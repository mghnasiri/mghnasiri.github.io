"""
Early-season prior for player and team stats.

Every model builds player stats from the current season's game log only and
requires MIN_GAMES_PLAYED games, so on opening nights no player qualifies and
the pipelines exit 1 (2026-09-29 outage). Team strength from standings/now has
the same problem (0 goals over 0 games -> gf_per_game 0).

Fix: pad each player's real games with up to PRIOR_GAMES pseudo-games at last
season's per-game rates, and pad team per-game rates toward the league average,
both fading out as real games accumulate. Once a player or team has
PRIOR_GAMES real games, stats are exactly the current season's again.

Last season's totals come from ONE bulk request (not per-player calls) and are
stored in data/season_priors/ — a finished season never changes.
"""

import json
import os
import time

import requests

PRIOR_GAMES = 10
PRIOR_DIR = "data/season_priors"
MIN_BULK_ROWS = 500  # a full season has ~900 skaters; fewer means a partial response
_BULK_URL = ("https://api.nhle.com/stats/rest/en/skater/summary?isAggregate=false"
             "&isGame=false&cayenneExp=seasonId={season}%20and%20gameTypeId=2&limit=-1")
_KEEP = ("playerId", "skaterFullName", "teamAbbrevs", "positionCode", "gamesPlayed",
         "goals", "shots", "assists", "points", "ppGoals", "ppPoints", "timeOnIcePerGame")

_priors = None


def previous_season_id(season_id):
    """'20262027' -> '20252026'."""
    start = int(str(season_id)[:4]) - 1
    return f"{start}{start + 1}"


def _fetch_bulk(season):
    for attempt in range(3):
        try:
            resp = requests.get(_BULK_URL.format(season=season), timeout=30)
            if resp.status_code == 200:
                rows = resp.json().get("data", [])
                if len(rows) >= MIN_BULK_ROWS:
                    return [{k: r.get(k) for k in _KEEP} for r in rows]
        except (requests.RequestException, ValueError):
            pass
        if attempt < 2:
            time.sleep(2 ** attempt)
    return None


def load_priors(season_id):
    """{player_id: last season's regular-season totals}. Empty dict if unavailable."""
    global _priors
    if _priors is not None:
        return _priors
    prev = previous_season_id(season_id)
    path = f"{PRIOR_DIR}/skaters_{prev}.json"
    rows = None
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                rows = json.load(f)
        except (OSError, ValueError):
            rows = None
    if rows is None:
        rows = _fetch_bulk(prev)
        if rows:
            os.makedirs(PRIOR_DIR, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False)
        else:
            print(f"   ⚠️ No {prev} season priors available — early-season "
                  f"players with < MIN_GAMES_PLAYED games will be skipped")
    _priors = {int(r["playerId"]): r for r in rows or [] if r.get("playerId")}
    return _priors


def pad_game_log(real_games, player_id, season_id):
    """Append up to PRIOR_GAMES - len(real_games) pseudo-games at last season's
    per-game rates (never more pseudo-games than the player actually played).

    real_games must be newest-first and already filtered to before today.
    Pseudo-games go last (oldest) so last-5 form uses real games first, and
    carry prior_season=True so callers can count them."""
    row = load_priors(season_id).get(int(player_id))
    gp = (row or {}).get("gamesPlayed") or 0
    k = min(PRIOR_GAMES - len(real_games), gp)
    if k <= 0:
        return real_games
    toi_sec = int(round(row.get("timeOnIcePerGame") or 0))
    pseudo = {
        "gameDate": "0000-00-00",
        "prior_season": True,
        "goals": (row.get("goals") or 0) / gp,
        "shots": (row.get("shots") or 0) / gp,
        "assists": (row.get("assists") or 0) / gp,
        "points": (row.get("points") or 0) / gp,
        "powerPlayGoals": (row.get("ppGoals") or 0) / gp,
        "powerPlayPoints": (row.get("ppPoints") or 0) / gp,
        "toi": f"{toi_sec // 60}:{toi_sec % 60:02d}",
    }
    return real_games + [dict(pseudo) for _ in range(k)]


def count_prior_games(games):
    return sum(1 for g in games if g.get("prior_season"))


def last5_is_real(stats):
    """False while pseudo-games still fill part of the last-5 window, so the
    'hot' flag reflects real recent form, not last season's rate. Identical
    to the old behavior whenever no padding is present."""
    padded = stats.get("prior_games") or 0
    return not padded or (stats.get("games_played", 0) - padded) >= 5


def shrink_team_rate(total, games, league_avg):
    """Per-game team rate padded toward the league average while the team has
    fewer than PRIOR_GAMES games (opening night would otherwise be 0 / 1 = 0)."""
    k = max(0, PRIOR_GAMES - games)
    return (total + k * league_avg) / (games + k) if games + k else league_avg
