"""
Per-game history for every skater and goalie, from the NHL stats API.

api.nhle.com/stats/rest returns one row per player-game (isGame=true) for a
whole date range in a single call (limit=-1), so a season is ~36 calls (30-day
windows x 3 reports) and the full history a couple of minutes. Completed seasons never change: they
are cached in data/cache/history (kept between CI runs by actions/cache)
and only the current season is fetched each run. Nothing here is
committed; a day's history in git would add ~10MB to the repo daily.
"""

import json
import os
import time
import urllib.parse
import urllib.request
from datetime import date, timedelta

import pandas as pd

CACHE_DIR = "data/cache/history"
BASE = "https://api.nhle.com/stats/rest/en"
FIRST_SEASON = 2019  # 2019-20 and 2020-21 only feed the positional priors
COLUMNS = {
    "skater/summary": ["playerId", "gameId", "gameDate", "teamAbbrev", "opponentTeamAbbrev",
                       "homeRoad", "positionCode", "goals", "shots", "ppGoals", "skaterFullName"],
    "skater/timeonice": ["playerId", "gameId", "timeOnIce", "ppTimeOnIce", "evTimeOnIce",
                         "shTimeOnIce"],
    "goalie/summary": ["playerId", "gameId", "gameDate", "teamAbbrev", "gamesStarted",
                       "shotsAgainst", "goalsAgainst"],
}


def season_of(d):
    """Start year of the NHL season a date belongs to."""
    return d.year if d.month >= 8 else d.year - 1


def _get(report, start, end, attempts=6):
    exp = (f'gameDate>="{start}" and gameDate<="{end}" '
           f'and (gameTypeId=2 or gameTypeId=3)')
    url = (f"{BASE}/{report}?isAggregate=false&isGame=true&start=0&limit=-1"
           f"&cayenneExp={urllib.parse.quote(exp)}")
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.load(resp)["data"]
        except Exception as e:
            if attempt == attempts - 1:
                raise RuntimeError(f"{report} {start}..{end}: {e}") from e
            time.sleep(5 * (attempt + 1))


def _fetch_season(y0, until):
    """One season's games through `until`, as {report: DataFrame}."""
    frames = {}
    for report, cols in COLUMNS.items():
        rows = []
        d, stop = date(y0, 8, 1), min(date(y0 + 1, 7, 31), until)
        while d <= stop:
            e = min(d + timedelta(days=29), stop)
            rows += _get(report, d.isoformat(), e.isoformat())
            d = e + timedelta(days=1)
        frames[report] = pd.DataFrame(rows, columns=cols)
    return frames


def load(until):
    """All regular-season and playoff player-games from FIRST_SEASON through
    `until` (a date, inclusive): (skaters, goalies) DataFrames."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    current = season_of(until)
    parts = {report: [] for report in COLUMNS}
    for y0 in range(FIRST_SEASON, current + 1):
        paths = {r: f"{CACHE_DIR}/{y0}_{r.replace('/', '_')}.csv.gz" for r in COLUMNS}
        if y0 < current and all(os.path.exists(p) for p in paths.values()):
            frames = {r: pd.read_csv(p) for r, p in paths.items()}
        else:
            frames = _fetch_season(y0, until)
            if y0 < current:
                for r, p in paths.items():
                    frames[r].to_csv(p, index=False)
        for r in COLUMNS:
            parts[r].append(frames[r])
    # An empty season (opening day: no games yet) has object columns and would
    # turn the whole concat to object; leave it out.
    cat = {r: pd.concat([f for f in v if len(f)] or v[:1], ignore_index=True)
           .drop_duplicates(["playerId", "gameId"]) for r, v in parts.items()}
    skaters = cat["skater/summary"].merge(cat["skater/timeonice"],
                                          on=["playerId", "gameId"], how="left")
    return skaters, cat["goalie/summary"]
