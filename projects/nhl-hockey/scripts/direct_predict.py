"""
NHL Goal Predictor - Direct LR v1
=================================
Scores tonight's skaters with the direct player-game model (direct_model.py):
one logistic regression refit every morning on every skater-game since
2021-22, from the NHL stats API's per-game history (nhl_history.py).

Runs in shadow: it is graded and shown on the dashboard but not used for
the Telegram picks until it has 14 clean days.

Author: Mohammad G. Nasiri
"""

import json
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import direct_model as dm
import nhl_history
from nhl_api import api_get
from tims_match import match_tims


class Config:
    MODEL_NAME = "direct_lr"
    MODEL_DISPLAY_NAME = "Direct LR v1"
    DATA_DIR = "data"
    PREDICTIONS_DIR = f"{DATA_DIR}/predictions/{MODEL_NAME}"
    TIMS_DIR = f"{DATA_DIR}/tims_players"
    # A date argument replays a past day (history strictly before it).
    TODAY = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y-%m-%d")
    RECENT_TEAM_GAMES = 3  # without a Tims pool: who dressed in the team's last 3 games


def get_todays_games(date):
    """Today's games; None if the schedule couldn't be fetched (distinct
    from [] = a day with no games)."""
    data = api_get(f"https://api-web.nhle.com/v1/schedule/{date}", attempts=6)
    if not data:
        return None
    games = []
    for day in data.get("gameWeek", []):
        if day["date"] == date:
            for g in day.get("games", []):
                if g.get("gameType") in [2, 3]:
                    games.append({
                        "game_id": g["id"],
                        "home_team": g["homeTeam"]["abbrev"],
                        "away_team": g["awayTeam"]["abbrev"],
                        "start_time": g.get("startTimeUTC", ""),
                        "game_type": g["gameType"],
                    })
    return games


def load_tims_players(date):
    path = f"{Config.TIMS_DIR}/{date}.json"
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def write(out):
    os.makedirs(Config.PREDICTIONS_DIR, exist_ok=True)
    for path in (f"{Config.PREDICTIONS_DIR}/{Config.TODAY}.json",
                 f"{Config.PREDICTIONS_DIR}/latest.json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {Config.PREDICTIONS_DIR}/{Config.TODAY}.json")


def candidates(sk, sched, tims):
    """Tonight's skaters on teams that play: who dressed in each team's last
    few games, plus every Tims pool player (call-ups, trades)."""
    last = sk.sort_values("date").groupby("pid").last()
    rows = {}
    for team in sched:
        team_games = sk[sk.team == team].drop_duplicates("gid").nlargest(
            Config.RECENT_TEAM_GAMES, "date").gid
        for pid in sk[sk.gid.isin(team_games) & (sk.team == team)].pid.unique():
            if last.loc[pid, "team"] != team:
                continue  # traded since: he plays for his latest team
            rows[int(pid)] = dict(pid=int(pid), team=team, name=last.loc[pid, "skaterFullName"])
    for entries in ((tims or {}).get("groups") or {}).values():
        for e in entries:
            if isinstance(e, dict) and e.get("player_id") and e.get("team") in sched:
                rows[int(e["player_id"])] = dict(pid=int(e["player_id"]), team=e["team"],
                                                 name=e.get("name", "?"),
                                                 tims_pos=str(e.get("position", "")))
    out = []
    for r in rows.values():
        s = sched[r["team"]]
        tims_pos = r.pop("tims_pos", "")
        # No NHL games yet (a debut): the pool's position decides F/D priors
        pos = (last.loc[r["pid"], "pos"] if r["pid"] in last.index
               else ("D" if tims_pos.startswith("Def") else "C"))
        out.append(dict(r, pos=pos, date=pd.Timestamp(Config.TODAY), **s))
    return pd.DataFrame(out)


def player_stats(sk, pids, season):
    """Display fields (season goals, last-5 goals) from history."""
    h = sk[sk.pid.isin(pids)].sort_values("date")
    cur = h[h.season == season].groupby("pid").agg(games_played=("gid", "size"),
                                                     season_goals=("goals", "sum"),
                                                     season_shots=("shots", "sum"))
    last5 = h.groupby("pid").tail(5).groupby("pid").agg(last5_goals=("goals", "sum"),
                                                        last5_games=("gid", "size"))
    return cur.join(last5, how="outer").fillna(0)


def main():
    print("=" * 70)
    print(f"  NHL GOAL PREDICTOR — {Config.MODEL_DISPLAY_NAME}")
    print(f"  {Config.TODAY}")
    print("=" * 70)

    games = get_todays_games(Config.TODAY)
    if games is None:
        print("  Could not fetch today's NHL schedule — not writing predictions.")
        return 1
    if not games:
        print("  No games today.")
        write({"date": Config.TODAY, "model": Config.MODEL_NAME,
               "model_display_name": Config.MODEL_DISPLAY_NAME, "games_count": 0, "games": [],
               "players_count": 0, "predictions": [], "tims_mode": False,
               "generated_at": datetime.now().isoformat()})
        return 0
    print(f"  {len(games)} games")
    sched = {}
    for g in games:
        playoff = int(g["game_type"] == 3)
        sched[g["home_team"]] = dict(opp=g["away_team"], home=1, gid=g["game_id"], playoff=playoff)
        sched[g["away_team"]] = dict(opp=g["home_team"], home=0, gid=g["game_id"], playoff=playoff)

    yesterday = (datetime.strptime(Config.TODAY, "%Y-%m-%d") - timedelta(days=1)).date()
    print(f"  Loading per-game history through {yesterday}...")
    skaters, goalies = nhl_history.load(yesterday)
    sk, gl = dm.prepare(skaters, goalies)
    today = pd.Timestamp(Config.TODAY)
    sk, gl = sk[sk.date < today], gl[gl.date < today]
    print(f"    {len(sk):,} skater-games, {len(gl):,} goalie-games")

    S = dm.build_states(sk, gl)
    model = dm.fit(sk, S)
    n_train = int((sk.season >= dm.TRAIN_FROM_SEASON).sum())
    print(f"  Fitted on {n_train:,} skater-games")

    tims = load_tims_players(Config.TODAY)
    Q = candidates(sk, sched, tims)
    Q["qid"] = np.arange(len(Q))
    F = dm.featurize(Q[["qid", "pid", "pos", "date", "team", "opp", "home", "gid", "playoff"]], S)
    Q["goal_probability"] = dm.predict(model, F.loc[Q.qid])

    season = int(dm.season_of(pd.Series([today]))[0])
    stats = player_stats(sk, Q.pid, season)
    players = []
    for r in Q.itertuples():
        st = stats.loc[r.pid] if r.pid in stats.index else None
        gp = int(st.games_played) if st is not None else 0
        goals = int(st.season_goals) if st is not None else 0
        players.append({
            "player_id": int(r.pid), "name": r.name, "team": r.team, "position": r.pos,
            "games_played": gp, "season_goals": goals,
            "avg_goals": round(goals / gp, 4) if gp else 0,
            "last5_goals": int(st.last5_goals) if st is not None else 0,
            "last5_games": int(st.last5_games) if st is not None else 0,
            "opponent": r.opp, "is_home": bool(r.home), "game_id": int(r.gid),
            "matchup": f"{r.team} vs {r.opp}" if r.home else f"{r.team} @ {r.opp}",
            "goal_probability": round(float(r.goal_probability), 4),
        })

    tims_mode = bool(tims and tims.get("groups"))
    if tims_mode:
        players, _ = match_tims(players, tims, Config.MODEL_NAME)
    # Ties go to more season goals, then a fixed id order (as in the other models)
    players.sort(key=lambda p: (p["goal_probability"], p["season_goals"], -p["player_id"]),
                 reverse=True)
    tims_group_rankings = {}
    for i, p in enumerate(players):
        p["rank"] = i + 1
        gid = p.get("tims_group")
        if gid:
            p["tims_rank"] = i + 1
            ranked = tims_group_rankings.setdefault(gid, [])
            ranked.append({"rank_in_group": len(ranked) + 1, "player_id": p["player_id"],
                           "name": p["name"], "team": p["team"],
                           "goal_probability": p["goal_probability"], "matchup": p["matchup"]})

    write({
        "date": Config.TODAY,
        "model": Config.MODEL_NAME,
        "model_display_name": Config.MODEL_DISPLAY_NAME,
        "games_count": len(games),
        "games": games,
        "players_count": len(players),
        "predictions": players,
        "tims_mode": tims_mode,
        "tims_source": tims.get("source") if tims else None,
        "tims_group_rankings": tims_group_rankings,
        "model_params": {
            "train_skater_games": n_train,
            "train_from_season": dm.TRAIN_FROM_SEASON,
            "history_through": str(yesterday),
            "features": len(dm.FEATS_BASE),
            "C": 0.1,
        },
        "generated_at": datetime.now().isoformat(),
    })

    for gid in sorted(tims_group_rankings):
        top = tims_group_rankings[gid][:3]
        print(f"  Group {gid}: " + ", ".join(f"{p['name']} {p['goal_probability']:.1%}" for p in top))
    return 0


if __name__ == "__main__":
    sys.exit(main())
