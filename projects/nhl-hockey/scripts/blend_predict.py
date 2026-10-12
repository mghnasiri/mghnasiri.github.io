"""
NHL Goal Predictor - Blend v1 (shadow)
======================================
Ranks each Tim Hortons group by the mean of the strongest models' log-odds,
each z-scored within the group so no model dominates by its spread: Direct
LR, Market, xG v3 and Neural v2, using whichever of them have a file today.
No fitted weights. On 82 past Tims days this ranking picked +10.7pp over
random, but it was found after screening ~20 combiners, so it runs in shadow
and is judged only on days after 2026-10-12.

goal_probability is the mean of the models' (calibrated) log-odds turned
back into a probability, for display; the group order uses the z-scores.

Author: Mohammad G. Nasiri
"""

import json
import math
import os
import sys
from datetime import datetime

MODEL_NAME = "blend"
MODEL_DISPLAY_NAME = "Blend v1"
DATA_DIR = "data"
PREDICTIONS_DIR = f"{DATA_DIR}/predictions/{MODEL_NAME}"
INPUTS = ["direct_lr", "market_odds", "xg_v3", "neural_v2"]
TODAY = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y-%m-%d")


def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def zscores(values):
    """{key: z} within one group; empty when there's no spread to scale by."""
    if len(values) < 2:
        return {}
    mean = sum(values.values()) / len(values)
    sd = math.sqrt(sum((v - mean) ** 2 for v in values.values()) / len(values))
    return {k: (v - mean) / sd for k, v in values.items()} if sd > 0 else {}


def write(out):
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    for path in (f"{PREDICTIONS_DIR}/{TODAY}.json", f"{PREDICTIONS_DIR}/latest.json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {PREDICTIONS_DIR}/{TODAY}.json")


def main():
    print(f"  BLEND v1 — {TODAY}")
    pool = load_json(f"{DATA_DIR}/tims_players/{TODAY}.json") or {}
    sources = {}
    for m in INPUTS:
        d = load_json(f"{DATA_DIR}/predictions/{m}/{TODAY}.json")
        if d and d.get("date") == TODAY and d.get("predictions"):
            sources[m] = {p["player_id"]: p for p in d["predictions"] if p.get("player_id")}
    print(f"  inputs: {sorted(sources) or 'none'}")

    base = {"date": TODAY, "model": MODEL_NAME, "model_display_name": MODEL_DISPLAY_NAME,
            "generated_at": datetime.now().isoformat()}
    if not sources:
        print("  no input model has picks today — writing no picks")
        write({**base, "games_count": 0, "games": [], "players_count": 0, "predictions": [],
               "tims_mode": False, "note": "no input model has picks today"})
        return 0
    tims_mode = pool.get("date", TODAY) == TODAY and bool(pool.get("groups"))
    if tims_mode:
        groups = {gid: [m["player_id"] for m in members if isinstance(m, dict) and m.get("player_id")]
                  for gid, members in pool["groups"].items()}
    else:
        # No pool today: one group of everyone the inputs scored, so the day
        # still gets a ranking (and the health check sees predictions).
        groups = {None: sorted({pid for preds in sources.values() for pid in preds})}

    players, rankings = [], {}
    for gid, ids in groups.items():
        z_by_model = {}
        for m, preds in sources.items():
            lo = {pid: logit(preds[pid].get("raw_goal_probability", preds[pid]["goal_probability"]))
                  for pid in ids if pid in preds}
            z_by_model[m] = zscores(lo)
        ranked = []
        for pid in ids:
            zs = [z[pid] for z in z_by_model.values() if pid in z]
            if not zs:
                continue  # no model covers him: left out (ranks last)
            shown = [logit(sources[m][pid]["goal_probability"]) for m in sources if pid in sources[m]]
            info = next(sources[m][pid] for m in INPUTS if m in sources and pid in sources[m])
            ranked.append({
                "player_id": pid, "name": info.get("name", "?"), "team": info.get("team", "?"),
                "position": info.get("position", ""), "opponent": info.get("opponent", ""),
                "matchup": info.get("matchup", ""), "is_home": info.get("is_home"),
                "season_goals": info.get("season_goals", 0),
                "last5_goals": info.get("last5_goals", 0),
                "goal_probability": round(1 / (1 + math.exp(-sum(shown) / len(shown))), 4),
                "blend_score": round(sum(zs) / len(zs), 4),
                "blend_models": len(zs),
                **({"tims_group": gid} if gid else {}),
            })
        ranked.sort(key=lambda p: (p["blend_score"], p["season_goals"], -p["player_id"]),
                    reverse=True)
        if gid is None:
            players += ranked
            continue
        rankings[gid] = [{"rank_in_group": i + 1, "player_id": p["player_id"], "name": p["name"],
                          "team": p["team"], "goal_probability": p["goal_probability"],
                          "matchup": p["matchup"]} for i, p in enumerate(ranked)]
        players += ranked

    players.sort(key=lambda p: (p["goal_probability"], p["season_goals"], -p["player_id"]),
                 reverse=True)
    for i, p in enumerate(players):
        p["rank"] = i + 1
    games = next((load_json(f"{DATA_DIR}/predictions/{m}/{TODAY}.json") or {}).get("games")
                 for m in INPUTS if m in sources)
    write({**base, "games_count": len(games or []), "games": games or [],
           "players_count": len(players), "predictions": players, "tims_mode": tims_mode,
           "tims_source": pool.get("source") if tims_mode else None, "tims_group_rankings": rankings,
           "model_params": {"inputs": sorted(sources), "method": "mean of within-group z-scored log-odds"}})
    for gid in sorted(rankings):
        print(f"  Group {gid}: " + ", ".join(p["name"] for p in rankings[gid][:3]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
