"""
Aggregate per-player real shot distributions from play-by-play data.

Reads shots collected by collect_xg_data.py into data/xg_training/shots.csv,
bucket-sorts them by player_id, keeps each player's recent shots, and writes
one compact JSON per player under data/player_shots/{player_id}.json.

Also writes data/player_shots/_position_priors.json — an aggregate shot
distribution per position code (C, L, R, D) that xg_predict.py uses as a
cold-start fallback for players below MIN_SHOTS_FOR_PLAYER_FILE.

Feeds xg_predict.py's new real-shot path. Schedule this AFTER collect_xg_data.py
and BEFORE xg_predict.py in the daily workflow.

Author: Mohammad G. Nasiri
"""

import csv
import glob
import json
import os
from collections import defaultdict
from datetime import datetime


# =============================================================================
# CONFIGURATION — tune these if validation shows different N works better
# =============================================================================
SHOTS_CSV = "data/xg_training/shots.csv"
PREDICTIONS_GLOB = "data/predictions/*/latest.json"
OUT_DIR = "data/player_shots"
PRIORS_FILE = "_position_priors.json"

# Decision A: how many recent shots per player
MAX_SHOTS_PER_PLAYER = 60
MIN_GAMES_EQUIVALENT = 15  # keep more shots if the 60-most-recent span < this many games

# Decision B: below this many shots, fall back to position prior instead of
# writing a player-specific file
MIN_SHOTS_FOR_PLAYER_FILE = 20

# Decision C: flat weighting (no decay). Not a knob yet — change here if
# you add time decay later.

# Feature schema — must match data/xg_model/metadata.json
SHOT_TYPES = ["backhand", "bat", "between-legs", "deflected", "poke",
              "slap", "snap", "tip-in", "wrap-around", "wrist"]
STRENGTH_STATES = ["1v0", "3v3", "3v4", "4v3", "4v4", "4v5", "4v6",
                   "5v3", "5v4", "5v5", "5v6", "6v4", "6v5"]
POSITION_KEYS = ("C", "L", "R", "D")  # priors are keyed by first letter
DEFAULT_POSITION = "C"


# =============================================================================
# ROW → FEATURE DICT
# =============================================================================
def row_to_feature_dict(row):
    """Convert one shots.csv row into the exact feature dict XGBoost expects."""
    try:
        dist = float(row["shot_distance"])
        angle = float(row["shot_angle"])
    except (ValueError, KeyError):
        return None
    feat = {
        "shot_distance": dist,
        "shot_angle": angle,
        "is_powerplay": int(row.get("is_powerplay", 0) or 0),
        "is_rebound": int(row.get("is_rebound", 0) or 0),
        "seconds_since_last_event": float(row.get("seconds_since_last_event", 0) or 0),
        "is_empty_net": int(row.get("is_empty_net", 0) or 0),
        "score_differential": int(row.get("score_differential", 0) or 0),
        "period": int(row.get("period", 1) or 1),
        "prior_event_distance": float(row.get("prior_event_distance", 0) or 0),
        "distance_x_angle": dist * angle,
        "is_slot_shot": int(dist < 20),
        "is_high_danger": int(dist < 30 and angle < 30),
    }
    shot_type = row.get("shot_type", "wrist") or "wrist"
    for st in SHOT_TYPES:
        feat[f"shot_type_{st}"] = 1 if shot_type == st else 0
    strength = row.get("strength_state", "5v5") or "5v5"
    for ss in STRENGTH_STATES:
        feat[f"strength_state_{ss}"] = 1 if strength == ss else 0
    return feat


# =============================================================================
# POSITION LOOKUP — scan all latest prediction files to build {player_id: pos}
# =============================================================================
def build_position_map():
    """Scan every model's latest.json + dated files for player position info.

    Shots.csv doesn't carry position. We scan predictions which always include
    it. Latest observed position wins (players can shift between F positions).
    """
    pos_map = {}
    # Walk all prediction JSONs (latest + recent dated) — position is stable
    # enough that we don't need to be clever about which one wins.
    candidates = glob.glob(PREDICTIONS_GLOB)
    candidates += glob.glob("data/predictions/*/2026-*.json")
    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        for p in data.get("predictions", []):
            pid = p.get("player_id")
            pos = p.get("position", "")
            if pid and pos:
                pos_map[int(pid)] = pos
    return pos_map


def normalize_position(pos):
    """Collapse NHL position codes to C/L/R/D bucket."""
    if not pos:
        return DEFAULT_POSITION
    first = pos[0].upper()
    return first if first in POSITION_KEYS else DEFAULT_POSITION


# =============================================================================
# MAIN AGGREGATION
# =============================================================================
def main():
    if not os.path.exists(SHOTS_CSV):
        print(f"  ERROR: {SHOTS_CSV} not found — run collect_xg_data.py first.")
        return 1

    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  NHL PLAYER-SHOT AGGREGATOR")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    print("\n  Building position map from prediction files...")
    pos_map = build_position_map()
    print(f"    {len(pos_map)} players with known position")

    print(f"\n  Reading {SHOTS_CSV}...")
    by_player = defaultdict(list)     # pid -> [(game_date, game_id, feature_dict)]
    by_position = defaultdict(list)   # pos_key -> [feature_dict]
    total_rows = 0

    with open(SHOTS_CSV, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            total_rows += 1
            feat = row_to_feature_dict(row)
            if feat is None:
                continue
            try:
                pid = int(row["player_id"])
            except (ValueError, KeyError):
                continue
            if not pid:
                continue
            by_player[pid].append((
                row.get("game_date", ""),
                row.get("game_id", ""),
                feat,
            ))
            # Feed position priors bucket
            pos_key = normalize_position(pos_map.get(pid, DEFAULT_POSITION))
            by_position[pos_key].append(feat)

    print(f"    Parsed {total_rows} rows into {len(by_player)} unique players")

    # --- Per-player files ---
    # Sort newest-first, truncate. Keep MAX_SHOTS_PER_PLAYER unless that
    # would cover fewer than MIN_GAMES_EQUIVALENT distinct games — in which
    # case keep more shots so we span enough games.
    print(f"\n  Writing per-player files (target: {MAX_SHOTS_PER_PLAYER} shots, "
          f">= {MIN_GAMES_EQUIVALENT} games, min {MIN_SHOTS_FOR_PLAYER_FILE} shots)...")
    written = 0
    skipped_few_shots = 0

    for pid, rows in by_player.items():
        rows.sort(key=lambda r: (r[0], r[1]), reverse=True)  # newest first
        if len(rows) < MIN_SHOTS_FOR_PLAYER_FILE:
            skipped_few_shots += 1
            continue

        # Keep enough rows to span MIN_GAMES_EQUIVALENT games (or
        # MAX_SHOTS_PER_PLAYER shots, whichever is more)
        kept = []
        game_ids_seen = set()
        for game_date, game_id, feat in rows:
            kept.append(feat)
            game_ids_seen.add(game_id)
            done = (len(kept) >= MAX_SHOTS_PER_PLAYER
                    and len(game_ids_seen) >= MIN_GAMES_EQUIVALENT)
            if done:
                break

        out = {
            "player_id": pid,
            "position": pos_map.get(pid, ""),
            "shot_count": len(kept),
            "games_covered": len(game_ids_seen),
            "latest_shot_date": rows[0][0],
            "shots": kept,
        }
        path = os.path.join(OUT_DIR, f"{pid}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, separators=(",", ":"))
        written += 1

    print(f"    {written} player files written")
    print(f"    {skipped_few_shots} players skipped (will use position priors)")

    # --- Position priors ---
    # For each position key, keep a representative sample of recent shots.
    # Cap at 500/bucket so any one position doesn't produce an absurdly
    # large file.
    POS_PRIOR_CAP = 500
    print(f"\n  Writing position priors ({PRIORS_FILE})...")
    priors = {}
    for pos_key in POSITION_KEYS:
        shots = by_position.get(pos_key, [])
        sample = shots[-POS_PRIOR_CAP:]  # take most recent (CSV is append-only chronological)
        priors[pos_key] = {"shot_count": len(sample), "shots": sample}
        print(f"    {pos_key}: {len(shots)} total, kept {len(sample)}")

    priors_path = os.path.join(OUT_DIR, PRIORS_FILE)
    with open(priors_path, "w", encoding="utf-8") as f:
        json.dump(priors, f, separators=(",", ":"))

    print(f"\n  Aggregation complete. Output: {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
