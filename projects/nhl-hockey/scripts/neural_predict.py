"""
NHL Goal Predictor — Neural v1 (MLP re-ranker).

Reads Monte Carlo v2's latest.json (which already has feature-rich per-player
rows for tonight's Tim Hortons pool), scores each player through the MLP
trained by train_neural_model.py, and writes a parallel prediction file
under data/predictions/neural_v1/.

Depends on MC v2 having run first. If MC v2 is missing/empty or the neural
model isn't trained yet, this falls through cleanly with an empty output so
the daily workflow doesn't fail.

Author: Mohammad G. Nasiri
"""

import json
import os
import sys
from datetime import datetime

try:
    import numpy as np
    import joblib
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install scikit-learn joblib numpy")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    MODEL_NAME = "neural_v1"
    MODEL_DISPLAY_NAME = "Neural MLP v1"

    DATA_DIR = "data"
    PREDICTIONS_DIR = f"{DATA_DIR}/predictions/{MODEL_NAME}"
    SRC_DIR = f"{DATA_DIR}/predictions/monte_carlo"      # upstream feature source
    SRC_LATEST = f"{SRC_DIR}/latest.json"

    NEURAL_MODEL_DIR = f"{DATA_DIR}/neural_model"
    MODEL_FILE = f"{NEURAL_MODEL_DIR}/model.joblib"
    SCALER_FILE = f"{NEURAL_MODEL_DIR}/scaler.joblib"
    METADATA_FILE = f"{NEURAL_MODEL_DIR}/metadata.json"

    TODAY = datetime.now().strftime("%Y-%m-%d")

    # Must match train_neural_model.py Config.FEATURES + BINARY_FEATURES in
    # the same order — StandardScaler is positional.
    FEATURES = [
        "avg_goals", "avg_shots", "shooting_pct",
        "pp_goals", "pp_points",
        "recent_gpg", "last5_goals", "last5_shots",
        "opp_ga_per_game", "team_gf_per_game",
    ]
    BINARY_FEATURES = ["is_home"]


os.makedirs(Config.PREDICTIONS_DIR, exist_ok=True)


def _empty_output(reason):
    """Write an empty prediction file so downstream consumers (meta ensemble,
    dashboard, stats) see a valid file and don't crash."""
    out = {
        "date": Config.TODAY,
        "model": Config.MODEL_NAME,
        "model_display_name": Config.MODEL_DISPLAY_NAME,
        "games_count": 0,
        "games": [],
        "players_count": 0,
        "predictions": [],
        "tims_mode": False,
        "note": reason,
        "generated_at": datetime.now().isoformat(),
    }
    for path in (f"{Config.PREDICTIONS_DIR}/{Config.TODAY}.json",
                 f"{Config.PREDICTIONS_DIR}/latest.json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)


def main():
    print("=" * 70)
    print(f"  NHL GOAL PREDICTOR — {Config.MODEL_DISPLAY_NAME}")
    print(f"  {Config.TODAY}")
    print("=" * 70)

    # 1. Load upstream features (MC v2's today file)
    if not os.path.exists(Config.SRC_LATEST):
        print(f"  No upstream: {Config.SRC_LATEST}")
        _empty_output("No upstream Monte Carlo predictions available")
        return 0
    with open(Config.SRC_LATEST, "r", encoding="utf-8") as f:
        src = json.load(f)
    upstream_preds = src.get("predictions", [])
    if not upstream_preds:
        print("  Upstream MC v2 has 0 predictions; nothing to score")
        _empty_output("Upstream Monte Carlo has no predictions")
        return 0

    # 2. Load the trained MLP + scaler
    if not (os.path.exists(Config.MODEL_FILE) and os.path.exists(Config.SCALER_FILE)):
        print(f"  No trained model at {Config.MODEL_FILE} — run train_neural_model.py first")
        _empty_output("Neural model not yet trained")
        return 0
    model = joblib.load(Config.MODEL_FILE)
    scaler = joblib.load(Config.SCALER_FILE)

    feature_cols = Config.FEATURES + Config.BINARY_FEATURES

    # 3. Build feature matrix aligned with training schema
    X_rows = []
    for p in upstream_preds:
        row = [float(p.get(f, 0) or 0) for f in Config.FEATURES]
        row += [int(bool(p.get(bf, False))) for bf in Config.BINARY_FEATURES]
        X_rows.append(row)
    X = np.array(X_rows, dtype=float)
    Xs = scaler.transform(X)
    probs = model.predict_proba(Xs)[:, 1]

    # 4. Assemble output, preserving MC v2's context per player
    scored = []
    for p, prob in zip(upstream_preds, probs):
        out_row = dict(p)  # copy MC v2 row wholesale
        out_row["goal_probability"] = round(float(prob), 4)
        # Preserve the MC v2 probability so the dashboard can show the delta
        out_row["mc_probability"] = p.get("goal_probability")
        out_row["model"] = Config.MODEL_NAME
        scored.append(out_row)

    scored.sort(key=lambda r: r["goal_probability"], reverse=True)
    for i, r in enumerate(scored):
        r["rank"] = i + 1
    # Tims rank within the Tims-filtered pool — same players since upstream
    # is already Tims-filtered, but numbering matches our own sort.
    for i, r in enumerate(scored):
        r["tims_rank"] = i + 1

    # 5. Rebuild Tims group rankings using Neural scores
    tims_group_rankings = {}
    if src.get("tims_mode"):
        for r in scored:
            gid = r.get("tims_group")
            if not gid:
                continue
            bucket = tims_group_rankings.setdefault(gid, [])
            bucket.append({
                "rank_in_group": len(bucket) + 1,
                "player_id": r["player_id"],
                "name": r["name"],
                "team": r["team"],
                "goal_probability": r["goal_probability"],
                "matchup": r.get("matchup", ""),
            })

    out = {
        "date": src.get("date", Config.TODAY),
        "model": Config.MODEL_NAME,
        "model_display_name": Config.MODEL_DISPLAY_NAME,
        "games_count": src.get("games_count", 0),
        "games": src.get("games", []),
        "players_count": len(scored),
        "predictions": scored,
        "tims_mode": src.get("tims_mode", False),
        "tims_source": src.get("tims_source"),
        "tims_group_rankings": tims_group_rankings,
        "upstream_source": "monte_carlo",
        "generated_at": datetime.now().isoformat(),
    }

    for path in (f"{Config.PREDICTIONS_DIR}/{Config.TODAY}.json",
                 f"{Config.PREDICTIONS_DIR}/latest.json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {Config.PREDICTIONS_DIR}/{Config.TODAY}.json")

    # 6. Console summary
    print(f"\n  Top 10 Neural v1 picks:")
    for r in scored[:10]:
        print(f"    {r['rank']:>2}  {r['name']:<22} {r['team']:<4} "
              f"{r['goal_probability']*100:>5.1f}%  (MC: "
              f"{(r.get('mc_probability') or 0)*100:>5.1f}%)")
    print("\n  Neural v1 predictions complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
