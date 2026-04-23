"""
Train Neural v1 — small MLP goal predictor.

Learns a nonlinear combination of raw player + matchup features to predict
whether a player scores 1+ goals in a given game. Trained end-to-end on
the repo's own prediction/results history (no new API calls needed) using
sklearn's MLPClassifier so training is CPU-friendly.

Training set is built by joining:
  data/predictions/monte_carlo/YYYY-MM-DD.json  (feature-rich per-player rows)
  data/results/YYYY-MM-DD.json                  (who actually scored that night)

Saves the fitted model + a scaler (for inference-time normalization) and
some metadata about training to data/neural_model/.

Run with --min-rows to guard against training on pathologically small data.

Author: Mohammad G. Nasiri
"""

import json
import os
import sys
from datetime import datetime

try:
    import numpy as np
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import roc_auc_score
    import joblib
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install scikit-learn joblib numpy")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    DATA_DIR = "data"
    PRED_DIR = f"{DATA_DIR}/predictions/monte_carlo"   # source of feature-rich rows
    RESULTS_DIR = f"{DATA_DIR}/results"
    OUT_DIR = f"{DATA_DIR}/neural_model"
    MODEL_FILE = f"{OUT_DIR}/model.joblib"
    SCALER_FILE = f"{OUT_DIR}/scaler.joblib"
    METADATA_FILE = f"{OUT_DIR}/metadata.json"
    MIN_ROWS_TO_TRAIN = 500

    # Features pulled per row (all available in MC v2 prediction JSON)
    FEATURES = [
        "avg_goals", "avg_shots", "shooting_pct",
        "pp_goals", "pp_points",
        "recent_gpg", "last5_goals", "last5_shots",
        "opp_ga_per_game", "team_gf_per_game",
    ]
    BINARY_FEATURES = ["is_home"]

    # Small-ish network — the training set is a few thousand rows, not millions.
    MLP_PARAMS = dict(
        hidden_layer_sizes=(64, 32, 16),
        activation="relu",
        solver="adam",
        alpha=1e-3,              # L2 regularization
        batch_size=64,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42,
    )


os.makedirs(Config.OUT_DIR, exist_ok=True)


# =============================================================================
# BUILD DATASET
# =============================================================================
def load_prediction_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def build_dataset():
    """Walk historical predictions + results, join on (date, player_id).
    Each row is one (player, date) observation with the player's features
    at prediction time and a 1/0 label for whether they scored that night."""
    rows = []
    if not os.path.isdir(Config.PRED_DIR):
        print(f"  No predictions dir: {Config.PRED_DIR}")
        return rows

    for fname in sorted(os.listdir(Config.PRED_DIR)):
        if not (fname.startswith("2026-") or fname.startswith("2025-")):
            continue
        if not fname.endswith(".json"):
            continue
        date_str = fname[:-5]
        pred = load_prediction_file(os.path.join(Config.PRED_DIR, fname))
        if not pred:
            continue

        results = load_prediction_file(os.path.join(Config.RESULTS_DIR, fname))
        if not results:
            continue

        scorer_ids = {s["player_id"] for s in results.get("all_scorers", [])}

        for p in pred.get("predictions", []):
            feats = {f: float(p.get(f, 0) or 0) for f in Config.FEATURES}
            for bf in Config.BINARY_FEATURES:
                feats[bf] = int(bool(p.get(bf, False)))
            feats["scored"] = int(p.get("player_id") in scorer_ids)
            feats["date"] = date_str
            feats["player_id"] = p.get("player_id")
            rows.append(feats)

    return rows


# =============================================================================
# TRAIN
# =============================================================================
def train_model(rows):
    feature_cols = Config.FEATURES + Config.BINARY_FEATURES
    X = np.array([[r[f] for f in feature_cols] for r in rows], dtype=float)
    y = np.array([r["scored"] for r in rows], dtype=int)

    print(f"\n  Dataset: {len(rows)} rows")
    print(f"  Positive rate: {y.mean():.3f} ({int(y.sum())} scorers)")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    print("\n  Training MLP...")
    model = MLPClassifier(**Config.MLP_PARAMS)

    # Quick sanity CV on small folds; final fit is on the whole thing
    print("  3-fold CV for AUC...")
    try:
        cv_aucs = cross_val_score(model, Xs, y, cv=3, scoring="roc_auc")
        cv_auc = float(cv_aucs.mean())
        print(f"  CV AUC: {cv_auc:.4f} (+/- {cv_aucs.std():.4f})")
    except Exception as e:
        print(f"  CV failed ({e}); proceeding with fit anyway")
        cv_auc = None

    model.fit(Xs, y)
    train_pred = model.predict_proba(Xs)[:, 1]
    train_auc = roc_auc_score(y, train_pred)
    print(f"  In-sample AUC: {train_auc:.4f}")

    return model, scaler, {
        "cv_auc": cv_auc,
        "in_sample_auc": train_auc,
        "num_rows": int(len(rows)),
        "positive_rate": float(y.mean()),
        "feature_names": feature_cols,
        "hidden_layer_sizes": list(Config.MLP_PARAMS["hidden_layer_sizes"]),
        "trained_at": datetime.now().isoformat(),
    }


# =============================================================================
# MAIN
# =============================================================================
def main():
    min_rows = Config.MIN_ROWS_TO_TRAIN
    if "--min-rows" in sys.argv:
        idx = sys.argv.index("--min-rows")
        if idx + 1 < len(sys.argv):
            min_rows = int(sys.argv[idx + 1])

    print("=" * 60)
    print("  NEURAL v1 MODEL TRAINER")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    print("\n  Building dataset from predictions + results...")
    rows = build_dataset()
    if len(rows) < min_rows:
        print(f"\n  Not enough training rows ({len(rows)} < {min_rows}). "
              "Run daily workflow longer or lower --min-rows.")
        # Exit cleanly so CI doesn't fail red when just bootstrapping
        sys.exit(0)

    model, scaler, meta = train_model(rows)

    joblib.dump(model, Config.MODEL_FILE)
    joblib.dump(scaler, Config.SCALER_FILE)
    with open(Config.METADATA_FILE, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Saved: {Config.MODEL_FILE}")
    print(f"  Saved: {Config.SCALER_FILE}")
    print(f"  Saved: {Config.METADATA_FILE}")
    print("\n  Training complete.")


if __name__ == "__main__":
    main()
