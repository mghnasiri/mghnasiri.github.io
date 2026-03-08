"""
NHL xG Model Trainer — XGBoost Expected Goals
==============================================
Trains an XGBClassifier on accumulated shot data to predict
the probability of each shot resulting in a goal.

Usage:
  python train_xg_model.py                   # Train on all available data
  python train_xg_model.py --min-shots 1000  # Only train if enough data

Author: Mohammad G. Nasiri
"""

import json
import os
import sys
import numpy as np
from datetime import datetime

try:
    import pandas as pd
    import xgboost as xgb
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import roc_auc_score, log_loss
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install pandas xgboost scikit-learn")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    DATA_DIR = "data"
    XG_TRAINING_DIR = f"{DATA_DIR}/xg_training"
    XG_MODEL_DIR = f"{DATA_DIR}/xg_model"
    SHOTS_CSV = f"{XG_TRAINING_DIR}/shots.csv"
    MODEL_FILE = f"{XG_MODEL_DIR}/model.json"
    METADATA_FILE = f"{XG_MODEL_DIR}/metadata.json"
    FALLBACK_TABLE = f"{XG_MODEL_DIR}/fallback_table.json"

    MIN_SHOTS_TO_TRAIN = 500
    TARGET_COL = 'is_goal'

    # Numeric features used directly
    NUMERIC_FEATURES = [
        'shot_distance', 'shot_angle', 'is_powerplay', 'is_rebound',
        'seconds_since_last_event', 'is_empty_net', 'score_differential',
        'period', 'prior_event_distance',
    ]

    # Categorical features to one-hot encode
    CATEGORICAL_FEATURES = ['shot_type', 'strength_state']

    # XGBoost hyperparameters
    XGB_PARAMS = {
        'n_estimators': 200,
        'max_depth': 5,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'logloss',
        'random_state': 42,
        'n_jobs': -1,
    }


os.makedirs(Config.XG_MODEL_DIR, exist_ok=True)


# =============================================================================
# DATA LOADING & PREPROCESSING
# =============================================================================
def load_and_preprocess():
    """
    Load shots CSV, clean data, engineer features, encode categoricals.
    Returns (X, y, feature_names).
    """
    if not os.path.exists(Config.SHOTS_CSV):
        print("  No training data found!")
        return None, None, None

    df = pd.read_csv(Config.SHOTS_CSV)
    print(f"  Loaded {len(df)} shots from CSV")

    if len(df) < Config.MIN_SHOTS_TO_TRAIN:
        print(f"  Not enough data ({len(df)} < {Config.MIN_SHOTS_TO_TRAIN})")
        return None, None, None

    # Clean data
    df = df.dropna(subset=['shot_distance', 'shot_angle', 'is_goal'])
    df['is_goal'] = df['is_goal'].astype(int)

    # Fill missing numeric values
    for col in Config.NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Engineer interaction features
    df['distance_x_angle'] = df['shot_distance'] * df['shot_angle']
    df['is_slot_shot'] = (df['shot_distance'] < 20).astype(int)
    df['is_high_danger'] = (
        (df['shot_distance'] < 30) & (df['shot_angle'] < 30)
    ).astype(int)

    # One-hot encode categoricals
    encoded_dfs = [df[Config.NUMERIC_FEATURES + ['distance_x_angle', 'is_slot_shot', 'is_high_danger']]]

    for cat_col in Config.CATEGORICAL_FEATURES:
        if cat_col in df.columns:
            dummies = pd.get_dummies(df[cat_col], prefix=cat_col, dtype=int)
            encoded_dfs.append(dummies)

    X = pd.concat(encoded_dfs, axis=1)
    y = df[Config.TARGET_COL]

    # Remove any remaining NaN
    mask = X.notna().all(axis=1)
    X = X[mask]
    y = y[mask]

    feature_names = list(X.columns)
    print(f"  Features: {len(feature_names)}")
    print(f"  Goals: {y.sum()} / {len(y)} ({y.mean()*100:.2f}%)")

    return X, y, feature_names


# =============================================================================
# MODEL TRAINING
# =============================================================================
def train_model(X, y):
    """
    Train XGBClassifier with cross-validation.
    Returns (model, cv_metrics).
    """
    # Handle class imbalance
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    scale_pos_weight = n_neg / max(n_pos, 1)

    params = Config.XGB_PARAMS.copy()
    params['scale_pos_weight'] = scale_pos_weight

    print(f"\n  Training XGBoost...")
    print(f"  Class balance: {n_pos} goals / {n_neg} non-goals (ratio: {scale_pos_weight:.1f})")

    model = xgb.XGBClassifier(**params)

    # Cross-validation
    print("  Running 5-fold cross-validation...")
    cv_auc = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
    cv_logloss = cross_val_score(model, X, y, cv=5, scoring='neg_log_loss')

    print(f"  CV AUC:     {cv_auc.mean():.4f} (+/- {cv_auc.std():.4f})")
    print(f"  CV LogLoss: {-cv_logloss.mean():.4f} (+/- {cv_logloss.std():.4f})")

    # Train final model on all data
    model.fit(X, y)

    # Feature importance
    importances = model.feature_importances_
    feature_names = list(X.columns)
    importance_pairs = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1], reverse=True
    )

    print("\n  Top 10 Features:")
    for name, imp in importance_pairs[:10]:
        bar = "█" * int(imp * 50)
        print(f"    {name:<30} {imp:.4f} {bar}")

    cv_metrics = {
        'cv_auc_mean': round(float(cv_auc.mean()), 4),
        'cv_auc_std': round(float(cv_auc.std()), 4),
        'cv_logloss_mean': round(float(-cv_logloss.mean()), 4),
        'cv_logloss_std': round(float(cv_logloss.std()), 4),
        'feature_importance': {
            name: round(float(imp), 4)
            for name, imp in importance_pairs
        },
    }

    return model, cv_metrics


# =============================================================================
# FALLBACK TABLE REGENERATION
# =============================================================================
def regenerate_fallback_table(csv_path):
    """
    Regenerate the distance/angle bin lookup table from real data.
    Overwrites the pre-computed values with empirical ones.
    """
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    df['is_goal'] = df['is_goal'].astype(int)
    df['shot_distance'] = pd.to_numeric(df['shot_distance'], errors='coerce')
    df['shot_angle'] = pd.to_numeric(df['shot_angle'], errors='coerce')
    df = df.dropna(subset=['shot_distance', 'shot_angle', 'is_goal'])

    if len(df) < 200:
        return  # Not enough data to be reliable

    # Distance bins
    dist_bins = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 200)]
    dist_labels = ['0-10', '10-20', '20-30', '30-40', '40-50', '50+']
    dist_names = ['Crease / Doorstep', 'Slot', 'High Slot', 'Mid Range', 'Perimeter', 'Far Out']

    distance_data = {}
    for (lo, hi), label, name in zip(dist_bins, dist_labels, dist_names):
        mask = (df['shot_distance'] >= lo) & (df['shot_distance'] < hi)
        subset = df[mask]
        if len(subset) > 10:
            xg = round(float(subset['is_goal'].mean()), 4)
        else:
            xg = 0.066  # Fallback to league average
        distance_data[label] = {"base_xg": xg, "label": name}

    # Shot type multipliers (relative to wrist)
    shot_types = df.groupby('shot_type')['is_goal'].mean()
    baseline = shot_types.get('wrist', 0.066)
    shot_mults = {}
    for st, rate in shot_types.items():
        shot_mults[st] = round(float(rate / max(baseline, 0.01)), 2)

    table = {
        "description": "Empirical xG lookup table regenerated from training data",
        "source": f"Training data: {len(df)} shots",
        "baseline_goal_rate": round(float(df['is_goal'].mean()), 4),
        "distance_bins": distance_data,
        "shot_type_multipliers": shot_mults,
        "regenerated_at": datetime.now().isoformat(),
    }

    with open(Config.FALLBACK_TABLE, 'w') as f:
        json.dump(table, f, indent=2)

    print(f"\n  Regenerated fallback table from {len(df)} shots")


# =============================================================================
# SAVE MODEL
# =============================================================================
def save_model(model, X, cv_metrics):
    """Save XGBoost model and metadata."""
    # Save model in XGBoost native format
    model.save_model(Config.MODEL_FILE)
    print(f"\n  Saved model: {Config.MODEL_FILE}")

    # Save metadata
    metadata = {
        'trained_at': datetime.now().isoformat(),
        'num_shots': int(len(X)),
        'num_features': int(len(X.columns)),
        'feature_names': list(X.columns),
        'cv_auc': cv_metrics['cv_auc_mean'],
        'cv_logloss': cv_metrics['cv_logloss_mean'],
        'feature_importance': cv_metrics['feature_importance'],
        'xgb_params': Config.XGB_PARAMS,
    }

    with open(Config.METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata: {Config.METADATA_FILE}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    # Parse args
    min_shots = Config.MIN_SHOTS_TO_TRAIN
    for i, arg in enumerate(sys.argv[1:]):
        if arg == '--min-shots' and i + 1 < len(sys.argv) - 1:
            min_shots = int(sys.argv[i + 2])

    Config.MIN_SHOTS_TO_TRAIN = min_shots

    print("=" * 60)
    print("  NHL xG MODEL TRAINER")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # Load and preprocess
    X, y, feature_names = load_and_preprocess()

    if X is None:
        print("\n  Cannot train — insufficient data.")
        print("  Run: python collect_xg_data.py --backfill 14")
        sys.exit(0)  # Exit cleanly, not an error

    # Train
    model, cv_metrics = train_model(X, y)

    # Save
    save_model(model, X, cv_metrics)

    # Regenerate fallback table from real data
    regenerate_fallback_table(Config.SHOTS_CSV)

    print("\n  Training complete!")
    print(f"  AUC: {cv_metrics['cv_auc_mean']:.4f}")
    print(f"  LogLoss: {cv_metrics['cv_logloss_mean']:.4f}")


if __name__ == "__main__":
    main()
