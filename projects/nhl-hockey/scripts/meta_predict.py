"""
NHL Goal Predictor — Meta Ensemble v1
======================================
Stacked LightGBM meta-learner over the base models. The intersection of the
three longest-history models (Weighted Linear [dir: neural_network], Monte
Carlo, xG XGBoost) defines the row universe; market_odds, lineup_v1 and
neural_v2 are left-joined as extra features so the ensemble can exploit the
current champions it previously ignored.

Features: p_linear, p_mc, p_xg, p_market, p_lineup, p_neural_v2,
          model_disagreement, is_home, opp_ga_per_game.

Pipeline:
  1. Auto-retrain if the saved model is stale or its feature set changed
  2. Load today's predictions from all available models
  3. Score through the trained meta-model
  4. Output in standard prediction JSON format

Training:
  Run with --train to (re)build the meta-model from historical predictions.
  The daily run self-heals via model_needs_retrain() — no manual schedule.

Author: Mohammad G. Nasiri
"""

import numpy as np
import json
import os
import sys
import math
import glob
from datetime import datetime, timedelta

from nhl_api import api_get
from season_prior import last5_is_real
from tims_match import match_tims


def current_season_id(date=None):
    """NHL seasonId like '20252026' for a date. Season starts in early October;
    month >= 8 rolls the ID forward so July/Aug/early-Sep still returns the
    upcoming season and keeps API calls pointed at data that actually exists."""
    d = date or datetime.now()
    if isinstance(d, str):
        d = datetime.strptime(d, "%Y-%m-%d")
    start = d.year if d.month >= 8 else d.year - 1
    return f"{start}{start + 1}"

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from sklearn.linear_model import LogisticRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    MODEL_NAME = "meta_ensemble"
    MODEL_DISPLAY_NAME = "Meta Ensemble v1"

    DATA_DIR = "data"
    PREDICTIONS_DIR = f"{DATA_DIR}/predictions/{MODEL_NAME}"
    META_MODEL_DIR = f"{DATA_DIR}/meta_model"
    META_MODEL_FILE = f"{META_MODEL_DIR}/model.txt"
    CALIBRATOR_FILE = f"{META_MODEL_DIR}/calibrator.json"
    META_METADATA_FILE = f"{META_MODEL_DIR}/metadata.json"
    TIMS_DIR = f"{DATA_DIR}/tims_players"

    TODAY = datetime.now().strftime("%Y-%m-%d")
    CURRENT_SEASON = current_season_id()

    # The intersection of these three (longest-history) models defines the
    # training/prediction row universe.
    BASE_MODELS = ['neural_network', 'monte_carlo', 'xg_v3']
    MARKET_MODEL = 'market_odds'
    # Optional models, left-joined as extra features: dir -> feature column.
    # Missing on dates before a model existed (LightGBM handles the gap).
    # lineup_v1 and neural_v2 are current champions the old ensemble ignored.
    OPTIONAL_MODELS = {
        'market_odds': 'p_market',
        'lineup_v1': 'p_lineup',
        'neural_v2': 'p_neural_v2',
    }
    # goalie_sv_pct was dropped: it was hardcoded to 0.900 in training, so the
    # model could never split on it (zero variance) — the live goalie fetch was
    # wasted. Re-add only with real historical SV% on BOTH train and predict.
    FEATURE_NAMES = [
        'p_linear', 'p_mc', 'p_xg', 'p_market', 'p_lineup', 'p_neural_v2',
        'model_disagreement',
        'is_home',
        'opp_ga_per_game',
    ]
    # Auto-retrain if the saved model is older than this, or if its feature set
    # no longer matches FEATURE_NAMES. Prevents silently serving a stale model
    # (the prior failure mode: model 6 weeks old, ignoring newer base models).
    RETRAIN_IF_OLDER_DAYS = 7


os.makedirs(Config.PREDICTIONS_DIR, exist_ok=True)
os.makedirs(Config.META_MODEL_DIR, exist_ok=True)

print("=" * 70)
print(f"  NHL GOAL PREDICTOR — {Config.MODEL_DISPLAY_NAME}")
print(f"  {Config.TODAY}")
print("=" * 70)


# =============================================================================
# NHL API HELPERS
# =============================================================================
def get_todays_games(date):
    """Get all NHL games scheduled for a date; None if the schedule couldn't
    be fetched (distinct from [] = a day with no games)."""
    # Patient retries (~30s): a failure now stops the run, so a short
    # rate-limit window (e.g. after the base models' fetches) shouldn't.
    data = api_get(f"https://api-web.nhle.com/v1/schedule/{date}", attempts=6)
    if not data:
        return None
    games = []
    for day in data.get('gameWeek', []):
        if day['date'] == date:
            for game in day.get('games', []):
                if game.get('gameType') in [2, 3]:
                    games.append({
                        'game_id': game['id'],
                        'home_team': game['homeTeam']['abbrev'],
                        'away_team': game['awayTeam']['abbrev'],
                        'start_time': game.get('startTimeUTC', ''),
                    })
    return games


# =============================================================================
# FEATURE CONSTRUCTION (shared by training + prediction — no train/serve skew)
# =============================================================================
def _input_prob(pred):
    """A base model's probability on the scale Meta's history uses. The shot
    models (xG, lineup, Neural v2) now publish calibrated probabilities but
    keep the old scale as raw_goal_probability; older files only have the
    old scale, so this keeps training rows and today's inputs comparable."""
    return pred.get('raw_goal_probability', pred.get('goal_probability', 0.0))


def _load_optional_preds(date):
    """feature_col -> {player_id: goal_probability} for each optional model on
    a given date ('latest' or 'YYYY-MM-DD'). Missing files yield empty dicts."""
    out = {}
    for mdir, col in Config.OPTIONAL_MODELS.items():
        path = f"{Config.DATA_DIR}/predictions/{mdir}/{date}.json"
        preds = {}
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                # For 'latest' guard against a stale date upstream.
                if date != 'latest' or data.get('date') == Config.TODAY:
                    preds = {p['player_id']: _input_prob(p)
                             for p in data.get('predictions', [])}
            except Exception:
                pass
        out[col] = preds
    return out


def _build_feature_row(pid, nn, mc, xg, optional_preds):
    """Build one model-input row. Used identically in training and prediction."""
    p_nn = _input_prob(nn)
    p_mc = _input_prob(mc)
    p_xg = _input_prob(xg)
    opt = {col: preds.get(pid, 0.0) for col, preds in optional_preds.items()}
    # Disagreement over whatever base signals are present (>0) that day.
    base_probs = [p_nn, p_mc, p_xg] + [v for v in opt.values() if v]
    return {
        'p_linear': p_nn,
        'p_mc': p_mc,
        'p_xg': p_xg,
        'p_market': opt.get('p_market', 0.0),
        'p_lineup': opt.get('p_lineup', 0.0),
        'p_neural_v2': opt.get('p_neural_v2', 0.0),
        'model_disagreement': float(np.std(base_probs)) if len(base_probs) > 1 else 0.0,
        'is_home': int(mc.get('is_home', xg.get('is_home', False))),
        'opp_ga_per_game': xg.get('opp_ga_per_game',
                                  mc.get('opp_ga_per_game', 3.07)),
    }


# =============================================================================
# TRAINING
# =============================================================================
def build_training_data():
    """Join historical predictions from all 3 models with actual results."""
    print("\n  Building training dataset...")

    model_dates = {}
    for m in Config.BASE_MODELS:
        files = glob.glob(f"{Config.DATA_DIR}/predictions/{m}/20*.json")
        model_dates[m] = {os.path.basename(f).replace('.json', ''): f for f in files}

    result_files = glob.glob(f"{Config.DATA_DIR}/results/20*.json")
    result_dates = {os.path.basename(f).replace('.json', ''): f for f in result_files}

    overlap = set(model_dates[Config.BASE_MODELS[0]].keys())
    for m in Config.BASE_MODELS[1:]:
        overlap &= set(model_dates[m].keys())
    overlap &= set(result_dates.keys())
    overlap = sorted(overlap)

    print(f"  Found {len(overlap)} days with all 3 models + results")

    rows = []
    labels = []

    for date in overlap:
        with open(result_dates[date], 'r') as f:
            result_data = json.load(f)
        # Ungraded games would label their scorers 0; fetch_results
        # completes partial days on a later run.
        if result_data.get('partial'):
            continue
        scorer_ids = {s['player_id'] for s in result_data.get('all_scorers', [])
                      if s.get('player_id')}

        preds_by_model = {}
        for m in Config.BASE_MODELS:
            with open(model_dates[m][date], 'r') as f:
                pred_data = json.load(f)
            preds_by_model[m] = {
                p['player_id']: p for p in pred_data.get('predictions', [])
            }

        # Optional models (left-joined). Missing dates -> empty -> 0.0 feature.
        optional_preds = _load_optional_preds(date)

        nn_preds = preds_by_model.get('neural_network', {})
        mc_preds = preds_by_model.get('monte_carlo', {})
        xg_preds = preds_by_model.get('xg_v3', {})
        common_ids = set(nn_preds) & set(mc_preds) & set(xg_preds)

        for pid in common_ids:
            rows.append(_build_feature_row(
                pid, nn_preds[pid], mc_preds[pid], xg_preds[pid], optional_preds))
            labels.append(1 if pid in scorer_ids else 0)

    print(f"  Built {len(rows)} training rows ({sum(labels)} goals, "
          f"{sum(labels)/len(labels)*100:.1f}% positive rate)")
    return rows, labels


def train_meta_model():
    """Train the LightGBM meta-model with Platt (monotonic) calibration."""
    if not HAS_LGB:
        print("  ERROR: lightgbm not installed. pip install lightgbm")
        return False
    if not HAS_SKLEARN:
        print("  ERROR: scikit-learn not installed. pip install scikit-learn")
        return False

    rows, labels = build_training_data()
    if len(rows) < 100:
        print("  ERROR: Not enough training data (need >= 100 rows)")
        return False

    import pandas as pd
    X = pd.DataFrame(rows, columns=Config.FEATURE_NAMES)
    y = np.array(labels)

    # Time-series split: hold out last 20% for calibration
    split_idx = int(len(X) * 0.8)
    X_train, X_cal = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_cal = y[:split_idx], y[split_idx:]

    print(f"\n  Training LightGBM on {len(X_train)} rows, calibrating on {len(X_cal)}...")

    pos_weight = (len(y_train) - sum(y_train)) / max(sum(y_train), 1)
    train_data = lgb.Dataset(X_train, label=y_train)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_leaves': 15,
        'max_depth': 4,
        'learning_rate': 0.05,
        'scale_pos_weight': pos_weight,
        'verbose': -1,
        'seed': 42,
    }
    model = lgb.train(params, train_data, num_boost_round=100)

    # Evaluate on calibration set
    cal_preds = model.predict(X_cal)

    # Top-10 style hit rate on cal set
    top_indices = np.argsort(cal_preds)[-int(len(cal_preds)*0.05):]
    top_hits = sum(y_cal[top_indices])
    print(f"  Top 5% hit rate: {top_hits}/{len(top_indices)} = "
          f"{top_hits/len(top_indices)*100:.1f}%")

    # Feature importance
    importance = {k: int(v) for k, v in zip(Config.FEATURE_NAMES, model.feature_importance())}
    print(f"  Feature importance: {json.dumps(importance, indent=4)}")

    # Platt calibration (logistic): p_cal = sigmoid(a * p_raw + b). Strictly
    # monotonic, so it makes the output an honest probability WITHOUT changing
    # player ranking — hit rate is unaffected, only the displayed numbers get
    # realistic. Replaces the old isotonic calibrator, which produced flat tie
    # regions that broke ranking (the reason calibration had been disabled).
    platt = LogisticRegression(C=1e6, solver='lbfgs')
    platt.fit(cal_preds.reshape(-1, 1), y_cal)
    platt_a = float(platt.coef_[0][0])
    platt_b = float(platt.intercept_[0])
    print(f"  Platt calibration: a={platt_a:.4f}, b={platt_b:.4f}")

    # Save model (LightGBM native format)
    model.save_model(Config.META_MODEL_FILE)

    with open(Config.CALIBRATOR_FILE, 'w') as f:
        json.dump({'method': 'platt', 'a': platt_a, 'b': platt_b}, f)

    # Metadata drives auto-retrain. Use a committed `trained_at` date rather
    # than the model file's mtime, because `git checkout` in CI resets mtimes
    # to the run time (staleness by mtime would never trigger).
    with open(Config.META_METADATA_FILE, 'w') as f:
        json.dump({
            'trained_at': Config.TODAY,
            'feature_names': Config.FEATURE_NAMES,
            'n_train_rows': int(len(X_train)),
            'n_cal_rows': int(len(X_cal)),
        }, f, indent=2)

    print(f"\n  Saved model: {Config.META_MODEL_FILE}")
    print(f"  Saved calibrator: {Config.CALIBRATOR_FILE}")
    print(f"  Saved metadata: {Config.META_METADATA_FILE}")
    return True


# =============================================================================
# DAILY PREDICTION
# =============================================================================
def model_needs_retrain():
    """Return (bool, reason). Retrain when the model is missing, stale, or was
    trained on a different feature set than Config.FEATURE_NAMES."""
    if not os.path.exists(Config.META_MODEL_FILE):
        return True, "no model file"
    meta = {}
    if os.path.exists(Config.META_METADATA_FILE):
        try:
            with open(Config.META_METADATA_FILE) as f:
                meta = json.load(f)
        except Exception:
            pass
    if list(meta.get('feature_names', [])) != list(Config.FEATURE_NAMES):
        return True, "feature set changed"
    trained_at = meta.get('trained_at')
    if not trained_at:
        return True, "no trained_at metadata"
    try:
        age = (datetime.strptime(Config.TODAY, "%Y-%m-%d")
               - datetime.strptime(trained_at, "%Y-%m-%d")).days
    except ValueError:
        return True, "unparseable trained_at"
    if age > Config.RETRAIN_IF_OLDER_DAYS:
        return True, f"model is {age} days old"
    return False, f"fresh ({age}d)"


def load_meta_model():
    """Load the trained meta-model and calibrator."""
    if not HAS_LGB:
        print("  lightgbm not installed")
        return None, None

    if not os.path.exists(Config.META_MODEL_FILE):
        print("  No trained meta-model found. Run with --train first.")
        return None, None

    model = lgb.Booster(model_file=Config.META_MODEL_FILE)

    # Calibrator is a Platt (a, b) tuple, applied as sigmoid(a*p_raw + b).
    calibrator = None
    if os.path.exists(Config.CALIBRATOR_FILE):
        try:
            with open(Config.CALIBRATOR_FILE, 'r') as f:
                cal_data = json.load(f)
            if cal_data.get('method') == 'platt':
                calibrator = (float(cal_data['a']), float(cal_data['b']))
        except Exception:
            calibrator = None

    return model, calibrator


def _write_no_predictions(games, note):
    """Write today's file with no predictions. games=[] is a real off-day;
    games with no predictions is flagged by the health check."""
    output = {
        "date": Config.TODAY,
        "model": Config.MODEL_NAME,
        "model_display_name": Config.MODEL_DISPLAY_NAME,
        "games_count": len(games), "games": games, "players_count": 0,
        "predictions": [], "tims_mode": False,
        "tims_source": None, "tims_group_rankings": {},
        "generated_at": datetime.now().isoformat(),
    }
    if note:
        output["note"] = note
    for path in [f"{Config.PREDICTIONS_DIR}/{Config.TODAY}.json",
                 f"{Config.PREDICTIONS_DIR}/latest.json"]:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"  Saved empty: {path}")


def predict_today():
    """Generate meta-ensemble predictions for today."""
    # Self-healing freshness: retrain before predicting if the model is stale
    # or its feature set changed. This is what keeps the ensemble from silently
    # running a months-old model that ignores newer base models.
    need, why = model_needs_retrain()
    if need:
        print(f"  Auto-retrain triggered: {why}")
        if HAS_LGB and HAS_SKLEARN:
            train_meta_model()
        else:
            print("  lightgbm/sklearn unavailable — skipping retrain.")

    model, calibrator = load_meta_model()
    if model is None:
        return None

    # Schedule first. On 2026-09-29/30 the base models had failed, so "all
    # bases have 0 players" was read as an off-day and an empty file was
    # written on 5- and 3-game nights — which the health check accepted.
    games = get_todays_games(Config.TODAY)
    if games is None:
        print("  ERROR: could not fetch today's NHL schedule — not writing predictions.")
        sys.exit(1)
    if not games:
        print("  No games today — writing empty predictions and exiting cleanly.")
        _write_no_predictions([], None)
        sys.exit(0)
    print(f"  {len(games)} games today")

    print("\n  Loading base model predictions...")
    base_preds = {}
    for m in Config.BASE_MODELS:
        path = f"{Config.DATA_DIR}/predictions/{m}/latest.json"
        if not os.path.exists(path):
            print(f"  WARNING: Missing {m} predictions at {path}")
            continue
        with open(path, 'r') as f:
            data = json.load(f)
        if data.get('date') != Config.TODAY:
            # Stacking a stale slate onto today's games ranks players who
            # aren't playing; skip it instead.
            print(f"  WARNING: skipping {m} — predictions are from {data.get('date')}, not today")
            continue
        base_preds[m] = {p['player_id']: p for p in data.get('predictions', [])}
        print(f"    {m}: {len(base_preds[m])} players")

    # Load optional-model probabilities (market_odds, lineup_v1, neural_v2),
    # each guarded to today's date inside the helper.
    optional_preds = _load_optional_preds('latest')
    for col, preds in optional_preds.items():
        if preds:
            print(f"    {col}: {len(preds)} players")

    nn = base_preds.get('neural_network', {})
    mc = base_preds.get('monte_carlo', {})
    xg = base_preds.get('xg_v3', {})
    common_ids = set(nn.keys()) & set(mc.keys()) & set(xg.keys())
    print(f"  Common players across all models: {len(common_ids)}")

    # Games today, but a base model failed, is stale, or the slates don't
    # overlap. Write the games with no predictions: the health check fails
    # that (games but no picks) and alerts, instead of accepting a fake
    # off-day. LightGBM would also crash on an empty DataFrame.
    if not common_ids:
        missing = [m for m in Config.BASE_MODELS if not base_preds.get(m)]
        note = (f"base models missing or stale for today: {', '.join(missing)}"
                if missing else "no overlap across base models")
        print(f"  {note} — writing no predictions.")
        _write_no_predictions(games, note)
        sys.exit(0)

    import pandas as pd
    player_rows = []
    player_meta = []

    for pid in common_ids:
        nn_p, mc_p, xg_p = nn[pid], mc[pid], xg[pid]
        player_rows.append(_build_feature_row(pid, nn_p, mc_p, xg_p, optional_preds))

        team = mc_p.get('team', xg_p.get('team', ''))
        player_meta.append({
            'player_id': pid,
            'name': mc_p.get('name', xg_p.get('name', '')),
            'position': mc_p.get('position', xg_p.get('position', '')),
            'team': team,
            'opponent': mc_p.get('opponent', xg_p.get('opponent', '')),
            'is_home': mc_p.get('is_home', xg_p.get('is_home', False)),
            'game_id': mc_p.get('game_id', xg_p.get('game_id', '')),
            'matchup': mc_p.get('matchup', xg_p.get('matchup', '')),
            'season_goals': mc_p.get('season_goals', xg_p.get('season_goals', 0)),
            'last5_goals': mc_p.get('last5_goals', xg_p.get('last5_goals', 0)),
            'games_played': mc_p.get('games_played', xg_p.get('games_played', 0)),
            'prior_games': mc_p.get('prior_games', xg_p.get('prior_games', 0)),
        })

    X = pd.DataFrame(player_rows, columns=Config.FEATURE_NAMES)
    raw_preds = model.predict(X)

    # Apply Platt calibration: p_cal = sigmoid(a*p_raw + b). It is strictly
    # monotonic, so the ranking (and thus hit rate) is identical to raw — only
    # the displayed probabilities become realistic. Guarded: any failure falls
    # back to raw output (the previous safe behavior). This replaces the old
    # isotonic path, which broke ranking via flat tie regions.
    calibrated = np.asarray(raw_preds, dtype=float)
    if calibrator is not None:
        try:
            a, b = calibrator
            calibrated = 1.0 / (1.0 + np.exp(-(a * calibrated + b)))
        except Exception as e:
            print(f"  Calibration failed ({e}); using raw output.")
            calibrated = np.asarray(raw_preds, dtype=float)

    all_players = []
    for i, meta in enumerate(player_meta):
        player = {
            **meta,
            'goal_probability': round(float(calibrated[i]), 4),
            'goal_probability_raw': round(float(raw_preds[i]), 4),
        }
        all_players.append(player)

    all_players.sort(key=lambda x: x['goal_probability'], reverse=True)
    for i, p in enumerate(all_players):
        p['rank'] = i + 1
        p['is_hot'] = p.get('last5_goals', 0) >= 3 and last5_is_real(p)

    return all_players, games


# =============================================================================
# TIM HORTONS
# =============================================================================
def load_tims_players(date):
    """Load Tim Hortons eligible players."""
    tims_file = f"{Config.TIMS_DIR}/{date}.json"
    if os.path.exists(tims_file):
        try:
            with open(tims_file, 'r') as f:
                data = json.load(f)
            count = sum(len(v) for v in data.get('groups', {}).values())
            print(f"  Tim Hortons players: {count} (source: {data.get('source', '?')})")
            return data
        except Exception:
            pass
    return None


# =============================================================================
# MAIN
# =============================================================================
if '--train' in sys.argv:
    print("\n  TRAINING MODE")
    success = train_meta_model()
    if not success:
        sys.exit(1)
    if '--predict' not in sys.argv:
        print("\n  Training complete. Run without --train for predictions.")
        sys.exit(0)

print("\n  PREDICTION MODE")
result = predict_today()
if result is None:
    print("  Could not generate predictions. Is the meta-model trained?")
    sys.exit(1)

all_players, todays_games = result

# Tim Hortons filtering
tims_data = load_tims_players(Config.TODAY)
tims_mode = tims_data is not None and bool(tims_data.get('groups'))
tims_group_rankings = {}
output_players = all_players

if tims_mode:
    filtered, _ = match_tims(all_players, tims_data, Config.MODEL_NAME)

    if filtered:
        for i, p in enumerate(filtered):
            p['tims_rank'] = i + 1
        output_players = filtered
        for p in filtered:
            gid = p.get('tims_group')
            if gid:
                if gid not in tims_group_rankings:
                    tims_group_rankings[gid] = []
                tims_group_rankings[gid].append({
                    'rank_in_group': len(tims_group_rankings[gid]) + 1,
                    'player_id': p['player_id'],
                    'name': p['name'],
                    'team': p['team'],
                    'goal_probability': p['goal_probability'],
                    'matchup': p.get('matchup', ''),
                })

# Save output
print("\n  Saving predictions...")
output = {
    "date": Config.TODAY,
    "model": Config.MODEL_NAME,
    "model_display_name": Config.MODEL_DISPLAY_NAME,
    "games_count": len(todays_games),
    "games": todays_games,
    "players_count": len(output_players),
    "predictions": output_players,
    "tims_mode": tims_mode,
    "tims_source": tims_data.get('source', 'unknown') if tims_data else None,
    "tims_group_rankings": tims_group_rankings if tims_mode else {},
    "model_params": {
        "base_models": Config.BASE_MODELS,
        "features": Config.FEATURE_NAMES,
    },
    "generated_at": datetime.now().isoformat()
}

for path in [f"{Config.PREDICTIONS_DIR}/{Config.TODAY}.json",
             f"{Config.PREDICTIONS_DIR}/latest.json"]:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {path}")

# Console output
print("\n" + "=" * 70)
print("  TOP 20 META ENSEMBLE PREDICTIONS")
print("=" * 70)
print(f"  {'#':<4} {'Name':<25} {'Team':<5} {'Prob':>7}")
print(f"  {'-' * 43}")
for p in output_players[:20]:
    hot = " *" if p.get('is_hot') else ""
    print(f"  {p['rank']:<4} {p['name']:<25} {p['team']:<5} "
          f"{p['goal_probability']*100:>6.1f}%{hot}")

print(f"\n  Total: {len(output_players)} players ranked")
print(f"\n  Meta Ensemble v1 predictions complete!")
