"""
NHL Goal Predictor — Meta Ensemble v1
======================================
Stacked meta-learner that combines predictions from all base models
(Weighted Linear, Monte Carlo, xG XGBoost) plus goalie quality and
context features. Trains a LightGBM classifier with isotonic calibration.

Pipeline:
  1. Load today's predictions from all 3 base models
  2. Fetch starting goalie data for tonight's games
  3. Score through trained meta-model + isotonic calibrator
  4. Output in standard prediction JSON format

Training:
  Run with --train flag to build the meta-model from historical predictions.

Author: Mohammad G. Nasiri
"""

import requests
import numpy as np
import json
import os
import sys
import math
import glob
from datetime import datetime, timedelta

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from sklearn.isotonic import IsotonicRegression
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
    GOALIE_CACHE_FILE = f"{META_MODEL_DIR}/goalie_cache.json"
    TIMS_DIR = f"{DATA_DIR}/tims_players"

    TODAY = datetime.now().strftime("%Y-%m-%d")
    CURRENT_SEASON = "20242025"

    BASE_MODELS = ['neural_network', 'monte_carlo', 'xg_v3']
    MARKET_MODEL = 'market_odds'
    FEATURE_NAMES = [
        'p_linear', 'p_mc', 'p_xg', 'p_market',
        'model_disagreement',
        'goalie_sv_pct',
        'is_home',
        'opp_ga_per_game',
    ]


os.makedirs(Config.PREDICTIONS_DIR, exist_ok=True)
os.makedirs(Config.META_MODEL_DIR, exist_ok=True)

print("=" * 70)
print(f"  NHL GOAL PREDICTOR — {Config.MODEL_DISPLAY_NAME}")
print(f"  {Config.TODAY}")
print("=" * 70)


# =============================================================================
# NHL API HELPERS
# =============================================================================
def api_get(url, timeout=15):
    """Safe API GET request with retry"""
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 404:
                return None
        except requests.RequestException:
            if attempt == 2:
                return None
    return None


def get_todays_games(date):
    """Get all NHL games scheduled for a date"""
    data = api_get(f"https://api-web.nhle.com/v1/schedule/{date}")
    if not data:
        return []
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
# GOALIE DATA
# =============================================================================
def load_goalie_cache():
    """Load cached goalie stats."""
    if os.path.exists(Config.GOALIE_CACHE_FILE):
        try:
            with open(Config.GOALIE_CACHE_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_goalie_cache(cache):
    """Save goalie cache to disk."""
    with open(Config.GOALIE_CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)


def fetch_goalie_sv_pct(goalie_id, cache):
    """Fetch goalie season save percentage from game log."""
    cache_key = str(goalie_id)
    if cache_key in cache:
        cached = cache[cache_key]
        if cached.get('fetched_date', '') >= Config.TODAY:
            return cached.get('sv_pct', 0.900)

    data = api_get(f"https://api-web.nhle.com/v1/player/{goalie_id}/game-log/now")
    if not data:
        return 0.900

    game_log = data.get('gameLog', [])
    prior = [g for g in game_log if g.get('gameDate', '9999') < Config.TODAY]
    if not prior:
        return 0.900

    total_saves = 0
    total_shots_against = 0
    for g in prior:
        sa = g.get('shotsAgainst', 0)
        ga = g.get('goalsAgainst', 0)
        total_shots_against += sa
        total_saves += (sa - ga)

    sv_pct = total_saves / total_shots_against if total_shots_against > 0 else 0.900

    cache[cache_key] = {
        'sv_pct': round(sv_pct, 4),
        'games': len(prior),
        'fetched_date': Config.TODAY,
    }
    return round(sv_pct, 4)


def get_opponent_goalie_sv_pct(team_abbrev, games, goalie_cache):
    """
    For a given team, find the opponent and look up their starting goalie SV%.
    Falls back to 0.900 (league average) if starter unknown.
    """
    opponent = None
    for g in games:
        if g['home_team'] == team_abbrev:
            opponent = g['away_team']
            break
        elif g['away_team'] == team_abbrev:
            opponent = g['home_team']
            break

    if not opponent:
        return 0.900

    data = api_get(f"https://api-web.nhle.com/v1/roster/{opponent}/current")
    if not data:
        return 0.900

    goalies = data.get('goalies', [])
    if not goalies:
        return 0.900

    # Use first listed goalie (typically starter)
    return fetch_goalie_sv_pct(goalies[0]['id'], goalie_cache)


# =============================================================================
# TRAINING
# =============================================================================
def build_training_data():
    """Join historical predictions from all 3 models with actual results."""
    print("\n  Building training dataset...")

    model_dates = {}
    for m in Config.BASE_MODELS:
        files = glob.glob(f"{Config.DATA_DIR}/predictions/{m}/2026-*.json")
        model_dates[m] = {os.path.basename(f).replace('.json', ''): f for f in files}

    result_files = glob.glob(f"{Config.DATA_DIR}/results/2026-*.json")
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
        scorer_ids = set(s['player_id'] for s in result_data.get('all_scorers', []))

        preds_by_model = {}
        for m in Config.BASE_MODELS:
            with open(model_dates[m][date], 'r') as f:
                pred_data = json.load(f)
            preds_by_model[m] = {
                p['player_id']: p for p in pred_data.get('predictions', [])
            }

        # Load market odds if available for this date
        market_path = f"{Config.DATA_DIR}/predictions/{Config.MARKET_MODEL}/{date}.json"
        market_preds = {}
        if os.path.exists(market_path):
            try:
                with open(market_path, 'r') as f:
                    mdata = json.load(f)
                market_preds = {p['player_id']: p for p in mdata.get('predictions', [])}
            except Exception:
                pass

        nn_preds = preds_by_model.get('neural_network', {})
        mc_preds = preds_by_model.get('monte_carlo', {})
        xg_preds = preds_by_model.get('xg_v3', {})

        common_ids = set(nn_preds.keys()) & set(mc_preds.keys()) & set(xg_preds.keys())

        for pid in common_ids:
            nn = nn_preds[pid]
            mc = mc_preds[pid]
            xg = xg_preds[pid]
            mkt = market_preds.get(pid, {})

            p_nn = nn.get('goal_probability', 0)
            p_mc = mc.get('goal_probability', 0)
            p_xg = xg.get('goal_probability', 0)
            p_mkt = mkt.get('goal_probability', 0)

            row = {
                'p_linear': p_nn,
                'p_mc': p_mc,
                'p_xg': p_xg,
                'p_market': p_mkt,
                'model_disagreement': float(np.std([p_nn, p_mc, p_xg])),
                'goalie_sv_pct': 0.900,
                'is_home': int(mc.get('is_home', xg.get('is_home', False))),
                'opp_ga_per_game': xg.get('opp_ga_per_game',
                                          mc.get('opp_ga_per_game', 3.07)),
            }
            rows.append(row)
            labels.append(1 if pid in scorer_ids else 0)

    print(f"  Built {len(rows)} training rows ({sum(labels)} goals, "
          f"{sum(labels)/len(labels)*100:.1f}% positive rate)")
    return rows, labels


def train_meta_model():
    """Train the LightGBM meta-model with isotonic calibration."""
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

    # Isotonic calibration
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(cal_preds, y_cal)

    # Save model (LightGBM native format)
    model.save_model(Config.META_MODEL_FILE)

    # Save calibrator as JSON (safe serialization)
    cal_data = {
        'X_thresholds': calibrator.X_thresholds_.tolist(),
        'y_thresholds': calibrator.y_thresholds_.tolist(),
        'increasing': bool(calibrator.increasing_),
    }
    with open(Config.CALIBRATOR_FILE, 'w') as f:
        json.dump(cal_data, f)

    print(f"\n  Saved model: {Config.META_MODEL_FILE}")
    print(f"  Saved calibrator: {Config.CALIBRATOR_FILE}")
    return True


# =============================================================================
# DAILY PREDICTION
# =============================================================================
def load_meta_model():
    """Load the trained meta-model and calibrator."""
    if not HAS_LGB:
        print("  lightgbm not installed")
        return None, None

    if not os.path.exists(Config.META_MODEL_FILE):
        print("  No trained meta-model found. Run with --train first.")
        return None, None

    model = lgb.Booster(model_file=Config.META_MODEL_FILE)

    calibrator = None
    if HAS_SKLEARN and os.path.exists(Config.CALIBRATOR_FILE):
        with open(Config.CALIBRATOR_FILE, 'r') as f:
            cal_data = json.load(f)
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.X_thresholds_ = np.array(cal_data['X_thresholds'])
        calibrator.y_thresholds_ = np.array(cal_data['y_thresholds'])
        calibrator.increasing_ = cal_data['increasing']
        calibrator.X_min_ = calibrator.X_thresholds_[0]
        calibrator.X_max_ = calibrator.X_thresholds_[-1]
        calibrator.f_ = None  # Will use thresholds directly

    return model, calibrator


def predict_today():
    """Generate meta-ensemble predictions for today."""
    model, calibrator = load_meta_model()
    if model is None:
        return None

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
            print(f"  WARNING: {m} predictions are from {data.get('date')}, not today")
        base_preds[m] = {p['player_id']: p for p in data.get('predictions', [])}
        print(f"    {m}: {len(base_preds[m])} players")

    if len(base_preds) < 2:
        print("  ERROR: Need at least 2 base models.")
        return None

    games = get_todays_games(Config.TODAY)
    if not games:
        print("  No games today.")
        return None
    print(f"  {len(games)} games today")

    # Build team -> opponent goalie SV% map
    print("  Fetching opponent goalie data...")
    goalie_cache = load_goalie_cache()
    team_goalie_sv = {}
    all_teams = set()
    for g in games:
        all_teams.add(g['home_team'])
        all_teams.add(g['away_team'])
    for team in all_teams:
        team_goalie_sv[team] = get_opponent_goalie_sv_pct(team, games, goalie_cache)
    save_goalie_cache(goalie_cache)

    # Load market odds if available
    market_preds = {}
    market_path = f"{Config.DATA_DIR}/predictions/{Config.MARKET_MODEL}/latest.json"
    if os.path.exists(market_path):
        with open(market_path, 'r') as f:
            mdata = json.load(f)
        if mdata.get('date') == Config.TODAY:
            market_preds = {p['player_id']: p for p in mdata.get('predictions', [])}
            print(f"    market_odds: {len(market_preds)} players")
        else:
            print(f"  WARNING: market_odds is from {mdata.get('date')}, not today")

    nn = base_preds.get('neural_network', {})
    mc = base_preds.get('monte_carlo', {})
    xg = base_preds.get('xg_v3', {})
    common_ids = set(nn.keys()) & set(mc.keys()) & set(xg.keys())
    print(f"  Common players across all models: {len(common_ids)}")

    import pandas as pd
    player_rows = []
    player_meta = []

    for pid in common_ids:
        nn_p, mc_p, xg_p = nn[pid], mc[pid], xg[pid]
        mkt_p = market_preds.get(pid, {})
        p_nn = nn_p.get('goal_probability', 0)
        p_mc = mc_p.get('goal_probability', 0)
        p_xg = xg_p.get('goal_probability', 0)
        p_mkt = mkt_p.get('goal_probability', 0)

        team = mc_p.get('team', xg_p.get('team', ''))

        row = {
            'p_linear': p_nn,
            'p_mc': p_mc,
            'p_xg': p_xg,
            'p_market': p_mkt,
            'model_disagreement': float(np.std([p_nn, p_mc, p_xg])),
            'goalie_sv_pct': team_goalie_sv.get(team, 0.900),
            'is_home': int(mc_p.get('is_home', xg_p.get('is_home', False))),
            'opp_ga_per_game': xg_p.get('opp_ga_per_game',
                                        mc_p.get('opp_ga_per_game', 3.07)),
        }
        player_rows.append(row)
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
        })

    X = pd.DataFrame(player_rows, columns=Config.FEATURE_NAMES)
    raw_preds = model.predict(X)

    if calibrator is not None:
        calibrated = np.interp(raw_preds,
                               calibrator.X_thresholds_,
                               calibrator.y_thresholds_)
    else:
        calibrated = raw_preds

    all_players = []
    for i, meta in enumerate(player_meta):
        player = {
            **meta,
            'goal_probability': round(float(calibrated[i]), 4),
            'goal_probability_raw': round(float(raw_preds[i]), 4),
            'opp_goalie_sv_pct': player_rows[i]['goalie_sv_pct'],
        }
        all_players.append(player)

    all_players.sort(key=lambda x: x['goal_probability'], reverse=True)
    for i, p in enumerate(all_players):
        p['rank'] = i + 1
        p['is_hot'] = p.get('last5_goals', 0) >= 3

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
    def normalize(name):
        return name.lower().strip().replace('.', '').replace("'", "").replace('-', ' ')

    all_tims_names = set()
    tims_groups = {}
    for gid, players in tims_data['groups'].items():
        group_names = set()
        for p in players:
            name = p if isinstance(p, str) else p.get('name', '')
            group_names.add(normalize(name))
            all_tims_names.add(normalize(name))
        tims_groups[gid] = group_names

    filtered = []
    for player in all_players:
        pname = normalize(player['name'])
        if pname in all_tims_names:
            for gid, names in tims_groups.items():
                if pname in names:
                    player['tims_group'] = gid
                    break
            filtered.append(player)

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
print(f"  {'#':<4} {'Name':<25} {'Team':<5} {'Prob':>7} {'SV%':>7}")
print(f"  {'-' * 50}")
for p in output_players[:20]:
    hot = " *" if p.get('is_hot') else ""
    sv = p.get('opp_goalie_sv_pct', 0)
    print(f"  {p['rank']:<4} {p['name']:<25} {p['team']:<5} "
          f"{p['goal_probability']*100:>6.1f}% {sv:>6.3f}{hot}")

print(f"\n  Total: {len(output_players)} players ranked")
print(f"\n  Meta Ensemble v1 predictions complete!")
