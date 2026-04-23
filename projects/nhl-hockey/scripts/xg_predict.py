"""
NHL Goal Predictor — xG XGBoost v3
===================================
Uses a trained xG model (or fallback lookup table) to predict which
players are most likely to score tonight.

Pipeline:
  1. Load trained XGBoost model (or fallback table)
  2. Fetch today's games, rosters, player stats
  3. For each player: estimate shot profile -> score through xG model
  4. Convert per-shot xG to game-level goal probability
  5. Rank and output in standard prediction JSON format

Author: Mohammad G. Nasiri
"""

import requests
import numpy as np
import json
import os
import sys
import math
from datetime import datetime


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
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# =============================================================================
# MODEL VARIANT FLAGS
# =============================================================================
# Multiple model variants share this script to keep logic in one place:
#   (no flag)         -> xg_v3: real-shot xG with retrained model
#   --synthetic-only  -> xg_v3_synthetic: Phase 4 A/B shadow, pre-upgrade behavior
#   --lineup-adjusted -> lineup_v1: xg_v3 + scale expected_shots by
#                        (recent TOI / season TOI) so scratches/demotions
#                        don't keep ranking on their full-minutes history
# These are mutually exclusive; lineup-adjusted wins if both are given.
LINEUP_ADJUSTED = "--lineup-adjusted" in sys.argv
SYNTHETIC_ONLY = not LINEUP_ADJUSTED and "--synthetic-only" in sys.argv


# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    if LINEUP_ADJUSTED:
        MODEL_NAME = "lineup_v1"
        MODEL_DISPLAY_NAME = "Lineup TOI v1"
    elif SYNTHETIC_ONLY:
        MODEL_NAME = "xg_v3_synthetic"
        MODEL_DISPLAY_NAME = "xG XGBoost v3 (synthetic shadow)"
    else:
        MODEL_NAME = "xg_v3"
        MODEL_DISPLAY_NAME = "xG XGBoost v3"

    DATA_DIR = "data"
    PREDICTIONS_DIR = f"{DATA_DIR}/predictions/{MODEL_NAME}"
    TIMS_DIR = f"{DATA_DIR}/tims_players"
    XG_MODEL_DIR = f"{DATA_DIR}/xg_model"
    MODEL_FILE = f"{XG_MODEL_DIR}/model.json"
    METADATA_FILE = f"{XG_MODEL_DIR}/metadata.json"
    FALLBACK_TABLE = f"{XG_MODEL_DIR}/fallback_table.json"
    PLAYER_SHOTS_DIR = f"{DATA_DIR}/player_shots"
    POSITION_PRIORS_FILE = f"{PLAYER_SHOTS_DIR}/_position_priors.json"

    TODAY = datetime.now().strftime("%Y-%m-%d")
    CURRENT_SEASON = current_season_id()

    LEAGUE_AVG_GOALS = 3.07
    HOME_ADVANTAGE = 1.026
    MIN_GAMES_PLAYED = 3

    # Synthetic shot generation parameters
    NUM_REPRESENTATIVE_SHOTS = 20  # Shots to score through xG model per player
    DEFAULT_DISTANCE_MEAN = 32.0
    DEFAULT_DISTANCE_STD = 12.0
    DEFAULT_ANGLE_MEAN = 20.0
    DEFAULT_ANGLE_STD = 12.0

    SHOT_TYPE_DISTRIBUTION = {
        'wrist': 0.50, 'snap': 0.15, 'slap': 0.15,
        'backhand': 0.10, 'tip-in': 0.05, 'deflected': 0.03,
        'wrap-around': 0.02,
    }

    # Position-specific shot profiles (angle + shot-type distributions)
    POSITION_PROFILES = {
        'D': {'angle_mean': 28, 'angle_std': 15,
              'shot_types': {'wrist': 0.30, 'slap': 0.30, 'snap': 0.20,
                             'tip-in': 0.05, 'backhand': 0.05,
                             'deflected': 0.05, 'wrap-around': 0.05}},
        'C': {'angle_mean': 18, 'angle_std': 10,
              'shot_types': {'wrist': 0.45, 'snap': 0.20, 'backhand': 0.12,
                             'tip-in': 0.10, 'slap': 0.05,
                             'deflected': 0.05, 'wrap-around': 0.03}},
        'L': {'angle_mean': 22, 'angle_std': 11,
              'shot_types': {'wrist': 0.42, 'snap': 0.22, 'slap': 0.10,
                             'backhand': 0.10, 'tip-in': 0.08,
                             'deflected': 0.05, 'wrap-around': 0.03}},
        'R': {'angle_mean': 22, 'angle_std': 11,
              'shot_types': {'wrist': 0.42, 'snap': 0.22, 'slap': 0.10,
                             'backhand': 0.10, 'tip-in': 0.08,
                             'deflected': 0.05, 'wrap-around': 0.03}},
    }


os.makedirs(Config.PREDICTIONS_DIR, exist_ok=True)

print("=" * 70)
print(f"  NHL GOAL PREDICTOR — {Config.MODEL_DISPLAY_NAME}")
print(f"  {Config.TODAY}")
print("=" * 70)


# =============================================================================
# NHL API HELPERS (reused from monte_carlo_predict.py)
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


def _parse_toi_minutes(toi_str):
    """Parse NHL TOI string 'MM:SS' to float minutes. Returns 0 on malformed."""
    try:
        parts = str(toi_str).split(':')
        return int(parts[0]) + int(parts[1]) / 60 if len(parts) == 2 else 0
    except (ValueError, IndexError):
        return 0


def get_todays_games(date):
    """Get all NHL games scheduled for today"""
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


def get_team_stats_from_standings():
    """Fetch all team stats from standings API."""
    data = api_get("https://api-web.nhle.com/v1/standings/now")
    if not data:
        return {}
    team_stats = {}
    for team in data.get('standings', []):
        abbrev = team.get('teamAbbrev', {}).get('default', '')
        gp = max(team.get('gamesPlayed', 1), 1)
        gf = team.get('goalFor', 0)
        ga = team.get('goalAgainst', 0)
        team_stats[abbrev] = {
            'games_played': gp,
            'goals_for': gf,
            'goals_against': ga,
            'gf_per_game': round(gf / gp, 3),
            'ga_per_game': round(ga / gp, 3),
        }
    return team_stats


def get_player_stats(player_id):
    """Get player stats from game log (leak-free: excludes today's games).

    Pulls the current season's regular-season (type 2) and playoff (type 3)
    logs explicitly and merges them newest-first. The `/game-log/now` endpoint
    silently switches to playoff-only data once a player's team enters the
    postseason, leaving healthy scorers with 0-3 games and causing them to
    fail the MIN_GAMES_PLAYED filter — this call pattern avoids that."""
    season = Config.CURRENT_SEASON
    reg = api_get(f"https://api-web.nhle.com/v1/player/{player_id}/game-log/{season}/2")
    po = api_get(f"https://api-web.nhle.com/v1/player/{player_id}/game-log/{season}/3")
    game_log = (reg.get('gameLog', []) if reg else []) + \
               (po.get('gameLog', []) if po else [])
    if not game_log:
        return None
    # Merge, newest-first, and strip today's games to stay leak-free
    game_log.sort(key=lambda g: g.get('gameDate', ''), reverse=True)
    prior = [g for g in game_log if g.get('gameDate', '9999') < Config.TODAY]
    gp = len(prior)
    if gp < Config.MIN_GAMES_PLAYED:
        return None
    goals = sum(g.get('goals', 0) for g in prior)
    shots = sum(g.get('shots', 0) for g in prior)
    assists = sum(g.get('assists', 0) for g in prior)
    points = sum(g.get('points', 0) for g in prior)
    pp_goals = sum(g.get('powerPlayGoals', 0) for g in prior)
    pp_points = sum(g.get('powerPlayPoints', 0) for g in prior)
    total_toi = sum(_parse_toi_minutes(g.get('toi', '0:00')) for g in prior)
    shooting_pct = goals / shots if shots > 0 else 0

    last5 = prior[:5]  # newest-first
    last5_goals = sum(g.get('goals', 0) for g in last5)
    last5_shots = sum(g.get('shots', 0) for g in last5)
    last5_count = len(last5)
    last5_toi = sum(_parse_toi_minutes(g.get('toi', '0:00')) for g in last5)

    return {
        'games_played': gp,
        'season_goals': goals,
        'season_shots': shots,
        'season_assists': assists,
        'season_points': points,
        'pp_goals': pp_goals,
        'pp_points': pp_points,
        'avg_toi_minutes': round(total_toi / gp, 1) if gp > 0 else 0,
        'recent_toi_minutes': round(last5_toi / last5_count, 1) if last5_count > 0 else 0,
        'avg_goals': round(goals / gp, 4),
        'avg_shots': round(shots / gp, 2) if gp > 0 else 0,
        'shooting_pct': round(shooting_pct, 4),
        'last5_goals': last5_goals,
        'last5_shots': last5_shots,
        'last5_games': last5_count,
        'recent_gpg': round(last5_goals / max(last5_count, 1), 4),
    }


def get_team_roster(team_abbrev):
    """Get team roster (forwards + defensemen)"""
    data = api_get(f"https://api-web.nhle.com/v1/roster/{team_abbrev}/current")
    if not data:
        return []
    players = []
    for group in ['forwards', 'defensemen']:
        for p in data.get(group, []):
            players.append({
                'player_id': p['id'],
                'name': f"{p['firstName']['default']} {p['lastName']['default']}",
                'position': p['positionCode'],
                'team': team_abbrev,
            })
    return players


# =============================================================================
# XG MODEL LOADING
# =============================================================================
def load_xg_model():
    """
    Load the trained XGBoost model and its metadata.
    Falls back to the lookup table if model is unavailable.

    Returns: (model_or_None, metadata_or_None, fallback_table, mode)
    """
    model = None
    metadata = None
    fallback = None
    mode = "fallback"

    # Try loading XGBoost model
    if HAS_XGB and os.path.exists(Config.MODEL_FILE):
        try:
            model = xgb.XGBClassifier()
            model.load_model(Config.MODEL_FILE)
            if os.path.exists(Config.METADATA_FILE):
                with open(Config.METADATA_FILE, 'r') as f:
                    metadata = json.load(f)
            mode = "xgboost"
            print(f"  Loaded XGBoost model (AUC: {metadata.get('cv_auc', '?')})")
        except Exception as e:
            print(f"  Could not load XGBoost model: {e}")
            model = None
            mode = "fallback"

    # Load fallback table
    if os.path.exists(Config.FALLBACK_TABLE):
        try:
            with open(Config.FALLBACK_TABLE, 'r') as f:
                fallback = json.load(f)
        except Exception:
            pass

    if mode == "fallback":
        if fallback:
            print("  Using fallback xG lookup table")
        else:
            print("  WARNING: No model or fallback table found!")

    return model, metadata, fallback, mode


# =============================================================================
# REAL-SHOT LOADING (from aggregate_player_shots.py output)
# =============================================================================
_POSITION_PRIORS_CACHE = {"loaded": False, "data": {}}


def load_position_priors():
    """Lazy-load position-based shot priors. Cold-start fallback when a
    player has no per-player shot file (rookies, call-ups, low-volume)."""
    if not _POSITION_PRIORS_CACHE["loaded"]:
        _POSITION_PRIORS_CACHE["loaded"] = True
        path = Config.POSITION_PRIORS_FILE
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    _POSITION_PRIORS_CACHE["data"] = json.load(f)
            except Exception as e:
                print(f"  Could not load position priors: {e}")
    return _POSITION_PRIORS_CACHE["data"]


def load_real_shots(player_id, position):
    """Return (shots, source) for a player.

    source in {"real", "position_prior", None} — None means caller should
    fall back to synthetic shot generation (cold-start rookie with no
    priors, or aggregator hasn't run yet)."""
    # First choice: player-specific recent shots
    if player_id:
        player_file = f"{Config.PLAYER_SHOTS_DIR}/{player_id}.json"
        if os.path.exists(player_file):
            try:
                with open(player_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                shots = data.get("shots")
                if shots:
                    return shots, "real"
            except Exception:
                pass  # fall through

    # Second choice: position-based prior
    priors = load_position_priors()
    pos_key = (position or "C")[0].upper()
    if pos_key not in ("C", "L", "R", "D"):
        pos_key = "C"
    prior = priors.get(pos_key)
    if prior:
        shots = prior.get("shots")
        if shots:
            return shots, "position_prior"

    # Neither available — caller synthesizes
    return None, None


# =============================================================================
# SHOT PROFILE ESTIMATION
# =============================================================================
def estimate_shot_profile(player, position):
    """
    Estimate a player's typical shot characteristics from their stats.

    High shooting% -> closer shots (lower distance)
    Defensemen -> farther shots on average
    High PP involvement -> more power play shots
    """
    shooting_pct = player.get('shooting_pct', 0.066)
    avg_shots = player.get('avg_shots', 2.0)
    pp_goals = player.get('pp_goals', 0)
    season_goals = max(player.get('season_goals', 1), 1)

    # Estimate average shot distance from shooting percentage
    # Higher shooting% implies closer / higher-quality shots
    # League avg shooting% ~10%, avg distance ~32ft
    if shooting_pct > 0:
        distance_mean = max(10, min(50, 45 - shooting_pct * 150))
    else:
        distance_mean = Config.DEFAULT_DISTANCE_MEAN

    # Defensemen shoot from farther
    if position in ('D', 'LD', 'RD'):
        distance_mean += 12
        distance_std = 15.0
    else:
        distance_std = Config.DEFAULT_DISTANCE_STD

    # Power play shot fraction
    pp_fraction = pp_goals / season_goals if season_goals > 0 else 0.15

    # Rebound likelihood: forwards in traffic get more rebounds
    if position in ('C', 'L', 'R', 'LW', 'RW'):
        rebound_fraction = 0.08
    else:
        rebound_fraction = 0.03

    # Position-specific angle and shot-type distributions
    pos_key = position[0] if position else 'C'  # normalize LD/RD→D, LW/RW→L/R
    if pos_key in ('D',):
        pos_key = 'D'
    pos_profile = Config.POSITION_PROFILES.get(pos_key, Config.POSITION_PROFILES['C'])

    return {
        'expected_shots': avg_shots,
        'distance_mean': distance_mean,
        'distance_std': distance_std,
        'angle_mean': pos_profile['angle_mean'],
        'angle_std': pos_profile['angle_std'],
        'shot_types': pos_profile['shot_types'],
        'pp_fraction': min(pp_fraction, 0.5),
        'rebound_fraction': rebound_fraction,
    }


# =============================================================================
# XG CALCULATION
# =============================================================================
def calculate_xg_with_model(model, metadata, profile, num_shots=20,
                            player_id=0, position=None):
    """Score a player's shots through XGBoost and return (avg_xg, source).

    Tries three sources in order:
      1. Real shots from data/player_shots/{player_id}.json (populated by
         aggregate_player_shots.py)
      2. Position prior from data/player_shots/_position_priors.json
      3. Synthetic shots from the player's estimated profile (original
         behavior — used when aggregator hasn't run or player has no data)

    Returning the source lets the caller tag predictions so A/B analysis
    can look at hit rate broken down by data quality.
    """
    if not HAS_PANDAS:
        return None, None

    feature_names = metadata.get('feature_names', [])
    if not feature_names:
        return None, None

    # ── Source 1 & 2: real shots or position prior ─────────────────────
    # In --synthetic-only shadow mode, skip real-shot lookup entirely so the
    # output reproduces the pre-upgrade behavior as a baseline control.
    loaded_shots, loaded_source = (
        (None, None) if SYNTHETIC_ONLY
        else load_real_shots(player_id, position)
    )
    if loaded_shots:
        df = pd.DataFrame(loaded_shots)
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_names]
        try:
            probas = model.predict_proba(df)[:, 1]
            return float(np.mean(probas)), loaded_source
        except Exception as e:
            # Real-shot path failed (corrupt file, schema mismatch) — fall
            # through to synthetic so the pipeline keeps producing output.
            print(f"    WARN: real-shot scoring failed for {player_id}: {e}")

    # ── Source 3: synthetic shots (fallback) ────────────────────────────
    # Generate synthetic shots — seed per player+date for reproducibility
    np.random.seed(hash(str(player_id) + Config.TODAY) % 2**31)

    distances = np.clip(
        np.random.normal(profile['distance_mean'], profile['distance_std'], num_shots),
        2, 80
    )
    angles = np.clip(
        np.abs(np.random.normal(profile['angle_mean'], profile['angle_std'], num_shots)),
        0, 80
    )

    # Use position-specific shot types if available, else global fallback
    pos_shot_types = profile.get('shot_types', Config.SHOT_TYPE_DISTRIBUTION)
    shot_types_list = list(pos_shot_types.keys())
    shot_types_probs = list(pos_shot_types.values())
    shot_types = np.random.choice(shot_types_list, size=num_shots, p=shot_types_probs)

    is_pp = np.random.binomial(1, profile['pp_fraction'], num_shots)
    is_rebound = np.random.binomial(1, profile['rebound_fraction'], num_shots)

    # Vary period and score differential across synthetic shots
    periods = np.random.choice([1, 2, 3], num_shots, p=[0.33, 0.34, 0.33])
    score_diffs = np.random.choice([-2, -1, 0, 1, 2], num_shots,
                                   p=[0.10, 0.20, 0.40, 0.20, 0.10])

    rows = []
    for i in range(num_shots):
        row = {
            'shot_distance': distances[i],
            'shot_angle': angles[i],
            'is_powerplay': int(is_pp[i]),
            'is_rebound': int(is_rebound[i]),
            'seconds_since_last_event': 5.0 if is_rebound[i] else 15.0,
            'is_empty_net': 0,
            'score_differential': int(score_diffs[i]),
            'period': int(periods[i]),
            'prior_event_distance': distances[i] + 10 if not is_rebound[i] else distances[i] + 3,
            'distance_x_angle': distances[i] * angles[i],
            'is_slot_shot': int(distances[i] < 20),
            'is_high_danger': int(distances[i] < 30 and angles[i] < 30),
        }

        # One-hot encode shot_type
        for st in shot_types_list:
            row[f'shot_type_{st}'] = 1 if shot_types[i] == st else 0

        # One-hot encode strength_state
        strength = '5v4' if is_pp[i] else '5v5'
        for ss in ['5v5', '5v4', '4v5', '5v3', '3v5', '4v4', '3v3']:
            row[f'strength_state_{ss}'] = 1 if strength == ss else 0

        rows.append(row)

    df = pd.DataFrame(rows)

    # Align columns with model's expected features
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    # Predict
    try:
        probas = model.predict_proba(df)[:, 1]
        return float(np.mean(probas)), "synthetic"
    except Exception:
        return None, None


def calculate_xg_with_fallback(fallback, profile):
    """
    Calculate average xG per shot using the lookup table.
    Returns average xG per shot.
    """
    if not fallback:
        return 0.066  # League average

    dist_bins = fallback.get('distance_bins', {})
    shot_mults = fallback.get('shot_type_multipliers', {})
    bonus = fallback.get('bonus_modifiers', {})

    # Determine distance bin
    dist = profile['distance_mean']
    if dist < 10:
        base_xg = dist_bins.get('0-10', {}).get('base_xg', 0.22)
    elif dist < 20:
        base_xg = dist_bins.get('10-20', {}).get('base_xg', 0.14)
    elif dist < 30:
        base_xg = dist_bins.get('20-30', {}).get('base_xg', 0.07)
    elif dist < 40:
        base_xg = dist_bins.get('30-40', {}).get('base_xg', 0.04)
    elif dist < 50:
        base_xg = dist_bins.get('40-50', {}).get('base_xg', 0.025)
    else:
        base_xg = dist_bins.get('50+', {}).get('base_xg', 0.015)

    # Angle multiplier (averaged)
    angle = profile['angle_mean']
    angle_mults = fallback.get('angle_multipliers', {})
    if angle < 15:
        angle_mult = angle_mults.get('0-15', 1.1)
    elif angle < 30:
        angle_mult = angle_mults.get('15-30', 1.0)
    elif angle < 45:
        angle_mult = angle_mults.get('30-45', 0.75)
    else:
        angle_mult = angle_mults.get('45-60', 0.5)

    # PP bonus
    pp_bonus = profile['pp_fraction'] * bonus.get('is_powerplay_bonus', 0.015)

    # Rebound bonus
    rebound_bonus = profile['rebound_fraction'] * bonus.get('is_rebound', 0.05)

    avg_xg = base_xg * angle_mult + pp_bonus + rebound_bonus

    return max(avg_xg, 0.005)


# =============================================================================
# PLAYER GOAL PROBABILITY
# =============================================================================
def calculate_player_goal_probability(player, profile, model, metadata,
                                       fallback, team_stats, mode):
    """
    Calculate the probability of a player scoring at least one goal tonight.

    Steps:
      1. Get average xG per shot (from model or fallback)
      2. Estimate expected shots tonight (adjusted for opponent)
      3. Calculate total expected goals: avg_xG × expected_shots
      4. Convert to probability: P(>=1 goal) = 1 - e^(-total_xG)
    """
    # Step 1: Average xG per shot. Tracks which data source was used so
    # the output JSON can carry that label for later A/B analysis.
    xg_source = "lookup_table"
    if mode == "xgboost" and model is not None:
        avg_shot_xg, xg_source = calculate_xg_with_model(
            model, metadata, profile, Config.NUM_REPRESENTATIVE_SHOTS,
            player_id=player.get('player_id', 0),
            position=player.get('position'),
        )
        if avg_shot_xg is None:
            avg_shot_xg = calculate_xg_with_fallback(fallback, profile)
            xg_source = "lookup_table"
    else:
        avg_shot_xg = calculate_xg_with_fallback(fallback, profile)

    # Step 2: Expected shots tonight
    base_shots = profile['expected_shots']

    # Opponent adjustment
    opponent = player.get('opponent', '')
    opp_ga = Config.LEAGUE_AVG_GOALS
    if opponent in team_stats:
        opp_ga = team_stats[opponent]['ga_per_game']

    opp_factor = opp_ga / Config.LEAGUE_AVG_GOALS

    # Home/away adjustment
    is_home = player.get('is_home', False)
    ha_factor = Config.HOME_ADVANTAGE if is_home else (2.0 - Config.HOME_ADVANTAGE)

    expected_shots = base_shots * opp_factor * ha_factor

    # Recent form blend (40% recent, 60% season)
    recent_gpg = player.get('recent_gpg', 0)
    season_gpg = player.get('avg_goals', 0)
    if recent_gpg > season_gpg * 1.5 and season_gpg > 0:
        # Hot player: boost expected shots slightly
        expected_shots *= 1.1
    elif recent_gpg > 0 and season_gpg > 0:
        form_ratio = recent_gpg / season_gpg
        expected_shots *= (0.6 + 0.4 * min(form_ratio, 2.0))

    # Lineup v1 adjustment: scale expected_shots by how much the player is
    # ACTUALLY playing lately. A 4th-line guy bumped to the top line gets
    # boosted; a healthy scratch gets zeroed out. Without this, scratches
    # and demotions keep ranking on their full-minutes career average.
    toi_factor = 1.0
    if LINEUP_ADJUSTED:
        recent_toi = player.get('recent_toi_minutes', 0)
        season_toi = player.get('avg_toi_minutes', 0)
        if season_toi > 0:
            toi_factor = recent_toi / season_toi
            # Clamp to a sane range: zero-ish TOI = effectively scratched;
            # 1.5x = called-up role change. Beyond those, noise dominates.
            toi_factor = max(0.05, min(1.5, toi_factor))
        else:
            # No season TOI means we have no way to normalize; fall back to
            # absolute minutes vs a league-typical 15 min/game.
            toi_factor = max(0.05, min(1.5, recent_toi / 15.0)) if recent_toi else 0.3
        expected_shots *= toi_factor

    # Step 3: Total expected goals
    player_xg = avg_shot_xg * expected_shots

    # Step 4: Poisson probability of at least 1 goal
    prob = 1.0 - math.exp(-player_xg)

    result = {
        'goal_probability': round(prob, 4),
        'player_xg': round(player_xg, 4),
        'expected_shots': round(expected_shots, 2),
        'avg_shot_xg': round(avg_shot_xg, 4),
        'xg_source': xg_source,
    }
    if LINEUP_ADJUSTED:
        result['toi_factor'] = round(toi_factor, 3)
        result['recent_toi_minutes'] = player.get('recent_toi_minutes', 0)
        result['avg_toi_minutes'] = player.get('avg_toi_minutes', 0)
    return result


# =============================================================================
# TIM HORTONS (same pattern as MC v2)
# =============================================================================
def load_tims_players(date):
    """Load Tim Hortons eligible players."""
    tims_file = f"{Config.TIMS_DIR}/{date}.json"
    if os.path.exists(tims_file):
        try:
            with open(tims_file, 'r') as f:
                data = json.load(f)
            count = sum(len(v) for v in data.get('groups', {}).values())
            source = data.get('source', 'unknown')
            print(f"  Loaded Tim Hortons players: {count} players (source: {source})")
            if source == 'cached_pool':
                print(f"  ⚠️ WARNING: Using cached pool — players may not match Tim Hortons app")
            return data
        except Exception:
            pass
    return None


def filter_tims_players(all_players, tims_data):
    """Filter predictions to Tim Hortons eligible players."""
    if not tims_data or 'groups' not in tims_data:
        return all_players, {}

    def normalize(name):
        return name.lower().strip().replace('.', '').replace("'", "").replace('-', ' ')

    tims_groups = {}
    all_tims_names = set()
    for group_id, players in tims_data['groups'].items():
        group_names = set()
        for p in players:
            name = p if isinstance(p, str) else p.get('name', '')
            group_names.add(normalize(name))
            all_tims_names.add(normalize(name))
        tims_groups[group_id] = group_names

    filtered = []
    player_groups = {}
    for player in all_players:
        pname = normalize(player['name'])
        if pname in all_tims_names:
            for gid, names in tims_groups.items():
                if pname in names:
                    player['tims_group'] = gid
                    player_groups[player['player_id']] = gid
                    break
            filtered.append(player)

    return filtered, player_groups


# =============================================================================
# MAIN PIPELINE
# =============================================================================

# Step 1: Load xG model
print("\n  Loading xG model...")
xg_model, xg_metadata, xg_fallback, xg_mode = load_xg_model()

# Step 2: Fetch today's games
print("\n  Fetching today's games...")
todays_games = get_todays_games(Config.TODAY)

if not todays_games:
    print("  No games found for today!")
    output = {
        "date": Config.TODAY,
        "model": Config.MODEL_NAME,
        "model_display_name": Config.MODEL_DISPLAY_NAME,
        "games_count": 0, "games": [], "players_count": 0,
        "predictions": [], "tims_mode": False,
        "generated_at": datetime.now().isoformat()
    }
    for path in [f"{Config.PREDICTIONS_DIR}/{Config.TODAY}.json",
                 f"{Config.PREDICTIONS_DIR}/latest.json"]:
        with open(path, 'w') as f:
            json.dump(output, f, indent=2)
    print("  Saved empty predictions")
    sys.exit(0)

print(f"  Found {len(todays_games)} games")
for g in todays_games:
    print(f"    {g['away_team']} @ {g['home_team']}")

# Step 3: Fetch team standings
print("\n  Fetching team standings...")
team_stats = get_team_stats_from_standings()
print(f"  Loaded stats for {len(team_stats)} teams")

# Step 4: Load Tim Hortons players
print("\n  Checking Tim Hortons player pool...")
tims_data = load_tims_players(Config.TODAY)
tims_mode = tims_data is not None

# Step 5: Fetch rosters and player stats
print("\n  Fetching rosters and player stats...")

all_teams = set()
game_matchups = {}
for g in todays_games:
    all_teams.add(g['home_team'])
    all_teams.add(g['away_team'])
    game_matchups[g['home_team']] = {
        'opponent': g['away_team'], 'is_home': True, 'game_id': g['game_id']
    }
    game_matchups[g['away_team']] = {
        'opponent': g['home_team'], 'is_home': False, 'game_id': g['game_id']
    }

all_players = []
for team in sorted(all_teams):
    print(f"    Fetching {team}...", end=" ")
    roster = get_team_roster(team)
    team_players = []
    for p in roster:
        stats = get_player_stats(p['player_id'])
        if stats:
            p.update(stats)
            matchup = game_matchups.get(team, {})
            p['opponent'] = matchup.get('opponent', '')
            p['is_home'] = matchup.get('is_home', False)
            p['game_id'] = matchup.get('game_id', '')
            p['matchup'] = (
                f"{team} vs {p['opponent']}" if p['is_home']
                else f"{team} @ {p['opponent']}"
            )
            team_players.append(p)
    all_players.extend(team_players)
    print(f"{len(team_players)} players")

print(f"\n  Total players with stats: {len(all_players)}")
if not all_players:
    print("  No player data available!")
    sys.exit(1)

# Step 6: Filter Tim Hortons players
player_groups = {}
if tims_mode:
    print("\n  Filtering for Tim Hortons eligible players...")
    filtered_players, player_groups = filter_tims_players(all_players, tims_data)
    if filtered_players:
        print(f"    Matched {len(filtered_players)} Tim Hortons players")
        tims_player_ids = set(p['player_id'] for p in filtered_players)
    else:
        print("    No Tim Hortons players matched, using all players")
        tims_mode = False
        tims_player_ids = None
else:
    tims_player_ids = None
    print("    No Tim Hortons data — predicting all players")

# Step 7: Calculate xG for each player
print(f"\n  Calculating xG predictions (mode: {xg_mode})...")

for p in all_players:
    profile = estimate_shot_profile(p, p.get('position', 'C'))
    xg_result = calculate_player_goal_probability(
        p, profile, xg_model, xg_metadata, xg_fallback, team_stats, xg_mode
    )
    p.update(xg_result)

    # Add opponent context
    opp = p.get('opponent', '')
    p['opp_ga_per_game'] = team_stats.get(opp, {}).get('ga_per_game', Config.LEAGUE_AVG_GOALS)
    p['team_gf_per_game'] = team_stats.get(p['team'], {}).get('gf_per_game', Config.LEAGUE_AVG_GOALS)

# Sort by probability
all_players.sort(key=lambda x: x['goal_probability'], reverse=True)

# Add rank and hot indicator
for i, p in enumerate(all_players):
    p['rank'] = i + 1
    p['is_hot'] = p.get('last5_goals', 0) >= 3

# Step 8: Tim Hortons group rankings
tims_group_rankings = {}
if tims_mode and player_groups:
    for p in all_players:
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

# Step 9: Save output
print("\n  Saving predictions...")

output_players = all_players
if tims_mode and tims_player_ids:
    tims_preds = [p for p in all_players if p['player_id'] in tims_player_ids]
    for i, p in enumerate(tims_preds):
        p['tims_rank'] = i + 1
    output_players = tims_preds

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
        "xg_model_mode": xg_mode,
        "training_shots": xg_metadata.get('num_shots', 0) if xg_metadata else 0,
        "model_cv_auc": xg_metadata.get('cv_auc', None) if xg_metadata else None,
        "league_avg_goals": Config.LEAGUE_AVG_GOALS,
        "home_advantage": Config.HOME_ADVANTAGE,
        "num_representative_shots": Config.NUM_REPRESENTATIVE_SHOTS,
    },
    "generated_at": datetime.now().isoformat()
}

# Save dated file
dated_file = f"{Config.PREDICTIONS_DIR}/{Config.TODAY}.json"
with open(dated_file, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"  Saved: {dated_file}")

# Save latest
latest_file = f"{Config.PREDICTIONS_DIR}/latest.json"
with open(latest_file, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"  Saved: {latest_file}")

# Step 10: Console output
print("\n" + "=" * 70)
if tims_mode and tims_group_rankings:
    print("  TIM HORTONS CHALLENGE — xG TOP PICKS BY GROUP")
    print("=" * 70)
    for gid in sorted(tims_group_rankings.keys()):
        print(f"\n    Group {gid}:")
        print(f"    {'#':<4} {'Name':<25} {'Team':<5} {'Prob':>7} {'xG':>6} {'Shots':>6}")
        print(f"    {'-' * 55}")
        for p in tims_group_rankings[gid][:5]:
            # Find full player data
            full = next((x for x in all_players if x['player_id'] == p['player_id']), {})
            xg = full.get('player_xg', 0)
            shots = full.get('expected_shots', 0)
            print(f"    {p['rank_in_group']:<4} {p['name']:<25} {p['team']:<5} "
                  f"{p['goal_probability']*100:>6.1f}% {xg:>5.2f} {shots:>5.1f}")
else:
    print("  TOP 20 PREDICTIONS (xG XGBoost)")
    print("=" * 70)
    print(f"  {'#':<4} {'Name':<25} {'Team':<5} {'Prob':>7} {'xG':>6} {'Shots':>6}")
    print(f"  {'-' * 55}")
    for p in output_players[:20]:
        hot = " *" if p['is_hot'] else ""
        print(f"  {p['rank']:<4} {p['name']:<25} {p['team']:<5} "
              f"{p['goal_probability']*100:>6.1f}% {p.get('player_xg',0):>5.2f} "
              f"{p.get('expected_shots',0):>5.1f}{hot}")

print(f"\n  Total: {len(output_players)} players ranked")
print(f"  Mode: {xg_mode}")
print(f"\n  xG v3 predictions complete!")
