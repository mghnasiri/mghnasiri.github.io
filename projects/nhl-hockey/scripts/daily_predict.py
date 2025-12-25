"""
NHL Goal Predictor - Daily Prediction Script
=============================================
Generates predictions and saves as JSON for web dashboard

Author: Mohammad
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    # Model name for this scenario
    MODEL_NAME = "neural_network"
    MODEL_DISPLAY_NAME = "Neural Network v1"
    
    # Paths
    DATA_DIR = "data"
    PREDICTIONS_DIR = f"{DATA_DIR}/predictions/{MODEL_NAME}"
    RESULTS_DIR = f"{DATA_DIR}/results"
    HISTORICAL_FILE = f"{DATA_DIR}/historical/all_raw_stats.csv"
    
    # Date
    TODAY = datetime.now().strftime("%Y-%m-%d")
    # TODAY = "2024-04-15"  # For testing

# Create directories
os.makedirs(Config.PREDICTIONS_DIR, exist_ok=True)
os.makedirs(Config.RESULTS_DIR, exist_ok=True)

print("=" * 70)
print(f"🏒 NHL GOAL PREDICTOR - {Config.MODEL_DISPLAY_NAME}")
print(f"📅 Date: {Config.TODAY}")
print("=" * 70)

# =============================================================================
# 1. FETCH TODAY'S GAMES
# =============================================================================
def get_todays_games(date):
    url = f"https://api-web.nhle.com/v1/schedule/{date}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return []
        
        data = resp.json()
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
    except Exception as e:
        print(f"❌ Error fetching games: {e}")
        return []

print("\n📡 Fetching today's games...")
todays_games = get_todays_games(Config.TODAY)

if not todays_games:
    print("⚠️ No games found for today!")
    # Save empty prediction
    output = {
        "date": Config.TODAY,
        "model": Config.MODEL_NAME,
        "model_display_name": Config.MODEL_DISPLAY_NAME,
        "games_count": 0,
        "games": [],
        "predictions": [],
        "generated_at": datetime.now().isoformat()
    }
    with open(f"{Config.PREDICTIONS_DIR}/{Config.TODAY}.json", 'w') as f:
        json.dump(output, f, indent=2)
    with open(f"{Config.PREDICTIONS_DIR}/latest.json", 'w') as f:
        json.dump(output, f, indent=2)
    exit()

print(f"✅ Found {len(todays_games)} games")

# =============================================================================
# 2. GET PLAYER ROSTERS
# =============================================================================
def get_team_roster(team_abbrev):
    url = f"https://api-web.nhle.com/v1/roster/{team_abbrev}/current"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return []
        
        data = resp.json()
        players = []
        
        for group in ['forwards', 'defensemen']:
            for p in data.get(group, []):
                players.append({
                    'player_id': p['id'],
                    'name': f"{p['firstName']['default']} {p['lastName']['default']}",
                    'position': p['positionCode'],
                    'team': team_abbrev,
                    'headshot': p.get('headshot', '')
                })
        return players
    except:
        return []

print("\n👥 Fetching rosters...")
all_teams = set()
for g in todays_games:
    all_teams.add(g['home_team'])
    all_teams.add(g['away_team'])

all_players = []
for team in all_teams:
    roster = get_team_roster(team)
    all_players.extend(roster)

print(f"✅ Total players: {len(all_players)}")

# =============================================================================
# 3. LOAD HISTORICAL DATA & CALCULATE FEATURES
# =============================================================================
print("\n📊 Loading historical data...")

hist_df = None
if os.path.exists(Config.HISTORICAL_FILE):
    hist_df = pd.read_csv(Config.HISTORICAL_FILE)
    hist_df['game_date'] = pd.to_datetime(hist_df['game_date'])
    print(f"✅ Loaded: {Config.HISTORICAL_FILE}")

def calculate_player_features(player_id, hist_df, current_date):
    if hist_df is None:
        return None
    
    player_data = hist_df[
        (hist_df['player_id'] == player_id) & 
        (hist_df['game_date'] < current_date)
    ].sort_values('game_date')
    
    if len(player_data) < 3:
        return None
    
    last_5 = player_data.tail(5)
    
    def parse_toi(toi):
        try:
            if ':' in str(toi):
                m, s = str(toi).split(':')
                return int(m) + int(s)/60
            return float(toi) if pd.notna(toi) else 15.0
        except:
            return 15.0
    
    return {
        'games_played': len(player_data),
        'season_goals': int(player_data['goals'].sum()),
        'season_shots': int(player_data['shots'].sum()),
        'avg_goals': round(player_data['goals'].mean(), 3),
        'avg_shots': round(player_data['shots'].mean(), 2),
        'avg_toi': round(player_data['toi'].apply(parse_toi).mean(), 1),
        'shooting_pct': round(player_data['goals'].sum() / max(player_data['shots'].sum(), 1), 3),
        'last5_goals': int(last_5['goals'].sum()),
        'last5_points': int(last_5['goals'].sum() + last_5['assists'].sum()),
    }

# Calculate features
current_date = pd.to_datetime(Config.TODAY)
player_features = []

for player in all_players:
    features = calculate_player_features(player['player_id'], hist_df, current_date)
    if features:
        player_features.append({**player, **features})

print(f"✅ Players with history: {len(player_features)}")

# =============================================================================
# 4. ADD MATCHUP INFO
# =============================================================================
game_matchups = {}
for g in todays_games:
    game_matchups[g['home_team']] = {'opponent': g['away_team'], 'is_home': True, 'game_id': g['game_id']}
    game_matchups[g['away_team']] = {'opponent': g['home_team'], 'is_home': False, 'game_id': g['game_id']}

for player in player_features:
    team = player['team']
    if team in game_matchups:
        player['opponent'] = game_matchups[team]['opponent']
        player['is_home'] = game_matchups[team]['is_home']
        player['game_id'] = game_matchups[team]['game_id']
        player['matchup'] = f"{team} vs {player['opponent']}" if player['is_home'] else f"{team} @ {player['opponent']}"

# =============================================================================
# 5. CALCULATE OPPONENT STRENGTH
# =============================================================================
if hist_df is not None:
    recent = hist_df[hist_df['game_date'] >= hist_df['game_date'].max() - timedelta(days=60)]
    team_defense = {}
    for team in recent['team'].unique():
        opp_goals = recent[recent['opponent'] == team].groupby('game_id')['goals'].sum()
        team_defense[team] = round(opp_goals.mean(), 2) if len(opp_goals) > 0 else 3.0
    
    for player in player_features:
        player['opponent_ga'] = team_defense.get(player.get('opponent', ''), 3.0)

# =============================================================================
# 6. CALCULATE GOAL PROBABILITY
# =============================================================================
print("\n🧠 Calculating probabilities...")

for player in player_features:
    prob = (
        player.get('avg_goals', 0) * 0.35 +
        player.get('avg_shots', 0) * 0.08 +
        player.get('last5_goals', 0) * 0.04 +
        player.get('opponent_ga', 3.0) * 0.015 +
        player.get('shooting_pct', 0) * 0.1
    )
    player['goal_probability'] = round(prob, 4)

# Normalize
if player_features:
    max_prob = max(p['goal_probability'] for p in player_features)
    if max_prob > 0:
        for player in player_features:
            player['goal_probability'] = round(
                min(max(player['goal_probability'] / max_prob * 0.55, 0.02), 0.60), 
                3
            )

# Sort
player_features.sort(key=lambda x: x['goal_probability'], reverse=True)

# Add rank
for i, player in enumerate(player_features):
    player['rank'] = i + 1
    player['is_hot'] = player.get('last5_goals', 0) >= 3

# =============================================================================
# 7. SAVE OUTPUT
# =============================================================================
print("\n💾 Saving predictions...")

output = {
    "date": Config.TODAY,
    "model": Config.MODEL_NAME,
    "model_display_name": Config.MODEL_DISPLAY_NAME,
    "games_count": len(todays_games),
    "games": todays_games,
    "players_count": len(player_features),
    "predictions": player_features,
    "generated_at": datetime.now().isoformat()
}

# Save dated file
dated_file = f"{Config.PREDICTIONS_DIR}/{Config.TODAY}.json"
with open(dated_file, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"✅ Saved: {dated_file}")

# Save latest
latest_file = f"{Config.PREDICTIONS_DIR}/latest.json"
with open(latest_file, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"✅ Saved: {latest_file}")

# Console output - Top 20
print("\n" + "=" * 70)
print("🎯 TOP 20 PREDICTIONS")
print("=" * 70)
print(f"{'#':<4} {'Name':<25} {'Team':<5} {'Prob':>7} {'Goals':>6} {'L5':>4}")
print("-" * 55)

for p in player_features[:20]:
    hot = "🔥" if p['is_hot'] else "  "
    print(f"{p['rank']:<4} {p['name']:<25} {p['team']:<5} {p['goal_probability']*100:>6.1f}% {p['season_goals']:>6} {p['last5_goals']:>4} {hot}")

print("\n🏒 Predictions complete!")
