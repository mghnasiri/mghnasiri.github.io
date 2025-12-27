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
    MODEL_NAME = "neural_network"
    MODEL_DISPLAY_NAME = "Neural Network v1"
    
    DATA_DIR = "data"
    PREDICTIONS_DIR = f"{DATA_DIR}/predictions/{MODEL_NAME}"
    RESULTS_DIR = f"{DATA_DIR}/results"
    HISTORICAL_FILE = f"{DATA_DIR}/historical/all_raw_stats.csv"
    
    TODAY = datetime.now().strftime("%Y-%m-%d")
    CURRENT_SEASON = "20242025"

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
    output = {
        "date": Config.TODAY,
        "model": Config.MODEL_NAME,
        "model_display_name": Config.MODEL_DISPLAY_NAME,
        "games_count": 0,
        "games": [],
        "players_count": 0,
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
# 2. GET PLAYER STATS FROM API (Current Season)
# =============================================================================
def get_player_current_stats(player_id):
    """Get current season stats for a player from NHL API"""
    url = f"https://api-web.nhle.com/v1/player/{player_id}/landing"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return None
        
        data = resp.json()
        
        # Get current season stats
        featured_stats = data.get('featuredStats', {})
        regular_season = featured_stats.get('regularSeason', {})
        sub_season = regular_season.get('subSeason', {})
        
        if not sub_season:
            return None
        
        games_played = sub_season.get('gamesPlayed', 0)
        if games_played < 1:
            return None
        
        goals = sub_season.get('goals', 0)
        shots = sub_season.get('shots', 0)
        points = sub_season.get('points', 0)
        
        # Get last 5 games
        last5 = data.get('last5Games', [])
        last5_goals = sum(g.get('goals', 0) for g in last5)
        last5_points = sum(g.get('points', 0) for g in last5)
        
        return {
            'games_played': games_played,
            'season_goals': goals,
            'season_shots': shots,
            'avg_goals': round(goals / games_played, 3) if games_played > 0 else 0,
            'avg_shots': round(shots / games_played, 2) if games_played > 0 else 0,
            'shooting_pct': round(goals / shots, 3) if shots > 0 else 0,
            'last5_goals': last5_goals,
            'last5_points': last5_points,
        }
    except Exception as e:
        return None

def get_team_roster_with_stats(team_abbrev):
    """Get team roster with current season stats"""
    url = f"https://api-web.nhle.com/v1/roster/{team_abbrev}/current"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return []
        
        data = resp.json()
        players = []
        
        for group in ['forwards', 'defensemen']:
            for p in data.get(group, []):
                player_id = p['id']
                name = f"{p['firstName']['default']} {p['lastName']['default']}"
                position = p['positionCode']
                
                # Get current stats
                stats = get_player_current_stats(player_id)
                
                if stats and stats['games_played'] >= 3:
                    players.append({
                        'player_id': player_id,
                        'name': name,
                        'position': position,
                        'team': team_abbrev,
                        **stats
                    })
        
        return players
    except Exception as e:
        print(f"   ⚠️ Error fetching {team_abbrev}: {e}")
        return []

# =============================================================================
# 3. COLLECT ALL PLAYERS
# =============================================================================
print("\n👥 Fetching rosters and stats...")

all_teams = set()
for g in todays_games:
    all_teams.add(g['home_team'])
    all_teams.add(g['away_team'])

all_players = []
for team in sorted(all_teams):
    print(f"   Fetching {team}...", end=" ")
    roster = get_team_roster_with_stats(team)
    all_players.extend(roster)
    print(f"{len(roster)} players")

print(f"\n✅ Total players with stats: {len(all_players)}")

if len(all_players) == 0:
    print("⚠️ No player data available!")
    output = {
        "date": Config.TODAY,
        "model": Config.MODEL_NAME,
        "model_display_name": Config.MODEL_DISPLAY_NAME,
        "games_count": len(todays_games),
        "games": todays_games,
        "players_count": 0,
        "predictions": [],
        "generated_at": datetime.now().isoformat()
    }
    with open(f"{Config.PREDICTIONS_DIR}/{Config.TODAY}.json", 'w') as f:
        json.dump(output, f, indent=2)
    with open(f"{Config.PREDICTIONS_DIR}/latest.json", 'w') as f:
        json.dump(output, f, indent=2)
    exit()

# =============================================================================
# 4. ADD MATCHUP INFO
# =============================================================================
print("\n🏒 Adding matchup info...")

game_matchups = {}
for g in todays_games:
    game_matchups[g['home_team']] = {'opponent': g['away_team'], 'is_home': True, 'game_id': g['game_id']}
    game_matchups[g['away_team']] = {'opponent': g['home_team'], 'is_home': False, 'game_id': g['game_id']}

for player in all_players:
    team = player['team']
    if team in game_matchups:
        player['opponent'] = game_matchups[team]['opponent']
        player['is_home'] = game_matchups[team]['is_home']
        player['game_id'] = game_matchups[team]['game_id']
        player['matchup'] = f"{team} vs {player['opponent']}" if player['is_home'] else f"{team} @ {player['opponent']}"

# =============================================================================
# 5. CALCULATE GOAL PROBABILITY
# =============================================================================
print("\n🧠 Calculating probabilities...")

for player in all_players:
    # Simple but effective formula based on our analysis
    prob = (
        player.get('avg_goals', 0) * 0.40 +
        player.get('avg_shots', 0) * 0.08 +
        player.get('last5_goals', 0) * 0.05 +
        player.get('shooting_pct', 0) * 0.15
    )
    player['goal_probability_raw'] = prob

# Normalize
max_prob = max(p['goal_probability_raw'] for p in all_players) if all_players else 1
for player in all_players:
    player['goal_probability'] = round(
        min(max(player['goal_probability_raw'] / max_prob * 0.55, 0.02), 0.60),
        3
    )
    del player['goal_probability_raw']

# Sort by probability
all_players.sort(key=lambda x: x['goal_probability'], reverse=True)

# Add rank and hot indicator
for i, player in enumerate(all_players):
    player['rank'] = i + 1
    player['is_hot'] = player.get('last5_goals', 0) >= 3

# =============================================================================
# 6. SAVE OUTPUT
# =============================================================================
print("\n💾 Saving predictions...")

output = {
    "date": Config.TODAY,
    "model": Config.MODEL_NAME,
    "model_display_name": Config.MODEL_DISPLAY_NAME,
    "games_count": len(todays_games),
    "games": todays_games,
    "players_count": len(all_players),
    "predictions": all_players,
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

for p in all_players[:20]:
    hot = "🔥" if p['is_hot'] else "  "
    print(f"{p['rank']:<4} {p['name']:<25} {p['team']:<5} {p['goal_probability']*100:>6.1f}% {p['season_goals']:>6} {p['last5_goals']:>4} {hot}")

print(f"\n✅ Total: {len(all_players)} players ranked")
print("\n🏒 Predictions complete!")
