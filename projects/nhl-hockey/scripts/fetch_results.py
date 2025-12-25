"""
NHL Goal Predictor - Fetch Actual Results
==========================================
Gets actual goal scorers from completed games
Compares with predictions

Author: Mohammad
"""

import requests
import json
import os
from datetime import datetime, timedelta

# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    DATA_DIR = "data"
    RESULTS_DIR = f"{DATA_DIR}/results"
    
    # Yesterday's date (we fetch results for completed games)
    YESTERDAY = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    # YESTERDAY = "2024-04-15"  # For testing

os.makedirs(Config.RESULTS_DIR, exist_ok=True)

print("=" * 70)
print("🏒 NHL GOAL PREDICTOR - FETCH ACTUAL RESULTS")
print(f"📅 Date: {Config.YESTERDAY}")
print("=" * 70)

# =============================================================================
# 1. GET GAMES FROM DATE
# =============================================================================
def get_games_for_date(date):
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
                            'game_state': game.get('gameState', ''),
                        })
        return games
    except Exception as e:
        print(f"❌ Error: {e}")
        return []

print("\n📡 Fetching games...")
games = get_games_for_date(Config.YESTERDAY)

if not games:
    print("⚠️ No games found!")
    exit()

print(f"✅ Found {len(games)} games")

# =============================================================================
# 2. GET GOAL SCORERS FOR EACH GAME
# =============================================================================
def get_game_goals(game_id):
    """Get all goal scorers from a game (excluding shootout)"""
    url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
    
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return []
        
        data = resp.json()
        goals = []
        
        for play in data.get('plays', []):
            # Goal in regular time or overtime (not shootout)
            if play.get('typeDescKey') == 'goal':
                period = play.get('periodDescriptor', {}).get('number', 0)
                
                # Skip shootout goals (period > 4 in regular season, varies in playoffs)
                if period <= 4:  # Reg time + OT
                    details = play.get('details', {})
                    scorer_id = details.get('scoringPlayerId')
                    
                    if scorer_id:
                        # Get scorer name
                        for player_type in ['scoringPlayer']:
                            if player_type + 'Id' in details:
                                goals.append({
                                    'player_id': details.get('scoringPlayerId'),
                                    'player_name': details.get('scoringPlayerTotal', {}).get('name', 'Unknown'),
                                    'period': period,
                                    'time': play.get('timeInPeriod', ''),
                                    'team': details.get('eventOwnerTeamId'),
                                })
        
        return goals
    
    except Exception as e:
        print(f"   ⚠️ Error fetching game {game_id}: {e}")
        return []

def get_boxscore_goals(game_id):
    """Alternative: Get goals from boxscore"""
    url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/boxscore"
    
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return []
        
        data = resp.json()
        scorers = []
        
        for team_type in ['homeTeam', 'awayTeam']:
            team_data = data.get(team_type, {})
            team_abbrev = team_data.get('abbrev', '')
            
            for player in team_data.get('forwards', []) + team_data.get('defense', []):
                goals = player.get('goals', 0)
                if goals > 0:
                    scorers.append({
                        'player_id': player.get('playerId'),
                        'player_name': player.get('name', {}).get('default', 'Unknown'),
                        'team': team_abbrev,
                        'goals': goals
                    })
        
        return scorers
    
    except Exception as e:
        print(f"   ⚠️ Error: {e}")
        return []

print("\n⚽ Fetching goal scorers...")
all_scorers = []

for game in games:
    game_id = game['game_id']
    print(f"   {game['away_team']} @ {game['home_team']}...", end=" ")
    
    scorers = get_boxscore_goals(game_id)
    
    for scorer in scorers:
        scorer['game_id'] = game_id
        scorer['matchup'] = f"{game['away_team']} @ {game['home_team']}"
    
    all_scorers.extend(scorers)
    print(f"{len(scorers)} scorers")

print(f"\n✅ Total scorers: {len(all_scorers)}")

# =============================================================================
# 3. LOAD PREDICTIONS AND COMPARE
# =============================================================================
print("\n📊 Comparing with predictions...")

# Find prediction files for this date
comparison_results = []
models_found = []

# Check all model directories
predictions_base = f"{Config.DATA_DIR}/predictions"
if os.path.exists(predictions_base):
    for model_name in os.listdir(predictions_base):
        pred_file = f"{predictions_base}/{model_name}/{Config.YESTERDAY}.json"
        
        if os.path.exists(pred_file):
            models_found.append(model_name)
            
            with open(pred_file, 'r') as f:
                predictions = json.load(f)
            
            # Get top 3 predictions
            top_predictions = predictions.get('predictions', [])[:3]
            
            # Check hits
            scorer_ids = set(s['player_id'] for s in all_scorers)
            hits = []
            
            for pred in top_predictions:
                scored = pred['player_id'] in scorer_ids
                hits.append({
                    'rank': pred['rank'],
                    'name': pred['name'],
                    'team': pred['team'],
                    'probability': pred['goal_probability'],
                    'scored': scored
                })
            
            hit_count = sum(1 for h in hits if h['scored'])
            
            comparison_results.append({
                'model': model_name,
                'model_display_name': predictions.get('model_display_name', model_name),
                'top3_picks': hits,
                'hits': hit_count,
                'total_predictions': 3
            })
            
            print(f"\n   📈 {model_name}:")
            for h in hits:
                status = "✅ GOAL!" if h['scored'] else "❌"
                print(f"      {h['rank']}. {h['name']} ({h['team']}) - {h['probability']*100:.1f}% {status}")
            print(f"      Result: {hit_count}/3 correct")

# =============================================================================
# 4. SAVE RESULTS
# =============================================================================
print("\n💾 Saving results...")

output = {
    "date": Config.YESTERDAY,
    "games_count": len(games),
    "games": games,
    "all_scorers": all_scorers,
    "scorers_count": len(all_scorers),
    "model_comparisons": comparison_results,
    "fetched_at": datetime.now().isoformat()
}

# Save dated file
dated_file = f"{Config.RESULTS_DIR}/{Config.YESTERDAY}.json"
with open(dated_file, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"✅ Saved: {dated_file}")

# Save latest
latest_file = f"{Config.RESULTS_DIR}/latest.json"
with open(latest_file, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"✅ Saved: {latest_file}")

# =============================================================================
# 5. SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("📊 SUMMARY")
print("=" * 70)

print(f"\n🏒 Games: {len(games)}")
print(f"⚽ Total goal scorers: {len(all_scorers)}")

if all_scorers:
    print("\n🎯 All Goal Scorers:")
    for s in all_scorers:
        goals_text = f"({s['goals']} goals)" if s.get('goals', 1) > 1 else ""
        print(f"   • {s['player_name']} ({s['team']}) {goals_text}")

if comparison_results:
    print("\n📈 Model Performance:")
    for comp in comparison_results:
        print(f"   {comp['model_display_name']}: {comp['hits']}/3")

print("\n🏒 Results fetch complete!")
