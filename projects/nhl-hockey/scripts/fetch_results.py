"""
NHL Goal Predictor - Fetch Actual Results
==========================================
Gets actual goal scorers from completed games
Compares Top 10 predictions with actual results

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
    PREDICTIONS_DIR = f"{DATA_DIR}/predictions"
    
    # Yesterday's date (we fetch results for completed games)
    YESTERDAY = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

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

# Handle no games gracefully
if not games:
    print("⚠️ No games found for yesterday!")
    output = {
        "date": Config.YESTERDAY,
        "games_count": 0,
        "games": [],
        "all_scorers": [],
        "scorers_count": 0,
        "model_comparisons": [],
        "message": "No games scheduled",
        "fetched_at": datetime.now().isoformat()
    }
    
    # Save anyway so dashboard knows there were no games
    dated_file = f"{Config.RESULTS_DIR}/{Config.YESTERDAY}.json"
    with open(dated_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved: {dated_file}")
    
    latest_file = f"{Config.RESULTS_DIR}/latest.json"
    with open(latest_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved: {latest_file}")
    
    print("\n🏒 No games to process. Done!")
    exit()

print(f"✅ Found {len(games)} games")

# =============================================================================
# 2. GET GOAL SCORERS FOR EACH GAME
# =============================================================================
def get_boxscore_goals(game_id):
    """Get goals from boxscore"""
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
# 3. LOAD PREDICTIONS AND COMPARE TOP 10
# =============================================================================
print("\n📊 Comparing Top 10 predictions...")

scorer_ids = set(s['player_id'] for s in all_scorers)
comparison_results = []

# Check all model directories
if os.path.exists(Config.PREDICTIONS_DIR):
    for model_name in os.listdir(Config.PREDICTIONS_DIR):
        model_dir = f"{Config.PREDICTIONS_DIR}/{model_name}"
        if not os.path.isdir(model_dir):
            continue
            
        pred_file = f"{model_dir}/{Config.YESTERDAY}.json"
        
        if os.path.exists(pred_file):
            with open(pred_file, 'r') as f:
                predictions = json.load(f)
            
            # Get Top 10 predictions
            top10_predictions = predictions.get('predictions', [])[:10]
            
            if not top10_predictions:
                print(f"   ⚠️ {model_name}: No predictions found")
                continue
            
            # Check hits for Top 10
            top10_picks = []
            hits = 0
            
            for pred in top10_predictions:
                scored = pred['player_id'] in scorer_ids
                if scored:
                    hits += 1
                
                top10_picks.append({
                    'rank': pred['rank'],
                    'player_id': pred['player_id'],
                    'name': pred['name'],
                    'team': pred['team'],
                    'probability': pred['goal_probability'],
                    'scored': scored
                })
            
            comparison_results.append({
                'model': model_name,
                'model_display_name': predictions.get('model_display_name', model_name),
                'top10_picks': top10_picks,
                'hits': hits,
                'total_predictions': 10,
                'hit_rate': round(hits / 10 * 100, 1)
            })
            
            # Print results
            print(f"\n   📈 {model_name} - Top 10:")
            for p in top10_picks:
                status = "✅" if p['scored'] else "❌"
                print(f"      {p['rank']:>2}. {p['name']:<25} ({p['team']}) {p['probability']*100:>5.1f}% {status}")
            print(f"      ─────────────────────────────────────────")
            print(f"      Result: {hits}/10 correct ({round(hits/10*100)}%)")

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
        goals_text = f" ({s['goals']} goals)" if s.get('goals', 1) > 1 else ""
        print(f"   • {s['player_name']} ({s['team']}){goals_text}")

if comparison_results:
    print("\n📈 Model Performance (Top 10):")
    for comp in comparison_results:
        print(f"   {comp['model_display_name']}: {comp['hits']}/10 ({comp['hit_rate']}%)")

print("\n🏒 Results fetch complete!")
