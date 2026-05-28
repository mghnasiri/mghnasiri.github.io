"""
NHL Goal Predictor - Fetch Results
==================================
Fetches actual goal scorers from completed games
Compares with Top 10 predictions

Author: Mohammad G. Nasiri
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
    
    YESTERDAY = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

# Create directories
os.makedirs(Config.RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("🏒 NHL GOAL PREDICTOR - FETCH RESULTS")
print(f"📅 Date: {Config.YESTERDAY}")
print("=" * 60)

# =============================================================================
# FETCH GAMES
# =============================================================================
def get_games(date):
    """Get all NHL games for a specific date"""
    url = f"https://api-web.nhle.com/v1/schedule/{date}"
    
    try:
        response = requests.get(url, timeout=15)
        if response.status_code != 200:
            return []
        
        data = response.json()
        games = []
        
        for day in data.get('gameWeek', []):
            if day['date'] == date:
                for game in day.get('games', []):
                    # Only regular season (2) and playoffs (3)
                    if game.get('gameType') in [2, 3]:
                        games.append({
                            'game_id': game['id'],
                            'home_team': game['homeTeam']['abbrev'],
                            'away_team': game['awayTeam']['abbrev'],
                            'game_state': game.get('gameState', '')
                        })
        
        return games
        
    except Exception as e:
        print(f"❌ Error fetching games: {e}")
        return []

# =============================================================================
# FETCH GOAL SCORERS
# =============================================================================
def get_scorers(game_id):
    """Get all goal scorers from a game's boxscore"""
    url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/boxscore"
    
    try:
        response = requests.get(url, timeout=15)
        if response.status_code != 200:
            print(f"   ⚠️ API returned {response.status_code}")
            return []
        
        data = response.json()
        scorers = []
        
        # NEW API STRUCTURE: playerByGameStats
        player_stats = data.get('playerByGameStats')
        
        if player_stats:
            for team_type in ['awayTeam', 'homeTeam']:
                team_data = player_stats.get(team_type, {})
                
                # Get team abbrev from main data
                team_abbrev = data.get(team_type, {}).get('abbrev', '')
                
                # Check forwards and defense
                for position_group in ['forwards', 'defense']:
                    for player in team_data.get(position_group, []):
                        goals = player.get('goals', 0)
                        if goals > 0:
                            # Handle name as dict or string
                            name = player.get('name', {})
                            if isinstance(name, dict):
                                player_name = name.get('default', 'Unknown')
                            else:
                                player_name = str(name) if name else 'Unknown'
                            
                            scorers.append({
                                'player_id': player.get('playerId'),
                                'player_name': player_name,
                                'team': team_abbrev,
                                'goals': goals
                            })
        else:
            # FALLBACK: Old structure
            for team_type in ['homeTeam', 'awayTeam']:
                team_data = data.get(team_type, {})
                team_abbrev = team_data.get('abbrev', '')
                
                players = team_data.get('forwards', []) + team_data.get('defense', [])
                
                for player in players:
                    goals = player.get('goals', 0)
                    if goals > 0:
                        name = player.get('name', {})
                        if isinstance(name, dict):
                            player_name = name.get('default', 'Unknown')
                        else:
                            player_name = str(name) if name else 'Unknown'
                        
                        scorers.append({
                            'player_id': player.get('playerId'),
                            'player_name': player_name,
                            'team': team_abbrev,
                            'goals': goals
                        })
        
        return scorers
        
    except Exception as e:
        print(f"   ⚠️ Error fetching boxscore: {e}")
        return []

# =============================================================================
# MAIN LOGIC
# =============================================================================

# 1. Get games
print("\n📡 Fetching games...")
games = get_games(Config.YESTERDAY)

# Handle no games
if not games:
    print("ℹ️  No games found for this date")
    
    output = {
        "date": Config.YESTERDAY,
        "games_count": 0,
        "games": [],
        "all_scorers": [],
        "scorers_count": 0,
        "model_comparisons": [],
        "fetched_at": datetime.now().isoformat()
    }
    
    # Save files
    with open(f"{Config.RESULTS_DIR}/{Config.YESTERDAY}.json", 'w') as f:
        json.dump(output, f, indent=2)
    with open(f"{Config.RESULTS_DIR}/latest.json", 'w') as f:
        json.dump(output, f, indent=2)
    
    print("✅ Saved empty results (no games)")
    exit()

print(f"✅ Found {len(games)} games")

# 2. Get all scorers — only from completed games. Grading a LIVE/scheduled
#    game records real scorers as "did not score" and understates hit rate.
print("\n⚽ Fetching goal scorers...")
FINAL_STATES = {"OFF", "FINAL"}
all_scorers = []
final_games = 0

for game in games:
    matchup = f"{game['away_team']} @ {game['home_team']}"
    state = game.get('game_state', '')
    if state not in FINAL_STATES:
        print(f"   {matchup}... SKIPPED (state={state or '?'}, not final)")
        continue
    final_games += 1
    print(f"   {matchup}...", end=" ")

    scorers = get_scorers(game['game_id'])
    for scorer in scorers:
        scorer['game_id'] = game['game_id']
        scorer['matchup'] = matchup
    all_scorers.extend(scorers)
    print(f"{len(scorers)} scorers")

print(f"\n✅ Total scorers: {len(all_scorers)} from {final_games}/{len(games)} final games")

# If nothing is final yet, do NOT grade — recording every pick as a miss
# would poison the hit-rate metric. Leave yesterday's result file untouched.
if final_games == 0:
    print("⛔ No completed games yet — skipping grading to avoid false zeros.")
    exit(0)

partial = final_games < len(games)
if partial:
    print(f"⚠️ {len(games) - final_games} game(s) not final — grading is partial.")

# 3. Compare with predictions
print("\n📊 Comparing with predictions...")

# Drop missing/None ids so a boxscore gap can't collapse to a single None
# that a player_id=0 prediction would then falsely match.
scorer_ids = {s['player_id'] for s in all_scorers if s.get('player_id')}
model_comparisons = []

# Check each model
if os.path.exists(Config.PREDICTIONS_DIR):
    for model_name in os.listdir(Config.PREDICTIONS_DIR):
        model_dir = f"{Config.PREDICTIONS_DIR}/{model_name}"
        
        if not os.path.isdir(model_dir):
            continue
        
        pred_file = f"{model_dir}/{Config.YESTERDAY}.json"
        
        if not os.path.exists(pred_file):
            continue
        
        # Load predictions
        with open(pred_file, 'r') as f:
            predictions = json.load(f)
        
        # Grade the model's top picks. Denominator is the number of picks
        # actually graded — a Tims-filtered model can have <10 on short slates,
        # and dividing by a hardcoded 10 understated hit rate on those days.
        TOP_N = 10
        top_picks = predictions.get('predictions', [])[:TOP_N]

        if not top_picks:
            print(f"   ⚠️ {model_name}: No predictions")
            continue

        hits = 0
        graded_picks = []

        for pred in top_picks:
            pid = pred.get('player_id')
            scored = bool(pid) and pid in scorer_ids
            if scored:
                hits += 1

            graded_picks.append({
                'rank': pred.get('rank', 0),
                'player_id': pid,
                'name': pred['name'],
                'team': pred['team'],
                'probability': pred.get('goal_probability', 0),
                'scored': scored
            })

        n = len(graded_picks)
        # Save comparison ('top10_picks' key kept for dashboard/telegram compat)
        model_comparisons.append({
            'model': model_name,
            'model_display_name': predictions.get('model_display_name', model_name),
            'top10_picks': graded_picks,
            'hits': hits,
            'total_predictions': n,
            'hit_rate': round(hits / n * 100, 1) if n else 0.0
        })

        # Print results
        print(f"\n   📈 {model_name}:")
        for p in graded_picks:
            icon = "✅" if p['scored'] else "❌"
            print(f"      {p['rank']:>2}. {p['name']:<24} {p['probability']*100:>5.1f}% {icon}")
        print(f"      {'─' * 45}")
        print(f"      Result: {hits}/{n} ({round(hits / n * 100) if n else 0}%)")

# 4. Save results
print("\n💾 Saving results...")

output = {
    "date": Config.YESTERDAY,
    "games_count": len(games),
    "final_games": final_games,
    "partial": partial,
    "games": games,
    "all_scorers": all_scorers,
    "scorers_count": len(all_scorers),
    "model_comparisons": model_comparisons,
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

# 5. Summary
print("\n" + "=" * 60)
print("📊 SUMMARY")
print("=" * 60)
print(f"🏒 Games: {len(games)}")
print(f"⚽ Scorers: {len(all_scorers)}")

if model_comparisons:
    print("\n📈 Model Results:")
    for comp in model_comparisons:
        print(f"   {comp['model']}: {comp['hits']}/10 ({comp['hit_rate']}%)")

print("\n✅ Done!")
