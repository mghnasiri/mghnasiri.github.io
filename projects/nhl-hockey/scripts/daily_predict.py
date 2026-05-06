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
import time
import warnings
warnings.filterwarnings('ignore')


def current_season_id(date=None):
    """NHL seasonId like '20252026' for a date. Season starts in early October;
    month >= 8 rolls the ID forward so July/Aug/early-Sep still returns the
    upcoming season and keeps API calls pointed at data that actually exists."""
    d = date or datetime.now()
    if isinstance(d, str):
        d = datetime.strptime(d, "%Y-%m-%d")
    start = d.year if d.month >= 8 else d.year - 1
    return f"{start}{start + 1}"

# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    MODEL_NAME = "neural_network"
    MODEL_DISPLAY_NAME = "Weighted Linear v1"

    DATA_DIR = "data"
    PREDICTIONS_DIR = f"{DATA_DIR}/predictions/{MODEL_NAME}"
    RESULTS_DIR = f"{DATA_DIR}/results"
    TIMS_DIR = f"{DATA_DIR}/tims_players"

    TODAY = datetime.now().strftime("%Y-%m-%d")
    CURRENT_SEASON = current_season_id()

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
def compute_stats_from_gamelog(game_log, today_date):
    """Compute season stats from game log, excluding games on or after today_date."""
    prior_games = [g for g in game_log if g.get('gameDate', '9999') < today_date]
    if not prior_games:
        return None

    games_played = len(prior_games)
    goals = sum(g.get('goals', 0) for g in prior_games)
    shots = sum(g.get('shots', 0) for g in prior_games)

    last5 = prior_games[:5]  # game log is newest-first
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

def _fetch_game_log(player_id, season, game_type):
    """One game-log fetch with retry. Returns list (possibly empty) or None on hard error."""
    url = f"https://api-web.nhle.com/v1/player/{player_id}/game-log/{season}/{game_type}"
    try:
        for attempt in range(3):
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                return resp.json().get('gameLog', [])
            if resp.status_code == 404:
                return []
            if attempt < 2:
                time.sleep(1)
        return None
    except Exception:
        return None


def get_player_current_stats(player_id):
    """Get current season stats for a player from NHL API game log (leak-free).

    Pulls regular-season (type 2) + playoff (type 3) logs for the current
    season explicitly and merges them. The `/game-log/now` endpoint silently
    switches to playoff-only mode once a team enters the postseason, so
    healthy scorers end up with 0-3 games and fall below the min-games filter."""
    season = Config.CURRENT_SEASON
    reg = _fetch_game_log(player_id, season, 2)
    po = _fetch_game_log(player_id, season, 3)
    if reg is None and po is None:
        return None
    game_log = (reg or []) + (po or [])
    if not game_log:
        return None
    game_log.sort(key=lambda g: g.get('gameDate', ''), reverse=True)
    stats = compute_stats_from_gamelog(game_log, Config.TODAY)
    if stats is None or stats['games_played'] < 1:
        return None
    return stats

def get_team_roster_with_stats(team_abbrev):
    """Get team roster with current season stats"""
    url = f"https://api-web.nhle.com/v1/roster/{team_abbrev}/current"
    data = None
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                break
        except requests.RequestException:
            pass
        if attempt < 2:
            time.sleep(2)

    if not data:
        print(f"   ⚠️ Failed to fetch roster for {team_abbrev} after 3 attempts")
        return []

    players = []
    roster_list = []
    for group in ['forwards', 'defensemen']:
        for p in data.get(group, []):
            roster_list.append(p)

    for i, p in enumerate(roster_list):
        if i > 0 and i % 10 == 0:
            time.sleep(0.5)  # Rate limit: pause every 10 players
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

# =============================================================================
# 3. COLLECT ALL PLAYERS
# =============================================================================
all_teams = set()
for g in todays_games:
    all_teams.add(g['home_team'])
    all_teams.add(g['away_team'])

# Per-day cross-script cache (shared with MC v2 + xG v3 + neural_v2). Linear
# v1 runs at 15:00 UTC, AFTER MC (14:00) and xG (14:05). On a normal day
# the cache is already populated when Linear starts — zero NHL API calls
# needed. Falls back to fetching if no script ran successfully earlier.
PLAYERS_CACHE_DIR = f"{Config.DATA_DIR}/cache"
PLAYERS_CACHE_FILE = f"{PLAYERS_CACHE_DIR}/players_{Config.TODAY}.json"
os.makedirs(PLAYERS_CACHE_DIR, exist_ok=True)

all_players = None
if os.path.exists(PLAYERS_CACHE_FILE):
    try:
        with open(PLAYERS_CACHE_FILE, 'r', encoding='utf-8') as f:
            cached = json.load(f)
        if cached.get('date') == Config.TODAY and cached.get('players'):
            all_players = cached['players']
            print(f"\n✅ Loaded {len(all_players)} players from cache: "
                  f"{PLAYERS_CACHE_FILE}")
    except Exception as e:
        print(f"\n⚠️ Cache read failed ({e}); will refetch")
        all_players = None

if all_players is None:
    print("\n👥 Fetching rosters and stats (no cache available)...")
    all_players = []
    for idx, team in enumerate(sorted(all_teams)):
        if idx > 0:
            time.sleep(1)  # Rate limit: pause between teams
        print(f"   Fetching {team}...", end=" ", flush=True)
        roster = get_team_roster_with_stats(team)
        all_players.extend(roster)
        print(f"{len(roster)} players")

    if all_players:
        try:
            with open(PLAYERS_CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump({'date': Config.TODAY, 'players': all_players,
                           'cached_at': datetime.now().isoformat()},
                          f, ensure_ascii=False)
            print(f"   Cached {len(all_players)} players to {PLAYERS_CACHE_FILE}")
        except Exception as e:
            print(f"   Cache write failed: {e}")

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
# 5.5 TIM HORTONS FILTERING
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

def normalize_name(name):
    return name.lower().strip().replace('.', '').replace("'", "").replace('-', ' ')

tims_data = load_tims_players(Config.TODAY)
tims_mode = tims_data is not None and bool(tims_data.get('groups'))
tims_group_rankings = {}
output_players = all_players

if tims_mode:
    print("\n🍩 Tim Hortons mode: filtering predictions...")
    tims_groups = {}
    all_tims_names = set()
    for group_id, players in tims_data['groups'].items():
        group_names = set()
        for p in players:
            name = p if isinstance(p, str) else p.get('name', '')
            group_names.add(normalize_name(name))
            all_tims_names.add(normalize_name(name))
        tims_groups[group_id] = group_names

    tims_filtered = []
    for player in all_players:
        pname = normalize_name(player['name'])
        if pname in all_tims_names:
            for gid, names in tims_groups.items():
                if pname in names:
                    player['tims_group'] = gid
                    break
            tims_filtered.append(player)

    # Build group rankings (sorted by probability within each group)
    for p in tims_filtered:
        gid = p.get('tims_group')
        if gid:
            if gid not in tims_group_rankings:
                tims_group_rankings[gid] = []
            tims_group_rankings[gid].append({
                'rank_in_group': len(tims_group_rankings[gid]) + 1,
                'player_id': p.get('player_id', 0),
                'name': p['name'],
                'team': p['team'],
                'goal_probability': p['goal_probability'],
                'matchup': p.get('matchup', ''),
            })

    # Re-rank filtered players
    for i, p in enumerate(tims_filtered):
        p['tims_rank'] = i + 1
    output_players = tims_filtered
    print(f"  ✅ Matched {len(tims_filtered)} Tim Hortons players")
else:
    print("\n📭 No Tim Hortons data — using full player list")

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
    "players_count": len(output_players),
    "predictions": output_players,
    "tims_mode": tims_mode,
    "tims_source": tims_data.get('source', 'unknown') if tims_data else None,
    "tims_group_rankings": tims_group_rankings if tims_mode else {},
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
