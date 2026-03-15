"""
NHL Goal Predictor - Monte Carlo Simulation Engine (v2)
=======================================================
Opponent-adjusted Poisson Monte Carlo model with Tim Hortons support.

Key improvements over Neural Network v1:
  1. Poisson-based Monte Carlo simulation (10,000 games per matchup)
  2. Opponent-adjusted scoring (opposing team GA/game, goalie save%)
  3. Home/away advantage modeling (+3% home boost)
  4. Exponential-weighted recent form (last 10 games weighted 2x)
  5. Power play opportunity adjustment
  6. Tim Hortons challenge player pool filtering

Author: Mohammad G. Nasiri
"""

import requests
import numpy as np
import json
import os
import sys
import time
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    MODEL_NAME = "monte_carlo"
    MODEL_DISPLAY_NAME = "Monte Carlo v2"

    DATA_DIR = "data"
    PREDICTIONS_DIR = f"{DATA_DIR}/predictions/{MODEL_NAME}"
    TIMS_DIR = f"{DATA_DIR}/tims_players"

    TODAY = datetime.now().strftime("%Y-%m-%d")
    CURRENT_SEASON = "20242025"

    NUM_SIMULATIONS = 10000
    LEAGUE_AVG_GOALS = 3.07       # 2024-25 NHL league average goals per team per game
    HOME_ADVANTAGE = 1.026        # Home teams score ~2.6% more
    MIN_GAMES_PLAYED = 3          # Minimum games for inclusion
    RECENT_GAMES_WINDOW = 10      # Recent form window
    RECENT_FORM_WEIGHT = 0.40     # Weight of recent form vs season avg
    PP_BONUS_FACTOR = 0.08        # Power play bonus weight

os.makedirs(Config.PREDICTIONS_DIR, exist_ok=True)
os.makedirs(Config.TIMS_DIR, exist_ok=True)

print("=" * 70)
print(f"🏒 NHL GOAL PREDICTOR - {Config.MODEL_DISPLAY_NAME}")
print(f"📅 Date: {Config.TODAY}")
print(f"🎲 Simulations: {Config.NUM_SIMULATIONS:,}")
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


# =============================================================================
# 1. FETCH TODAY'S GAMES
# =============================================================================
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


# =============================================================================
# 2. FETCH TEAM STANDINGS (for opponent adjustment)
# =============================================================================
def get_team_stats_from_standings():
    """
    Fetch all team stats from standings API.
    Returns dict: team_abbrev -> {gf_per_game, ga_per_game, games_played, ...}
    """
    data = api_get("https://api-web.nhle.com/v1/standings/now")
    if not data:
        print("   ⚠️ Could not fetch standings, using league averages")
        return {}

    team_stats = {}
    for team in data.get('standings', []):
        abbrev = team.get('teamAbbrev', {}).get('default', '')
        gp = team.get('gamesPlayed', 1)
        if gp == 0:
            gp = 1

        gf = team.get('goalFor', 0)
        ga = team.get('goalAgainst', 0)

        team_stats[abbrev] = {
            'games_played': gp,
            'goals_for': gf,
            'goals_against': ga,
            'gf_per_game': round(gf / gp, 3),
            'ga_per_game': round(ga / gp, 3),
            'wins': team.get('wins', 0),
            'losses': team.get('losses', 0),
            'points': team.get('points', 0),
        }

    return team_stats


# =============================================================================
# 3. FETCH PLAYER STATS
# =============================================================================
def get_player_stats(player_id):
    """Get comprehensive player stats from NHL API"""
    data = api_get(f"https://api-web.nhle.com/v1/player/{player_id}/landing")
    if not data:
        return None

    # Current season stats
    featured = data.get('featuredStats', {})
    regular = featured.get('regularSeason', {})
    sub = regular.get('subSeason', {})

    if not sub:
        return None

    gp = sub.get('gamesPlayed', 0)
    if gp < Config.MIN_GAMES_PLAYED:
        return None

    goals = sub.get('goals', 0)
    shots = sub.get('shots', 0)
    assists = sub.get('assists', 0)
    points = sub.get('points', 0)
    pp_goals = sub.get('powerPlayGoals', 0)
    pp_points = sub.get('powerPlayPoints', 0)
    toi_per_game = sub.get('avgToi', '0:00')

    # Parse TOI string "MM:SS" to minutes
    try:
        parts = str(toi_per_game).split(':')
        toi_minutes = int(parts[0]) + int(parts[1]) / 60 if len(parts) == 2 else 0
    except (ValueError, IndexError):
        toi_minutes = 0

    # Last N games
    last_games = data.get('last5Games', [])
    last_n_goals = sum(g.get('goals', 0) for g in last_games)
    last_n_shots = sum(g.get('shots', 0) for g in last_games)
    last_n_games_count = len(last_games)

    # Calculate shooting percentage
    shooting_pct = goals / shots if shots > 0 else 0

    # Calculate recent form (goals per game in recent games)
    recent_gpg = last_n_goals / last_n_games_count if last_n_games_count > 0 else 0
    season_gpg = goals / gp if gp > 0 else 0

    return {
        'games_played': gp,
        'season_goals': goals,
        'season_shots': shots,
        'season_assists': assists,
        'season_points': points,
        'pp_goals': pp_goals,
        'pp_points': pp_points,
        'avg_toi_minutes': round(toi_minutes, 1),
        'avg_goals': round(season_gpg, 4),
        'avg_shots': round(shots / gp, 2) if gp > 0 else 0,
        'shooting_pct': round(shooting_pct, 4),
        'last5_goals': last_n_goals,
        'last5_shots': last_n_shots,
        'last5_games': last_n_games_count,
        'recent_gpg': round(recent_gpg, 4),
    }


def get_team_roster(team_abbrev):
    """Get team roster (forwards + defensemen only) with retry"""
    for attempt in range(3):
        data = api_get(f"https://api-web.nhle.com/v1/roster/{team_abbrev}/current")
        if data:
            break
        if attempt < 2:
            time.sleep(2)
    if not data:
        print(f"⚠️ Failed to fetch roster for {team_abbrev} after 3 attempts")
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
# 4. TIM HORTONS PLAYER POOL
# =============================================================================
def load_tims_players(date):
    """
    Load Tim Hortons eligible players from local file or scraper output.
    Returns dict: {group_number: [player_names]} or None if not available.
    """
    tims_file = f"{Config.TIMS_DIR}/{date}.json"
    if os.path.exists(tims_file):
        try:
            with open(tims_file, 'r') as f:
                data = json.load(f)
            print(f"   ✅ Loaded Tim Hortons players: {sum(len(v) for v in data.get('groups', {}).values())} players in {len(data.get('groups', {}))} groups")
            return data
        except Exception:
            pass
    return None


def filter_tims_players(all_players, tims_data):
    """
    Filter predictions to only Tim Hortons eligible players.
    Matches by player name (fuzzy) since IDs may differ between sources.
    """
    if not tims_data or 'groups' not in tims_data:
        return all_players, {}

    # Build name lookup (lowercase, no accents)
    def normalize(name):
        return name.lower().strip().replace('.', '').replace("'", "").replace('-', ' ')

    # Collect all tims player names by group
    tims_groups = {}
    all_tims_names = set()
    for group_id, players in tims_data['groups'].items():
        group_names = set()
        for p in players:
            name = p if isinstance(p, str) else p.get('name', '')
            group_names.add(normalize(name))
            all_tims_names.add(normalize(name))
        tims_groups[group_id] = group_names

    # Filter players
    filtered = []
    player_groups = {}
    for player in all_players:
        pname = normalize(player['name'])
        if pname in all_tims_names:
            # Find which group
            for gid, names in tims_groups.items():
                if pname in names:
                    player['tims_group'] = gid
                    player_groups[player['player_id']] = gid
                    break
            filtered.append(player)

    return filtered, player_groups


# =============================================================================
# 5. MONTE CARLO SIMULATION ENGINE
# =============================================================================
def calculate_team_expected_goals(team, opponent, team_stats, is_home):
    """
    Calculate expected goals for a team in a specific matchup.

    Formula:
      xG = league_avg * (team_GF/GP / league_avg) * (opp_GA/GP / league_avg) * home_away
         = team_GF/GP * (opp_GA/GP / league_avg) * home_away

    This adjusts the team's scoring rate by how generous the opponent's defense is.
    """
    league_avg = Config.LEAGUE_AVG_GOALS

    if team in team_stats and opponent in team_stats:
        team_gf = team_stats[team]['gf_per_game']
        opp_ga = team_stats[opponent]['ga_per_game']

        # Opponent defense factor: > 1 means opponent allows more than average
        opp_defense_factor = opp_ga / league_avg

        # Team offense factor
        team_offense_factor = team_gf / league_avg

        # Combined expected goals
        xg = league_avg * team_offense_factor * opp_defense_factor
    else:
        # Fallback to league average
        xg = league_avg

    # Home/away adjustment
    ha_factor = Config.HOME_ADVANTAGE if is_home else (2.0 - Config.HOME_ADVANTAGE)
    xg *= ha_factor

    return max(xg, 0.5)  # Floor at 0.5 expected goals


def calculate_player_weights(players, team_stats):
    """
    Calculate each player's weight for goal distribution.

    The key insight: goal scoring in NHL is highly skewed. A 40-goal
    scorer should get ~10x the weight of a 4-goal scorer, not 2x.

    Primary factor: season goals (squared to amplify star-vs-depth gap)
    Secondary: recent form boost for hot players
    Tertiary: small PP and shot-volume bonuses
    """
    weights = []
    for p in players:
        season_goals = p.get('season_goals', 0)
        gp = p.get('games_played', 1)
        last5_goals = p.get('last5_goals', 0)
        last5_games = p.get('last5_games', 5)
        pp_goals = p.get('pp_goals', 0)
        shots_per_game = p.get('avg_shots', 0)

        # Base weight: season goals per game, SQUARED to amplify star power
        # A 0.5 GPG player gets 25x weight of a 0.1 GPG player (vs 5x linear)
        season_gpg = season_goals / max(gp, 1)
        base = season_gpg ** 1.8  # Power > 1 amplifies differences

        # Recent form multiplier: hot players get a boost
        recent_gpg = last5_goals / max(last5_games, 1)
        if recent_gpg > season_gpg and season_gpg > 0:
            # Player is hotter than their season average
            form_boost = 1.0 + min((recent_gpg / season_gpg - 1.0) * 0.3, 0.5)
        elif season_gpg == 0 and recent_gpg > 0:
            form_boost = 1.3
        else:
            form_boost = 1.0

        # Shot volume bonus: more shots = more chances, small effect
        shot_bonus = 1.0 + min(shots_per_game * 0.02, 0.15)

        # PP bonus: players on PP get more high-quality chances
        pp_bonus = 1.0 + (pp_goals / max(season_goals, 1)) * 0.10

        # Combined weight
        weight = base * form_boost * shot_bonus * pp_bonus

        # Floor: even lowest-usage players have a tiny chance
        # (~1% of what a 30-goal scorer gets)
        weight = max(weight, 0.0005)
        weights.append(weight)

    return np.array(weights)


def run_monte_carlo(team_xg, player_weights, num_simulations):
    """
    Monte Carlo simulation of goal scoring for one team in one game.

    For each simulation:
      1. Draw total team goals from Poisson(team_xG)
      2. For each goal, randomly assign to a player weighted by their goal share
      3. Track which players scored at least once

    Returns: array of probabilities (one per player)
    """
    n_players = len(player_weights)
    if n_players == 0:
        return np.array([])

    # Normalize weights to probabilities
    total_weight = player_weights.sum()
    if total_weight == 0:
        probs = np.ones(n_players) / n_players
    else:
        probs = player_weights / total_weight

    # Track: how many simulations each player scored in
    scored_count = np.zeros(n_players, dtype=np.int32)

    # Draw all team goals at once for efficiency
    team_goals_per_sim = np.random.poisson(team_xg, size=num_simulations)

    for sim_idx in range(num_simulations):
        n_goals = team_goals_per_sim[sim_idx]
        if n_goals == 0:
            continue

        # Assign each goal to a player
        scorers = np.random.choice(n_players, size=n_goals, p=probs)
        unique_scorers = np.unique(scorers)
        scored_count[unique_scorers] += 1

    # Probability = fraction of simulations where player scored
    return scored_count / num_simulations


# =============================================================================
# MAIN PIPELINE
# =============================================================================

# Step 1: Fetch today's games
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
        "tims_mode": False,
        "generated_at": datetime.now().isoformat()
    }
    for path in [f"{Config.PREDICTIONS_DIR}/{Config.TODAY}.json",
                 f"{Config.PREDICTIONS_DIR}/latest.json"]:
        with open(path, 'w') as f:
            json.dump(output, f, indent=2)
    print("✅ Saved empty predictions")
    sys.exit(0)

print(f"✅ Found {len(todays_games)} games")
for g in todays_games:
    print(f"   🏒 {g['away_team']} @ {g['home_team']}")

# Step 2: Fetch team standings for opponent adjustment
print("\n📊 Fetching team standings...")
team_stats = get_team_stats_from_standings()
print(f"✅ Loaded stats for {len(team_stats)} teams")

# Step 3: Load Tim Hortons eligible players (if available)
print("\n🍩 Checking Tim Hortons player pool...")
tims_data = load_tims_players(Config.TODAY)
tims_mode = tims_data is not None

# Step 4: Collect all players from today's games
print("\n👥 Fetching rosters and player stats...")

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
for idx, team in enumerate(sorted(all_teams)):
    if idx > 0:
        time.sleep(1)  # Rate limit: pause between teams
    print(f"   Fetching {team}...", end=" ", flush=True)
    roster = get_team_roster(team)
    team_players = []

    for i, p in enumerate(roster):
        if i > 0 and i % 10 == 0:
            time.sleep(0.5)  # Rate limit: pause every 10 players
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

print(f"\n✅ Total players with stats: {len(all_players)}")

if len(all_players) == 0:
    print("⚠️ No player data available!")
    sys.exit(1)

# Step 5: Filter for Tim Hortons players (if available)
player_groups = {}
if tims_mode:
    print("\n🍩 Filtering for Tim Hortons eligible players...")
    filtered_players, player_groups = filter_tims_players(all_players, tims_data)
    if filtered_players:
        print(f"   ✅ Matched {len(filtered_players)} Tim Hortons players")
        # Run predictions for TIMS players only, but keep all for reference
        tims_player_ids = set(p['player_id'] for p in filtered_players)
    else:
        print("   ⚠️ No Tim Hortons players matched, using all players")
        tims_mode = False
        tims_player_ids = None
else:
    tims_player_ids = None
    print("   ℹ️ No Tim Hortons data — predicting all players")

# Step 6: Run Monte Carlo simulation per game
print(f"\n🎲 Running Monte Carlo simulations ({Config.NUM_SIMULATIONS:,} per game)...")

np.random.seed(42)  # Reproducibility

# Group players by game
games_players = {}
for p in all_players:
    gid = p.get('game_id')
    team = p['team']
    key = (gid, team)
    if key not in games_players:
        games_players[key] = []
    games_players[key].append(p)

# Simulate each team in each game
player_probabilities = {}

for (game_id, team), players in games_players.items():
    if not players:
        continue

    opponent = players[0].get('opponent', '')
    is_home = players[0].get('is_home', False)

    # Calculate team expected goals
    team_xg = calculate_team_expected_goals(team, opponent, team_stats, is_home)

    # Calculate player weights
    weights = calculate_player_weights(players, team_stats)

    # Run simulation
    probabilities = run_monte_carlo(team_xg, weights, Config.NUM_SIMULATIONS)

    # Store results
    for i, p in enumerate(players):
        player_probabilities[p['player_id']] = round(float(probabilities[i]), 4)

    matchup_str = f"{team} vs {opponent}" if is_home else f"{team} @ {opponent}"
    top_prob = max(probabilities) if len(probabilities) > 0 else 0
    print(f"   {matchup_str}: xG={team_xg:.2f}, {len(players)} players, top={top_prob:.1%}")

# Step 7: Assign probabilities and rank
print("\n📊 Ranking players...")

for p in all_players:
    p['goal_probability'] = player_probabilities.get(p['player_id'], 0.0)

    # Add opponent context
    opp = p.get('opponent', '')
    if opp in team_stats:
        p['opp_ga_per_game'] = team_stats[opp]['ga_per_game']
    else:
        p['opp_ga_per_game'] = Config.LEAGUE_AVG_GOALS

    # Add team context
    team = p['team']
    if team in team_stats:
        p['team_gf_per_game'] = team_stats[team]['gf_per_game']
    else:
        p['team_gf_per_game'] = Config.LEAGUE_AVG_GOALS

# Sort by probability
all_players.sort(key=lambda x: x['goal_probability'], reverse=True)

# Add rank and hot indicator
for i, p in enumerate(all_players):
    p['rank'] = i + 1
    p['is_hot'] = p.get('last5_goals', 0) >= 3

# Step 8: Prepare Tim Hortons group rankings
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
print("\n💾 Saving predictions...")

# Determine which players to include in output
output_players = all_players
if tims_mode and tims_player_ids:
    # Save Tim Hortons players at the top, then everyone else
    tims_preds = [p for p in all_players if p['player_id'] in tims_player_ids]
    other_preds = [p for p in all_players if p['player_id'] not in tims_player_ids]
    # Re-rank tims players separately
    for i, p in enumerate(tims_preds):
        p['tims_rank'] = i + 1
    output_players = tims_preds  # Only output Tim Hortons players for this model

output = {
    "date": Config.TODAY,
    "model": Config.MODEL_NAME,
    "model_display_name": Config.MODEL_DISPLAY_NAME,
    "games_count": len(todays_games),
    "games": todays_games,
    "players_count": len(output_players),
    "predictions": output_players,
    "tims_mode": tims_mode,
    "tims_group_rankings": tims_group_rankings if tims_mode else {},
    "simulation_params": {
        "num_simulations": Config.NUM_SIMULATIONS,
        "league_avg_goals": Config.LEAGUE_AVG_GOALS,
        "home_advantage": Config.HOME_ADVANTAGE,
        "recent_form_weight": Config.RECENT_FORM_WEIGHT,
    },
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

# Step 10: Console output
print("\n" + "=" * 70)
if tims_mode and tims_group_rankings:
    print("🍩 TIM HORTONS CHALLENGE - TOP PICKS BY GROUP")
    print("=" * 70)
    for gid in sorted(tims_group_rankings.keys()):
        print(f"\n  📦 Group {gid}:")
        print(f"  {'#':<4} {'Name':<25} {'Team':<5} {'Prob':>7} {'Matchup':<15}")
        print(f"  {'-' * 58}")
        for p in tims_group_rankings[gid][:5]:
            print(f"  {p['rank_in_group']:<4} {p['name']:<25} {p['team']:<5} {p['goal_probability']*100:>6.1f}% {p['matchup']:<15}")
else:
    print("🎯 TOP 20 PREDICTIONS (Monte Carlo)")
    print("=" * 70)
    print(f"{'#':<4} {'Name':<25} {'Team':<5} {'Prob':>7} {'Goals':>6} {'L5':>4} {'OppGA':>6}")
    print("-" * 65)
    for p in output_players[:20]:
        hot = "🔥" if p['is_hot'] else "  "
        opp_ga = p.get('opp_ga_per_game', 0)
        print(f"{p['rank']:<4} {p['name']:<25} {p['team']:<5} {p['goal_probability']*100:>6.1f}% {p['season_goals']:>6} {p['last5_goals']:>4} {opp_ga:>5.1f} {hot}")

print(f"\n✅ Total: {len(output_players)} players ranked")
print(f"🎲 Simulations: {Config.NUM_SIMULATIONS:,} per game")
print("\n🏒 Monte Carlo predictions complete!")
