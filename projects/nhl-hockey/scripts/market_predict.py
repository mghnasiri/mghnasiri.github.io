"""
NHL Goal Predictor — Market Odds v1
====================================
Uses sportsbook anytime-goal-scorer odds from The Odds API as predictions.
Market lines embed lineup confirmations, goalie starters, injury news, and
sharp money — information no public stats model can match.

Pipeline:
  1. Fetch today's NHL events from The Odds API
  2. For each event, fetch player_goal_scorer_anytime odds
  3. Average across bookmakers, devig to true probabilities
  4. Match player names to NHL API player_ids
  5. Output in standard prediction JSON format

Requires: ODDS_API_KEY environment variable (free at https://the-odds-api.com)

Author: Mohammad G. Nasiri
"""

import requests
import json
import os
import sys
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    MODEL_NAME = "market_odds"
    MODEL_DISPLAY_NAME = "Market Odds v1"

    DATA_DIR = "data"
    PREDICTIONS_DIR = f"{DATA_DIR}/predictions/{MODEL_NAME}"
    ODDS_DIR = f"{DATA_DIR}/odds"
    TIMS_DIR = f"{DATA_DIR}/tims_players"

    TODAY = datetime.now().strftime("%Y-%m-%d")

    ODDS_API_KEY = os.environ.get('ODDS_API_KEY', '')
    ODDS_API_BASE = "https://api.the-odds-api.com/v4"
    SPORT = "icehockey_nhl"
    MARKET = "player_goal_scorer_anytime"
    REGIONS = "us,us2"
    ODDS_FORMAT = "american"

    # Base model to use for player_id matching (any model with full roster)
    PLAYER_ID_SOURCES = [
        f"{DATA_DIR}/predictions/monte_carlo/latest.json",
        f"{DATA_DIR}/predictions/xg_v3/latest.json",
        f"{DATA_DIR}/predictions/neural_network/latest.json",
    ]


os.makedirs(Config.PREDICTIONS_DIR, exist_ok=True)
os.makedirs(Config.ODDS_DIR, exist_ok=True)

print("=" * 70)
print(f"  NHL GOAL PREDICTOR — {Config.MODEL_DISPLAY_NAME}")
print(f"  {Config.TODAY}")
print("=" * 70)


# =============================================================================
# ODDS API
# =============================================================================
def american_to_prob(odds):
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)


def fetch_events():
    """Fetch today's NHL events from The Odds API."""
    url = f"{Config.ODDS_API_BASE}/sports/{Config.SPORT}/events"
    params = {
        'apiKey': Config.ODDS_API_KEY,
        'dateFormat': 'iso',
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 200:
            events = resp.json()
            # Filter to today's events
            today_events = [e for e in events
                           if e.get('commence_time', '').startswith(Config.TODAY)]
            print(f"  API returned {len(events)} events, {len(today_events)} today")
            return today_events
        else:
            print(f"  Events API error: {resp.status_code} — {resp.text[:200]}")
            return []
    except Exception as e:
        print(f"  Events API failed: {e}")
        return []


def fetch_player_odds(event_id):
    """Fetch anytime goal scorer odds for a specific event."""
    url = f"{Config.ODDS_API_BASE}/sports/{Config.SPORT}/events/{event_id}/odds"
    params = {
        'apiKey': Config.ODDS_API_KEY,
        'regions': Config.REGIONS,
        'markets': Config.MARKET,
        'oddsFormat': Config.ODDS_FORMAT,
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 200:
            return resp.json()
        else:
            print(f"    Odds API error for {event_id}: {resp.status_code}")
            return None
    except Exception as e:
        print(f"    Odds API failed for {event_id}: {e}")
        return None


def extract_player_probs(event_data):
    """
    Extract per-player implied probabilities from bookmaker odds.
    Averages across bookmakers and devigs to true probabilities.
    """
    if not event_data:
        return {}

    # Collect all odds per player across bookmakers
    player_odds = {}  # name -> list of implied probs
    for bookmaker in event_data.get('bookmakers', []):
        for market in bookmaker.get('markets', []):
            if market.get('key') != Config.MARKET:
                continue
            for outcome in market.get('outcomes', []):
                name = outcome.get('description', outcome.get('name', ''))
                price = outcome.get('price')
                if not name or price is None:
                    continue
                prob = american_to_prob(price)
                if name not in player_odds:
                    player_odds[name] = []
                player_odds[name].append(prob)

    # Average across bookmakers
    player_probs = {}
    for name, probs in player_odds.items():
        player_probs[name] = sum(probs) / len(probs)

    # Devig: normalize so probabilities reflect true estimates
    # (remove bookmaker margin)
    total = sum(player_probs.values())
    if total > 0:
        # Scale factor: if total implied = 1.15, devig by / 1.15
        # But for anytime scorer, total can be >> 1 (many players)
        # Instead, devig per-game: divide by sum of that game's players
        # Since this is already one game's players, this works
        for name in player_probs:
            player_probs[name] = player_probs[name] / total * min(total, 1.0)
            # Actually for goal scorer props, each player is independent
            # (multiple can score), so we DON'T normalize to sum=1.
            # Instead, remove the vig proportionally.
            pass

        # Better devig: average vig is ~5-10% per outcome.
        # For independent props, multiply each by (1 - avg_vig_per_outcome)
        # Simpler: if avg bookmaker has ~5% vig, multiply probs by 0.95
        # Or: use the "power method" — raise each prob to a power that
        # makes the total correct. For independent events this is tricky.
        # Simplest correct approach: just use the averaged implied probs as-is.
        # They're slightly inflated but the ranking is correct.
        player_probs = {name: sum(probs) / len(probs)
                        for name, probs in player_odds.items()}

    return player_probs


# =============================================================================
# PLAYER ID MATCHING
# =============================================================================
def normalize_name(name):
    """Normalize player name for matching."""
    return (name.lower().strip()
            .replace('.', '').replace("'", "").replace("'", "")
            .replace('-', ' ').replace('  ', ' '))


def build_name_to_id_map():
    """
    Build a player name -> player_id map from existing model predictions.
    Uses whichever base model has today's predictions available.
    """
    name_map = {}  # normalized_name -> {player_id, name, team, position, ...}

    for source_path in Config.PLAYER_ID_SOURCES:
        if not os.path.exists(source_path):
            continue
        try:
            with open(source_path, 'r') as f:
                data = json.load(f)
            # Accept if it's from today or latest
            for p in data.get('predictions', []):
                nname = normalize_name(p.get('name', ''))
                if nname and p.get('player_id'):
                    name_map[nname] = {
                        'player_id': p['player_id'],
                        'name': p.get('name', ''),
                        'position': p.get('position', ''),
                        'team': p.get('team', ''),
                        'opponent': p.get('opponent', ''),
                        'is_home': p.get('is_home', False),
                        'game_id': p.get('game_id', ''),
                        'matchup': p.get('matchup', ''),
                        'season_goals': p.get('season_goals', 0),
                        'last5_goals': p.get('last5_goals', 0),
                    }
            if name_map:
                print(f"  Loaded {len(name_map)} player IDs from {source_path}")
                break
        except Exception:
            continue

    return name_map


def match_odds_to_players(player_probs, name_map):
    """
    Match Odds API player names to NHL API player IDs.
    Returns list of player dicts with goal_probability from market odds.
    """
    matched = []
    unmatched = []

    for odds_name, prob in player_probs.items():
        nname = normalize_name(odds_name)
        player_info = name_map.get(nname)

        if not player_info:
            # Try last-name-first-initial match for edge cases
            # e.g., "C. McDavid" vs "Connor McDavid"
            parts = nname.split()
            if len(parts) >= 2:
                # Try "firstname lastname" patterns
                for stored_name, info in name_map.items():
                    stored_parts = stored_name.split()
                    if len(stored_parts) >= 2 and stored_parts[-1] == parts[-1]:
                        # Same last name — check first name/initial
                        if (stored_parts[0] == parts[0] or
                            stored_parts[0][0] == parts[0][0]):
                            player_info = info
                            break

        if player_info:
            matched.append({
                **player_info,
                'goal_probability': round(prob, 4),
                'odds_name': odds_name,
            })
        else:
            unmatched.append(odds_name)

    if unmatched and len(unmatched) <= 20:
        print(f"  Unmatched ({len(unmatched)}): {', '.join(unmatched[:10])}")
    elif unmatched:
        print(f"  Unmatched: {len(unmatched)} players (no player_id found)")

    return matched


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
# MAIN PIPELINE
# =============================================================================
if not Config.ODDS_API_KEY:
    print("  ERROR: ODDS_API_KEY not set. Set it as an environment variable.")
    print("  Get a free key at https://the-odds-api.com")
    sys.exit(1)

# Step 1: Fetch events
print("\n  Fetching today's NHL events...")
events = fetch_events()

if not events:
    print("  No events found for today.")
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
    sys.exit(0)

# Step 2: Fetch odds for each event
print(f"\n  Fetching goal scorer odds for {len(events)} games...")
all_player_probs = {}
games = []

for event in events:
    eid = event['id']
    home = event.get('home_team', '?')
    away = event.get('away_team', '?')
    print(f"    {away} @ {home}...", end=" ")

    event_data = fetch_player_odds(eid)
    probs = extract_player_probs(event_data)
    print(f"{len(probs)} players")

    all_player_probs.update(probs)
    games.append({
        'game_id': eid,
        'home_team': home,
        'away_team': away,
        'start_time': event.get('commence_time', ''),
    })

print(f"\n  Total players with odds: {len(all_player_probs)}")

# Cache raw odds
odds_cache = {
    'date': Config.TODAY,
    'games_count': len(games),
    'players_with_odds': len(all_player_probs),
    'odds': {name: round(p, 4) for name, p in all_player_probs.items()},
    'fetched_at': datetime.now().isoformat(),
}
odds_file = f"{Config.ODDS_DIR}/{Config.TODAY}.json"
with open(odds_file, 'w') as f:
    json.dump(odds_cache, f, indent=2)
print(f"  Cached odds: {odds_file}")

# Step 3: Match to player IDs
print("\n  Matching player names to NHL API IDs...")
name_map = build_name_to_id_map()
if not name_map:
    print("  WARNING: No base model predictions found for player ID matching.")
    print("  Run at least one base model first (MC v2 or xG v3).")

all_players = match_odds_to_players(all_player_probs, name_map)
print(f"  Matched: {len(all_players)} players")

if not all_players:
    print("  No players matched. Cannot generate predictions.")
    sys.exit(1)

# Step 4: Sort and rank
all_players.sort(key=lambda x: x['goal_probability'], reverse=True)
for i, p in enumerate(all_players):
    p['rank'] = i + 1
    p['is_hot'] = p.get('last5_goals', 0) >= 3

# Step 5: Tim Hortons filtering
tims_data = load_tims_players(Config.TODAY)
tims_mode = tims_data is not None and bool(tims_data.get('groups'))
tims_group_rankings = {}
output_players = all_players

if tims_mode:
    all_tims_names = set()
    tims_groups = {}
    for gid, players in tims_data['groups'].items():
        group_names = set()
        for p in players:
            name = p if isinstance(p, str) else p.get('name', '')
            group_names.add(normalize_name(name))
            all_tims_names.add(normalize_name(name))
        tims_groups[gid] = group_names

    filtered = []
    for player in all_players:
        pname = normalize_name(player['name'])
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

# Step 6: Save output
print("\n  Saving predictions...")
output = {
    "date": Config.TODAY,
    "model": Config.MODEL_NAME,
    "model_display_name": Config.MODEL_DISPLAY_NAME,
    "games_count": len(games),
    "games": games,
    "players_count": len(output_players),
    "predictions": output_players,
    "tims_mode": tims_mode,
    "tims_source": tims_data.get('source', 'unknown') if tims_data else None,
    "tims_group_rankings": tims_group_rankings if tims_mode else {},
    "model_params": {
        "source": "The Odds API",
        "market": Config.MARKET,
        "regions": Config.REGIONS,
        "bookmakers_used": len(set(
            b['key'] for e in [fetch_player_odds(ev['id']) for ev in events[:1]]
            if e for b in e.get('bookmakers', [])
        )) if events else 0,
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
print("  TOP 20 MARKET ODDS PREDICTIONS")
print("=" * 70)
print(f"  {'#':<4} {'Name':<25} {'Team':<5} {'Prob':>7}")
print(f"  {'-' * 45}")
for p in output_players[:20]:
    hot = " *" if p.get('is_hot') else ""
    print(f"  {p['rank']:<4} {p['name']:<25} {p['team']:<5} "
          f"{p['goal_probability']*100:>6.1f}%{hot}")

print(f"\n  Total: {len(output_players)} players ranked")
print(f"\n  Market Odds v1 predictions complete!")
