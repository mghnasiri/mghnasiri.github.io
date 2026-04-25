"""
NHL xG Data Collector — Play-by-Play Shot Extractor
====================================================
Fetches play-by-play data from the NHL API and extracts shot events
with engineered features for xG model training.

Usage:
  python collect_xg_data.py                  # Collect yesterday's games
  python collect_xg_data.py --backfill 14    # Collect last 14 days
  python collect_xg_data.py --date 2026-01-15  # Collect specific date

Author: Mohammad G. Nasiri
"""

import requests
import json
import os
import sys
import csv
import math
import time
from datetime import datetime, timedelta


# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    DATA_DIR = "data"
    XG_TRAINING_DIR = f"{DATA_DIR}/xg_training"
    SHOTS_CSV = f"{XG_TRAINING_DIR}/shots.csv"
    COLLECTION_LOG = f"{XG_TRAINING_DIR}/collection_log.json"
    RATE_LIMIT_SECONDS = 0.5

    NHL_API = "https://api-web.nhle.com/v1"

    # CSV column order
    CSV_COLUMNS = [
        'game_id', 'game_date', 'event_index', 'player_id', 'team',
        'shot_distance', 'shot_angle', 'shot_type', 'strength_state',
        'is_powerplay', 'is_rebound', 'seconds_since_last_event',
        'is_empty_net', 'score_differential', 'period',
        'x_coord', 'y_coord', 'prior_event_type', 'prior_event_distance',
        'is_goal',
    ]

    # Shot event type codes
    SHOT_EVENTS = {'goal', 'shot-on-goal', 'missed-shot'}


os.makedirs(Config.XG_TRAINING_DIR, exist_ok=True)


# =============================================================================
# API HELPERS
# =============================================================================
def api_get(url, timeout=15):
    """Safe API GET with exponential-backoff retry (1s, 2s).
    Without backoff between retries, all 3 attempts hit the same
    throttled state and fail in <1s. Backs off on 429, 5xx, and
    network exceptions; returns immediately on 200 or 404."""
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 404:
                return None
        except requests.RequestException:
            pass
        if attempt < 2:
            time.sleep(2 ** attempt)
    return None


# =============================================================================
# GAME DISCOVERY
# =============================================================================
def get_completed_games(date_str):
    """Get all completed games for a given date."""
    data = api_get(f"{Config.NHL_API}/schedule/{date_str}")
    if not data:
        return []

    games = []
    for day in data.get('gameWeek', []):
        if day['date'] != date_str:
            continue
        for game in day.get('games', []):
            game_type = game.get('gameType')
            state = game.get('gameState', '')
            if game_type in [2, 3] and state in ('OFF', 'FINAL'):
                games.append({
                    'game_id': game['id'],
                    'home_team': game['homeTeam']['abbrev'],
                    'away_team': game['awayTeam']['abbrev'],
                    'date': date_str,
                })
    return games


# =============================================================================
# PLAY-BY-PLAY EXTRACTION
# =============================================================================
def parse_situation_code(situation_code, event_team_id, home_team_id):
    """
    Parse the 4-digit situationCode into strength state.

    Format: [AwayGoalie][AwaySkaters][HomeSkaters][HomeGoalie]
    Example: "1551" = both goalies, 5v5
    """
    if not situation_code or len(str(situation_code)) != 4:
        return '5v5', False

    code = str(situation_code)
    away_goalies = int(code[0])
    away_skaters = int(code[1])
    home_skaters = int(code[2])
    home_goalies = int(code[3])

    # Determine from shooting team's perspective
    if event_team_id == home_team_id:
        my_skaters = home_skaters
        opp_skaters = away_skaters
    else:
        my_skaters = away_skaters
        opp_skaters = home_skaters

    state = f"{my_skaters}v{opp_skaters}"
    is_pp = my_skaters > opp_skaters

    return state, is_pp


def extract_shots_from_game(game_info):
    """
    Parse play-by-play data and extract shot events with features.

    Returns list of dicts (one per shot), ready for CSV output.
    """
    game_id = game_info['game_id']
    game_date = game_info['date']

    data = api_get(f"{Config.NHL_API}/gamecenter/{game_id}/play-by-play")
    if not data:
        return []

    plays = data.get('plays', [])
    home_team_id = data.get('homeTeam', {}).get('id', 0)
    away_team_id = data.get('awayTeam', {}).get('id', 0)
    home_abbrev = game_info['home_team']
    away_abbrev = game_info['away_team']

    shots = []
    prev_event = None

    # PBP score-differential leakage fix:
    # The NHL PBP API only populates `homeScore`/`awayScore` on GOAL events
    # (post-goal). For shot-on-goal / missed-shot events, those fields are
    # absent and default to 0 — making score_diff = 0 for ~99% of non-goal
    # shots while goal rows have arbitrary values. The xG model then learns
    # the trivial "score_diff != 0 -> goal" leak, inflating CV AUC and
    # producing miscalibrated predictions.
    #
    # Fix: track running home/away score ourselves through the play list.
    # We increment ONLY after a goal event has been recorded, so the score
    # at the moment of any shot is the score BEFORE the shot's outcome —
    # the same view a real-time predictor would have.
    running_home_score = 0
    running_away_score = 0

    for i, play in enumerate(plays):
        event_type = play.get('typeDescKey', '')

        # Track all events for "prior event" features
        details = play.get('details', {})

        if event_type not in Config.SHOT_EVENTS:
            prev_event = {
                'type': event_type,
                'time_str': play.get('timeInPeriod', '0:00'),
                'period': play.get('periodDescriptor', {}).get('number', 1),
                'x': details.get('xCoord', 0),
                'y': details.get('yCoord', 0),
            }
            continue

        # ── Extract raw data ──
        x_raw = details.get('xCoord', 0) or 0
        y_raw = details.get('yCoord', 0) or 0
        shot_type = details.get('shotType', 'wrist')
        situation_code = play.get('situationCode', '1551')
        period_num = play.get('periodDescriptor', {}).get('number', 1)
        time_str = play.get('timeInPeriod', '0:00')
        event_team_id = details.get('eventOwnerTeamId', 0)

        # Determine shooting team
        if event_team_id == home_team_id:
            team_abbrev = home_abbrev
        elif event_team_id == away_team_id:
            team_abbrev = away_abbrev
        else:
            team_abbrev = ''

        # Get player ID
        if event_type == 'goal':
            player_id = details.get('scoringPlayerId', 0)
        else:
            player_id = details.get('shootingPlayerId', 0)

        if not player_id:
            prev_event = {
                'type': event_type, 'time_str': time_str,
                'period': period_num, 'x': x_raw, 'y': y_raw,
            }
            continue

        # ── Normalize coordinates ──
        # Ensure all shots attack towards positive x (net at x=89)
        if x_raw < 0:
            x_norm = -x_raw
            y_norm = -y_raw
        else:
            x_norm = x_raw
            y_norm = y_raw

        # ── Calculate shot distance and angle ──
        dx = 89 - x_norm
        dy = y_norm
        shot_distance = math.sqrt(dx * dx + dy * dy)
        shot_angle = abs(math.degrees(math.atan2(dy, max(dx, 0.1))))

        # ── Strength state ──
        strength_state, is_powerplay = parse_situation_code(
            situation_code, event_team_id, home_team_id
        )

        # ── Empty net ──
        goalie_id = details.get('goalieInNetId', None)
        is_empty_net = 1 if (goalie_id is None or goalie_id == 0) else 0

        # ── Score differential ──
        # Use our running tally (pre-shot score) instead of the API field
        # (which is post-goal-only and creates label leakage). See the
        # comment at the top of this loop.
        if event_team_id == home_team_id:
            score_diff = running_home_score - running_away_score
        else:
            score_diff = running_away_score - running_home_score

        # ── Time-based features ──
        seconds_since_last = 0.0
        is_rebound = 0
        prior_event_type = 'none'
        prior_event_distance = 0.0

        if prev_event:
            prior_event_type = prev_event.get('type', 'none')
            # Calculate time since last event
            try:
                cur_parts = time_str.split(':')
                prev_parts = prev_event['time_str'].split(':')
                cur_secs = int(cur_parts[0]) * 60 + int(cur_parts[1])
                prev_secs = int(prev_parts[0]) * 60 + int(prev_parts[1])

                if play.get('periodDescriptor', {}).get('number', 1) == prev_event.get('period', 1):
                    # NHL clock counts down, so previous event has higher time
                    seconds_since_last = max(prev_secs - cur_secs, 0)
                else:
                    seconds_since_last = 30.0  # Different period
            except (ValueError, IndexError):
                seconds_since_last = 30.0

            # Rebound: shot within 3 seconds of a prior shot
            if prior_event_type in ('shot-on-goal', 'goal', 'missed-shot'):
                is_rebound = 1 if seconds_since_last <= 3.0 else 0

            # Prior event distance from net
            px = abs(prev_event.get('x', 0))
            py = prev_event.get('y', 0)
            prior_event_distance = math.sqrt((89 - px) ** 2 + py ** 2)

        # ── Target: is_goal ──
        is_goal = 1 if event_type == 'goal' else 0

        shots.append({
            'game_id': game_id,
            'game_date': game_date,
            'event_index': i,
            'player_id': player_id,
            'team': team_abbrev,
            'shot_distance': round(shot_distance, 2),
            'shot_angle': round(shot_angle, 2),
            'shot_type': shot_type or 'wrist',
            'strength_state': strength_state,
            'is_powerplay': int(is_powerplay),
            'is_rebound': is_rebound,
            'seconds_since_last_event': round(seconds_since_last, 1),
            'is_empty_net': is_empty_net,
            'score_differential': score_diff,
            'period': period_num,
            'x_coord': round(x_norm, 1),
            'y_coord': round(y_norm, 1),
            'prior_event_type': prior_event_type,
            'prior_event_distance': round(prior_event_distance, 2),
            'is_goal': is_goal,
        })

        # If this shot was a goal, advance the running score AFTER recording
        # the shot row — so the row itself reflects the pre-shot score.
        if is_goal:
            if event_team_id == home_team_id:
                running_home_score += 1
            elif event_team_id == away_team_id:
                running_away_score += 1

        # Update prev_event
        prev_event = {
            'type': event_type, 'time_str': time_str,
            'period': period_num, 'x': x_raw, 'y': y_raw,
        }

    return shots


# =============================================================================
# CSV MANAGEMENT
# =============================================================================
def load_collection_log():
    """Load the set of already-processed game IDs."""
    try:
        if os.path.exists(Config.COLLECTION_LOG):
            with open(Config.COLLECTION_LOG, 'r') as f:
                data = json.load(f)
            return set(data.get('processed_games', []))
    except Exception:
        pass
    return set()


def save_collection_log(processed_games):
    """Save the set of processed game IDs."""
    with open(Config.COLLECTION_LOG, 'w') as f:
        json.dump({
            'processed_games': sorted(processed_games),
            'total_games': len(processed_games),
            'last_updated': datetime.now().isoformat(),
        }, f, indent=2)


def append_shots_to_csv(shots):
    """Append shot rows to the CSV file."""
    if not shots:
        return

    file_exists = os.path.exists(Config.SHOTS_CSV)

    with open(Config.SHOTS_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=Config.CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        for shot in shots:
            writer.writerow({col: shot.get(col, '') for col in Config.CSV_COLUMNS})


# =============================================================================
# MAIN
# =============================================================================
def main():
    # Parse arguments
    backfill_days = 1  # Default: yesterday only
    target_date = None

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == '--backfill' and i + 1 < len(args):
            backfill_days = int(args[i + 1])
            i += 2
        elif args[i] == '--date' and i + 1 < len(args):
            target_date = args[i + 1]
            i += 2
        else:
            i += 1

    print("=" * 60)
    print("  NHL xG DATA COLLECTOR")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    processed_games = load_collection_log()
    print(f"  Previously processed: {len(processed_games)} games")

    # Determine dates to collect
    if target_date:
        dates = [target_date]
    else:
        today = datetime.now()
        dates = [
            (today - timedelta(days=d)).strftime('%Y-%m-%d')
            for d in range(1, backfill_days + 1)
        ]

    print(f"  Collecting {len(dates)} day(s): {dates[0]} to {dates[-1]}")

    total_shots = 0
    total_games = 0
    new_games = 0

    for date_str in dates:
        games = get_completed_games(date_str)
        if not games:
            continue

        for game in games:
            gid = game['game_id']

            # Skip already processed
            if gid in processed_games:
                continue

            # Rate limit
            time.sleep(Config.RATE_LIMIT_SECONDS)

            shots = extract_shots_from_game(game)
            if shots:
                append_shots_to_csv(shots)
                processed_games.add(gid)
                total_shots += len(shots)
                new_games += 1
                print(f"    {game['away_team']} @ {game['home_team']} "
                      f"({date_str}): {len(shots)} shots "
                      f"({sum(1 for s in shots if s['is_goal'])} goals)")

            total_games += 1

    # Save updated log
    save_collection_log(processed_games)

    print(f"\n  New games collected: {new_games}")
    print(f"  New shots extracted: {total_shots}")
    print(f"  Total games in log: {len(processed_games)}")

    # Show CSV size
    if os.path.exists(Config.SHOTS_CSV):
        size_kb = os.path.getsize(Config.SHOTS_CSV) / 1024
        print(f"  CSV file size: {size_kb:.1f} KB")

    print("\n  Collection complete!")


if __name__ == "__main__":
    main()
