"""
NHL Goal Predictor - Fetch Results
==================================
Fetches actual goal scorers from completed games
Compares with Top 10 predictions

Grades yesterday, plus any day in the last BACKFILL_DAYS that has
predictions but no results or only partial ones (games not final yet at
run time, a throttled boxscore, a failed run). Those used to stay
ungraded or half-graded for good.

Author: Mohammad G. Nasiri
"""

import json
import os
import sys
from datetime import datetime, timedelta

from nhl_api import api_get

# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    DATA_DIR = "data"
    RESULTS_DIR = f"{DATA_DIR}/results"
    PREDICTIONS_DIR = f"{DATA_DIR}/predictions"

    # NHL dates are Eastern; the workflow runs with TZ=America/Toronto.
    YESTERDAY = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    BACKFILL_DAYS = 7

FINAL_STATES = {"OFF", "FINAL"}

# Create directories
os.makedirs(Config.RESULTS_DIR, exist_ok=True)

# =============================================================================
# FETCH GAMES
# =============================================================================
def get_games(date):
    """All NHL games for a date; None if the schedule couldn't be fetched
    (distinct from [] = a day with no games)."""
    data = api_get(f"https://api-web.nhle.com/v1/schedule/{date}", attempts=6)
    if not data:
        return None
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
                        'game_state': game.get('gameState', ''),
                        # OK, or PPD/CNCL for a postponed/cancelled game
                        'schedule_state': game.get('gameScheduleState', 'OK'),
                    })
    return games

# =============================================================================
# FETCH GOAL SCORERS
# =============================================================================
def _scorer(player, team_abbrev):
    # Handle name as dict or string
    name = player.get('name', {})
    if isinstance(name, dict):
        player_name = name.get('default', 'Unknown')
    else:
        player_name = str(name) if name else 'Unknown'
    return {
        'player_id': player.get('playerId'),
        'player_name': player_name,
        'team': team_abbrev,
        'goals': player.get('goals', 0)
    }


def get_scorers(game_id):
    """All goal scorers from a game's boxscore; None if it didn't load, so
    a fetch failure can't grade every pick in the game as a miss."""
    data = api_get(f"https://api-web.nhle.com/v1/gamecenter/{game_id}/boxscore")
    if data is None:
        return None
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
                    if player.get('goals', 0) > 0:
                        scorers.append(_scorer(player, team_abbrev))
    else:
        # FALLBACK: Old structure
        for team_type in ['homeTeam', 'awayTeam']:
            team_data = data.get(team_type, {})
            team_abbrev = team_data.get('abbrev', '')
            for player in team_data.get('forwards', []) + team_data.get('defense', []):
                if player.get('goals', 0) > 0:
                    scorers.append(_scorer(player, team_abbrev))

    return scorers

# =============================================================================
# GRADING
# =============================================================================
def compare_models(date, all_scorers, ungraded_teams):
    """Grade every model's top-10 picks for `date` against its scorers.
    Picks on teams whose game wasn't graded don't count either way."""
    # Drop missing/None ids so a boxscore gap can't collapse to a single None
    # that a player_id=0 prediction would then falsely match.
    scorer_ids = {s['player_id'] for s in all_scorers if s.get('player_id')}
    model_comparisons = []
    if not os.path.exists(Config.PREDICTIONS_DIR):
        return model_comparisons

    for model_name in sorted(os.listdir(Config.PREDICTIONS_DIR)):
        pred_file = f"{Config.PREDICTIONS_DIR}/{model_name}/{date}.json"
        if not os.path.exists(pred_file):
            continue
        with open(pred_file, 'r') as f:
            predictions = json.load(f)

        # Grade the model's top picks. Denominator is the number of picks
        # actually graded — a Tims-filtered model can have <10 on short slates,
        # and dividing by a hardcoded 10 understated hit rate on those days.
        TOP_N = 10
        top_picks = [p for p in predictions.get('predictions', [])[:TOP_N]
                     if p.get('team') not in ungraded_teams]
        if not top_picks:
            print(f"   ⚠️ {model_name}: No predictions")
            continue

        graded_picks = []
        for pred in top_picks:
            pid = pred.get('player_id')
            graded_picks.append({
                'rank': pred.get('rank', 0),
                'player_id': pid,
                'name': pred['name'],
                'team': pred['team'],
                'probability': pred.get('goal_probability', 0),
                'scored': bool(pid) and pid in scorer_ids
            })
        hits = sum(p['scored'] for p in graded_picks)
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
        print(f"   📈 {model_name}: {hits}/{n}")
    return model_comparisons


def grade(date, games):
    """Results for `date`'s games, or None when no game can be graded yet."""
    # A postponed/cancelled game never goes final; counting it would leave
    # the day partial for good.
    unplayed = [g for g in games if g['schedule_state'] != 'OK']
    games = [g for g in games if g['schedule_state'] == 'OK']
    ungraded_teams = {t for g in unplayed for t in (g['home_team'], g['away_team'])}
    for g in unplayed:
        print(f"   {g['away_team']} @ {g['home_team']}... not played "
              f"({g['schedule_state']}) — its picks aren't graded")
    if not games:
        print("   ℹ️  No games found for this date")
        return {
            "date": date,
            "games_count": 0,
            "games": [],
            "all_scorers": [],
            "scorers_count": 0,
            "model_comparisons": [],
            "fetched_at": datetime.now().isoformat()
        }

    # Only completed games whose boxscore loaded count. Grading a LIVE or
    # unloaded game would record its real scorers as "did not score".
    all_scorers = []
    graded_games = 0
    for game in games:
        matchup = f"{game['away_team']} @ {game['home_team']}"
        state = game.get('game_state', '')
        if state not in FINAL_STATES:
            print(f"   {matchup}... SKIPPED (state={state or '?'}, not final)")
            ungraded_teams |= {game['home_team'], game['away_team']}
            continue
        scorers = get_scorers(game['game_id'])
        if scorers is None:
            print(f"   {matchup}... SKIPPED (boxscore didn't load)")
            ungraded_teams |= {game['home_team'], game['away_team']}
            continue
        graded_games += 1
        for scorer in scorers:
            scorer['game_id'] = game['game_id']
            scorer['matchup'] = matchup
        all_scorers.extend(scorers)
        print(f"   {matchup}... {len(scorers)} scorers")

    # If nothing is graded, do NOT write a result — every pick would count
    # as a miss and poison the hit-rate metric. A later run retries.
    if graded_games == 0:
        print("   ⛔ No completed game graded yet — leaving this day for a later run.")
        return None

    partial = graded_games < len(games)
    if partial:
        print(f"   ⚠️ {len(games) - graded_games} game(s) not graded — partial; "
              "a later run completes it.")

    return {
        "date": date,
        "games_count": len(games),
        "final_games": graded_games,
        "partial": partial,
        "games": games,
        "all_scorers": all_scorers,
        "scorers_count": len(all_scorers),
        "model_comparisons": compare_models(date, all_scorers, ungraded_teams),
        "fetched_at": datetime.now().isoformat()
    }


def dates_to_grade():
    """Yesterday, plus recent days with predictions whose results are
    missing or partial."""
    dates = []
    yesterday = datetime.strptime(Config.YESTERDAY, "%Y-%m-%d")
    for back in range(Config.BACKFILL_DAYS, 0, -1):
        date = (yesterday - timedelta(days=back)).strftime("%Y-%m-%d")
        has_preds = any(os.path.exists(f"{Config.PREDICTIONS_DIR}/{m}/{date}.json")
                        for m in os.listdir(Config.PREDICTIONS_DIR)) \
            if os.path.isdir(Config.PREDICTIONS_DIR) else False
        result_file = f"{Config.RESULTS_DIR}/{date}.json"
        if not has_preds:
            continue
        if not os.path.exists(result_file):
            dates.append(date)
            continue
        with open(result_file, 'r') as f:
            if json.load(f).get('partial'):
                dates.append(date)
    return dates + [Config.YESTERDAY]


def save(path, output):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"   ✅ Saved: {path}")


# =============================================================================
# MAIN LOGIC
# =============================================================================
def main():
    print("=" * 60)
    print("🏒 NHL GOAL PREDICTOR - FETCH RESULTS")
    print(f"📅 Date: {Config.YESTERDAY}")
    print("=" * 60)

    dates = dates_to_grade()
    if len(dates) > 1:
        print(f"\n🔁 Also re-grading missed/partial days: {dates[:-1]}")

    for date in dates:
        print(f"\n📅 {date}")
        games = get_games(date)
        if games is None:
            # An outage would otherwise pass as a quiet night: fail so the
            # alert fires; the next run re-grades the day.
            print("   ❌ Could not fetch the schedule")
            if date == Config.YESTERDAY:
                return 1
            continue
        output = grade(date, games)
        if output is None:
            continue
        path = f"{Config.RESULTS_DIR}/{date}.json"
        if os.path.exists(path):
            with open(path, 'r') as f:
                before = json.load(f).get('final_games', 0)
            if output.get('final_games', 0) < before:
                print(f"   Keeping the earlier result ({before} games graded)")
                continue
        save(path, output)
        if date == Config.YESTERDAY:
            save(f"{Config.RESULTS_DIR}/latest.json", output)

    print("\n✅ Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
