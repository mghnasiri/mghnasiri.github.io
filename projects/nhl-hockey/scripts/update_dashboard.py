"""
NHL Goal Predictor - Update Stats
=================================
Aggregates historical results into stats.json
Does NOT touch index.html

Author: Mohammad G. Nasiri
"""

import json
import os
import random
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    DATA_DIR = "data"
    RESULTS_DIR = f"{DATA_DIR}/results"
    STATS_FILE = f"{DATA_DIR}/stats.json"
    TIMS_DIR = f"{DATA_DIR}/tims_players"
    PREDICTIONS_DIR = f"{DATA_DIR}/predictions"

print("=" * 60)
print("📊 NHL GOAL PREDICTOR - UPDATE STATS")
print("=" * 60)

# =============================================================================
# LOAD ALL RESULT FILES
# =============================================================================
def load_json(filepath):
    """Safely load a JSON file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return None

print("\n📂 Loading result files...")

results = []

if os.path.exists(Config.RESULTS_DIR):
    for filename in sorted(os.listdir(Config.RESULTS_DIR)):
        # Skip non-date files
        if not filename.endswith('.json'):
            continue
        if filename in ['latest.json', 'stats.json']:
            continue
        
        filepath = f"{Config.RESULTS_DIR}/{filename}"
        data = load_json(filepath)
        
        # Only include days with actual games
        if data and data.get('games_count', 0) > 0:
            results.append(data)
            print(f"   ✅ {filename} ({data.get('games_count')} games)")

print(f"\n✅ Loaded {len(results)} result files with games")

GROUP_WINDOW = 150   # most recent graded Tims groups per model (display)
BOOTSTRAP_RESAMPLES = 2000


def season_start(now=None):
    """Aug 1 of the current season's start year."""
    d = now or datetime.now()
    return f"{d.year if d.month >= 8 else d.year - 1}-08-01"


def _group_picks(model_name, results):
    """(date, hit, baseline) for every real Tims group the model ranked. Its
    pick is its highest-ranked player who is in that day's real group: a
    model can rank a stale pool, and 43 published picks (2026-03..05) were
    players not in the real group. Only days with the real Tims pool and a
    complete grade count; the baseline is the group's own scoring rate (the
    expected hit rate of a random pick)."""
    picks = []
    for result in results:
        if result.get('partial'):
            continue
        date = result.get('date')
        tims = load_json(f"{Config.TIMS_DIR}/{date}.json") or {}
        if tims.get('source') != 'timnhlassist.com' or not tims.get('groups'):
            continue
        pred = load_json(f"{Config.PREDICTIONS_DIR}/{model_name}/{date}.json") or {}
        scorers = {s['player_id'] for s in result.get('all_scorers', []) if s.get('player_id')}
        for gid, ranked in (pred.get('tims_group_rankings') or {}).items():
            members = {p['player_id'] for p in tims['groups'].get(gid, [])
                       if isinstance(p, dict) and p.get('player_id')}
            pick = next((r.get('player_id') for r in ranked if r.get('player_id') in members), None)
            if pick is None:
                continue
            baseline = sum(pid in scorers for pid in members) / len(members)
            picks.append((date, pick in scorers, baseline))
    return picks


def _summary(picks):
    """Hit rate, random baseline and lift, with a 90% CI for the lift from
    resampling days (a fixed seed keeps it stable between runs)."""
    n = len(picks)
    hits = sum(h for _, h, _ in picks)
    base = sum(b for _, _, b in picks) / n
    by_day = {}
    for d, h, b in picks:
        by_day.setdefault(d, []).append(h - b)
    days = sorted(by_day)
    rng = random.Random(0)
    boots = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        rows = [x for d in (rng.choice(days) for _ in days) for x in by_day[d]]
        boots.append(sum(rows) / len(rows) * 100)
    boots.sort()
    return {
        'hits': hits,
        'groups': n,
        'days': len(days),
        'rate': round(hits / n * 100, 1),
        'baseline_rate': round(base * 100, 1),
        'lift': round((hits / n - base) * 100, 1),
        'lift_ci90': [round(boots[int(0.05 * BOOTSTRAP_RESAMPLES)], 1),
                      round(boots[int(0.95 * BOOTSTRAP_RESAMPLES)], 1)],
    }


def group_top1(model_name, results):
    """The metric that matches the Tims game (one pick per group): did the
    model's pick in each real group score, against a random pick from the
    same group? Top-10 overall is mostly group-1 players, where no model
    beats random, so it can't see the gains in groups 2-3. Recent window for
    display, plus this season with its CI (what model status is judged on)."""
    picks = _group_picks(model_name, results)
    if not picks:
        return None
    out = _summary(picks[-GROUP_WINDOW:])
    season = [p for p in picks if p[0] >= season_start()]
    out['season'] = _summary(season) if season else None
    return out


# =============================================================================
# CALCULATE MODEL STATS
# =============================================================================
print("\n📈 Calculating model stats...")

# Find all models
model_names = set()
for result in results:
    for comp in result.get('model_comparisons', []):
        if comp.get('model'):
            model_names.add(comp['model'])

print(f"   Models found: {model_names or 'None'}")

# Calculate stats per model
model_stats = {}

for model_name in model_names:
    total_hits = 0
    total_predictions = 0
    daily_results = []
    
    for result in results:
        for comp in result.get('model_comparisons', []):
            if comp.get('model') == model_name:
                hits = comp.get('hits', 0)
                # Denominator = picks actually graded that day, not a hardcoded
                # 10. Reading len(top10_picks) heals historical files (which
                # stored total_predictions: 10) without re-fetching.
                picks = comp.get('top10_picks', [])
                total = len(picks) if picks else comp.get('total_predictions', 0)

                total_hits += hits
                total_predictions += total
                
                daily_results.append({
                    'date': result.get('date'),
                    'hits': hits,
                    'total': total,
                    'hit_rate': round(hits / total * 100, 1) if total > 0 else 0
                })
    
    # Calculate aggregates
    hit_rate = (total_hits / total_predictions * 100) if total_predictions > 0 else 0
    avg_hits = (total_hits / len(daily_results)) if daily_results else 0
    
    # Last 7 days stats
    last_7 = daily_results[-7:] if len(daily_results) >= 7 else daily_results
    last_7_hits = sum(d['hits'] for d in last_7)
    last_7_total = sum(d['total'] for d in last_7)
    last_7_rate = (last_7_hits / last_7_total * 100) if last_7_total > 0 else 0
    
    model_stats[model_name] = {
        'total_days': len(daily_results),
        'total_hits': total_hits,
        'total_predictions': total_predictions,
        'hit_rate': round(hit_rate, 1),
        'avg_hits_per_day': round(avg_hits, 2),
        'last_7_days': {
            'days': len(last_7),
            'hits': last_7_hits,
            'total': last_7_total,
            'hit_rate': round(last_7_rate, 1)
        },
        'daily_results': daily_results[-30:],  # Last 30 days
        'group_top1': group_top1(model_name, results),
    }

    print(f"\n   📊 {model_name}:")
    print(f"      Days tracked: {len(daily_results)}")
    print(f"      Total: {total_hits}/{total_predictions} ({round(hit_rate, 1)}%)")
    print(f"      Last 7: {last_7_hits}/{last_7_total} ({round(last_7_rate, 1)}%)")
    g = model_stats[model_name]['group_top1']
    if g:
        print(f"      Group top-1: {g['hits']}/{g['groups']} ({g['rate']}%) vs random {g['baseline_rate']}% "
              f"over {g['days']} real-pool days")

# =============================================================================
# SAVE STATS
# =============================================================================
print("\n💾 Saving stats...")

output = {
    "generated_at": datetime.now().isoformat(),
    "total_days_tracked": len(results),
    "models": model_stats
}

# Ensure directory exists
os.makedirs(Config.DATA_DIR, exist_ok=True)

with open(Config.STATS_FILE, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"✅ Saved: {Config.STATS_FILE}")

print("\n" + "=" * 60)
print("✅ Stats update complete!")
print("=" * 60)
