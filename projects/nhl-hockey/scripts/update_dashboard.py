"""
NHL Goal Predictor - Update Stats
=================================
Aggregates historical results into stats.json
Does NOT touch index.html

Author: Mohammad G. Nasiri
"""

import json
import os
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

GROUP_WINDOW = 150   # most recent graded Tims groups per model


def group_top1(model_name, results):
    """The metric that matches the Tims game (one pick per group): in each
    group, did the model's top-ranked player score? Graded only on days with
    the real Tims pool, against the same-day random baseline (share of the
    group's players who scored). Top-10 overall is dominated by group 1,
    where no model beats random, so it can't see the gains in groups 2-3."""
    picks = []   # (date, hit, baseline)
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
            members = [p.get('player_id') for p in tims['groups'].get(gid, [])
                       if isinstance(p, dict) and p.get('player_id')]
            if not ranked or not members:
                continue
            baseline = sum(pid in scorers for pid in members) / len(members)
            picks.append((date, ranked[0].get('player_id') in scorers, baseline))
    picks = picks[-GROUP_WINDOW:]
    if not picks:
        return None
    n = len(picks)
    hits = sum(h for _, h, _ in picks)
    base = sum(b for _, _, b in picks) / n
    return {
        'hits': hits,
        'groups': n,
        'days': len({d for d, _, _ in picks}),
        'rate': round(hits / n * 100, 1),
        'baseline_rate': round(base * 100, 1),
        'lift': round((hits / n - base) * 100, 1),
    }


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
