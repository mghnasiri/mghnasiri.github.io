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
                total = comp.get('total_predictions', 10)
                
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
        'daily_results': daily_results[-30:]  # Last 30 days
    }
    
    print(f"\n   📊 {model_name}:")
    print(f"      Days tracked: {len(daily_results)}")
    print(f"      Total: {total_hits}/{total_predictions} ({round(hit_rate, 1)}%)")
    print(f"      Last 7: {last_7_hits}/{last_7_total} ({round(last_7_rate, 1)}%)")

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
