"""
NHL Goal Predictor - Update Dashboard Stats
============================================
Updates stats JSON file for the dashboard
Does NOT overwrite index.html anymore

Author: Mohammad
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    DATA_DIR = "data"
    PREDICTIONS_DIR = f"{DATA_DIR}/predictions"
    RESULTS_DIR = f"{DATA_DIR}/results"
    # Output to stats.json instead of index.html
    STATS_FILE = f"{DATA_DIR}/stats.json"

print("=" * 70)
print("🎨 NHL GOAL PREDICTOR - UPDATE STATS")
print("=" * 70)

# =============================================================================
# 1. LOAD RESULT FILES
# =============================================================================
def load_json(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return None

# Load all result files for historical stats
historical_results = []
if os.path.exists(Config.RESULTS_DIR):
    for filename in sorted(os.listdir(Config.RESULTS_DIR)):
        if filename.endswith('.json') and filename != 'latest.json' and filename != 'stats.json':
            data = load_json(f"{Config.RESULTS_DIR}/{filename}")
            if data and data.get('games_count', 0) > 0:
                historical_results.append(data)

print(f"✅ Loaded {len(historical_results)} historical result files")

# =============================================================================
# 2. CALCULATE STATS FOR EACH MODEL
# =============================================================================
def calculate_model_stats(model_name, historical_results):
    total_hits = 0
    total_predictions = 0
    daily_results = []
    
    for result in historical_results:
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
    
    hit_rate = total_hits / total_predictions * 100 if total_predictions > 0 else 0
    avg_hits = total_hits / len(daily_results) if daily_results else 0
    
    return {
        'total_days': len(daily_results),
        'total_hits': total_hits,
        'total_predictions': total_predictions,
        'hit_rate': round(hit_rate, 1),
        'avg_hits_per_day': round(avg_hits, 1),
        'last_7_days': daily_results[-7:] if len(daily_results) >= 7 else daily_results,
        'all_daily_results': daily_results
    }

# Get all model names
model_names = set()
for result in historical_results:
    for comp in result.get('model_comparisons', []):
        model_names.add(comp.get('model'))

print(f"✅ Found models: {model_names}")

# Calculate stats for each model
model_stats = {}
for model_name in model_names:
    if model_name:
        model_stats[model_name] = calculate_model_stats(model_name, historical_results)
        print(f"   {model_name}: {model_stats[model_name]['total_hits']}/{model_stats[model_name]['total_predictions']} ({model_stats[model_name]['hit_rate']}%)")

# =============================================================================
# 3. SAVE STATS JSON
# =============================================================================
print("\n💾 Saving stats...")

output = {
    "generated_at": datetime.now().isoformat(),
    "total_days_tracked": len(historical_results),
    "models": model_stats
}

# Save stats file
os.makedirs(Config.DATA_DIR, exist_ok=True)
with open(Config.STATS_FILE, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"✅ Saved: {Config.STATS_FILE}")
print("\n🎨 Stats update complete!")
print("ℹ️  Note: index.html is no longer overwritten by this script")
