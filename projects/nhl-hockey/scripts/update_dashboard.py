"""
NHL Goal Predictor - Update Dashboard
=====================================
Generates HTML dashboard from predictions and results

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
    OUTPUT_FILE = "index.html"

print("=" * 70)
print("🎨 NHL GOAL PREDICTOR - UPDATE DASHBOARD")
print("=" * 70)

# =============================================================================
# 1. LOAD LATEST DATA
# =============================================================================
def load_json(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return None

# Load latest predictions from all models
predictions_by_model = {}
if os.path.exists(Config.PREDICTIONS_DIR):
    for model_name in os.listdir(Config.PREDICTIONS_DIR):
        model_dir = f"{Config.PREDICTIONS_DIR}/{model_name}"
        if os.path.isdir(model_dir):
            latest_file = f"{model_dir}/latest.json"
            if os.path.exists(latest_file):
                predictions_by_model[model_name] = load_json(latest_file)
                print(f"✅ Loaded predictions: {model_name}")

# Load latest results
latest_results = load_json(f"{Config.RESULTS_DIR}/latest.json")
if latest_results:
    print(f"✅ Loaded results for: {latest_results.get('date', 'unknown')}")

# Load historical results for stats
historical_results = []
if os.path.exists(Config.RESULTS_DIR):
    for filename in sorted(os.listdir(Config.RESULTS_DIR)):
        if filename.endswith('.json') and filename != 'latest.json':
            data = load_json(f"{Config.RESULTS_DIR}/{filename}")
            if data:
                historical_results.append(data)

print(f"✅ Loaded {len(historical_results)} historical results")

# =============================================================================
# 2. CALCULATE STATISTICS
# =============================================================================
def calculate_model_stats(model_name, historical_results):
    total_hits = 0
    total_predictions = 0
    daily_results = []
    
    for result in historical_results:
        for comp in result.get('model_comparisons', []):
            if comp.get('model') == model_name:
                hits = comp.get('hits', 0)
                total_hits += hits
                total_predictions += 3
                daily_results.append({
                    'date': result.get('date'),
                    'hits': hits
                })
    
    hit_rate = total_hits / total_predictions * 100 if total_predictions > 0 else 0
    
    return {
        'total_days': len(daily_results),
        'total_hits': total_hits,
        'total_predictions': total_predictions,
        'hit_rate': round(hit_rate, 1),
        'daily_results': daily_results[-10:]  # Last 10 days
    }

model_stats = {}
for model_name in predictions_by_model.keys():
    model_stats[model_name] = calculate_model_stats(model_name, historical_results)

# =============================================================================
# 3. GENERATE HTML
# =============================================================================
print("\n📝 Generating HTML...")

# Get first model's predictions for display
primary_model = list(predictions_by_model.keys())[0] if predictions_by_model else None
primary_predictions = predictions_by_model.get(primary_model, {}) if primary_model else {}

html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🏒 NHL Goal Predictor - Tim Hortons Challenge</title>
    <style>
        :root {{
            --primary: #c8102e;
            --secondary: #041e42;
            --accent: #ffd700;
            --bg: #f5f5f5;
            --card-bg: #ffffff;
            --text: #333333;
            --success: #28a745;
            --danger: #dc3545;
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        header {{
            background: linear-gradient(135deg, var(--secondary) 0%, var(--primary) 100%);
            color: white;
            padding: 30px 20px;
            text-align: center;
            margin-bottom: 30px;
            border-radius: 10px;
        }}
        
        header h1 {{
            font-size: 2.5rem;
            margin-bottom: 10px;
        }}
        
        header .date {{
            font-size: 1.2rem;
            opacity: 0.9;
        }}
        
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .card {{
            background: var(--card-bg);
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .card h2 {{
            color: var(--secondary);
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid var(--primary);
        }}
        
        .games-list {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }}
        
        .game-badge {{
            background: var(--secondary);
            color: white;
            padding: 8px 15px;
            border-radius: 20px;
            font-weight: 600;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        
        th {{
            background: var(--secondary);
            color: white;
            font-weight: 600;
        }}
        
        tr:hover {{
            background: #f8f9fa;
        }}
        
        .prob-bar {{
            width: 100%;
            height: 8px;
            background: #eee;
            border-radius: 4px;
            overflow: hidden;
        }}
        
        .prob-fill {{
            height: 100%;
            background: linear-gradient(90deg, var(--primary) 0%, var(--accent) 100%);
            border-radius: 4px;
        }}
        
        .hot {{
            color: #ff6b00;
            font-weight: bold;
        }}
        
        .result-hit {{
            color: var(--success);
            font-weight: bold;
        }}
        
        .result-miss {{
            color: var(--danger);
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin-bottom: 20px;
        }}
        
        .stat-box {{
            text-align: center;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        
        .stat-value {{
            font-size: 2rem;
            font-weight: bold;
            color: var(--primary);
        }}
        
        .stat-label {{
            font-size: 0.9rem;
            color: #666;
        }}
        
        .model-tabs {{
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }}
        
        .model-tab {{
            padding: 10px 20px;
            background: var(--secondary);
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-weight: 600;
        }}
        
        .model-tab.active {{
            background: var(--primary);
        }}
        
        footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9rem;
        }}
        
        @media (max-width: 768px) {{
            header h1 {{
                font-size: 1.8rem;
            }}
            
            .grid {{
                grid-template-columns: 1fr;
            }}
            
            table {{
                font-size: 0.9rem;
            }}
            
            th, td {{
                padding: 8px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🏒 NHL Goal Predictor</h1>
            <p class="date">Tim Hortons Hockey Challenge</p>
            <p class="date">📅 {primary_predictions.get('date', datetime.now().strftime('%Y-%m-%d'))}</p>
        </header>
'''

# Add yesterday's results section if available
if latest_results and latest_results.get('model_comparisons'):
    html += '''
        <div class="card" style="margin-bottom: 30px; border-left: 4px solid var(--success);">
            <h2>📊 Yesterday's Results</h2>
    '''
    
    for comp in latest_results.get('model_comparisons', []):
        html += f'''
            <h3 style="margin: 15px 0 10px 0;">{comp.get('model_display_name', comp.get('model'))}</h3>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Player</th>
                    <th>Team</th>
                    <th>Probability</th>
                    <th>Result</th>
                </tr>
        '''
        
        for pick in comp.get('top3_picks', []):
            result_class = 'result-hit' if pick['scored'] else 'result-miss'
            result_text = '✅ GOAL!' if pick['scored'] else '❌ Miss'
            html += f'''
                <tr>
                    <td>{pick['rank']}</td>
                    <td>{pick['name']}</td>
                    <td>{pick['team']}</td>
                    <td>{pick['probability']*100:.1f}%</td>
                    <td class="{result_class}">{result_text}</td>
                </tr>
            '''
        
        html += f'''
            </table>
            <p style="margin-top: 10px; font-size: 1.2rem;">
                <strong>Score: {comp.get('hits', 0)}/3</strong>
            </p>
        '''
    
    # Show all scorers
    if latest_results.get('all_scorers'):
        html += '''
            <h3 style="margin: 20px 0 10px 0;">⚽ All Goal Scorers</h3>
            <div class="games-list">
        '''
        for scorer in latest_results['all_scorers']:
            goals = f" ({scorer['goals']})" if scorer.get('goals', 1) > 1 else ""
            html += f'<span class="game-badge">{scorer["player_name"]} ({scorer["team"]}){goals}</span>'
        html += '</div>'
    
    html += '</div>'

# Today's games
if primary_predictions.get('games'):
    html += '''
        <div class="card" style="margin-bottom: 30px;">
            <h2>🏒 Today's Games</h2>
            <div class="games-list">
    '''
    for game in primary_predictions['games']:
        html += f'<span class="game-badge">{game["away_team"]} @ {game["home_team"]}</span>'
    html += '</div></div>'

# Model statistics
if model_stats:
    html += '<div class="grid">'
    
    for model_name, stats in model_stats.items():
        display_name = predictions_by_model.get(model_name, {}).get('model_display_name', model_name)
        html += f'''
        <div class="card">
            <h2>📈 {display_name} Stats</h2>
            <div class="stats-grid">
                <div class="stat-box">
                    <div class="stat-value">{stats['hit_rate']}%</div>
                    <div class="stat-label">Hit Rate</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{stats['total_hits']}</div>
                    <div class="stat-label">Total Hits</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{stats['total_days']}</div>
                    <div class="stat-label">Days Tracked</div>
                </div>
            </div>
        </div>
        '''
    
    html += '</div>'

# Today's predictions
if primary_predictions.get('predictions'):
    html += '''
        <div class="card">
            <h2>🎯 Today's Predictions</h2>
            <table>
                <tr>
                    <th>#</th>
                    <th>Player</th>
                    <th>Team</th>
                    <th>Pos</th>
                    <th>Probability</th>
                    <th>Season Goals</th>
                    <th>Last 5</th>
                    <th>Matchup</th>
                </tr>
    '''
    
    for player in primary_predictions['predictions'][:50]:
        hot_indicator = ' <span class="hot">🔥</span>' if player.get('is_hot') else ''
        prob_width = player.get('goal_probability', 0) * 100 / 0.6 * 100  # Scale to max
        
        html += f'''
            <tr>
                <td>{player.get('rank', '')}</td>
                <td>{player.get('name', '')}{hot_indicator}</td>
                <td>{player.get('team', '')}</td>
                <td>{player.get('position', '')}</td>
                <td>
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span>{player.get('goal_probability', 0)*100:.1f}%</span>
                        <div class="prob-bar" style="width: 60px;">
                            <div class="prob-fill" style="width: {prob_width}%;"></div>
                        </div>
                    </div>
                </td>
                <td>{player.get('season_goals', 0)}</td>
                <td>{player.get('last5_goals', 0)}</td>
                <td>{player.get('matchup', '')}</td>
            </tr>
        '''
    
    html += '</table></div>'

# Footer
html += f'''
        <footer>
            <p>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}</p>
            <p>🤖 Automated predictions using Machine Learning</p>
            <p>⚠️ For entertainment purposes only</p>
        </footer>
    </div>
</body>
</html>
'''

# Save HTML
with open(Config.OUTPUT_FILE, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"✅ Saved: {Config.OUTPUT_FILE}")
print("\n🎨 Dashboard update complete!")
