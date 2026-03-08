"""
NHL Goal Predictor - Telegram Bot Notifications
================================================
Sends daily predictions and results via Telegram Bot API.

Setup:
  1. Create a bot via @BotFather on Telegram -> get BOT_TOKEN
  2. Send /start to your bot, then get your CHAT_ID via
     https://api.telegram.org/bot<TOKEN>/getUpdates
  3. Set as environment variables or GitHub Secrets:
     - TELEGRAM_BOT_TOKEN
     - TELEGRAM_CHAT_ID

Author: Mohammad G. Nasiri
"""

import requests
import json
import os
import sys
from datetime import datetime, timedelta

# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
    CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')

    DATA_DIR = "data"
    MC_PREDICTIONS_DIR = f"{DATA_DIR}/predictions/monte_carlo"
    NN_PREDICTIONS_DIR = f"{DATA_DIR}/predictions/neural_network"
    RESULTS_DIR = f"{DATA_DIR}/results"
    STATS_FILE = f"{DATA_DIR}/stats.json"
    TIMS_DIR = f"{DATA_DIR}/tims_players"

    TODAY = datetime.now().strftime("%Y-%m-%d")
    YESTERDAY = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"
    DASHBOARD_URL = "https://mghnasiri.github.io/projects/nhl-hockey/"


# =============================================================================
# TELEGRAM API
# =============================================================================
def send_message(text, parse_mode='HTML'):
    """Send a message via Telegram Bot API"""
    if not Config.BOT_TOKEN or not Config.CHAT_ID:
        print("⚠️ Telegram credentials not set. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
        print(f"Message would be:\n{text}")
        return False

    url = f"{Config.TELEGRAM_API}/sendMessage"
    payload = {
        'chat_id': Config.CHAT_ID,
        'text': text,
        'parse_mode': parse_mode,
        'disable_web_page_preview': True,
    }

    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 200:
            print("✅ Telegram message sent")
            return True
        else:
            print(f"❌ Telegram API error: {resp.status_code} - {resp.text}")
            return False
    except Exception as e:
        print(f"❌ Failed to send Telegram message: {e}")
        return False


def load_json(filepath):
    """Safely load JSON file"""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return None


# =============================================================================
# MESSAGE FORMATTERS
# =============================================================================
def format_predictions_message():
    """Format today's predictions as a Telegram message"""
    # Load Monte Carlo predictions
    mc_data = load_json(f"{Config.MC_PREDICTIONS_DIR}/latest.json")
    nn_data = load_json(f"{Config.NN_PREDICTIONS_DIR}/latest.json")
    tims_data = load_json(f"{Config.TIMS_DIR}/latest.json")

    if not mc_data:
        return "⚠️ No Monte Carlo predictions available for today."

    date = mc_data.get('date', Config.TODAY)
    games_count = mc_data.get('games_count', 0)
    predictions = mc_data.get('predictions', [])
    tims_groups = mc_data.get('tims_group_rankings', {})

    lines = []
    lines.append(f"🏒 <b>NHL Goal Predictor</b>")
    lines.append(f"📅 {date} | {games_count} games")
    lines.append("")

    # Games list
    for g in mc_data.get('games', []):
        lines.append(f"  ⚔️ {g['away_team']} @ {g['home_team']}")
    lines.append("")

    # Tim Hortons groups (if available)
    if tims_groups:
        lines.append("🍩 <b>Tim Hortons Challenge Picks</b>")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━")

        for gid in sorted(tims_groups.keys()):
            group = tims_groups[gid]
            lines.append(f"\n📦 <b>Group {gid}</b>")

            for p in group[:5]:
                prob = p['goal_probability'] * 100
                bar = "█" * int(prob / 5) + "░" * (10 - int(prob / 5))
                medal = "🥇" if p['rank_in_group'] == 1 else ("🥈" if p['rank_in_group'] == 2 else "  ")
                lines.append(f"  {medal} {p['name']} ({p['team']})")
                lines.append(f"     {bar} {prob:.1f}%")

        lines.append("")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━")

        # Best pick summary
        best_picks = []
        for gid in sorted(tims_groups.keys()):
            if tims_groups[gid]:
                best = tims_groups[gid][0]
                best_picks.append(f"G{gid}: {best['name']} ({best['goal_probability']*100:.1f}%)")
        lines.append(f"\n🎯 <b>Recommended:</b> {' | '.join(best_picks)}")

    else:
        # Non-Tims mode: show top 10
        lines.append("🎯 <b>Top 10 Predictions (Monte Carlo v2)</b>")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━")

        for p in predictions[:10]:
            prob = p['goal_probability'] * 100
            hot = "🔥" if p.get('is_hot') else ""
            opp_ga = p.get('opp_ga_per_game', 0)
            lines.append(f"  {p['rank']:>2}. {p['name']} ({p['team']}) — <b>{prob:.1f}%</b> {hot}")
            lines.append(f"     vs {p.get('opponent', '?')} | OppGA: {opp_ga:.1f}")

    lines.append("")

    # Model comparison (if both available)
    if nn_data:
        nn_preds = nn_data.get('predictions', [])[:5]
        lines.append("\n📊 <b>Model Comparison (Top 5)</b>")
        lines.append("┌─────────────────────────────────┐")
        lines.append("│ MC v2          │ NN v1          │")
        lines.append("├─────────────────────────────────┤")
        for i in range(min(5, len(predictions), len(nn_preds))):
            mc_p = predictions[i]
            nn_p = nn_preds[i]
            mc_str = f"{mc_p['name'][:12]} {mc_p['goal_probability']*100:.0f}%"
            nn_str = f"{nn_p['name'][:12]} {nn_p['goal_probability']*100:.0f}%"
            lines.append(f"│ {mc_str:<15}│ {nn_str:<15}│")
        lines.append("└─────────────────────────────────┘")

    lines.append(f"\n🔗 <a href='{Config.DASHBOARD_URL}'>Full Dashboard</a>")
    lines.append(f"🎲 10,000 Monte Carlo simulations per game")

    return "\n".join(lines)


def format_results_message():
    """Format yesterday's results as a Telegram message"""
    results = load_json(f"{Config.RESULTS_DIR}/latest.json")
    stats = load_json(Config.STATS_FILE)

    if not results:
        return "⚠️ No results available."

    date = results.get('date', Config.YESTERDAY)
    games_count = results.get('games_count', 0)
    comparisons = results.get('model_comparisons', [])

    lines = []
    lines.append(f"📊 <b>NHL Predictor Results</b>")
    lines.append(f"📅 {date} | {games_count} games")
    lines.append("")

    if not comparisons:
        lines.append("ℹ️ No model comparisons available for this day")
        return "\n".join(lines)

    # Show each model's results
    for comp in comparisons:
        model = comp.get('model_display_name', comp.get('model', ''))
        hits = comp.get('hits', 0)
        total = comp.get('total_predictions', 10)
        rate = comp.get('hit_rate', 0)

        # Visual hit rate
        emoji = "🟢" if rate >= 40 else ("🟡" if rate >= 25 else "🔴")
        lines.append(f"{emoji} <b>{model}</b>: {hits}/{total} ({rate:.0f}%)")

        # Show individual picks
        for pick in comp.get('top10_picks', [])[:10]:
            icon = "✅" if pick['scored'] else "❌"
            lines.append(f"  {icon} {pick['name']} ({pick['team']}) — {pick['probability']*100:.0f}%")
        lines.append("")

    # Overall stats
    if stats and stats.get('models'):
        lines.append("📈 <b>Season Performance</b>")

        for model_name, model_data in stats['models'].items():
            overall_rate = model_data.get('hit_rate', 0)
            total_days = model_data.get('total_days', 0)
            total_hits = model_data.get('total_hits', 0)
            l7 = model_data.get('last_7_days', {})
            l7_rate = l7.get('hit_rate', 0)

            lines.append(f"  {model_name}: {overall_rate:.1f}% ({total_hits} hits / {total_days} days)")
            lines.append(f"    Last 7 days: {l7_rate:.1f}%")

    lines.append(f"\n🔗 <a href='{Config.DASHBOARD_URL}'>Full Dashboard</a>")

    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================
def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "predictions"

    print("=" * 60)
    print(f"📱 TELEGRAM NOTIFIER - {mode.upper()}")
    print(f"📅 {Config.TODAY}")
    print("=" * 60)

    if mode == "predictions":
        msg = format_predictions_message()
    elif mode == "results":
        msg = format_results_message()
    elif mode == "both":
        pred_msg = format_predictions_message()
        send_message(pred_msg)

        results_msg = format_results_message()
        send_message(results_msg)
        return
    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python telegram_notify.py [predictions|results|both]")
        return

    send_message(msg)


if __name__ == "__main__":
    main()
