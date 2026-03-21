"""
NHL Goal Predictor - Telegram Bot Notifications
================================================
Sends a single daily morning message combining:
  - Yesterday's results (if available)
  - Today's predictions + Tim Hortons picks

Gracefully handles missing data — never errors out.

Setup:
  1. Create a bot via @BotFather on Telegram -> get BOT_TOKEN
  2. Get your CHAT_ID via @userinfobot
  3. Set as GitHub Secrets:
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
    XG_PREDICTIONS_DIR = f"{DATA_DIR}/predictions/xg_v3"
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
        print("  Telegram credentials not set. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
        print(f"Message preview:\n{text}")
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
            print("  Telegram message sent successfully")
            return True
        else:
            print(f"  Telegram API error: {resp.status_code} - {resp.text}")
            return False
    except Exception as e:
        print(f"  Failed to send Telegram message: {e}")
        return False


def load_json(filepath):
    """Safely load JSON file — returns None on any failure"""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return None


# =============================================================================
# MESSAGE BUILDER
# =============================================================================
def build_daily_message():
    """
    Build the single daily message combining results + predictions.
    Gracefully skips any section where data is unavailable.
    """
    lines = []

    # ── Header ──
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append("  🏒  <b>NHL GOAL PREDICTOR</b>")
    lines.append(f"  📅  {Config.TODAY}")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # ── Section 1: Yesterday's Results (optional) ──
    results_added = add_results_section(lines)

    # ── Section 2: Today's Predictions ──
    predictions_added = add_predictions_section(lines)

    if not predictions_added:
        lines.append("")
        lines.append("📭  No predictions available for today.")
        lines.append("    (No NHL games scheduled or data not ready)")

    # ── Footer ──
    lines.append("")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"🔗  <a href='{Config.DASHBOARD_URL}'>Full Dashboard</a>")
    lines.append("🧠  3 Models: xG XGBoost · Monte Carlo · Neural Net")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━")

    return "\n".join(lines)


def add_results_section(lines):
    """
    Add yesterday's results to message.
    Returns True if results were found, False otherwise.
    Never raises errors — silently skips if data unavailable.
    """
    results = load_json(f"{Config.RESULTS_DIR}/latest.json")
    stats = load_json(Config.STATS_FILE)

    # Try to find yesterday's results
    if not results:
        results = load_json(f"{Config.RESULTS_DIR}/{Config.YESTERDAY}.json")

    if not results:
        # No results data at all — that's fine, just skip
        return False

    date = results.get('date', Config.YESTERDAY)
    comparisons = results.get('model_comparisons', [])

    if not comparisons:
        return False

    lines.append("")
    lines.append("📊  <b>YESTERDAY'S RESULTS</b>")
    lines.append(f"     {date}")
    lines.append("┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄")

    for comp in comparisons:
        model = comp.get('model_display_name', comp.get('model', 'Unknown'))
        hits = comp.get('hits', 0)
        total = comp.get('total_predictions', 10)
        rate = comp.get('hit_rate', 0)

        # Score emoji
        if rate >= 40:
            emoji = "🟢"
        elif rate >= 25:
            emoji = "🟡"
        else:
            emoji = "🔴"

        lines.append(f"  {emoji}  <b>{model}</b>: {hits}/{total} ({rate:.0f}%)")

        # Show individual picks (top 5 only to keep message compact)
        picks = comp.get('top10_picks', [])[:5]
        for pick in picks:
            icon = "✅" if pick.get('scored') else "❌"
            prob = pick.get('probability', 0) * 100
            lines.append(f"      {icon} {pick.get('name', '?')} ({pick.get('team', '?')}) {prob:.0f}%")

        if len(comp.get('top10_picks', [])) > 5:
            remaining = len(comp['top10_picks']) - 5
            lines.append(f"      ... +{remaining} more on dashboard")

    # Season performance summary
    if stats and stats.get('models'):
        lines.append("")
        lines.append("  📈  <b>Season Stats</b>")
        for model_name, model_data in stats['models'].items():
            overall_rate = model_data.get('hit_rate', 0)
            total_days = model_data.get('total_days', 0)
            total_hits = model_data.get('total_hits', 0)
            short_name = model_name.replace('_', ' ').title()
            lines.append(f"      {short_name}: {overall_rate:.1f}% ({total_hits} hits / {total_days}d)")

    return True


def add_predictions_section(lines):
    """
    Add today's predictions to message.
    Prefers xG v3 data, falls back to MC v2, then NN v1.
    Returns True if predictions were found, False otherwise.
    Never raises errors — silently skips if data unavailable.
    """
    # Load predictions — prefer today's dated file, fall back to latest.json
    xg_data = load_json(f"{Config.XG_PREDICTIONS_DIR}/{Config.TODAY}.json") or load_json(f"{Config.XG_PREDICTIONS_DIR}/latest.json")
    mc_data = load_json(f"{Config.MC_PREDICTIONS_DIR}/{Config.TODAY}.json") or load_json(f"{Config.MC_PREDICTIONS_DIR}/latest.json")
    nn_data = load_json(f"{Config.NN_PREDICTIONS_DIR}/{Config.TODAY}.json") or load_json(f"{Config.NN_PREDICTIONS_DIR}/latest.json")

    # Only use data that matches today's date (reject stale predictions)
    if xg_data and xg_data.get('date') != Config.TODAY:
        print(f"  Warning: xG v3 latest is {xg_data.get('date')}, not today ({Config.TODAY}) — skipping")
        xg_data = None
    if mc_data and mc_data.get('date') != Config.TODAY:
        print(f"  Warning: MC v2 latest is {mc_data.get('date')}, not today ({Config.TODAY}) — skipping")
        mc_data = None
    if nn_data and nn_data.get('date') != Config.TODAY:
        print(f"  Warning: NN v1 latest is {nn_data.get('date')}, not today ({Config.TODAY}) — skipping")
        nn_data = None

    # Use best available model as primary (xG v3 > MC v2 > NN v1)
    primary = xg_data or mc_data or nn_data
    if not primary:
        return False

    date = primary.get('date', Config.TODAY)
    games_count = primary.get('games_count', 0)
    predictions = primary.get('predictions', [])
    tims_groups = primary.get('tims_group_rankings', {})
    games = primary.get('games', [])

    primary_name = "xG v3" if xg_data else ("MC v2" if mc_data else "NN v1")

    if not predictions:
        return False

    lines.append("")
    lines.append("🎯  <b>TODAY'S PREDICTIONS</b>")
    lines.append(f"     {date}  •  {games_count} games  •  {primary_name}")
    lines.append("┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄")

    # Games list
    if games:
        for g in games:
            away = g.get('away_team', '?')
            home = g.get('home_team', '?')
            lines.append(f"  ⚔️  {away}  @  {home}")
        lines.append("")

    # Tim Hortons groups
    tims_source = primary.get('tims_source')
    if tims_groups:
        lines.append("🍩  <b>TIM HORTONS PICKS</b>")
        if tims_source == 'cached_pool':
            lines.append("     ⚠️ <i>Source unavailable — using cached roster</i>")
            lines.append("     <i>Players may differ from Tim Hortons app</i>")
        lines.append("")

        best_picks = []

        for gid in sorted(tims_groups.keys()):
            group = tims_groups[gid]
            if not group:
                continue

            lines.append(f"  📦  <b>Group {gid}</b>")

            for i, p in enumerate(group[:5]):
                prob = p.get('goal_probability', 0) * 100
                name = p.get('name', '?')
                team = p.get('team', '?')

                # Visual bar (10 chars wide)
                filled = min(10, int(prob / 7))
                bar = "█" * filled + "░" * (10 - filled)

                if i == 0:
                    lines.append(f"    ⭐  <b>{name}</b> ({team})")
                    lines.append(f"         {bar}  <b>{prob:.1f}%</b>")
                    best_picks.append(f"G{gid}: {name} ({prob:.1f}%)")
                else:
                    rank = i + 1
                    lines.append(f"    {rank}.   {name} ({team})")
                    lines.append(f"         {bar}  {prob:.1f}%")
            lines.append("")

        # Recommended picks summary
        if best_picks:
            lines.append("  🏆  <b>RECOMMENDED PICKS</b>")
            for pick in best_picks:
                lines.append(f"      ▸ {pick}")

    else:
        # No Tim Hortons data — show top 10 overall
        lines.append("  <b>Top 10 Goal Scorers</b>")
        lines.append("")

        for p in predictions[:10]:
            prob = p.get('goal_probability', 0) * 100
            name = p.get('name', '?')
            team = p.get('team', '?')
            rank = p.get('rank', 0)
            hot = " 🔥" if p.get('is_hot') else ""

            filled = min(10, int(prob / 7))
            bar = "█" * filled + "░" * (10 - filled)

            lines.append(f"  {rank:>2}.  {name} ({team}){hot}")
            lines.append(f"       {bar}  {prob:.1f}%")

    # 3-model comparison (compact — show top 3 from each available model)
    model_tops = []
    if xg_data and xg_data.get('predictions'):
        model_tops.append(("xG v3", xg_data['predictions'][:3]))
    if mc_data and mc_data.get('predictions'):
        model_tops.append(("MC v2", mc_data['predictions'][:3]))
    if nn_data and nn_data.get('predictions'):
        model_tops.append(("NN v1", nn_data['predictions'][:3]))

    if len(model_tops) >= 2:
        lines.append("")
        names = " vs ".join(m[0] for m in model_tops)
        lines.append(f"  📊  <b>{names}</b>")
        for rank in range(3):
            parts = []
            for model_name, preds in model_tops:
                if rank < len(preds):
                    p = preds[rank]
                    name = p.get('name', '?')[:12]
                    prob = p.get('goal_probability', 0) * 100
                    parts.append(f"{name} {prob:.0f}%")
            if parts:
                lines.append(f"   {rank+1}.  {'  │  '.join(parts)}")

    return True


# =============================================================================
# MAIN
# =============================================================================
def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "daily"

    print("=" * 50)
    print(f"  TELEGRAM NOTIFIER — {mode.upper()}")
    print(f"  {Config.TODAY}")
    print("=" * 50)

    if mode in ("daily", "predictions", "both"):
        # Single combined message: results + predictions
        msg = build_daily_message()
        send_message(msg)

    elif mode == "test":
        # Send a test message to verify connection
        test_msg = (
            "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "  🏒  <b>NHL GOAL PREDICTOR</b>\n"
            "  🧪  Connection Test\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "\n"
            "✅  Telegram bot is connected!\n"
            f"📅  {Config.TODAY}\n"
            "\n"
            "You will receive daily predictions\n"
            "every morning at 10:00 AM Toronto time.\n"
            "\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🔗  <a href='{Config.DASHBOARD_URL}'>Full Dashboard</a>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        send_message(test_msg)

    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python telegram_notify.py [daily|test]")


if __name__ == "__main__":
    main()
