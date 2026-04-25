"""
Push the health report to Telegram if any model failed validation.

Reads data/health_report.json (written by validate_predictions.py), composes
a short summary of failures, and posts via the existing TELEGRAM_BOT_TOKEN /
TELEGRAM_CHAT_ID secrets. Used by the daily health-check workflow.

Silent if all models passed (we only want noise when there's a real problem).
"""

import json
import os
import sys
import urllib.parse
import urllib.request

REPORT = "data/health_report.json"


def main():
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not (token and chat):
        print("  No Telegram credentials in env — skipping alert.")
        return 0
    if not os.path.exists(REPORT):
        print(f"  No report at {REPORT} — nothing to alert about.")
        return 0
    with open(REPORT, "r", encoding="utf-8") as f:
        report = json.load(f)

    fails = [
        (name, m) for name, m in report.get("models", {}).items()
        if m.get("status") == "fail"
    ]
    if not fails:
        print("  All models passed — no alert needed.")
        return 0

    today = report.get("today", "?")
    lines = [
        f"NHL HEALTH CHECK — {today}",
        f"{len(fails)} model(s) failing validation:",
        "",
    ]
    for name, meta in fails:
        failed_checks = [
            k for k, v in meta.get("checks", {}).items() if v[0] == "fail"
        ]
        msg_pieces = [
            v[1] for k, v in meta.get("checks", {}).items() if v[0] == "fail"
        ]
        lines.append(f"• {name}: {', '.join(failed_checks)}")
        for piece in msg_pieces[:2]:  # cap detail per model
            lines.append(f"    {piece}")
    lines.append("")
    lines.append("Full report committed to data/health_report.json")

    msg = "\n".join(lines)
    print(f"  Sending alert ({len(msg)} chars):")
    print("\n".join("    " + line for line in lines))

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode({"chat_id": chat, "text": msg}).encode()
    try:
        with urllib.request.urlopen(url, data=data, timeout=10) as resp:
            resp.read()
    except Exception as e:
        print(f"  Telegram send failed: {e}")
        return 1
    print("  Sent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
