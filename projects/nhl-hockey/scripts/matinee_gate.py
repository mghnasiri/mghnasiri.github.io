"""
Decide whether today's early ("matinee") pass should run.

The regular pass starts 2-4 h after its 14:00-15:20 UTC crons and finished
after 17:00 UTC on every September day, so on days with an early first puck
(weekend and holiday matinees) picks arrived after the Tims pick deadline.
The MC, xG, Linear and Meta workflows also have an early cron; this gate lets
that early pass run only when today's first game starts before
LATE_PASS_SAFE_UTC. On other days it would just duplicate the regular pass.

Writes run=true/false to $GITHUB_OUTPUT. Outside the early cron (EARLY is not
'true') it always says run=true. If the schedule can't be fetched, the early
pass runs anyway: an extra pass is cheaper than late picks. stdlib only (the
gate job installs nothing).
"""

import json
import os
import urllib.request
from datetime import datetime, timezone

# First puck at/after 20:00 UTC: the regular pass is in time. (In September
# Meta finished after 19:00 UTC on ~40% of days, so 19:00 starts are at risk.)
LATE_PASS_SAFE_UTC = 20


def first_puck(today):
    """Earliest regular-season/playoff start today as an aware UTC datetime,
    or None if there are no games. Raises if the schedule can't be fetched."""
    req = urllib.request.Request(f"https://api-web.nhle.com/v1/schedule/{today}",
                                 headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.load(resp)
    starts = [g["startTimeUTC"] for d in data.get("gameWeek", []) if d.get("date") == today
              for g in d.get("games", []) if g.get("gameType") in (2, 3) and g.get("startTimeUTC")]
    if not starts:
        return None
    return min(datetime.fromisoformat(s.replace("Z", "+00:00")) for s in starts)


def should_run(today):
    try:
        first = first_puck(today)
    except Exception as e:
        print(f"  Schedule unavailable ({e}); running the early pass to be safe.")
        return True
    if first is None:
        print(f"  No games on {today}; skipping the early pass.")
        return False
    cutoff = datetime.strptime(today, "%Y-%m-%d").replace(hour=LATE_PASS_SAFE_UTC, tzinfo=timezone.utc)
    early = first < cutoff
    print(f"  First puck {first:%Y-%m-%d %H:%M} UTC -> "
          f"{'matinee: running the early pass' if early else 'regular pass is in time: skipping'}.")
    return early


def main():
    run = True
    if os.environ.get("EARLY") == "true":
        run = should_run(datetime.now().strftime("%Y-%m-%d"))
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as f:
            f.write(f"run={'true' if run else 'false'}\n")
    print(f"run={'true' if run else 'false'}")


if __name__ == "__main__":
    main()
