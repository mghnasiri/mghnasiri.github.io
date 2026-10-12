"""
Auto-promotion / auto-retirement status tracker.

Reads data/stats.json (maintained by update_dashboard.py) and writes
data/model_status.json — one row per model with a status tag that the
dashboard consumes to badge and visually sort models.

Status is judged on the Tims game's own metric: this season's group top-1
lift (the model's pick in each real Tims group vs a random pick from that
group) and its 90% day-bootstrap CI. Trailing top-10 hit rates are still
reported for display only: top-10 picks are mostly group-1 players, where
no model beats random, so they don't measure the game.

Status tags (one per model):
  new          Fewer than GRACE_TIMS_DAYS graded Tims days this season.
  active       Normal.
  warning      The lift's whole CI is below 0 after WARNING_TIMS_DAYS days.
  deprecated   The same after DEPRECATED_TIMS_DAYS days. Retained in the
               workflows so it can be re-evaluated; retire it by hand.
  champion     The Telegram pick source (pick_policy.SOURCE_ORDER[0]),
               chosen on evidence, not by who leads a noisy window.

Workflow position: runs in nhl_daily_update.yml after update_dashboard.py,
inside the results job (06:00 UTC daily).

Author: Mohammad G. Nasiri
"""

import json
import os
from datetime import datetime

from pick_policy import SOURCE_ORDER


# =============================================================================
# THRESHOLDS — tune these if the rules feel too aggressive / too lenient
# =============================================================================
TRAILING_WINDOW = 14              # days for the displayed trailing top-10 rates
CHAMPION_WINDOW = 30

# Status is judged on the game's own metric: the season's group top-1 lift
# over a random pick from the same group, with its 90% day-bootstrap CI
# (update_dashboard.py). Detecting a +3pp gap takes ~450+ Tims days, so a
# model is only flagged once its CI sits wholly below zero.
GRACE_TIMS_DAYS = 10              # fewer graded Tims days this season: "new"
WARNING_TIMS_DAYS = 60            # CI below 0 after this many days: warning
DEPRECATED_TIMS_DAYS = 150        # ... after this many: deprecated
RETIRED = {"neural_v1", "meta_ensemble"}           # no longer run; history stays in stats.json

IN = "data/stats.json"
OUT = "data/model_status.json"


def season_start(now=None):
    """Aug 1 of the current season's start year. Older days are last season's
    (often playoff) slates and shouldn't decide this season's badges."""
    d = now or datetime.now()
    return f"{d.year if d.month >= 8 else d.year - 1}-08-01"


def _trailing_hit_rate(daily_results, window):
    """Overall hit rate across the last `window` days with games; None when
    nothing is graded yet (e.g. opening week), not a misleading 0%."""
    sample = daily_results[-window:] if len(daily_results) >= window else daily_results
    total_hits = sum(d.get("hits", 0) for d in sample)
    total = sum(d.get("total", 0) for d in sample)
    if total == 0:
        return None, len(sample)
    return round(total_hits / total * 100, 1), len(sample)


def classify_model(model_data, start):
    """Return a dict with status + supporting metrics for one model, judged
    on this season's graded days only."""
    daily = [d for d in model_data.get("daily_results", []) if d.get("date", "") >= start]
    season_days = len(daily)

    trailing_14, n14 = _trailing_hit_rate(daily, TRAILING_WINDOW)
    trailing_30, n30 = _trailing_hit_rate(daily, CHAMPION_WINDOW)

    season = (model_data.get("group_top1") or {}).get("season") or {}
    tims_days = season.get("days", 0)
    ci = season.get("lift_ci90") or [None, None]
    below = ci[1] is not None and ci[1] < 0   # whole CI under random
    if tims_days < GRACE_TIMS_DAYS:
        status = "new"
    elif below and tims_days >= DEPRECATED_TIMS_DAYS:
        status = "deprecated"
    elif below and tims_days >= WARNING_TIMS_DAYS:
        status = "warning"
    else:
        status = "active"

    g = model_data.get("group_top1") or {}
    return {
        "status": status,
        "total_days": model_data.get("total_days", season_days),
        "season_days": season_days,
        "last_graded": daily[-1]["date"] if daily else None,
        "trailing_14_hit_rate": trailing_14,
        "trailing_14_sample_size": n14,
        "trailing_30_hit_rate": trailing_30,
        "trailing_30_sample_size": n30,
        "overall_hit_rate": model_data.get("hit_rate", 0),
        # The metric that matches the game: top pick per Tims group.
        "group_top1_rate": g.get("rate"),
        "group_top1_n": g.get("groups"),
        "group_top1_baseline": g.get("baseline_rate"),
        "group_top1_lift": g.get("lift"),
        "group_top1_lift_ci90": g.get("lift_ci90"),
        "season_tims_days": tims_days,
        "season_lift": season.get("lift"),
        "season_lift_ci90": season.get("lift_ci90"),
    }


def main():
    if not os.path.exists(IN):
        print(f"  {IN} not found; run update_dashboard.py first.")
        return 1

    with open(IN, "r", encoding="utf-8") as f:
        stats = json.load(f)

    models = {k: v for k, v in stats.get("models", {}).items() if k not in RETIRED}
    start = season_start()
    statuses = {name: classify_model(data, start) for name, data in models.items()}

    # Champion: the Telegram pick source (pick_policy.SOURCE_ORDER), set on
    # evidence, not by a trailing window. A deprecated model isn't crowned.
    champion = next((m for m in SOURCE_ORDER if m in statuses), None)
    if champion and statuses[champion]["status"] != "deprecated":
        statuses[champion]["is_champion"] = True
        if statuses[champion]["status"] in ("active", "new"):
            statuses[champion]["status"] = "champion"

    output = {
        "generated_at": datetime.now().isoformat(),
        "rules": {
            "metric": "season group top-1 lift over a random pick from the same group, 90% day-bootstrap CI",
            "grace_tims_days": GRACE_TIMS_DAYS,
            "warning": f"CI upper bound < 0 after {WARNING_TIMS_DAYS} Tims days",
            "deprecated": f"CI upper bound < 0 after {DEPRECATED_TIMS_DAYS} Tims days",
            "season_start": start,
            "champion_rule": "the Telegram pick source (pick_policy.SOURCE_ORDER[0])",
        },
        "models": statuses,
    }

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"  Wrote {OUT}")
    print(f"  Status breakdown:")
    from collections import Counter
    c = Counter(s["status"] for s in statuses.values())
    for status, count in c.most_common():
        names = [n for n, s in statuses.items() if s["status"] == status]
        print(f"    {status}: {count}  ({', '.join(names)})")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
