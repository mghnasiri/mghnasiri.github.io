"""
Auto-promotion / auto-retirement status tracker.

Reads data/stats.json (per-model daily hit-rate history maintained by
update_dashboard.py), applies trailing-window rules, and writes
data/model_status.json — one row per model with a status tag that the
dashboard consumes to badge and visually sort models.

Status tags (one per model):
  new          Fewer than GRACE_PERIOD_DAYS of results. Shielded from
               warnings while the sample is still noisy.
  active       Normal. Passes all thresholds.
  warning      14-day trailing hit rate below WARNING_HIT_RATE. Soft
               signal — dashboard shows a yellow badge but model stays.
  deprecated   14-day trailing hit rate below DEPRECATED_HIT_RATE AND
               total tracked days >= DEPRECATED_MIN_DAYS. Dashboard
               marks with a red badge and moves to the bottom of the
               comparison list. Retained in workflow runs so you can
               re-evaluate; remove from nhl_xg_v3_update.yml manually
               once you're sure.
  champion     Highest 30-day trailing hit rate among non-new models.
               Ties within CHAMPION_TIE_TOLERANCE share the title.

Workflow position: runs in nhl_daily_update.yml after update_dashboard.py,
inside the results job (06:00 UTC daily).

Author: Mohammad G. Nasiri
"""

import json
import os
from datetime import datetime


# =============================================================================
# THRESHOLDS — tune these if the rules feel too aggressive / too lenient
# =============================================================================
GRACE_PERIOD_DAYS = 7             # new models shielded from warning/retire
TRAILING_WINDOW = 14              # days to evaluate warning/deprecated
DEPRECATED_WINDOW = 14            # separate knob if you want it different
CHAMPION_WINDOW = 30              # days to pick champion on

WARNING_HIT_RATE = 15.0           # below this on 14-day = yellow badge
DEPRECATED_HIT_RATE = 10.0        # below this AND >= DEPRECATED_MIN_DAYS
DEPRECATED_MIN_DAYS = 10          # require enough evidence before red flag

CHAMPION_TIE_TOLERANCE = 1.0      # models within 1% of leader share champion

IN = "data/stats.json"
OUT = "data/model_status.json"


def _trailing_hit_rate(daily_results, window):
    """Overall hit rate across the last `window` days with games."""
    sample = daily_results[-window:] if len(daily_results) >= window else daily_results
    total_hits = sum(d.get("hits", 0) for d in sample)
    total = sum(d.get("total", 0) for d in sample)
    rate = (total_hits / total * 100) if total > 0 else 0.0
    return round(rate, 1), len(sample)


def classify_model(model_data):
    """Return a dict with status + supporting metrics for one model."""
    daily = model_data.get("daily_results", [])
    total_days = model_data.get("total_days", len(daily))

    trailing_14, n14 = _trailing_hit_rate(daily, TRAILING_WINDOW)
    trailing_30, n30 = _trailing_hit_rate(daily, CHAMPION_WINDOW)

    # Grace period
    if total_days < GRACE_PERIOD_DAYS:
        status = "new"
    # Deprecated: below red threshold AND enough evidence
    elif (trailing_14 < DEPRECATED_HIT_RATE
          and total_days >= DEPRECATED_MIN_DAYS):
        status = "deprecated"
    elif trailing_14 < WARNING_HIT_RATE:
        status = "warning"
    else:
        status = "active"

    return {
        "status": status,
        "total_days": total_days,
        "trailing_14_hit_rate": trailing_14,
        "trailing_14_sample_size": n14,
        "trailing_30_hit_rate": trailing_30,
        "trailing_30_sample_size": n30,
        "overall_hit_rate": model_data.get("hit_rate", 0),
    }


def main():
    if not os.path.exists(IN):
        print(f"  {IN} not found; run update_dashboard.py first.")
        return 1

    with open(IN, "r", encoding="utf-8") as f:
        stats = json.load(f)

    models = stats.get("models", {})
    statuses = {name: classify_model(data) for name, data in models.items()}

    # Champion: among non-new models, pick the highest 30-day trailing rate.
    # Tie tolerance allows multiple champions so we don't punish near-ties.
    eligible = {
        name: meta for name, meta in statuses.items()
        if meta["status"] != "new"
        and meta["trailing_30_sample_size"] >= GRACE_PERIOD_DAYS
    }
    if eligible:
        top_rate = max(m["trailing_30_hit_rate"] for m in eligible.values())
        champions = [
            name for name, meta in eligible.items()
            if meta["trailing_30_hit_rate"] >= top_rate - CHAMPION_TIE_TOLERANCE
        ]
        # Champion overrides active/warning (but not deprecated — a deprecated
        # model shouldn't be crowned, that would be a contradictory signal).
        for name in champions:
            if statuses[name]["status"] == "deprecated":
                continue
            statuses[name]["is_champion"] = True
            if statuses[name]["status"] == "active":
                statuses[name]["status"] = "champion"

    output = {
        "generated_at": datetime.now().isoformat(),
        "rules": {
            "grace_period_days": GRACE_PERIOD_DAYS,
            "trailing_window": TRAILING_WINDOW,
            "champion_window": CHAMPION_WINDOW,
            "warning_hit_rate": WARNING_HIT_RATE,
            "deprecated_hit_rate": DEPRECATED_HIT_RATE,
            "deprecated_min_days": DEPRECATED_MIN_DAYS,
            "champion_tie_tolerance": CHAMPION_TIE_TOLERANCE,
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
