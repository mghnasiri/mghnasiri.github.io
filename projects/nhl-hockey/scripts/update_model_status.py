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
  champion     Highest 30-day trailing hit rate this season among non-new,
               non-stale models; models the leader isn't shown to beat on
               the same days (paired bootstrap) share the title.

Workflow position: runs in nhl_daily_update.yml after update_dashboard.py,
inside the results job (06:00 UTC daily).

Author: Mohammad G. Nasiri
"""

import json
import math
import os
import random
from datetime import datetime, timedelta


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

# Co-champion unless the leader is shown better on the same days: paired
# day-bootstrap of (leader - model) hit rate; share the title if its 5th
# percentile is <= 0. (The old fixed 1.0 pp tolerance sat far inside the
# day-to-day noise, so ties and gaps were both mostly luck.)
BOOTSTRAP_RESAMPLES = 2000
STALE_DAYS = 2                    # last graded this much before newest -> ineligible
RETIRED = {"neural_v1"}           # no longer run; history stays in stats.json

IN = "data/stats.json"
OUT = "data/model_status.json"


def season_start(now=None):
    """Aug 1 of the current season's start year. Older days are last season's
    (often playoff) slates and shouldn't decide this season's badges."""
    d = now or datetime.now()
    return f"{d.year if d.month >= 8 else d.year - 1}-08-01"


def _paired_diff_ci(leader_daily, other_daily):
    """(diff_pp, [p5, p95]) of leader-minus-model hit rate over the days both
    were graded, resampling days with a fixed seed. (None, None) if no overlap."""
    a = {d["date"]: d for d in leader_daily}
    b = {d["date"]: d for d in other_daily}
    dates = sorted(set(a) & set(b))
    if not dates:
        return None, None

    def diff(sample):
        at = sum(a[x]["total"] for x in sample)
        bt = sum(b[x]["total"] for x in sample)
        if not at or not bt:
            return 0.0
        return (sum(a[x]["hits"] for x in sample) / at - sum(b[x]["hits"] for x in sample) / bt) * 100

    rng = random.Random(0)
    boots = sorted(diff([rng.choice(dates) for _ in dates]) for _ in range(BOOTSTRAP_RESAMPLES))
    return round(diff(dates), 1), [round(boots[int(0.05 * BOOTSTRAP_RESAMPLES)], 1),
                                   round(boots[int(0.95 * BOOTSTRAP_RESAMPLES)], 1)]


def _wilson90(hits, n):
    """90% Wilson interval for a hit rate, in percent."""
    if not n:
        return None
    z, p = 1.645, hits / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round((centre - half) * 100, 1), round((centre + half) * 100, 1)]


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

    # Grace period
    if season_days < GRACE_PERIOD_DAYS:
        status = "new"
    # Deprecated: below red threshold AND enough evidence
    elif (trailing_14 < DEPRECATED_HIT_RATE
          and season_days >= DEPRECATED_MIN_DAYS):
        status = "deprecated"
    elif trailing_14 < WARNING_HIT_RATE:
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
        "group_top1_ci90": _wilson90(g.get("hits", 0), g.get("groups", 0)),
        "group_top1_baseline": g.get("baseline_rate"),
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

    # A model that stopped producing output keeps its old trailing rate; it
    # must not stay champion.
    graded = [s["last_graded"] for s in statuses.values() if s["last_graded"]]
    newest = max(graded) if graded else None
    cutoff = ((datetime.strptime(newest, "%Y-%m-%d") - timedelta(days=STALE_DAYS)).strftime("%Y-%m-%d")
              if newest else None)

    # Champion: the highest 30-day (this season) rate, plus any model the
    # leader isn't shown to beat on the same days (paired bootstrap).
    eligible = {
        name: meta for name, meta in statuses.items()
        if meta["status"] != "new"
        and meta["trailing_30_sample_size"] >= GRACE_PERIOD_DAYS
        and meta["last_graded"] and meta["last_graded"] >= cutoff
    }
    if eligible:
        leader = max(eligible, key=lambda n: eligible[n]["trailing_30_hit_rate"])
        window = lambda n: [d for d in models[n].get("daily_results", [])
                            if d.get("date", "") >= start][-CHAMPION_WINDOW:]
        champions = [leader]
        for name in eligible:
            if name == leader:
                continue
            diff, ci = _paired_diff_ci(window(leader), window(name))
            statuses[name]["champion_diff_pp"] = diff
            statuses[name]["champion_diff_ci90"] = ci
            if ci is not None and ci[0] <= 0:
                champions.append(name)
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
            "season_start": start,
            "stale_days": STALE_DAYS,
            "champion_rule": "leader + models whose paired-bootstrap 90% CI of (leader - model) includes 0",
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
