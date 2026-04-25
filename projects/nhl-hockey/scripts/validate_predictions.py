"""
Daily prediction sanity validator.

Loads every model's latest.json, applies a small set of "if this trips, the
output is probably broken" checks, and prints a structured report. Designed
to catch the silent-failure pattern that hid the meta_ensemble saturation
and market_odds missing-games bugs — both shipped with workflows reporting
green for days while the actual prediction output was wrong.

Checks per model:
  freshness        latest.json date == today
  team_coverage    predictions span >= MIN_TEAM_COVERAGE % of game teams
  player_count     predictions list isn't suspiciously empty
  no_saturation    no probability == 1.0 exactly (calibrator runaway)
  no_zero          no probability == 0.0 exactly (signal is dead)
  prob_max_sane    top probability <= MAX_TOP_PROB (NHL skaters cap ~40%)
  no_flat_ties     not too many predictions tied at the same probability

Outputs:
  data/health_report.json    machine-readable per-model status
  Console: human-readable findings, exits non-zero if any FAIL

Run via daily health-check workflow after all prediction workflows finish.
Calling it from within an individual prediction workflow would be tighter
but adds maintenance overhead per workflow; one centralized check is cheaper.

Author: Mohammad G. Nasiri
"""

import json
import os
import sys
from collections import Counter
from datetime import datetime


# =============================================================================
# THRESHOLDS — tune as data quality patterns emerge
# =============================================================================
PRED_DIR = "data/predictions"
HEALTH_REPORT = "data/health_report.json"

# How much of the night's team list must appear in predictions to count as
# "full coverage". 0.6 is generous: it allows a Tims-filtered model to skip
# teams whose Tim Hortons players don't qualify, but flags the case where
# the bug from 2026-04-24 dropped 5 of 6 teams.
MIN_TEAM_COVERAGE = 0.60

# If NHL has games today, the model should have at least this many predictions.
# Below this typically means something filtered too aggressively.
MIN_PREDICTIONS_IF_GAMES = 5

# Soft ceiling for top probability. Above WARN_TOP_PROB is suspicious but
# not necessarily broken (Poisson P = 1 - exp(-xG) gives ~0.63 for an elite
# scorer with xG ~1.0 in a hot matchup — legitimate). Above HARD_TOP_PROB
# is almost certainly broken — even xG = 2.0 only yields P = 0.86, which
# would require a player who's expected to score multiple goals.
WARN_TOP_PROB = 0.55
HARD_TOP_PROB = 0.85

# Models that legitimately produce inflated probabilities and shouldn't
# fail prob_max_sane validation:
#   xg_v3_synthetic — A/B control, broken-baseline by design
#   meta_ensemble   — calibrator disabled in commit 6c46a59 due to a
#                     saturation bug; raw LightGBM output is uncalibrated
#                     and runs hot. Re-enable validation once we have
#                     >=2,000 held-out rows and can refit the isotonic.
EXEMPT_FROM_TOP_PROB = {"xg_v3_synthetic", "meta_ensemble"}

# How many predictions can share the same probability before we flag it
# as a flat-tie issue (e.g., the meta_ensemble bug where 5 players were
# all tied at exactly 0.1538).
MAX_FLAT_TIES = 4


# =============================================================================
# HELPERS
# =============================================================================
def load_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"_load_error": str(e)}


def discover_models():
    """Return list of model_name strings from data/predictions/* with a latest.json."""
    if not os.path.isdir(PRED_DIR):
        return []
    out = []
    for name in sorted(os.listdir(PRED_DIR)):
        path = os.path.join(PRED_DIR, name, "latest.json")
        if os.path.isfile(path):
            out.append(name)
    return out


# =============================================================================
# CHECKS — each returns (status, message) where status in {"ok", "warn", "fail"}
# =============================================================================
def check_freshness(data, today):
    pred_date = data.get("date")
    if pred_date == today:
        return "ok", f"date={pred_date}"
    return "fail", f"date={pred_date} (expected {today}) — workflow may have failed"


def check_team_coverage(data):
    games = data.get("games", [])
    preds = data.get("predictions", [])
    if not games:
        return "ok", "no games today"
    # Some models (market_odds) store full team names in games[] (from the
    # Odds API) while predictions[] uses NHL abbreviations. Compare via a
    # match if EITHER full names or abbreviations cover game teams.
    game_full = set()
    for g in games:
        for k in ("home_team", "away_team"):
            v = g.get(k)
            if v:
                game_full.add(v)
    pred_teams = {p.get("team") for p in preds if p.get("team")}
    if not game_full:
        return "ok", "could not determine game teams"
    # Approximate per-team match: a game's team is "covered" if any
    # prediction's abbreviation appears as the last word of the full name
    # (e.g., "Carolina Hurricanes" -> last word "Hurricanes" doesn't match
    # "CAR", so we instead match by team count: if pred_teams matches at
    # least len(games) distinct values, treat as covered).
    direct_match = pred_teams & game_full
    if direct_match:
        coverage = len(direct_match) / len(game_full)
    else:
        # Predictions use abbreviations; games use full names. Use raw
        # team-count parity instead of name matching.
        coverage = min(len(pred_teams) / max(len(game_full), 1), 1.0)
    pct = round(coverage * 100, 1)
    if coverage >= MIN_TEAM_COVERAGE:
        return "ok", f"{pct}% team coverage ({len(pred_teams)} pred teams vs {len(game_full)} game teams)"
    missing = sorted(game_full - pred_teams) if direct_match else \
              [f"({len(game_full) - len(pred_teams)} teams short)"]
    return "fail", f"only {pct}% of game teams covered (missing {missing})"


def check_player_count(data):
    games = data.get("games", [])
    preds = data.get("predictions", [])
    if not games:
        return "ok", f"{len(preds)} predictions, no games"
    if len(preds) < MIN_PREDICTIONS_IF_GAMES:
        return "fail", (f"{len(preds)} predictions but {len(games)} games "
                        f"scheduled — filter likely over-aggressive")
    return "ok", f"{len(preds)} predictions for {len(games)} games"


def check_no_saturation(data):
    preds = data.get("predictions", [])
    if not preds:
        return "ok", "no predictions"
    sat = [p for p in preds if p.get("goal_probability", 0) >= 0.999]
    if sat:
        names = ", ".join(p["name"] for p in sat[:3])
        return "fail", (f"{len(sat)} predictions saturated at p>=0.999 "
                        f"({names}{'...' if len(sat) > 3 else ''})")
    return "ok", "no saturated probabilities"


def check_no_zero(data):
    preds = data.get("predictions", [])
    if not preds:
        return "ok", "no predictions"
    zeros = [p for p in preds if p.get("goal_probability", 0) <= 0.0]
    if zeros:
        return "warn", f"{len(zeros)} predictions at p<=0.0"
    return "ok", "no zero probabilities"


def check_prob_max_sane(data, model_name=None):
    preds = data.get("predictions", [])
    if not preds:
        return "ok", "no predictions"
    top = max(p.get("goal_probability", 0) for p in preds)
    if model_name in EXEMPT_FROM_TOP_PROB:
        return "ok", f"top prob = {top:.3f} (model exempt — A/B control)"
    if top > HARD_TOP_PROB:
        return "fail", (f"top prob = {top:.3f} > {HARD_TOP_PROB} — "
                        f"calibration / normalization bug suspected")
    if top > WARN_TOP_PROB:
        return "warn", f"top prob = {top:.3f} (high but plausible)"
    return "ok", f"top prob = {top:.3f}"


def check_no_flat_ties(data):
    preds = data.get("predictions", [])
    if len(preds) < 5:
        return "ok", "too few predictions to check"
    probs = Counter(round(p.get("goal_probability", 0), 4) for p in preds)
    biggest_tie_value, biggest_tie_count = probs.most_common(1)[0]
    if biggest_tie_count > MAX_FLAT_TIES:
        return "fail", (f"{biggest_tie_count} predictions tied at exactly "
                        f"p={biggest_tie_value:.4f} — calibrator flat-region "
                        f"or NaN propagation suspected")
    return "ok", f"max tie size = {biggest_tie_count}"


CHECKS = {
    "freshness": check_freshness,
    "team_coverage": check_team_coverage,
    "player_count": check_player_count,
    "no_saturation": check_no_saturation,
    "no_zero": check_no_zero,
    "prob_max_sane": check_prob_max_sane,
    "no_flat_ties": check_no_flat_ties,
}


# =============================================================================
# MAIN
# =============================================================================
def main():
    today = datetime.now().strftime("%Y-%m-%d")
    print("=" * 70)
    print(f"  PREDICTION VALIDATOR — {today}")
    print("=" * 70)

    models = discover_models()
    if not models:
        print(f"  No models found under {PRED_DIR}")
        return 0

    report = {
        "generated_at": datetime.now().isoformat(),
        "today": today,
        "thresholds": {
            "min_team_coverage": MIN_TEAM_COVERAGE,
            "min_predictions_if_games": MIN_PREDICTIONS_IF_GAMES,
            "warn_top_prob": WARN_TOP_PROB,
            "hard_top_prob": HARD_TOP_PROB,
            "max_flat_ties": MAX_FLAT_TIES,
        },
        "models": {},
    }

    overall_status = "ok"
    fail_summary = []

    for name in models:
        data = load_json(f"{PRED_DIR}/{name}/latest.json")
        if not data or "_load_error" in (data or {}):
            err = (data or {}).get("_load_error", "missing latest.json")
            print(f"\n  {name}: COULD NOT LOAD ({err})")
            report["models"][name] = {"status": "fail",
                                       "checks": {"load": ["fail", err]}}
            overall_status = "fail"
            fail_summary.append(f"{name}: load error")
            continue

        # Freshness needs `today` and prob_max_sane needs model name
        results = {"freshness": check_freshness(data, today)}
        for k, fn in CHECKS.items():
            if k == "freshness":
                continue
            if k == "prob_max_sane":
                results[k] = fn(data, model_name=name)
            else:
                results[k] = fn(data)

        worst = max(results.values(), key=lambda r: ["ok", "warn", "fail"].index(r[0]))
        model_status = worst[0]
        report["models"][name] = {
            "status": model_status,
            "checks": {k: list(v) for k, v in results.items()},
        }

        icon = {"ok": "OK", "warn": "WARN", "fail": "FAIL"}[model_status]
        print(f"\n  [{icon}] {name}")
        for cname, (cstatus, cmsg) in results.items():
            mark = {"ok": " ", "warn": "?", "fail": "!"}[cstatus]
            print(f"        {mark} {cname}: {cmsg}")

        if model_status == "fail":
            overall_status = "fail"
            failed_checks = [k for k, v in results.items() if v[0] == "fail"]
            fail_summary.append(f"{name}: {', '.join(failed_checks)}")
        elif model_status == "warn" and overall_status == "ok":
            overall_status = "warn"

    # Save report
    os.makedirs(os.path.dirname(HEALTH_REPORT), exist_ok=True)
    with open(HEALTH_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print(f"  OVERALL: {overall_status.upper()}")
    if fail_summary:
        print("  Failures:")
        for f in fail_summary:
            print(f"    - {f}")
    print(f"  Report: {HEALTH_REPORT}")
    print("=" * 70)

    return 1 if overall_status == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())
