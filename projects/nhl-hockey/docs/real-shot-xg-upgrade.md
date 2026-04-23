# Real-Shot xG Upgrade — Step-by-Step Guide

**Goal:** Stop feeding the xG model 20 synthesized shot profiles per player and start feeding it the player's **actual shots from this season's play-by-play data**. Expected lift: +5 to +10 points of Top-10 hit rate.

**Status of the codebase as this guide was written (2026-04-23):**
- `scripts/collect_xg_data.py` already pulls PBP shots and writes `data/xg_training/shots.csv` (35,357 rows, 411 games, last run 2026-04-20). ✅ Ingestion exists.
- `scripts/xg_predict.py` still uses `calculate_xg_with_model()` at [xg_predict.py:328](../scripts/xg_predict.py#L328), which generates fake shots. ❌ Prediction still synthetic.
- `data/xg_model/metadata.json` shows the model was trained on 7,942 shots with 35 features at CV AUC 0.9293. The CSV has grown 4.5× since — model can be retrained on more data as a bonus.
- No scheduled job: ingestion runs manually. Playoff games are only 7 of the last week's games, because nobody's re-running the collector daily.

**What this guide does NOT do:**
- Multi-season backfill (deferred — see roadmap memory)
- Line combinations / TOI projection (deferred)
- Neural sequence model (deferred)

---

## Architecture before/after

**Before (synthetic shots):**
```
xg_predict.py
  └─ for each player:
       └─ estimate_shot_profile()        # position-based averages
       └─ calculate_xg_with_model()
            └─ generate 20 synthetic shots via np.random.normal
            └─ score through XGBoost
            └─ return mean xG per shot
  └─ player_xg = avg_shot_xg × expected_shots
```

**After (real shots):**
```
aggregate_player_shots.py              (NEW, runs daily after collector)
  └─ read data/xg_training/shots.csv
  └─ per player: keep last N shots (chronological), one-hot encode,
                 write data/player_shots/{player_id}.json
  └─ per player_cold_start: write data/player_shots/_position_priors.json

xg_predict.py
  └─ for each player:
       └─ load data/player_shots/{player_id}.json  (fallback to position prior)
       └─ score loaded shots through XGBoost
       └─ return mean xG per shot
  └─ player_xg = avg_shot_xg × expected_shots
```

The `expected_shots` calculation stays exactly the same (already uses real season averages). Only `avg_shot_xg` — the quality side — becomes real instead of synthetic.

---

## Phases

- **Phase 1**: Confirm + automate ingestion (2-4 hrs)
- **Phase 2**: Build per-player shot aggregator (4-8 hrs)
- **Phase 3**: Swap prediction logic (2-4 hrs)
- **Phase 4**: Validate A/B vs synthetic (1-2 weeks of live running)
- **Phase 5** (optional): Retrain xG on the fuller dataset (2-4 hrs)
- **Phase 6**: Maintenance & monitoring (ongoing, ~1 hr/month)

Total active work: **1 to 2 weeks**. Plus the 1-2 weeks of validation running in parallel.

---

## Phase 1 — Confirm + automate PBP ingestion

The collector already works. The gap is that it stopped at 2026-04-20. Check what's missed, backfill, and automate.

### Step 1.1 — Audit current ingestion coverage

```bash
cd projects/nhl-hockey
python3 -c "
import json
with open('data/xg_training/collection_log.json') as f:
    log = json.load(f)
games = log.get('processed_games', [])
print(f'Total: {len(games)}  Last update: {log[\"last_updated\"]}')
print(f'First: {games[0]}  Last: {games[-1]}')
"
```

**Expected state today (2026-04-23):** ~411 games, last update 2026-04-20, missing 3+ days of playoffs.

**Done when:** you know how many games and how many days are missing.

### Step 1.2 — Backfill the gap

```bash
python3 scripts/collect_xg_data.py --backfill 7
```

This calls [collect_xg_data.py:386](../scripts/collect_xg_data.py#L386) with a 7-day lookback. The script already uses `processed_games` as a dedup set, so re-running is safe.

**Gotchas:**
- Rate limit is `0.5s` per game ([collect_xg_data.py:33](../scripts/collect_xg_data.py#L33)). 7 days × ~8 games/day × 0.5s ≈ 30 sec. Fine.
- Playoff games use IDs like `20250301XX` where `XX` encodes round/matchup. The collector treats them the same as regular-season (`gameType in [2, 3]`). ✅ no change needed.

**Done when:** `shots.csv` row count grew, `collection_log.json` includes playoff games through yesterday.

### Step 1.3 — Schedule daily runs

Your existing daily predictions run via GitHub Actions (implied by commits like "xG v3 predictions for 2026-04-20"). Find the workflow file — likely `.github/workflows/daily.yml` — and add a step that runs **before** the prediction scripts:

```yaml
- name: Collect yesterday's shots
  run: python3 projects/nhl-hockey/scripts/collect_xg_data.py --backfill 2
```

The `--backfill 2` gives a 2-day safety window in case yesterday's run failed.

**Decision point (yours):** Should the collector run fail the whole workflow, or just warn? Recommend: **fail loudly** for the first month so you notice schema drift. Switch to `|| true` only after it's been stable.

**Done when:** a scheduled run has completed and `collection_log.json.last_updated` is today's date.

---

## Phase 2 — Build the per-player shot aggregator

This is the new script. It reads `shots.csv` once per day and writes one JSON per player.

### Step 2.1 — Design decisions (read before coding)

Three decisions you'll want to make deliberately. Learning mode kicks in here — these shape the feature's behavior.

**Decision A: How many recent shots per player?**
- Too few (e.g., last 20): noisy, dominated by 1-2 hot games
- Too many (e.g., whole season): stale, doesn't capture in-season development
- **Recommended: last 60 shots OR last 15 games, whichever is more**. For a top scorer that's ~3 weeks of form; for a depth player it's ~2 months. Document your choice; this is the biggest lever for "recent form" sensitivity.

**Decision B: How to handle players with < 20 career shots this season?**
Rookies, recent call-ups, traded-in players. Options:
1. Fall back to **position-based priors** (mean shot distribution for C / L / R / D) — simple, stable
2. Fall back to the **synthetic generator** (current behavior) — familiar, known-working
3. Fall back to a **similar-player embedding** (e.g., same age + position + team tier) — best, but complex

**Recommended: option 1** for v1. Revisit if you see low-N players dominating top-10 picks.

**Decision C: Weight recent shots more than older ones?**
A rebound Feb 1 shouldn't count the same as a rebound last week. Options:
1. Flat weighting (no decay) — simplest
2. Exponential decay by days (e.g., `weight = 0.5 ** (days_ago / 30)`)
3. Game-index decay (last game = 1.0, prior game = 0.95, etc.)

**Recommended: flat for v1**. The xG model doesn't see shot recency anyway — weighting changes the *sample* it scores, which has second-order effects that are hard to reason about. Don't add this complexity until you've validated real-shot replay beats synthetic.

### Step 2.2 — Script sketch: `scripts/aggregate_player_shots.py`

```python
"""
Aggregate per-player real shot distributions from data/xg_training/shots.csv.
Outputs one JSON per player under data/player_shots/{player_id}.json, plus a
position-prior file for cold-start players.

Runs daily after collect_xg_data.py.
"""
import csv
import json
import os
from collections import defaultdict
from datetime import datetime

SHOTS_CSV = "data/xg_training/shots.csv"
OUT_DIR = "data/player_shots"
MAX_SHOTS_PER_PLAYER = 60          # Decision A
MIN_SHOTS_FOR_PLAYER_FILE = 20     # below this, rely on position prior

FEATURE_NAMES = [                  # from data/xg_model/metadata.json
    "shot_distance", "shot_angle", "is_powerplay", "is_rebound",
    "seconds_since_last_event", "is_empty_net", "score_differential",
    "period", "prior_event_distance", "distance_x_angle",
    "is_slot_shot", "is_high_danger",
]
SHOT_TYPES = ["backhand","bat","between-legs","deflected","poke","slap",
              "snap","tip-in","wrap-around","wrist"]
STRENGTH_STATES = ["1v0","3v3","3v4","4v3","4v4","4v5","4v6","5v3",
                   "5v4","5v5","5v6","6v4","6v5"]

def row_to_feature_dict(row):
    """Convert one CSV row into the exact feature dict the xG model expects."""
    dist = float(row["shot_distance"])
    angle = float(row["shot_angle"])
    feat = {
        "shot_distance": dist,
        "shot_angle": angle,
        "is_powerplay": int(row["is_powerplay"]),
        "is_rebound": int(row["is_rebound"]),
        "seconds_since_last_event": float(row["seconds_since_last_event"]),
        "is_empty_net": int(row["is_empty_net"]),
        "score_differential": int(row["score_differential"]),
        "period": int(row["period"]),
        "prior_event_distance": float(row["prior_event_distance"]),
        "distance_x_angle": dist * angle,
        "is_slot_shot": int(dist < 20),
        "is_high_danger": int(dist < 30 and angle < 30),
    }
    for st in SHOT_TYPES:
        feat[f"shot_type_{st}"] = 1 if row["shot_type"] == st else 0
    for ss in STRENGTH_STATES:
        feat[f"strength_state_{ss}"] = 1 if row["strength_state"] == ss else 0
    return feat

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    by_player = defaultdict(list)  # player_id -> list of (game_date, feat_dict)

    with open(SHOTS_CSV) as f:
        for row in csv.DictReader(f):
            pid = int(row["player_id"])
            if not pid:
                continue
            by_player[pid].append((row["game_date"], row_to_feature_dict(row)))

    # Sort newest-first, truncate to MAX_SHOTS_PER_PLAYER
    for pid, shots in by_player.items():
        shots.sort(key=lambda x: x[0], reverse=True)
        recent = [feat for _, feat in shots[:MAX_SHOTS_PER_PLAYER]]
        if len(recent) < MIN_SHOTS_FOR_PLAYER_FILE:
            continue
        out = {
            "player_id": pid,
            "shot_count": len(recent),
            "latest_shot_date": shots[0][0],
            "shots": recent,
        }
        with open(f"{OUT_DIR}/{pid}.json", "w") as f:
            json.dump(out, f)

    # Build position priors (for cold-start players, looked up later)
    # ... see Step 2.3

    print(f"  Wrote {len(os.listdir(OUT_DIR))} player files")

if __name__ == "__main__":
    main()
```

**Where the 5-10 lines of your judgment go:** this script intentionally has no smoothing, no season-boundary handling, no team-change correction. Add only what the next phase's validation actually demands.

### Step 2.3 — Position priors for cold-start players

Run the same loop a second time, bucketing by position (need to join to roster data since the shots CSV doesn't carry position). Save `data/player_shots/_position_priors.json` with one `shots` array per position code (C, L, R, D). These are used in Phase 3 for any player not in the main directory.

Simplest approach: look up position from `data/predictions/neural_network/latest.json` (every player in a prediction has `position`) and aggregate by position. Roughly 5K shots per position code after filtering.

**Done when:**
- `data/player_shots/` contains ~600-800 player files
- `data/player_shots/_position_priors.json` exists
- A quick sanity check of one star's file shows sensible shots:
  ```bash
  python3 -c "import json; d=json.load(open('data/player_shots/8477956.json')); print('Pastrnak:', d['shot_count'], 'shots; latest', d['latest_shot_date'])"
  ```

### Step 2.4 — Integrate into the daily workflow

In the GitHub Actions workflow file, add:

```yaml
- name: Aggregate per-player shots
  run: python3 projects/nhl-hockey/scripts/aggregate_player_shots.py
```

Order: `collect_xg_data.py` → `aggregate_player_shots.py` → prediction scripts.

---

## Phase 3 — Swap prediction logic

### Step 3.1 — Modify `xg_predict.py` `calculate_xg_with_model`

Current code at [xg_predict.py:328-407](../scripts/xg_predict.py#L328):

```python
def calculate_xg_with_model(model, metadata, profile, num_shots=20, player_id=0):
    # ... generates np.random.normal shots and scores them
```

Replace the body so that synthetic generation is a **fallback**, not the default:

```python
def calculate_xg_with_model(model, metadata, profile, num_shots=20, player_id=0):
    if not HAS_PANDAS:
        return None
    feature_names = metadata.get('feature_names', [])
    if not feature_names:
        return None

    # NEW: try real shots first
    real_shots = _load_real_shots(player_id, profile.get('position'))
    if real_shots is not None:
        df = pd.DataFrame(real_shots)
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_names]
        try:
            probas = model.predict_proba(df)[:, 1]
            return float(np.mean(probas))
        except Exception:
            pass  # fall through to synthetic

    # Existing synthetic generation as fallback (unchanged) ...
```

And add the loader at module scope:

```python
PLAYER_SHOTS_DIR = "data/player_shots"
_POSITION_PRIORS = None

def _load_real_shots(player_id, position):
    """Return list of feature dicts for this player's recent shots, or None."""
    path = f"{PLAYER_SHOTS_DIR}/{player_id}.json"
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f).get("shots")
        except Exception:
            return None
    # Cold start: position prior
    global _POSITION_PRIORS
    if _POSITION_PRIORS is None:
        priors_path = f"{PLAYER_SHOTS_DIR}/_position_priors.json"
        if os.path.exists(priors_path):
            with open(priors_path) as f:
                _POSITION_PRIORS = json.load(f)
        else:
            _POSITION_PRIORS = {}
    pos_key = (position or "C")[0]
    if pos_key not in ("C","L","R","D"):
        pos_key = "C"
    prior = _POSITION_PRIORS.get(pos_key)
    return prior.get("shots") if prior else None
```

**Done when:** a dry-run prints "loaded real shots" for top scorers and "using position prior" for a rookie you've chosen as a check case.

### Step 3.2 — Add a dashboard flag in the prediction output

In the output JSON ([xg_predict.py:730-749](../scripts/xg_predict.py#L730)), add per-player:

```python
"xg_source": "real" | "position_prior" | "synthetic"
```

This lets you filter the dashboard and the `validate_predictions.py` script (Phase 4) to see hit rates by source.

**Done when:** `data/predictions/xg_v3/latest.json` has an `xg_source` field populated for every player.

---

## Phase 4 — Validate A/B

Do NOT delete the synthetic path yet. Run both side by side for ~2 weeks.

### Step 4.1 — Parallel-scoring script

Create `scripts/xg_predict_synthetic_only.py` — a copy of `xg_predict.py` with Phase 3's changes reverted (i.e., forces synthetic). Output to `data/predictions/xg_v3_synthetic/`. Add it to the daily workflow as a separate job.

### Step 4.2 — Extend `fetch_results.py` to compare both

[fetch_results.py:199-248](../scripts/fetch_results.py#L199) already loops over every folder under `data/predictions/`. The new `xg_v3_synthetic` folder will be picked up automatically. ✅ no code change needed.

### Step 4.3 — Evaluation criteria

Track hit rate for **both** models across the validation window. Also track:
- Hits on picks where `xg_source == "real"` vs `"position_prior"` vs `"synthetic"` in the real-shot model
- Calibration: mean predicted probability vs observed rate, bucketed (reliability diagram)
- Coverage: what % of Tim Hortons predictions had real-shot data vs fell back to prior

**Promote real-shot to primary when:**
- Real-shot hit rate ≥ synthetic hit rate for 10+ game-days
- No catastrophic day (< 5% hit rate) that synthetic would have avoided
- Cold-start picks aren't systematically worse than warm picks (indicates priors need work)

**Decision point (yours):** how strict is "no catastrophic day"? One 0% day out of 14 is normal variance; four in a row is a signal. Write down your criteria before running — easier than deciding under pressure.

### Step 4.4 — Remove the synthetic fallback once promoted

Only after validation passes: delete `xg_predict_synthetic_only.py` and the `data/predictions/xg_v3_synthetic/` folder. Keep the synthetic *fallback* inside `calculate_xg_with_model` — it's load-bearing for cold-start players with no position prior.

---

## Phase 5 (optional) — Retrain the xG model on fuller data

The current model was trained on 7,942 shots ([metadata.json](../data/xg_model/metadata.json)). The CSV is at 35,357. A retrain should improve CV AUC modestly and stabilize feature importances.

### Step 5.1 — Inspect the existing trainer

```bash
ls scripts/train_xg_model.py
```

Read it. Confirm it reads `data/xg_training/shots.csv` end-to-end and does proper CV (temporal split, not random — shots from later games should not leak into training folds for earlier predictions).

### Step 5.2 — Retrain, compare, replace

```bash
python3 scripts/train_xg_model.py
```

Compare new `metadata.json` CV AUC to 0.9293. If it's **within ±0.005**, accept. If it drops meaningfully, investigate — schema drift, outlier-game contamination, or overfitting to late-season shot types.

### Step 5.3 — Feature importance sanity check

If features that were ~0% important (`strength_state_3v4`, `shot_type_poke`, `seconds_since_last_event`) jump to high importance, that's a red flag. It usually means a data quality issue: garbage features gaining signal in noise.

---

## Phase 6 — Maintenance

### 6.1 Monitoring

Add to `data/xg_training/collection_log.json` daily health check:
- Alert if `last_updated` is > 36 hours ago
- Alert if daily game count drops below 1 during season

Cheapest way: a post-collection step that writes a status file + uses Telegram (you already have [telegram_notify.py](../scripts/telegram_notify.py)) to ping on anomalies.

### 6.2 Schema drift

Every October, spot-check one collector run for:
- New shot types appearing in the CSV (the one-hot encoding will silently drop unknown types)
- `situationCode` format change (the parser at [collect_xg_data.py:105](../scripts/collect_xg_data.py#L105) assumes 4 digits)
- Missing `xCoord`/`yCoord` fields (the `or 0` default at [collect_xg_data.py:174](../scripts/collect_xg_data.py#L174) silently produces a dead-center shot — bad)

### 6.3 Cold-start coverage

Monthly, log:
- % of predictions served from real shots vs position prior
- Rookies called up mid-season → first-time predictions should prefer the prior, then transition to real shots once they have ≥20 shots

---

## Gotchas list (learned the hard way — read before starting)

1. **The 7,942 → 35,357 growth means retraining changes behavior.** Don't retrain in the same PR as Phase 3. Land the real-shot swap with the current model first, validate it, then retrain as a separate change.

2. **Rebounds across periods.** [collect_xg_data.py:250](../scripts/collect_xg_data.py#L250) treats inter-period shots as 30 sec apart, which is fine, but double-check the `is_rebound` logic isn't triggering across period breaks for your validation set.

3. **Empty-net shots skew averages.** A player who potted one empty-netter on 4/15 will have that 0.98-xG shot in their "recent 60." Consider excluding `is_empty_net == 1` from the recent-shot bucket — or leave it in and let the model's own handling (it has `is_empty_net` as a feature) work. Debate with yourself; I'd leave it in for v1.

4. **Traded players have mixed team context.** If Rantanen was traded mid-season, his shot history carries two teams. The xG model doesn't care (team isn't a feature) but recent-form features in MC and linear models do. Out of scope here, but note for future line-combo upgrade.

5. **Playoff shots look different.** Tighter checking, lower shot volumes, different shot type mix. Phase 4's validation will live through that — set expectations accordingly. The upgrade won't magically restore regular-season hit rates during playoffs.

6. **Cache invalidation on `_position_priors.json`.** The module-level `_POSITION_PRIORS` in `xg_predict.py` is loaded once per script run, which is correct — but if you refactor to a long-running service, remember to reload on file change.

7. **Player ID mismatches.** The shots CSV uses `scoringPlayerId` for goals but `shootingPlayerId` for other shots ([collect_xg_data.py:191-194](../scripts/collect_xg_data.py#L191)). For the aggregator this is fine (both are the shooter), but verify in spot-checks.

---

## Verification checklist

Copy this into a tracking issue and tick as you go:

- [ ] Phase 1.1 — audited coverage, know the gap
- [ ] Phase 1.2 — backfilled to yesterday
- [ ] Phase 1.3 — collector in daily workflow, fails loudly
- [ ] Phase 2.1 — documented the three decisions (N shots, cold start, decay)
- [ ] Phase 2.2 — aggregator script exists, produces per-player JSON
- [ ] Phase 2.3 — position priors file exists
- [ ] Phase 2.4 — aggregator in daily workflow
- [ ] Phase 3.1 — xg_predict.py loads real shots with fallback
- [ ] Phase 3.2 — output JSON carries `xg_source` field
- [ ] Phase 4.1 — synthetic-only shadow run exists
- [ ] Phase 4.3 — 10+ days of A/B data collected
- [ ] Phase 4.4 — real-shot promoted, shadow removed
- [ ] Phase 5 (optional) — model retrained, AUC didn't regress
- [ ] Phase 6 — monitoring ping + schema drift check scheduled

---

## Expected timeline with honesty

If you do this evenings + weekends: **3-4 weeks end to end**, most of that being Phase 4 validation running in the background while you do nothing.

If you dedicate a weekend to it: **Phases 1-3 in ~2 days**, then passively wait 2 weeks for Phase 4 data, then a Sunday afternoon for Phase 5.

The one step that tends to take longer than expected is **Phase 2 decisions** — it's tempting to add smoothing / decay / fancy priors before you've shown the naive version works. Resist. Ship naive, measure, then add complexity only where the data says you should.
