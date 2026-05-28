# Line & TOI Projection Upgrade — Step 10 Guide

**Goal:** Weight each player's shot projection by **tonight's expected role** —
confirmed lineup, line slot, and power-play unit — instead of only their
trailing time-on-ice. Expected lift: **+3 to +7 points of Top-N hit rate**,
concentrated on nights with unusual rotations (post-trade, injury return,
call-ups, blowout aftermath).

**Status when this guide was written (2026-05-28):**
- `lineup_v1` (`xg_predict.py --lineup-adjusted`) already scales
  `expected_shots` by `recent_toi / season_toi` (last-5-game avg vs season avg),
  clamped to `[0.05, 1.5]` — [xg_predict.py:669-686](../scripts/xg_predict.py#L669).
  It's a **champion (~24% hit rate)**. ✅ Trailing-TOI scaling exists.
- What it **cannot** do, and this upgrade adds:
  1. **Tonight's scratch.** A healthy scratch announced 30-60 min pre-puck still
     carries a normal last-5 TOI, so lineup_v1 keeps projecting shots for a
     player who won't dress. This is the single biggest miss.
  2. **Tonight's line promotion/demotion.** A 4th-liner bumped to the top line
     (injury fill-in) gets the same projection as last week.
  3. **PP unit membership.** PP goals are ~30% of all goals; lineup_v1 uses
     *total* TOI, so it can't tell PP1 (3-4 elite min) from no-PP.

**This guide does NOT cover** linemate-quality modeling (a secondary model —
defer to a v2) or shift-level simulation (out of scope).

---

## Architecture

```
fetch_projected_lines.py            (NEW — runs in daily workflow, pre-predictions)
  └─ scrape tonight's projected lines for each team
  └─ sanity-gate the result (min teams/players; reject partial scrapes)
  └─ write data/projected_lines/{date}.json:
       per team -> { scratches:[pid], lines:{L1..L4:[pid]},
                     pp_units:{PP1:[pid], PP2:[pid]}, source, scraped_at }

xg_predict.py  (--lineup-adjusted, i.e. lineup_v1 → lineup_v2)
  └─ load data/projected_lines/{today}.json
  └─ per player:
       scratched         -> expected_shots = 0   (drop from picks entirely)
       line slot (L1..L4)-> line_factor          (top line up, 4th line down)
       PP unit (PP1/PP2) -> pp_factor            (folds into existing PP logic)
  └─ these multiply the EXISTING toi_factor (decision C below governs the blend)
  └─ cold start (no entry, e.g. call-up): fall back to today's toi_factor
```

Keep the current `lineup_v1` running unchanged and add a **`lineup_v2`** variant
(another `xg_predict.py` flag, like the synthetic/lineup variants already do at
[xg_predict.py:56-70](../scripts/xg_predict.py#L56)) so you can A/B them.

---

## Data sources (pick one — all are fragile)

| Source | Coverage | Form | Risk |
|---|---|---|---|
| **Daily Faceoff** projected/confirmed lines | Lines + PP units + scratches, all teams | HTML scrape (no API) | Layout changes; the make-or-break dependency |
| **NHL API** roster + pregame `gamecenter` | Official scratches sometimes appear pregame | JSON | No projected *lines* or *PP units* |
| Beat-writer morning-skate posts | Most accurate | Unstructured text | Not machine-parseable at scale |

**Recommended:** Daily Faceoff as primary, cross-checked against the NHL roster
API for player-id resolution. **Mandatory:** a freshness + size sanity gate
(mirror the Tims-scraper gate at [fetch_tims_players.py](../scripts/fetch_tims_players.py))
— a silently-partial scrape must NOT overwrite a good cache, or you poison the
pool exactly like the Tims bug did.

---

## Phased plan

- **Phase 1 — Prove a source.** Manually scrape one night, confirm you can map
  names→player_ids and get scratches+lines+PP for ≥90% of dressed skaters.
  *This phase is make-or-break; if the source is too fragile, stop here.*
- **Phase 2 — Build the fetcher.** `fetch_projected_lines.py` + the JSON schema
  above + the sanity gate. Add to the daily workflow **before** the prediction
  scripts. Stamp `source` and `scraped_at`.
- **Phase 3 — Scratch zeroing only** (biggest single win, smallest risk). In the
  lineup-adjusted path, set `expected_shots = 0` for scratched players. Ship
  `lineup_v2` with *only* this and A/B vs `lineup_v1`.
- **Phase 4 — Line + PP weighting.** Add `line_factor` and `pp_factor` once
  scratch-zeroing is validated.
- **Phase 5 — Validate & promote.** 2+ weeks A/B; promote `lineup_v2` when it
  beats `lineup_v1` on Top-N hit rate, especially on high-rotation nights.
- **Phase 6 — Late refresh + maintenance.** See decision C.

---

## Decision points (yours — these shape the feature)

**A. Scratch handling.** Hard-zero (player drops out) vs heavy-discount?
*Recommend hard-zero* — a scratch literally cannot score, and a false-positive
scratch only costs you one pick.

**B. PP-unit weight.** How much to boost PP1? PP1 elite shooters get ~3-4 min of
high-danger time; this is a large share of their scoring. Start with a modest
multiplier (e.g. PP1 ×1.15, PP2 ×1.05, none ×1.0) and **calibrate against
realized PP-goal rates** — don't guess and forget.

**C. Blend with the existing `toi_factor`.** The current `toi_factor` already
captures *trailing* usage. Projected role and trailing TOI overlap — do you
**replace** `toi_factor` with the projected signal, or **blend** them? Naive
stacking double-counts. *Recommend:* when a projected-line entry exists, let it
dominate (it's tonight's truth); fall back to `toi_factor` only on cold start.

**D. Late refresh.** Scratches drop 30-60 min before puck. Do you re-run
predictions on a late cron, or accept morning-skate lines and miss late changes?
Late refresh is the highest-accuracy option but adds workflow complexity.

---

## Gotchas (read before starting)

1. **Lines break mid-game** — only the *pre-game projected* lines matter for
   tonight; never derive tonight's lines from last game's in-game shifts.
2. **Source fragility** — without the sanity gate, a format change silently
   feeds an empty/partial lineup and tanks a night's picks. Non-negotiable.
3. **Playoffs differ** — lines are more stable, but injury scratches matter more
   and PP time concentrates further.
4. **Player-id resolution** — same name-collision risk as the market model
   ([market_predict.py](../scripts/market_predict.py)); disambiguate by team.
5. **Cold start** — call-ups won't have a projected-line entry on their first
   night; fall back to `toi_factor`, don't zero them.

---

## Validation criteria (write these down before running)

- `lineup_v2` Top-N hit rate ≥ `lineup_v1` over **10+ game-days**, and clearly
  higher on identified high-rotation nights.
- **Coverage:** % of picks that had a projected-line entry (target ≥90% on a
  full slate).
- **Sanity:** a confirmed scratch must NEVER appear in the top picks.

**Dependency:** do the real-shot xG **re-collect** (`collect_xg_data.py
--rebuild`) first — clean shot *quality* × clean projected *volume* is the point;
stacking projected volume on corrupted shot quality muddies the A/B.
