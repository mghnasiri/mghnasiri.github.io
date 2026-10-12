"""
Offline backtest for changes to Direct LR (run by hand, not daily).

Live Tims days can't judge a change: a +3pp difference takes ~450+ days to
detect. This replays the direct model leak-free over whole seasons, where a
log-loss difference is measurable, and also reports the paired top pick per
group on the real Tims days. Adopt a candidate only if its log-loss gain's
90% CI excludes 0, it is positive in most test seasons, and it is not worse
on the real Tims days (the 2026-10-12 study's gate).

usage: python scripts/backtest_direct.py [--add f1,f2] [--drop f1,f2] [--C 0.1]
       The incumbent is direct_model.FEATS_BASE with C=0.1; the candidate is
       that list with --add/--drop applied (names from direct_model.featurize).
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import direct_model as dm
import nhl_history

TEST_SEASONS = (2023, 2024, 2025)
BOOT = 2000


def fit(X, y, cols, C):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
    return model.fit(dm.lr_design(X[cols]), y)


def proba(model, X, cols):
    return model.predict_proba(dm.lr_design(X[cols]))[:, 1]


def day_boot(by_day):
    """Mean of per-row values and its 90% CI, resampling days."""
    days = sorted(by_day)
    sums = np.array([by_day[d].sum() for d in days])
    counts = np.array([len(by_day[d]) for d in days])
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(days), size=(BOOT, len(days)))
    boots = np.sort(sums[idx].sum(1) / counts[idx].sum(1))
    return sums.sum() / counts.sum(), boots[int(0.05 * BOOT)], boots[int(0.95 * BOOT)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--add", default="")
    ap.add_argument("--drop", default="")
    ap.add_argument("--C", type=float, default=0.1)
    a = ap.parse_args()
    base = list(dm.FEATS_BASE)
    add = [f.strip() for f in a.add.split(",") if f.strip()]
    drop = [f.strip() for f in a.drop.split(",") if f.strip()]
    if [f for f in drop if f not in base]:
        sys.exit(f"--drop: not in the incumbent: {[f for f in drop if f not in base]}")
    # featurize fills these from the game's actual starter, which the 10am
    # predictor never knows: testing them here would leak.
    if [f for f in add if f in dm.FEATS_GOALIE]:
        sys.exit(f"{dm.FEATS_GOALIE} use the game's actual starting goalie (unknown at 10am)")
    cand = [f for f in base if f not in drop] + [f for f in add if f not in base]
    if cand == base and a.C == 0.1:
        sys.exit("the candidate is identical to the incumbent: nothing to compare")
    print(f"incumbent: {len(base)} features, C=0.1 | candidate: +{add} -{drop}, C={a.C}")

    until = datetime.now().date() - timedelta(days=1)
    sk, gl = dm.prepare(*nhl_history.load(until))
    S = dm.build_states(sk, gl)
    rows = sk[sk.season >= dm.TRAIN_FROM_SEASON][
        ["pid", "pos", "date", "team", "opp", "home", "gid", "playoff", "season", "scored"]
    ].reset_index(drop=True)
    rows["qid"] = rows.index
    X = dm.featurize(rows.drop(columns=["season", "scored"]), S)
    unknown = [f for f in cand if f not in X.columns]
    if unknown:
        sys.exit(f"unknown feature(s): {unknown}")
    rows = rows.set_index("qid").loc[X.index]

    # 1) Season walk-forward: train on earlier seasons, test on the next one.
    report = {"candidate": {"add": add, "drop": drop, "C": a.C}, "seasons": {}}
    diffs = {}
    models = {}
    for T in TEST_SEASONS + (dm.season_of(pd.Series([pd.Timestamp(until)]))[0],):
        tr = rows.season < T
        mi = fit(X[tr], rows.scored[tr], base, 0.1)
        mc = fit(X[tr], rows.scored[tr], cand, a.C)
        models[int(T)] = (mi, mc)
        te = rows.season == T
        if T not in TEST_SEASONS or not te.any():
            continue
        y = rows.scored[te].to_numpy()
        pi = np.clip(proba(mi, X[te], base), 1e-6, 1 - 1e-6)
        pc = np.clip(proba(mc, X[te], cand), 1e-6, 1 - 1e-6)
        ll = lambda p: -(y * np.log(p) + (1 - y) * np.log(1 - p))
        d = pd.Series(ll(pi) - ll(pc), index=rows.date[te].to_numpy())  # >0: candidate better
        by_day = {k: v.to_numpy() for k, v in d.groupby(level=0)}
        diffs.update({(T, k): v for k, v in by_day.items()})
        m, lo, hi = day_boot(by_day)
        report["seasons"][int(T)] = dict(n=int(te.sum()), logloss_gain=m, ci90=[lo, hi],
                                         auc_incumbent=roc_auc_score(y, pi),
                                         auc_candidate=roc_auc_score(y, pc))
        print(f"  season {T}: n={te.sum():,} log-loss gain {m:+.5f} (90% CI {lo:+.5f}..{hi:+.5f})"
              f"  AUC {roc_auc_score(y, pi):.4f} -> {roc_auc_score(y, pc):.4f}")
    m, lo, hi = day_boot(diffs)
    report["pooled_logloss_gain"] = [m, lo, hi]
    print(f"  pooled log-loss gain {m:+.5f} (90% CI {lo:+.5f}..{hi:+.5f})")

    # 2) Real Tims days: each day scored by the model trained before its season.
    hits = {}
    for path in sorted(glob.glob("data/tims_players/20*.json")):
        pool = json.load(open(path))
        date = pool.get("date") or os.path.basename(path)[:-5]
        res_path = f"data/results/{date}.json"
        if pool.get("source") != "timnhlassist.com" or not os.path.exists(res_path):
            continue
        res = json.load(open(res_path))
        if res.get("partial") or not res.get("games_count"):
            continue
        D = pd.Timestamp(date)
        T = int(dm.season_of(pd.Series([D]))[0])
        if T not in models:
            continue
        scorers = {s["player_id"] for s in res.get("all_scorers", []) if s.get("player_id")}
        games = S["sched"][S["sched"].date == D].set_index("team")
        home = sk[sk.date == D].groupby("team").home.max()
        last_pos = sk[sk.date < D].sort_values("date").groupby("pid").pos.last()
        q = []
        for gid, members in pool["groups"].items():
            for e in members:
                if isinstance(e, dict) and e.get("player_id") and e.get("team") in games.index:
                    g = games.loc[e["team"]]
                    q.append(dict(group=gid, pid=int(e["player_id"]), team=e["team"], date=D,
                                  pos=last_pos.get(int(e["player_id"]), "C"), opp=g.opp,
                                  home=int(home.get(e["team"], 0)),
                                  gid=int(g.gid), playoff=int((g.gid // 10000) % 100 == 3)))
        if not q:
            continue
        Q = pd.DataFrame(q)
        Q["qid"] = np.arange(len(Q))
        F = dm.featurize(Q[["qid", "pid", "pos", "date", "team", "opp", "home", "gid", "playoff"]], S)
        mi, mc = models[T]
        Q["pi"], Q["pc"] = proba(mi, F.loc[Q.qid], base), proba(mc, F.loc[Q.qid], cand)
        for gid, grp in Q.groupby("group"):
            members = {e["player_id"] for e in pool["groups"][gid] if isinstance(e, dict)}
            rnd = sum(p in scorers for p in members) / len(members)
            hi_i = int(grp.loc[grp.pi.idxmax(), "pid"] in scorers)
            hi_c = int(grp.loc[grp.pc.idxmax(), "pid"] in scorers)
            hits.setdefault(date, []).append((hi_i - rnd, hi_c - rnd, hi_c - hi_i))
    if hits:
        for k, name in enumerate(("incumbent lift vs random", "candidate lift vs random",
                                  "candidate - incumbent")):
            m, lo, hi = day_boot({d: np.array([r[k] for r in v]) for d, v in hits.items()})
            report[name] = [m, lo, hi]
            print(f"  Tims days ({len(hits)}): {name} {m * 100:+.1f}pp (90% CI {lo * 100:+.1f}..{hi * 100:+.1f})")
    os.makedirs("data/backtest", exist_ok=True)
    out = f"data/backtest/direct_{datetime.now():%Y-%m-%d_%H%M}.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
