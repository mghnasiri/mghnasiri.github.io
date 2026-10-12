"""
Direct player-game goal model ("Direct LR"), ported from the 2026-10-12 study
(~/Projects/nhl-research/2026-10-12-model-study, verified by an independent
re-run). One logistic regression over every skater-game since 2021-22, with
features built only from games strictly before each game's date.

Design: build "state after game" tables for players, teams and goalies; every
query (player, date, team, opp, home, gameId) reads the latest state strictly
before the query date via merge_asof(allow_exact_matches=False). Training rows
and today's candidates go through the same featurize() path, so a player who
does not dress tonight gets exactly the features he would have had.

On 82 real Tims days it picked +7.8pp over random (22.8% vs 15.0%), tied
with xG v3; it is the one strong ranker that can be replayed leak-free over
five seasons, which is what lets later changes be tested offline.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# EW half-lives in player games / team games / goalie games
H_PLAYER = (5, 10, 40, 150)
H_TEAM = 20
H_GOALIE = 50
K_TOI_MIN = 120.0     # pseudo-minutes of TOI for shrinking per-60 rates
K_SHOTS = 100.0       # pseudo-shots for shooting %
K_GSA = 500.0         # pseudo shots-against for goalie save %
GOALIE_PRIOR = 0.900  # unknown/new goalies are shrunk here (below league avg ~.905)
TRAIN_FROM_SEASON = 2021


def season_of(dates):
    d = pd.to_datetime(dates)
    return np.where(d.dt.month >= 8, d.dt.year, d.dt.year - 1)


def prepare(skaters, goalies):
    """nhl_history.load() frames -> (sk, gl) in the model's column names."""
    sk = skaters.rename(columns={"playerId": "pid", "gameId": "gid", "gameDate": "date",
                                 "teamAbbrev": "team", "opponentTeamAbbrev": "opp",
                                 "positionCode": "pos"}).copy()
    sk["date"] = pd.to_datetime(sk["date"])
    for c in ["timeOnIce", "ppTimeOnIce", "evTimeOnIce", "shTimeOnIce"]:
        sk[c] = sk[c].fillna(0) / 60.0  # minutes
    sk["season"] = sk["gid"] // 1000000
    sk["playoff"] = ((sk["gid"] // 10000) % 100 == 3).astype(int)
    sk["home"] = (sk["homeRoad"] == "H").astype(int)
    sk["scored"] = (sk["goals"] > 0).astype(int)
    gl = goalies.rename(columns={"playerId": "gpid", "gameId": "gid", "gameDate": "date",
                                 "teamAbbrev": "team"}).copy()
    gl["date"] = pd.to_datetime(gl["date"])
    return sk, gl


def positional_priors(sk):
    """Per-position (F/D) rates computed ONLY from the 2019-21 history seasons (pre-training)."""
    h = sk[sk["season"] <= 2020].copy()
    h["F"] = (h["pos"] != "D")
    out = {}
    for f, g in h.groupby("F"):
        out[bool(f)] = dict(gpm=g.goals.sum() / g.timeOnIce.sum(), spm=g.shots.sum() / g.timeOnIce.sum(),
                            sh=g.goals.sum() / g.shots.sum(), toi=g.timeOnIce.median(),
                            pptoi=g.ppTimeOnIce.mean())
    return out


def _ew_by(df, key, cols, h):
    a = 1 - 0.5 ** (1.0 / h)
    m = df.groupby(key, sort=False)[cols].transform(lambda x: x.ewm(alpha=a, adjust=True).mean())
    n = df.groupby(key, sort=False).cumcount() + 1
    w = (1 - (1 - a) ** n) / a   # sum of weights -> decayed sums = mean * w
    return m, w


def build_states(sk, gl):
    sk = sk.sort_values(["pid", "date"]).reset_index(drop=True)
    # ---------- team-game table ----------
    tg = sk.groupby(["gid", "team"]).agg(date=("date", "first"), opp=("opp", "first"), gf=("goals", "sum"),
                                          sf=("shots", "sum"), pp_time=("ppTimeOnIce", "max"),
                                          pk_time=("shTimeOnIce", "max")).reset_index()
    o = tg[["gid", "team", "gf", "sf"]].rename(columns={"team": "opp", "gf": "ga", "sf": "sa"})
    tg = tg.merge(o, on=["gid", "opp"], how="left").sort_values(["team", "date"]).reset_index(drop=True)
    cols = ["gf", "ga", "sf", "sa", "pp_time", "pk_time"]
    m, _ = _ew_by(tg, "team", cols, H_TEAM)
    team_state = tg[["team", "date"]].copy()
    for c in cols:
        team_state["t_" + c] = m[c].values
    team_state["t_last_date"] = tg["date"].values
    # schedule (who plays whom, which game) for any date
    sched = tg[["gid", "date", "team", "opp"]].copy()

    # ---------- player state ----------
    sk = sk.merge(tg[["gid", "team", "pp_time"]].rename(columns={"pp_time": "team_pp"}), on=["gid", "team"], how="left")
    sk["one"] = 1.0
    st = sk[["pid", "date", "season", "pos"]].copy()
    base = ["goals", "shots", "timeOnIce", "ppTimeOnIce", "team_pp", "ppGoals"]
    for h in H_PLAYER:
        m, w = _ew_by(sk, "pid", base, h)
        for c in base:
            st[f"d{h}_{c}"] = (m[c] * w).values   # decayed sums
        st[f"d{h}_n"] = w.values
    st["career_gp"] = sk.groupby("pid").cumcount() + 1
    g = sk.groupby(["pid", "season"])
    st["s_gp"] = g.cumcount() + 1
    for c in ["goals", "shots", "timeOnIce", "ppTimeOnIce"]:
        st["s_" + c] = g[c].cumsum()
    st["last_toi"] = sk["timeOnIce"]
    st["last_date"] = sk["date"]
    st["last_team"] = sk["team"]
    # completed-season totals for "prior season" features
    seas = sk.groupby(["pid", "season"]).agg(p_gp=("one", "sum"), p_goals=("goals", "sum"),
                                              p_shots=("shots", "sum"), p_toi=("timeOnIce", "sum"),
                                              p_pptoi=("ppTimeOnIce", "sum")).reset_index()
    # ---------- goalie state ----------
    gl = gl.sort_values(["gpid", "date"]).reset_index(drop=True)
    gl["saves_"] = gl["shotsAgainst"] - gl["goalsAgainst"]
    m, w = _ew_by(gl, "gpid", ["saves_", "shotsAgainst"], H_GOALIE)
    gstate = gl[["gpid", "date"]].copy()
    gstate["g_sv"] = (m["saves_"] * w).values
    gstate["g_sa"] = (m["shotsAgainst"] * w).values
    gstate["g_gp"] = gl.groupby("gpid").cumcount() + 1
    starters = gl[gl["gamesStarted"] == 1][["gid", "team", "gpid"]].rename(columns={"team": "opp", "gpid": "opp_starter"})
    starters = starters.drop_duplicates(["gid", "opp"])
    return dict(player=st, seasons=seas, team=team_state, goalie=gstate, starters=starters, sched=sched,
                priors=positional_priors(sk))


def _asof(q, state, qkey, skey, cols):
    q = q.sort_values("date")
    s = state[[skey, "date"] + cols].sort_values("date")
    if qkey != skey:
        s = s.rename(columns={skey: qkey})
    return pd.merge_asof(q, s, on="date", by=qkey, allow_exact_matches=False)


def featurize(q, S):
    """q columns: qid, pid, pos, date, team, opp, home, gid, playoff. Returns feature frame indexed by qid."""
    q = q.copy()
    q["date"] = pd.to_datetime(q["date"])
    q["season"] = season_of(q["date"])
    pr = S["priors"]

    pcols = [c for c in S["player"].columns if c not in ("pid", "date", "pos")]
    st = S["player"].rename(columns={"season": "st_season"})
    pcols = [("st_season" if c == "season" else c) for c in pcols]
    x = _asof(q, st, "pid", "pid", pcols)
    x = x.sort_values("qid").reset_index(drop=True)
    q = q.sort_values("qid").reset_index(drop=True)
    isF = (q["pos"] != "D").values
    gpm = np.where(isF, pr[True]["gpm"], pr[False]["gpm"])
    spm = np.where(isF, pr[True]["spm"], pr[False]["spm"])
    shp = np.where(isF, pr[True]["sh"], pr[False]["sh"])
    toi0 = np.where(isF, pr[True]["toi"], pr[False]["toi"]) * 0.85  # unknowns are usually depth/debut players

    F = pd.DataFrame({"qid": q["qid"].values})
    dcols = [c for c in x.columns if c.startswith("d") and "_" in c and c[1:].split("_")[0].isdigit()]
    x[dcols] = x[dcols].fillna(0.0)
    for h in H_PLAYER:
        toi = x[f"d{h}_timeOnIce"]
        n = x[f"d{h}_n"]
        F[f"g60_h{h}"] = 60 * (x[f"d{h}_goals"] + K_TOI_MIN * gpm) / (toi + K_TOI_MIN)
        F[f"s60_h{h}"] = 60 * (x[f"d{h}_shots"] + K_TOI_MIN * spm) / (toi + K_TOI_MIN)
        F[f"toi_h{h}"] = (toi + 3 * toi0) / (n + 3)                       # minutes / game, shrunk by 3 games
        F[f"pptoi_h{h}"] = (x[f"d{h}_ppTimeOnIce"]) / (n + 3)
        F[f"ppshare_h{h}"] = x[f"d{h}_ppTimeOnIce"] / (x[f"d{h}_team_pp"] + 3.0)
        F[f"gpg_h{h}"] = (x[f"d{h}_goals"] + 3 * gpm * toi0) / (n + 3)
    F["shpct_reg"] = (x["d150_goals"] + K_SHOTS * shp) / (x["d150_shots"] + K_SHOTS)
    F["ppg_share_h150"] = (x["d150_ppGoals"] + 1) / (x["d150_goals"] + 4)
    same = (x["st_season"] == q["season"]).values
    F["season_gp"] = np.where(same, x["s_gp"], 0)
    F["season_gpg"] = (np.where(same, x["s_goals"], 0) + 5 * F["gpg_h150"]) / (F["season_gp"] + 5)
    F["season_spg"] = (np.where(same, x["s_shots"], 0) + 5 * F["s60_h150"] * F["toi_h150"] / 60) / (F["season_gp"] + 5)
    F["season_toi"] = (np.where(same, x["s_timeOnIce"], 0) + 5 * F["toi_h150"]) / (F["season_gp"] + 5)
    F["career_gp"] = x["career_gp"].fillna(0).clip(upper=400)
    F["last_toi"] = x["last_toi"].fillna(pd.Series(toi0, index=x.index))
    rest = (q["date"] - x["last_date"]).dt.days
    F["rest_days"] = rest.fillna(30).clip(upper=30)
    # prior season (completed)
    ps = S["seasons"].copy(); ps["season"] = ps["season"] + 1
    pq = q[["qid", "pid", "season"]].merge(ps, on=["pid", "season"], how="left").sort_values("qid")
    F["prev_gp"] = pq["p_gp"].fillna(0).values
    F["prev_gpg"] = ((pq["p_goals"].fillna(0) + 10 * gpm * toi0) / (pq["p_gp"].fillna(0) + 10)).values
    F["prev_toi"] = ((pq["p_toi"].fillna(0) + 10 * toi0) / (pq["p_gp"].fillna(0) + 10)).values
    # team / opponent context
    tcols = ["t_gf", "t_ga", "t_sf", "t_sa", "t_pp_time", "t_pk_time", "t_last_date"]
    tm = _asof(q[["qid", "date", "team"]], S["team"], "team", "team", tcols).sort_values("qid")
    op = _asof(q[["qid", "date", "opp"]], S["team"].rename(columns={"team": "opp"}), "opp", "opp", tcols).sort_values("qid")
    F["tm_gf"] = tm["t_gf"].fillna(3.0).values
    F["tm_sf"] = tm["t_sf"].fillna(30.0).values
    F["tm_pp"] = tm["t_pp_time"].fillna(5.0).values
    F["opp_ga"] = op["t_ga"].fillna(3.0).values
    F["opp_sa"] = op["t_sa"].fillna(30.0).values
    F["opp_pk"] = op["t_pk_time"].fillna(5.0).values
    F["opp_team_svp"] = 1 - (op["t_ga"].fillna(3.0) / op["t_sa"].fillna(30.0)).values
    F["tm_b2b"] = ((q["date"].values - tm["t_last_date"].values) == np.timedelta64(1, "D")).astype(int)
    F["opp_b2b"] = ((q["date"].values - op["t_last_date"].values) == np.timedelta64(1, "D")).astype(int)
    # opponent starting goalie (known pregame only when confirmed)
    sq = q[["qid", "gid", "opp", "date"]].merge(S["starters"], on=["gid", "opp"], how="left")
    gs = _asof(sq[["qid", "date", "opp_starter"]].dropna(subset=["opp_starter"]).astype({"opp_starter": "int64"}),
               S["goalie"], "opp_starter", "gpid", ["g_sv", "g_sa", "g_gp"])
    F = F.merge(gs[["qid", "g_sv", "g_sa", "g_gp"]], on="qid", how="left")
    F["opp_g_svp"] = (F["g_sv"].fillna(0) + K_GSA * GOALIE_PRIOR) / (F["g_sa"].fillna(0) + K_GSA)
    F["opp_g_gp"] = F["g_gp"].fillna(0).clip(upper=300)
    F = F.drop(columns=["g_sv", "g_sa", "g_gp"])
    F["home"] = q["home"].values
    F["playoff"] = q["playoff"].values
    F["is_D"] = (q["pos"] == "D").astype(int).values
    F["is_C"] = (q["pos"] == "C").astype(int).values
    # composite expectation: shots/game * shooting %
    F["xg_proxy"] = F["s60_h40"] * F["toi_h10"] / 60 * F["shpct_reg"]
    return F.set_index("qid")


FEATS_BASE = (
    [f"{k}_h{h}" for h in H_PLAYER for k in ("g60", "s60", "toi", "pptoi", "ppshare", "gpg")]
    + ["shpct_reg", "ppg_share_h150", "season_gp", "season_gpg", "season_spg", "season_toi", "career_gp",
       "last_toi", "rest_days", "prev_gp", "prev_gpg", "prev_toi", "tm_gf", "tm_sf", "tm_pp", "opp_ga",
       "opp_sa", "opp_pk", "opp_team_svp", "tm_b2b", "opp_b2b", "home", "playoff", "is_D", "is_C", "xg_proxy"])
FEATS_GOALIE = ["opp_g_svp", "opp_g_gp"]


# --- the model: logistic regression, no goalie feature (a confirmed starter
# isn't known at ~10am, and the study found the feature adds nothing) ---
LOG_FEATS = [c for c in FEATS_BASE if c.split("_")[0] in ("g60", "s60", "gpg")] + [
    "xg_proxy", "season_gpg", "season_spg", "prev_gpg", "shpct_reg"]


def lr_design(X):
    """Log-transform the rate and count features present in X (the backtest
    passes variant feature sets)."""
    Z = X.copy()
    for c in LOG_FEATS:
        if c in Z:
            Z[c] = np.log(Z[c].clip(lower=1e-4))
    if "rest_days" in Z:
        Z["rest_days"] = np.log(Z["rest_days"].clip(1, 30))
    for c in ("career_gp", "season_gp", "prev_gp", "opp_g_gp"):
        if c in Z:
            Z[c] = np.log1p(Z[c])
    return Z


def fit(sk, S, C=0.1):
    """Fit on every skater-game from TRAIN_FROM_SEASON in `sk` (all before
    the prediction date). Returns a fitted pipeline."""
    train = sk[sk.season >= TRAIN_FROM_SEASON][
        ["pid", "pos", "date", "team", "opp", "home", "gid", "playoff", "scored"]].reset_index(drop=True)
    train["qid"] = train.index
    X = featurize(train.drop(columns="scored"), S)
    y = train.set_index("qid").loc[X.index, "scored"]
    model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
    model.fit(lr_design(X[FEATS_BASE]), y)
    return model


def predict(model, X):
    return model.predict_proba(lr_design(X[FEATS_BASE]))[:, 1]
