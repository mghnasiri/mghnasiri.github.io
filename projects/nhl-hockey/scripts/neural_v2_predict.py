"""
NHL Goal Predictor — Neural v2 (MLP + player embeddings).

Uses the same real-shot ingestion pipeline as xg_v3:
  collect_xg_data.py     -> data/xg_training/shots.csv
  aggregate_player_shots -> data/player_shots/{pid}.json
  neural_v2_predict.py   -> data/predictions/neural_v2/

For each tonight's Tim Hortons player we read their recent shots, score
each through the trained PyTorch MLP (with the player's learned embedding
when known), then convert mean xG per shot into P(>=1 goal) the same way
xg_v3 does. Falls back to a neutral player embedding (index 0) for
players not seen during training (rookies, call-ups).

Depends on:
  data/neural_v2_model/model.pt           (weights)
  data/neural_v2_model/metadata.json      (player map + feature stats)
  data/player_shots/{player_id}.json      (real recent shots)
  data/player_shots/_position_priors.json (cold-start fallback)

Scheduled in nhl_xg_v3_update.yml alongside the other xG variants.

Author: Mohammad G. Nasiri
"""

import json
import math
import os
import sys
from datetime import datetime

try:
    import requests
    import numpy as np
    import torch
    from torch import nn
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install torch numpy requests")
    sys.exit(1)

from nhl_api import api_get
from season_prior import pad_game_log, shrink_team_rate
from tims_match import match_tims
from shot_calibration import ATTEMPTS_PER_SOG, undo_class_weight


# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    MODEL_NAME = "neural_v2"
    MODEL_DISPLAY_NAME = "Neural Embed v2"

    DATA_DIR = "data"
    PREDICTIONS_DIR = f"{DATA_DIR}/predictions/{MODEL_NAME}"
    TIMS_DIR = f"{DATA_DIR}/tims_players"

    NEURAL_DIR = f"{DATA_DIR}/neural_v2_model"
    MODEL_FILE = f"{NEURAL_DIR}/model.pt"
    META_FILE = f"{NEURAL_DIR}/metadata.json"

    PLAYER_SHOTS_DIR = f"{DATA_DIR}/player_shots"
    POSITION_PRIORS_FILE = f"{PLAYER_SHOTS_DIR}/_position_priors.json"

    TODAY = datetime.now().strftime("%Y-%m-%d")
    PLAYOFFS = False  # set from today's schedule

    LEAGUE_AVG_GOALS = 3.07
    HOME_ADVANTAGE = 1.026
    MIN_GAMES_PLAYED = 3

    # Architecture (must match train_neural_v2.py Config)
    EMB_DIM = 32
    HIDDEN = 128
    NUM_RES_BLOCKS = 3
    DROPOUT = 0.0  # dropout off at inference


os.makedirs(Config.PREDICTIONS_DIR, exist_ok=True)


def current_season_id(date=None):
    d = date or datetime.now()
    if isinstance(d, str):
        d = datetime.strptime(d, "%Y-%m-%d")
    start = d.year if d.month >= 8 else d.year - 1
    return f"{start}{start + 1}"


# =============================================================================
# MODEL (must match train_neural_v2.py)
# =============================================================================
class NeuralV2(nn.Module):
    def __init__(self, num_players, num_feats,
                 emb_dim=Config.EMB_DIM, hidden=Config.HIDDEN,
                 num_blocks=Config.NUM_RES_BLOCKS, dropout=Config.DROPOUT):
        super().__init__()
        self.player_emb = nn.Embedding(num_players + 1, emb_dim)
        self.shot_proj = nn.Linear(num_feats, hidden)
        self.player_proj = nn.Linear(emb_dim, hidden)
        self.act = nn.ReLU()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
            )
            for _ in range(num_blocks)
        ])
        self.head = nn.Linear(hidden, 1)

    def forward(self, feats, player_idx):
        p = self.player_proj(self.player_emb(player_idx))
        s = self.shot_proj(feats)
        x = self.act(p + s)
        for block in self.blocks:
            x = x + block(x)
        return self.head(x).squeeze(-1)


# =============================================================================
# NHL API (subset needed for tonight's matchups)
# =============================================================================
def get_todays_games(date):
    """None if the schedule couldn't be fetched (distinct from [] = no games)."""
    # Patient retries (~30s): a failure now stops the run.
    data = api_get(f"https://api-web.nhle.com/v1/schedule/{date}", attempts=6)
    if not data:
        return None
    games = []
    for day in data.get("gameWeek", []):
        if day["date"] != date:
            continue
        for g in day.get("games", []):
            if g.get("gameType") in [2, 3]:
                games.append({
                    "game_id": g["id"],
                    "home_team": g["homeTeam"]["abbrev"],
                    "away_team": g["awayTeam"]["abbrev"],
                    "start_time": g.get("startTimeUTC", ""),
                    "game_type": g["gameType"],
                })
    return games


def get_team_stats():
    data = api_get("https://api-web.nhle.com/v1/standings/now")
    if not data:
        return {}
    stats = {}
    for t in data.get("standings", []):
        ab = t.get("teamAbbrev", {}).get("default", "")
        gp = t.get("gamesPlayed", 0) or 0
        stats[ab] = {
            "gf_per_game": round(shrink_team_rate(t.get("goalFor", 0), gp, Config.LEAGUE_AVG_GOALS), 3),
            "ga_per_game": round(shrink_team_rate(t.get("goalAgainst", 0), gp, Config.LEAGUE_AVG_GOALS), 3),
        }
    return stats


def get_team_roster(team):
    data = api_get(f"https://api-web.nhle.com/v1/roster/{team}/current")
    if not data:
        return []
    out = []
    for group in ("forwards", "defensemen"):
        for p in data.get(group, []):
            out.append({
                "player_id": p["id"],
                "name": f"{p['firstName']['default']} {p['lastName']['default']}",
                "position": p["positionCode"],
                "team": team,
            })
    return out


def get_player_stats(player_id):
    """Raises ConnectionError when a game log doesn't load (players without
    games get 200 + an empty log), so a fetch failure can't pass as a
    player who doesn't qualify."""
    season = current_season_id()
    reg = api_get(f"https://api-web.nhle.com/v1/player/{player_id}/game-log/{season}/2")
    # No playoff games exist this season until the playoffs start.
    po = (api_get(f"https://api-web.nhle.com/v1/player/{player_id}/game-log/{season}/3")
          if Config.PLAYOFFS else {})
    if reg is None or po is None:
        raise ConnectionError(f"game log for player {player_id} did not load")
    gl = reg.get("gameLog", []) + po.get("gameLog", [])
    gl.sort(key=lambda g: g.get("gameDate", ""), reverse=True)
    prior = [g for g in gl if g.get("gameDate", "9999") < Config.TODAY]
    # Early season: pad with last season's per-game rates (season_prior.py)
    prior = pad_game_log(prior, player_id, season)
    gp = len(prior)
    if gp < Config.MIN_GAMES_PLAYED:
        return None
    goals = sum(g.get("goals", 0) for g in prior)
    shots = sum(g.get("shots", 0) for g in prior)
    return {
        "games_played": gp,
        "season_goals": goals,
        "season_shots": shots,
        "avg_shots": round(shots / gp, 2) if gp > 0 else 0,
    }


# =============================================================================
# TIM HORTONS FILTERING (reused pattern)
# =============================================================================
def load_tims_players(date):
    path = f"{Config.TIMS_DIR}/{date}.json"
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def filter_tims(all_players, tims_data):
    """Tim Hortons eligible players, matched by NHL player_id (tims_match.py)."""
    if not tims_data or "groups" not in tims_data:
        return all_players, {}
    return match_tims(all_players, tims_data, Config.MODEL_NAME)


# =============================================================================
# REAL-SHOT LOADING (mirrors xg_predict.py)
# =============================================================================
_PRIORS_CACHE = {"loaded": False, "data": {}}


def load_real_shots(player_id, position):
    if player_id:
        path = f"{Config.PLAYER_SHOTS_DIR}/{player_id}.json"
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f).get("shots"), "real"
            except Exception:
                pass
    if not _PRIORS_CACHE["loaded"]:
        _PRIORS_CACHE["loaded"] = True
        if os.path.exists(Config.POSITION_PRIORS_FILE):
            try:
                with open(Config.POSITION_PRIORS_FILE, "r", encoding="utf-8") as f:
                    _PRIORS_CACHE["data"] = json.load(f)
            except Exception:
                pass
    pos_key = (position or "C")[0].upper()
    if pos_key not in ("C", "L", "R", "D"):
        pos_key = "C"
    prior = _PRIORS_CACHE["data"].get(pos_key)
    return (prior.get("shots") if prior else None), "position_prior"


# =============================================================================
# SCORING
# =============================================================================
def score_player_shots(model, shots, feature_names, feat_mean, feat_std,
                       player_idx, device, pos_weight):
    """Return (mean xG per shot, mean raw model output) for a player, or
    (None, None) if we can't score. The raw output is inflated by the
    training loss's pos_weight; the xG has it undone."""
    if not shots:
        return None, None
    X = np.array(
        [[float(s.get(f, 0) or 0) for f in feature_names] for s in shots],
        dtype=np.float32,
    )
    # Zero-variance training columns become exactly 0 (what every training
    # row had). Otherwise a feature that was constant in training — e.g.
    # seconds_since_last_event before the collector fix — passes raw values
    # into never-trained weights and adds noise.
    X = np.where(feat_std > 0, (X - feat_mean) / np.where(feat_std > 0, feat_std, 1.0), 0.0)
    x = torch.from_numpy(X.astype(np.float32)).to(device)
    p = torch.full((len(shots),), int(player_idx), dtype=torch.long, device=device)
    model.train(False)
    with torch.no_grad():
        logits = model(x, p)
        probs = torch.sigmoid(logits).cpu().numpy()
    return float(undo_class_weight(probs, pos_weight).mean()), float(probs.mean())


def goal_probability_from_xg(avg_shot_xg, raw_shot_xg, player, team_stats,
                             attempts_per_sog):
    """Same opponent + home/away adjustment as xg_v3 so the two models
    differ ONLY in how they score shot quality. Returns (probability,
    raw probability, player_xg, expected_shots); the raw probability is the
    pre-calibration scale Meta was trained on (meta_predict.py)."""
    base_shots = player.get("avg_shots", 0) or 0
    opp_ga = Config.LEAGUE_AVG_GOALS
    opp = player.get("opponent", "")
    if opp in team_stats:
        opp_ga = team_stats[opp]["ga_per_game"]
    opp_factor = opp_ga / Config.LEAGUE_AVG_GOALS
    ha_factor = (
        Config.HOME_ADVANTAGE if player.get("is_home")
        else (2.0 - Config.HOME_ADVANTAGE)
    )
    expected_shots = base_shots * opp_factor * ha_factor
    # xG is per unblocked attempt; expected_shots counts shots on goal.
    player_xg = avg_shot_xg * expected_shots * attempts_per_sog
    return (
        1.0 - math.exp(-player_xg),
        1.0 - math.exp(-raw_shot_xg * expected_shots),
        round(player_xg, 4),
        round(expected_shots, 2),
    )


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print(f"  NHL GOAL PREDICTOR — {Config.MODEL_DISPLAY_NAME}")
    print(f"  {Config.TODAY}")
    print("=" * 70)

    if not (os.path.exists(Config.MODEL_FILE) and os.path.exists(Config.META_FILE)):
        print(f"  Model not found at {Config.MODEL_FILE} — run train_neural_v2.py first")
        return 0
    with open(Config.META_FILE, "r") as f:
        meta = json.load(f)

    feature_names = meta["feature_names"]
    feat_mean = np.array(meta["feature_mean"], dtype=np.float32)
    feat_std = np.array(meta["feature_std"], dtype=np.float32)
    player_map = {int(k): v for k, v in meta["player_map"].items()}
    num_players = meta["num_players"]
    pos_weight = meta.get("pos_weight")
    attempts_per_sog = ATTEMPTS_PER_SOG
    if pos_weight is None:
        # Inflated outputs x 1.4 would only run hotter: keep the old scale.
        print("  WARNING: model predates pos_weight in metadata; probabilities "
              "stay uncalibrated until the next retrain")
        pos_weight = attempts_per_sog = 1.0

    # CPU inference: lowest-common-denominator across laptop + CI
    device = torch.device("cpu")
    model = NeuralV2(num_players=num_players, num_feats=len(feature_names)).to(device)
    model.load_state_dict(torch.load(Config.MODEL_FILE, map_location=device))
    model.train(False)

    # --- tonight's games ---
    print("\n  Fetching today's games...")
    games = get_todays_games(Config.TODAY)
    if games is None:
        # Don't write an empty "no games" file the health check would accept.
        print("  Could not fetch today's NHL schedule — not writing predictions.")
        return 1
    Config.PLAYOFFS = any(g["game_type"] == 3 for g in games)
    if not games:
        print("  No games today.")
        empty = {
            "date": Config.TODAY,
            "model": Config.MODEL_NAME,
            "model_display_name": Config.MODEL_DISPLAY_NAME,
            "games_count": 0, "games": [], "players_count": 0,
            "predictions": [], "tims_mode": False,
            "generated_at": datetime.now().isoformat(),
        }
        for path in (f"{Config.PREDICTIONS_DIR}/{Config.TODAY}.json",
                     f"{Config.PREDICTIONS_DIR}/latest.json"):
            with open(path, "w") as f:
                json.dump(empty, f, indent=2)
        return 0

    print(f"  {len(games)} games")
    team_stats = get_team_stats()

    # Tim Hortons pool
    tims = load_tims_players(Config.TODAY)

    # Build matchups
    all_teams = set()
    matchups = {}
    for g in games:
        all_teams.add(g["home_team"]); all_teams.add(g["away_team"])
        matchups[g["home_team"]] = {
            "opponent": g["away_team"], "is_home": True, "game_id": g["game_id"]
        }
        matchups[g["away_team"]] = {
            "opponent": g["home_team"], "is_home": False, "game_id": g["game_id"]
        }

    # Reuse the cache xg_predict.py wrote earlier in the same workflow run.
    # Without this, neural_v2 independently re-fetches the same ~150 players,
    # and at this point in the pipeline (4th in sequence) the NHL API is
    # already throttled — yesterday's neural_v2 produced 0 predictions.
    cache_dir = f"{Config.DATA_DIR}/cache"
    cache_file = f"{cache_dir}/players_{Config.TODAY}.json"
    all_players = None
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("date") == Config.TODAY and cached.get("players"):
                all_players = cached["players"]
                print(f"\n  Loaded {len(all_players)} players from cache "
                      f"(written by xg_predict.py)")
        except Exception as e:
            print(f"\n  Cache read failed ({e}); refetching")
            all_players = None

    if all_players is None:
        print("\n  Fetching rosters + stats (no cache available)...")
        all_players = []
        failed = []
        for team in sorted(all_teams):
            print(f"    {team}...", end=" ", flush=True)
            roster = get_team_roster(team)
            if not roster:
                failed.append(f"{team} roster")
            team_players = []
            for p in roster:
                try:
                    stats = get_player_stats(p["player_id"])
                except ConnectionError:
                    failed.append(p["name"])
                    continue
                if not stats:
                    continue
                p.update(stats)
                mu = matchups.get(team, {})
                p["opponent"] = mu.get("opponent", "")
                p["is_home"] = mu.get("is_home", False)
                p["game_id"] = mu.get("game_id", "")
                p["matchup"] = (
                    f"{team} vs {p['opponent']}" if p["is_home"]
                    else f"{team} @ {p['opponent']}"
                )
                team_players.append(p)
            all_players.extend(team_players)
            print(f"{len(team_players)}")
        # A partial slate can hide a group's best pick: fail loudly instead.
        if failed:
            print(f"  Could not load {failed} — not writing predictions.")
            return 1

    # Filter Tim Hortons
    tims_player_ids = None
    if tims:
        filtered, _ = filter_tims(all_players, tims)
        if filtered:
            tims_player_ids = {p["player_id"] for p in filtered}

    print("\n  Scoring shots through Neural v2...")
    cold_count = 0
    real_count = 0
    for p in all_players:
        shots, src = load_real_shots(p["player_id"], p.get("position"))
        player_idx = player_map.get(p["player_id"], 0)
        avg_xg = raw_xg = None
        if shots:
            avg_xg, raw_xg = score_player_shots(
                model, shots, feature_names, feat_mean, feat_std,
                player_idx, device, pos_weight,
            )
        p["_avg_shot_xg"] = avg_xg if avg_xg is not None else 0.066
        p["_raw_shot_xg"] = raw_xg if raw_xg is not None else 0.066
        p["_xg_source"] = src if shots else "none"
        p["_known_player"] = player_idx != 0
        if p["_xg_source"] == "real":
            real_count += 1
        elif p["_xg_source"] == "position_prior":
            cold_count += 1

    print(f"    real={real_count}  position_prior={cold_count}  "
          f"known_players_in_emb={sum(1 for q in all_players if q['_known_player'])}")

    # Convert to goal probability with opponent + home/away adjustment
    for p in all_players:
        prob, raw_prob, pxg, esh = goal_probability_from_xg(
            p["_avg_shot_xg"], p["_raw_shot_xg"], p, team_stats, attempts_per_sog
        )
        p["goal_probability"] = round(prob, 4)
        p["raw_goal_probability"] = round(raw_prob, 4)
        p["player_xg"] = pxg
        p["expected_shots"] = esh
        p["avg_shot_xg"] = round(p["_avg_shot_xg"], 4)
        p["xg_source"] = p["_xg_source"]
        p["known_in_embedding"] = p["_known_player"]
        for k in ("_avg_shot_xg", "_raw_shot_xg", "_xg_source", "_known_player"):
            del p[k]

    all_players.sort(key=lambda q: q["goal_probability"], reverse=True)
    for i, p in enumerate(all_players):
        p["rank"] = i + 1

    output_players = all_players
    if tims_player_ids:
        output_players = [q for q in all_players if q["player_id"] in tims_player_ids]
        for i, q in enumerate(output_players):
            q["tims_rank"] = i + 1

    tims_group_rankings = {}
    if tims_player_ids:
        for q in output_players:
            gid = q.get("tims_group")
            if not gid:
                continue
            bucket = tims_group_rankings.setdefault(gid, [])
            bucket.append({
                "rank_in_group": len(bucket) + 1,
                "player_id": q["player_id"],
                "name": q["name"],
                "team": q["team"],
                "goal_probability": q["goal_probability"],
                "matchup": q.get("matchup", ""),
            })

    out = {
        "date": Config.TODAY,
        "model": Config.MODEL_NAME,
        "model_display_name": Config.MODEL_DISPLAY_NAME,
        "games_count": len(games),
        "games": games,
        "players_count": len(output_players),
        "predictions": output_players,
        "tims_mode": bool(tims_player_ids),
        "tims_source": tims.get("source") if tims else None,
        "tims_group_rankings": tims_group_rankings,
        "model_params": {
            "val_auc": meta.get("cv_val_auc"),
            "num_params": meta.get("num_players", 0) * Config.EMB_DIM,
            "num_train_shots": meta.get("num_train"),
            "pos_weight_undone": pos_weight,
            "attempts_per_sog": attempts_per_sog,
        },
        "generated_at": datetime.now().isoformat(),
    }
    for path in (f"{Config.PREDICTIONS_DIR}/{Config.TODAY}.json",
                 f"{Config.PREDICTIONS_DIR}/latest.json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {Config.PREDICTIONS_DIR}/{Config.TODAY}.json")

    if tims_group_rankings:
        print("\n  Top Neural v2 Tim Hortons picks by group:")
        for gid in sorted(tims_group_rankings.keys()):
            print(f"    Group {gid}:")
            for p in tims_group_rankings[gid][:5]:
                print(f"      {p['rank_in_group']}  {p['name']:<22} {p['team']:<4} "
                      f"{p['goal_probability']*100:5.1f}%")
    print("\n  Neural v2 predictions complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
