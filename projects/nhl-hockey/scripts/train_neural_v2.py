"""
Train Neural v2 — deep MLP with per-player embeddings for xG prediction.

One-shot training on data/xg_training/shots.csv. Designed so a single CPU
run (~30-60 min on GitHub Actions ubuntu-latest) produces a deployable
model — no hyperparameter sweep, no CV, no seed averaging. Defaults chosen
to be robust rather than optimal: Adam, lr=3e-4, batch 256, dropout 0.1,
early stop on plateaued val loss.

The architecture difference vs xg_v3 (XGBoost) that makes this worth
running: a learned per-player embedding. XGBoost can't represent
"Pastrnak is a different shooter than Svechnikov independent of the
per-shot features." A 32-dim embedding vector per player can.

Outputs data/neural_v2_model/:
  model.pt              the trained nn.Module weights
  metadata.json         feature list, player map, val AUC, scaler stats

Run via `python scripts/train_neural_v2.py` (uses MPS on Mac if available,
CUDA if on a GPU runner, else CPU). Takes ~3-8 min on M2 Pro GPU, 30-60
min on CI CPU.

Author: Mohammad G. Nasiri
"""

import csv
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime

try:
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install torch numpy")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    SHOTS_CSV = "data/xg_training/shots.csv"
    OUT_DIR = "data/neural_v2_model"
    MODEL_FILE = f"{OUT_DIR}/model.pt"
    META_FILE = f"{OUT_DIR}/metadata.json"

    # Same feature set xg_v3 uses — guarantees 1:1 comparability. is_empty_net
    # is excluded (label leakage: it's ~unknowable at predict time and near-
    # perfectly correlated with goals); empty-net rows are also dropped in
    # load_dataset. neural_v2_predict.py reads feature_names from metadata, so
    # this takes effect automatically on the next retrain with no mismatch.
    NUMERIC_FEATURES = [
        "shot_distance", "shot_angle", "is_powerplay", "is_rebound",
        "seconds_since_last_event", "score_differential",
        "period", "prior_event_distance",
    ]
    SHOT_TYPES = ["backhand", "bat", "between-legs", "cradle", "deflected",
                  "poke", "slap", "snap", "tip-in", "wrap-around", "wrist"]
    STRENGTH_STATES = ["1v0", "3v3", "3v4", "3v5", "4v3", "4v4", "4v5", "4v6",
                       "5v3", "5v4", "5v5", "5v6", "6v3", "6v4", "6v5"]

    EMB_DIM = 32
    HIDDEN = 128
    NUM_RES_BLOCKS = 3
    DROPOUT = 0.1

    BATCH_SIZE = 256
    LR = 3e-4
    MAX_EPOCHS = 50
    EARLY_STOP_PATIENCE = 5
    VAL_FRACTION = 0.10

    SEED = 42


os.makedirs(Config.OUT_DIR, exist_ok=True)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =============================================================================
# DATA LOADING
# =============================================================================
def row_to_feature_vec(row, feature_names):
    """Convert one shots.csv row to a dense vector matching feature_names."""
    try:
        dist = float(row["shot_distance"])
        angle = float(row["shot_angle"])
    except (ValueError, KeyError):
        return None
    base = {
        "shot_distance": dist,
        "shot_angle": angle,
        "is_powerplay": float(row.get("is_powerplay", 0) or 0),
        "is_rebound": float(row.get("is_rebound", 0) or 0),
        "seconds_since_last_event": float(row.get("seconds_since_last_event", 0) or 0),
        "is_empty_net": float(row.get("is_empty_net", 0) or 0),
        "score_differential": float(row.get("score_differential", 0) or 0),
        "period": float(row.get("period", 1) or 1),
        "prior_event_distance": float(row.get("prior_event_distance", 0) or 0),
        "distance_x_angle": dist * angle,
        "is_slot_shot": float(dist < 20),
        "is_high_danger": float(dist < 30 and angle < 30),
    }
    st = row.get("shot_type", "wrist") or "wrist"
    for t in Config.SHOT_TYPES:
        base[f"shot_type_{t}"] = 1.0 if st == t else 0.0
    ss = row.get("strength_state", "5v5") or "5v5"
    for s in Config.STRENGTH_STATES:
        base[f"strength_state_{s}"] = 1.0 if ss == s else 0.0
    return [base[name] for name in feature_names]


def build_feature_names():
    names = [
        "shot_distance", "shot_angle", "is_powerplay", "is_rebound",
        "seconds_since_last_event", "score_differential",
        "period", "prior_event_distance",
        "distance_x_angle", "is_slot_shot", "is_high_danger",
    ]
    names += [f"shot_type_{t}" for t in Config.SHOT_TYPES]
    names += [f"strength_state_{s}" for s in Config.STRENGTH_STATES]
    return names


def load_dataset():
    feature_names = build_feature_names()
    rows = []
    skipped = 0
    with open(Config.SHOTS_CSV, "r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            vec = row_to_feature_vec(r, feature_names)
            if vec is None:
                skipped += 1
                continue
            # Drop empty-net / pulled-goalie shots (different scoring process).
            if int(float(r.get("is_empty_net", 0) or 0)) == 1:
                skipped += 1
                continue
            try:
                pid = int(r["player_id"])
                label = int(r["is_goal"])
                date = r.get("game_date", "")
            except (ValueError, KeyError):
                skipped += 1
                continue
            if not pid:
                skipped += 1
                continue
            rows.append((date, pid, vec, label))

    rows.sort(key=lambda r: r[0])
    print(f"  Loaded {len(rows)} shots ({skipped} skipped)")
    return feature_names, rows


def build_player_map(rows):
    """Contiguous 0-based index per player. Index 0 reserved for UNK so
    unseen players at inference don't index out of range."""
    ids = sorted({r[1] for r in rows})
    pmap = {pid: i + 1 for i, pid in enumerate(ids)}
    return pmap


class ShotsDataset(Dataset):
    def __init__(self, rows, pmap, feat_mean, feat_std):
        self.player_idx = np.array(
            [pmap.get(r[1], 0) for r in rows], dtype=np.int64
        )
        X = np.array([r[2] for r in rows], dtype=np.float32)
        self.X = (X - feat_mean) / np.where(feat_std > 0, feat_std, 1.0)
        self.y = np.array([r[3] for r in rows], dtype=np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        # All floats explicitly float32 — MPS (Apple Silicon GPU) refuses
        # float64 tensors and a bare Python float becomes float64.
        return (
            torch.from_numpy(self.X[i]).float(),
            int(self.player_idx[i]),
            torch.tensor(self.y[i], dtype=torch.float32),
        )


# =============================================================================
# MODEL
# =============================================================================
class NeuralV2(nn.Module):
    def __init__(self, num_players, num_feats,
                 emb_dim=Config.EMB_DIM, hidden=Config.HIDDEN,
                 num_blocks=Config.NUM_RES_BLOCKS, dropout=Config.DROPOUT):
        super().__init__()
        self.player_emb = nn.Embedding(num_players + 1, emb_dim)
        nn.init.normal_(self.player_emb.weight, std=0.02)

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
        return self.head(x).squeeze(-1)  # logits


# =============================================================================
# TRAIN
# =============================================================================
def run_training():
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    device = pick_device()
    print(f"  Device: {device}")

    print("\n  Loading shots...")
    feature_names, rows = load_dataset()
    pmap = build_player_map(rows)

    split_idx = int(len(rows) * (1 - Config.VAL_FRACTION))
    train_rows, val_rows = rows[:split_idx], rows[split_idx:]
    print(f"  Train: {len(train_rows)}  Val: {len(val_rows)}")

    X_train = np.array([r[2] for r in train_rows], dtype=np.float32)
    feat_mean = X_train.mean(axis=0)
    feat_std = X_train.std(axis=0)

    train_ds = ShotsDataset(train_rows, pmap, feat_mean, feat_std)
    val_ds = ShotsDataset(val_rows, pmap, feat_mean, feat_std)

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE * 4)

    pos_rate = float(train_ds.y.mean())
    pos_weight = torch.tensor(
        [(1 - pos_rate) / max(pos_rate, 1e-6)], device=device
    )
    print(f"  Positive rate: {pos_rate:.3f}  pos_weight: {pos_weight.item():.2f}")

    model = NeuralV2(
        num_players=len(pmap),
        num_feats=len(feature_names),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} params "
          f"({Config.NUM_RES_BLOCKS} blocks x {Config.HIDDEN} hidden, emb {Config.EMB_DIM})")

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=Config.LR)

    best_val = math.inf
    patience = 0
    best_state = None

    print(f"\n  Training up to {Config.MAX_EPOCHS} epochs "
          f"(early stop after {Config.EARLY_STOP_PATIENCE} stale):")
    for epoch in range(Config.MAX_EPOCHS):
        model.train()
        train_loss_sum = 0.0
        n = 0
        for x, p, y in train_loader:
            x = x.to(device); p = p.to(device); y = y.to(device)
            opt.zero_grad()
            logits = model(x, p)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            train_loss_sum += loss.item() * y.size(0)
            n += y.size(0)
        train_loss = train_loss_sum / max(n, 1)

        model.train(False)
        val_loss_sum = 0.0
        total = 0
        all_logits = []
        all_y = []
        with torch.no_grad():
            for x, p, y in val_loader:
                x = x.to(device); p = p.to(device); y = y.to(device)
                logits = model(x, p)
                val_loss_sum += loss_fn(logits, y).item() * y.size(0)
                all_logits.append(logits.cpu().numpy())
                all_y.append(y.cpu().numpy())
                total += y.size(0)
        val_loss = val_loss_sum / max(total, 1)

        from sklearn.metrics import roc_auc_score
        lg = np.concatenate(all_logits); yy = np.concatenate(all_y)
        val_auc = roc_auc_score(yy, lg) if len(np.unique(yy)) > 1 else float("nan")

        print(f"    epoch {epoch + 1:>2}  train {train_loss:.4f}  "
              f"val {val_loss:.4f}  val AUC {val_auc:.4f}")

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= Config.EARLY_STOP_PATIENCE:
                print(f"    early stopping at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.train(False)
    all_logits = []
    all_y = []
    with torch.no_grad():
        for x, p, y in val_loader:
            x = x.to(device); p = p.to(device)
            all_logits.append(model(x, p).cpu().numpy())
            all_y.append(y.numpy())
    lg = np.concatenate(all_logits); yy = np.concatenate(all_y)
    from sklearn.metrics import roc_auc_score, log_loss
    final_auc = float(roc_auc_score(yy, lg))
    final_logloss = float(log_loss(yy, 1 / (1 + np.exp(-lg))))
    print(f"\n  Best val AUC: {final_auc:.4f}  LogLoss: {final_logloss:.4f}")

    torch.save(model.state_dict(), Config.MODEL_FILE)
    meta = {
        "trained_at": datetime.now().isoformat(),
        "num_shots": int(len(rows)),
        "num_train": int(len(train_rows)),
        "num_val": int(len(val_rows)),
        "num_players": int(len(pmap)),
        "feature_names": feature_names,
        "feature_mean": feat_mean.tolist(),
        "feature_std": feat_std.tolist(),
        "player_map": {str(k): v for k, v in pmap.items()},
        "cv_val_auc": final_auc,
        "cv_val_logloss": final_logloss,
        "architecture": {
            "emb_dim": Config.EMB_DIM,
            "hidden": Config.HIDDEN,
            "num_res_blocks": Config.NUM_RES_BLOCKS,
            "dropout": Config.DROPOUT,
        },
        "hyperparams": {
            "batch_size": Config.BATCH_SIZE,
            "lr": Config.LR,
            "max_epochs": Config.MAX_EPOCHS,
            "early_stop_patience": Config.EARLY_STOP_PATIENCE,
        },
    }
    with open(Config.META_FILE, "w") as f:
        json.dump(meta, f)
    print(f"\n  Saved: {Config.MODEL_FILE}")
    print(f"  Saved: {Config.META_FILE}")


def main():
    print("=" * 70)
    print("  NEURAL v2 TRAINER — MLP with player embeddings")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)
    if not os.path.exists(Config.SHOTS_CSV):
        print(f"  {Config.SHOTS_CSV} not found — run collect_xg_data.py first")
        sys.exit(0)
    run_training()
    print("\n  Training complete.")


if __name__ == "__main__":
    main()
