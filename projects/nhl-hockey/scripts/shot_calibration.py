"""
Turn the shot models' outputs into honest goal probabilities.

Both shot-quality models (xg_predict.py's XGBoost and Neural v2) up-weight
goals in training (~13x), which multiplies every raw shot probability's
odds by that weight: the 2025-26 shots average 0.38 raw vs a 0.072 goal
rate, and player probabilities averaged 44% vs 15% actual. Undoing the
weight per shot fixes that; a 2026-03..06 backtest (2,950 player-games)
also lifted xG's within-day AUC (0.700 -> 0.702) and group top-1 picks
(21.9% -> 24.4%).

The models are trained on unblocked attempts (goal + shot-on-goal +
missed-shot) but avg_shots counts shots on goal, so a player's expected
goals also needs ATTEMPTS_PER_SOG.
"""

# Unblocked attempts per shot on goal. League ratio ~1.4; the backtest's
# best fits, net of each model's own per-shot bias, recover 1.36 (xG) and
# 1.40 (Neural v2).
ATTEMPTS_PER_SOG = 1.4


def undo_class_weight(p, weight):
    """Map a probability from a model trained with positives up-weighted by
    `weight` back to the true scale (its odds are ~weight x too high).
    Works on floats and numpy arrays."""
    return p / (p + (1 - p) * weight)
