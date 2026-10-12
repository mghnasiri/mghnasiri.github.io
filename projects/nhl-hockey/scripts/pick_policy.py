"""
Which model's Tims picks go out, in order of preference (the first one with
picks today is used). The dashboard's champion badge marks the first entry.

Changed only on evidence (the offline backtest plus real Tims days), never by
a trailing window: on 82 real Tims days 27 of 28 model pairs were
statistically tied, and one extra day moves a model's measured lift ~2pp.
xG v3 leads (+7.3pp over random on 79 days) ahead of Market (+1.9pp, with
whole games missing odds on some days).
"""

SOURCE_ORDER = ["xg_v3", "market_odds", "monte_carlo", "neural_network"]
