# Regime Selector

Step 26 adds a third strategy:

```text
regime_switch
```

This strategy does not invent a new signal. It chooses which existing behavior
is appropriate:

```text
MOMENTUM       -> allow the momentum strategy
MEAN_REVERSION -> allow the mean-reversion strategy
FLAT           -> stay out, or exit an existing position
```

## Why This Matters

Momentum and mean reversion want opposite market conditions.

Momentum wants:

```text
strong, smooth, persistent direction
```

Mean reversion wants:

```text
stable baseline, contained trend, meaningful deviation
```

The selector prevents a large z-score from blindly overriding a strong trend.
That is important because a price can look "far from average" simply because the
market is trending.

## What It Checks

The selector reads a longer window and estimates:

- Momentum move in basis points.
- Momentum score.
- Momentum trend efficiency.
- Mean-reversion z-score.
- Mean-reversion trend strength.
- Mean-reversion path efficiency.
- Spread in basis points.

It chooses:

- `MOMENTUM` for strong, smooth directional movement.
- `MEAN_REVERSION` for a large deviation without a strong directional trend.
- `FLAT` for ambiguous, unstable, or expensive conditions.

## Hysteresis

The selector has `hysteresis_bars`.

That means a regime has to repeat before the strategy switches. This reduces
churn when the market flickers around a boundary.

## Config

In `configs/default.toml`:

```toml
[strategy.regime_switch]
symbol = "EURUSD"
lookback = 10
momentum_min_move_bps = 8.0
momentum_min_score = 1.0
momentum_min_efficiency = 0.6
mean_reversion_min_abs_zscore = 1.0
mean_reversion_max_trend_bps = 8.0
mean_reversion_max_efficiency = 0.5
max_spread_bps = 10.0
hysteresis_bars = 2
```

## Run In VS Code Terminal

Inspect the selector:

```bash
python scripts/evaluation/strategy_demo.py --strategy regime_switch --scenario up
python scripts/evaluation/strategy_demo.py --strategy regime_switch --scenario flat
python scripts/evaluation/strategy_demo.py --strategy regime_switch --scenario spike_up
```

Backtest it:

```bash
python scripts/evaluation/run_backtest.py --strategy regime_switch
```

Compare it:

```bash
python scripts/evaluation/compare_strategies.py
```

Run tests:

```bash
python -m unittest tests.test_strategy
```

## Important Limitation

The selector is still a hypothesis. It is not "final" until it survives:

- larger historical data;
- cost stress;
- walk-forward tests;
- and eventually MT5 symbol/quote/fill behavior.

For now, it gives us a cleaner offline strategy architecture.
