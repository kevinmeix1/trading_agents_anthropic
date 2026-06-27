# Breakout Strategy

Step 28 adds:

```text
breakout
```

Breakout is a simple Donchian-style strategy. It watches a recent price channel
and trades only when the latest price breaks outside that prior range.

## Core Idea

```text
prior upper band = max(previous prices)
prior lower band = min(previous prices)

latest > upper band + buffer -> long
latest < lower band - buffer -> short
```

The latest price is excluded from the channel. That prevents the strategy from
moving the goalpost with the same price it is judging.

## Why Add It

Breakout is related to momentum, but it asks a different question:

```text
Momentum: did price move enough?
Breakout: did price clear an important recent level?
```

That makes it useful as a separate sleeve inside `alpha_router`.

## Config

In `configs/default.toml`:

```toml
[strategy.breakout]
symbol = "EURUSD"
lookback = 8
breakout_buffer_bps = 2.0
exit_buffer_bps = 1.0
min_channel_width_bps = 2.0
target_notional_usd = 50000.0
```

## Router Role

`alpha_router` now listens to:

```text
momentum
ma_crossover
breakout
mean_reversion
ml_alpha
```

If momentum and breakout agree, the router has stronger evidence for a trend.
If mean reversion disagrees, the conflict penalty can still keep the final
decision flat.

## Run In VS Code Terminal

Inspect breakout:

```bash
python scripts/evaluation/strategy_demo.py --strategy breakout --scenario up
```

Backtest it:

```bash
python scripts/evaluation/run_backtest.py --strategy breakout
```

Compare all strategies:

```bash
python scripts/evaluation/compare_strategies.py
```

Run tests:

```bash
python -m unittest tests.test_strategy
```

## Important Limitation

Breakouts can overtrade in noisy markets. The strategy has spread, cost, channel
width, and exit filters, but it still needs larger historical data before we
trust it.
