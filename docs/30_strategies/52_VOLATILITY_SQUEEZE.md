# Volatility Squeeze Breakout

`volatility_squeeze` is a selective breakout strategy.

It tries to avoid the most common breakout problem: trading every small new high
or low in noisy chop. Instead, it requires the market to compress first, then
trades only if price breaks outside a volatility band.

## Core Idea

```text
1. Build a Bollinger-style band from prior prices.
2. Measure recent volatility versus earlier volatility.
3. Trade only if recent volatility is much lower than prior volatility.
4. Enter only when the latest price breaks outside the band by enough bps.
```

In plain English:

```text
quiet range + confirmed escape = possible trade
```

## Why This Is Useful For FX And Crypto

FX and crypto often move in phases:

```text
range -> compression -> breakout -> follow-through or failure
```

The strategy is meant to capture the breakout stage while avoiding ordinary
range noise. It is especially useful to test around session changes and larger
portfolio backtests where each symbol can have a different volatility regime.

## Main Filters

`volatility_squeeze` will not enter unless all of these pass:

- `band_width_bps` is above the minimum band width.
- `prior_volatility_bps` is high enough to make compression meaningful.
- `squeeze_ratio = recent_volatility / prior_volatility` is below the configured maximum.
- `breakout_bps` clears the entry buffer.
- estimated spread/slippage/fees are below the expected edge.

This makes it slower than raw `breakout`, but that is intentional.

## Config

In `configs/default.toml`:

```toml
[strategy.volatility_squeeze]
symbol = "EURUSD"
lookback = 24
squeeze_window = 8
band_stdev_multiplier = 2.0
breakout_buffer_bps = 2.5
exit_buffer_bps = 1.0
max_squeeze_ratio = 0.50
min_prior_volatility_bps = 0.5
min_band_width_bps = 1.0
forex_allowed_utc_hours = [11, 12, 13, 14, 15, 16, 17, 18, 19]
metal_allowed_utc_hours = [11, 12, 13, 14, 15, 16, 17, 18, 19]
target_notional_usd = 50000.0
position_sizing = "volatility"
max_target_notional_usd = 75000.0
max_holding_period = 24
```

## Run In VS Code Terminal

Inspect a hand-built squeeze pattern:

```bash
python scripts/evaluation/strategy_demo.py \
  --strategy volatility_squeeze \
  --lookback 10 \
  --prices 1.0000,1.0010,0.9990,1.0010,0.9990,1.0000,1.00005,0.99995,1.0000,1.0025
```

Backtest one symbol:

```bash
python scripts/evaluation/run_backtest.py --strategy volatility_squeeze
```

Compare it with the other standalone strategies:

```bash
python scripts/evaluation/compare_strategies.py \
  --strategy breakout \
  --strategy session_breakout \
  --strategy volatility_squeeze \
  --strategy mean_reversion \
  --strategy alpha_router
```

Optimize conservative squeeze parameters on a portfolio basket:

```bash
python scripts/evaluation/volatility_squeeze_optimize.py \
  --price-csv data/full_20gb_15m_prices.csv \
  --quote-csv data/full_20gb_15m_quotes.csv \
  --symbol EURUSD \
  --symbol GBPUSD \
  --symbol USDJPY \
  --symbol USDCHF \
  --symbol XAUUSD \
  --symbol XAGUSD
```

This writes:

```text
outputs/backtests/volatility_squeeze_optimization.csv
```

## Current Research Status

This is a research sleeve and an optional part of `alpha_router`.

The first FX/metals portfolio smoke test preferred the stricter default:

```text
lookback=24, squeeze_window=8, max_squeeze_ratio=0.50, breakout_buffer_bps=2.5
```

Hour attribution then showed weak realized P&L around 05, 22, and 23 UTC. The
optimizer preferred allowing new FX/metals entries during 11-19 UTC, while
leaving crypto unrestricted until we have crypto sample data.

The next question is empirical:

```text
Does the squeeze filter reduce bad breakout trades enough to justify fewer signals?
```

It is kept at `volatility_squeeze_weight = 0.0` in the default router config.
Research commands can now test it explicitly with an eight-number router
candidate such as `0,0,0,0,0,0,0,1`, which means "squeeze only." The important
rule is unchanged: let walk-forward optimization decide whether it deserves
capital before promoting it toward live MT5 use.
