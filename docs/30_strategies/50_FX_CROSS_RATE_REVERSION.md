# FX Cross-Rate Reversion

This strategy looks for temporary disagreement between an FX pair and the
synthetic cross rate implied by other FX pairs.

Example:

```text
synthetic EURGBP = EURUSD / GBPUSD
```

If live `EURGBP` is much higher than that synthetic value, `EURGBP` is rich and
the strategy wants to short it. If live `EURGBP` is much lower, it is cheap and
the strategy wants to buy it.

## Why This Helps

FX pairs are linked by currency arithmetic. They will not match perfectly after
spreads, slippage, latency, and broker microstructure, but large short-lived
deviations can be useful alpha candidates.

This is different from normal momentum or mean reversion:

- Momentum asks: did this pair move strongly in one direction?
- Mean reversion asks: is this pair far from its own recent average?
- Cross-rate reversion asks: is this pair far from the value implied by other
  related FX pairs?

## Current Implementation

The code lives in:

```text
src/quanthack/strategies/strategy.py
```

Main objects:

- `CrossRateReversionConfig`
- `CrossRateReversionReading`
- `CrossRateReversionStrategy`

The strategy is portfolio-aware. It needs context for multiple FX symbols, so it
is most useful through portfolio backtests or live dry-run loops that call
`update_portfolio_context(...)`.

## Safety Rules

The strategy fails closed:

- non-FX symbols return no action
- missing synthetic paths return no action
- deviations below the edge threshold return no action
- deviations above the stability guard return no action or exit
- spreads and estimated costs must be cleared before entry

Default target:

```toml
[strategy.cross_rate_reversion]
symbol = "EURUSD"
allowed_symbols = []
lookback = 12
entry_zscore = 1.0
max_abs_deviation_bps = 80.0
```

`EURUSD` keeps the tiny built-in sample data paths working. For real portfolio
research, `EURGBP`, `EURCHF`, and `EURUSD` are usually more interesting
starting points than running every FX symbol blindly.

## How To Try It

Quick signal diagnostic:

```bash
quanthack-signal-diagnostics \
  --strategy cross_rate_reversion \
  --price-csv data/full_20gb_15m_prices.csv \
  --quote-csv data/full_20gb_15m_quotes.csv \
  --symbol EURGBP \
  --symbol EURUSD \
  --symbol GBPUSD \
  --horizon-bars 4
```

Portfolio backtest:

```bash
quanthack-portfolio-backtest \
  --strategy cross_rate_reversion \
  --price-csv data/full_20gb_15m_prices.csv \
  --quote-csv data/full_20gb_15m_quotes.csv \
  --symbol EURGBP \
  --symbol EURUSD \
  --symbol GBPUSD
```

The important thing to inspect is whether active signals have positive average
signed forward return after costs. If that number is weak, the next research
step is to tighten session filters, increase the deviation threshold, or limit
the strategy to the best cross pairs.

## Router Sleeve

The alpha router can include cross-rate reversion as one more weighted vote.
It is disabled by default:

```toml
[strategy.alpha_router]
cross_rate_weight = 0.0
```

For research, try a small weight first:

```toml
[strategy.alpha_router]
cross_rate_weight = 0.10
```

This is intentionally a sleeve, not the main engine. In the first 15-minute
diagnostics pass, realistic costs made standalone cross-rate trades rare, while
looser research settings suggested the direction can be useful on selected
pairs such as `EURGBP`, `EURUSD`, and `EURCHF`.

Use `40_evaluation/51_CROSS_RATE_OPTIMIZATION.md` to rank symbols and thresholds
before promoting a nonzero `cross_rate_weight` into portfolio router tests.

After optimization, you can restrict the sleeve to researched symbols:

```toml
[strategy.cross_rate_reversion]
allowed_symbols = ["EURGBP", "EURUSD", "EURCHF"]
```

An empty list means no allowlist restriction.
