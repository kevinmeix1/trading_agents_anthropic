# Dual Squeeze Strategy

`dual_squeeze` is a conservative volatility-breakout strategy.

It does two things:

```text
1. Fast squeeze: find a short-window volatility compression breakout.
2. Slow confirmation: only enter if a slower window agrees with the breakout direction.
```

Why this exists:

```text
raw breakout and raw momentum overtraded badly on the downloaded FX/metals data.
volatility_squeeze was positive, but small.
dual_squeeze keeps the same idea and adds a slower confirmation gate.
```

Current default:

```text
fast lookback: 14 prices
fast squeeze window: 4 returns
fast max squeeze ratio: 0.60
fast breakout buffer: 2.5 bps
slow confirmation lookback: 24 prices
slow confirmation window: 8 returns
slow max squeeze ratio: 0.70
confirmation mode: squeeze_bias
max holding period: 12 bars
```

Backtest checkpoint on `data/full_20gb_15m_prices.csv`:

```text
dual_squeeze:
  return: 0.124%
  max drawdown: 0.055%
  Sharpe 15m: 0.023
  trades: 48
  risk discipline: 100/100
```

Important warning:

```text
walk-forward activity is still sparse.
Treat this as the best current paper-trading candidate, not an automatic MT5 live strategy.
```

Useful commands:

```bash
python scripts/evaluation/portfolio_compare.py \
  --price-csv data/full_20gb_15m_prices.csv \
  --quote-csv data/full_20gb_15m_quotes.csv \
  --strategy dual_squeeze \
  --strategy volatility_squeeze \
  --output outputs/backtests/full_20gb_dual_squeeze_compare.csv

python scripts/evaluation/strategy_attribution.py \
  --price-csv data/full_20gb_15m_prices.csv \
  --quote-csv data/full_20gb_15m_quotes.csv \
  --strategy dual_squeeze \
  --strategy volatility_squeeze \
  --output outputs/backtests/full_20gb_dual_squeeze_attribution.csv

python scripts/evaluation/dual_squeeze_optimize.py \
  --price-csv data/full_20gb_15m_prices.csv \
  --quote-csv data/full_20gb_15m_quotes.csv \
  --walk-forward \
  --output outputs/backtests/dual_squeeze_walk_forward_optimization.csv
```
