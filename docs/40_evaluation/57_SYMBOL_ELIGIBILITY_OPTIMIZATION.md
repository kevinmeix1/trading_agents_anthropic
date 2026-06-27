# Symbol Eligibility Optimization

Symbol eligibility optimization asks:

```text
Given one strategy, which symbols should it actually trade?
```

This matters because a strategy can be positive overall while one or two symbols
quietly drag the portfolio down. The optimizer:

```text
1. Runs per-symbol attribution for a strategy.
2. Builds candidate universes:
   - all symbols
   - positive attribution symbols
   - positive attribution plus flat/no-trade symbols
   - drop worst N symbols
   - top N symbols by attribution P&L
   - optional combinations from the top attribution-ranked symbols
3. Replays each candidate universe through the shared-risk portfolio backtest.
4. Optionally attaches fixed-warmup walk-forward metrics.
5. Writes both candidate results and source attribution ranks.
```

The walk-forward-aware mode reports both raw and selective-strategy metrics:

```text
positive_fold_fraction: percent of all folds with positive return
active_fold_fraction: percent of folds where the strategy actually traded
active_positive_fold_fraction: percent of active folds with positive return
non_negative_fold_fraction: percent of folds that avoided losing money
median_active_test_return_pct: median return among active folds only
```

This distinction matters because a selective strategy can correctly stay flat
for many weak periods. A zero-return no-trade fold should not be treated like a
losing fold.

Current `dual_squeeze` finding:

```text
all symbols:
  return: 0.124%
  max drawdown: 0.055%
  trades: 48

positive_active / no EURUSD:
  return: 0.137%
  max drawdown: 0.049%
  trades: 40
  risk discipline: 100/100
```

Walk-forward still rejects promotion:

```text
positive_active stable folds: 33.3%
top_5 stable folds: 33.3%
promotion: REJECT
```

Useful command:

```bash
python scripts/evaluation/symbol_eligibility_optimize.py \
  --strategy dual_squeeze \
  --price-csv data/full_20gb_15m_prices.csv \
  --quote-csv data/full_20gb_15m_quotes.csv \
  --min-symbols 3 \
  --output outputs/backtests/dual_squeeze_symbol_eligibility.csv
```

Combination search example:

```bash
quanthack-symbol-eligibility-optimize \
  --strategy champion_ensemble \
  --price-csv data/full_20gb_15m_prices.csv \
  --quote-csv data/full_20gb_15m_quotes.csv \
  --min-symbols 5 \
  --max-symbols 5 \
  --include-combinations \
  --combination-pool-size 7 \
  --max-combinations 21 \
  --include-walk-forward \
  --train-size 480 \
  --test-size 96 \
  --step-size 96
```

Research verdict:

```text
Dropping EURUSD improves the paper backtest.
Do not promote the filtered universe to live MT5 until it passes stronger
walk-forward or live dry-run evidence.
```
