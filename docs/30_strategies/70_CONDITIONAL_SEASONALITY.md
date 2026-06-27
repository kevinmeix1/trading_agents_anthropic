# Conditional Seasonality

`conditional_seasonality` tests a time-of-day hypothesis without using future
data.

## Intuition

The hourly scans showed large symbol-hour effects, especially in metals. Raw
hour-of-day effects are easy to overfit, so this strategy only uses historical
same-slot examples where the recent momentum condition matches the current bar:

```text
current condition:
  4-bar momentum up / down / flat

historical examples:
  previous same 15-minute slot from prior days
  keep only examples with the same condition
  measure their next 4-bar forward return
```

At the current bar, those historical forward returns are already known, so the
feature is leakage-safe.

## Modes

```text
signal_mode = "momentum"
  trade in the direction of the historical forward drift

signal_mode = "reversal"
  fade the historical forward drift
```

The first direct-drift prototype was high-churn and rejected. The default is now
the safer reversal-mode research profile.

## Current Defaults

```text
period_bars: 96
lookback_periods: 4
horizon_bars: 4
momentum_lookback: 4
momentum_threshold_bps: 2.0
signal_mode: reversal
min_samples: 3
entry_threshold_bps: 10.0
min_abs_tstat: 1.50
min_consistency: 0.67
max_holding_period: 4
```

## Validation

Command:

```bash
quanthack portfolio-fixed-warmup-walk-forward \
  --strategy conditional_seasonality \
  --symbol XAGUSD --symbol XAUUSD --symbol USDCHF --symbol AUDUSD \
  --symbol GBPUSD --symbol EURUSD --symbol EURGBP \
  --price-csv data/full_20gb_15m_prices.csv \
  --quote-csv data/full_20gb_15m_quotes.csv \
  --train-size 480 \
  --test-size 96 \
  --step-size 96 \
  --summary-output outputs/backtests/conditional_seasonality_7_fixed_warmup_summary.csv \
  --folds-output outputs/backtests/conditional_seasonality_7_fixed_warmup_folds.csv
```

Result:

```text
positive fold fraction: 5.9%
active fold fraction: 23.5%
active positive fold fraction: 25.0%
non-negative fold fraction: 82.4%
median active return: -0.001%
worst drawdown: 0.026%
risk discipline: 100/100
evaluation fills: 20
promotion: REJECT
```

## Scan Notes

Saved scans:

```text
outputs/backtests/conditional_seasonality_mode_scan.csv
outputs/backtests/conditional_seasonality_strict_scan.csv
```

Important outcomes:

```text
momentum_default:
  fills: 1712
  median active return: -0.067%
  promotion: REJECT

reversal_strict_edge6_t1:
  positive folds: 52.9%
  median active return: 0.003%
  fills: 522
  promotion: REJECT

reversal_min3_edge10_t1.5:
  non-negative folds: 82.4%
  worst drawdown: 0.026%
  fills: 20
  promotion: REJECT
```

Verdict: keep the implementation because it is a clean leakage-safe research
pipeline, but do not include it in the current paper/live candidate stack.
