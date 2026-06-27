# Range Expansion Trend

`range_expansion_trend` is a selective breakout-continuation sleeve.

## Intuition

The strongest current candidates tend to work when price pressure is one-sided,
but raw momentum and raw breakout can churn. This sleeve only enters when the
latest short impulse both:

```text
1. breaks outside the prior range
2. is unusually large versus the prior baseline volatility
```

The goal is to trade confirmed range escape, not every small continuation move.

## Signal Timing

At each bar:

1. Use the latest `lookback` prices only.
2. Split the window into a prior baseline and the latest `trigger_window`.
3. Measure the prior high/low range.
4. Measure the trigger move in basis points.
5. Measure expansion z-score:

```text
abs(trigger move) / (baseline volatility * sqrt(trigger window))
```

6. Enter only if the trigger move, range break, and path efficiency agree.
7. Exit when the impulse fades back into the prior range or the holding cap is
   reached.

This is leakage-safe because the prior range and trigger impulse use only prices
available at the current bar.

## Current Defaults

The first loose profile traded too often and was rejected. Defaults now use the
stricter scan winner:

```text
lookback: 40
trigger_window: 4
min_trigger_move_bps: 10.0
min_range_break_bps: 3.0
min_expansion_zscore: 2.5
min_trend_efficiency: 0.65
min_expected_edge_bps: 6.0
max_holding_period: 6 bars
FX/metals session: 10-14 UTC
crypto session: disabled by default
```

## Seven-Symbol Validation

Command:

```bash
quanthack portfolio-fixed-warmup-walk-forward \
  --strategy range_expansion_trend \
  --symbol XAGUSD --symbol XAUUSD --symbol USDCHF --symbol AUDUSD \
  --symbol GBPUSD --symbol EURUSD --symbol EURGBP \
  --price-csv data/full_20gb_15m_prices.csv \
  --quote-csv data/full_20gb_15m_quotes.csv \
  --train-size 480 \
  --test-size 96 \
  --step-size 96 \
  --summary-output outputs/backtests/range_expansion_trend_7_fixed_warmup_summary.csv \
  --folds-output outputs/backtests/range_expansion_trend_7_fixed_warmup_folds.csv
```

Result:

```text
positive fold fraction: 29.4%
active fold fraction: 58.8%
active positive fold fraction: 50.0%
non-negative fold fraction: 70.6%
median active return: 0.003%
worst drawdown: 0.068%
risk discipline: 100/100
evaluation fills: 80
promotion: PAPER_ONLY
```

## Symbol Eligibility

Best eligible basket:

```text
XAGUSD
XAUUSD
USDCHF
AUDUSD
```

Top-4 fixed-warmup result:

```text
positive fold fraction: 23.5%
active fold fraction: 41.2%
active positive fold fraction: 57.1%
non-negative fold fraction: 82.4%
median active return: 0.005%
worst drawdown: 0.053%
evaluation fills: 44
largest positive fold contribution: 42.7%
promotion: PAPER_ONLY
```

This is more robust than the broad seven-symbol run, but still too sparse to
stand alone.

## Adaptive Selector Check

Adding the top-4 range-expansion recipe to the current adaptive stack produced:

```text
candidates:
  kalman_trend, champion_ensemble, macd_momentum, range_expansion_top4

positive fold fraction: 35.3%
active positive folds: 66.7%
non-negative folds: 82.4%
median active return: 0.005%
stitched OOS final equity: $1,003,527.17
promotion: PAPER_ONLY
```

This did not beat the current adaptive leader:

```text
stitched OOS final equity: $1,004,225.57
active positive folds: 72.7%
```

Verdict: keep `range_expansion_trend` as a documented paper sleeve and
diagnostic. Do not include it in the current top adaptive candidate.
