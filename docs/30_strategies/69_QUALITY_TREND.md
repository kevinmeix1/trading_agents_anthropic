# Quality Trend

`quality_trend` is a conservative trend-confirmation strategy.

## Intuition

The current best candidates are trend-oriented, but several losing folds come
from taking trend signals when the market is noisy or the signal is too thin.
This sleeve requires two different trend views to agree:

```text
MACD histogram direction == Kalman trend direction
```

It then sizes by the weaker confidence, so a strong MACD reading cannot override
a weak Kalman regime, and a strong Kalman slope cannot override a weak MACD
histogram.

## Signal Timing

At each bar:

1. Use only prices available up to the current bar.
2. Read MACD 6/18/5 histogram direction.
3. Read Kalman trend direction on the latest 80 bars.
4. Enter only if both agree, estimated edge clears costs, and the current UTC
   hour is allowed.
5. Exit when agreement fades, reverses, confidence drops, or the holding cap is
   reached.

The default FX/metals session is 10-14 UTC. Crypto is disabled by default.

## Current Parameters

```text
kalman_lookback: 80
kalman_min_abs_slope_bps: 0.25
kalman_min_trend_efficiency: 0.20
kalman_min_expected_edge_bps: 5.0
macd_fast/slow/signal: 6/18/5
macd_min_histogram_bps: 2.0
macd_min_macd_bps: 1.0
min_combined_confidence: 0.30
min_expected_edge_bps: 2.0
max_holding_period: 16 bars
```

## Seven-Symbol Validation

Command:

```bash
quanthack portfolio-fixed-warmup-walk-forward \
  --strategy quality_trend \
  --symbol XAGUSD --symbol XAUUSD --symbol USDCHF --symbol AUDUSD \
  --symbol GBPUSD --symbol EURUSD --symbol EURGBP \
  --price-csv data/full_20gb_15m_prices.csv \
  --quote-csv data/full_20gb_15m_quotes.csv \
  --train-size 480 \
  --test-size 96 \
  --step-size 96 \
  --summary-output outputs/backtests/quality_trend_7_fixed_warmup_summary.csv \
  --folds-output outputs/backtests/quality_trend_7_fixed_warmup_folds.csv
```

Result:

```text
folds: 17
positive fold fraction: 17.6%
active fold fraction: 23.5%
active positive fold fraction: 75.0%
non-negative fold fraction: 94.1%
median active test return: 0.023%
worst test drawdown: 0.055%
risk discipline: 100/100
evaluation fills: 30
promotion: PAPER_ONLY
```

## Parameter Scan

The first default was too strict. A small controlled scan is saved at:

```text
outputs/backtests/quality_trend_parameter_scan.csv
```

Best quick revision:

```text
session_10_14_conf030_edge2:
  active positive folds: 75.0%
  non-negative folds: 94.1%
  median active return: 0.023%
  worst drawdown: 0.055%
  fills: 30
```

## Adaptive Selector Check

Adding `quality_trend` to the current adaptive stack did not improve the paper
candidate.

```text
candidates:
  kalman_trend, champion_ensemble, macd_momentum, quality_trend

stitched OOS final equity: $1,003,721.12
active positive folds: 62.5%
non-negative folds: 82.4%
evaluation fills: 68
```

The current adaptive leader without `quality_trend` remains stronger:

```text
stitched OOS final equity: $1,004,225.57
active positive folds: 72.7%
evaluation fills: 86
```

Verdict: keep `quality_trend` as a conservative diagnostic sleeve. Do not add it
to the live/paper candidate stack unless future data shows higher coverage
without damaging active-fold consistency.
