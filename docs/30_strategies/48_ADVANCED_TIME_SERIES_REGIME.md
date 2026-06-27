# Advanced Time-Series Regime Report

This report adds a lightweight Kalman-style trend diagnostic. It classifies each
symbol as trend-friendly, choppy, or too volatile.

The diagnostic now also powers the standalone `kalman_trend` strategy documented
in `60_KALMAN_TREND.md`.

## Run It

```bash
quanthack time-series-report --price-csv data/downloaded_scan_prices.csv
```

The grouped script path works too:

```bash
python scripts/evaluation/time_series_report.py \
  --price-csv data/downloaded_scan_prices.csv
```

Outputs are written to:

```text
outputs/backtests/time_series_regimes.csv
```

## What It Measures

- `kalman_slope_bps`: latest smoothed directional slope
- `trend_efficiency`: how direct the path was versus how noisy the path was
- `realized_volatility_bps`: recent root-mean-square return size
- `regime`: `TREND_UP`, `TREND_DOWN`, `CHOP`, or `HIGH_VOLATILITY`

## Current Result

On `data/downloaded_scan_prices.csv`, the strict default classified all symbols
as `CHOP`. A looser sensitivity run:

```bash
python scripts/evaluation/time_series_report.py \
  --price-csv data/downloaded_scan_prices.csv \
  --min-trend-efficiency 0.02 \
  --output outputs/backtests/time_series_regimes_loose.csv
```

classified only metals as trend-up:

- XAGUSD: `TREND_UP`
- XAUUSD: `TREND_UP`
- FX pairs: still `CHOP`

That matches the later `kalman_trend` finding: the strongest in-sample behavior
is mostly a metals phenomenon, with only selected FX support. Use it as a
research diagnostic and validation companion rather than a blind live routing
rule.
