# ML Alpha Evaluation

The ML router can now be evaluated directly.

`strategy-demo` answers:

```text
What would the router do for this one scenario?
```

`ml-alpha-report` answers:

```text
Did the ML alpha predict next-bar direction well over historical data?
```

## Run It

```bash
quanthack ml-alpha-report
```

The grouped script path works too:

```bash
python scripts/evaluation/ml_alpha_report.py
```

Use a specific symbol or price file:

```bash
quanthack ml-alpha-report --symbol EURUSD
quanthack ml-alpha-report --price-csv data/backtest_prices.csv
```

Use research overrides for faster ML probes without editing `configs/default.toml`:

```bash
quanthack ml-alpha-report \
  --symbol EURUSD \
  --price-csv data/full_20gb_15m_prices.csv \
  --ml-lookback 8 \
  --ml-train-window 80 \
  --ml-epochs 3 \
  --ml-label-threshold-bps 2.0
```

Evaluate all downloaded symbols and calibrate filters:

```bash
quanthack ml-alpha-report \
  --price-csv data/downloaded_scan_prices.csv \
  --all-symbols \
  --calibrate \
  --walk-forward-calibrate
```

## Output

By default, predictions are written to:

```text
outputs/backtests/ml_alpha_predictions.csv
```

Portfolio calibration writes additional files:

```text
outputs/backtests/ml_alpha_portfolio_predictions.csv
outputs/backtests/ml_alpha_calibration.csv
outputs/backtests/ml_alpha_symbol_calibration.csv
outputs/backtests/ml_alpha_walk_forward_calibration.csv
```

Each row is one historical decision point:

- `probability_up`: model probability that the next move is up
- `score`: probability transformed into a -1 to +1 directional score
- `prediction`: LONG, SHORT, or FLAT
- `actual`: next-bar direction after the label threshold
- `forward_return_bps`: next return in basis points
- `signed_return_bps`: return from following the prediction
- `training_accuracy`: in-sample fit on the rolling training window

## How To Read The Summary

`Coverage` is the percentage of scored rows where the model was confident enough
to say LONG or SHORT.

`Actionable accuracy` checks only those LONG/SHORT rows.

`Average signed return` is the more trading-like number. Positive means the
model's actionable calls pointed in the right direction on average before
transaction costs.

This report does not prove a strategy is profitable. It is a diagnostic tool:
it tells us whether the ML signal is worth trusting enough to include in the
router, tune further, or switch off.

## Current Result

On `data/downloaded_scan_prices.csv`, the raw ML signal had 67.2% coverage but
negative average signed return. A stricter in-sample calibration looked better,
but walk-forward calibration rejected promotion:

- positive fold rate: 25.0%
- average test signed return: -3.65 bps
- decision: keep ML disabled by default

The best in-sample calibration was mostly a metals result:

- XAGUSD: 63 actions, +2.86 bps average signed return
- XAUUSD: 74 actions, +0.79 bps average signed return
- most FX pairs: no action under the strict best filter

This is useful progress: it prevents us from promoting an overfit ML filter.

On the current seven-symbol candidate basket
`XAGUSD XAUUSD USDCHF AUDUSD GBPUSD EURUSD EURGBP`, a faster 3-epoch ML pass
also rejected promotion:

- raw average signed return: +0.01 bps
- best in-sample filter: +1.62 bps, mostly `XAGUSD`
- walk-forward folds: 17
- positive fold rate: 41.2%
- average test signed return: +0.24 bps
- decision: reject because fold stability is below the 60% promotion bar
