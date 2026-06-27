# Fold Regime Diagnostics

Date: 2026-06-23

`fold-regime-diagnostics` explains each fixed-warmup fold using only information
available before the fold's test window starts. It reads the fold CSV plus price
history, then writes:

- detail rows by `fold / symbol`;
- summary rows by `fold`;
- Kalman slope, trend efficiency, realized volatility, train return, lookback
  return, and asset-class net slope.

The purpose is to avoid making filters from hindsight. If a fold was profitable,
we want to know whether it looked different before trading started.

## Command

Core metals/USDCHF sleeve:

```bash
python -c 'from quanthack.cli.fold_regime_diagnostics import main; main()' \
  --price-csv data/full_20gb_15m_prices.csv \
  --folds-csv outputs/research/wf_core_plain_folds.csv \
  --symbol USDCHF --symbol XAGUSD --symbol XAUUSD \
  --detail-output outputs/research/core_plain_fold_regime_detail.csv \
  --summary-output outputs/research/core_plain_fold_regime_summary.csv
```

Baseline:

```bash
python -c 'from quanthack.cli.fold_regime_diagnostics import main; main()' \
  --price-csv data/full_20gb_15m_prices.csv \
  --folds-csv outputs/research/wf_baseline_folds.csv \
  --detail-output outputs/research/baseline_fold_regime_detail.csv \
  --summary-output outputs/research/baseline_fold_regime_summary.csv
```

## Latest Core Findings

The default Kalman regime labels every core fold as `CHOP`, including the best
fold. That means the binary regime label is too coarse for this data window.
The raw features are more useful.

Core plain summary:

| Fold | Return | Net slope bps | Avg abs slope bps | Avg efficiency | Avg vol bps | Metal slope bps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | -0.136% | -1.101 | 0.474 | 0.164 | 19.0 | -1.261 |
| 2 | 0.000% | -21.975 | 7.863 | 0.148 | 14.1 | -22.782 |
| 3 | 0.019% | -0.651 | 1.400 | 0.084 | 12.2 | 1.216 |
| 4 | -0.134% | -13.001 | 5.077 | 0.026 | 13.1 | -14.116 |
| 5 | 5.658% | -2.212 | 0.744 | 0.113 | 15.1 | -2.222 |
| 6 | 0.000% | 3.633 | 1.211 | 0.037 | 20.6 | 3.086 |

Interpretation:

- Fold 5 was a huge winner, but it did not have a clean ex-ante trend label.
- Folds 1 and 4 had similar negative metal pressure and still lost money.
- A simple `TREND_DOWN` or UTC-hour gate cannot reliably isolate the winning
  episode.

## Trend-Pullback Follow-Up

The fold diagnostics suggested testing a messier continuation/pullback profile.
The optimizer now includes low-efficiency, longer-lookback continuation profiles.

Focused core scan:

```text
best full-sample profile:
  messy_cont_l64_p8_t10_e0_10_pb1_35_r0_5_edge3
  return: 0.092%
  max drawdown: 0.214%
  trades: 58
```

Walk-forward result:

```text
messy_cont_l64_p8_t10_e0_10_pb1_35_r0_5_edge3
  median test return: -0.026%
  total test fills: 44
  eligible: false
```

Verdict: keep the expanded optimizer grid for research, but do not promote
trend-pullback. It adds activity, but not robust alpha.

## Next Research Direction

The next alpha improvement should not be a static regime label. Better options:

1. Symbol-level recent evidence: only allow a sleeve when the same symbol has
   shown positive recent out-of-sample contribution.
2. Portfolio-level complement checks: add sleeves only when they improve losing
   folds without concentrating the positive return in one fold.
3. Crypto coverage: official crypto data is still the biggest missing return
   source because crypto can trade when FX/metals are quiet.
