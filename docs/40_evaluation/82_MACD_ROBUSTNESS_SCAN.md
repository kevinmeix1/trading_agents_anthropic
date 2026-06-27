# MACD Robustness Scan

Date: 2026-06-21

Objective: improve fold quality without adding leverage. The scan tested MACD
variants around the current session window, histogram threshold, slope filter,
and max holding period.

Command output:

- `outputs/research/macd_asset_stop_parameter_scan.csv`

## Best Result

`strict_hist25_10_14` is the best current MACD profile:

```text
fast=6
slow=18
signal=5
min_histogram_bps=2.5
min_macd_bps=1.0
min_trend_efficiency=0.20
max_holding_period=12
allowed UTC hours=10,11,12,13,14
```

Compared with the previous base MACD:

| Candidate | Return | Max DD | Sharpe15 | Trades | WF non-negative | WF active-positive | WF active median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base hist 2.0 | 5.896% | 1.694% | 0.035 | 110 | 70.6% | 54.5% | 0.336% |
| strict hist 2.5 | 6.001% | 0.855% | 0.038 | 84 | 82.4% | 66.7% | 0.641% |

This is a rare good kind of refinement: fewer trades, higher return, lower
drawdown, better Sharpe, and better fold quality.

## Remaining Weakness

Total positive folds are still only `35.3%`. The reason is inactivity: many
folds have zero fills and therefore count as flat, not positive. This means
MACD filtering has reached the point where extra strictness improves active
quality but cannot solve the score's per-round positive-return problem alone.

## Decision

Promote `min_histogram_bps = 2.5` into `configs/competition.toml`.

The next alpha work should focus on adding independent return streams for the
flat folds:

- crypto coverage, because official crypto trades 24/7;
- a low-turnover range/reversion sleeve for quiet FX/metals periods;
- regime gating so trend and reversion sleeves are not fighting each other.

## Recheck

Date: 2026-06-23

Focused recheck:

- `outputs/research/macd_targeted_parameter_scan_full20gb.csv`
- `outputs/research/macd_targeted_parameter_scan_mixed_overlap.csv`

The scan retested the current competition MACD against nearby alternatives:

```text
current:
  6/18/5
  histogram >= 2.5 bps
  MACD >= 1.0 bps
  trend efficiency >= 0.20
  max hold 12 bars
  UTC hours 10-14

nearby variants:
  wider UTC hours 9-15
  shifted UTC hours 12-16
  slope filter 0.25 bps
  looser histogram / trend filters
  smoother 8/21/8 MACD
```

Full official 10-symbol result:

| Candidate | Return | Max DD | Sharpe15 | WF active-positive | WF non-negative |
| --- | ---: | ---: | ---: | ---: | ---: |
| current 6/18/5 h2.5 10-14 | 6.001% | 0.855% | 0.038 | 100.0% | 100.0% |
| slope filter 0.25 | 5.471% | 0.860% | 0.035 | 100.0% | 100.0% |
| smoother 8/21/8 | 7.127% | 2.443% | 0.032 | 80.0% | 83.3% |
| wider hours 9-15 | 5.978% | 1.252% | 0.037 | 80.0% | 83.3% |

Mixed official plus crypto-proxy overlap:

| Candidate | Return | Max DD | Sharpe15 | WF active-positive | WF non-negative |
| --- | ---: | ---: | ---: | ---: | ---: |
| wider hours 9-15 | 0.538% | 2.410% | 0.011 | 100.0% | 100.0% |
| current 6/18/5 h2.5 10-14 | 0.508% | 2.231% | 0.010 | 100.0% | 100.0% |
| smoother 8/21/8 | 1.299% | 2.417% | 0.019 | 33.3% | 33.3% |

Decision:

```text
Keep configs/competition.toml unchanged.

The wider-hours and smoother variants are research-only. They add return in
some samples, but the official walk-forward quality is worse than the current
competition MACD.
```
