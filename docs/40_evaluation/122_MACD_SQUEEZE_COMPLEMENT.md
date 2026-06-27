# MACD Squeeze Complement

Date: 2026-06-23

Purpose:

```text
Strict MACD is still the strongest official-data strategy, but it has many flat
folds. Test whether a sparse volatility-squeeze sleeve can add return when MACD
is inactive.
```

Implementation:

- `MacdSqueezeComplementConfig`
- `MacdSqueezeComplementStrategy`
- Strategy name: `macd_squeeze_complement`
- Alias: `macd_squeeze`
- Config section: `[strategy.macd_squeeze_complement]`

Behavior:

```text
1. Ask MACD first.
2. If MACD has trade intent or says HOLD, keep MACD priority.
3. If MACD is inactive, allow volatility_squeeze to act.
4. Scale squeeze entries with squeeze_notional_multiplier.
5. Default safety gate: only allow squeeze after MACD reasons containing "below".
```

Why the safety gate matters:

```text
The ungated version allowed squeeze after any inactive MACD reason, including
session and neutral-state reasons. In full walk-forward replay this increased
drawdown and did not improve round coverage.
```

## Evidence

Baseline fixed-warmup run:

- `outputs/research/complement_macd_baseline_wf_summary.csv`
- `outputs/research/complement_macd_baseline_wf_folds.csv`

Standalone sleeve scan:

- `outputs/research/complement_sleeve_summary.csv`
- `outputs/research/complement_sleeve_detail.csv`

The standalone `volatility_squeeze` sleeve looked attractive in fold-complement
diagnostics:

```text
combined positive folds: 47.1%
positive on baseline-flat folds: 2
hurt positive MACD folds: 0
incremental return: +0.197%
```

But the embedded strategy is the authoritative test because it includes warmup
positions, position carry, allocation, risk, and strategy priority:

- `outputs/research/complement_macd_squeeze_fullsample_compare.csv`
- `outputs/research/complement_macd_squeeze_complement_wf_summary.csv`
- `outputs/research/complement_macd_squeeze_complement_wf_folds.csv`

Full-sample comparison:

| Strategy | Return | Max DD | Sharpe15 | Trades |
| --- | ---: | ---: | ---: | ---: |
| `macd_momentum` | 6.001% | 0.855% | 0.038 | 84 |
| `macd_squeeze_complement` | 5.659% | 1.598% | 0.035 | 98 |

Fixed-warmup comparison:

| Strategy | Positive folds | Active positive | Non-negative | Median active return | Worst DD |
| --- | ---: | ---: | ---: | ---: | ---: |
| `macd_momentum` | 35.3% | 66.7% | 82.4% | 0.641% | 0.824% |
| `macd_squeeze_complement` | 35.3% | 60.0% | 76.5% | 0.599% | 0.882% |

Variant scan:

- `outputs/research/complement_macd_squeeze_variant_scan.csv`

```text
m025_all   return 5.445%, drawdown 1.647%, active positive 54.5%
m050_all   return 5.359%, drawdown 1.632%, active positive 54.5%
m100_all   return 5.171%, drawdown 1.606%, active positive 54.5%
m025_below return 5.567%, drawdown 1.645%, active positive 60.0%
m050_below return 5.603%, drawdown 1.628%, active positive 60.0%
m100_below return 5.659%, drawdown 1.598%, active positive 60.0%
```

## Decision

Status: **PAPER_ONLY / do not promote**.

The new wrapper is useful infrastructure and a reminder that standalone
fold-complement diagnostics are only a first-pass screen. The actual embedded
portfolio replay is worse than strict MACD, so the competition profile remains
`macd_momentum`.

Next implication:

```text
The flat-fold solution probably needs either true crypto coverage or a separate
portfolio-level selector that can choose a sleeve per fold/window, not a naive
same-symbol fallback inside MACD.
```
