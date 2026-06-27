# Adaptive Handoff Diagnostic

`adaptive-handoff-diagnostic` merges three views:

```text
1. adaptive selector oracle folds
2. oracle candidate rows
3. ex-ante fold regime summaries
```

It labels selector misses into research buckets so we do not chase hindsight.

Command:

```bash
PYTHONPATH=src .venv/bin/python -c 'from quanthack.cli.adaptive_handoff_diagnostic import main; main()' \
  --oracle-folds-csv outputs/research/adaptive_current_top_oracle_folds.csv \
  --oracle-candidates-csv outputs/research/adaptive_current_top_oracle_candidates.csv \
  --regime-summary-csv outputs/research/adaptive_current_top_regime_summary.csv \
  --output outputs/research/adaptive_current_top_handoff_diagnostic.csv
```

2026-06-23 result:

```text
folds: 17
total regret: 5.355%

diagnosis counts:
  NO_REGRET: 11
  CASH_AVOIDABLE_LOSS: 2
  HINDSIGHT_CHOP_BREAKOUT: 2
  MACD_MISSED_AFTER_CHOP: 1
  CHAMPION_HANDOFF_MISS: 1
```

Largest miss:

```text
fold 14:
  selected: macd_momentum, +2.207%
  oracle: champion_ensemble, +5.598%
  regret: +3.391%
  ex-ante regime: 100% chop, 0% trend consensus
  champion train-adjusted return: -2.063%
  macd train-adjusted return: +0.625%
```

Interpretation:

```text
The largest regret is a chop-to-breakout event. The selector had a rational
reason to avoid champion_ensemble because champion's recent training score was
deeply negative. A simple "choose champion after chop" rule would be hindsight
overfit.
```

Next alpha direction:

```text
Build or refine a compression-breakout feature that is evaluated before the
test window opens:
  - low realized volatility / high chop fraction
  - narrowing range or Bollinger width
  - rising short-term momentum impulse
  - optional metal-specific confirmation

Then test it as a standalone paper sleeve and as a gated complement. Do not
promote it unless it beats the current adaptive stack on compounded OOS return
or improves non-negative folds without killing active-positive folds.
```
