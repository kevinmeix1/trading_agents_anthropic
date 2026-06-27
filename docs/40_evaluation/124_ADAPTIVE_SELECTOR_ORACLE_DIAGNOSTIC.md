# Adaptive Selector Oracle Diagnostic

`adaptive-strategy-oracle` compares the adaptive selector's chosen strategy with
the ex-post best candidate on each unseen fold.

It answers:

```text
When the selector was wrong, was there a realistic ex-ante clue, or only
hindsight?
```

Command:

```bash
PYTHONPATH=src .venv/bin/python -c 'from quanthack.cli.adaptive_strategy_oracle import main; main()' \
  --config configs/competition.toml \
  --strategy kalman_trend \
  --strategy champion_ensemble \
  --strategy macd_momentum \
  --symbol XAGUSD --symbol XAUUSD --symbol USDCHF --symbol AUDUSD \
  --symbol GBPUSD --symbol EURUSD --symbol EURGBP \
  --price-csv data/full_20gb_15m_prices.csv \
  --quote-csv data/full_20gb_15m_quotes.csv \
  --train-size 480 \
  --test-size 96 \
  --step-size 96 \
  --loss-cooldown-folds 1 \
  --output-prefix outputs/research/adaptive_current_top_oracle
```

Artifacts:

```text
outputs/research/adaptive_current_top_oracle_summary.csv
outputs/research/adaptive_current_top_oracle_folds.csv
outputs/research/adaptive_current_top_oracle_candidates.csv
```

2026-06-23 result:

```text
folds: 17
selected was oracle: 47.1%
total regret: 5.355%
average regret: 0.315%
regret folds: 6
negative selected folds: 2
cash oracle folds: 2
policy compounded OOS return: 5.245%
```

Largest regret folds:

```text
fold 14:
  selected: macd_momentum, +2.207%
  oracle: champion_ensemble, +5.598%
  regret: +3.391%
  ex-ante clue: weak. champion_ensemble train-adjusted return was -2.063%,
    so choosing it from past evidence would likely be overfit.

fold 9:
  selected: kalman_trend, 0.000%
  oracle: macd_momentum, +1.142%
  regret: +1.142%
  ex-ante clue: partial. macd_momentum was flat in training while kalman_trend
    had a slightly negative train-adjusted score.

fold 11:
  selected: macd_momentum, -0.744%
  oracle: cash, 0.000%
  regret: +0.744%
  ex-ante clue: weak. macd_momentum had the strongest recent train score before
    the fold.
```

Interpretation:

```text
The selector leaves hindsight regret, but most of the largest regret is not
obviously tradable from past-only evidence.

The practical improvement is not to force the oracle strategy. The better path
is to add ex-ante regime features that can recognize when a previously weak
strategy is about to become valid, or when a strong recent trend is becoming
exhausted.
```

Current decision:

```text
Keep current adaptive stack as paper leader.
Do not add an oracle-inspired rule yet.
Next alpha research should focus on ex-ante regime transition features around
MACD/champion handoff, not broader strategy stuffing.
```

Follow-up:

```text
docs/40_evaluation/125_ADAPTIVE_HANDOFF_DIAGNOSTIC.md classifies these oracle
misses with ex-ante regime features. The largest miss is labeled
HINDSIGHT_CHOP_BREAKOUT, reinforcing that the next research target is a
compression-breakout detector, not an oracle-forcing selector rule.
```
