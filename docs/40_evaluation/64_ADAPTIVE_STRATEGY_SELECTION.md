# Adaptive Strategy Selection

`adaptive-strategy-select` is a walk-forward research tool for choosing between
complete portfolio strategies.

The question it answers:

```text
If we only knew the previous training window, which strategy would we have run
for the next unseen window?
```

This is different from `strategy-map-optimize`:

- `strategy-map-optimize` searches symbol -> strategy assignments.
- `adaptive-strategy-select` searches strategy -> time-window assignments.
- `adaptive-strategy-select --candidate-map` can also compare complete
  deployable maps as time-window candidates.
- `adaptive-strategy-select --recipe-map` can compare deployable maps with
  different symbol universes.

## Method

For each fold:

1. Slice a training window.
2. Backtest each candidate strategy on that training window.
3. Rank strategies by risk discipline, activity, drawdown-adjusted return,
   Sharpe, raw return, and drawdown.
4. Optionally apply training gates such as minimum train fills or minimum
   drawdown-adjusted train return.
5. Select the best recent eligible strategy.
6. Replay the selected strategy over train + test history.
7. Score only the test window.

The train window is used as warmup for indicators and positions, but only the
next test window is counted as out-of-sample evidence.

## Command

```bash
quanthack adaptive-strategy-select \
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
  --summary-output outputs/backtests/adaptive_strategy_selection_summary.csv \
  --folds-output outputs/backtests/adaptive_strategy_selection_folds.csv \
  --scores-output outputs/backtests/adaptive_strategy_selection_scores.csv \
  --stitched-equity-output outputs/backtests/adaptive_strategy_selection_stitched_equity.csv \
  --promotion-output outputs/backtests/adaptive_strategy_selection_promotion.csv
```

## How To Read It

Use the summary CSV first:

```text
positive_fold_fraction
active_positive_fold_fraction
non_negative_fold_fraction
median_active_test_return_pct
worst_test_drawdown_pct
selection_counts
```

Then inspect the folds CSV:

```text
selected_strategy
selected_train_return_pct
selected_train_drawdown_adjusted_return_pct
return_pct
max_drawdown_pct
risk_discipline_score
```

The scores CSV shows every candidate strategy's training score in every fold,
which helps explain why the selector picked a strategy.

Optional training gates:

```text
--min-train-fills N
--min-train-adjusted-return-pct X
--train-fill-penalty-pct X
--allow-cash-fallback
```

These gates prevent a candidate from being selected when its training run is too
thin or too weak. If every candidate fails the gates, the selector falls back to
the best raw training score so a fold is still evaluated. If
`--allow-cash-fallback` is enabled, the selector can instead sit out the next
fold with a flat cash allocation. The folds CSV records
`train_gate_blocked_strategies`, and the scores CSV records `train_gate_passed`.

`--train-fill-penalty-pct` is a softer churn guardrail. It subtracts a small
decimal return amount for every training-window fill before ranking candidates:

```text
training score = return - drawdown penalty - (fills * fill penalty)
```

On the current seven-symbol `kalman_trend / champion_ensemble / macd_momentum`
candidate, penalties from `0.000001` through `0.00002` did not change selection
counts or out-of-sample metrics. Verdict: keep it as a robustness probe; do not
enable it by default because it currently adds no improvement.

Cash fallback scan:

```text
candidate stack:
  kalman_trend, champion_ensemble, macd_momentum

no cash fallback:
  stitched OOS final equity: $1,004,225.57
  active positive folds: 72.7%

cash fallback with min_train_adjusted_return_pct = 0:
  cash selected in 9 of 17 folds
  stitched OOS final equity: $1,001,601.32
  active positive folds: 60.0%
```

Verdict: useful infrastructure and a defensive diagnostic, but too blunt for the
current top candidate.

For `--candidate-map`, the label is what appears as the candidate name, and the
score CSV includes a `strategy_map` column with the exact symbol recipe.

Use `--recipe-map` when a candidate should trade only the symbols listed in the
recipe. This is useful for comparing a conservative basket against a broader
candidate without forcing every strategy to run on every symbol.

The stitched equity CSV compounds test-window returns into a single
out-of-sample research curve. It is useful for the dashboard and demo, but it is
not a claim that positions were carried continuously across folds.

The promotion audit CSV lists each research/live gate separately. Use it to see
exactly why a run is `REJECT`, `PAPER_ONLY`, or `PROMOTE`.

## Promotion Rule Of Thumb

Treat adaptive selection as paper research until it beats the simple baseline
strategy in walk-forward validation.

Good signs:

```text
active positive folds >= 60%
non-negative folds >= 75%
median active return > 0
risk discipline near 100/100
selection counts are not dominated by one lucky fold
```

Bad signs:

```text
full-sample return improves but walk-forward median active return worsens
selector keeps chasing the previous fold's winner and then loses
one strategy is always selected and the tool adds no value
```

## Current Finding

Seven-symbol run:

```text
symbols:
  XAGUSD, XAUUSD, USDCHF, AUDUSD, GBPUSD, EURUSD, EURGBP

candidate strategies:
  kalman_trend, champion_ensemble, macd_momentum

folds:
  17
```

2026-06-23 recheck with session-filtered MACD and one-fold loss cooldown:

```text
command option: --loss-cooldown-folds 1
positive fold fraction: 41.2%
active fold fraction: 52.9%
active positive fold fraction: 77.8%
non-negative fold fraction: 88.2%
compounded OOS return: 5.245%
median active test return: 0.411%
worst test drawdown: 0.812%
risk discipline: 100/100
evaluation fills: 72
stitched OOS final equity: $1,052,445.87
selection counts: kalman_trend=7, champion_ensemble=3, macd_momentum=7
promotion: PAPER_ONLY
```

Selector variant check:

```text
current adaptive stack:
  compounded OOS return: 5.245%
  active positive folds: 77.8%
  non-negative folds: 88.2%

plus standalone volatility_squeeze:
  compounded OOS return: 4.845%
  active positive folds: 83.3%
  non-negative folds: 94.1%
  verdict: reject for now because it gives up OOS return

plus macd_squeeze_complement:
  compounded OOS return: 5.238%
  active positive folds: 75.0%
  non-negative folds: 88.2%
  verdict: near tie, but slightly weaker than current stack

train-stability-preferred variants:
  compounded OOS return range: 1.294% to 2.054%
  verdict: diagnostic only; do not use as primary selector ranking

cash fallback with min_train_adjusted_return_pct = 0:
  compounded OOS return: 5.530%
  active positive folds: 62.5%
  non-negative folds: 82.4%
  verdict: higher return, but weaker round consistency; challenger only

cash fallback with min_train_adjusted_return_pct = 0.0001:
  compounded OOS return: 4.841%
  active positive folds: 80.0%
  non-negative folds: 94.1%
  verdict: safer, but too idle and lower return than current leader

broad idle-sleeve scan:
  candidates added: quality_trend, trend_pullback, range_expansion_trend,
    relative_strength, cross_rate_reversion, fixing_reversal,
    liquidity_sweep_reversal, session_momentum, ma_crossover
  compounded OOS return: 4.540%
  active positive folds: 50.0%
  non-negative folds: 76.5%
  verdict: reject; adding many sleeves makes the selector chase weak recent wins

cached policy sweep:
  output: outputs/research/adaptive_strategy_policy_sweep.csv
  candidates: 36
  top policy: loss_cooldown_folds=1, no training-return gate,
    transition_risk_multiplier=1.0, no cash fallback
  compounded OOS return: 5.245%
  active positive folds: 77.8%
  non-negative folds: 88.2%
  verdict: confirms current baseline policy remains the paper leader

oracle diagnostic:
  output prefix: outputs/research/adaptive_current_top_oracle
  selected was oracle: 47.1%
  total regret: 5.355%
  largest regret: fold 14, where champion_ensemble beat macd_momentum by 3.391%
  verdict: largest regret was not cleanly harvestable because the oracle
    strategy had weak past evidence; use this to guide regime-transition
    research, not to force hindsight rules

handoff diagnostic:
  output: outputs/research/adaptive_current_top_handoff_diagnostic.csv
  largest miss label: HINDSIGHT_CHOP_BREAKOUT
  ex-ante fold 14 regime: 100% chop, 0% trend consensus
  verdict: next alpha target is a compression-breakout feature, not a simple
    champion handoff rule
```

No-cooldown result with session-filtered MACD:

```text
--loss-cooldown-folds 0:
  positive fold fraction: 41.2%
  active positive folds: 63.6%
  non-negative folds: 76.5%
  median active return: 0.033%
  worst drawdown: 0.071%
  promotion: PAPER_ONLY
```

Earlier cooldown check before the MACD session filter:

```text
One-fold cooldown was only a tiny improvement before session filtering.
Two-fold cooldown was too blunt and rejected.
```

Fair fixed-strategy comparison on the same folds:

```text
fixed kalman_trend:
  active positive folds: 55.6%
  non-negative folds: 76.5%
  median active return: 0.003%

fixed champion_ensemble:
  active positive folds: 44.4%
  non-negative folds: 70.6%
  median active return: -0.000%

fixed macd_momentum:
  active positive folds: 60.0%
  non-negative folds: 76.5%
  median active return: 0.017%
```

Quality-trend inclusion check:

```text
Added candidate:
  quality_trend

result:
  positive folds: 29.4%
  active positive folds: 62.5%
  non-negative folds: 82.4%
  stitched OOS final equity: $1,003,721.12
  selection counts: champion_ensemble=5, macd_momentum=7, quality_trend=5
```

This was worse than the current `kalman_trend / champion_ensemble /
macd_momentum` stack. `quality_trend` is safe but sparse; the adaptive selector
sometimes picked it into test windows where it produced no fills. Do not include
it in the current top paper candidate.

Range-expansion top-4 recipe check:

```text
Added recipe:
  range_expansion_top4:
    XAGUSD=range_expansion_trend
    XAUUSD=range_expansion_trend
    USDCHF=range_expansion_trend
    AUDUSD=range_expansion_trend

result:
  positive folds: 35.3%
  active positive folds: 66.7%
  non-negative folds: 82.4%
  median active return: 0.005%
  stitched OOS final equity: $1,003,527.17
  selection counts: champion_ensemble=5, macd_momentum=6, range_expansion_top4=6
```

This was safer than the rejected loose range-expansion profile, but still did
not beat the current top stack. A `--min-train-fills 6` variant reached
`$1,002,753.06`, also below the no-gate range recipe and below the current
leader. Keep it as a documented paper sleeve.

Verdict:

```text
Adaptive selection with session-filtered MACD and one-fold cooldown is the best
current paper candidate. It passes active-positive and non-negative validation
quality, but remains paper-only because total positive folds are below the
stricter 67% live gate.
```

Broad strategy-set scan:

```text
Adding dual_squeeze, asset_adaptive_dual_squeeze, fixing_reversal, and
trend_pullback worsened validation:

active positive folds: 25.0%
non-negative folds: 64.7%
median active return: -0.020%
```

Interpretation:

```text
The adaptive layer should use a small candidate set with prior evidence.
Adding many weaker sleeves makes the selector overfit the latest training
window.
```

Recent rejected alpha sleeves:

```text
session_momentum:
  best optimized default had positive full-sample return but weak walk-forward
  median active return. Keep as research infrastructure only.

intraday_seasonality:
  same-time-of-day momentum and reversal were both too noisy on the current
  seven-symbol sample. The adaptive selector naturally ignored it when included.
```

## Candidate Map Check

The selector can now compare static maps as first-class candidates:

```bash
quanthack adaptive-strategy-select \
  --strategy kalman_trend \
  --strategy champion_ensemble \
  --strategy macd_momentum \
  --candidate-map 'top5_static_map:XAGUSD=champion_ensemble,XAUUSD=macd_momentum,AUDUSD=macd_momentum,USDCHF=macd_momentum,EURUSD=macd_momentum' \
  --symbol XAGUSD --symbol XAUUSD --symbol AUDUSD --symbol USDCHF --symbol EURUSD \
  --price-csv data/full_20gb_15m_prices.csv \
  --quote-csv data/full_20gb_15m_quotes.csv \
  --train-size 480 \
  --test-size 96 \
  --step-size 96 \
  --loss-cooldown-folds 1 \
  --summary-output outputs/backtests/adaptive_candidate_map_top5_summary.csv \
  --folds-output outputs/backtests/adaptive_candidate_map_top5_folds.csv \
  --scores-output outputs/backtests/adaptive_candidate_map_top5_scores.csv \
  --stitched-equity-output outputs/backtests/adaptive_candidate_map_top5_equity.csv
```

Five-symbol result:

```text
candidates:
  kalman_trend, champion_ensemble, macd_momentum, top5_static_map

selection counts:
  kalman_trend=1
  champion_ensemble=5
  macd_momentum=8
  top5_static_map=3

active positive folds: 62.5%
non-negative folds: 82.4%
median active return: 0.029%
worst drawdown: 0.075%
stitched OOS final equity: $1,003,637.00
promotion: PAPER_ONLY
```

Interpretation:

```text
Candidate-map selection is useful and deployable, but this five-symbol map run
does not beat the seven-symbol adaptive selector with session-filtered MACD.
Keep it as backup evidence, not the top candidate.
```

## Recipe Map Check

Recipe maps allow partial symbol universes:

```bash
quanthack adaptive-strategy-select \
  --no-default-strategies \
  --recipe-map 'conservative_macd:AUDUSD=macd_momentum,EURCHF=macd_momentum,EURUSD=macd_momentum,USDCAD=macd_momentum,USDJPY=macd_momentum,XAGUSD=macd_momentum,XAUUSD=macd_momentum' \
  --recipe-map 'top5_static_map:XAGUSD=champion_ensemble,XAUUSD=macd_momentum,AUDUSD=macd_momentum,USDCHF=macd_momentum,EURUSD=macd_momentum' \
  --price-csv data/full_20gb_15m_prices.csv \
  --quote-csv data/full_20gb_15m_quotes.csv \
  --train-size 480 \
  --test-size 96 \
  --step-size 96 \
  --loss-cooldown-folds 1
```

Latest recipe-selector run:

```text
candidates:
  conservative_macd
  top5_static_map
  current7_macd
  xag_champion_scan_macd

positive folds: 29.4%
active positive folds: 55.6%
non-negative folds: 76.5%
median active return: 0.013%
stitched OOS final equity: $1,004,169.77
promotion: PAPER_ONLY
```

Interpretation:

```text
The feature is useful for fair static-recipe comparisons, but this recipe mix
does not beat the current top adaptive candidate.
```

## Training Gate Check

Training gates were added after recipe experiments over-selected sparse
candidates such as `multi_horizon_top3`.

Two checks were run:

```text
main plus clean recipes, min_train_fills=12:
  stitched OOS final equity: $1,002,768.91
  active positive folds: 55.6%
  non-negative folds: 76.5%
  median active return: 0.003%

main seven-symbol adaptive, min_train_adjusted_return=0.0:
  stitched OOS final equity: $1,003,914.26
  active positive folds: 63.6%
  non-negative folds: 76.5%
  median active return: 0.033%
```

Interpretation:

```text
Training gates are useful diagnostics and may prevent obvious sparse-candidate
mistakes, but they do not beat the current ungated one-fold-cooldown adaptive
candidate on this data.
```

## Training Stability Diagnostics

`adaptive-strategy-select` can now split each training window into chronological
subwindows and write stability diagnostics for every candidate:

```bash
quanthack adaptive-strategy-select \
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
  --train-stability-splits 4 \
  --prefer-train-stability
```

The scores CSV records:

```text
train_stability_active_fraction
train_stability_positive_fraction
train_stability_active_positive_fraction
train_stability_non_negative_fraction
train_stability_median_return_pct
train_stability_median_active_return_pct
```

Latest result:

```text
positive folds: 35.3%
active positive folds: 60.0%
non-negative folds: 76.5%
median active return: 0.003%
stitched OOS final equity: $1,001,788.05
```

Interpretation:

```text
Useful diagnostic infrastructure, but preferring training stability directly
over-selected locally clean candidates and did not beat the current leader.
Keep the flag off for the main paper candidate.
```

## Transition Risk Cap

`adaptive-strategy-select` can reduce actual target notionals for the evaluation
fold immediately after the selector changes strategy:

```bash
quanthack adaptive-strategy-select \
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
  --transition-risk-multiplier 0.75
```

The folds CSV records `evaluation_risk_multiplier`, so switch folds are visible.
The multiplier scales actual portfolio target notionals as well as risk budgets.

Latest checks:

```text
transition multiplier 0.50:
  stitched OOS final equity: $1,002,915.14
  median active return: 0.016%
  active positive folds: 72.7%

transition multiplier 0.75:
  stitched OOS final equity: $1,003,570.61
  median active return: 0.025%
  active positive folds: 72.7%

uncapped current leader:
  stitched OOS final equity: $1,004,225.57
  median active return: 0.033%
  active positive folds: 72.7%
```

Interpretation:

```text
The transition cap behaves correctly and reduces exposure on switch folds, but
it gives up too much return on the current data. Keep it as a defensive option,
not the main paper-candidate setting.
```

## Per-Symbol Adaptive Check

`--per-symbol-selection` adds a dynamic candidate named `per_symbol_adaptive`.
For each fold it scores the recent training window separately for each symbol
and builds a deployable map such as:

```text
XAGUSD=kalman_trend XAUUSD=kalman_trend USDCHF=macd_momentum ...
```

Use `--per-symbol-only` to force that dynamic map every fold:

```bash
quanthack adaptive-strategy-select \
  --strategy kalman_trend \
  --strategy champion_ensemble \
  --strategy macd_momentum \
  --per-symbol-selection \
  --per-symbol-only \
  --symbol XAGUSD --symbol XAUUSD --symbol USDCHF --symbol AUDUSD \
  --symbol GBPUSD --symbol EURUSD --symbol EURGBP \
  --price-csv data/full_20gb_15m_prices.csv \
  --quote-csv data/full_20gb_15m_quotes.csv \
  --train-size 480 \
  --test-size 96 \
  --step-size 96 \
  --loss-cooldown-folds 1
```

Latest forced dynamic-map run:

```text
positive folds: 29.4%
active positive folds: 55.6%
non-negative folds: 76.5%
median active return: 0.003%
worst drawdown: 0.106%
stitched OOS final equity: $1,003,938.32
promotion: PAPER_ONLY
```

Interpretation:

```text
Per-symbol selection is useful diagnostics, but the forced dynamic map did not
beat the current global adaptive selector. Keep it out of the main candidate
until it improves active-positive folds and drawdown.
```
