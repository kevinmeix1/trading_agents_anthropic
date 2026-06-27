# Current Paper Candidate

This is the current best research profile from the imported 15-minute data.

## Top Candidate

Use adaptive strategy selection with:

```text
symbols:
  XAGUSD, XAUUSD, USDCHF, AUDUSD, GBPUSD, EURUSD, EURGBP

candidate strategies:
  kalman_trend
  champion_ensemble
  macd_momentum

MACD session filter:
  FX/metals 10-14 UTC

loss cooldown:
  1 fold
```

Latest validation:

```text
folds: 17
positive fold fraction: 47.1%
active fold fraction: 64.7%
active positive fold fraction: 72.7%
non-negative fold fraction: 82.4%
median active test return: 0.033%
worst test drawdown: 0.071%
risk discipline: 100/100
evaluation fills: 86
stitched OOS final equity: $1,004,225.57
promotion: PAPER_ONLY
blocking live gate: total positive folds 47.1% vs 67.0% required
```

Position stop-loss refresh:

```text
artifact prefix:
  outputs/backtests/adaptive_strategy_selection_session_macd_cooldown1_stoploss_*

result:
  headline metrics unchanged
  stitched OOS final equity: $1,004,225.57
  position_stop fills: 0
```

The new 1% entry-notional position stop did not fire on this candidate. Treat it
as a dormant safety guard on the current data, not a source of the reported
return.

Transition risk cap check:

```text
--transition-risk-multiplier 0.75:
  stitched OOS final equity: $1,003,570.61
  median active return: 0.025%

--transition-risk-multiplier 0.50:
  stitched OOS final equity: $1,002,915.14
  median active return: 0.016%
```

The cap reduces actual target notionals after selector switches, but it gives
up too much return here. Leave it off for the current main paper profile.

Portfolio volatility targeting check:

```text
baseline static map:
  final equity: $1,002,968.29
  return: 0.297%
  max drawdown: 0.118%
  official 15m Sharpe: 0.027

vol target 0.00035, max scale 1.00:
  final equity: $1,002,575.50
  return: 0.258%
  max drawdown: 0.106%
  official 15m Sharpe: 0.027

vol target 0.00015, max scale 1.00:
  final equity: $1,001,767.59
  return: 0.177%
  max drawdown: 0.069%
  official 15m Sharpe: 0.030
```

Volatility targeting is now available as a defensive overlay, but on this sample
it gives up too much return for the main return-heavy profile. Leave it off by
default unless live volatility rises materially. Full notes:
`docs/40_evaluation/73_PORTFOLIO_VOLATILITY_TARGETING.md`.

Regime tilt check:

```text
default regime tilt:
  return: 0.162%
  max drawdown: 0.101%
  official 15m Sharpe: 0.022

gentle regime tilt:
  return: 0.208%
  max drawdown: 0.095%
  official 15m Sharpe: 0.024
```

The Kalman-regime overlay works, but it resizes the current trend-heavy map too
often and gives up too much return. Leave it off for the main profile. Full
notes: `docs/40_evaluation/74_REGIME_TILT_RESULTS.md`.

Alpha-router behavior optimization:

```text
default router behavior:
  return: -0.468%
  max drawdown: 0.747%
  trades: 2282

conservative behavior:
  return: 0.112%
  max drawdown: 0.044%
  trades: 60

walk-forward:
  stable fold fraction: 0.0%
  median test return: 0.000%
  promotion: REJECT
```

The stricter router is a meaningful improvement over the old noisy router, but
it is still not better than the current paper candidate. Keep it in research.
Full notes: `docs/40_evaluation/75_ROUTER_BEHAVIOR_OPTIMIZATION.md`.

This is not automatic-live ready yet because total positive folds are still
below the stricter 67% live gate. It is the strongest paper/dry-run candidate.

Run `quanthack hackathon-readiness` for the combined go/no-go view. The current
readiness report also blocks full hackathon-live readiness because the local
15-minute data file has no crypto coverage for `BARUSD`, `BTCUSD`, `ETHUSD`,
`SOLUSD`, or `XRPUSD`.

## Aggressive Research Profile

`configs/competition.toml` now provides a higher-sizing research profile. The
10-symbol trend map reached a much stronger full-sample return:

```text
return: 3.318%
max drawdown: 1.962%
official 15m Sharpe: 0.019
trades: 178
risk discipline: 100/100
worst leverage: 5.28x
```

But the same map failed fixed-warmup walk-forward:

```text
positive fold fraction: 29.4%
active positive fold fraction: 35.7%
non-negative fold fraction: 47.1%
median active test return: -0.101%
promotion: REJECT
```

So the sizing profile is useful, but the aggressive map is not the main
candidate. Full notes:
`docs/40_evaluation/77_COMPETITION_PROFILE_AND_AGGRESSIVE_MAP.md`.

## Strongest Aggressive Paper Profile

The latest competition-profile strategy-map optimizer found a simpler all-MACD
portfolio across the 10 available FX/metals symbols:

```text
80% symbol cap:
  MACD minimum histogram: 2.5 bps
  FX position stop: 1.0%
  metal position stop: 2.0%
  return: 6.001%
  max drawdown: 0.855%
  official 15m Sharpe: 0.038
  risk discipline: 100/100
  worst leverage: 5.81x
  walk-forward non-negative folds: 82.4%
  walk-forward active-positive folds: 66.7%
  walk-forward median active return: 0.641%
```

This is stronger than the mixed aggressive map and is now the main aggressive
paper candidate, but it is still not automatic-live ready because total positive
folds remain below the stricter live gate and crypto coverage is still missing.
Full notes:
`docs/40_evaluation/79_ALL_MACD_AGGRESSIVE_CANDIDATE.md`.

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
  --summary-output outputs/backtests/adaptive_strategy_selection_session_macd_cooldown1_summary.csv \
  --folds-output outputs/backtests/adaptive_strategy_selection_session_macd_cooldown1_folds.csv \
  --scores-output outputs/backtests/adaptive_strategy_selection_session_macd_cooldown1_scores.csv \
  --stitched-equity-output outputs/backtests/adaptive_strategy_selection_session_macd_cooldown1_equity.csv \
  --promotion-output outputs/backtests/adaptive_strategy_selection_session_macd_cooldown1_promotion.csv
```

The stitched equity CSV is an out-of-sample research curve for the dashboard.
It compounds fold test-window returns into one curve, while still keeping fold
and selected-strategy columns for inspection.

The promotion audit CSV is the fastest way to explain the current paper-only
status in a demo or live-readiness review.

## Defensive MACD Variants

A controlled MACD/session refinement scan tested stricter histogram, slope, and
hour filters. The best defensive adaptive variant so far is `hist25`:

```text
MACD profile:
  fast / slow / signal: 6 / 18 / 5
  minimum histogram: 2.5 bps
  minimum MACD line: 1.0 bps
  minimum trend efficiency: 0.20
  max holding period: 12 bars
  session filter: 10-14 UTC

adaptive selector result:
  positive fold fraction: 41.2%
  active fold fraction: 52.9%
  active positive fold fraction: 77.8%
  non-negative fold fraction: 88.2%
  median active return: 0.033%
  worst drawdown: 0.071%
  evaluation fills: 72
  stitched OOS final equity: $1,003,519.23
```

This is cleaner than the return leader on active-positive and non-negative
folds, but it trails the current top stack's stitched OOS final equity of
`$1,004,225.57`. Keep it as a defensive paper backup, not the main profile.

The stricter `hist3` variant was even cleaner on non-negative folds but too
sparse:

```text
active positive folds: 75.0%
non-negative folds: 88.2%
stitched OOS final equity: $1,003,174.05
```

## Simpler Backup

If adaptive strategy selection feels too complex to operate, use the static
top-5 strategy map:

```text
XAGUSD=champion_ensemble
XAUUSD=macd_momentum
AUDUSD=macd_momentum
USDCHF=macd_momentum
EURUSD=macd_momentum
```

Validation:

```text
active positive folds: 62.5%
non-negative folds: 82.4%
median active return: 0.019%
max drawdown: 0.100%
risk discipline: 100/100
```

Command:

```bash
quanthack portfolio-fixed-warmup-walk-forward \
  --strategy champion_ensemble \
  --strategy-map XAUUSD=macd_momentum \
  --strategy-map AUDUSD=macd_momentum \
  --strategy-map USDCHF=macd_momentum \
  --strategy-map EURUSD=macd_momentum \
  --symbol XAGUSD --symbol XAUUSD --symbol AUDUSD --symbol USDCHF --symbol EURUSD \
  --price-csv data/full_20gb_15m_prices.csv \
  --quote-csv data/full_20gb_15m_quotes.csv \
  --train-size 480 \
  --test-size 96 \
  --step-size 96
```

CSV live dry-run check:

```bash
quanthack live-dry-run \
  --adapter csv \
  --strategy champion_ensemble \
  --strategy-map XAUUSD=macd_momentum \
  --strategy-map AUDUSD=macd_momentum \
  --strategy-map USDCHF=macd_momentum \
  --strategy-map EURUSD=macd_momentum \
  --symbol XAGUSD --symbol XAUUSD --symbol AUDUSD --symbol USDCHF --symbol EURUSD \
  --price-csv data/full_20gb_15m_prices.csv \
  --quote-csv data/full_20gb_15m_quotes.csv \
  --bars 120 \
  --iterations 1 \
  --journal outputs/top5_static_map_live_dry_run_journal.jsonl \
  --monitor-output outputs/top5_static_map_live_monitor.csv
```

This command is useful before MT5 work because it proves the live dry-run path
can run the same per-symbol strategy map used by the static backtest.

Adaptive map-selection check:

```text
The adaptive selector can now compare this static map as a whole candidate via
--candidate-map. On the five-symbol universe, it selected the map in 3 of 17
folds and produced:

active positive folds: 62.5%
non-negative folds: 82.4%
median active return: 0.029%
stitched OOS final equity: $1,003,637.00
promotion: PAPER_ONLY
```

That is solid backup evidence, but it does not beat the seven-symbol adaptive
candidate above.

## Conservative MACD Basket

A fresh universe scan found a clean MACD-only backup basket:

```text
AUDUSD
EURCHF
EURUSD
USDCAD
USDJPY
XAGUSD
XAUUSD
```

Fixed-warmup validation:

```text
positive fold fraction: 35.3%
active fold fraction: 41.2%
active positive fold fraction: 85.7%
non-negative fold fraction: 94.1%
median active return: 0.072%
worst drawdown: 0.052%
risk discipline: 100/100
evaluation fills: 62
promotion: PAPER_ONLY
```

This is more selective than the top adaptive candidate but cleaner when it
actually trades. Keep it as a conservative paper/live-dry-run comparison path.

Recipe-level adaptive selection can compare this basket against other static
recipes using `--recipe-map`. The first recipe-selection run reached stitched
OOS final equity of `$1,004,169.77`, close to the main adaptive candidate, but
with weaker active-positive validation:

```text
active positive folds: 55.6%
non-negative folds: 76.5%
median active return: 0.013%
promotion: PAPER_ONLY
```

So `--recipe-map` is useful infrastructure, but the main top candidate remains
the adaptive `kalman_trend / champion_ensemble / macd_momentum` run.

## Quality Trend Sleeve

`quality_trend` was added as a stricter MACD-plus-Kalman confirmation sleeve.
Standalone validation is safe but sparse:

```text
positive fold fraction: 17.6%
active positive fold fraction: 75.0%
non-negative folds: 94.1%
median active return: 0.023%
worst drawdown: 0.055%
fills: 30
promotion: PAPER_ONLY
```

Adding it to the adaptive selector reduced stitched OOS equity from
`$1,004,225.57` to `$1,003,721.12`, so the top candidate remains unchanged.

## Range Expansion Sleeve

`range_expansion_trend` was added as a stricter breakout-continuation sleeve.
The first loose profile churned and was rejected. The stricter default is
paper-only, especially after symbol eligibility:

```text
top-4 symbols:
  XAGUSD, XAUUSD, USDCHF, AUDUSD

positive fold fraction: 23.5%
active positive fold fraction: 57.1%
non-negative folds: 82.4%
median active return: 0.005%
worst drawdown: 0.053%
fills: 44
promotion: PAPER_ONLY
```

Adaptive selection with this top-4 recipe reached stitched OOS final equity of
`$1,003,527.17`, below the current top stack. Keep it as a paper sleeve and
diagnostic, not as part of the current best candidate.

## Multi-Horizon Momentum Backup

`multi_horizon_momentum` was added as a volatility-managed trend sleeve. Broad
seven-symbol use was rejected, but symbol eligibility found a cleaner top-3
basket:

```text
AUDUSD
USDCHF
XAUUSD
```

Top-3 fixed-warmup validation:

```text
positive fold fraction: 35.3%
active fold fraction: 58.8%
active positive fold fraction: 60.0%
non-negative fold fraction: 82.4%
median active return: 0.009%
worst drawdown: 0.039%
risk discipline: 100/100
evaluation fills: 72
promotion: PAPER_ONLY
```

Adaptive recipe selection can compare this top-3 basket with the main
strategies:

```text
recipe: multi_horizon_top3
stitched OOS final equity: $1,003,145.74
active positive folds: 54.5%
non-negative folds: 76.5%
median active return: 0.013%
promotion: PAPER_ONLY
```

This is a useful research/paper backup, but it does not beat the main adaptive
candidate or the conservative MACD basket.

## Avoid For Now

```text
broad adaptive selector:
  too many weaker sleeves caused overfitting

loss cooldown = 2:
  over-corrected and rejected

naive best_per_symbol_all map:
  worsened active-fold validation

automatic live MT5 execution:
  wait until paper/live dry-run evidence is stronger

session_momentum and intraday_seasonality:
  useful research infrastructure, but rejected by latest walk-forward evidence

broad multi_horizon_momentum:
  positive full-sample, but seven-symbol walk-forward active median was negative

usd_pressure_router:
  selected twice in adaptive testing but lowered stitched equity and fold quality

relative_strength and cross_rate_reversion:
  adaptive selector ignored them, so they are not additive on this sample

session_breakout:
  symbol eligibility was negative across broad baskets, with only inactive sparse variants

main plus clean recipes:
  over-selected sparse recipes and did not beat the main seven-symbol adaptive run

adaptive training gates:
  useful diagnostics, but min_train_fills=12 and min_train_adjusted_return=0.0
  both trailed the ungated one-fold-cooldown main candidate

adaptive training-stability preference:
  new diagnostics split the training window into chronological subwindows
  stability-preferred run trailed the leader:
    active positive folds: 60.0%
    non-negative folds: 76.5%
    stitched OOS final equity: $1,001,788.05

autocorrelation_regime:
  default seven-symbol fixed-warmup run was rejected
  optimizer found a sparse strict variant, but full-sample return stayed negative
  keep the optimizer as a diagnostic only
```
