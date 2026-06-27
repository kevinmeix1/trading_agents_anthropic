# Strategy Research Sprint

This note records the current evidence from the full normalized 15-minute data
sample in `data/full_20gb_15m_prices.csv` and
`data/full_20gb_15m_quotes.csv`.

## Best Current Finding

Latest update: `champion_ensemble` is the best current paper candidate, with
`XAGUSD XAUUSD USDCHF AUDUSD GBPUSD` as the practical five-symbol basket.

```text
five-symbol full sample after no-churn allocator:
  return: 0.436%
  max drawdown: 0.147%
  Sharpe 15m: 0.031
  fills: 90
  risk discipline: 100/100

short fixed-warmup validation:
  folds: 17
  active fold fraction: 41.2%
  active positive fold fraction: 42.9%
  non-negative fold fraction: 76.5%
  median active test return: -0.003%
  promotion: REJECT
```

Interpretation:

```text
The strategy is selective, not continuously profitable. Zero-return no-trade
folds are acceptable, but automatic live MT5 execution still needs stronger
active-fold consistency.
```

## Position Stop-Loss Guardrail

Added a reusable position cost-basis tracker and a per-position stop-loss check.
The default risk limit is `max_position_loss_pct = 0.01`, measured as open loss
divided by entry notional.

Where it runs:

```text
single-symbol backtest:
  stop check before strategy on each bar

portfolio backtest:
  stop check becomes a symbol intent targeting flat
```

Candidate refresh:

```text
adaptive strategy selector, same seven-symbol leader:
  stitched OOS final equity: $1,004,225.57
  positive folds: 47.1%
  active positive folds: 72.7%
  position_stop fills: 0
```

Interpretation:

```text
The guard did not change the current leader. It improves risk discipline and
future MT5 readiness, but it should not be counted as alpha.
```

## Adaptive Transition Risk Cap

Added `--transition-risk-multiplier` to adaptive selection. When the selected
strategy changes from the previous fold, the next evaluation fold scales actual
target notionals and risk budgets by the multiplier.

Seven-symbol leader check:

```text
uncapped:
  stitched OOS final equity: $1,004,225.57
  median active return: 0.033%

transition multiplier 0.75:
  stitched OOS final equity: $1,003,570.61
  median active return: 0.025%

transition multiplier 0.50:
  stitched OOS final equity: $1,002,915.14
  median active return: 0.016%
```

Interpretation:

```text
The feature is correctly reducing switch-fold exposure, but the current leader's
positive switch folds matter. Keep transition caps as a defensive/live-protection
option, not as the default paper candidate.
```

## MACD Momentum Finding

`macd_momentum` was added and optimized as a momentum-acceleration sleeve.

Best short walk-forward parameter set:

```text
fast / slow / signal: 6 / 18 / 5
minimum histogram: 2.0 bps
minimum MACD line: 1.0 bps
minimum trend efficiency: 0.20
max holding period: 12 bars
session filter: 10-14 UTC for FX/metals
```

Session-filtered seven-symbol evidence:

```text
symbols:
  XAGUSD, XAUUSD, USDCHF, AUDUSD, GBPUSD, EURUSD, EURGBP

full sample:
  return: 0.271%
  max drawdown: 0.127%
  Sharpe 15m: 0.026
  fills: 78

short fixed-warmup validation:
  active fold fraction: 58.8%
  active positive fold fraction: 60.0%
  non-negative fold fraction: 76.5%
  median active test return: 0.017%
  promotion: PAPER_ONLY
```

Champion blend scan:

```text
20% optimized MACD blend:
  full-sample return: 0.557%
  max drawdown: 0.140%
  fills: 84

wider walk-forward:
  active median worsened versus baseline
```

Research verdict:

```text
Keep MACD as a tested research sleeve, optimizer target, and second paper
candidate. Do not promote MACD into champion defaults yet.
```

Hybrid map check:

```text
champion on XAGUSD/XAUUSD and MACD on AUDUSD/USDCHF/EURUSD/EURGBP:
  full-sample return: 0.235%
  max drawdown: 0.105%
  active positive fold fraction: 44.4%
  median active test return: -0.005%
  promotion: REJECT
```

Interpretation:

```text
The new strategy-map tool is useful, but this first hybrid diluted the strong
metal champion trades and did not pass walk-forward. Keep champion and MACD as
separate paper candidates for now.
```

## Strategy Map Optimization Finding

`strategy-map-optimize` now builds controlled per-symbol strategy maps from
full-portfolio symbol attribution.

Seven-symbol scan:

```text
symbols:
  XAGUSD, XAUUSD, USDCHF, AUDUSD, GBPUSD, EURUSD, EURGBP
strategies:
  champion_ensemble, macd_momentum, kalman_trend
```

Best walk-forward-ranked map after the session-filtered MACD update:

```text
top_5_best_symbol_strategies:
  XAGUSD=champion_ensemble
  XAUUSD=macd_momentum
  AUDUSD=macd_momentum
  USDCHF=macd_momentum
  EURUSD=macd_momentum

  return: 0.308%
  max drawdown: 0.100%
  active positive fold fraction: 62.5%
  non-negative fold fraction: 82.4%
  median active return: 0.019%
```

Naive best-per-symbol hybrid:

```text
best_per_symbol_all:
  return: 0.264%
  active positive fold fraction: 45.5%
  median active return: -0.011%
```

Research verdict:

```text
The top-5 static map is now a simpler paper backup candidate. It is easier to
operate than adaptive selection, but the adaptive selector still has stronger
active-fold validation.
```

## Adaptive Strategy Selection Finding

`adaptive-strategy-select` now tests a simple online research idea:

```text
Pick the strategy with the best recent training-window evidence, then score
only the next unseen fold.
```

Seven-symbol scan:

```text
symbols:
  XAGUSD, XAUUSD, USDCHF, AUDUSD, GBPUSD, EURUSD, EURGBP
strategies:
  kalman_trend, champion_ensemble, macd_momentum
```

Result:

```text
folds: 17
loss cooldown: 1 fold
positive fold fraction: 47.1%
active fold fraction: 64.7%
active positive fold fraction: 72.7%
non-negative fold fraction: 82.4%
median active test return: 0.033%
worst test drawdown: 0.071%
risk discipline: 100/100
evaluation fills: 86
stitched OOS final equity: $1,004,225.57
selection counts: kalman_trend=2, champion_ensemble=5, macd_momentum=10
promotion: PAPER_ONLY
```

Fair fixed-strategy comparison on the same folds:

```text
kalman_trend median active return: 0.003%
champion_ensemble median active return: -0.000%
session-filtered macd_momentum median active return: 0.017%
adaptive selector median active return: 0.033%
```

Research verdict:

```text
This is the best current validation profile. Keep it as the top paper
candidate, but do not switch to automatic live MT5 execution until total
positive folds improve or it remains stable on more data.
```

Quality-trend sleeve check:

```text
quality_trend intuition:
  require MACD momentum and Kalman trend to agree before entering

standalone seven-symbol fixed-warmup:
  positive fold fraction: 17.6%
  active fold fraction: 23.5%
  active positive fold fraction: 75.0%
  non-negative fold fraction: 94.1%
  median active return: 0.023%
  worst drawdown: 0.055%
  risk discipline: 100/100
  evaluation fills: 30
  promotion: PAPER_ONLY

adaptive selector with quality_trend added:
  positive fold fraction: 29.4%
  active positive fold fraction: 62.5%
  non-negative fold fraction: 82.4%
  stitched OOS final equity: $1,003,721.12
```

Research verdict:

```text
Keep quality_trend as a conservative diagnostic sleeve. It is clean when active
but too sparse, and adding it to adaptive selection reduced the current best
stitched OOS result.
```

Conditional-seasonality check:

```text
conditional_seasonality intuition:
  use prior same-time-of-day examples whose 4-bar momentum condition matches
  the current bar, then trade/fade their historical 4-bar forward drift

direct momentum default prototype:
  active folds: 100.0%
  median active return: -0.067%
  fills: 1712
  verdict: REJECT

conservative reversal default:
  positive fold fraction: 5.9%
  active positive fold fraction: 25.0%
  non-negative fold fraction: 82.4%
  median active return: -0.001%
  worst drawdown: 0.026%
  fills: 20
  verdict: REJECT
```

Research verdict:

```text
The implementation is leakage-safe and useful for future diagnostics, but the
tested signal does not generate enough positive active folds. Do not add it to
adaptive selection or live dry-run candidates.
```

Range-expansion trend check:

```text
range_expansion_trend intuition:
  trade only when the latest short impulse breaks the prior range and is
  unusually large versus baseline volatility

loose first profile:
  active folds: 94.1%
  active positive folds: 43.8%
  non-negative folds: 47.1%
  fills: 202
  verdict: REJECT

stricter default profile:
  min_trigger_move_bps: 10.0
  min_range_break_bps: 3.0
  min_expansion_zscore: 2.5
  min_trend_efficiency: 0.65
  max_holding_period: 6

seven-symbol fixed-warmup:
  positive folds: 29.4%
  active positive folds: 50.0%
  non-negative folds: 70.6%
  median active return: 0.003%
  fills: 80
  promotion: PAPER_ONLY

top-4 eligible basket:
  XAGUSD, XAUUSD, USDCHF, AUDUSD
  active positive folds: 57.1%
  non-negative folds: 82.4%
  median active return: 0.005%
  worst drawdown: 0.053%
  fills: 44
  promotion: PAPER_ONLY

adaptive selector with range_expansion_top4:
  active positive folds: 66.7%
  non-negative folds: 82.4%
  stitched OOS final equity: $1,003,527.17

min-train-fills 6 gate:
  stitched OOS final equity: $1,002,753.06
```

Research verdict:

```text
The stricter range-expansion profile is a valid paper sleeve and improves
stability after symbol eligibility, but adaptive inclusion still trails the
current top stack at $1,004,225.57. Keep it documented; do not include it in
the current top candidate.
```

Adaptive candidate-map extension:

```text
The adaptive selector can now compare deployable static maps as first-class
candidates with --candidate-map.

Five-symbol top5_static_map experiment:
  selected top5_static_map in 3 of 17 folds
  active positive folds: 62.5%
  non-negative folds: 82.4%
  median active return: 0.029%
  stitched OOS final equity: $1,003,637.00
  promotion: PAPER_ONLY
```

Interpretation:

```text
Useful backup evidence and an operationally simpler recipe, but still below the
seven-symbol adaptive selector with session-filtered MACD.
```

## Session Momentum And Intraday Seasonality Finding

Two additional alpha ideas were implemented and tested:

```text
session_momentum:
  simple momentum restricted to configurable UTC sessions
  optimizer command: quanthack session-momentum-optimize

intraday_seasonality:
  same-time-of-day return pattern using prior 15-minute daily slots
  supports momentum and reversal modes
```

Latest seven-symbol evidence:

```text
session_momentum best optimizer candidate:
  full-sample return: 0.104%
  walk-forward active positive folds: 50.0%
  walk-forward non-negative folds: 70.6%
  median active test return: -0.006%
  verdict: REJECT

intraday_seasonality reversal mode:
  active fold fraction: 100.0%
  positive fold fraction: 29.4%
  non-negative fold fraction: 29.4%
  median active test return: -0.071%
  verdict: REJECT
```

Adaptive check:

```text
Adding intraday_seasonality to the best adaptive candidate set did not change
the selected folds or stitched OOS final equity. The selector ignored it, which
is the right behavior for a weak sleeve.
```

Research verdict:

```text
Keep both as modular research infrastructure, but do not include them in the
current paper/live candidate set.
```

## Universe Rescan Finding

After adding the rejected sleeves, a broader universe scan over the imported
10-symbol FX/metals sample was rerun for:

```text
champion_ensemble
kalman_trend
macd_momentum
```

Best full-sample basket:

```text
strategy: macd_momentum
symbols: AUDUSD EURCHF EURUSD USDCAD USDJPY XAGUSD XAUUSD
full-sample return: 0.398%
full-sample drawdown: 0.052%
full-sample Sharpe 15m: 0.039
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
promotion: PAPER_ONLY
```

Research verdict:

```text
This MACD basket is a strong conservative backup. It is not the main candidate
because it is very selective, but its active folds are cleaner than the current
top adaptive run.
```

Loss-cooldown check before the MACD session filter:

```text
one-fold cooldown:
  median active return: 0.018%
  active positive folds: 60.0%
  non-negative folds: 76.5%
  promotion: PAPER_ONLY

two-fold cooldown:
  median active return: -0.012%
  active positive folds: 44.4%
  promotion: REJECT
```

Interpretation:

```text
After the MACD session filter, one-fold cooldown is useful because it skips a
known follow-on MACD loss. Two-fold cooldown still over-corrects and should not
be used.
```

Broad candidate warning:

```text
Adding dual_squeeze, asset_adaptive_dual_squeeze, fixing_reversal, and
trend_pullback worsened validation. Keep the selector candidate set small.
```

Older checkpoint:

`dual_squeeze` is the strongest conservative default candidate.

It is a conservative extension of `volatility_squeeze`: the fast squeeze creates
the entry, and a slower 24-price squeeze/bias filter must confirm the direction.

Latest all-symbol comparison:

```text
dual_squeeze:
  return: 0.124%
  max drawdown: 0.055%
  Sharpe 15m: 0.023
  trades: 48
  risk discipline: 100/100
```

This is an improvement over the old `volatility_squeeze` checkpoint:

```text
volatility_squeeze:
  return: 0.027%
  max drawdown: 0.021%
  Sharpe 15m: 0.015
  trades: 46
```

Important caveat:

```text
dual_squeeze is still sparse in walk-forward.
Use it as the best paper-trading candidate, not as an automatic live-MT5 switch.
```

Latest eligible-basket run:

```text
symbols: XAGUSD, XAUUSD, USDCHF, GBPUSD, AUDUSD, EURCHF, USDJPY, USDCAD
return: 0.137%
max drawdown: 0.049%
Sharpe 15m: 0.026
fills: 40
risk discipline: 100/100
```

Warmup-aware scoring from `2026-05-21T00:00:00+00:00`:

```text
return after warmup: 0.083%
Sharpe 15m after warmup: 0.020
evaluation fills: 28
risk discipline: 100/100
```

That means the current candidate is below the 30-trade Sharpe-prize threshold
after this warmup start. It needs either more robust activity or a different
evaluation window before promotion.

Fixed warmup walk-forward on the eligible basket:

```text
command: quanthack portfolio-fixed-warmup-walk-forward
train/test/step: 960 / 192 / 192 timestamps
folds: 6
positive fold fraction: 16.7%
median test return: 0.000%
worst test drawdown: 0.031%
evaluation fills: 12
risk discipline: 100/100 average
```

Research verdict:

```text
dual_squeeze remains the best current candidate, but it is not robust enough
for automatic live MT5 execution. Next alpha work should focus on independent
activity sources or a better portfolio-level router gate, not more leverage.
```

The best current full-sample research variant is now
`asset_adaptive_dual_squeeze`, which applies a faster dual-squeeze profile only
to metals:

```text
all-symbol comparison:
  return: 0.138%
  max drawdown: 0.061%
  Sharpe 15m: 0.023
  fills: 52
  rank: 1 / 15 current strategy candidates

eligible-basket comparison:
  symbols: XAGUSD, XAUUSD, USDCHF, GBPUSD, AUDUSD, EURCHF, USDJPY, USDCAD
asset_adaptive_dual_squeeze:
  return: 0.156%
  max drawdown: 0.049%
  Sharpe 15m: 0.026
  fills: 44
  risk discipline: 100/100
```

Its warmup walk-forward still rejects promotion:

```text
positive fold fraction: 16.7%
median test return: 0.000%
evaluation fills: 12
```

Latest optimizer command:

```bash
python scripts/evaluation/dual_squeeze_optimize.py \
  --price-csv data/full_20gb_15m_prices.csv \
  --quote-csv data/full_20gb_15m_quotes.csv \
  --walk-forward \
  --output outputs/backtests/dual_squeeze_walk_forward_optimization.csv
```

Best parameter set:

```text
fast_confirm_l14_w4_r0_60_b2_5_c24_cw8_cr0_70:
  return: 0.124%
  max drawdown: 0.055%
  trades: 48
  walk-forward stable folds: 16.7%
  walk-forward eligible: false
```

## Router Finding

The router can now optimize an explicit volatility-squeeze weight.

Full-data router optimization with squeeze candidates:

```text
squeeze-only router:
  return: 0.025%
  max drawdown: 0.007%
  Sharpe 15m: 0.022
  trades: 28
  risk discipline: 100/100
```

Blending squeeze with older noisy sleeves hurt performance. The current research
view is:

```text
squeeze-only > squeeze blended with old router sleeves
```

Router walk-forward selected squeeze-only consistently, but the stricter
promotion gate rejects it because median test return is economically tiny.

## Trend Pullback Finding

`trend_pullback` was added as a new continuation strategy:

```text
larger trend -> controlled pullback -> resume move with trend
```

Full-data strategy comparison:

```text
trend_pullback:
  return: -0.105%
  max drawdown: 0.268%
  trades: 230
```

The optimizer found one positive full-sample variant:

```text
ny_l32_p4_t8_e0_40_pb1_12_r1_edge3:
  return: 0.040%
  max drawdown: 0.083%
  trades: 98
```

But walk-forward did not support promoting that NY-hours variant. A faster
variant was walk-forward-eligible on selected baskets but negative on the
all-symbol full sample.

Research verdict:

```text
trend_pullback stays in research; do not promote to live MT5 yet.
```

## Fixing Reversal Finding

`fixing_reversal` was added as an intraday FX/metals research sleeve:

```text
strong pre-window move -> opposite confirmation bar -> short-lived fade
```

The first naive UTC profile lost broadly. A compact grid search found a
tiny-positive profile:

```text
hours: 14 UTC
pre_fix_lookback: 4
min_pre_fix_move_bps: 8.0
min_reversal_confirmation_bps: 1.5
min_pre_fix_efficiency: 0.35
max_holding_period: 4 bars
```

Eligible-basket full sample:

```text
symbols: XAGUSD, XAUUSD, USDCHF, GBPUSD, AUDUSD, USDJPY, USDCAD
return: 0.023%
max drawdown: 0.043%
official 15m Sharpe: 0.009
fills: 92
risk discipline: 100/100
```

Fixed warmup walk-forward:

```text
positive fold fraction: 33.3%
median test return: 0.000%
median test Sharpe 15m: 0.000
evaluation fills: 60
```

Research verdict:

```text
fixing_reversal is a useful tested sleeve, but not promoted. The edge is too
small and forward stability is not strong enough.
```

## Kalman Trend Finding

`kalman_trend` was added as a standalone advanced time-series strategy:

```text
Kalman-smoothed trend slope + path efficiency -> trend-following target
```

Grid search found a better profile than the strict default:

```text
lookback: 80
min_abs_slope_bps: 0.25
min_trend_efficiency: 0.20
min_expected_edge_bps: 5.0
expected_holding_bars: 6
max_holding_period: 32
```

Eight-symbol eligible basket:

```text
symbols: XAGUSD, XAUUSD, USDCHF, GBPUSD, AUDUSD, EURCHF, USDJPY, USDCAD
return: 0.227%
max drawdown: 0.202%
official 15m Sharpe: 0.019
fills: 192
risk discipline: 100/100
```

Positive-attribution five-symbol basket:

```text
symbols: XAGUSD, XAUUSD, USDCHF, AUDUSD, GBPUSD
return: 0.357%
max drawdown: 0.104%
official 15m Sharpe: 0.032
fills: 188
risk discipline: 100/100
```

Latest all-strategy comparison on the five-symbol basket:

```text
rank 1 / 17: kalman_trend
return: 0.357%
max drawdown: 0.104%
official 15m Sharpe: 0.032
trades: 188
```

Fixed warmup walk-forward on that five-symbol basket:

```text
positive fold fraction: 50.0%
median test return: 0.001%
median test Sharpe 15m: 0.002
evaluation fills: 108
worst drawdown: 0.078%
```

Research verdict:

```text
kalman_trend is the strongest standalone research sleeve and the fallback behind
champion_ensemble. It is more active and higher-return than dual_squeeze, but
still not robust enough for blind automatic MT5 execution because one fold
contributes much of the gain.
```

## Champion Ensemble Finding

`champion_ensemble` was added after `kalman_trend` became the strongest
standalone sleeve. The first loose profile allowed asset-adaptive squeeze to
trade alone:

```text
loose profile:
  return: 0.607%
  max drawdown: 0.124%
  official 15m Sharpe: 0.040
  trades: 242
  walk-forward positive folds: 33.3%
```

That was rejected as too clustered. The current default is stricter:

```text
kalman_trend_weight: 0.70
asset_adaptive_dual_squeeze_weight: 0.30
dual_squeeze_weight: 0.00
trend_pullback_weight: 0.00
entry_score: 0.50
strong_lead_score: 0.50
conflict_penalty: 0.70
```

Positive-attribution five-symbol basket:

```text
symbols: XAGUSD, XAUUSD, USDCHF, AUDUSD, GBPUSD
return: 0.475%
max drawdown: 0.113%
official 15m Sharpe: 0.035
fills: 186
risk discipline: 100/100
```

Fixed warmup walk-forward:

```text
positive fold fraction: 50.0%
median test return: 0.001%
median test Sharpe 15m: 0.001
evaluation fills: 106
worst drawdown: 0.090%
largest positive fold contribution: about 99%
promotion: PAPER_ONLY
```

Latest all-strategy comparison on the five-symbol basket:

```text
rank 1 / 18: champion_ensemble
return: 0.475%
max drawdown: 0.113%
official 15m Sharpe: 0.035
trades: 186
```

Research verdict:

```text
champion_ensemble is the best current paper candidate. It improves full-sample
return versus standalone kalman_trend while preserving the same positive-fold
fraction. Keep it in dry-run/manual validation before automatic MT5 execution.
```

## Data Coverage Finding

The downloaded 15-minute backtest CSV currently covers only FX and metals:

```text
AUDUSD, EURCHF, EURGBP, EURUSD, GBPUSD, USDCAD, USDCHF, USDJPY, XAGUSD, XAUUSD
```

The stricter full-competition validation command:

```bash
quanthack-validate-data \
  --price-csv data/full_20gb_15m_prices.csv \
  --quote-csv data/full_20gb_15m_quotes.csv \
  --competition-symbols \
  --output outputs/backtests/full_20gb_competition_data_health.csv
```

currently fails because crypto rows are missing:

```text
BARUSD, BTCUSD, ETHUSD, SOLUSD, XRPUSD
```

Research implication:

```text
Do not claim crypto backtest evidence from this dataset. Crypto strategy work is
structurally supported, but real crypto validation needs MT5/organizer data.
```

## June 21 Research Branches

Adaptive squeeze fallback:

```text
idea: allow volatility-squeeze fallback entries only when Kalman trend agrees
best static result: 0.140% return on eligible basket
walk-forward result: still sparse, 1/6 positive folds in fixed test windows
verdict: not promoted; improvement is too clustered
```

Rolling intraday seasonality:

```text
idea: learn same-hour forward-return bias using only prior bars
result: broad variants overtraded and lost after costs; strict metals variants
were mostly inactive
verdict: research artifact, not a strategy candidate
```

Session filtering:

```text
idea: restrict dual_squeeze to observed profitable UTC entry hours
result: no improvement over the current 11-19 UTC default
verdict: keep current dual_squeeze session settings
```

Per-symbol dual-squeeze tuning:

```text
idea: select a parameter set per symbol using a research allocation policy
result: individually selected settings collapsed when recombined in the portfolio
verdict: optimize at portfolio level, not symbol-by-symbol
```

Metal-fast / FX-default dual squeeze:

```text
idea: use faster, looser dual_squeeze settings only for metals, while keeping
current default dual_squeeze settings for FX
full-sample result: 0.156% return, 0.049% max drawdown, 44 fills, 100/100 risk
warmup walk-forward: 1/6 positive folds, 12 evaluation fills
verdict: promising full-sample research variant, but not robust enough to
replace the default configuration yet
```

Promoted tooling:

```text
portfolio_fills.csv: trade-level audit report from portfolio backtests
--metrics-start: warmup-aware competition scoring view
portfolio-fixed-warmup-walk-forward: fold-by-fold warmup-aware validation
```

## Attribution Finding

Per-symbol attribution was generated with:

```text
outputs/backtests/full_20gb_volatility_squeeze_attribution_pnl.csv
outputs/backtests/full_20gb_trend_pullback_attribution_pnl.csv
```

Reusable command:

```bash
python scripts/evaluation/strategy_attribution.py \
  --price-csv data/full_20gb_15m_prices.csv \
  --quote-csv data/full_20gb_15m_quotes.csv \
  --strategy volatility_squeeze \
  --strategy trend_pullback \
  --output outputs/backtests/full_20gb_strategy_attribution.csv
```

For `volatility_squeeze`, most of the profit came from metals:

```text
XAGUSD: +$150.66
XAUUSD: +$82.64
```

For `trend_pullback`, metals were also positive, but most FX pairs were negative:

```text
XAGUSD: +$445.05
XAUUSD: +$131.50
EURUSD: -$513.16
USDCHF: -$395.94
GBPUSD: -$317.54
```

Single-symbol checks confirmed that XAUUSD and XAGUSD pullback can be profitable,
but they trigger risk-discipline concentration penalties when traded alone. A
hybrid test using squeeze for FX and pullback for metals did not improve the
portfolio because isolated metal pullback signals were trimmed by the allocator
when there was not enough concurrent diversification.

Updated interpretation:

```text
pullback is a possible metals alpha, but only if paired with enough real,
positive-expectancy diversifying signals.
```

## Exhaustion Reversal Finding

`exhaustion_reversal` was added to test a shock-and-reversal idea:

```text
large short-window shock -> first reversal bar -> trade against the shock
```

Default result:

```text
exhaustion_reversal:
  return: -0.269%
  max drawdown: 0.372%
  trades: 231
  risk discipline: 100/100
```

A stricter 72-candidate grid did not rescue it:

```text
best stricter candidate:
  return: -0.052%
  trades: 10
```

Research verdict:

```text
keep as a documented rejected sleeve; do not promote.
```

## Dual Squeeze Attribution

Reusable command:

```bash
python scripts/evaluation/strategy_attribution.py \
  --price-csv data/full_20gb_15m_prices.csv \
  --quote-csv data/full_20gb_15m_quotes.csv \
  --strategy dual_squeeze \
  --strategy volatility_squeeze \
  --output outputs/backtests/full_20gb_dual_squeeze_attribution.csv
```

Attribution checkpoint:

```text
XAGUSD: +$485.67
XAUUSD: +$474.22
USDCHF: +$140.98
GBPUSD: +$139.45
AUDUSD: +$70.12
EURUSD: -$102.32
```

## Symbol Eligibility Finding

The current `dual_squeeze` attribution shows that EURUSD is the main drag. A
symbol-eligibility optimization was run with:

```bash
python scripts/evaluation/symbol_eligibility_optimize.py \
  --strategy dual_squeeze \
  --price-csv data/full_20gb_15m_prices.csv \
  --quote-csv data/full_20gb_15m_quotes.csv \
  --min-symbols 3 \
  --output outputs/backtests/dual_squeeze_symbol_eligibility.csv
```

Best filtered universe:

```text
positive_active:
  symbols: XAGUSD, XAUUSD, USDCHF, GBPUSD, AUDUSD, EURCHF, USDJPY, USDCAD
  excluded: EURGBP, EURUSD
  return: 0.137%
  max drawdown: 0.049%
  Sharpe 15m: 0.026
  trades: 40
  risk discipline: 100/100
```

Fixed-basket walk-forward:

```text
positive_active stable folds: 33.3%
top_5 stable folds: 33.3%
promotion: REJECT
```

Updated interpretation:

```text
symbol gating improves the paper backtest, but still does not pass promotion.
Use the filtered universe for paper/live dry-run observation, not automatic MT5 live trading.
```

## Multi-Horizon Momentum Finding

`multi_horizon_momentum` was added as a volatility-managed two-horizon trend
sleeve. It requires fast and slow momentum to agree, then filters entries by
trend efficiency, spread/costs, session, and a recent-versus-baseline volatility
ratio.

Seven-symbol broad evidence:

```text
symbols:
  XAGUSD, XAUUSD, USDCHF, AUDUSD, GBPUSD, EURUSD, EURGBP

full sample:
  return: 0.122%
  max drawdown: 0.184%
  Sharpe 15m: 0.010
  fills: 266

fixed-warmup validation:
  active positive fold fraction: 40.0%
  non-negative fold fraction: 47.1%
  median active return: -0.010%
  verdict: REJECT
```

Symbol eligibility found a better top-3 basket:

```text
symbols: AUDUSD, USDCHF, XAUUSD
full-sample return: 0.205%
max drawdown: 0.061%
Sharpe 15m: 0.034
fills: 80

fixed-warmup validation:
  active positive fold fraction: 60.0%
  non-negative fold fraction: 82.4%
  median active return: 0.009%
  worst drawdown: 0.039%
  promotion: PAPER_ONLY
```

Research verdict:

```text
Keep the top-3 multi-horizon basket as a paper backup and router experiment.
Do not use broad multi_horizon_momentum in the main candidate because it lowers
fold quality.
```

## Adaptive Training Gate Finding

`adaptive-strategy-select` now supports optional training gates:

```text
--min-train-fills
--min-train-adjusted-return-pct
--train-fill-penalty-pct
```

The purpose is to avoid selecting candidates that look good in the training
window only because they barely traded or had weak drawdown-adjusted evidence.

Latest checks:

```text
main plus clean recipes with min_train_fills=12:
  active positive folds: 55.6%
  non-negative folds: 76.5%
  median active return: 0.003%
  stitched OOS final equity: $1,002,768.91

main adaptive candidate with min_train_adjusted_return=0.0:
  active positive folds: 63.6%
  non-negative folds: 76.5%
  median active return: 0.033%
  stitched OOS final equity: $1,003,914.26

main adaptive candidate with train_fill_penalty_pct=0.000001 to 0.00002:
  selection counts unchanged
  active positive folds: 72.7%
  non-negative folds: 82.4%
  median active return: 0.033%
  stitched OOS final equity: $1,004,225.57
```

Research verdict:

```text
Keep training gates and fill penalties as diagnostics and optional safeguards.
Do not enable them by default because the ungated one-fold-cooldown adaptive
candidate remains stronger or unchanged on the current sample.
```

## Per-Symbol Adaptive Selection Finding

`adaptive-strategy-select` now has two dynamic per-symbol modes:

```text
--per-symbol-selection
--per-symbol-only
```

The selector can build a fold-specific map by choosing the best recent strategy
for each symbol separately, then scoring that map out of sample.

Seven-symbol result when the dynamic map is available but not forced:

```text
selection counts unchanged:
  kalman_trend=2
  champion_ensemble=5
  macd_momentum=10
  per_symbol_adaptive=0

metrics unchanged versus the main adaptive candidate:
  active positive folds: 72.7%
  non-negative folds: 82.4%
  median active return: 0.033%
  stitched OOS final equity: $1,004,225.57
```

Forced dynamic-map result:

```text
per_symbol_adaptive=17 folds
active positive folds: 55.6%
non-negative folds: 76.5%
median active return: 0.003%
worst drawdown: 0.106%
stitched OOS final equity: $1,003,938.32
```

Research verdict:

```text
Keep per-symbol adaptive selection as a diagnostic and demo feature. Do not
promote it over the existing global adaptive selector because it lowers
active-positive fold quality and increases drawdown on the current data.
```

## ML And Autocorrelation Rejection

The ML alpha report now supports research overrides such as `--ml-epochs`,
`--ml-lookback`, and `--symbols`, which makes candidate-basket probes faster.

Seven-symbol ML candidate-basket check:

```text
symbols: XAGUSD, XAUUSD, USDCHF, AUDUSD, GBPUSD, EURUSD, EURGBP
raw average signed return: +0.01 bps
best in-sample filter: +1.62 bps, mostly XAGUSD
walk-forward positive fold rate: 41.2%
average test signed return: +0.24 bps
promotion: REJECT
```

`autocorrelation_regime` was added as a serial-correlation diagnostic:

```text
seven-symbol full sample:
  return: -0.476%
  max drawdown: 0.574%
  trades: 508

stricter probes:
  strict_r30_m8_z12_edge5: -0.057%
  reversion_only_r35: -0.003%

asset split:
  metals only: -0.018%
  FX subset: -0.188%
```

Research verdict:

```text
Keep both as diagnostics. Do not promote ML alpha or autocorrelation_regime into
the champion ensemble, adaptive selector, or live MT5 path.
```

Follow-up optimizer check:

```text
command: quanthack autocorrelation-regime-optimize --include-walk-forward

best strict profile:
  lookback/signal: 48 / 8
  min abs autocorrelation: 0.28
  min momentum: 8.0 bps
  min trend efficiency: 0.45
  min expected edge: 6.0 bps
  UTC hours: 10-14

full sample return: -0.022%
walk-forward active folds: 17.6%
walk-forward active positive folds: 66.7%
walk-forward non-negative folds: 94.1%
median active return: 0.002%
```

Updated verdict:

```text
Optimization reduced churn and drawdown but did not create enough return or
coverage. Keep the optimizer for research discipline; do not promote the sleeve.
```

## Adaptive Stability And MACD Refinement

`adaptive-strategy-select` now records optional training-window stability
diagnostics by splitting each training window into chronological subwindows.

Latest stability-preferred run:

```text
train stability splits: 4
positive folds: 35.3%
active positive folds: 60.0%
non-negative folds: 76.5%
median active return: 0.003%
stitched OOS final equity: $1,001,788.05
verdict: diagnostic only
```

MACD refinement scan:

```text
best defensive adaptive variant: hist25
  min histogram: 2.5 bps
  active positive folds: 77.8%
  non-negative folds: 88.2%
  median active return: 0.033%
  stitched OOS final equity: $1,003,519.23

return leader remains:
  current h2.0 adaptive stack
  stitched OOS final equity: $1,004,225.57
```

Research verdict:

```text
Keep the current adaptive stack as the paper return leader. Keep hist25/slope020
as defensive backups if risk stability matters more than stitched return.
```

## Current Ranking

Practical ranking based on current evidence:

1. Adaptive `kalman_trend / champion_ensemble / macd_momentum` on seven symbols: strongest paper candidate.
2. Conservative MACD basket: clean active folds and strong non-negative coverage, but sparse.
3. Static top-5 strategy map: simpler backup, weaker than adaptive selection.
4. Multi-horizon top-3 basket: useful paper backup, not broad enough for main use.
5. `champion_ensemble`: useful component, weaker as a standalone walk-forward candidate.
6. `asset_adaptive_dual_squeeze`: clean low-drawdown confirmation sleeve, too sparse alone.
7. Defensive adaptive MACD variants: cleaner folds, lower stitched return.
8. Rejected research sleeves: ML alpha, autocorrelation regime, broad multi-horizon momentum, session momentum, intraday seasonality, broad dual squeeze, fixing/exhaustion reversal.

## Next Research Steps

1. Build basket-specific promotion checks so a candidate is judged on the exact
   basket it would trade live.
2. Build a low-churn ensemble that can switch between squeeze and pullback only
   when each sleeve has recent symbol-level evidence.
3. Keep MT5 execution read-only/manual until a strategy passes meaningful
   walk-forward return, fill count, drawdown, and risk-discipline gates.
