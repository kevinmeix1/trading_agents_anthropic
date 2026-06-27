# Competition Profile And Aggressive Map

This note validates the higher-leverage profile ported from the
`quanthackclaude` comparison pass into the current main codebase.

## What Changed

Added:

- `configs/competition.toml`
- competition-safe optional `RiskLimits` controls:
  - reduce-only margin tier;
  - drawdown derisk brake;
  - daily-loss freeze toggle;
  - `RiskLimits.competition_safe()`;
  - explicit `STOP_OUT_MARGIN_LEVEL_PCT = 30.0`.
- CI/dev-tooling config in `pyproject.toml` and `.github/workflows/ci.yml`.

The competition profile uses larger target notionals for the strongest trend
sleeves while keeping internal risk below official red-line/penalty zones:

```text
max gross leverage: 6.0x
max symbol notional: 80% of equity
margin hard floor: 300%
reduce-only margin tier: 500%
drawdown brake: scale new size from 4% -> 10% drawdown
position stop: 1% entry-notional loss
```

## Aggressive Full-Sample Map

Strategy map:

```text
AUDUSD = macd_momentum
EURCHF = volatility_squeeze
EURGBP = volatility_squeeze
EURUSD = macd_momentum
GBPUSD = multi_horizon_momentum
USDCAD = macd_momentum
USDCHF = multi_horizon_momentum
USDJPY = quality_trend
XAGUSD = macd_momentum
XAUUSD = macd_momentum
```

Full-sample result on `data/full_20gb_15m_*.csv`:

```text
final equity: $1,033,175.37
return: 3.318%
max drawdown: 1.962%
official 15m Sharpe: 0.019
trades: 178
risk discipline: 100/100
worst leverage: 5.28x
worst net directional exposure: 69.2%
worst largest-symbol concentration: 68.6%
```

Interpretation: sizing can materially improve return rank while staying
risk-clean. The portfolio allocator and risk engine trimmed 153 of 2128 periods,
so this is not unbounded leverage.

## Walk-Forward Reality Check

Fixed-warmup walk-forward with `train=480`, `test=96`, `step=96`:

```text
folds: 17
positive fold fraction: 29.4%
active fold fraction: 82.4%
active positive fold fraction: 35.7%
non-negative fold fraction: 47.1%
median active test return: -0.101%
worst test drawdown: 1.295%
risk discipline: 100/100
evaluation fills: 154
promotion: REJECT
```

Interpretation: the full-sample return is concentrated in a few strong folds.
This profile is useful research evidence for sizing and return potential, but it
is not robust enough to replace the current paper candidate.

## Current Decision

Keep `configs/competition.toml` as an aggressive research/deployment profile,
but do not make the aggressive map the main candidate yet.

Sizing frontier check:

```text
25% cap: return 1.197%, DD 0.784%, WF non-negative 47.1%
40% cap: return 1.975%, DD 1.251%, WF non-negative 47.1%
60% cap: return 2.682%, DD 1.685%, WF non-negative 47.1%
80% cap: return 3.318%, DD 1.962%, WF non-negative 47.1%
```

Sizing changes the return/drawdown tradeoff but does not fix fold sign
robustness. Full notes:
`docs/40_evaluation/78_SIZING_FRONTIER.md`.

The practical next step is to use the higher sizing only after selection gates
improve:

1. find strategies with stronger positive/non-negative fold distribution;
2. retest sizing after fold robustness improves;
3. add crypto coverage because 24/7 crypto is the most direct way to reduce flat
   weekend/idle folds.

## Commands

Full-sample check:

```bash
quanthack portfolio-backtest \
  --config configs/competition.toml \
  --strategy-map EURUSD=macd_momentum \
  --strategy-map GBPUSD=multi_horizon_momentum \
  --strategy-map AUDUSD=macd_momentum \
  --strategy-map USDCHF=multi_horizon_momentum \
  --strategy-map USDCAD=macd_momentum \
  --strategy-map EURGBP=volatility_squeeze \
  --strategy-map EURCHF=volatility_squeeze \
  --strategy-map XAUUSD=macd_momentum \
  --strategy-map XAGUSD=macd_momentum \
  --strategy-map USDJPY=quality_trend
```

Walk-forward check:

```bash
quanthack portfolio-fixed-warmup-walk-forward \
  --config configs/competition.toml \
  --strategy champion_ensemble \
  --strategy-map EURUSD=macd_momentum \
  --strategy-map GBPUSD=multi_horizon_momentum \
  --strategy-map AUDUSD=macd_momentum \
  --strategy-map USDCHF=multi_horizon_momentum \
  --strategy-map USDCAD=macd_momentum \
  --strategy-map EURGBP=volatility_squeeze \
  --strategy-map EURCHF=volatility_squeeze \
  --strategy-map XAUUSD=macd_momentum \
  --strategy-map XAGUSD=macd_momentum \
  --strategy-map USDJPY=quality_trend \
  --train-size 480 \
  --test-size 96 \
  --step-size 96
```
