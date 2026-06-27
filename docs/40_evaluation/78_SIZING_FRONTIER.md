# Sizing Frontier

`quanthack sizing-frontier` sweeps per-symbol notional caps for a fixed strategy
map and reports full-sample competition metrics plus optional walk-forward
robustness.

The point is to separate two questions:

```text
1. Does bigger sizing improve return without breaking risk discipline?
2. Does bigger or smaller sizing improve fold robustness?
```

For most linear portfolios, sizing changes return and drawdown but does not fix
the sign of bad strategy decisions. This tool makes that visible.

## Latest Aggressive Map Frontier

Command output:

```text
cap=25%:
  return=1.197%
  drawdown=0.784%
  official 15m Sharpe=0.017
  risk=100/100
  worst leverage=1.75x
  WF non-negative folds=47.1%
  WF active-positive folds=35.7%
  WF median active return=-0.075%

cap=40%:
  return=1.975%
  drawdown=1.251%
  official 15m Sharpe=0.018
  risk=100/100
  worst leverage=2.79x
  WF non-negative folds=47.1%
  WF active-positive folds=35.7%
  WF median active return=-0.121%

cap=60%:
  return=2.682%
  drawdown=1.685%
  official 15m Sharpe=0.018
  risk=100/100
  worst leverage=4.10x
  WF non-negative folds=47.1%
  WF active-positive folds=35.7%
  WF median active return=-0.130%

cap=80%:
  return=3.318%
  drawdown=1.962%
  official 15m Sharpe=0.019
  risk=100/100
  worst leverage=5.28x
  WF non-negative folds=47.1%
  WF active-positive folds=35.7%
  WF median active return=-0.101%
```

Interpretation:

```text
Sizing up improves full-sample return while preserving risk discipline, but it
does not improve fold robustness. The current aggressive map has an alpha
selection problem, not mainly a sizing problem.
```

## Decision

Use the frontier for every candidate before considering live/manual deployment:

- If fold robustness is weak, do not solve it by leverage.
- If fold robustness is acceptable, use the frontier to choose a return/drawdown
  operating point.
- Keep `80%` cap as an aggressive research point, not a default live candidate.

## Command

```bash
quanthack sizing-frontier \
  --config configs/competition.toml \
  --price-csv data/full_20gb_15m_prices.csv \
  --quote-csv data/full_20gb_15m_quotes.csv \
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
  --symbol-notional-pct 0.25 \
  --symbol-notional-pct 0.40 \
  --symbol-notional-pct 0.60 \
  --symbol-notional-pct 0.80 \
  --include-walk-forward \
  --train-size 480 \
  --test-size 96 \
  --step-size 96 \
  --output outputs/research/competition_map_sizing_frontier.csv
```
