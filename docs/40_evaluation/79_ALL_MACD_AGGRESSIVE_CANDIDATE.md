# All-MACD Aggressive Candidate

The competition-profile strategy-map optimizer found a simpler and stronger
aggressive paper candidate than the mixed hand map:

```text
AUDUSD = macd_momentum
EURCHF = macd_momentum
EURGBP = macd_momentum
EURUSD = macd_momentum
GBPUSD = macd_momentum
USDCAD = macd_momentum
USDCHF = macd_momentum
USDJPY = macd_momentum
XAGUSD = macd_momentum
XAUUSD = macd_momentum
```

## Why It Matters

Compared with the mixed aggressive map, all-MACD is:

- simpler to explain and operate;
- lower turnover than the multi-horizon-heavy alternatives;
- stronger full-sample return;
- better walk-forward non-negative consistency.

## Current Risk Profile

The competition profile now uses a stricter MACD entry filter plus
asset-class-specific per-position stops:

```text
MACD min histogram: 2.5 bps
MACD session hours: 10-14 UTC for FX/metals
```

The position-stop policy is:

```text
FX:     1.0%
Metals: 2.0%
Crypto: 2.5% once crypto data is available
```

This keeps the stop-loss framework live, but avoids stopping silver/gold at the
same threshold as quieter FX pairs. The stricter histogram filter reduced noisy
entries, improved drawdown, and lifted active-positive fold quality.

## Sizing Frontier

`configs/competition.toml`, 10 FX/metals symbols, fixed-warmup walk-forward
enabled:

```text
25% cap:
  return: 2.489%
  max drawdown: 0.490%
  official 15m Sharpe: 0.036
  risk discipline: 100/100
  worst leverage: 2.00x
  WF non-negative folds: 82.4%
  WF active-positive folds: 66.7%
  WF median active return: 0.221%

40% cap:
  return: 3.963%
  max drawdown: 0.601%
  official 15m Sharpe: 0.039
  risk discipline: 100/100
  worst leverage: 3.17x
  WF non-negative folds: 82.4%
  WF active-positive folds: 66.7%
  WF median active return: 0.354%

60% cap:
  return: 5.179%
  max drawdown: 0.739%
  official 15m Sharpe: 0.039
  risk discipline: 100/100
  worst leverage: 4.58x
  WF non-negative folds: 82.4%
  WF active-positive folds: 66.7%
  WF median active return: 0.531%

80% cap:
  return: 6.001%
  max drawdown: 0.855%
  official 15m Sharpe: 0.038
  risk discipline: 100/100
  worst leverage: 5.81x
  WF non-negative folds: 82.4%
  WF active-positive folds: 66.7%
  WF median active return: 0.641%
```

## Decision

This is the current strongest aggressive paper profile, but not automatic-live
ready:

```text
Pros:
  stronger full-sample return after wider metal stops
  improved non-negative and active-positive fold fractions
  clean risk discipline
  simple operating story

Cons:
  total positive folds still below the 67% live gate because inactive folds remain flat
  crypto is still missing
  one 30-day FX/metals window remains the dominant overfit risk
```

For demo/paper ranking, compare this against the safer adaptive candidate. For
live/manual MT5, only use after a fresh round-specific data check.

## Command

```bash
quanthack sizing-frontier \
  --config configs/competition.toml \
  --price-csv data/full_20gb_15m_prices.csv \
  --quote-csv data/full_20gb_15m_quotes.csv \
  --strategy-map AUDUSD=macd_momentum \
  --strategy-map EURCHF=macd_momentum \
  --strategy-map EURGBP=macd_momentum \
  --strategy-map EURUSD=macd_momentum \
  --strategy-map GBPUSD=macd_momentum \
  --strategy-map USDCAD=macd_momentum \
  --strategy-map USDCHF=macd_momentum \
  --strategy-map USDJPY=macd_momentum \
  --strategy-map XAGUSD=macd_momentum \
  --strategy-map XAUUSD=macd_momentum \
  --symbol-notional-pct 0.25 \
  --symbol-notional-pct 0.40 \
  --symbol-notional-pct 0.60 \
  --symbol-notional-pct 0.80 \
  --include-walk-forward \
  --train-size 480 \
  --test-size 96 \
  --step-size 96 \
  --output outputs/research/all_macd_strict25_asset_stop_sizing_frontier.csv
```
