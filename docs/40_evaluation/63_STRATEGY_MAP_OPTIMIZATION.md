# Strategy Map Optimization

`strategy-map-optimize` searches simple per-symbol strategy maps.

The purpose:

```text
Some strategies work on some symbols and fail on others.
Instead of one strategy for every symbol, test controlled symbol -> strategy maps.
```

The optimizer:

1. Runs each candidate strategy on the selected symbol set.
2. Extracts per-symbol P&L attribution from those full portfolio runs.
3. Builds conservative maps:
   - one all-symbol map per strategy
   - best strategy per symbol
   - positive-only best strategy map
   - top-N best symbol-strategy subsets
4. Replays each map through the shared-risk portfolio backtester.
5. Optionally runs fixed-warmup validation.

It deliberately avoids huge combinatorial searches. The goal is robust research,
not finding a fragile perfect-looking map.

## Current Finding

Seven-symbol scan:

```text
symbols: XAGUSD, XAUUSD, USDCHF, AUDUSD, GBPUSD, EURUSD, EURGBP
strategies: champion_ensemble, macd_momentum, kalman_trend
```

Best walk-forward-ranked map after the session-filtered MACD update:

```text
top_5_best_symbol_strategies:
  map:
    XAGUSD=champion_ensemble
    XAUUSD=macd_momentum
    AUDUSD=macd_momentum
    USDCHF=macd_momentum
    EURUSD=macd_momentum
  full-sample return: 0.308%
  max drawdown: 0.100%
  official 15m Sharpe: 0.029
  trades: 54
  active positive folds: 62.5%
  non-negative folds: 82.4%
  median active return: 0.019%
```

For comparison:

```text
all_macd_momentum:
  full-sample return: 0.271%
  active positive folds: 60.0%
  non-negative folds: 76.5%
  median active return: 0.017%

all_kalman_trend:
  full-sample return: 0.309%
  active positive folds: 55.6%
  non-negative folds: 76.5%
  median active return: 0.003%

all_champion_ensemble:
  full-sample return: 0.385%
  max drawdown: 0.216%
  active positive folds: 44.4%
  median active return: -0.000%
```

The naive best-per-symbol map still overfits:

```text
best_per_symbol_all:
  full-sample return: 0.264%
  active positive folds: 45.5%
  non-negative folds: 64.7%
  median active return: -0.011%
```

Verdict:

```text
The top-5 static map is now a credible paper candidate and easier to operate
than adaptive selection. Adaptive selection with one-fold cooldown still has
better fold quality, so keep the map as the simpler backup candidate.
```

## Command

```bash
quanthack strategy-map-optimize \
  --strategy champion_ensemble \
  --strategy macd_momentum \
  --strategy kalman_trend \
  --symbol XAGUSD --symbol XAUUSD --symbol USDCHF --symbol AUDUSD \
  --symbol GBPUSD --symbol EURUSD --symbol EURGBP \
  --price-csv data/full_20gb_15m_prices.csv \
  --quote-csv data/full_20gb_15m_quotes.csv \
  --include-walk-forward \
  --train-size 480 \
  --test-size 96 \
  --step-size 96 \
  --output outputs/backtests/strategy_map_session_macd_wf.csv \
  --score-output outputs/backtests/strategy_map_session_macd_scores.csv
```
