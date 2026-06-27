# Kalman Trend

`kalman_trend` turns the existing Kalman-style regime reader into a tradable
strategy.

The idea:

```text
Smooth recent log-prices with a lightweight Kalman filter.
If slope and path efficiency confirm TREND_UP, go long.
If slope and path efficiency confirm TREND_DOWN, go short.
If regime becomes CHOP or HIGH_VOLATILITY, exit.
```

The strategy still passes through:

- UTC session filters
- spread/cost filters
- volatility sizing
- max holding-period exits
- portfolio allocation
- risk engine checks

Current tuned research profile:

```text
lookback: 80
min_abs_slope_bps: 0.25
min_trend_efficiency: 0.20
expected_holding_bars: 6
min_expected_edge_bps: 5.0
max_holding_period: 32
```

## Evidence

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

Fixed warmup walk-forward on the five-symbol basket:

```text
folds: 6
positive fold fraction: 50.0%
median test return: 0.001%
median test Sharpe 15m: 0.002
worst test drawdown: 0.078%
evaluation fills: 108
risk discipline: 100/100
```

Verdict:

```text
This is the strongest standalone sleeve and remains the fallback baseline behind
champion_ensemble. It still needs more validation before automatic MT5 execution
because one fold contributes a large share of the gain.
```

## Commands

Five-symbol candidate:

```bash
quanthack-portfolio-backtest \
  --strategy kalman_trend \
  --symbol XAGUSD --symbol XAUUSD --symbol USDCHF --symbol AUDUSD --symbol GBPUSD \
  --price-csv data/full_20gb_15m_prices.csv \
  --quote-csv data/full_20gb_15m_quotes.csv \
  --equity-output outputs/backtests/kalman_trend_positive_subset_equity.csv \
  --pnl-output outputs/backtests/kalman_trend_positive_subset_pnl.csv \
  --allocation-output outputs/backtests/kalman_trend_positive_subset_allocation.csv \
  --fills-output outputs/backtests/kalman_trend_positive_subset_fills.csv
```

Fixed warmup validation:

```bash
quanthack-portfolio-fixed-warmup-walk-forward \
  --strategy kalman_trend \
  --symbol XAGUSD --symbol XAUUSD --symbol USDCHF --symbol AUDUSD --symbol GBPUSD \
  --price-csv data/full_20gb_15m_prices.csv \
  --quote-csv data/full_20gb_15m_quotes.csv
```
