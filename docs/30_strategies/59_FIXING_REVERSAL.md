# Fixing Reversal

`fixing_reversal` is an intraday FX/metals research sleeve.

The idea:

```text
If price moves strongly into a fixing/session window,
and the next bar confirms reversal,
fade the pre-window move for a short holding period.
```

The implementation is deliberately conservative:

- Uses only recent prices and the current quote timestamp.
- Trades only configured UTC hours.
- Disables crypto by default.
- Requires a clean pre-window move, an opposite confirmation bar, minimum expected edge, and cost clearance.
- Exits after the configured session window or max holding period.
- Still passes through portfolio allocation, market quality, slippage, and risk checks.

Current default research profile:

```text
pre_fix_lookback: 4 prices
allowed UTC hours: 14
minimum pre-fix move: 8.0 bps
minimum reversal confirmation: 1.5 bps
minimum pre-fix efficiency: 0.35
max holding period: 4 bars
```

## Evidence

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
folds: 6
positive fold fraction: 33.3%
median test return: 0.000%
median test Sharpe 15m: 0.000
worst test drawdown: 0.026%
evaluation fills: 60
risk discipline: 100/100
```

Verdict:

```text
Useful tested research sleeve, not a live MT5 strategy.
The edge is too small and the forward folds are mixed.
```

## Commands

Run the optimizer:

```bash
quanthack-fixing-reversal-optimize \
  --price-csv data/full_20gb_15m_prices.csv \
  --quote-csv data/full_20gb_15m_quotes.csv \
  --symbol XAGUSD --symbol XAUUSD --symbol USDCHF --symbol GBPUSD \
  --symbol AUDUSD --symbol USDJPY --symbol USDCAD \
  --output outputs/backtests/fixing_reversal_optimization.csv
```

Run the current default profile:

```bash
quanthack-portfolio-backtest \
  --strategy fixing_reversal \
  --price-csv data/full_20gb_15m_prices.csv \
  --quote-csv data/full_20gb_15m_quotes.csv
```
