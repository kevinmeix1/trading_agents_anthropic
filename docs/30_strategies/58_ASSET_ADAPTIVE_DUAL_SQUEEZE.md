# Asset-Adaptive Dual Squeeze

`asset_adaptive_dual_squeeze` is a research variant of `dual_squeeze`.

It keeps the current conservative dual-squeeze profile for FX and crypto, but
uses a faster, looser profile for metals:

```text
FX/crypto profile:
  fast lookback: 14
  fast squeeze window: 4
  breakout buffer: 2.5 bps
  confirmation lookback: 24
  confirmation squeeze window: 8

Metal profile:
  fast lookback: 12
  fast squeeze window: 4
  breakout buffer: 2.0 bps
  confirmation lookback: 20
  confirmation squeeze window: 6
```

Why this exists:

```text
The fast-loose profile improved XAGUSD/XAUUSD activity and P&L, but hurt several
FX pairs. The wrapper applies that profile only to metals and leaves FX on the
default dual-squeeze profile.
```

Latest eligible-basket full-sample result on `data/full_20gb_15m_prices.csv`:

```text
return: 0.156%
max drawdown: 0.049%
Sharpe 15m: 0.026
fills: 44
risk discipline: 100/100
```

Warmup walk-forward check:

```text
folds: 6
positive fold fraction: 16.7%
median test return: 0.000%
worst test drawdown: 0.038%
evaluation fills: 12
risk discipline: 100/100 average
```

Verdict:

```text
Useful research sleeve. Do not promote to automatic live MT5 execution until it
shows better out-of-sample activity.
```

Run the full-sample portfolio check:

```bash
quanthack portfolio-backtest \
  --strategy asset_adaptive_dual_squeeze \
  --symbol XAGUSD --symbol XAUUSD --symbol USDCHF --symbol GBPUSD \
  --symbol AUDUSD --symbol EURCHF --symbol USDJPY --symbol USDCAD \
  --price-csv data/full_20gb_15m_prices.csv \
  --quote-csv data/full_20gb_15m_quotes.csv
```

Run the warmup walk-forward check:

```bash
quanthack portfolio-fixed-warmup-walk-forward \
  --strategy asset_adaptive_dual_squeeze \
  --symbol XAGUSD --symbol XAUUSD --symbol USDCHF --symbol GBPUSD \
  --symbol AUDUSD --symbol EURCHF --symbol USDJPY --symbol USDCAD \
  --price-csv data/full_20gb_15m_prices.csv \
  --quote-csv data/full_20gb_15m_quotes.csv \
  --train-size 960 \
  --test-size 192 \
  --step-size 192
```
