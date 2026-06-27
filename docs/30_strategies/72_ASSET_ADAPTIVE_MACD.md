# Asset-Adaptive MACD

Date: 2026-06-22

## Purpose

`asset_adaptive_macd` is a lightweight registry strategy that reuses the
existing MACD momentum implementation but tightens crypto entries.

Why: crypto MACD helped the mixed FX/metals/crypto research portfolio, but it
also increased drawdown and turnover. A stricter crypto filter reduced drawdown
on the 14-day research proxy while keeping positive P&L.

## Behavior

For FX and metals, `asset_adaptive_macd` behaves like the configured
`macd_momentum`.

For crypto only, it applies minimum guardrails:

- `min_histogram_bps >= 5.0`
- `min_macd_bps >= 2.0`
- `min_trend_efficiency >= 0.25`
- `max_holding_period <= 10`

It is intentionally not the default champion. It is a candidate sleeve for
crypto validation once MT5/official crypto quotes are available.

## Usage

```bash
PYTHONPATH=src python -c "from quanthack.cli.portfolio_backtest import main; main()" \
  --config configs/competition.toml \
  --price-csv data/research_crypto_proxy_14d_prices.csv \
  --quote-csv data/research_crypto_proxy_14d_quotes.csv \
  --strategy asset_adaptive_macd \
  --symbol BARUSD --symbol BTCUSD --symbol ETHUSD --symbol SOLUSD --symbol XRPUSD \
  --clock-open-at 2026-06-07T23:45:00+00:00
```

Aliases:

- `adaptive_macd`
- `crypto_macd`
- `crypto_strict_macd`
