# Crypto Mean Reversion

`crypto_mean_reversion` is a crypto-specific profile over the existing
`MeanReversionStrategy`.

It exists because the plain crypto `mean_reversion` sleeve was stable, but too
small and too busy:

- 14-day proxy result: +0.102% with 310 trades
- fixed-warmup folds: promoted
- problem: too much turnover for too little P&L

The crypto profile keeps the same signal logic but changes the operating point:

- `lookback = 16`
- `entry_zscore = 1.0`
- `max_trend_bps = 50`
- `position_sizing = volatility`
- `target_notional_usd = 500_000`
- `max_target_notional_usd = 150_000`
- `max_holding_period = 20`

## Why These Settings

The winning test was not a wider z-score. A wider z-score reduced activity but
performed badly. The better tradeoff was a longer 16-bar baseline with modest
z-score and capped volatility sizing.

On the 14-day crypto proxy data:

- plain `mean_reversion`: +0.102%, 0.145% drawdown, 310 trades
- `crypto_mean_reversion`: +0.654%, 0.449% drawdown, 240 trades

So the profile made the sleeve more meaningful while reducing full-sample trade
count.

## Current Status

This is a promising paper/research sleeve, not a blind live sleeve.

It promoted on the 14-day crypto proxy walk-forward, but only reached
`PAPER_ONLY` on the short mixed official/proxy overlap. Re-run it on official MT5
crypto data before using it in a live portfolio.

Run:

```bash
PYTHONPATH=src .venv/bin/python -c 'from quanthack.cli.crypto_sleeve_compare import main; main()' \
  --config configs/competition.toml \
  --strategy crypto_mean_reversion \
  --strategy mean_reversion \
  --strategy macd_momentum \
  --price-csv data/research_crypto_proxy_14d_prices.csv \
  --quote-csv data/research_crypto_proxy_14d_quotes.csv \
  --output outputs/research/crypto_reversion_check.csv
```
