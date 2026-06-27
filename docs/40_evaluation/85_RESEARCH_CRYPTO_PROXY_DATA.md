# Research Crypto Proxy Data

Date: 2026-06-22

## Purpose

The official downloaded pricer archive does not contain the five crypto
competition symbols. To avoid leaving the crypto sleeve completely untested, I
added a research-only proxy data path using Binance spot USDT klines. Binance's
public Spot API documents the kline endpoint as `GET /api/v3/klines` with a
maximum request limit of 1000 rows:

- https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints

This is not official competition data and should not be treated as a final
validation source. It is useful for strategy development only.

## New Fetch Command

```bash
PYTHONPATH=src python -c "from quanthack.cli.fetch_crypto_proxy_data import main; main()" \
  --all-crypto \
  --days 14 \
  --interval 15m \
  --price-output data/research_crypto_proxy_14d_prices.csv \
  --quote-output data/research_crypto_proxy_14d_quotes.csv \
  --confirm-research-proxy
```

The command maps:

- `BARUSD` -> `BARUSDT`
- `BTCUSD` -> `BTCUSDT`
- `ETHUSD` -> `ETHUSDT`
- `SOLUSD` -> `SOLUSDT`
- `XRPUSD` -> `XRPUSDT`

It writes QuanHack-compatible prices and synthetic quotes. The synthetic spread
uses the instrument metadata slippage/spread assumptions, so the backtest still
pays an estimated trading cost.

## Data Health

Command:

```bash
PYTHONPATH=src python -c "from quanthack.cli.validate_market_data import main; main()" \
  --config configs/competition.toml \
  --price-csv data/research_crypto_proxy_14d_prices.csv \
  --quote-csv data/research_crypto_proxy_14d_quotes.csv \
  --symbol BARUSD --symbol BTCUSD --symbol ETHUSD --symbol SOLUSD --symbol XRPUSD \
  --max-gap-seconds 1200 \
  --output outputs/research/research_crypto_proxy_14d_data_health.csv
```

Result:

- overall status: `OK`
- rows per symbol: `1,344`
- window: `2026-06-07T23:45:00+00:00` to `2026-06-21T23:30:00+00:00`
- spread breaches: `0.0%` for all five symbols

## Backtest Notes

Because this historical proxy window mostly predates the configured competition
open, the portfolio backtest CLI now supports a research-only clock override:

```bash
--clock-open-at 2026-06-07T23:45:00+00:00
```

This does not change live trading behavior. It only lets historical replays use
the competition-mode machinery on older data.

## Crypto MACD Results

Five-symbol crypto MACD, conservative allocation:

- return: `+0.913%`
- max drawdown: `0.350%`
- official 15m Sharpe view: `0.029`
- fills: `24`
- risk discipline score: `100/100`
- worst leverage: `1.74x`
- worst net directional exposure: `80.0%`
- worst largest-symbol concentration: `65.5%`

Five-symbol crypto MACD with portfolio volatility targeting:

- return: `+0.223%`
- max drawdown: `0.060%`
- official 15m Sharpe view: `0.034`
- fills: `43`
- risk discipline score: `100/100`
- worst leverage: `0.56x`
- worst net directional exposure: `80.0%`
- worst largest-symbol concentration: `68.0%`

The vol-targeted version gives up return but improves smoothness, drawdown, and
trade-count eligibility. It is a potential Sharpe-prize sleeve, not the current
return-maximizing choice.

## Important Negative Test

I tested a less conservative net-exposure interpretation that allowed one-sided
crypto baskets. It produced many more trades but failed the competition-style
risk monitor because net directional concentration stayed above the penalty
threshold, and P&L became negative. That experiment supports keeping the
allocator conservative even when it leaves some crypto signals unused.

## Current Takeaway

Crypto is promising for reducing flat rounds, but only as a carefully gated,
offset-aware sleeve. The next robust step is to combine this research proxy
workflow with real MT5 read-only capture once the Windows/MT5 environment is
available, then re-run the same tests on actual competition quotes.
