# Crypto Allocation Comparison

This pass tests portfolio-level crypto allocation by assigning each crypto symbol
to one sleeve:

- `macd_momentum`
- `crypto_mean_reversion`

It does not blend signals inside a symbol. Instead it asks a cleaner portfolio
question: which crypto symbols should run trend and which should run reversion?

Run:

```bash
PYTHONPATH=src .venv/bin/python -c 'from quanthack.cli.crypto_allocation_compare import main; main()' \
  --config configs/competition.toml \
  --price-csv data/research_crypto_proxy_14d_prices.csv \
  --quote-csv data/research_crypto_proxy_14d_quotes.csv \
  --output outputs/research/crypto_allocation_comparison.csv
```

The command enumerates all symbol-level maps. With 5 crypto symbols and 2
strategies, that is 32 maps.

## 14-Day Crypto Proxy Result

Output: `outputs/research/crypto_allocation_comparison.csv`

Best robust candidate:

```text
BARUSD=crypto_mean_reversion
BTCUSD=crypto_mean_reversion
ETHUSD=crypto_mean_reversion
SOLUSD=macd_momentum
XRPUSD=crypto_mean_reversion
```

Metrics:

- return: +5.600%
- max drawdown: 0.783%
- Sharpe15: 0.057
- trades: 520
- risk discipline: 100/100
- fixed-warmup: PROMOTE
- fold coverage: 100% non-negative, 100% active-positive

The highest-return proxy map added `ETHUSD=macd_momentum` too, reaching +6.750%,
but with 1.553% drawdown. The SOL-only MACD map is the cleaner robustness pick.

## Mixed Official/Proxy Overlap Result

Output: `outputs/research/crypto_allocation_mixed_overlap_comparison.csv`

Top overlap map:

```text
BARUSD=crypto_mean_reversion
BTCUSD=macd_momentum
ETHUSD=crypto_mean_reversion
SOLUSD=macd_momentum
XRPUSD=crypto_mean_reversion
```

Metrics:

- return: +1.161%
- max drawdown: 0.331%
- Sharpe15: 0.070
- fixed-warmup: PROMOTE

The robust SOL-only MACD map ranked second on the overlap:

```text
BARUSD=crypto_mean_reversion
BTCUSD=crypto_mean_reversion
ETHUSD=crypto_mean_reversion
SOLUSD=macd_momentum
XRPUSD=crypto_mean_reversion
```

Metrics:

- return: +0.308%
- max drawdown: 0.257%
- Sharpe15: 0.033
- fixed-warmup: PROMOTE
- fold coverage: 100% non-negative, 100% active-positive

## Decision

Use the SOL-only MACD map as the robust paper candidate:

```text
BARUSD=crypto_mean_reversion
BTCUSD=crypto_mean_reversion
ETHUSD=crypto_mean_reversion
SOLUSD=macd_momentum
XRPUSD=crypto_mean_reversion
```

Why:

- top selection score on the longer 14-day proxy
- promoted on the mixed overlap too
- much better return than all-reversion
- less concentrated than all-MACD
- preserves the strongest crypto trend signal, SOL, without forcing MACD onto every coin

The BTC+SOL MACD map is an aggressive alternate, but should wait for official
MT5 crypto data confirmation.

Next:

1. Re-run this command on official MT5 crypto captures.
2. If the SOL-only map remains promoted, test it inside the full FX/metals/crypto portfolio.
3. Only consider the BTC+SOL aggressive map if official data confirms it across multiple windows.

Update: the full mixed-portfolio overlay test is now documented in
`96_CRYPTO_OVERLAY_COMPARISON.md`. In that stricter test, the SOL-only map was
rejected and the BTC+SOL map became the stronger paper-only research candidate.
