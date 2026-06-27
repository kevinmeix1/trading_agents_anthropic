# Crypto Sleeve Comparison

This pass adds a repeatable crypto-only comparison loop:

```bash
PYTHONPATH=src .venv/bin/python -c 'from quanthack.cli.crypto_sleeve_compare import main; main()' \
  --config configs/competition.toml \
  --price-csv data/research_crypto_proxy_14d_prices.csv \
  --quote-csv data/research_crypto_proxy_14d_quotes.csv \
  --output outputs/research/crypto_sleeve_comparison.csv
```

The command only evaluates crypto symbols and writes one CSV combining:

- full-sample competition proxy score
- official-style return, drawdown, Sharpe, risk score, and trade count
- fixed-warmup fold stability
- promotion decision

## Why this matters

Crypto is the missing 24/7 diversifier. FX and metals can be quiet or closed over weekends; the crypto symbols can still create positive rounds. The risk is overfitting because the current crypto data is research-only Binance proxy data, not official MT5 competition data.

## 14-day proxy result

Output: `outputs/research/crypto_sleeve_comparison.csv`

Best evidence:

- `crypto_mean_reversion`: +0.654%, 0.449% drawdown, Sharpe15 0.028, 240 trades, fixed-warmup promoted.
- `mean_reversion`: +0.102%, 0.145% drawdown, Sharpe15 0.018, 310 trades, fixed-warmup promoted.
- `macd_momentum`: +0.913%, 0.350% drawdown, Sharpe15 0.029, 24 trades, paper-only due fold concentration.
- `asset_adaptive_macd`: +0.543%, 0.051% drawdown, Sharpe15 0.031, 10 trades, paper-only due fold concentration.
- `crypto_trend_reversion`: +0.415%, 1.524% drawdown, Sharpe15 0.005, 180 trades, rejected by folds.

Interpretation:

`crypto_mean_reversion` improved the original reversion sleeve: fewer trades, higher return, and promoted fixed-warmup folds on the 14-day proxy. `macd_momentum` remains the strongest crypto trend sleeve. The `crypto_trend_reversion` router is useful as research evidence, but it should not be promoted because the longer proxy folds were unstable.

## Mixed official/proxy overlap result

Output: `outputs/research/crypto_sleeve_mixed_overlap_comparison.csv`

This uses the short overlap file with smaller folds:

```bash
PYTHONPATH=src .venv/bin/python -c 'from quanthack.cli.crypto_sleeve_compare import main; main()' \
  --config configs/competition.toml \
  --price-csv data/mixed_official_crypto_proxy_overlap_prices.csv \
  --quote-csv data/mixed_official_crypto_proxy_overlap_quotes.csv \
  --train-size 96 \
  --test-size 48 \
  --step-size 48 \
  --output outputs/research/crypto_sleeve_mixed_overlap_comparison.csv
```

Best evidence:

- `macd_momentum`: +0.461%, 0.306% drawdown, Sharpe15 0.051, paper-only.
- `crypto_trend_reversion`: +0.080%, 0.167% drawdown, Sharpe15 0.021, paper-only.
- `crypto_mean_reversion`: +0.006%, 0.081% drawdown, Sharpe15 0.003, paper-only.
- `mean_reversion`: approximately flat, rejected.

Interpretation:

The alternate short window supports `macd_momentum` most strongly. `crypto_mean_reversion` is not a big return source on the overlap, but it no longer fails the way plain mean reversion does. That makes it a paper candidate pending official MT5 crypto data, not a live-ready allocation.

## Decision

Do not promote a live crypto blend yet.

Keep:

- `macd_momentum` as the primary crypto trend candidate.
- `asset_adaptive_macd` as the safer low-drawdown crypto MACD variant.
- `crypto_mean_reversion` as the better crypto reversion candidate, but only at small sleeve weight unless official MT5 crypto data confirms it.
- `mean_reversion` as the conservative baseline/reference sleeve.
- `crypto_trend_reversion` as research-only infrastructure.

Next best work:

1. Capture official MT5 crypto quotes as soon as possible.
2. Re-run `crypto-sleeve-compare` on official crypto data.
3. Use `95_CRYPTO_ALLOCATION_COMPARISON.md` for the current portfolio-level allocation evidence between `macd_momentum` and `crypto_mean_reversion`.
4. Add portfolio-level crypto sleeve weighting only after official-data confirmation.
