# Liquidity Sweep Reversal

`liquidity_sweep_reversal` is a range-regime alpha sleeve for FX, metals, and
crypto. It looks for a close outside a recent range followed by an immediate
close back inside that range, then fades the failed breakout toward the prior
range midpoint.

This is meant to diversify the trend-heavy book. Momentum and MACD sleeves want
breakouts to continue; this sleeve wants weak breakouts to fail. In hackathon
terms, it is an attempt to improve positive-round frequency and Sharpe without
raising gross risk.

## Signal Logic

For each symbol:

1. Use the last `lookback` closes.
2. Treat all but the final two closes as the prior range.
3. If the previous close swept above the prior high and the latest close is back
   inside the range, generate a short signal.
4. If the previous close swept below the prior low and the latest close is back
   inside the range, generate a long signal.
5. Estimate edge as distance from latest close back to the geometric midpoint of
   the prior range.

The strategy refuses entries when:

- the prior range is too narrow,
- the prior move was too trend-efficient,
- the sweep is too large and likely represents real news/momentum,
- expected edge is too small to cover estimated costs,
- the current UTC hour is outside the configured asset-class session.

## Exits

Open positions exit when:

- price reaches the prior range midpoint,
- the range break is renewed and invalidates the fade,
- max holding period is reached,
- the allowed trading session ends.

## Optimizer

Run a fast portfolio parameter comparison:

```bash
quanthack liquidity-sweep-reversal-optimize \
  --config configs/competition.toml \
  --price-csv data/mixed_official_crypto_proxy_overlap_prices.csv \
  --quote-csv data/mixed_official_crypto_proxy_overlap_quotes.csv \
  --output outputs/research/liquidity_sweep_reversal_optimization.csv
```

Run with fixed-warmup walk-forward:

```bash
quanthack liquidity-sweep-reversal-optimize \
  --config configs/competition.toml \
  --price-csv data/mixed_official_crypto_proxy_overlap_prices.csv \
  --quote-csv data/mixed_official_crypto_proxy_overlap_quotes.csv \
  --include-walk-forward \
  --train-size 960 \
  --test-size 192 \
  --step-size 192 \
  --output outputs/research/liquidity_sweep_reversal_optimization.csv
```

Interpretation rule: promote this sleeve only if it adds active positive folds
or improves complementarity. A strong full-sample result alone is not enough
because the current research data window is short and overfitting risk is high.

## Initial Evidence

Artifacts generated:

- `outputs/research/liquidity_sweep_reversal_optimization.csv`
- `outputs/research/liquidity_sweep_reversal_optimization_fullsample.csv`
- `outputs/research/liquidity_sweep_reversal_official10_overlap_optimization.csv`
- `outputs/research/liquidity_sweep_reversal_full20gb_optimization.csv`
- `outputs/research/liquidity_sweep_strategy_comparison.csv`
- `outputs/research/liquidity_sweep_strategy_comparison_full20gb.csv`
- `outputs/research/liquidity_sweep_reversal_attribution.csv`

Findings:

- Mixed official plus crypto-proxy overlap: all default candidates lost money.
  Crypto losses, especially `SOLUSD` and `ETHUSD`, dominated the full result.
- Official 10-symbol overlap only: the widest/slowest candidate made a tiny
  positive return, but with only four trades.
- Longer official 10-symbol window: the sleeve was negative and materially
  underperformed `macd_momentum`, `champion_ensemble`, and `volatility_squeeze`.

Current decision: keep this sleeve as `PAPER_ONLY`. It is useful research
coverage for false-breakout regimes, but it should not enter the deployment
profile or alpha router until it shows stronger walk-forward complementarity.
