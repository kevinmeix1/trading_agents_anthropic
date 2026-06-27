# Deployment Profile Dependency Replacement

Purpose: test whether fragile single-symbol return can be replaced by capped,
diversified refill symbols instead of merely trimming the fragile symbol.

The prior dependency scaler showed that trimming `USDCAD` from the
`symbol_refined` deployment profile barely reduced dependency. This refiner tries
a harder question:

1. Identify fragile dependency symbols from leave-one-symbol robustness.
2. Select replacement symbols that contributed positively and still have spare
   multiplier capacity below `1.0`.
3. Reduce or remove the dependency symbol.
4. Refill some or all of the freed multiplier budget into the replacement basket.
5. Re-run exact portfolio backtest, fixed-warmup walk-forward, and dependency
   stress.

Default command:

```bash
PYTHONPATH=src .venv/bin/python -c 'from quanthack.cli.deployment_profile_dependency_replace import main; main()' \
  --config configs/competition.toml \
  --profile-pack-json outputs/research/deployment_profile_symbol_gated_pack.json \
  --slot symbol_refined \
  --robustness-csv outputs/research/deployment_profile_symbol_refined_robustness.csv \
  --price-csv data/mixed_official_crypto_proxy_overlap_prices.csv \
  --quote-csv data/mixed_official_crypto_proxy_overlap_quotes.csv \
  --output outputs/research/deployment_profile_dependency_replacement.csv \
  --refined-pack-json outputs/research/deployment_profile_dependency_replacement_pack.json \
  --refined-slot dependency_replacement
```

## Current Real-Data Result

Input profile: `symbol_refined`

Fragile dependency:

- `USDCAD`

Replacement pool selected from leave-one-symbol contribution and multiplier
capacity:

- `XAUUSD`
- `XRPUSD`
- `ETHUSD`
- `XAGUSD`

Results:

- Baseline: return `0.619%`, max drawdown `0.389%`, Sharpe15 `0.062`,
  risk `100/100`, dependency loss `0.446%`.
- Partial `USDCAD` reduction with refill often improves full-sample return, for
  example `USDCAD=0.50` with `XAUUSD/XRPUSD` refill reached `0.746%`.
- But those partial replacements make dependency worse: the same example had
  dependency loss `0.543%`, worse than the baseline `0.446%`.
- Full `USDCAD=0.00` replacement removes dependency, but only preserves about
  `31-33%` of baseline return and falls to `PAPER_ONLY` walk-forward status.

Conclusion: existing underweighted sleeves cannot replace `USDCAD` strongly
enough. Reallocation is not the fix. The next research step should be new alpha
generation or a truly adaptive gate that turns `USDCAD` on only when current
evidence is fresh, rather than trying to statically replace it with current
profile components.

The generated `dependency_replacement` profile pack intentionally keeps the
baseline multiplier map when no replacement candidate clears the return and
dependency gates.
