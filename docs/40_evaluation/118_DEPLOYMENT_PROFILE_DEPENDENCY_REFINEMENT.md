# Deployment Profile Dependency Refinement

Purpose: reduce hidden overfitting risk when one symbol explains too much of a
deployment profile's backtest return.

The robustness report already stress-tests a profile by removing one symbol at a
time. If removing a symbol removes most of the return, that symbol is a fragile
dependency. This is dangerous in the hackathon because our research data is a
single short window; a symbol that looked perfect in that window may not repeat
in the real competition window.

The dependency refiner reads a robustness CSV, finds fragile dependency symbols,
then tests smaller multipliers for only those symbols. It does not change the
strategy map, session gates, or non-dependent symbols. It ranks candidates by:

- risk discipline first
- walk-forward promotion status
- return retained versus the base profile
- reduction in leave-one-symbol dependency loss
- drawdown and Sharpe as tie-breakers

Default command:

```bash
PYTHONPATH=src .venv/bin/python -c 'from quanthack.cli.deployment_profile_dependency_refine import main; main()' \
  --config configs/competition.toml \
  --profile-pack-json outputs/research/deployment_profile_symbol_gated_pack.json \
  --slot symbol_refined \
  --robustness-csv outputs/research/deployment_profile_symbol_refined_robustness.csv \
  --price-csv data/mixed_official_crypto_proxy_overlap_prices.csv \
  --quote-csv data/mixed_official_crypto_proxy_overlap_quotes.csv \
  --output outputs/research/deployment_profile_dependency_refinement.csv \
  --refined-pack-json outputs/research/deployment_profile_dependency_refined_pack.json \
  --refined-slot dependency_refined
```

Key report fields:

- `dependent_symbols`: symbols selected from the robustness report.
- `dependency_scale`: multiplier applied to those symbols.
- `return_retention_vs_base`: how much of base return survived the scale change.
- `dependency_loss_reduction`: how much the leave-one-symbol dependency loss fell.
- `candidate_decision`: whether the return/dependency tradeoff is acceptable.
- `multiplier_map`: exact profile multipliers to deploy in paper/live dry run.

Interpretation:

- `BALANCED_REDUCE_DEPENDENCY` means the candidate retained enough return while
  reducing dependency enough to be worth considering.
- `KEEP_BASELINE_DEPENDENCY` means the original profile still ranks best under
  the configured thresholds.
- `WATCHLIST_TRADEOFF` means the dependency improved, but the return cost is
  probably too high for the default deployment threshold.
- `FAIL_*` or `REJECT_*` means do not promote without fresh evidence.

This is still paper-only evidence. Before live MT5 deployment, rerun this on the
latest official data window and compare the resulting `dependency_refined` slot
against `symbol_refined` with the challenger scorecard.

## Current Real-Data Result

Input:

- Profile: `outputs/research/deployment_profile_symbol_gated_pack.json`
- Slot: `symbol_refined`
- Robustness: `outputs/research/deployment_profile_symbol_refined_robustness.csv`
- Output: `outputs/research/deployment_profile_dependency_refinement.csv`
- Pack: `outputs/research/deployment_profile_dependency_refined_pack.json`

The robustness report selected `USDCAD` as the only fragile dependency.

Results:

- Base `USDCAD=1.00`: return `0.619%`, max drawdown `0.389%`, Sharpe15
  `0.062`, risk `100/100`, dependency loss `0.446%`.
- `USDCAD=0.85`: return retention `99.8%`, dependency loss reduction `0.3%`.
- `USDCAD=0.75`: return retention `99.6%`, dependency loss reduction `0.5%`.
- `USDCAD=0.50`: return retention `99.2%`, dependency loss reduction `1.1%`.
- `USDCAD=0.25`: return retention `98.8%`, dependency loss reduction `1.6%`.
- `USDCAD=0.00`: dependency loss reduction `100%`, but return retention only
  `28.0%`, so this is too expensive.

Conclusion: partial scaling does not materially fix the `USDCAD` dependency,
probably because the portfolio/risk allocation still leaves the same trade path
as the dominant contributor. Full removal fixes dependency but destroys too much
return. The generated `dependency_refined` slot therefore keeps the baseline
`symbol_refined` multipliers instead of pretending a weaker profile is an
improvement.
