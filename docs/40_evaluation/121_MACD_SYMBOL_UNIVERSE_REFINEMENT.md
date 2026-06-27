# MACD Symbol Universe Refinement

This refinement tests whether the strongest current standalone sleeve,
`macd_momentum`, should trade every available symbol or a narrower defensive
universe.

The new bridge command converts a symbol-eligibility optimizer row into an
executable deployment-profile pack:

```bash
PYTHONPATH=src .venv/bin/python -c 'from quanthack.cli.deployment_profile_symbol_universe_refine import main; main()' \
  --symbol-eligibility-csv outputs/research/macd_symbol_eligibility_full20gb.csv \
  --candidate top_5_pnl \
  --selected-slot macd_defensive_top5 \
  --selected-label macd_top5_defensive \
  --data-source official_research \
  --output-json outputs/research/deployment_profile_macd_symbol_universe_pack.json \
  --output-csv outputs/research/deployment_profile_macd_symbol_universe_pack.csv
```

Why this matters:

```text
The strategy remains generic.
The profile pack decides which symbols are eligible.
That keeps overfit symbol decisions outside the reusable strategy code.
```

Current evidence on `data/full_20gb_15m_prices.csv`:

```text
all_symbols:
  return: 6.001%
  max drawdown: 0.855%
  Sharpe 15m: 0.038
  walk-forward active positive folds: 100.0%

drop_worst_1 / no EURCHF:
  return: 5.922%
  max drawdown: 0.834%
  Sharpe 15m: 0.038
  walk-forward active positive folds: 100.0%

top_5_pnl / XAUUSD XAGUSD AUDUSD EURUSD USDCHF:
  return: 4.349%
  max drawdown: 0.824%
  Sharpe 15m: 0.034
  walk-forward active positive folds: 100.0%
  non-negative folds: 100.0%
```

Current evidence on the mixed official/crypto proxy overlap:

```text
all_symbols:
  return: 0.970%
  max drawdown: 2.439%
  Sharpe 15m: 0.018

drop_worst_3 / excludes ETHUSD GBPUSD XRPUSD:
  return: 1.711%
  max drawdown: 1.727%
  Sharpe 15m: 0.044
```

Research verdict:

```text
Do not replace the full MACD universe yet.

Use the generated top-5 universe as a defensive challenger profile. It gives up
return but lowers drawdown slightly and keeps perfect active-fold hit rate in the
available official window.

The crypto/short-overlap filters improve mixed results but are too concentrated
to promote without real competition crypto data.
```
