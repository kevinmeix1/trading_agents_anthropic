# Deployment Profile Symbol-Gate Refinement

The symbol-gate refiner is a more precise version of session-gate refinement. Instead
of dropping an entire asset class for a UTC hour, it drops only the weak
`symbol:hour` buckets found in session attribution.

Run:

```bash
PYTHONPATH=src .venv/bin/python -m quanthack.cli.deployment_profile_symbol_gate_refine \
  --config configs/competition.toml \
  --profile-pack-json outputs/research/deployment_profile_refined_pack.json \
  --slot refined \
  --attribution-csv outputs/research/deployment_profile_refined_session_attribution.csv \
  --price-csv data/mixed_official_crypto_proxy_overlap_prices.csv \
  --quote-csv data/mixed_official_crypto_proxy_overlap_quotes.csv \
  --max-dropped-symbol-hours 6 \
  --output outputs/research/deployment_profile_symbol_gate_refinement.csv \
  --refined-pack-json outputs/research/deployment_profile_symbol_gated_pack.json \
  --refined-slot symbol_refined
```

What it does:

1. Aggregates session attribution by symbol and UTC hour.
2. Finds negative-P&L symbol-hour buckets.
3. Tests candidates that remove the worst `N` symbol-hour buckets from the profile.
4. Backtests each candidate and, by default, runs fixed-warmup walk-forward gates.
5. Writes a ranking CSV and a research profile pack containing the best candidate.

Current result on the refined profile:

| Candidate | Dropped symbol-hours | Return | Max DD | Sharpe15 | Risk | Fills | P&L |
|---|---|---:|---:|---:|---:|---:|---:|
| `refined_symbol_session_drop_2h` | `XAGUSD:13 GBPUSD:13` | +0.619% | 0.389% | 0.062 | 100/100 | 21 | $6,194 |
| `refined_symbol_session_base` | none | +0.487% | 0.389% | 0.048 | 100/100 | 27 | $4,872 |

This matches the broader asset-class session-gate improvement, but is less blunt:
it keeps the rest of FX and metals open while blocking only `GBPUSD` and `XAGUSD`
new/increased exposure at UTC 13.

Backtest the generated profile:

```bash
PYTHONPATH=src .venv/bin/python -m quanthack.cli.deployment_profile_backtest \
  --config configs/competition.toml \
  --profile-pack-json outputs/research/deployment_profile_symbol_gated_pack.json \
  --slot symbol_refined \
  --price-csv data/mixed_official_crypto_proxy_overlap_prices.csv \
  --quote-csv data/mixed_official_crypto_proxy_overlap_quotes.csv \
  --output-prefix outputs/research/deployment_profile_symbol_refined_backtest
```

As with all attribution-derived refinements, this is in-sample research evidence.
Validate on fresh official MT5 data before live deployment.
