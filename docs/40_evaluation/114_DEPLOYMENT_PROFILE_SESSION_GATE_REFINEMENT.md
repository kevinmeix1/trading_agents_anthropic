# Deployment Profile Session-Gate Refinement

The session-gate refiner turns deployment-profile session attribution into
research-only UTC-hour gates by asset class. It is meant to answer a narrow
question: if an asset class loses money in a specific UTC hour, does blocking new
or larger entries in that hour improve the full profile without harming risk
discipline?

Run:

```bash
PYTHONPATH=src .venv/bin/python -m quanthack.cli.deployment_profile_session_gate_refine \
  --config configs/competition.toml \
  --profile-pack-json outputs/research/deployment_profile_refined_pack.json \
  --slot refined \
  --attribution-csv outputs/research/deployment_profile_refined_session_attribution.csv \
  --price-csv data/mixed_official_crypto_proxy_overlap_prices.csv \
  --quote-csv data/mixed_official_crypto_proxy_overlap_quotes.csv \
  --max-dropped-hours 3 \
  --output outputs/research/deployment_profile_session_gate_refinement.csv \
  --refined-pack-json outputs/research/deployment_profile_session_gated_pack.json \
  --refined-slot session_refined
```

What it does:

1. Aggregates session attribution by asset class and UTC hour.
2. Finds negative-P&L asset-hour buckets with enough fills.
3. Tests candidates that drop the worst `N` asset-hours from profile session gates.
4. Backtests each candidate and, by default, runs fixed-warmup walk-forward gates.
5. Writes a ranking CSV and a research profile pack containing the best candidate.

Current result on the refined profile:

| Candidate | Dropped hours | Return | Max DD | Sharpe15 | Risk | Fills | P&L |
|---|---|---:|---:|---:|---:|---:|---:|
| `refined_session_drop_1h` | `FOREX:13` | +0.619% | 0.389% | 0.062 | 100/100 | 21 | $6,194 |
| `refined_session_base` | none | +0.487% | 0.389% | 0.048 | 100/100 | 27 | $4,872 |

Interpretation: the weak FX 13:00 UTC bucket looks worth avoiding in the current
research sample. This is still in-sample attribution, so treat `session_refined`
as a candidate to validate on fresh official MT5 data, not as an automatic live
deployment.

Backtest the generated profile:

```bash
PYTHONPATH=src .venv/bin/python -m quanthack.cli.deployment_profile_backtest \
  --config configs/competition.toml \
  --profile-pack-json outputs/research/deployment_profile_session_gated_pack.json \
  --slot session_refined \
  --price-csv data/mixed_official_crypto_proxy_overlap_prices.csv \
  --quote-csv data/mixed_official_crypto_proxy_overlap_quotes.csv \
  --output-prefix outputs/research/deployment_profile_session_refined_backtest
```
