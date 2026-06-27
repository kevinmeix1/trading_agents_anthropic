# Deployment Profile Challenger Scorecard

The challenger scorecard compares exact deployment profiles against a baseline. It is
the promotion sanity check after attribution refinements: a candidate should improve
return without weakening drawdown, fold stability, or risk discipline.

Run:

```bash
PYTHONPATH=src .venv/bin/python -m quanthack.cli.deployment_profile_challenger \
  --config configs/competition.toml \
  --price-csv data/mixed_official_crypto_proxy_overlap_prices.csv \
  --quote-csv data/mixed_official_crypto_proxy_overlap_quotes.csv \
  --output outputs/research/deployment_profile_challenger_scorecard.csv
```

Default candidates:

- `survival`: `outputs/research/deployment_profile_pack.json`, slot `survival`
- `refined`: `outputs/research/deployment_profile_refined_pack.json`, slot `refined`
- `session_refined`: `outputs/research/deployment_profile_session_gated_pack.json`, slot `session_refined`
- `symbol_refined`: `outputs/research/deployment_profile_symbol_gated_pack.json`, slot `symbol_refined`

The first candidate is the baseline for deltas. You can override the set:

```bash
PYTHONPATH=src .venv/bin/python -m quanthack.cli.deployment_profile_challenger \
  --candidate survival,outputs/research/deployment_profile_pack.json,survival \
  --candidate symbol_refined,outputs/research/deployment_profile_symbol_gated_pack.json,symbol_refined
```

Current result:

| Rank | Candidate | Decision | Return | Delta | Max DD | Sharpe15 | Risk | P&L | Gate complexity |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `symbol_refined` | `PROMOTE_CHALLENGER` | +0.619% | +0.178% | 0.389% | 0.062 | 100/100 | $6,194 | 77 |
| 2 | `session_refined` | `PROMOTE_CHALLENGER` | +0.619% | +0.178% | 0.389% | 0.062 | 100/100 | $6,194 | 83 |
| 3 | `refined` | `PROMOTE_CHALLENGER` | +0.487% | +0.046% | 0.389% | 0.048 | 100/100 | $4,872 | 75 |
| 4 | `survival` | `BASELINE` | +0.441% | 0.000% | 0.466% | 0.040 | 100/100 | $4,409 | 75 |

Gate complexity counts restricted symbol-hours. This makes the comparison prefer
`symbol_refined` over `session_refined`: they have identical P&L and risk in the
current sample, but `symbol_refined` blocks only `GBPUSD:13` and `XAGUSD:13`
instead of a broader asset-class hour.

Interpretation: `symbol_refined` is currently the best research challenger. It is
still proxy/in-sample evidence, so the live decision remains: validate on fresh
official MT5 data before promotion.
