# Deployment Profile Robustness

This stress test reruns an exact deployment profile while preserving its strategy map,
symbol multipliers, asset-class gates, and symbol-specific session gates.

It checks two practical live risks:

1. Higher transaction costs.
2. Dependence on one symbol.

Run:

```bash
PYTHONPATH=src .venv/bin/python -m quanthack.cli.deployment_profile_robustness \
  --config configs/competition.toml \
  --profile-pack-json outputs/research/deployment_profile_symbol_gated_pack.json \
  --slot symbol_refined \
  --price-csv data/mixed_official_crypto_proxy_overlap_prices.csv \
  --quote-csv data/mixed_official_crypto_proxy_overlap_quotes.csv \
  --slippage-multiplier 1.5 \
  --slippage-multiplier 2 \
  --slippage-multiplier 3 \
  --output outputs/research/deployment_profile_symbol_refined_robustness.csv
```

Current `symbol_refined` baseline:

- Return: +0.619%
- Max drawdown: 0.389%
- Sharpe15: 0.062
- Risk discipline: 100/100
- Fills: 21
- P&L: $6,194

## Cost Stress

| Scenario | Return | Delta | Max DD | Risk | P&L | Decision |
|---|---:|---:|---:|---:|---:|---|
| `slippage_1.5x` | +0.591% | -0.028% | 0.401% | 100/100 | $5,912 | `PASS_WEAKER` |
| `slippage_2x` | +0.563% | -0.056% | 0.412% | 100/100 | $5,630 | `PASS_WEAKER` |
| `slippage_3x` | +0.507% | -0.113% | 0.434% | 100/100 | $5,066 | `PASS_WEAKER` |

Interpretation: the profile survives substantial cost stress. This is a useful
sign for live execution because spreads and fills will not match the research CSV
perfectly.

## Symbol Dependence

Most dependent exclusions:

| Excluded | Return | Delta | Risk | Decision |
|---|---:|---:|---:|---|
| `USDCAD` | +0.173% | -0.446% | 100/100 | `FRAGILE` |
| `AUDUSD` | +0.399% | -0.221% | 100/100 | `PASS_WEAKER` |
| `SOLUSD` | +0.433% | -0.186% | 100/100 | `PASS_WEAKER` |
| `XAUUSD` | +0.469% | -0.150% | 100/100 | `PASS_WEAKER` |
| `XRPUSD` | +0.475% | -0.144% | 100/100 | `PASS_WEAKER` |

Interpretation: `symbol_refined` is not cost-fragile, but it does have symbol
dependency. Removing `USDCAD` cuts more than half of the baseline return. That does
not mean remove `USDCAD`; it means the current challenger should be considered
dependent on one helpful FX leg and must be revalidated on fresh official MT5 data.

Next research direction:

- add a dependency-aware profile refiner that tests smaller `USDCAD` multipliers
  against walk-forward gates;
- compare `symbol_refined` against a lower-dependency variant rather than only
  same-window return;
- keep the current profile paper-only until official-data validation confirms the
  `USDCAD` contribution.
