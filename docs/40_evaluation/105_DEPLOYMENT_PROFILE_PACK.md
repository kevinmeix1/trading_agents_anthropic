# Deployment Profile Pack

This command turns the latest promotion and asset-class stability evidence into
three named deployment profiles:

- `aggressive`: highest-return profile
- `conservative`: highest-return stable profile
- `survival`: lowest fold-contribution stable profile

Run:

```bash
PYTHONPATH=src .venv/bin/python \
  -c 'from quanthack.cli.deployment_profile_pack import main; main()'
```

Artifacts:

- `outputs/research/deployment_profile_pack.csv`
- `outputs/research/deployment_profile_pack.json`

## Current Result

The current profile pack is still `paper_only` because the evidence source is
`mixed_proxy`.

| Slot | Profile | Return | Max DD | Sharpe15 | Fold Contribution | Status |
|---|---|---:|---:|---:|---:|---|
| aggressive | `current_london_full` | +1.852% | 1.446% | 0.044 | 93.9% | PAPER_ONLY |
| conservative | `current_asia_metal75` | +0.761% | 0.744% | 0.040 | 79.1% | PAPER_ONLY |
| survival | `current_asia_metal25` | +0.441% | 0.466% | 0.040 | 65.8% | PAPER_ONLY |

The pack recommends:

```text
paper_only
```

Reason:

```text
mixed_proxy data cannot justify live deployment
```

## How To Use It

The CSV is readable in Excel. The JSON is useful for automation or MT5 wiring
because each profile contains:

- strategy map by symbol
- target-notional multiplier map
- FX and metal multipliers
- crypto session hours
- return, drawdown, Sharpe, and fold-contribution evidence
- recommended slot and reason

When official MT5 crypto data is available, rerun:

1. `crypto-promotion-pipeline --data-source official`
2. `deployment-profile-pack`

Then:

- If `recommended_slot=aggressive`, the primary profile passed official gates.
- If `recommended_slot=conservative`, the primary profile failed but the stable
  backup passed official gates.
- If `recommended_slot=paper_only`, do not run live.

This gives us a clean bridge from research evidence to live-deployment choices
without manually interpreting multiple CSVs during the competition.
