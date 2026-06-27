# Deployment Profile Backtest

This command loads `deployment_profile_pack.json`, selects one profile slot, and
reruns the exact strategy map, symbol multipliers, and session gates. Deployment
profiles can specify `allowed_utc_hours`, `forex_allowed_utc_hours`,
`metal_allowed_utc_hours`, `crypto_allowed_utc_hours`, and
`symbol_allowed_utc_hours`; omitted fields mean all hours are allowed for that
scope. Symbol-specific hours override asset-class hours.

It is the bridge from research evidence to executable strategy configuration.

Run the conservative profile:

```bash
PYTHONPATH=src .venv/bin/python \
  -c 'from quanthack.cli.deployment_profile_backtest import main; main()' \
  --config configs/competition.toml \
  --profile-pack-json outputs/research/deployment_profile_pack.json \
  --slot conservative \
  --price-csv data/mixed_official_crypto_proxy_overlap_prices.csv \
  --quote-csv data/mixed_official_crypto_proxy_overlap_quotes.csv \
  --output-prefix outputs/research/deployment_profile_conservative_backtest
```

Run the aggressive profile:

```bash
PYTHONPATH=src .venv/bin/python \
  -c 'from quanthack.cli.deployment_profile_backtest import main; main()' \
  --config configs/competition.toml \
  --profile-pack-json outputs/research/deployment_profile_pack.json \
  --slot aggressive \
  --price-csv data/mixed_official_crypto_proxy_overlap_prices.csv \
  --quote-csv data/mixed_official_crypto_proxy_overlap_quotes.csv \
  --output-prefix outputs/research/deployment_profile_aggressive_backtest
```

Artifacts per run:

- `{prefix}_summary.csv`
- `{prefix}_equity.csv`
- `{prefix}_pnl.csv`
- `{prefix}_fills.csv`
- `{prefix}_allocation.csv`

## Current Replays

| Slot | Profile | Return | Max DD | Sharpe15 | Fills | Risk |
|---|---|---:|---:|---:|---:|---:|
| aggressive | `current_london_full` | +1.852% | 1.446% | 0.044 | 53 | 100/100 |
| conservative | `current_asia_metal75` | +0.761% | 0.744% | 0.040 | 26 | 100/100 |

The replay matches the profile-pack evidence, which means the JSON is not only a
reporting artifact. It is an executable strategy configuration.

## MT5 Relevance

When the project moves to Windows/MT5:

1. regenerate the profile pack on official MT5 data
2. run `deployment-profile-backtest` for the chosen slot
3. use the selected profile's strategy map, multipliers, and session hours in
   the live dry-run or manual-ticket workflow

`--slot recommended` resolves to the pack's `recommended_slot` only when that slot is
one of the executable profiles in the JSON. If the pack says `paper_only`, choose a
profile explicitly for research replays. Live use still requires official-data gates.
