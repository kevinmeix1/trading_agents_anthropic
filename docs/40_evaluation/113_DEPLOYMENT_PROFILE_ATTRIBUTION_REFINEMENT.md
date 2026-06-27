# Deployment Profile Attribution Refinement

The attribution refiner turns session-attribution weaknesses into research-only profile
variants.

Run:

```bash
PYTHONPATH=src .venv/bin/python -m quanthack.cli.deployment_profile_attribution_refine \
  --config configs/competition.toml \
  --profile-pack-json outputs/research/deployment_profile_pack.json \
  --recommendation-json outputs/research/deployment_profile_recommendation.json \
  --attribution-csv outputs/research/deployment_profile_session_attribution.csv \
  --price-csv data/mixed_official_crypto_proxy_overlap_prices.csv \
  --quote-csv data/mixed_official_crypto_proxy_overlap_quotes.csv \
  --output outputs/research/deployment_profile_attribution_refinement.csv \
  --refined-pack-json outputs/research/deployment_profile_refined_pack.json
```

What it does:

1. Reads the recommended profile slot.
2. Reads session attribution rows.
3. Aggregates total P&L by symbol.
4. Finds symbols with negative attribution P&L.
5. Tests profile variants where those weak symbols are scaled to `75%`, `50%`,
   `25%`, or `0%` of their current multiplier.
6. Backtests each candidate and, by default, runs fixed-warmup walk-forward gates.
7. Writes a ranked CSV plus a research-only profile pack containing the best candidate.

This is intentionally simple. It does not try to curve-fit individual hour/signal
rules into the live system. Instead, it asks a safer question: if a symbol is hurting
the current profile, does reducing its capital improve the full profile without
breaking risk/fold stability?

Important limitation: this is attribution-driven and therefore in-sample. Treat the
best refined profile as a hypothesis. It must be re-run on fresh official MT5 data
before live use.

Typical next step:

```bash
PYTHONPATH=src .venv/bin/python -m quanthack.cli.deployment_profile_backtest \
  --config configs/competition.toml \
  --profile-pack-json outputs/research/deployment_profile_refined_pack.json \
  --slot refined
```

The refined pack can also be passed directly to the selector/recommendation workflow.
The commands read profile slots from the JSON pack, so `refined` participates like any
other executable deployment profile:

```bash
PYTHONPATH=src .venv/bin/python -m quanthack.cli.deployment_profile_recommendation \
  --config configs/competition.toml \
  --profile-pack-json outputs/research/deployment_profile_refined_pack.json
```
