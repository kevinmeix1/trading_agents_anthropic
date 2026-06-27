# Deployment Profile Selector

This module compares the fixed deployment profiles in
`outputs/research/deployment_profile_pack.json` against a simple adaptive selector.
The purpose is to test whether switching between `aggressive`, `conservative`, and
`survival` profiles improves fold stability without weakening risk discipline.

Run:

```bash
PYTHONPATH=src .venv/bin/python -m quanthack.cli.deployment_profile_selector \
  --config configs/competition.toml \
  --profile-pack-json outputs/research/deployment_profile_pack.json \
  --price-csv data/mixed_official_crypto_proxy_overlap_prices.csv \
  --quote-csv data/mixed_official_crypto_proxy_overlap_quotes.csv \
  --summary-output outputs/research/deployment_profile_selector_summary.csv \
  --folds-output outputs/research/deployment_profile_selector_folds.csv
```

What it does:

1. Loads the deployment profile pack.
2. Runs each selected profile through the same fixed-warmup portfolio walk-forward.
3. For each test fold, chooses the next profile using only completed past folds.
4. Writes a summary CSV comparing fixed profiles versus the adaptive selector.
5. Writes a fold CSV showing which profile was selected and why.

The selector score is intentionally transparent:

```text
past score = cumulative past return - drawdown_penalty * worst past drawdown
```

Profiles must also meet a minimum average risk-discipline score over completed past
folds. By default, a profile must also have at least one active completed fold before
it can be selected; this prevents a no-trade profile from winning only because it has
zero drawdown. The first folds use the fallback profile because there is not enough
past evidence yet.

This is useful for hackathon preparation because the live competition may move through
different regimes. A fixed aggressive profile can win on trend days and struggle on
quiet or choppy days; a survival profile can preserve drawdown rank but miss return
opportunities. This selector tests a middle path while keeping the decision auditable.

Important limitation: this does not create new alpha by itself. It only chooses among
already-built profile recipes. If the profiles are too similar, or if the historical
window is not representative of the live round, adaptive selection may add noise.
