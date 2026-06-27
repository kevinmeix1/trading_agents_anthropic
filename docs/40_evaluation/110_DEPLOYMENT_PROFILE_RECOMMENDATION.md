# Deployment Profile Recommendation

The recommendation command turns selector-sweep evidence into one current operating
choice: the next deployment profile slot to run in paper/live dry-run.

By default the command reads executable profile slots from the supplied profile pack.
That means research packs such as `deployment_profile_refined_pack.json` can introduce
new slots like `refined` without changing the command code. If the pack recommends a
non-executable state such as `paper_only`, choose an executable slot explicitly.

Run:

```bash
PYTHONPATH=src .venv/bin/python -m quanthack.cli.deployment_profile_recommendation \
  --config configs/competition.toml \
  --profile-pack-json outputs/research/deployment_profile_pack.json \
  --price-csv data/mixed_official_crypto_proxy_overlap_prices.csv \
  --quote-csv data/mixed_official_crypto_proxy_overlap_quotes.csv \
  --recommendation-csv outputs/research/deployment_profile_recommendation.csv \
  --recommendation-json outputs/research/deployment_profile_recommendation.json \
  --sweep-output outputs/research/deployment_profile_recommendation_sweep.csv \
  --snapshot-output outputs/research/deployment_profile_recommended_snapshot.csv
```

What it does:

1. Runs the deployment-profile selector sweep.
2. Picks the highest-ranked selector policy.
3. Uses all completed folds to recommend the next profile slot.
4. Writes a CSV and JSON recommendation artifact.
5. Optionally writes a signal snapshot for the recommended slot.

The recommendation is not an order. It is a controlled handoff between research and
operation. The snapshot still goes through allocation and risk checks, and the MT5
ticket-sheet step still requires symbol contract specs before any manual order sizing.

Typical interpretation:

- `recommended_slot`: profile to use next.
- `recommendation_reason`: why that profile won from completed fold evidence.
- `promotion_status`: whether the historical adaptive sequence passed promotion gates.
- `historical_selected_sequence`: how the policy would have selected past folds.
- `past_scores`: per-profile score using all completed folds.

Use this file as the source of truth for the current paper/live dry-run profile. If the
recommended slot changes after new data arrives, treat that as a deliberate profile
rotation and regenerate the snapshot/ticket sheet.

Run a refined research pack:

```bash
PYTHONPATH=src .venv/bin/python -m quanthack.cli.deployment_profile_recommendation \
  --config configs/competition.toml \
  --profile-pack-json outputs/research/deployment_profile_refined_pack.json \
  --price-csv data/mixed_official_crypto_proxy_overlap_prices.csv \
  --quote-csv data/mixed_official_crypto_proxy_overlap_quotes.csv \
  --recommendation-csv outputs/research/deployment_profile_refined_recommendation.csv \
  --recommendation-json outputs/research/deployment_profile_refined_recommendation.json \
  --sweep-output outputs/research/deployment_profile_refined_recommendation_sweep.csv \
  --snapshot-output outputs/research/deployment_profile_refined_recommendation_snapshot.csv
```
