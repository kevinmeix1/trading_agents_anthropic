# Deployment Profile Action Scan

The action scan answers a practical question after a profile has been recommended:
when does this profile actually produce risk-approved actions?

Run:

```bash
PYTHONPATH=src .venv/bin/python -m quanthack.cli.deployment_profile_action_scan \
  --config configs/competition.toml \
  --profile-pack-json outputs/research/deployment_profile_pack.json \
  --recommendation-json outputs/research/deployment_profile_recommendation.json \
  --price-csv data/mixed_official_crypto_proxy_overlap_prices.csv \
  --quote-csv data/mixed_official_crypto_proxy_overlap_quotes.csv \
  --summary-output outputs/research/deployment_profile_action_scan_summary.csv \
  --events-output outputs/research/deployment_profile_action_scan_events.csv \
  --hours-output outputs/research/deployment_profile_action_scan_hours.csv
```

Outputs:

1. Summary CSV: one-row profile activity summary.
2. Events CSV: one row per non-HOLD action.
3. Hours CSV: activity grouped by UTC hour.

By default the scan is stateful: it carries a local portfolio forward so the same
target is not counted as a fresh action every bar. Use `--stateless` for pure signal
frequency checks.

The command defaults to the slot in `deployment_profile_recommendation.json` when
`--slot` is omitted. This makes it a natural follow-up after the recommendation step.

Use this to diagnose whether a profile is:

- Too quiet to compete on the return-heavy score.
- Active only in certain UTC hours.
- Concentrated in one symbol.
- Producing blocked actions because of risk or allocation controls.

The scan is read-only and does not journal or trade.
