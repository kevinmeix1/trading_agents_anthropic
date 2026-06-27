# Deployment Profile Selector Sweep

The selector sweep tests whether the adaptive deployment-profile selector is robust
across policy settings.

Run:

```bash
PYTHONPATH=src .venv/bin/python -m quanthack.cli.deployment_profile_selector_sweep \
  --config configs/competition.toml \
  --profile-pack-json outputs/research/deployment_profile_pack.json \
  --price-csv data/mixed_official_crypto_proxy_overlap_prices.csv \
  --quote-csv data/mixed_official_crypto_proxy_overlap_quotes.csv \
  --output outputs/research/deployment_profile_selector_sweep.csv
```

What it sweeps:

1. Fallback profile before enough evidence exists.
2. Number of completed folds required before adaptive selection.
3. Drawdown penalty in the past-evidence score.
4. Minimum risk-discipline floor.
5. Whether inactive profiles can be selected from absence of evidence.

The implementation runs each fixed profile walk-forward once, then evaluates many
selector policies on top of those fold results. This is much faster than rerunning the
portfolio backtest for every policy setting.

The sweep ranking is deliberately conservative but not return-blind. It prioritizes
promotion status, active positive folds, non-negative fold consistency, risk
discipline, cumulative test-fold return, lower fold concentration, and lower drawdown.
This mirrors the hackathon tension: return matters most, but a fragile high-return
setting should not beat a safer setting unless it still passes the risk and fold gates.

Use this after rebuilding `deployment_profile_pack.json`. If many nearby settings
produce similar promote-able results, the profile selector is more trustworthy. If one
fragile setting wins by a wide margin, treat it as paper-only until the live data window
confirms it.
