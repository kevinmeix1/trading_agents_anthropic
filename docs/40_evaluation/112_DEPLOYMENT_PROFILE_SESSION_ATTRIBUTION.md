# Deployment Profile Session Attribution

The action scan shows when a profile trades. Session attribution shows whether those
entry hours historically helped or hurt P&L.

Run:

```bash
PYTHONPATH=src .venv/bin/python -m quanthack.cli.deployment_profile_session_attribution \
  --config configs/competition.toml \
  --profile-pack-json outputs/research/deployment_profile_pack.json \
  --recommendation-json outputs/research/deployment_profile_recommendation.json \
  --price-csv data/mixed_official_crypto_proxy_overlap_prices.csv \
  --quote-csv data/mixed_official_crypto_proxy_overlap_quotes.csv \
  --output outputs/research/deployment_profile_session_attribution.csv
```

What it does:

1. Resolves the profile slot from `deployment_profile_recommendation.json` unless
   `--slot` is provided.
2. Runs the exact deployment profile backtest.
3. Matches fills into entry lots.
4. Credits realized P&L and remaining open P&L back to the entry hour, symbol, signal,
   and side.
5. Writes a CSV attribution table.

This is useful after the action scan. If many actions fire around UTC `00-03`, this
report tells whether that hour bucket actually made money or merely traded often.

Interpretation:

- `utc_hour`: hour when the entry lot was opened.
- `primary_signal`: signal that opened the lot.
- `realized_pnl_usd`: closed P&L credited to that entry bucket.
- `open_pnl_usd`: mark-to-market P&L on remaining open lots from that bucket.
- `total_pnl_usd`: realized plus open P&L.

Use weak rows to decide what to reduce or session-gate. Use strong rows to decide what
may deserve more capital, subject to risk discipline and fold stability.
