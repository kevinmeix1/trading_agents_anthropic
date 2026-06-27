# Deployment Profile Signal Snapshot

This command loads one slot from `deployment_profile_pack.json` and converts the
latest aligned price/quote row into a current target sheet.

It is read-only:

- no MT5 order is sent
- no dry-run journal record is written
- strategy, session gate, portfolio allocation, and risk preview still run

Run the conservative profile snapshot:

```bash
PYTHONPATH=src .venv/bin/python \
  -c 'from quanthack.cli.deployment_profile_snapshot import main; main()' \
  --config configs/competition.toml \
  --profile-pack-json outputs/research/deployment_profile_pack.json \
  --slot conservative \
  --price-csv data/mixed_official_crypto_proxy_overlap_prices.csv \
  --quote-csv data/mixed_official_crypto_proxy_overlap_quotes.csv \
  --output outputs/research/deployment_profile_conservative_signal_snapshot.csv
```

Optional current-position context:

```bash
PYTHONPATH=src .venv/bin/python \
  -c 'from quanthack.cli.deployment_profile_snapshot import main; main()' \
  --slot conservative \
  --journal outputs/dry_run_journal.jsonl
```

The journal option reconstructs current dry-run positions before calculating
target changes. Without it, the snapshot assumes a flat book.

Optional historical rehearsal:

```bash
PYTHONPATH=src .venv/bin/python \
  -c 'from quanthack.cli.deployment_profile_snapshot import main; main()' \
  --slot conservative \
  --as-of 2026-06-10T10:00:00+00:00
```

`--as-of` chooses the latest common symbol row at or before that ISO timestamp.
This is useful when the newest row is outside the strategy's trading session.

## Output

The CSV has one row per profile symbol with:

- current bid, ask, and mid
- current notional
- raw strategy target
- session-gated target
- allocator-adjusted target
- BUY/SELL/HOLD change
- risk approval and risk reason
- strategy signal labels and allocation trim reasons

## MT5 Relevance

This is the safe bridge between research and execution. On Windows/MT5, the same
shape can be fed by live quotes instead of CSV rows, then converted into manual
tickets or gated API orders.

For now, treat mixed official/proxy profile snapshots as paper-only. Before live
trading, regenerate the profile pack and snapshots from official MT5 data.
