# Live Operator Dashboard

The operator dashboard is a static HTML page for competition-day review. It
combines the current deployment profile, latest profile snapshot, live
allocation report, live monitor row, and MT5 ticket sheet.

Build it from the current research artifacts:

```bash
PYTHONPATH=src .venv/bin/python \
  -c 'from quanthack.cli.operator_dashboard import main; main()' \
  --profile-pack-json outputs/research/deployment_profile_pack.json \
  --snapshot-csv outputs/research/deployment_profile_conservative_signal_snapshot_asof_0815.csv \
  --allocation-csv outputs/research/profile_live_allocation.csv \
  --monitor-csv outputs/research/profile_live_monitor.csv \
  --ticket-csv outputs/research/mt5_ticket_sheet_asof_0815.csv \
  --output outputs/reports/operator_dashboard.html
```

Open:

```text
outputs/reports/operator_dashboard.html
```

## What It Shows

- operating status and next action
- active profile and timestamp
- actionable signal count
- ticket status counts
- latest equity, leverage, and accepted dry-run trades
- MT5 ticket rows and blockers
- source files, row counts, and freshness

## Current Use

The report is read-only. It does not connect to MT5 or place orders.

If ticket rows say `NEEDS_CONTRACT_SPEC`, inspect MT5 Symbol Specification on
Windows and fill in the contract-spec CSV before trusting lot sizes.

