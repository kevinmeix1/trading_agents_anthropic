# Local Results Dashboard

The project now has a local browser dashboard for backtest results and live
dry-run monitoring.

## Run It

From the QuanHack folder in VS Code:

```bash
python scripts/reporting/run_dashboard.py
```

Then open:

```text
http://127.0.0.1:8765
```

If that port is already busy, the server automatically tries the next few ports
and prints the actual URL.

## What It Reads

Backtest panels read:

```text
outputs/backtests/*comparison.csv
outputs/backtests/*walk_forward*summary.csv
outputs/backtests/*equity*.csv
```

Live panels read:

```text
outputs/live_competition_monitor.csv
outputs/live_dry_run_journal.jsonl
```

The dry-run audit tile reads:

```text
outputs/dry_run_journal.jsonl
```

## How To Use It During Development

Run a backtest or comparison first:

```bash
python scripts/evaluation/portfolio_compare.py \
  --price-csv data/downloaded_scan_prices.csv \
  --quote-csv data/downloaded_scan_quotes.csv
```

Refresh the dashboard. It polls every five seconds, so new CSVs should appear
without restarting the server.

For live dry-run, run:

```bash
python scripts/dry_run/live_dry_run.py
```

When the live monitor or live journal files exist, the Live tab moves from
waiting status to ready status.
