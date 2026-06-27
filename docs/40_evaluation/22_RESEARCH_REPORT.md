# Research Report

Step 22 adds a demo-ready research report.

The goal is to stop jumping between terminal output and CSV files when we want to
answer:

```text
What strategy looks best?
Why?
What did risk do?
Where did P&L come from?
Is the system still dry-run safe?
```

## New Command

```bash
python scripts/reporting/build_research_report.py
```

This writes:

```text
outputs/reports/research_report.html
```

It also refreshes supporting CSVs:

```text
outputs/backtests/equity_curve.csv
outputs/backtests/pnl_ledger.csv
outputs/backtests/strategy_comparison.csv
outputs/backtests/parameter_sweep.csv
outputs/backtests/data_health.csv
outputs/backtests/walk_forward_summary.csv
outputs/backtests/walk_forward_folds.csv
```

## What The Report Includes

- Preflight status.
- Market data health.
- Selected strategy.
- Strategy comparison.
- Walk-forward evaluation.
- Momentum parameter sweep.
- Backtest metrics.
- Equity curve chart.
- Recent fills.
- Realized, open, and total attributed P&L.
- Risk settings.
- Dry-run-only safety reminder.

## Run In VS Code Terminal

Make sure your prompt shows `(.venv)`.

```bash
python scripts/reporting/build_research_report.py
```

To choose a specific strategy for the main backtest section:

```bash
python scripts/reporting/build_research_report.py --strategy mean_reversion
```

Then open:

```text
outputs/reports/research_report.html
```

## How To Read It

Use the report from top to bottom:

1. Check `Preflight`.
2. Check `Market Data Health`.
3. Check the best strategy comparison row.
4. Check the walk-forward row. This is more important than a single backtest.
5. Check the best momentum sweep candidate.
6. Check backtest return, Sharpe, drawdown, and P&L attribution.
7. Confirm risk settings are still conservative.

## Important Limitation

The report is only as good as the input data.

Right now the included historical CSV is tiny and synthetic. Treat the report as
proof that the research workflow works, not proof that the strategy is ready for
live or paper execution.
