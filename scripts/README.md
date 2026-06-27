# Script Launchers

These files are thin VS Code terminal launchers. The real command implementations
live in `src/quanthack/cli/`.

Prefer the installed commands after running `python -m pip install -e .`:

```bash
quanthack --help
quanthack show-instruments
quanthack backtest
quanthack backtest --strategy ma_crossover
quanthack compare --strategy ma_crossover --strategy simple_momentum
quanthack walk-forward --strategy ma_crossover
quanthack portfolio-backtest
quanthack generate-sample-data
quanthack import-backtest-data --symbol EURUSD --max-files-per-symbol 2
quanthack portfolio-compare
quanthack strategy-attribution
quanthack symbol-eligibility-optimize
quanthack portfolio-universe-scan
quanthack portfolio-walk-forward
quanthack portfolio-router-walk-forward
quanthack ml-alpha-report
quanthack time-series-report
quanthack router-report
quanthack router-optimize
quanthack dual-squeeze-optimize
quanthack trend-pullback-optimize
quanthack manual-ticket --symbol EURUSD --side BUY --price 1.1000
quanthack mt5-probe --symbol EURUSD
quanthack mt5-capture --confirm-read-only-mt5 --symbol EURUSD
quanthack live-dry-run --adapter csv --symbol EURUSD
quanthack preflight
quanthack research-report
```

Folder map:

- `setup/`: environment, project status, and preflight readiness checks.
- `inspect/`: read config, market data, journals, and reconstructed positions.
- `dry_run/`: simulated strategy/risk/execution journal workflows, including the read-only CSV/MT5 live dry-run loop.
- `evaluation/`: single-symbol/portfolio backtests, strategy comparison, synthetic sample data, portfolio comparison, router optimization, sweeps, walk-forward testing, ML/router reports, and data validation.
- `reporting/`: journal and research report builders.
