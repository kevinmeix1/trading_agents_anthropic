# QuantHack

Starter workspace for the Model to Market / quantitative trading hackathon.

This project starts slowly on purpose. Before choosing a strategy, we first want:

- A clean VS Code workspace.
- Python 3.11 running in a project-local virtual environment.
- Git tracking the work.
- A clear understanding of the rules.
- A safe rule: no real or paper orders until dry-run checks are built and tested.
- A read-only MT5 path that can observe ticks, bars, and account state without sending orders.

## First Goal

Open this folder in VS Code and run the environment check:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python scripts/setup/check_environment.py
```

Expected result:

```text
Python: 3.11.x
London time: ...
Status: environment looks ready for step 1
```

After Step 3, install the local project and run the first project check:

```bash
python -m pip install -e .
python scripts/setup/project_status.py
python -m unittest discover -s tests
```

## Documentation

The docs are grouped by purpose so the workspace is easier to scan:

- `docs/00_onboarding/`: first setup and project map.
- `docs/10_foundations/`: rules, clock, risk, config, and preflight.
- `docs/20_data_execution/`: data, journals, reports, P&L, and validation.
- `docs/30_strategies/`: strategies, routers, ML alpha, and allocation.
- `docs/40_evaluation/`: backtests, sweeps, comparisons, and research reports.

Start with `docs/README.md` for the full reading order.

For the live-style workflow, read
`docs/20_data_execution/37_MT5_READ_ONLY_LIVE_DRY_RUN.md` and start with:

```bash
quanthack live-dry-run --adapter csv --symbol EURUSD
quanthack mt5-probe --symbol EURUSD
```

## Important

This is for a paper-trading hackathon. It is not financial, investment, legal, tax,
or real-money trading advice. The official participant portal and organizer rules
override these notes whenever they differ.
