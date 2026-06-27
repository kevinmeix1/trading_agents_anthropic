# Project Structure

Step 3 turns the folder into a small Python project.

## Folders

```text
quanthack/
  configs/    TOML settings for risk, strategy, data, and outputs.
  data/       Small local CSV datasets for practice and tests.
  docs/       Notes and step-by-step guides, grouped by topic.
  outputs/    Generated journals, backtests, and reports.
  scripts/    Thin command launchers grouped by workflow.
  src/        Reusable Python package code.
  tests/      Checks that protect the project from mistakes.
```

The docs are organized as:

```text
docs/
  00_onboarding/
  10_foundations/
  20_data_execution/
  30_strategies/
  40_evaluation/
```

The reusable package code is organized as:

```text
src/quanthack/
  core/         clock, config, instruments, and project status.
  market/       price data, quote quality, validation, and sample data.
  trading/      risk engine, dry-run execution, and preflight checks.
  strategies/   strategy logic and ML alpha helpers.
  backtesting/  backtests, metrics, scoring, allocation, and optimization.
  reporting/    journal, HTML, research, and router reports.
  cli/          command-line entry points.
```

## Why We Use `src/`

Putting code inside `src/quanthack/` makes imports cleaner and prevents accidental
imports from random files in the project root.

## What We Built In This Step

- `pyproject.toml`: tells Python this folder is a package.
- `src/quanthack/core/status.py`: a tiny reusable module.
- `scripts/setup/project_status.py`: a command you can run.
- `tests/test_status.py`: a simple test.

## Commands For VS Code Terminal

Make sure your terminal prompt shows `(.venv)`.

Install this project into your virtual environment:

```bash
python -m pip install -e .
```

Run the status script:

```bash
python scripts/setup/project_status.py
```

Run the tests:

```bash
python -m unittest discover -s tests
```

Expected result:

```text
Ran 1 test
OK
```

Use the explicit `discover -s tests` form so Python always looks inside the
`tests/` folder.
