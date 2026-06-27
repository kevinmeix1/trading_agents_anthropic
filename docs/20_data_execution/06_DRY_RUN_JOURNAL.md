# Dry-Run Journal

Step 6 adds a dry-run execution journal.

This is still not real trading. It does not connect to MT5, Syphonix, an API, or
any broker. It only writes a record to a local file.

## Why We Need A Journal

In a trading hackathon, terminal output is not enough.

We need to answer questions like:

- What did the strategy ask for?
- What account state did risk see?
- Did risk approve or block?
- If approved, did risk shrink the size?
- What mode were we in: qualify or checkpoint protection?
- When did the decision happen?

The journal is our audit trail.

## New Files

- `src/quanthack/execution.py`
- `scripts/dry_run/dry_run_trade.py`
- `scripts/inspect/show_journal.py`
- `tests/test_execution.py`

## What Happens In This Step

The flow is:

```text
TradeRequest
  -> RiskEngine
  -> RiskDecision
  -> DryRunExecutor
  -> outputs/dry_run_journal.jsonl
```

The dry-run executor writes both approved and blocked decisions. That is important:
blocked trades are evidence that the safety layer is working.

## Run In VS Code Terminal

Make sure your prompt shows `(.venv)`.

Run a normal fake trade:

```bash
python scripts/dry_run/dry_run_trade.py
```

Run a fake trade that risk shrinks:

```bash
python scripts/dry_run/dry_run_trade.py --target-notional 900000
```

Run a fake trade that risk blocks:

```bash
python scripts/dry_run/dry_run_trade.py --equity 974000 --day-start-equity 1000000
```

Show recent journal entries:

```bash
python scripts/inspect/show_journal.py
```

Run all tests:

```bash
python -m unittest discover -s tests
```

## Where The Journal Goes

By default:

```text
outputs/dry_run_journal.jsonl
```

The `outputs/` folder is ignored by Git. That is intentional because journal files
can become large and may later include sensitive account details.

## How To Read One Journal Line

Each line is one JSON record. It includes:

- Account snapshot.
- Trade request.
- Risk decision.
- Dry-run status.
- Competition mode.
- Timestamp.

Later, this same pattern can support a dashboard, replay tool, or final presentation
evidence.

