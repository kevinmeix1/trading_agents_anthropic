# Preflight

Step 12 adds a preflight command.

This command answers:

```text
Is the local system ready for dry-run?
```

It does not trade. It does not call MT5. It does not call Syphonix. It only checks
whether the project is internally coherent.

## Why This Matters

Hackathon conditions are time-sensitive and stressful. Before each work session,
you want one command that checks the boring essentials:

- Python is 3.11+.
- Config loads.
- London-time clock works.
- Price CSV has enough rows for the strategy.
- Quote CSV has a quote for the configured symbol.
- Market quality passes.
- Risk limits are still conservative.
- Dry-run journal path is writable.

This is the local version of a trading desk start-of-day checklist.

## New Files

- `src/quanthack/preflight.py`
- `scripts/setup/preflight.py`
- `tests/test_preflight.py`

## Run In VS Code Terminal

Make sure your prompt shows `(.venv)`.

```bash
python scripts/setup/preflight.py
```

Expected result:

```text
Preflight
  Python: OK - ...
  Config: OK - ...
  Clock: OK - ...
  Prices: OK - ...
  Quotes: OK - ...
  Market quality: OK - ...
  Risk limits: OK - ...
  Journal: OK - ...
  Overall: READY_FOR_DRY_RUN
```

Preflight can now also return:

```text
Overall: READY_WITH_WARNINGS
```

That means dry-run can continue, but a safety margin is getting thin.

Run a deliberately stale quote check:

```bash
python scripts/setup/preflight.py --quote-as-of "2026-06-22T10:20:10+01:00"
```

That should end with:

```text
Overall: ATTENTION_REQUIRED
```

because the sample quote is 10 seconds old and the configured limit is 5 seconds.

## How It Fits The Architecture

```text
preflight.py
  -> config.py
  -> clock.py
  -> market_data.py
  -> market_quality.py
  -> risk.py settings
  -> execution journal path
```

It checks the system, but does not create a strategy request and does not write a
trade journal entry.

## Hackathon Rule Awareness

The preflight command checks that:

- The project is using London-time competition logic.
- Risk limits are much more conservative than the allowed 1:30 leverage.
- The internal margin floor is still high.
- Market quality is checked before strategy/risk/execution.

This helps prevent accidental rule-blind changes as the project grows.
