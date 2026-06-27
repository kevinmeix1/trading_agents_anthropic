# Journal Summary

Step 13 adds a journal summary command.

The dry-run journal already records individual decisions. The summary turns those
records into useful evidence:

- How many decisions were recorded?
- How many were accepted?
- How many were blocked?
- How much notional did strategies request?
- How much notional did risk actually allow?
- How much exposure did risk trim?
- Which modes and symbols appear in the journal?

## Why This Matters For The Hackathon

The competition is not only about signals. You also need to show that your system
is controlled, explainable, and replayable.

A journal summary helps with:

- Daily reviews.
- Debugging.
- Final presentation evidence.
- Showing that risk controls are active, not decorative.

## New Files

- `src/quanthack/journal_report.py`
- `scripts/reporting/journal_summary.py`
- `tests/test_journal_report.py`

## Key Terms

### Requested Notional

What the strategy asked for.

Example:

```text
BUY EURUSD $50,000
```

### Adjusted Notional

What the risk engine allowed.

Example:

```text
Strategy requested $50,000.
Checkpoint protection allowed $25,000.
```

### Trimmed By Risk

The difference between requested and adjusted notional.

```text
trimmed = requested - adjusted
```

This is a good number to track. It shows when risk controls actually reduced
exposure.

## Run In VS Code Terminal

Make sure your prompt shows `(.venv)`.

```bash
python scripts/reporting/journal_summary.py
```

Show fewer recent records:

```bash
python scripts/reporting/journal_summary.py --recent 2
```

Show only aggregate summary:

```bash
python scripts/reporting/journal_summary.py --recent 0
```

Run tests:

```bash
python -m unittest discover -s tests
```

## How It Fits The Architecture

```text
outputs/dry_run_journal.jsonl
  -> read_journal()
  -> summarize_journal()
  -> journal_summary.py output
```

This is downstream from execution. It does not create trades, approve trades, or
change risk. It only reads the audit trail.

## What To Watch

If `Trimmed by risk` is always zero, risk may not be doing much yet.

If `Blocked` grows quickly, the system may be too aggressive, the data may be bad,
or the limits may be too strict.

If most records occur in `CHECKPOINT_PROTECT`, review whether you are testing too
close to checkpoint windows.

