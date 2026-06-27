# HTML Report

Step 14 adds a standalone HTML report.

The report turns the dry-run journal into a browser-readable dashboard. It does
not trade, connect to MT5, connect to Syphonix, or call any external service.

## Why This Matters

The hackathon rewards systems that can be explained.

The HTML report helps show:

- How many dry-run decisions happened.
- How many were accepted or blocked.
- How much notional the strategy requested.
- How much notional risk actually allowed.
- How much notional risk trimmed.
- Which competition modes appear in the journal.
- Recent decision records.

This is useful for daily review and final presentation evidence.

## New Files

- `src/quanthack/html_report.py`
- `scripts/reporting/build_html_report.py`
- `tests/test_html_report.py`

## Run In VS Code Terminal

Make sure your prompt shows `(.venv)`.

```bash
python scripts/reporting/build_html_report.py
```

Expected output:

```text
Report: outputs/reports/journal_report.html
Records: ...
```

Then open this file:

```text
outputs/reports/journal_report.html
```

## Report Sections

### Summary

Shows headline metrics:

- Records.
- Accepted.
- Blocked.
- Accepted rate.
- Requested notional.
- Adjusted notional.
- Trimmed by risk.

### By Status

Counts dry-run statuses such as:

- `DRY_RUN_ACCEPTED`
- `DRY_RUN_BLOCKED`

### By Mode

Counts competition modes such as:

- `QUALIFY`
- `CHECKPOINT_PROTECT`

This helps prove the system is aware of checkpoint windows.

### By Symbol

Shows per-symbol totals.

Right now most data is `EURUSD` because we only have one simple configured
strategy.

### Recent Records

Shows the latest journal entries.

## Important

This report is an explanation layer. It reads the journal after dry-run decisions
already happened. It does not approve trades, block trades, or execute trades.

