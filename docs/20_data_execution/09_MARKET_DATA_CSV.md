# Offline Market Data CSV

Step 9 adds offline market data input.

We are still not using MT5, Syphonix, broker APIs, or live market feeds. This step
uses a tiny local CSV so we can practice the data pipeline safely.

## Why This Matters For The Hackathon

Trading systems fail when they trust bad data.

Before we connect to a live platform, we want the code to answer:

- Does the file have the columns we expect?
- Are timestamps timezone-aware?
- Are prices positive?
- Which symbols are available?
- Do we have enough prices for the strategy lookback?

This keeps us aligned with the rules: London time matters, and we should not trade
from stale, malformed, or misunderstood data.

## New Files

- `data/sample_prices.csv`
- `src/quanthack/market_data.py`
- `scripts/inspect/show_prices.py`
- `scripts/dry_run/data_strategy_dry_run.py`
- `tests/test_market_data.py`

## CSV Format

The CSV uses three columns:

```csv
timestamp,symbol,close
2026-06-22T10:00:00+01:00,EURUSD,1.1000
```

Timestamps must include a timezone offset like `+01:00`. That is intentional:
the hackathon uses London time, and naive timestamps are dangerous.

## Run In VS Code Terminal

Make sure your prompt shows `(.venv)`.

Inspect the configured data file:

```bash
python scripts/inspect/show_prices.py
```

Run the configured strategy using CSV prices:

```bash
python scripts/dry_run/data_strategy_dry_run.py
python scripts/dry_run/data_strategy_dry_run.py --symbol XAUUSD
python scripts/dry_run/data_strategy_dry_run.py --mode CHECKPOINT_PROTECT --equity 1020000
python scripts/inspect/show_journal.py --limit 8
```

Run tests:

```bash
python -m unittest discover -s tests
```

## What To Notice

The strategy no longer uses built-in scenario lists. It reads prices from:

```text
data/sample_prices.csv
```

The full flow is now:

```text
CSV prices
  -> market data loader
  -> strategy
  -> risk engine
  -> dry-run journal
```

This is one step closer to a live platform adapter, but still safely offline.

