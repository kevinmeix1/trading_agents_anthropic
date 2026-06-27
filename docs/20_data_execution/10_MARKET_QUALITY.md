# Market Quality Checks

Step 10 adds quote quality checks.

We are still offline and dry-run only. No MT5, no Syphonix, no API, and no live
paper orders.

## Why Market Quality Matters

A strategy can be reasonable and still fail if the market data is bad.

Examples:

- The quote is stale.
- The quote timestamp is after the time you are pretending to evaluate.
- The spread is too wide.
- The symbol has no quote.

For the hackathon, this matters because the rules include elimination and margin
risk. Trading bad quotes near a checkpoint is a very avoidable way to damage the
account path.

## New Files

- `data/sample_quotes.csv`
- `src/quanthack/market_quality.py`
- `scripts/inspect/show_quotes.py`
- `scripts/dry_run/quality_data_strategy_dry_run.py`
- `tests/test_market_quality.py`

## New Config Settings

In `configs/default.toml`:

```toml
[market_data]
price_csv = "data/sample_prices.csv"
quote_csv = "data/sample_quotes.csv"

[market_quality]
max_spread_bps = 10.0
max_quote_age_seconds = 5.0
```

These are starter values. For a live platform, we will revisit them by symbol
after seeing real spreads and quote timestamps.

The offline data-health validator now uses instrument metadata for symbol-specific
spread limits when you run `quanthack-validate-data`. For example, EURUSD,
XAUUSD, and XAGUSD are not judged against one identical spread threshold.

## New Flow

The safer flow is now:

```text
quote CSV
  -> market quality check
  -> price CSV
  -> strategy
  -> risk engine
  -> dry-run journal
```

If market quality fails, we do not even ask the strategy to propose a trade.

## Run In VS Code Terminal

Make sure your prompt shows `(.venv)`.

Show the configured quote:

```bash
python scripts/inspect/show_quotes.py
```

Show a wide-spread quote:

```bash
python scripts/inspect/show_quotes.py --symbol BTCUSD
```

Show a stale quote:

```bash
python scripts/inspect/show_quotes.py --as-of "2026-06-22T10:20:10+01:00"
```

Run the full quality-gated dry-run pipeline:

```bash
python scripts/dry_run/quality_data_strategy_dry_run.py
python scripts/dry_run/quality_data_strategy_dry_run.py --symbol BTCUSD
python scripts/dry_run/quality_data_strategy_dry_run.py --as-of "2026-06-22T10:20:10+01:00"
python scripts/dry_run/quality_data_strategy_dry_run.py --mode CHECKPOINT_PROTECT --equity 1020000
python scripts/inspect/show_journal.py --limit 8
```

Run tests:

```bash
python -m unittest discover -s tests
```

## What To Notice

- `EURUSD` should pass market quality.
- `BTCUSD` should block because the sample spread is wide.
- A quote checked 10 seconds after its timestamp should block because our starter
  limit is 5 seconds.
- `CHECKPOINT_PROTECT` should still reduce notional when the account is up.

This is how we keep hackathon rule awareness layered into the project instead of
bolting it on at the end.
