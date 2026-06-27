# Stateful Dry-Run Positions

Step 20 makes dry-run risk checks remember journaled positions.

Before this step, many scripts passed an empty `PortfolioSnapshot` into risk.
That meant each dry-run decision acted like the account had no existing exposure.

Now the dry-run executor can rebuild portfolio exposure from the journal:

```text
accepted BUY  EURUSD $50,000 -> EURUSD long  $50,000
accepted SELL EURUSD $75,000 -> EURUSD short $75,000
blocked trades                 -> ignored for position state
```

The latest accepted target for each symbol is treated as the current dry-run
position.

## Why This Matters

Risk checks need to know existing exposure.

If we already have accepted dry-run exposure in the journal, the next risk check
should not pretend the account is flat. This makes the dry-run workflow closer to
real account behavior while still avoiding live or paper orders.

## What Changed

- `DryRunExecutor.current_portfolio()` reconstructs positions from the journal.
- `portfolio_from_journal(...)` converts accepted records into a `PortfolioSnapshot`.
- New journal records include `portfolio_before`, so we can audit what risk saw.
- Dry-run scripts now pass the reconstructed portfolio into `RiskEngine`.
- Allocated exits targeting flat also receive a risk-engine decision before they
  are journaled.
- `scripts/inspect/show_positions.py` displays the reconstructed portfolio.

## Run In VS Code Terminal

Use a temporary journal so the demo is clean:

```bash
python scripts/dry_run/dry_run_trade.py --journal outputs/demo_positions.jsonl --side BUY --target-notional 50000
python scripts/inspect/show_positions.py --journal outputs/demo_positions.jsonl
python scripts/dry_run/dry_run_trade.py --journal outputs/demo_positions.jsonl --side SELL --target-notional 75000
python scripts/inspect/show_positions.py --journal outputs/demo_positions.jsonl
```

Expected idea:

```text
EURUSD: LONG $50,000
EURUSD: SHORT $75,000
```

The second accepted trade replaces the symbol target rather than adding another
separate EURUSD line.

## Important Limitation

This is still dry-run bookkeeping. It does not confirm broker fills, fees,
partial executions, or real account positions.

That is intentional. The project remains safe until a real broker seam is added
behind explicit gates.
