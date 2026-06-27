# Backtest P&L Attribution

Step 21 adds trade-level P&L attribution for backtests.

Before this step, the backtest could say:

```text
Final equity changed by X.
```

Now it can also explain:

```text
How much P&L was realized by fills?
How much P&L remains open in the final position?
Which fills changed the running average entry price?
```

## Why This Matters

Final equity is useful, but it is not enough for debugging.

If a strategy looks good or bad, we need to understand whether the result came
from:

- Closed trades.
- Open mark-to-market P&L.
- Reversing from long to short.
- Repeatedly resizing the same position.

This is especially useful for a hackathon demo because it shows that the strategy
is not just producing a number; the system can explain where that number came
from.

## New Files

- `src/quanthack/pnl.py`
- `tests/test_pnl.py`

## Output

Running:

```bash
python scripts/evaluation/run_backtest.py
```

now writes:

```text
outputs/backtests/equity_curve.csv
outputs/backtests/pnl_ledger.csv
```

The P&L ledger includes:

- Fill timestamp.
- Side.
- Fill price.
- Trade units.
- Realized P&L from that fill.
- Cumulative realized P&L.
- Position units after the fill.
- Average entry price after the fill.
- Open P&L at the fill price.

## Important Accounting Rule

Positions use signed units:

```text
positive units = long
negative units = short
```

For a long position:

```text
realized P&L = closed_units * (exit_price - entry_price)
```

For a short position:

```text
realized P&L = closed_units * (entry_price - exit_price)
```

The same implementation handles partial exits and flips from long to short or
short to long.

## Limitation

This is still a simulation. It uses the backtest fill prices and final mark price.
It does not claim to match real broker fills, commissions, swaps, or financing.
