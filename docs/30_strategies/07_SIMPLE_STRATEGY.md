# Simple Strategy

Step 7 adds the first strategy.

This strategy is intentionally simple. It does not use machine learning, news,
LLMs, or broker data yet. It only looks at a small list of recent prices.

## What The Strategy Does

The strategy measures price momentum:

```text
move_bps = (last_price / first_price - 1) * 10000
```

`bps` means basis points:

```text
100 bps = 1%
8 bps = 0.08%
```

Then:

- If the move is above the threshold, it proposes `BUY`.
- If the move is below the negative threshold, it proposes `SELL`.
- If the move is small, it proposes nothing.

## Important Architecture Rule

The strategy does not trade.

It only creates a `TradeRequest`.

The flow remains:

```text
strategy proposes
  -> risk checks
  -> dry-run executor journals
```

## New Files

- `src/quanthack/strategy.py`
- `scripts/evaluation/strategy_demo.py`
- `scripts/dry_run/strategy_dry_run.py`
- `tests/test_strategy.py`

## Run In VS Code Terminal

Make sure your prompt shows `(.venv)`.

Show what the strategy proposes:

```bash
python scripts/evaluation/strategy_demo.py --scenario up
python scripts/evaluation/strategy_demo.py --scenario down
python scripts/evaluation/strategy_demo.py --scenario flat
```

Run the strategy through risk and the dry-run journal:

```bash
python scripts/dry_run/strategy_dry_run.py --scenario up
python scripts/dry_run/strategy_dry_run.py --scenario down
python scripts/dry_run/strategy_dry_run.py --scenario flat
python scripts/inspect/show_journal.py
```

Run all tests:

```bash
python -m unittest discover -s tests
```

## What To Notice

The `flat` scenario should say:

```text
Strategy output: NO TRADE
```

That is good. A trading system should be comfortable doing nothing.

The `up` and `down` scenarios should create requests, but risk still decides the
final adjusted notional before the dry-run executor writes a journal record.

