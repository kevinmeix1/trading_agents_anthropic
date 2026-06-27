# Competition Clock

Step 4 adds the first real hackathon-specific logic: the competition clock.

Before choosing a strategy, our code needs to know what kind of period we are in:

- `PRE_LIVE`: before the live trading window.
- `QUALIFY`: normal qualification period.
- `CHECKPOINT_PROTECT`: close to an elimination/checkpoint time.
- `FINAL_RANK_PUSH`: finalist mode if pursuing account equity rank.
- `FINAL_SHARPE`: finalist mode if pursuing smoother Sharpe-style returns.

## Why This Matters

The rules make time part of the strategy. A trade that might be acceptable at noon
can be too risky 30 minutes before a daily elimination checkpoint.

So we build the clock before we build any trading signal.

## Files Added

- `src/quanthack/clock.py`
- `scripts/inspect/show_competition_mode.py`
- `tests/test_clock.py`

## Run In VS Code Terminal

Make sure your prompt shows `(.venv)`, then run:

```bash
python scripts/inspect/show_competition_mode.py
python scripts/inspect/show_competition_mode.py --at "2026-06-22T21:15:00+01:00"
python -m unittest discover -s tests
```

The second command should show:

```text
Mode: CHECKPOINT_PROTECT
```

The test command should end with:

```text
OK
```

## Important Schedule Note

The supplied document flags a possible difference between private rules and the
public event page around the final dates. We are using configurable dates in code,
and we should verify the official participant portal before live trading.

