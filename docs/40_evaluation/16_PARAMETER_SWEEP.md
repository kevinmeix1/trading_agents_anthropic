# Parameter Sweep

Step 16 adds a simple train/test parameter sweep.

The goal is to stop guessing strategy parameters by hand.

We now test combinations of:

- `lookback`
- `threshold_bps`

and rank them by test-set performance.

## Why This Matters

Backtesting one parameter set answers:

```text
How did this exact setup behave?
```

Sweeping answers:

```text
Which settings look better, and do they still look decent out of sample?
```

This is a first guard against overfitting. It is not full research-grade validation
yet, but it is much better than tuning from vibes.

## New Files

- `src/quanthack/sweep.py`
- `scripts/evaluation/run_sweep.py`
- `tests/test_sweep.py`

## Config

In `configs/default.toml`:

```toml
[sweep]
lookbacks = [3, 5, 7]
threshold_bps = [4.0, 8.0, 12.0]
train_fraction = 0.6
results_csv = "outputs/backtests/parameter_sweep.csv"
```

The current sample data has 20 bars. With `train_fraction = 0.6`, the sweep uses:

```text
12 train bars
8 test bars
```

## Run In VS Code Terminal

Make sure your prompt shows `(.venv)`.

```bash
python scripts/evaluation/run_sweep.py
```

This writes:

```text
outputs/backtests/parameter_sweep.csv
```

Run tests:

```bash
python -m unittest discover -s tests
```

## How Ranking Works

Candidates are sorted by:

1. Test Sharpe ratio.
2. Test total return.
3. Lower test max drawdown.

We rank by test metrics because train-only ranking is a classic overfitting trap.

## Important Limitations

The current data is tiny and synthetic. Do not treat the best parameter as a real
trading conclusion.

This step proves that the evaluation machinery works:

- Split history into train/test.
- Run many strategy configs.
- Use the same backtest engine.
- Save results to CSV.
- Rank candidates consistently.

Next we can add larger historical data, walk-forward windows, and more strategies.
