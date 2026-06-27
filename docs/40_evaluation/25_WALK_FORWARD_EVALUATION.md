# Walk-Forward Evaluation

Step 25 adds chronological out-of-sample evaluation.

A single backtest answers:

```text
What happened on this one historical sample?
```

Walk-forward evaluation asks a stricter question:

```text
If we chose parameters using only past data, what happened on the next unseen window?
```

## New Command

```bash
python scripts/evaluation/run_walk_forward.py
```

This writes:

```text
outputs/backtests/walk_forward_summary.csv
outputs/backtests/walk_forward_folds.csv
```

## How It Works

The configured sample uses:

```text
train_size = 10 bars
test_size = 5 bars
step_size = 5 bars
```

For each fold:

1. Use the training window first.
2. Select strategy parameters using only that training window.
3. Freeze the selected parameters.
4. Evaluate on the next test window.
5. Repeat forward in time.

The test window always starts after the train window, so future data is not used
to pick parameters for the past.

For `simple_momentum`, walk-forward tests configured momentum lookbacks and
thresholds.

For `ma_crossover`, walk-forward tests configured fast windows, slow windows,
and minimum separation thresholds:

```toml
[walk_forward]
ma_fast_windows = [2, 3]
ma_slow_windows = [5, 8]
ma_min_separation_bps = [1.0, 2.0]
```

Invalid pairs such as `fast_window >= slow_window` are skipped.

## What It Reports

The summary CSV includes:

- Median out-of-sample Sharpe.
- Lower-quartile out-of-sample return.
- Worst out-of-sample drawdown.
- Profitable fold fraction.
- Total and median fills.
- Total turnover.
- Returns under `1.5x` and `2.0x` cost stress.
- Eligibility flag.

The fold CSV includes the train/test dates, selected parameters, train metrics,
and test metrics for each fold.

## Why This Matters

The parameter sweep can overfit because it ranks candidates from one train/test
split. Walk-forward evaluation is still not proof of profitability, especially
with tiny synthetic data, but it is a better habit:

```text
choose using past data -> judge on future data
```

That is closer to how a real trading strategy must survive.

## Run In VS Code Terminal

```bash
python scripts/evaluation/run_walk_forward.py
```

Try one strategy:

```bash
python scripts/evaluation/run_walk_forward.py --strategy simple_momentum
python scripts/evaluation/run_walk_forward.py --strategy ma_crossover
```

Use different fold sizes:

```bash
python scripts/evaluation/run_walk_forward.py --train-size 12 --test-size 4 --step-size 2
```

Try a custom moving-average grid:

```bash
quanthack walk-forward --strategy ma_crossover \
  --ma-fast-window 2 --ma-fast-window 3 \
  --ma-slow-window 5 --ma-slow-window 8 \
  --ma-min-separation-bps 1 --ma-min-separation-bps 2
```

Run tests:

```bash
python -m unittest tests.test_walk_forward
```

## Important Limitation

The included CSV is intentionally tiny. Use this tool to prove the evaluation
workflow works. Do not claim a strategy is good until it survives larger,
cleaner, cost-aware historical data.
