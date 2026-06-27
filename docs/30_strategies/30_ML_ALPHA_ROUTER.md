# ML Alpha Router

The alpha router now has an optional lightweight ML signal named `ml_alpha`.

It does not use a heavy library like scikit-learn. Instead, it trains a small
logistic model from recent price history using only information available before
the current decision.

## What It Learns From

For each historical point, the model builds simple price-action features:

- cumulative return over the ML lookback window
- latest return
- realized volatility
- trend efficiency
- z-score versus the recent baseline
- channel position inside or outside the recent range

The label is the next return direction, ignoring tiny moves below
`ml_label_threshold_bps`.

## Why This Is Safe For A Hackathon Demo

This is intentionally transparent and conservative:

- no external ML dependency
- no future data in the current decision
- no automatic live trading
- router still passes through market quality, risk, and dry-run journaling
- diagnostics include ML probability, score, sample count, and training accuracy

## Run It

The default config keeps ML available but disabled inside `alpha_router`.
The 2026-06-20 portfolio walk-forward calibration rejected promotion because
only 25.0% of test folds were positive and average test signed return was
-3.65 bps. Keep it as an experiment until new data says otherwise.

```bash
quanthack strategy-demo --strategy alpha_router --scenario up
quanthack backtest --strategy alpha_router
quanthack portfolio-backtest --strategy alpha_router
```

The ML signal appears alongside the existing router signals:

```text
momentum
ma_crossover
breakout
mean_reversion
ml_alpha
```

Evaluate the ML signal over historical bars:

```bash
quanthack ml-alpha-report
```

## Config

The settings live under `[strategy.alpha_router]` in `configs/default.toml`:

```toml
ml_enabled = false
ml_weight = 0.30
ml_lookback = 5
ml_train_window = 80
ml_min_train_samples = 8
ml_entry_probability = 0.58
ml_min_training_accuracy = 0.55
ml_min_samples_for_trade = 12
ml_min_expected_edge_bps = 3.0
ml_disable_on_negative_signed_return = true
```

The important idea: ML does not replace the router. When deliberately enabled
for research, it becomes one more vote, and the router still penalizes conflict
between signals.

## Guardrails

`ml_alpha` is allowed to vote only when recent evidence is strong enough:

- enough labeled samples exist
- training accuracy clears the minimum
- expected edge clears the minimum
- the rolling training signed return is positive, if that guard is enabled

If any guardrail fails, `ml_alpha` becomes `FLAT` and prints the reason.
