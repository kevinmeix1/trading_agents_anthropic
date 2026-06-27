# Adaptive Strategy Policy Sweep

`adaptive-strategy-policy-sweep` ranks policy settings for the raw adaptive
strategy selector.

It is different from `adaptive-strategy-select`:

```text
adaptive-strategy-select:
  run one policy

adaptive-strategy-policy-sweep:
  run one or more policies and rank them in one CSV
```

Default command:

```bash
PYTHONPATH=src .venv/bin/python -c 'from quanthack.cli.adaptive_strategy_policy_sweep import main; main()' \
  --config configs/competition.toml \
  --strategy kalman_trend \
  --strategy champion_ensemble \
  --strategy macd_momentum \
  --symbol XAGUSD --symbol XAUUSD --symbol USDCHF --symbol AUDUSD \
  --symbol GBPUSD --symbol EURUSD --symbol EURGBP \
  --price-csv data/full_20gb_15m_prices.csv \
  --quote-csv data/full_20gb_15m_quotes.csv \
  --train-size 480 \
  --test-size 96 \
  --step-size 96 \
  --output outputs/research/adaptive_strategy_policy_sweep.csv
```

The default policy is intentionally safe and small:

```text
loss_cooldown_folds = 1
min_train_adjusted_return_pct = none
transition_risk_multiplier = 1.0
cash_fallback = no
```

Why the default is small:

```text
The implementation caches each fold's candidate training scores and candidate
test-window evaluations, then replays policy settings from that cache.

That makes compact grids usable, but not free:
  one baseline policy: about 57 seconds before caching, about 26 seconds after
  compact 36-policy grid: about 39 seconds after caching
```

Use repeated flags for controlled sweeps:

```bash
PYTHONPATH=src .venv/bin/python -c 'from quanthack.cli.adaptive_strategy_policy_sweep import main; main()' \
  --config configs/competition.toml \
  --strategy kalman_trend \
  --strategy champion_ensemble \
  --strategy macd_momentum \
  --symbol XAGUSD --symbol XAUUSD --symbol USDCHF --symbol AUDUSD \
  --symbol GBPUSD --symbol EURUSD --symbol EURGBP \
  --price-csv data/full_20gb_15m_prices.csv \
  --quote-csv data/full_20gb_15m_quotes.csv \
  --train-size 480 \
  --test-size 96 \
  --step-size 96 \
  --loss-cooldown-folds 1 \
  --min-train-adjusted-return-pct none \
  --min-train-adjusted-return-pct 0.0 \
  --min-train-adjusted-return-pct 0.0001 \
  --cash-fallback no \
  --cash-fallback yes \
  --output outputs/research/adaptive_strategy_policy_sweep.csv
```

2026-06-23 compact grid output:

```text
output:
  outputs/research/adaptive_strategy_policy_sweep.csv

grid:
  loss_cooldown_folds: 0, 1, 2
  min_train_adjusted_return_pct: none, 0.0, 0.0001
  transition_risk_multiplier: 1.0, 0.75
  cash_fallback: no, yes

top policy:
  status: PAPER_ONLY
  selector score: 85.20
  loss_cooldown_folds: 1
  min_train_adjusted_return_pct: none
  transition_risk_multiplier: 1.0
  cash_fallback: no
  positive folds: 41.2%
  active-positive folds: 77.8%
  non-negative folds: 88.2%
  compounded OOS return: 5.245%
  median active return: 0.411%
  worst drawdown: 0.812%
  fills: 72
  selection counts:
    kalman_trend=7 champion_ensemble=3 macd_momentum=7

notable challengers:
  transition_risk_multiplier=0.75:
    same fold quality, lower compounded return at 4.303%
  min_train_adjusted_return_pct=0.0001 with cash fallback:
    non-negative folds improve to 94.1%, but compounded return drops to 4.841%
  min_train_adjusted_return_pct=0.0:
    compounded return rises to 5.530%, but active-positive folds fall to 62.5%
```

Current conclusion:

```text
Keep the current adaptive stack as the paper leader.
Cash fallback remains a challenger from one-off tests, not a default.
Transition-risk scaling is safer-looking but too return-dilutive on this window.
The cached compact sweep is now safe enough to run after major strategy changes.
```
