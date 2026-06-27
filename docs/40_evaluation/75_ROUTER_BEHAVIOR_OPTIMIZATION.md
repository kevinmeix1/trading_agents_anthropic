# Router Behavior Optimization

The first alpha-router optimizer only searched signal weights. That was not
enough: the router could still overtrade because its confirmation behavior was
too permissive.

This update adds behavior profiles to router optimization:

- `entry_score`
- `exit_score`
- `min_signal_confidence`
- `cost_buffer`
- `conflict_penalty`
- `primary_signal_override_enabled`

## Full-Sample Result

Command:

```bash
quanthack router-optimize \
  --symbol XAGUSD --symbol XAUUSD --symbol USDCHF --symbol AUDUSD \
  --symbol GBPUSD --symbol EURUSD --symbol EURGBP \
  --price-csv data/full_20gb_15m_prices.csv \
  --quote-csv data/full_20gb_15m_quotes.csv \
  --candidate 0.30,0.15,0.15,0.35,0.25,0.00,0.00,0.00,0.00 \
  --conservative-behavior-grid \
  --output outputs/backtests/router_behavior_conservative_grid.csv
```

Best profile:

```text
entry_score: 0.55
exit_score: 0.12
min_signal_confidence: 0.20
cost_buffer: 1.20
conflict_penalty: 0.70
primary_signal_override_enabled: false
```

Result:

```text
return: 0.112%
max drawdown: 0.044%
official 15m Sharpe: 0.026
trades: 60
risk discipline: 100/100
```

For comparison, the current default router behavior produced:

```text
return: -0.468%
max drawdown: 0.747%
official 15m Sharpe: -0.031
trades: 2282
```

Interpretation: stricter confirmation materially improves the router by cutting
noise and turnover. This is a real engineering improvement.

## Walk-Forward Result

Command:

```bash
quanthack portfolio-router-walk-forward \
  --symbol XAGUSD --symbol XAUUSD --symbol USDCHF --symbol AUDUSD \
  --symbol GBPUSD --symbol EURUSD --symbol EURGBP \
  --price-csv data/full_20gb_15m_prices.csv \
  --quote-csv data/full_20gb_15m_quotes.csv \
  --candidate 0.30,0.15,0.15,0.35,0.25,0.00,0.00,0.00,0.00 \
  --behavior-candidate 0.35,0.20,1.20,0.50,true \
  --behavior-candidate 0.55,0.20,1.20,0.70,false \
  --train-size 480 \
  --test-size 96 \
  --step-size 480
```

Result:

```text
folds: 4
most selected behavior: entry=0.55; override=off
stable fold fraction: 0.0%
selected was test-best: 75.0%
median test return: 0.000%
worst test drawdown: 0.075%
promotion: REJECT
```

Interpretation: the stricter router is much safer and less noisy, but it is not
yet a live candidate. It often goes flat in the next test window. Keep it as a
research improvement and use it as the baseline for future router work.

## Current Decision

Do not replace the current paper candidate with alpha-router yet. The current
adaptive/static trend-heavy profile still has better realized return. The router
needs better underlying signals, not just better confirmation thresholds.
