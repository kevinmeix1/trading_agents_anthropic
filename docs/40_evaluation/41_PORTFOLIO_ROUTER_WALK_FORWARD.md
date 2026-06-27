# Portfolio Router Walk-Forward

This workflow tunes `alpha_router` weights on train windows, then evaluates the
chosen weights on later unseen test windows.

It answers:

```text
Do the router weights that look best in training still behave well out of sample?
```

## Why It Exists

`router-optimize` is useful, but it can overfit one full backtest window.

`portfolio-router-walk-forward` is stricter:

1. Split data into chronological train/test folds.
2. Run router weight optimization on the train window only.
3. Freeze the best train-window weights.
4. Evaluate those exact weights on the next test window.
5. Repeat and summarize stability.

## Run It

```bash
quanthack portfolio-router-walk-forward \
  --symbol EURUSD --symbol GBPUSD --symbol USDJPY \
  --symbol XAUUSD --symbol XAGUSD \
  --price-csv data/downloaded_scan_prices.csv \
  --quote-csv data/downloaded_scan_quotes.csv \
  --train-size 96 \
  --test-size 48 \
  --step-size 48
```

Default outputs:

```text
outputs/backtests/portfolio_router_walk_forward_summary.csv
outputs/backtests/portfolio_router_walk_forward_folds.csv
```

## Custom Weight Candidates

```bash
quanthack portfolio-router-walk-forward \
  --symbol EURUSD --symbol USDJPY --symbol XAUUSD \
  --candidate 0.40,0.20,0.35,0.25 \
  --candidate 0.25,0.15,0.15,0.35,0.20,0.10 \
  --candidate 0.20,0.10,0.10,0.40,0.15,0.05,0.10 \
  --candidate 0,0,0,0,0,0,0,1 \
  --price-csv data/downloaded_scan_prices.csv \
  --quote-csv data/downloaded_scan_quotes.csv
```

Four-number candidates are:

```text
momentum_weight,moving_average_weight,breakout_weight,mean_reversion_weight
```

Six-number candidates add the newer router sleeves:

```text
momentum_weight,moving_average_weight,breakout_weight,mean_reversion_weight,session_breakout_weight,cross_rate_weight
```

Seven-number candidates append relative strength:

```text
momentum_weight,moving_average_weight,breakout_weight,mean_reversion_weight,session_breakout_weight,cross_rate_weight,relative_strength_weight
```

Eight-number candidates append volatility squeeze:

```text
momentum_weight,moving_average_weight,breakout_weight,mean_reversion_weight,session_breakout_weight,cross_rate_weight,relative_strength_weight,volatility_squeeze_weight
```

For example, `0,0,0,0,0,0,0,1` tests a squeeze-only router candidate under the
same allocator and risk gates as the blended router candidates.

## How To Read The Summary

Important fields:

- `eligible`: whether the out-of-sample result passes stability filters.
- `promotion_status`: conservative verdict for whether to promote the researched
  weights, keep them paper-only, or reject them.
- `promotion_reason`: the plain-English reason behind that verdict.
- `most_selected_weights`: weights most often selected during train windows.
- `stable_fold_fraction`: share of folds with positive return, acceptable
  drawdown, enough fills, and acceptable risk discipline.
- `selected_was_test_best_fraction`: how often the train-selected weights were
  also best on the test window.
- `median_test_proxy_score`: median out-of-sample competition-style proxy score.
- `median_test_return_pct`: median test-window return.
- `worst_test_drawdown_pct`: worst test-window drawdown.
- `average_risk_discipline_score`: average risk discipline on test windows.

The promotion gate intentionally rejects candidates whose median test-window
return is economically tiny, even when drawdown and risk discipline look clean.
The default minimum is about `0.005%` median test return. A flat but safe router
belongs in research, not live MT5 execution.

## Recommended Research Order

```text
import larger downloaded sample
-> portfolio-universe-scan
-> portfolio-walk-forward
-> portfolio-router-walk-forward
-> detailed portfolio-backtest on the best candidate
```

If the router walk-forward is not stable, do not move to live MT5 execution yet.
If `promotion_status` is not `PROMOTE`, keep the weights in research mode.
