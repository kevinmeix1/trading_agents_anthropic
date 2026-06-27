# Portfolio Walk-Forward Validation

Portfolio universe scan is useful, but it can still overfit one window.

Portfolio walk-forward validation adds a stricter question:

```text
If we pick the best basket and strategy on earlier data, does that choice still
look good on later unseen data?
```

## What It Does

For each fold:

1. Take an earlier train window.
2. Run `portfolio-universe-scan` logic on that train window.
3. Select the best basket + strategy by proxy score.
4. Take the next unseen test window.
5. Evaluate the selected candidate against the same basket/strategy candidates.
6. Mark whether the selected candidate was stable out-of-sample.

This is more realistic than picking the best result from one full backtest.

## Run It

```bash
quanthack portfolio-walk-forward \
  --strategy alpha_router \
  --strategy ma_crossover \
  --price-csv data/downloaded_scan_prices.csv \
  --quote-csv data/downloaded_scan_quotes.csv \
  --train-size 96 \
  --test-size 48 \
  --step-size 48
```

Default outputs:

```text
outputs/backtests/portfolio_walk_forward_summary.csv
outputs/backtests/portfolio_walk_forward_folds.csv
```

## Use A Custom Basket Set

```bash
quanthack portfolio-walk-forward \
  --strategy alpha_router \
  --strategy ma_crossover \
  --basket core_fx:EURUSD,GBPUSD,USDJPY \
  --basket fx_gold:EURUSD,USDJPY,XAUUSD \
  --price-csv data/downloaded_scan_prices.csv \
  --quote-csv data/downloaded_scan_quotes.csv \
  --train-size 96 \
  --test-size 48 \
  --step-size 48
```

Custom baskets are useful when you want to keep the search focused and avoid
testing too many combinations too early.

## How To Read The Summary

Important summary fields:

- `eligible`: whether the walk-forward result passes the stability filters.
- `promotion_status`: `PROMOTE`, `PAPER_ONLY`, or `REJECT`. This is the live
  gate; do not promote a full-sample winner unless walk-forward promotion passes.
- `promotion_reason`: plain-English explanation for the promotion decision.
- `stable_fold_fraction`: share of folds where the selected candidate made money,
  respected drawdown, respected risk discipline, and had enough fills.
- `median_test_proxy_score`: median out-of-sample rank score.
- `median_test_return_pct`: median return on unseen windows.
- `worst_test_drawdown_pct`: worst fold drawdown.
- `average_risk_discipline_score`: average test-window risk score.
- `most_selected_basket`: basket most often selected by train windows.
- `most_selected_strategy`: strategy most often selected by train windows.

Important fold fields:

- `selected_basket`
- `selected_symbols`
- `selected_strategy`
- `train_proxy_score`
- `test_proxy_score`
- `test_best_candidate`
- `selected_was_test_best`
- `stable_candidate`
- `test_return_pct`
- `test_drawdown_pct`
- `test_sharpe_15m`
- `test_risk_discipline_score`

## Why It Matters

The hackathon rewards return, drawdown, Sharpe, and risk discipline. A candidate
that only wins on the same data used to select it is not trustworthy enough for a
live MT5 dry run.

Treat the promotion fields as the final research gate:

```text
full-sample winner -> research candidate only
walk-forward eligible but weak lower quartile -> paper-only candidate
walk-forward promoted -> live dry-run candidate
```

Use this workflow before spending time on router optimization:

```text
import downloaded data -> universe scan -> portfolio walk-forward -> router optimization
```

After a basket survives this workflow, tune `alpha_router` weights out of sample:

```bash
quanthack portfolio-router-walk-forward \
  --symbol EURUSD --symbol USDJPY --symbol XAUUSD \
  --price-csv data/downloaded_portfolio_prices.csv \
  --quote-csv data/downloaded_portfolio_quotes.csv
```
