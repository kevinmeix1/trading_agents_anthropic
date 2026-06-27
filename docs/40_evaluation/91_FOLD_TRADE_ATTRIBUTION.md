# Fold Trade Attribution

Date: 2026-06-23

`fold-trade-attribution` assigns realized fill P&L back to the walk-forward fold
where the opening lot was entered. This helps answer a more useful question than
"which full backtest won?":

```text
Which fold, symbol, signal, UTC hour, and side actually made or lost money?
```

## Command

Core metals/USDCHF sleeve:

```bash
python -c 'from quanthack.cli.fold_trade_attribution import main; main()' \
  --fills-csv outputs/research/subset_metals_usdchf_core_fills.csv \
  --folds-csv outputs/research/wf_core_plain_folds.csv \
  --output outputs/research/core_plain_fold_trade_attribution.csv \
  --limit 8
```

Baseline sleeve:

```bash
python -c 'from quanthack.cli.fold_trade_attribution import main; main()' \
  --fills-csv outputs/research/compare_current_own_competition_fills.csv \
  --folds-csv outputs/research/wf_baseline_folds.csv \
  --output outputs/research/baseline_fold_trade_attribution.csv \
  --limit 8
```

## Latest Findings

Core plain sleeve:

- Weakest rows were mainly fold 1 and fold 4 `kalman_trend` trades.
- The biggest winner was fold 5 `XAGUSD` `kalman_trend` short around UTC hour 12.
- The fold 5 metal short dominated the positive P&L, which is why full-sample
  return looked much better than fold stability.

Core session/regime/vol-target sleeve:

- The session gate moved exposure later in the day and sharply reduced drawdown.
- It also shrank the large fold 5 winner into a much smaller gain.
- That improves stability but removes too much return for a winning main book.

Baseline sleeve:

- Weakest rows were fold 4 `XAUUSD` and fold 1 metals trades.
- Strongest rows were still fold 5 metal shorts, especially `XAGUSD`.

## Interpretation

This is not evidence that "metals are always good." It is evidence that our
current paper return comes from a specific metals trend episode. The same signal
family also loses in other folds.

Useful next work:

1. Keep fold attribution in every promotion review.
2. Avoid promoting a candidate whose positive return is dominated by one fold.
3. Prefer strategy changes that improve folds 1, 3, and 4 without deleting all
   of fold 5's return.
