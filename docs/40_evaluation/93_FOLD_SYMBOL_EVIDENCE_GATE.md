# Fold Symbol Evidence Gate

Date: 2026-06-23

`fold-symbol-evidence` is a research diagnostic for a possible adaptive symbol
gate. It uses only prior fold evidence for each symbol, then simulates whether
the next fold's symbol contribution would have been allowed or blocked.

This is not a live execution rule yet. It is attribution math, so it should be
treated as a hypothesis generator before wiring anything into the portfolio
backtest engine.

## Commands

Core metals/USDCHF evidence sweep:

```bash
python -c 'from quanthack.cli.fold_symbol_evidence_sweep import main; main()' \
  --attribution-csv outputs/research/core_plain_fold_trade_attribution.csv \
  --folds-csv outputs/research/wf_core_plain_folds.csv \
  --symbol USDCHF --symbol XAGUSD --symbol XAUUSD \
  --output outputs/research/core_plain_fold_symbol_evidence_sweep.csv \
  --lookback-folds 1 --lookback-folds 2 --lookback-folds 3 \
  --min-prior-pnl-usd -5000 --min-prior-pnl-usd -1000 \
  --min-prior-pnl-usd -250 --min-prior-pnl-usd 0 \
  --min-prior-pnl-usd 250 --min-prior-pnl-usd 1000 \
  --min-prior-win-rate 0 --min-prior-win-rate 0.5
```

Baseline evidence sweep:

```bash
python -c 'from quanthack.cli.fold_symbol_evidence_sweep import main; main()' \
  --attribution-csv outputs/research/baseline_fold_trade_attribution.csv \
  --folds-csv outputs/research/wf_baseline_folds.csv \
  --output outputs/research/baseline_fold_symbol_evidence_sweep.csv \
  --lookback-folds 1 --lookback-folds 2 --lookback-folds 3 \
  --min-prior-pnl-usd -5000 --min-prior-pnl-usd -1000 \
  --min-prior-pnl-usd -250 --min-prior-pnl-usd 0 \
  --min-prior-pnl-usd 250 --min-prior-pnl-usd 1000 \
  --min-prior-win-rate 0 --min-prior-win-rate 0.5
```

## Latest Result

Best policy in both tested samples:

```text
lookback_folds = 1
min_prior_pnl_usd = 1000
min_prior_win_rate = 0.0
allow_without_history = true
```

Core plain sleeve:

```text
ungated realized P&L: $53,995.16
gated realized P&L:   $55,329.66
simulated delta:       $1,334.50
avoided loss:          $1,334.50
missed gain:               $0.00
allowed symbol-folds:       77.8%
```

Baseline sleeve:

```text
ungated realized P&L: $51,791.35
gated realized P&L:   $60,364.49
simulated delta:       $8,573.14
avoided loss:          $9,202.41
missed gain:             $629.27
allowed symbol-folds:       66.7%
```

## Interpretation

This is the first adaptive symbol-gate idea that improves the attribution
simulation without deleting fold 5's large metal winner. The one-fold lookback
works better than longer lookbacks because it reacts quickly and does not keep
symbols blocked for too long.

Important caveat:

```text
This does not prove live performance.
```

The report works on realized attribution rows, not a full rerun where the gate
changes positions, allocator state, risk trimming, and later cost basis. Treat
it as a candidate for the next engine-level backtest, not as a promotion signal.

## Next Step

An online version has now been wired into the portfolio backtest as an optional
gate:

- `--symbol-evidence-gate`
- `--symbol-evidence-lookback-events`
- `--symbol-evidence-min-pnl-usd`
- `--symbol-evidence-stale-after-bars`

It:

- allows cold-start probes;
- tracks only P&L from trades that actually executed;
- blocks new or increased exposure after poor recent symbol evidence;
- always allows exits and reductions;
- can expire stale evidence after a configured number of bars.

## Online Backtest Result

Core USDCHF/metals sleeve, online gate with `min_prior_pnl_usd=1000` and stale
evidence after 96 bars:

```text
full-sample return: 2.390%
max drawdown:       1.789%
trades:             32
risk score:         100/100
```

Broad 10-symbol baseline, same gate:

```text
full-sample return: 4.425%
max drawdown:       2.584%
trades:             108
risk score:         100/100
```

Candidate scorecard:

| Candidate | Return | Max DD | Sharpe 15m | Trades | Score |
| --- | ---: | ---: | ---: | ---: | ---: |
| `core_plain` | 5.511% | 1.456% | 0.036 | 42 | 97.5 |
| `baseline_gated` | 4.425% | 2.584% | 0.026 | 108 | 42.5 |
| `core_gated` | 2.390% | 1.789% | 0.018 | 32 | 10.0 |

Fixed-warmup validation still rejects the online gate:

| Candidate | Non-negative folds | Active positive folds | Median active return | Fills | Promotion |
| --- | ---: | ---: | ---: | ---: | --- |
| `core_gated_stale96` | 66.7% | 50.0% | -0.021% | 18 | REJECT |
| `baseline_gated_stale96` | 16.7% | 16.7% | -0.035% | 70 | REJECT |

Verdict:

```text
Keep the online symbol evidence gate as research infrastructure.
Do not promote it as the live MT5 strategy layer yet.
```

The online version is more honest than attribution simulation. It improves some
same-window paper metrics, but it does not solve the fold-stability problem.
The next alpha work should focus on a genuine return diversifier, especially
official crypto data/strategy coverage, rather than more filters on the same
metals trend episode.
