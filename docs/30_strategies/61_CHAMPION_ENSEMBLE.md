# Champion Ensemble

`champion_ensemble` is the current research router for the strongest tested
sleeves. It does not listen to every strategy. It uses the current evidence to
combine only:

```text
kalman_trend
asset_adaptive_dual_squeeze
fixing_reversal (optional, default weight 0.00)
macd_momentum (optional, default weight 0.00)
```

The default profile is intentionally stricter than the first high-return trial:

```text
kalman_trend_weight: 0.70
asset_adaptive_dual_squeeze_weight: 0.30
dual_squeeze_weight: 0.00
trend_pullback_weight: 0.00
fixing_reversal_weight: 0.00
macd_momentum_weight: 0.00
entry_score: 0.50
strong_lead_score: 0.50
conflict_penalty: 0.70
```

Meaning:

```text
Kalman can lead.
Asset-adaptive squeeze can confirm or block.
Asset-adaptive squeeze does not trade alone by default.
Conflicts are penalized hard.
```

The looser profile allowed asset-adaptive squeeze to trade alone and produced a
higher full-sample return, but fixed warmup walk-forward weakened. The stricter
profile is the current default because it preserved the Kalman walk-forward
shape while improving the full-sample result.

`fixing_reversal` was added as an optional diversifier after fixed-warmup scans
showed it had lower positive-fold concentration than the trend/squeeze sleeves.
Heavier fixing blends reduced concentration but did not improve active-fold
quality enough to replace the strict default.

`macd_momentum` was added as an optional momentum-acceleration diversifier after
the optimized 6/18/5 MACD profile passed the standalone short walk-forward
screen. After the MACD session filter was tightened to 10-14 UTC, a 30% MACD
champion blend improved full-sample return and median active walk-forward
return, but it became sparse and still did not beat the adaptive selector's
fold quality. MACD remains disabled in the champion default.

## Evidence

Current no-churn five-symbol basket:

```text
symbols: XAGUSD, XAUUSD, USDCHF, AUDUSD, GBPUSD
return: 0.436%
max drawdown: 0.147%
official 15m Sharpe: 0.031
fills: 90
risk discipline: 100/100
```

Fixed warmup walk-forward on the same basket:

```text
short-window folds: 17
positive fold fraction: 23.5%
active fold fraction: 41.2%
active positive fold fraction: 42.9%
non-negative fold fraction: 76.5%
median active test return: -0.003%
worst test drawdown: 0.078%
evaluation fills: 52
risk discipline: 100/100
promotion: REJECT
```

Session-filtered MACD blend scan on seven symbols:

```text
macd30 blend:
  return: 0.468%
  max drawdown: 0.143%
  official 15m Sharpe: 0.034
  trades: 30
  active positive folds: 50.0%
  non-negative folds: 82.4%
  median active return: 0.020%
  promotion: paper research only

base champion:
  return: 0.385%
  active positive folds: 44.4%
  non-negative folds: 70.6%
  median active return: -0.000%
```

Interpretation: the candidate is selective. Many folds correctly stay flat, so
the active/non-negative fold metrics are more informative than raw positive-fold
fraction alone. The no-churn allocator made the test more realistic and lowered
the promotion status, so this remains paper-only research, not automatic live
execution.

Eight-symbol eligible basket:

```text
symbols: XAGUSD, XAUUSD, USDCHF, GBPUSD, AUDUSD, EURCHF, USDJPY, USDCAD
return: 0.331%
max drawdown: 0.230%
official 15m Sharpe: 0.023
fills: 196
risk discipline: 100/100
```

Research verdict:

```text
Best current paper candidate. Keep it in dry-run and further validation before
automatic MT5 execution. The active-fold profile is acceptable for paper mode,
but not strong enough for automatic live MT5 execution.
```

## Commands

Five-symbol candidate:

```bash
quanthack-portfolio-backtest \
  --strategy champion_ensemble \
  --symbol XAGUSD --symbol XAUUSD --symbol USDCHF --symbol AUDUSD --symbol GBPUSD \
  --price-csv data/full_20gb_15m_prices.csv \
  --quote-csv data/full_20gb_15m_quotes.csv \
  --equity-output outputs/backtests/champion_ensemble_positive_subset_equity.csv \
  --pnl-output outputs/backtests/champion_ensemble_positive_subset_pnl.csv \
  --allocation-output outputs/backtests/champion_ensemble_positive_subset_allocation.csv \
  --fills-output outputs/backtests/champion_ensemble_positive_subset_fills.csv
```

Fixed warmup validation:

```bash
quanthack-portfolio-fixed-warmup-walk-forward \
  --strategy champion_ensemble \
  --symbol XAGUSD --symbol XAUUSD --symbol USDCHF --symbol AUDUSD --symbol GBPUSD \
  --price-csv data/full_20gb_15m_prices.csv \
  --quote-csv data/full_20gb_15m_quotes.csv \
  --train-size 960 \
  --test-size 192 \
  --step-size 192
```

Optimizer:

```bash
quanthack-champion-ensemble-optimize \
  --symbol XAGUSD --symbol XAUUSD --symbol USDCHF --symbol AUDUSD --symbol GBPUSD \
  --price-csv data/full_20gb_15m_prices.csv \
  --quote-csv data/full_20gb_15m_quotes.csv \
  --include-walk-forward \
  --output outputs/backtests/champion_ensemble_optimization.csv
```
