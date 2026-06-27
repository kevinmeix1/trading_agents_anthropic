# Asset-Class Position Stops

Date: 2026-06-21

The Claude comparison showed a useful lesson: main `quanthack` was safer, but
its single global `1%` per-position stop was too tight for silver. `XAGUSD`
accounted for the full all-MACD return gap versus `quanthackclaude`.

## Change

`RiskLimits` now supports asset-class-specific per-position stop overrides:

```toml
max_position_loss_pct = 0.01
max_forex_position_loss_pct = 0.01
max_metal_position_loss_pct = 0.02
max_crypto_position_loss_pct = 0.025
```

Backtests call:

```python
risk_limits.max_position_loss_for_symbol(symbol)
```

So each symbol keeps a stop, but the stop can match the volatility of its asset
class. Unknown symbols fall back to the global value.

## Result

All-MACD, 10 FX/metals symbols:

| Profile | Return | Max DD | Official 15m Sharpe | Risk | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Global 1% stop | 5.262% | 1.694% | 0.032 | 100/100 | Safer but clipped `XAGUSD` |
| FX 1%, metals 2% | 5.896% | 1.694% | 0.035 | 100/100 | Recovered silver upside |
| FX 1%, metals 2%, MACD hist 2.5 | 6.001% | 0.855% | 0.038 | 100/100 | Current stronger paper profile |

The asset stop alone did **not** improve fixed-warmup robustness, but the
stricter MACD histogram filter did improve active-fold quality:

```text
Positive folds: 35.3%
Active-positive folds: 66.7%
Non-negative folds: 82.4%
Median active return: 0.641%
Worst test drawdown: 0.824%
Promotion: PAPER_ONLY
```

This is useful because it improves return, drawdown, Sharpe, and active-fold
quality. It does not solve the flat-fold problem because many folds have no
trades at all.

## Sizing Frontier With Asset Stops

```text
25% cap: return=2.489%, DD=0.490%, Sharpe15=0.036, worst lev=2.00x
40% cap: return=3.963%, DD=0.601%, Sharpe15=0.039, worst lev=3.17x
60% cap: return=5.179%, DD=0.739%, Sharpe15=0.039, worst lev=4.58x
80% cap: return=6.001%, DD=0.855%, Sharpe15=0.038, worst lev=5.81x
```

All points kept `100/100` risk discipline on this dataset.

## Interpretation

Use asset-class stops in main `quanthack`; do not switch back to an uncapped
position-risk model. The new profile captures the good part of the Claude result
while preserving a live-trading safety mechanism.

The next alpha task is not more leverage. It is improving positive fold
frequency through crypto coverage and a regime/diversification sleeve.
