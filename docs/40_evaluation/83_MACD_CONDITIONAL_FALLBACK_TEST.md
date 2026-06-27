# MACD Conditional Fallback Test

Date: 2026-06-22

## Purpose

Strict all-MACD is the current strongest paper candidate, but its weak point is
round coverage: many walk-forward folds are flat because no symbol clears the
MACD entry filters. This test added a small conditional-seasonality fallback
sleeve that can trade only when MACD is inactive.

Implementation:

- `MacdConditionalFallbackConfig`
- `MacdConditionalFallbackStrategy`
- Config section: `[strategy.macd_conditional_fallback]`

The fallback wraps:

- primary sleeve: `MacdMomentumStrategy`
- fallback sleeve: `ConditionalSeasonalityStrategy`

MACD pass-through decisions remain unchanged. Conditional fallback trades are
scaled by `conditional_notional_multiplier`, currently `0.25`.

## Safety Gate

The first loose version allowed conditional trades whenever MACD had no trade
intent. That was too broad: it allowed trades when MACD was outside its allowed
session or internally conflicted.

The current version adds `macd_inactive_reason_keywords = ["below"]`, so the
fallback can only trade when MACD is weak because a threshold is below the
required level. It does not trade just because the time is outside MACD hours,
the histogram is inside the exit band, or MACD line and histogram disagree.

## Full-Period Backtest

Dataset:

- `data/full_20gb_15m_prices.csv`
- `data/full_20gb_15m_quotes.csv`
- 10 symbols: FX majors/crosses plus `XAGUSD` and `XAUUSD`

| Candidate | Return | Max DD | Sharpe15 | Fills | Risk |
| --- | ---: | ---: | ---: | ---: | ---: |
| strict all-MACD | 6.001% | 0.855% | 0.038 | 84 | 100/100 |
| loose MACD conditional fallback | 4.998% | 0.824% | 0.036 | 116 | 100/100 |
| gated MACD conditional fallback | 6.000% | 0.855% | 0.038 | 102 | 100/100 |

The gated fallback is close to strict MACD, but it adds trades without improving
return or Sharpe.

## Walk-Forward Result

Fixed warmup settings:

- train size: `480`
- test size: `96`
- step size: `96`
- folds: `17`

| Candidate | Positive folds | Active folds | Active positive | Non-negative | Median active return | Worst DD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| strict all-MACD | 35.3% | 52.9% | 66.7% | 82.4% | 0.641% | 0.824% |
| gated MACD conditional fallback | 35.3% | 64.7% | 54.5% | 70.6% | 0.411% | 0.824% |

The fallback increases activity, but activity quality deteriorates. It does not
solve the round-coverage problem.

## Fill Diagnostics

The gated fallback generated only 6 conditional fills:

- `XAGUSD`: 2
- `USDCHF`: 2
- `EURUSD`: 1
- `AUDUSD`: 1

Those fills were too small and inconsistent to improve the fold distribution.
The loose version generated 13 fallback fills and was clearly worse.

## Decision

Status: **do not promote**.

Keep the strategy in code as an evaluated research tool, but do not use it as
the current competition candidate. Strict all-MACD remains the better benchmark.

Next research direction:

1. Add crypto data and test crypto trend/reversion coverage.
2. Build a portfolio-level complement that explicitly targets flat MACD folds.
3. Avoid single-symbol/tiny-position fallback sleeves that increase activity
   without improving fold-level returns.
