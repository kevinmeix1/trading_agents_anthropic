# Regime Tilt Results

The regime-tilt overlay uses the Kalman regime detector before allocation:

- trend-style signals are reduced in chop;
- reversion-style signals are boosted in chop;
- trend-style signals are boosted only when aligned with a detected trend;
- high-volatility regimes reduce exposure.

This is an allocation overlay, not a new signal. It is opt-in through
`portfolio-backtest --regime-tilt`.

## Static Candidate Check

Current static paper map:

| Mode | Return | Max DD | Official 15m Sharpe | Fills | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Baseline | 0.297% | 0.118% | 0.027 | 88 | current direct static check |
| Default regime tilt | 0.162% | 0.101% | 0.022 | 335 | too much resizing |
| Gentle regime tilt | 0.208% | 0.095% | 0.024 | 342 | still gives up too much return |

Conclusion: leave regime tilt off for the current static map. It reduces some
drawdown, but the return penalty is too large for a return-heavy competition.

## Alpha Router Check

The alpha router is the more natural place for trend/reversion regime tilting,
but the current router book is not profitable on this data:

| Mode | Return | Max DD | Official 15m Sharpe | Fills |
| --- | ---: | ---: | ---: | ---: |
| Alpha router baseline | -0.681% | 0.840% | -0.048 | 2276 |
| Alpha router + gentle regime tilt | -0.692% | 0.805% | -0.056 | 2776 |

Conclusion: regime tilt is useful infrastructure, but not a promotion candidate
until the underlying router weights/signals improve.

## Current Use

Keep this feature for experiments and live risk defense. Do not enable it in the
main paper profile yet.
