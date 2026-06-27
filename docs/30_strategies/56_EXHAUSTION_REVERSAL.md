# Exhaustion Reversal Strategy

`exhaustion_reversal` looks for a sharp short-window shock, then waits for the
first reversal bar before entering against the shock.

It was added because intraday research often finds short-horizon reversal or
continuation effects around large moves. The implementation is deliberately
gated:

```text
shock size
shock z-score versus baseline volatility
shock path efficiency
reversal bar size
session/spread/cost filter
short max holding period
```

Current research result:

```text
default full-sample portfolio:
  return: -0.269%
  max drawdown: 0.372%
  trades: 231
  risk discipline: 100/100
```

A stricter 72-candidate grid was also negative:

```text
best stricter variant:
  return: -0.052%
  trades: 10
```

Verdict:

```text
keep as a tested research sleeve.
do not use for live MT5 trading unless future data shows a reversal edge.
```

