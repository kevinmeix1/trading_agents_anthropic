# Signal Diagnostics

Full portfolio backtests are expensive because they simulate strategy, allocation,
risk, fills, equity, and competition scoring.

`signal-diagnostics` is a cheaper first pass. It asks:

```text
When each alpha-router sleeve fires, what happens over the next N bars?
```

It does not replace a real backtest. It helps decide which ideas deserve one.

## What It Measures

For each symbol and signal sleeve:

- how often the signal fired
- long/short counts
- hit rate
- average signed forward return in basis points
- average confidence
- average adjusted router weight
- average edge after estimated cost

The most important field is `average_signed_forward_return_bps`.

- Positive means the signal direction tended to be right over the chosen horizon.
- Negative means the signal direction tended to be wrong.
- Near zero means the signal is probably noise at that horizon.

## Run It

```bash
quanthack signal-diagnostics \
  --strategy alpha_router \
  --price-csv data/full_20gb_15m_prices.csv \
  --quote-csv data/full_20gb_15m_quotes.csv \
  --symbol EURUSD \
  --symbol GBPUSD \
  --symbol USDJPY \
  --horizon-bars 1 \
  --output outputs/backtests/signal_diagnostics_core_fx.csv
```

Use a longer horizon when testing slower ideas:

```bash
quanthack signal-diagnostics \
  --strategy alpha_router \
  --price-csv data/full_20gb_15m_prices.csv \
  --quote-csv data/full_20gb_15m_quotes.csv \
  --symbol EURUSD \
  --symbol XAUUSD \
  --horizon-bars 4
```

## How To Use It

Use this order:

```text
signal diagnostics -> small focused backtest -> portfolio walk-forward -> MT5 dry run
```

Do not promote a signal just because diagnostics look good. A signal can have
positive next-bar return and still fail once spread, position sizing, allocation,
and risk gates are included. But if diagnostics are poor, a full backtest is
usually not worth running yet.

## Current Refinement

The alpha router now includes `session_breakout` as a soft sleeve. This lets the
router see whether breakouts are occurring during the better session/volatility
windows without letting the session breakout strategy dominate alone.

The router also has adaptive weighting. Signal diagnostics uses the same
`generate_signals(...)` path as backtests, so the CSV reflects adjusted weights
for asset class and regime. This makes it a good first screen before spending
time on full portfolio simulations.
