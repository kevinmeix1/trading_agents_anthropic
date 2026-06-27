# Trend Pullback Continuation

`trend_pullback` is a selective continuation strategy for FX, metals, and crypto.

It looks for:

```text
larger trend -> controlled pullback -> resume move with the trend
```

The goal is to avoid raw momentum chasing while still participating when a
trend pauses briefly and then restarts.

## Entry Logic

The strategy splits the recent window into three parts:

- `trend_move_bps`: move from the start of the window to the pre-pullback anchor.
- `pullback_bps`: opposite move from the anchor to the prior bar.
- `resume_bps`: latest move back in the trend direction.

Long example:

```text
trend up enough + small pullback down + latest resume up = possible long
```

Short example:

```text
trend down enough + small pullback up + latest resume down = possible short
```

It also requires trend efficiency, expected edge after estimated costs, spread
guardrails, and liquid entry hours.

## Config

In `configs/default.toml`:

```toml
[strategy.trend_pullback]
lookback = 32
pullback_window = 4
min_trend_bps = 8.0
min_trend_efficiency = 0.35
min_pullback_bps = 1.0
max_pullback_bps = 12.0
min_resume_bps = 1.0
min_expected_edge_bps = 3.0
forex_allowed_utc_hours = [11, 12, 13, 14, 15, 16, 17, 18, 19]
metal_allowed_utc_hours = [11, 12, 13, 14, 15, 16, 17, 18, 19]
position_sizing = "volatility"
max_holding_period = 24
```

## Run It

Demo a hand-built setup:

```bash
python scripts/evaluation/strategy_demo.py \
  --strategy trend_pullback \
  --lookback 8 \
  --prices 1.0000,1.0020,1.0040,1.0060,1.0080,1.0100,1.0085,1.0095
```

Compare it against other strategies:

```bash
python scripts/evaluation/portfolio_compare.py \
  --price-csv data/full_20gb_15m_prices.csv \
  --quote-csv data/full_20gb_15m_quotes.csv \
  --strategy volatility_squeeze \
  --strategy trend_pullback \
  --strategy session_breakout \
  --strategy alpha_router
```

Optimize parameters:

```bash
python scripts/evaluation/trend_pullback_optimize.py \
  --price-csv data/full_20gb_15m_prices.csv \
  --quote-csv data/full_20gb_15m_quotes.csv \
  --walk-forward
```

## Current Research Status

The first full-data comparison ranked `trend_pullback` below
`volatility_squeeze`, but above the older high-churn strategies.

Optimizer notes:

- NY-hours variant had positive full-sample return: about `0.040%`, with low
  drawdown and 98 trades.
- The same NY-hours variant failed walk-forward because test windows had no
  selected fills.
- The faster variant was walk-forward-eligible on selected baskets, but had
  negative all-symbol full-sample return.

Verdict: keep `trend_pullback` in research. It may become a basket-selected
supporting sleeve, but it is not strong enough to replace `volatility_squeeze`
or move toward live MT5 execution yet.
