# Session Breakout Strategy

`session_breakout` is the first research-backed alpha added after the larger
downloaded-data scan.

## Purpose

Raw breakouts are easy to overtrade. This version keeps the useful part of the
Donchian breakout idea, then only allows new entries when the setup also passes:

- an allowed UTC hour filter
- an asset-specific session override for metals
- a minimum and maximum realized-volatility filter
- a minimum expected-edge filter before costs
- an optional time-series regime confirmation gate
- the existing spread, slippage, and fee cost filter
- the existing risk engine and portfolio allocator

The default active hours are:

```text
12:00, 13:00, 14:00, 15:00 UTC
```

Those hours came from the local imported-data scan, where average absolute
returns were higher than quieter sessions.

## Trading Logic

For each symbol:

1. Build a rolling price channel from the previous prices in the lookback window.
2. Treat the latest price as a breakout only when it clears the channel by
   `breakout_buffer_bps`.
3. Block new entries outside `allowed_utc_hours`.
4. For metals such as XAUUSD/XAGUSD, use `metal_allowed_utc_hours` when present.
5. Block quiet ranges below `min_realized_volatility_bps`.
6. Block extreme noise above `max_realized_volatility_bps`.
7. Block breakouts below `min_expected_edge_bps`.
8. If `require_regime_confirmation = true`, only enter long in `TREND_UP` and
   only enter short in `TREND_DOWN`.
9. Size the target notional with volatility sizing by default.
10. Allow exits whenever the breakout fades back inside the channel, unless
    `min_holding_period` says the position is too new to churn out.

That last point matters: the session filter blocks new entries outside the active
window, but it does not stop the strategy from closing a stale position.

## Config

```toml
[strategy.session_breakout]
symbol = "EURUSD"
lookback = 8
breakout_buffer_bps = 2.0
exit_buffer_bps = 1.0
min_channel_width_bps = 2.0
min_expected_edge_bps = 4.0
min_holding_period = 2
min_realized_volatility_bps = 1.5
max_realized_volatility_bps = 80.0
allowed_utc_hours = [12, 13, 14, 15]
metal_allowed_utc_hours = [12, 13, 14, 15, 16, 17]
require_regime_confirmation = false
regime_lookback = 80
regime_min_abs_slope_bps = 0.75
regime_min_trend_efficiency = 0.25
regime_max_realized_volatility_bps = 120.0
target_notional_usd = 50000.0
position_sizing = "volatility"
max_target_notional_usd = 75000.0
```

For research runs, flip `require_regime_confirmation` to `true` when you want a
stricter version that only trades breakouts aligned with the Kalman-style regime
classifier. Keep it off for baseline comparisons so you can measure the effect.

## Commands

Inspect one decision:

```bash
python scripts/evaluation/strategy_demo.py --strategy session_breakout --scenario spike_up
```

Compare against existing strategies on a single symbol:

```bash
python scripts/evaluation/compare_strategies.py \
  --strategy simple_momentum \
  --strategy ma_crossover \
  --strategy breakout \
  --strategy session_breakout \
  --strategy mean_reversion \
  --price-csv data/downloaded_scan_prices.csv \
  --quote-csv data/downloaded_scan_quotes.csv \
  --symbol XAUUSD \
  --output outputs/backtests/session_breakout_single_symbol_comparison.csv
```

Compare as a diversified portfolio:

```bash
python scripts/evaluation/portfolio_compare.py \
  --strategy breakout \
  --strategy session_breakout \
  --strategy alpha_router \
  --price-csv data/downloaded_scan_prices.csv \
  --quote-csv data/downloaded_scan_quotes.csv \
  --output outputs/backtests/session_breakout_portfolio_comparison.csv
```

## Interpretation

Use this strategy if it improves drawdown-adjusted results or risk discipline
versus raw `breakout`. If it makes fewer trades but protects Sharpe and drawdown,
that can still be useful because the competition score rewards risk-adjusted
performance, not only raw return.
