# USD Pressure Router

`usd_pressure_router` is a portfolio-aware strategy wrapper.

It uses the existing `alpha_router` as the base signal, then filters entries and
reversals with a broad USD pressure score from other FX pairs.

## Why It Exists

FX symbols are linked by common currency exposure. A long `EURUSD`, long
`GBPUSD`, and short `USDJPY` are all broadly USD-weakness trades. If a single
pair shows a signal but the rest of the USD basket points the other way, that
trade is more likely to be noise.

The filter tries to reduce false single-symbol trades.

## Pressure Sign

The pressure score is measured in basis points.

```text
positive USD pressure = broad USD strength
negative USD pressure = broad USD weakness
```

Examples:

- `EURUSD` up contributes negative USD pressure.
- `GBPUSD` up contributes negative USD pressure.
- `AUDUSD` up contributes negative USD pressure.
- `USDJPY` up contributes positive USD pressure.
- `USDCAD` up contributes positive USD pressure.

## Decision Rules

For a target trade:

1. Run the base `alpha_router`.
2. Compute broad USD pressure from other available USD FX pairs.
3. Require enough component symbols with enough history.
4. Require enough components confirming the broad pressure direction.
5. Require the target symbol's recent realized volatility to clear a small floor.
6. Allow the trade only if its USD direction agrees with the basket.
7. Optionally exit an existing position if the basket flips against it.

The volatility floor applies only to new entries and reversals. Explicit exits
are still allowed, so a quiet market does not trap existing risk.

For `EURUSD`, a long trade needs USD weakness.

For `USDJPY`, a long trade needs USD strength.

## Config

```toml
[strategy.usd_pressure]
symbol = "EURUSD"
lookback = 8
pressure_threshold_bps = 2.0
component_threshold_bps = 0.5
min_target_volatility_bps = 0.0
min_component_symbols = 3
min_confirming_symbols = 2
exit_on_conflict = true
```

`min_target_volatility_bps` is default-off. The first controlled experiment
tested `1.0 bps`, but it did not change the trade set on the downloaded scan
data, so it was not promoted into the default strategy.

## Commands

Demo:

```bash
python scripts/evaluation/strategy_demo.py --strategy usd_pressure_router --scenario up
```

Portfolio comparison:

```bash
python scripts/evaluation/portfolio_compare.py \
  --strategy alpha_router \
  --strategy usd_pressure_router \
  --price-csv data/downloaded_scan_prices.csv \
  --quote-csv data/downloaded_scan_quotes.csv \
  --output outputs/backtests/usd_pressure_portfolio_comparison.csv
```

## Important Limitation

This strategy needs multi-symbol context. It is most meaningful in portfolio
backtests and live dry-run monitoring. In a single-symbol backtest, there is not
enough basket context, so it will usually block entries.
