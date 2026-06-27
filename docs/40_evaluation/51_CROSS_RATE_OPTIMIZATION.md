# Cross-Rate Optimization

This tool ranks FX cross-rate reversion candidates before running slow portfolio
backtests.

It answers:

```text
Which FX symbols and thresholds have positive forward-return diagnostics?
```

## Why It Exists

Standalone cross-rate reversion was very selective under realistic spread and
slippage assumptions. That is not automatically bad. It means the idea is better
handled as a researched router sleeve:

1. scan symbols and thresholds quickly
2. keep only candidates with enough active signals
3. promote the best candidates into portfolio router tests

## Command

From the project folder:

```bash
python -c 'from quanthack.cli.cross_rate_optimize import main; main([
  "--price-csv", "data/full_20gb_15m_prices.csv",
  "--quote-csv", "data/full_20gb_15m_quotes.csv",
  "--symbol", "EURGBP",
  "--symbol", "EURCHF",
  "--symbol", "EURUSD",
  "--symbol", "GBPUSD",
  "--symbol", "USDCHF",
  "--horizon-bars", "4",
  "--output", "outputs/backtests/full_20gb_cross_rate_parameter_optimization_h4.csv",
])'
```

If the package is installed in editable mode, the equivalent console script is:

```bash
quanthack-cross-rate-optimize \
  --price-csv data/full_20gb_15m_prices.csv \
  --quote-csv data/full_20gb_15m_quotes.csv \
  --symbol EURGBP \
  --symbol EURCHF \
  --symbol EURUSD \
  --symbol GBPUSD \
  --symbol USDCHF \
  --horizon-bars 4 \
  --output outputs/backtests/full_20gb_cross_rate_parameter_optimization_h4.csv
```

## Output Columns

Important fields:

- `eligible`: passed minimum activity, hit-rate, and signed-return gates
- `quality_score`: ranking score combining signed return, hit-rate, sample count,
  and edge after cost
- `active_count`: number of tradeable cross-rate signals
- `hit_rate`: share of signals whose forward return had the right sign
- `average_signed_forward_return_bps`: average forward return after applying the
  signal direction
- `average_edge_after_cost_bps`: signal edge minus estimated cost

## How To Interpret

Prefer candidates with:

- enough active signals, not just one lucky event
- positive average signed forward return
- hit rate above 50%
- positive edge after cost
- stable behavior across neighboring parameter sets

Do not promote every top row blindly. If `GBPUSD` is negative across most rows
but one permissive row looks good, treat that as suspicious. If `EURGBP` or
`EURCHF` is positive across several nearby rows, that is much more interesting.

## Next Step

After this screen, take the best symbols and parameter settings into:

```bash
quanthack-signal-diagnostics --strategy alpha_router
quanthack-portfolio-backtest --strategy alpha_router
quanthack-portfolio-router-walk-forward
```

You can also promote the symbol list into the cross-rate sleeve allowlist:

```toml
[strategy.cross_rate_reversion]
allowed_symbols = ["EURGBP", "EURUSD", "EURCHF"]
```

The goal is not to maximize the optimizer CSV. The goal is to find robust
router inputs that survive portfolio allocation, spread/slippage, drawdown
control, and competition risk discipline.
