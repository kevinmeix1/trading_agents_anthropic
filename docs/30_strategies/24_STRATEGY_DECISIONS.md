# Strategy Decisions

Step 24 strengthens the strategy layer without removing the old interface.

Old scripts can still call:

```text
strategy.generate_request(prices) -> TradeRequest | None
```

Newer code can call:

```text
strategy.generate_decision(...)
```

The decision object says whether the strategy wants to:

```text
ENTER
HOLD
EXIT
REVERSE
NO_ACTION
```

This matters because `TradeRequest` can only represent a positive target
notional. It cannot directly say "go flat." `StrategyDecision` can.

## Momentum Changes

`SimpleMomentumStrategy` now calculates signal strength with log returns.

It also reports:

- First price.
- Last price.
- Cumulative log return.
- Move in basis points.
- Realized volatility.
- Normalized momentum score.
- Trend efficiency.

Trend efficiency is:

```text
abs(net log return) / sum(abs(each log return))
```

A smooth directional path gets a value near `1.0`. A noisy path with the same
endpoint gets a lower value.

Momentum entries now require:

- Enough absolute movement, or enough configured normalized momentum.
- Enough trend efficiency.
- Estimated edge above spread, slippage, and fee cost.

Existing positions use an exit threshold, so the strategy can hold through small
noise instead of flipping near the entry threshold.

## Mean-Reversion Changes

`MeanReversionStrategy` no longer includes the latest price in the benchmark
used to judge that same latest price.

It now calculates the baseline from prior observations, then compares the latest
price against that prior baseline.

It also reports:

- Baseline mean.
- Baseline standard deviation.
- Latest price.
- Residual.
- Z-score.
- Deviation in basis points.
- Trend strength.
- Trend efficiency.

Mean reversion now has:

- Entry z-score.
- Smaller exit z-score.
- Trend filter.
- Maximum holding period.
- Stop z-score.
- Cost filter.

This prevents a large z-score from blindly creating a trade during a strong
directional move.

## Position Sizing

Both strategies still default to fixed notional sizing.

They can also use volatility-aware sizing:

```text
target_notional =
    base_notional
    * target_volatility / max(realized_volatility, volatility_floor)
    * signal_confidence
```

Then the result is capped by `max_target_notional_usd` and ignored if it is below
`min_trade_notional_usd`.

## Backtest Behavior

The backtest engine now uses quote midpoints for signal prices when quotes are
available:

```text
signal price = (bid + ask) / 2
```

Fills still use executable bid or ask prices plus the `FillModel` slippage.

That means the strategy reads a fair mid price, but the simulated trade pays
realistic spread and slippage.

## Run In VS Code Terminal

```bash
python scripts/evaluation/strategy_demo.py --strategy simple_momentum --scenario up
python scripts/evaluation/strategy_demo.py --strategy ma_crossover --scenario up
python scripts/evaluation/strategy_demo.py --strategy mean_reversion --scenario spike_up
python scripts/evaluation/run_backtest.py --strategy simple_momentum
python scripts/evaluation/run_backtest.py --strategy ma_crossover
python scripts/evaluation/run_backtest.py --strategy mean_reversion
python -m unittest tests.test_strategy
```

## Compatibility

This is still dry-run and offline research code.

The old strategy names remain:

```text
simple_momentum
ma_crossover
mean_reversion
```

The old `generate_request(prices)` method remains available, so existing scripts
continue to run. The richer decision model is additive.
