# Strategy Comparison

Step 18 adds a side-by-side strategy comparison report.

Now that strategies are selectable, we need a quick answer to:

```text
Which configured strategy behaved better on the same data?
```

The comparison runs each strategy through the same pipeline:

```text
historical bars + quotes
  -> selected strategy
  -> market quality
  -> risk engine
  -> simulated fills
  -> performance metrics
```

## Run In VS Code Terminal

```bash
python scripts/evaluation/compare_strategies.py
```

This writes:

```text
outputs/backtests/strategy_comparison.csv
```

Compare only one strategy if you want:

```bash
python scripts/evaluation/compare_strategies.py --strategy simple_momentum
python scripts/evaluation/compare_strategies.py --strategy ma_crossover
python scripts/evaluation/compare_strategies.py --strategy breakout
python scripts/evaluation/compare_strategies.py --strategy volatility_squeeze
python scripts/evaluation/compare_strategies.py --strategy mean_reversion
python scripts/evaluation/compare_strategies.py --strategy regime_switch
python scripts/evaluation/compare_strategies.py --strategy alpha_router
```

Or pass `--strategy` twice:

```bash
python scripts/evaluation/compare_strategies.py --strategy simple_momentum --strategy ma_crossover --strategy breakout --strategy volatility_squeeze --strategy mean_reversion --strategy regime_switch --strategy alpha_router
```

## How Ranking Works

Rows are ranked by:

1. Sharpe ratio.
2. Total return.
3. Lower max drawdown.

This is the same ranking shape used by the parameter sweep.

## Important Limitation

The sample data is still tiny and synthetic. A better strategy on this file is
not automatically a better hackathon strategy.

This tool is useful because it proves the comparison workflow:

- Same data.
- Same risk limits.
- Same fill model.
- Same metric definitions.
- Different strategy logic.

That is the habit we want before touching any live or broker-connected path.
