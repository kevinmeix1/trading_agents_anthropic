# Backtesting

Step 15 adds the first offline backtest engine.

This is still not live trading. It replays local CSV data through the same core
layers we already built:

```text
price bars + quotes
  -> position stop-loss check
  -> market quality
  -> strategy
  -> risk engine
  -> simulated fill
  -> update position cost basis
  -> evolving cash, position, equity
  -> performance metrics
```

## Why This Matters

Until now, we could answer:

```text
Can the strategy generate a request?
Can risk approve, shrink, or block it?
Can the journal record it?
```

Backtesting starts answering the bigger question:

```text
Would this strategy have made money on historical data?
```

This is one of the biggest gaps for the hackathon because the competition has
equity ranking and a Sharpe-style award.

## New Files

- `src/quanthack/backtest.py`
- `src/quanthack/metrics.py`
- `scripts/evaluation/run_backtest.py`
- `data/backtest_prices.csv`
- `data/backtest_quotes.csv`
- `tests/test_backtest.py`
- `tests/test_metrics.py`

## What The Backtest Engine Does

For each bar:

1. Load the matching quote.
2. Mark current position to quote mid.
3. Build an evolving `AccountSnapshot`.
4. Check whether an open position has breached its entry-notional stop-loss.
5. Check market quality.
6. Let the strategy propose a request.
7. Let risk approve, shrink, or block the request.
8. If approved, simulate a fill using bid/ask plus slippage.
9. Update cash, position units, average entry, and equity.
10. Save an equity-curve point.

## Fill Model

The fill model is intentionally simple:

```text
BUY fills at ask plus slippage
SELL fills at bid minus slippage
```

Configured default:

```toml
[backtest]
slippage_bps = 1.0
```

This is not a perfect broker simulation. It is a first conservative approximation.

## Metrics

The first metrics are:

- Final equity.
- Total return.
- Sharpe ratio.
- Max drawdown.
- Win rate.
- Win rate including flat bars.
- Profit factor.
- Turnover.
- Trade-level realized/open P&L attribution.

Important: the headline win rate and profit factor are still based on bar-to-bar
equity changes. The P&L ledger separately explains realized and open P&L from
fills.

Profit factor is capped at `999.0` when there are gains but no losses. That
avoids `inf` values in CSV reports and optimizer rankings.

## Run In VS Code Terminal

Make sure your prompt shows `(.venv)`.

```bash
python scripts/evaluation/run_backtest.py
```

This writes:

```text
outputs/backtests/equity_curve.csv
outputs/backtests/pnl_ledger.csv
```

Run tests:

```bash
python -m unittest discover -s tests
```

## What To Look For

Do not over-trust results from the sample CSV. The data is tiny and synthetic.

The purpose of this step is the engine:

- Bar-by-bar replay works.
- Risk sees evolving equity.
- Positions and cash update.
- Fills include bid/ask and slippage.
- Metrics are computed.

Once this works, we can add larger historical datasets, parameter sweeps, and
walk-forward evaluation.

## Downloaded Competition Data

The large downloaded archive in `~/Downloads` contains tick-level Parquet files.
Convert it before backtesting:

```bash
python -m pip install -e ".[data]"
quanthack import-backtest-data --symbol EURUSD --max-files-per-symbol 2
quanthack backtest \
  --strategy simple_momentum \
  --symbol EURUSD \
  --price-csv data/downloaded_backtest_prices.csv \
  --quote-csv data/downloaded_backtest_quotes.csv
```

Read `docs/40_evaluation/38_DOWNLOADED_BACKTEST_DATA.md` for the full workflow.
