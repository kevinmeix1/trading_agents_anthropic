# Portfolio Backtesting

The single-symbol backtest answers:

```text
How did one strategy perform on one symbol?
```

The portfolio backtest answers:

```text
What happens when the same strategy trades multiple symbols through one shared account?
```

That matters because the hackathon account has shared equity, leverage, drawdown,
and margin risk. A trade on one symbol can reduce the risk budget available for
another symbol.

## Run It

After activating the virtual environment:

```bash
quanthack portfolio-backtest
```

The grouped script path works too:

```bash
python scripts/evaluation/run_portfolio_backtest.py
```

Use a specific strategy:

```bash
quanthack portfolio-backtest --strategy alpha_router
```

Use explicit symbols:

```bash
quanthack portfolio-backtest --symbol EURUSD --symbol GBPUSD
```

If you do not pass `--symbol`, the command uses symbols that appear in both the
price CSV and quote CSV.

Use a hybrid per-symbol strategy map:

```bash
quanthack portfolio-backtest \
  --strategy champion_ensemble \
  --strategy-map AUDUSD=macd_momentum \
  --strategy-map USDCHF=macd_momentum \
  --symbol XAGUSD --symbol XAUUSD --symbol AUDUSD --symbol USDCHF \
  --price-csv data/full_20gb_15m_prices.csv \
  --quote-csv data/full_20gb_15m_quotes.csv
```

`--strategy` is the fallback. Each `--strategy-map SYMBOL=STRATEGY` overrides
one symbol. This is useful for research portfolios where one sleeve works better
on metals and another works better on selected FX pairs.

Score only after a warmup period:

```bash
quanthack portfolio-backtest \
  --strategy dual_squeeze \
  --price-csv data/full_20gb_15m_prices.csv \
  --quote-csv data/full_20gb_15m_quotes.csv \
  --metrics-start 2026-05-21T00:00:00+00:00
```

This still writes full-run CSV artifacts, but the printed competition view uses
only equity points and fills at or after `--metrics-start`.

Run fixed-symbol warmup walk-forward:

```bash
quanthack portfolio-fixed-warmup-walk-forward \
  --strategy dual_squeeze \
  --symbol XAGUSD --symbol XAUUSD --symbol USDCHF --symbol GBPUSD \
  --symbol AUDUSD --symbol EURCHF --symbol USDJPY --symbol USDCAD \
  --price-csv data/full_20gb_15m_prices.csv \
  --quote-csv data/full_20gb_15m_quotes.csv \
  --train-size 960 \
  --test-size 192 \
  --step-size 192
```

This command runs each fold through the train window first, then scores only the
forward test window. Use it when a strategy needs lookback history before it can
make a realistic decision.

The fixed-warmup command also accepts the same `--strategy-map` overrides. Do not
trust a hybrid full-sample result until the same map has survived fixed-warmup
validation.

## Outputs

By default, the command writes:

```text
outputs/backtests/portfolio_equity_curve.csv
outputs/backtests/portfolio_pnl_summary.csv
outputs/backtests/portfolio_allocation_report.csv
outputs/backtests/portfolio_fills.csv
```

The equity curve records shared account equity, cash, gross exposure, net
exposure, drawdown, and open positions at each timestamp.

The P&L summary keeps attribution by symbol and adds a final `PORTFOLIO` row.

The allocation report shows requested exposure, adjusted exposure, trimming
reasons, active symbols, and estimated risk status.

The fills report is the trade-level audit trail:

```text
timestamp, symbol, side, fill_price, trade_units, turnover_notional_usd,
requested_notional_usd, adjusted_notional_usd, risk_reason, primary_signal
```

Use it for entry-hour analysis, signal attribution, and checking whether a
strategy meets the hackathon trade-count constraints after warmup.

## Why This Is Different From `quanthack backtest`

`quanthack backtest` is still useful for quick strategy work. It is simpler and
focused on one symbol.

`quanthack portfolio-backtest` is better when checking account-level behavior:

- multiple symbols sharing one cash balance
- shared gross leverage and risk limits
- per-symbol P&L attribution
- portfolio-level equity curve and drawdown

Right now the sample backtest CSVs only contain `EURUSD`, so the command behaves
like a one-symbol portfolio. The engine and tests already support multiple
symbols; richer CSV or MT5-exported data can plug into the same command later.
