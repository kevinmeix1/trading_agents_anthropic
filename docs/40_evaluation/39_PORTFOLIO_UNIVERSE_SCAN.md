# Portfolio Universe Scan

The single-symbol backtest answers:

```text
Does this strategy work on one market?
```

The portfolio universe scan answers:

```text
Which diversified symbol basket is worth testing more seriously?
```

That matters for this hackathon because the scoring is not only about return. It
also cares about drawdown, Sharpe, and risk discipline. A basket that spreads risk
across FX, metals, and crypto is usually a better starting point than only trading
one symbol.

## Run A Scan

After importing downloaded data into normalized CSVs:

```bash
quanthack portfolio-universe-scan \
  --price-csv data/downloaded_backtest_prices.csv \
  --quote-csv data/downloaded_backtest_quotes.csv
```

Default output:

```text
outputs/backtests/portfolio_universe_scan.csv
```

By default, the scan evaluates `alpha_router` across automatically selected
diversified baskets. It only uses symbols that exist in both the price and quote
CSV files and are in the official instrument metadata.

## Compare Several Strategies

```bash
quanthack portfolio-universe-scan \
  --strategy alpha_router \
  --strategy ma_crossover \
  --strategy simple_momentum \
  --price-csv data/downloaded_backtest_prices.csv \
  --quote-csv data/downloaded_backtest_quotes.csv
```

Each row is one basket plus one strategy. The command ranks rows using the same
proxy scoring shape used elsewhere in the project:

```text
70% return rank + 15% drawdown rank + 10% 15-minute Sharpe rank + 5% risk discipline
```

## Custom Basket

Use a custom basket when you want to test a specific idea:

```bash
quanthack portfolio-universe-scan \
  --strategy alpha_router \
  --basket fx_gold:EURUSD,USDJPY,XAUUSD \
  --price-csv data/downloaded_portfolio_prices.csv \
  --quote-csv data/downloaded_portfolio_quotes.csv
```

The basket name before `:` is just a label for the report.

## How To Read The CSV

Important columns:

- `proxy_score`: project leaderboard proxy for comparing candidates.
- `symbols`: symbols in the basket.
- `asset_mix`: how many FX, metal, and crypto symbols are included.
- `official_return_pct`: return over the imported backtest window.
- `official_max_drawdown_pct`: account-level drawdown.
- `official_15m_sharpe`: non-annualized 15-minute Sharpe shape.
- `risk_discipline_score`: account-level risk discipline estimate.
- `trimmed_allocation_periods`: how often the allocator reduced strategy targets.
- `worst_largest_symbol_concentration`: highest basket concentration seen.

## Next Step

After choosing a promising row, run the detailed portfolio backtest for that
strategy and symbol list:

```bash
quanthack portfolio-backtest \
  --strategy alpha_router \
  --symbol EURUSD --symbol USDJPY --symbol XAUUSD \
  --price-csv data/downloaded_portfolio_prices.csv \
  --quote-csv data/downloaded_portfolio_quotes.csv
```

Use the universe scan for discovery, then use `portfolio-backtest` for detailed
equity curve, P&L, and allocation reports.

Before treating a basket as robust, run portfolio walk-forward validation:

```bash
quanthack portfolio-walk-forward \
  --strategy alpha_router \
  --strategy ma_crossover \
  --price-csv data/downloaded_portfolio_prices.csv \
  --quote-csv data/downloaded_portfolio_quotes.csv
```
