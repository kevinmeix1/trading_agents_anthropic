# Portfolio Allocator And Router Optimization

The allocator sits between strategy signals and execution:

```text
market data -> strategy/router intents -> portfolio allocator -> risk engine -> fills
```

Strategies now say what final position they want per symbol. The allocator then
adjusts those targets before the risk engine sees them.

## Allocation Rules

Default rules are conservative:

- Max gross leverage: `2.0x`.
- Max one-symbol budget: `25%` of the gross leverage budget.
- Max net directional exposure: `80%`.
- Max crypto budget: `40%` of the gross leverage budget.
- Max metals budget: `25%` of the gross leverage budget.
- Prefer at least `3` active symbols when possible.

This does not guarantee profit. It makes the portfolio less likely to drift into
official risk-discipline problems such as net directional exposure above `95%` or
single-instrument concentration above `90%`.

## Run Allocator-Aware Portfolio Backtest

```bash
quanthack portfolio-backtest \
  --strategy alpha_router \
  --price-csv data/syphonix_sample_prices.csv \
  --quote-csv data/syphonix_sample_quotes.csv
```

Outputs now include:

- `outputs/backtests/portfolio_equity_curve.csv`
- `outputs/backtests/portfolio_pnl_summary.csv`
- `outputs/backtests/portfolio_allocation_report.csv`

The allocation report shows requested exposure, adjusted exposure, trimming
reasons, largest-symbol concentration, net directional exposure, and an estimated
risk status for each timestamp.

## Optimize Router Weights

After allocation exists, router optimization becomes more honest because each
candidate is scored after allocation trimming and risk-discipline checks.

```bash
quanthack router-optimize \
  --price-csv data/syphonix_sample_prices.csv \
  --quote-csv data/syphonix_sample_quotes.csv
```

Custom candidates use:

```bash
quanthack router-optimize \
  --candidate 0.4,0.2,0.35,0.25 \
  --candidate 0.25,0.15,0.15,0.35,0.20,0.10 \
  --candidate 0.20,0.10,0.10,0.40,0.15,0.05,0.10 \
  --candidate 0,0,0,0,0,0,0,1
```

The four numbers are:

```text
momentum, moving-average crossover, breakout, mean-reversion
```

The six-number format adds the newer sleeves:

```text
momentum, moving-average crossover, breakout, mean-reversion, session-breakout, cross-rate
```

The seven-number format appends relative strength:

```text
momentum, moving-average crossover, breakout, mean-reversion, session-breakout, cross-rate, relative-strength
```

The eight-number format appends volatility squeeze:

```text
momentum, moving-average crossover, breakout, mean-reversion, session-breakout, cross-rate, relative-strength, volatility-squeeze
```

Use the longer formats when researching whether session-breakout, FX cross-rate
confirmation, relative strength, or volatility squeeze deserves nonzero router
weight. Keep any new sleeve small until walk-forward validation supports it.

The output CSV ranks candidates by a local proxy for the official formula:

```text
70% return rank + 15% drawdown rank + 10% Sharpe rank + 5% risk discipline
```

The real leaderboard ranks against other teams, so treat this as a local research
tool, not a promise of competition rank.
