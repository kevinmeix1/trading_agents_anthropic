# Multi-Symbol Data And Portfolio Compare

This layer is for practicing the competition workflow before MT5 data is connected.
It creates deterministic CSV data for the official 15 instruments, then compares
strategies as shared-risk portfolios.

## Generate Practice Data

```bash
quanthack generate-sample-data
```

Default outputs:

- `data/syphonix_sample_prices.csv`
- `data/syphonix_sample_quotes.csv`

Useful smaller examples:

```bash
quanthack generate-sample-data --asset-class crypto --periods 96
quanthack generate-sample-data --symbol EURUSD --symbol BTCUSD --periods 64
```

The generator is not market truth. It is a controlled sandbox for checking whether
the backtest, risk monitor, and router can handle FX, metals, and crypto together.
Later, MT5/live data should replace this CSV source.

## Compare Portfolio Strategies

```bash
quanthack portfolio-compare \
  --price-csv data/syphonix_sample_prices.csv \
  --quote-csv data/syphonix_sample_quotes.csv
```

The comparison runs each strategy across all symbols found in both CSV files. It
then computes a competition-style proxy score:

- Return rank among the candidates.
- Drawdown rank among the candidates.
- Official non-annualized 15-minute Sharpe rank among the candidates.
- Actual risk discipline score from the risk monitor.

This is not the real leaderboard score because the real formula ranks against
other teams. It is still useful because it uses the same scoring shape:

```text
70% return rank + 15% drawdown rank + 10% Sharpe rank + 5% risk discipline
```

## Why This Matters

A strategy can look good on one EURUSD backtest and fail once BTCUSD, XAUUSD, and
other FX pairs are included. The portfolio comparison catches three common issues:

- One strategy only works on one symbol.
- A strategy trades too little to get useful Sharpe observations.
- A strategy makes money but drifts toward risk-discipline penalties.

Good next experiments:

```bash
quanthack portfolio-compare --strategy alpha_router --strategy ma_crossover
quanthack portfolio-universe-scan --strategy alpha_router --strategy ma_crossover
quanthack portfolio-backtest --strategy alpha_router \
  --price-csv data/syphonix_sample_prices.csv \
  --quote-csv data/syphonix_sample_quotes.csv
```
