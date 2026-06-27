# Competition Scoring And Instruments

This step adds Syphonix-aware scoring and instrument metadata.

## Why This Matters

The competition does not rank only by raw P&L.

The official score is:

```text
70% Return Rank
15% Drawdown Rank
10% Sharpe Rank
5% Risk Discipline
```

The Sharpe ratio is also not the annualized research Sharpe used in many
backtests. Syphonix computes Sharpe from 15-minute account-equity returns:

```text
Sharpe = mean(15m returns) / std(15m returns)
```

No annualization multiplier is applied.

## New Modules

```text
src/quanthack/competition_score.py
src/quanthack/instruments.py
```

`competition_score.py` computes:

- official return
- official max drawdown
- non-annualized 15-minute Sharpe
- 15-minute return observation count
- Sharpe rank cap warning for fewer than 8 observations
- trade count versus the 30-trade Sharpe prize minimum
- risk discipline score
- risk discipline breaches and compliance-review flags

`instruments.py` defines all 15 tradable instruments:

```text
AUDUSD, EURCHF, EURGBP, EURUSD, GBPUSD, USDCAD, USDCHF, USDJPY
XAGUSD, XAUUSD
BARUSD, BTCUSD, ETHUSD, SOLUSD, XRPUSD
```

## Risk Discipline Monitor

The monitor checks the rule thresholds from the competition text:

- margin usage above 90%, 95%, and 98%
- leverage above 28x, 29x, and approaching 30x
- single-instrument concentration above 90%
- net directional exposure above 95%

Penalty rules reduce the score from 100. Review rules do not directly subtract
points, but they flag that compliance review would be required.

## Run In VS Code Terminal

Show the official instrument universe:

```bash
quanthack show-instruments
quanthack show-instruments --asset-class crypto
python scripts/inspect/show_instruments.py
```

Run a backtest and inspect the competition scoring view:

```bash
quanthack backtest --strategy simple_momentum
```

Run a portfolio backtest:

```bash
quanthack portfolio-backtest --strategy simple_momentum
```

The output now includes:

```text
Competition scoring view
  Official return
  Official max drawdown
  Official 15m Sharpe
  15m return observations
  Sharpe rank cap active
  Trades
  Risk discipline score
  Compliance review required
```

## Important Interpretation

The local code can compute raw metrics and rule breaches. It cannot know your
final percentile rank without the full leaderboard.

So locally we compute:

```text
your return, drawdown, Sharpe, risk discipline
```

The platform computes:

```text
your rank among all active participants
```

