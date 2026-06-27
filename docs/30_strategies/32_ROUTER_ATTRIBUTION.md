# Router Attribution

The alpha router now carries signal attribution through each backtest fill.

For each router decision, the strategy records:

- `primary_signal`: strongest signal in the chosen direction
- `supporting_signals`: active signals agreeing with the chosen direction
- `conflicting_signals`: active signals pointing the other way

This is useful because a combined router can otherwise become hard to explain.
The attribution report answers:

```text
Which signal was leading the fills, and how did those fills perform?
```

## Run It

```bash
quanthack router-report
```

Grouped script path:

```bash
python scripts/evaluation/router_report.py
```

Output CSV:

```text
outputs/backtests/router_attribution.csv
```

## How To Read It

The report groups fills by `primary_signal` and shows:

- fill count
- realized P&L
- win rate on realized events
- turnover
- average adjusted notional
- conflict fill count

A high conflict count means the router traded even though another active signal
disagreed. That is not automatically bad, but it is exactly the kind of thing we
want visible before trusting the strategy.
