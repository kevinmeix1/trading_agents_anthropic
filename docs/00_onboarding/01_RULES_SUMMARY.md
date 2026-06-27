# Rules Summary

Based on the hackathon document you provided.

## What Matters Most

- Starting capital: 1,000,000 virtual dollars.
- Leverage: up to 1:30, but we should use much less at first.
- Instruments: 8 FX pairs, 2 metals, and 5 crypto assets.
- Timezone: London time.
- Equity ranking carries over. There is no daily reset.
- Daily cuts happen around official checkpoints.
- Margin-call style elimination is immediate, so risk controls must trigger much earlier.
- Final score is 70% return rank, 15% drawdown rank, 10% Sharpe rank, and
  5% risk discipline.
- Sharpe is non-annualized and computed from 15-minute account-equity returns.
- Best Sharpe prize eligibility requires reaching finals, finishing top 50
  overall, no red-line violations, and at least 30 trades.

## Practical Meaning

Do not build a bot that simply maximizes aggression.

Build a tournament-aware system:

1. Clock knows whether we are before live, qualifying, near checkpoint, or in finals.
2. Strategy proposes trades.
3. Risk engine approves, reduces, or blocks trades.
4. Execution adapter records every decision.
5. Official scoring metrics and logs prove what happened.

## Tradable Assets

```text
Forex: AUD/USD, EUR/CHF, EUR/GBP, EUR/USD, GBP/USD, USD/CAD, USD/CHF, USD/JPY
Metals: XAG/USD, XAU/USD
Crypto: BAR/USD, BTC/USD, ETH/USD, SOL/USD, XRP/USD
```

## Official Score Formula

```text
Final Score =
  70% * Return Rank
  15% * Drawdown Rank
  10% * Sharpe Rank
   5% * Risk Discipline
```

Local backtests can compute raw return, drawdown, Sharpe, and risk discipline.
Only the platform can compute final rank percentiles across all participants.

## Schedule To Verify

The notes mention a possible discrepancy:

- Round 1 cutoff: June 22, 2026 at 22:00 BST.
- Round 2 cutoff: June 23, 2026 at 22:00 BST.
- Round 3 cutoff: June 24, 2026 at 22:00 BST.
- Finals: June 24, 2026 at 22:00 BST through June 26, 2026 at 22:00 BST.
- Results announcement: June 27, 2026.

## Starting Risk Defaults

Use these only as initial training wheels:

- Max gross leverage: 2x.
- Max single-symbol notional: 25% of equity.
- Max daily loss: 2.5%.
- Max drawdown throttle: 6%.
- Checkpoint protection: reduce risk 90 minutes before cuts.

These are not strategy recommendations. They are safety defaults for building and
testing the system.
