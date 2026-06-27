# Crypto Overlay Sizing Comparison

The previous overlay test found that BTC+SOL trend exposure improved the mixed
portfolio, but the candidate remained paper-only because most positive
walk-forward return came from one fold.

This pass tests whether reducing crypto notional can keep the return edge while
improving drawdown and fold concentration.

Run:

```bash
PYTHONPATH=src .venv/bin/python -c 'from quanthack.cli.crypto_overlay_sizing_compare import main; main()' \
  --config configs/competition.toml \
  --price-csv data/mixed_official_crypto_proxy_overlap_prices.csv \
  --quote-csv data/mixed_official_crypto_proxy_overlap_quotes.csv \
  --train-size 96 \
  --test-size 48 \
  --step-size 48 \
  --output outputs/research/crypto_overlay_sizing_comparison.csv
```

Then gate the result:

```bash
PYTHONPATH=src .venv/bin/python -c 'from quanthack.cli.research_candidate_gate import main; main()' \
  --source path=outputs/research/crypto_overlay_sizing_comparison.csv,data_source=mixed_proxy \
  --output outputs/research/research_candidate_gate_crypto_overlay_sizing.csv
```

## Result

| Rank | Candidate | Return | Max DD | Sharpe15 | Fold concentration | Gate |
|---:|---|---:|---:|---:|---:|---|
| 1 | `btc075_sol100_reversion075` | +1.524% | 1.445% | 0.036 | 90.1% | PAPER_ONLY |
| 2 | `trend075_reversion050` | +1.481% | 1.330% | 0.039 | 92.3% | PAPER_ONLY |
| 3 | `crypto_100` | +1.377% | 1.531% | 0.031 | 91.1% | PAPER_ONLY |
| 4 | `btc050_sol100_reversion075` | +1.306% | 1.363% | 0.033 | 89.2% | PAPER_ONLY |
| 5 | `crypto_075` | +1.287% | 1.330% | 0.033 | 92.6% | PAPER_ONLY |
| 6 | `crypto_050` | +1.243% | 1.128% | 0.036 | 94.6% | PAPER_ONLY |
| 7 | `crypto_035` | +1.139% | 1.008% | 0.038 | 95.8% | PAPER_ONLY |

Best paper sizing map:

```text
BARUSD=0.750
BTCUSD=0.750
ETHUSD=0.750
SOLUSD=1.000
XRPUSD=0.750
```

Strategy map:

```text
BARUSD=crypto_mean_reversion
BTCUSD=macd_momentum
ETHUSD=crypto_mean_reversion
SOLUSD=macd_momentum
XRPUSD=crypto_mean_reversion
```

## Interpretation

The best trade-off is not simply shrinking all crypto. Keeping SOL at full size
while trimming BTC and the reversion crypto sleeve to 75% improved return,
drawdown, and Sharpe versus the unsized BTC+SOL overlay:

- return improved from +1.377% to +1.524%
- drawdown improved from 1.531% to 1.445%
- Sharpe15 improved from 0.031 to 0.036
- fold concentration improved slightly from 91.1% to 90.1%

The concentration problem did not disappear. This means the remaining weakness is
mostly temporal: the current short overlap sample has one fold doing most of the
work. Sizing can improve the trade-off, but official MT5 crypto data is still
needed before promotion.

## Decision

Replace the plain aggressive BTC+SOL overlay with the sized BTC+SOL overlay as
the leading crypto research candidate.

It remains `PAPER_ONLY`, not live-ready, because:

- the data source is mixed official/proxy
- the largest positive fold still contributes about 90% of positive fold return

Next:

1. Test this sized overlay on official MT5 crypto captures.
2. If it remains ahead, add it to the main candidate scorecard.
3. Consider a temporal/session filter for crypto only if official data also
   shows the same fold-concentration pattern.

Follow-up: `99_CRYPTO_SESSION_OVERLAY_COMPARISON.md` added crypto-only session
filters. London-hours improved return and Sharpe on the mixed proxy overlap, but
also worsened fold concentration, so it is still paper-only research evidence.
