# Crypto Session Overlay Comparison

This pass extends the sized BTC+SOL overlay by testing crypto-only entry session
filters. The portfolio session gate only blocks new exposure outside the
specified UTC hours; it still allows reductions, exits, and risk controls.

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

Gate:

```bash
PYTHONPATH=src .venv/bin/python -c 'from quanthack.cli.research_candidate_gate import main; main()' \
  --source path=outputs/research/crypto_overlay_sizing_comparison.csv,data_source=mixed_proxy \
  --output outputs/research/research_candidate_gate_crypto_overlay_sizing.csv
```

## Result

The session candidates use the same strategy and sizing map:

```text
BARUSD=crypto_mean_reversion at 0.75x
BTCUSD=macd_momentum at 0.75x
ETHUSD=crypto_mean_reversion at 0.75x
SOLUSD=macd_momentum at 1.00x
XRPUSD=crypto_mean_reversion at 0.75x
```

Ranked output:

| Rank | Candidate | Crypto entry hours UTC | Return | Max DD | Sharpe15 | Fold concentration | Gate |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | `btc075_sol100_reversion075_london` | 7-16 | +1.852% | 1.446% | 0.044 | 93.9% | PAPER_ONLY |
| 2 | `btc075_sol100_reversion075_london_us` | 8-21 | +1.757% | 1.446% | 0.042 | 93.9% | PAPER_ONLY |
| 3 | `trend075_reversion050` | all | +1.481% | 1.330% | 0.039 | 92.3% | PAPER_ONLY |
| 4 | `btc075_sol100_reversion075` | all | +1.524% | 1.445% | 0.036 | 90.1% | PAPER_ONLY |
| 5 | `btc075_sol100_reversion075_us` | 13-21 | +1.746% | 1.444% | 0.042 | 100.0% | PAPER_ONLY |
| 10 | `btc075_sol100_reversion075_asia` | 0-8 | +0.922% | 0.884% | 0.039 | 82.5% | PAPER_ONLY |

## Interpretation

The London-hour crypto filter is the highest-return paper candidate so far on
the mixed overlap sample. It improves the no-session sized overlay:

- return: +1.524% to +1.852%
- Sharpe15: 0.036 to 0.044
- drawdown: effectively unchanged at about 1.45%
- trade count: 84 down to 53

It does not solve the main robustness problem. Fold concentration worsens from
90.1% to 93.9%, meaning one positive fold still contributes most of the positive
walk-forward return.

The Asia filter is interesting as a defensive reference: lower return, lower
drawdown, and lower concentration, but only 26 trades, which is below the 30
trade threshold that matters for the Sharpe prize.

## Decision

Use `btc075_sol100_reversion075_london` as the **highest-return crypto research
candidate**, not as a live candidate.

Keep the no-session `btc075_sol100_reversion075` as the more conservative
comparison because it has lower fold concentration and more trades.

Both remain `PAPER_ONLY` until official MT5 crypto data confirms the behavior.

Next:

1. Re-run this exact comparison on official MT5 crypto captures.
2. If London hours still dominate, add a fold-level trade attribution report for
   the winning fold to understand whether the signal is BTC, SOL, or reversion.
3. Do not promote the London filter unless it reduces concentration on a longer
   official crypto window.

Follow-up: `100_CRYPTO_OVERLAY_FOLD_DIAGNOSTIC.md` shows the strongest fold was
driven by BTC/SOL plus metals, while crypto reversion names lost in that fold.
A trend-only crypto follow-up was worse overall, so the reversion sleeve remains
useful as coverage rather than as the main return driver.
