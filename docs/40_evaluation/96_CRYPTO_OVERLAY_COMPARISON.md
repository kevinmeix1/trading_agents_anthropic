# Crypto Overlay Comparison

This pass answers the portfolio question that the isolated crypto allocation
test could not answer by itself:

If the official FX/metals book is already running, does adding crypto improve the
full portfolio after the shared allocator, shared risk engine, and shared equity
curve are applied?

Run:

```bash
PYTHONPATH=src .venv/bin/python -c 'from quanthack.cli.crypto_overlay_compare import main; main()' \
  --config configs/competition.toml \
  --price-csv data/mixed_official_crypto_proxy_overlap_prices.csv \
  --quote-csv data/mixed_official_crypto_proxy_overlap_quotes.csv \
  --train-size 96 \
  --test-size 48 \
  --step-size 48 \
  --output outputs/research/crypto_overlay_comparison.csv
```

Output:

```text
outputs/research/crypto_overlay_comparison.csv
```

## Candidates

The comparison tests five maps:

- `official_only_base`: official FX/metals symbols only, using the base strategy.
- `all_symbols_base`: official FX/metals plus crypto, all using the base strategy.
- `crypto_all_reversion_overlay`: official symbols use the base strategy; all crypto symbols use `crypto_mean_reversion`.
- `crypto_robust_sol_overlay`: official symbols use the base strategy; SOL uses `macd_momentum`; other crypto symbols use `crypto_mean_reversion`.
- `crypto_aggressive_btc_sol_overlay`: official symbols use the base strategy; BTC and SOL use `macd_momentum`; other crypto symbols use `crypto_mean_reversion`.

The default base strategy is `macd_momentum`.

## Mixed Official/Crypto Overlap Result

Ranked output:

| Rank | Candidate | Return | Max DD | Sharpe15 | Risk | Fixed-warmup |
|---:|---|---:|---:|---:|---:|---|
| 1 | `crypto_aggressive_btc_sol_overlay` | +1.377% | 1.531% | 0.031 | 100/100 | PAPER_ONLY |
| 2 | `all_symbols_base` | +0.970% | 2.439% | 0.018 | 100/100 | PAPER_ONLY |
| 3 | `official_only_base` | +0.718% | 0.727% | 0.032 | 100/100 | PAPER_ONLY |
| 4 | `crypto_robust_sol_overlay` | +0.138% | 1.936% | 0.004 | 100/100 | REJECT |
| 5 | `crypto_all_reversion_overlay` | -0.490% | 1.852% | -0.018 | 100/100 | REJECT |

Best candidate crypto map:

```text
BARUSD=crypto_mean_reversion
BTCUSD=macd_momentum
ETHUSD=crypto_mean_reversion
SOLUSD=macd_momentum
XRPUSD=crypto_mean_reversion
```

## Interpretation

The isolated crypto allocation result did not fully transfer into the mixed
portfolio. The SOL-only robust map looked good in the crypto-only sleeve test,
but when combined with the official FX/metals book it fell to fourth and failed
fixed-warmup promotion.

The BTC+SOL aggressive overlay is the better mixed-portfolio research candidate:

- higher return than all-symbol MACD: +1.377% versus +0.970%
- lower drawdown than all-symbol MACD: 1.531% versus 2.439%
- clean risk discipline score: 100/100
- 100% non-negative fixed-warmup folds on the overlap sample

It is still not live-ready because the mixed overlap is short and the crypto data
is proxy data, not official MT5 competition data.

## Decision

Do not replace the current paper candidate solely from this result.

Keep `crypto_aggressive_btc_sol_overlay` as the next full-portfolio research
candidate, and treat `crypto_robust_sol_overlay` as crypto-only evidence that
failed the stronger mixed-portfolio test.

Next:

1. Re-run this comparison on official MT5 crypto captures.
2. If BTC+SOL remains ahead, compare it against the current paper candidate over
   the larger official FX/metals data.
3. Add a candidate scorecard row only after it survives official-data validation.

Follow-up: `97_RESEARCH_CANDIDATE_GATE.md` now formalizes this into a repeatable
gate. On the mixed proxy overlay output, BTC+SOL is `PAPER_ONLY` and the SOL-only
map is `REJECT`.

Sizing follow-up: `98_CRYPTO_OVERLAY_SIZING_COMPARISON.md` improves the BTC+SOL
research candidate by keeping SOL full-size and trimming BTC plus the reversion
crypto sleeve to 75%.
