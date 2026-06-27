# Crypto Overlay Fold Diagnostic

This diagnostic investigates why the current top crypto paper candidate remains
paper-only: one positive fold contributes most of the walk-forward return.

Candidate:

```text
btc075_sol100_reversion075_london
crypto entry hours UTC: 7-16
BARUSD=crypto_mean_reversion at 0.75x
BTCUSD=macd_momentum at 0.75x
ETHUSD=crypto_mean_reversion at 0.75x
SOLUSD=macd_momentum at 1.00x
XRPUSD=crypto_mean_reversion at 0.75x
```

Run:

```bash
PYTHONPATH=src .venv/bin/python -c 'from quanthack.cli.crypto_overlay_fold_diagnostic import main; main()' \
  --config configs/competition.toml \
  --price-csv data/mixed_official_crypto_proxy_overlap_prices.csv \
  --quote-csv data/mixed_official_crypto_proxy_overlap_quotes.csv \
  --candidate 'label=btc075_sol100_reversion075_london,crypto=0.75,btc=0.75,sol=1.0,crypto_hours=7|8|9|10|11|12|13|14|15|16' \
  --train-size 96 \
  --test-size 48 \
  --step-size 48 \
  --output-prefix outputs/research/crypto_overlay_london_fold_diagnostic
```

Artifacts:

- `outputs/research/crypto_overlay_london_fold_diagnostic_summary.csv`
- `outputs/research/crypto_overlay_london_fold_diagnostic_folds.csv`
- `outputs/research/crypto_overlay_london_fold_diagnostic_fills.csv`
- `outputs/research/crypto_overlay_london_fold_diagnostic_attribution.csv`
- `outputs/research/crypto_overlay_london_fold_diagnostic_symbol_summary.csv`

## Fold Result

The candidate produced three fixed-warmup folds:

- positive fold fraction: 66.7%
- non-negative fold fraction: 100.0%
- strongest fold: fold 2
- strongest fold return: +2.320%
- largest positive fold contribution: 93.9%
- promotion: PAPER_ONLY

The concentration problem is real. The paper result is dominated by one fold.

## Strongest Fold Attribution

Fold 2 realized attribution by asset class:

| Asset class | Realized P&L |
|---|---:|
| Metals | +$11,404.82 |
| Crypto | +$9,489.66 |
| FX | +$2,302.95 |

Fold 2 symbol attribution:

| Symbol | Asset class | Realized P&L |
|---|---|---:|
| BTCUSD | Crypto | +$6,914.83 |
| SOLUSD | Crypto | +$6,892.53 |
| XAUUSD | Metal | +$6,000.97 |
| XAGUSD | Metal | +$5,403.86 |
| AUDUSD | FX | +$2,205.03 |
| USDCAD | FX | +$97.91 |
| XRPUSD | Crypto | -$853.49 |
| BARUSD | Crypto | -$1,577.67 |
| ETHUSD | Crypto | -$1,886.55 |

Interpretation:

- The fold was not a pure crypto win.
- BTC and SOL trend exposure helped strongly.
- Metals helped even more than crypto as a group.
- The crypto mean-reversion names were negative in the strongest fold.

## Trend-Only Follow-Up

Because crypto reversion names lost money in the strongest fold, we tested a
trend-only variant that keeps BTC/SOL and turns BAR/ETH/XRP to zero exposure.

Run:

```bash
PYTHONPATH=src .venv/bin/python -c 'from quanthack.cli.crypto_overlay_sizing_compare import main; main()' \
  --config configs/competition.toml \
  --price-csv data/mixed_official_crypto_proxy_overlap_prices.csv \
  --quote-csv data/mixed_official_crypto_proxy_overlap_quotes.csv \
  --train-size 96 \
  --test-size 48 \
  --step-size 48 \
  --candidate 'label=trend_only_london,crypto=0.0,trend=0.75,btc=0.75,sol=1.0,crypto_hours=7|8|9|10|11|12|13|14|15|16' \
  --candidate 'label=trend_only_all,crypto=0.0,trend=0.75,btc=0.75,sol=1.0' \
  --candidate 'label=trend_only_london_us,crypto=0.0,trend=0.75,btc=0.75,sol=1.0,crypto_hours=8|9|10|11|12|13|14|15|16|17|18|19|20|21' \
  --candidate 'label=trend_only_asia,crypto=0.0,trend=0.75,btc=0.75,sol=1.0,crypto_hours=0|1|2|3|4|5|6|7|8' \
  --output outputs/research/crypto_trend_only_sizing_comparison.csv
```

Result:

| Candidate | Return | Max DD | Sharpe15 | Trades | Fold concentration |
|---|---:|---:|---:|---:|---:|
| `trend_only_london` | +1.041% | 1.455% | 0.029 | 18 | 100.0% |
| `trend_only_all` | +1.041% | 1.455% | 0.029 | 18 | 100.0% |
| `trend_only_asia` | +0.718% | 0.727% | 0.032 | 14 | 100.0% |

Trend-only is worse than the mixed trend/reversion overlay:

- lower return
- lower Sharpe
- fewer trades
- worse fold concentration

So the crypto reversion sleeve should not be removed based only on the strongest
fold. It loses in that fold, but it appears to add coverage across the full
overlap sample.

## Decision

Keep `btc075_sol100_reversion075_london` as the highest-return paper candidate.

Do not promote it live. The current evidence says:

- useful alpha candidate
- mixed-proxy data only
- strong dependence on one fold
- strongest fold is partly metals-driven, not purely crypto-driven

Next work should focus on official MT5 crypto data or a longer data window. More
micro-optimizing this short overlap risks overfitting.

Follow-up: `101_CRYPTO_OVERLAY_COMPONENT_ABLATION.md` confirms the paper edge is
a portfolio interaction: BTC/SOL, metals, and crypto reversion each contribute
meaningfully, while a trend-only crypto variant is not enough.
