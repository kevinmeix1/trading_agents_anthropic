# Crypto Promotion Pipeline

This adds a single command for the crypto overlay promotion workflow. It turns
the manual sequence into one reproducible evidence bundle:

1. market data health check
2. crypto overlay sizing comparison
3. research candidate gate for sizing
4. component ablation
5. research candidate gate for components
6. asset-class stability search for a conservative backup candidate
7. fold diagnostic and attribution
8. final go/no-go summary

Run:

```bash
PYTHONPATH=src .venv/bin/python \
  -c 'from quanthack.cli.crypto_promotion_pipeline import main; main()' \
  --config configs/competition.toml \
  --price-csv data/mixed_official_crypto_proxy_overlap_prices.csv \
  --quote-csv data/mixed_official_crypto_proxy_overlap_quotes.csv \
  --data-source mixed_proxy \
  --train-size 96 \
  --test-size 48 \
  --step-size 48 \
  --output-prefix outputs/research/crypto_promotion_pipeline
```

Artifacts:

- `outputs/research/crypto_promotion_pipeline_data_health.csv`
- `outputs/research/crypto_promotion_pipeline_sizing.csv`
- `outputs/research/crypto_promotion_pipeline_sizing_gate.csv`
- `outputs/research/crypto_promotion_pipeline_component_ablation.csv`
- `outputs/research/crypto_promotion_pipeline_component_gate.csv`
- `outputs/research/crypto_promotion_pipeline_asset_class_stability.csv`
- `outputs/research/crypto_promotion_pipeline_fold_diagnostic_folds.csv`
- `outputs/research/crypto_promotion_pipeline_fold_diagnostic_fills.csv`
- `outputs/research/crypto_promotion_pipeline_fold_diagnostic_attribution.csv`
- `outputs/research/crypto_promotion_pipeline_fold_diagnostic_summary.csv`
- `outputs/research/crypto_promotion_pipeline_fold_diagnostic_symbol_summary.csv`
- `outputs/research/crypto_promotion_pipeline_summary.csv`

## Current Mixed-Proxy Result

The pipeline selected:

```text
btc075_sol100_reversion075_london
crypto entry hours UTC: 7-16
```

Summary:

| Field | Value |
|---|---:|
| Return | +1.852% |
| Max drawdown | 1.446% |
| Sharpe15 | 0.044 |
| Risk discipline | 100/100 |
| Folds | 3 |
| Strongest fold | 2 |
| Strongest fold return | +2.320% |
| Largest positive fold contribution | 93.9% |
| Stable backup | `current_asia_metal75` |
| Stable backup return | +0.761% |
| Stable backup fold contribution | 79.1% |
| Decision | PAPER_ONLY |

The decision reason is intentionally conservative:

```text
mixed_proxy data cannot be live-ready;
market data has warnings;
sizing gate is PAPER_ONLY;
largest positive fold contribution 93.9% exceeds 80.0% live threshold
```

The stable backup candidate is also intentionally conservative:

```text
current_asia_metal75
crypto entry hours UTC: 0-8
metal multiplier: 0.75
return: +0.761%
fold contribution: 79.1%
fixed-warmup promotion status: PROMOTE
```

Because the data source is still `mixed_proxy`, the backup is a validation
candidate rather than a live candidate. Its purpose is to give us a lower-return,
lower-concentration profile to test immediately when official MT5 crypto data is
available.

Data-health warnings were not catastrophic:

- `EURCHF` and `USDCHF` had tiny spread breaches just above their internal
  spread limits.
- `XAGUSD` and `XAUUSD` had 1-hour gaps in the overlap slice.

## Why This Matters

This command is now the promotion checklist for official MT5 crypto captures.
When we replace mixed proxy crypto with official MT5 quotes, rerun the same
pipeline with:

```bash
--data-source official
```

Only consider a live deployment if:

- the final summary says `LIVE_READY`
- data health is `OK`
- fold concentration is below the live threshold
- risk discipline remains near 100/100
- the component ablation still shows the alpha is diversified, not one symbol or
  one lucky window
- the stable backup either remains positive on official data or is no longer
  needed because the primary candidate becomes stable

For now, the current candidate remains a useful paper alpha, not a deployable
live strategy.
