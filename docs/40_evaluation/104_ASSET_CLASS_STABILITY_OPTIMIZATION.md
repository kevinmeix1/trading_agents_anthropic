# Asset-Class Stability Optimization

The crypto-only stability search showed that changing crypto sizing alone does
not solve fold concentration. The strongest fold also had large metals P&L, so
this optimizer searches FX and metal exposure multipliers around the crypto
overlay profiles.

Run:

```bash
PYTHONPATH=src .venv/bin/python \
  -c 'from quanthack.cli.asset_class_stability_optimize import main; main()' \
  --config configs/competition.toml \
  --price-csv data/mixed_official_crypto_proxy_overlap_prices.csv \
  --quote-csv data/mixed_official_crypto_proxy_overlap_quotes.csv \
  --train-size 96 \
  --test-size 48 \
  --step-size 48 \
  --output outputs/research/asset_class_stability_optimization.csv
```

The default grid tests 30 variants:

- current London crypto overlay
- current Asia crypto overlay
- current all-day crypto overlay
- soft London crypto overlay
- soft Asia crypto overlay
- full FX/metals, reduced metals, and reduced FX+metals

## Result

The search found 8 `STABLE_PROFILE` candidates. They are all Asia-session
variants with lower metals exposure.

Highest-return candidate remains:

| Candidate | Return | Max DD | Sharpe15 | Fold Positive | Fold Concentration | Status |
|---|---:|---:|---:|---:|---:|---|
| `current_london_full` | +1.852% | 1.446% | 0.044 | 66.7% | 93.9% | FRAGILE |

Stable-profile candidates:

| Candidate | Return | Max DD | Sharpe15 | Fold Positive | Fold Concentration |
|---|---:|---:|---:|---:|---:|
| `current_asia_metal25` | +0.441% | 0.466% | 0.040 | 100.0% | 65.8% |
| `current_asia_fx75_metal50` | +0.582% | 0.563% | 0.042 | 100.0% | 72.3% |
| `current_asia_fx75_metal75` | +0.742% | 0.702% | 0.041 | 100.0% | 78.0% |
| `current_asia_metal75` | +0.761% | 0.744% | 0.040 | 100.0% | 79.1% |
| `current_asia_metal50` | +0.601% | 0.605% | 0.040 | 100.0% | 74.1% |
| `soft_asia_metal25` | +0.395% | 0.418% | 0.038 | 100.0% | 71.6% |
| `soft_asia_fx75_metal50` | +0.536% | 0.515% | 0.040 | 100.0% | 77.3% |
| `soft_asia_metal50` | +0.555% | 0.557% | 0.038 | 100.0% | 78.9% |

## Interpretation

This is the first optimizer to produce fold-stable profiles under the current
promotion logic.

The tradeoff is clear:

- London profile: best return, still fragile.
- Asia profile with reduced metals: much smoother folds, lower return.
- Reducing metals directly lowers fold concentration.
- Reducing FX adds little extra benefit compared with metal reduction.

This supports a two-sleeve idea:

1. a high-return paper candidate: `current_london_full`
2. a robustness candidate: `current_asia_metal75` or
   `current_asia_fx75_metal75`

The stable Asia candidates are not live-ready yet because this is still
mixed-proxy data. They are now priority candidates for official MT5 validation.

## Decision

Do not replace the high-return paper candidate yet.

Next official-data workflow:

1. collect official MT5 crypto quotes
2. rerun `crypto-promotion-pipeline`
3. rerun `asset-class-stability-optimize`
4. compare `current_london_full` against stable Asia candidates
5. promote only if the official data confirms fold stability and return remains
   positive
