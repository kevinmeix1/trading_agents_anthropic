# Crypto Fold Stability Optimization

This optimizer searches crypto overlay sizing and session variants with an
explicit penalty for fold concentration. It exists because the current best
crypto overlay has attractive return but is too dependent on one positive fold.

Run:

```bash
PYTHONPATH=src .venv/bin/python \
  -c 'from quanthack.cli.crypto_fold_stability_optimize import main; main()' \
  --config configs/competition.toml \
  --price-csv data/mixed_official_crypto_proxy_overlap_prices.csv \
  --quote-csv data/mixed_official_crypto_proxy_overlap_quotes.csv \
  --train-size 96 \
  --test-size 48 \
  --step-size 48 \
  --output outputs/research/crypto_fold_stability_optimization.csv
```

The default grid tests 40 variants:

- current crypto overlay sizing
- softer BTC/SOL/reversion sizing
- lower balanced crypto sizing
- BTC-heavy and SOL-heavy low-reversion variants
- trend-only and reversion-only variants
- all-day, Asia, London, London/US, and US crypto entry windows

## Result

No candidate reached `STABLE_PROFILE`.

Top ranked candidates:

| Rank | Candidate | Return | Max DD | Sharpe15 | Fold Positive | Fold Concentration | Status |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `current_london` | +1.852% | 1.446% | 0.044 | 66.7% | 93.9% | FRAGILE |
| 2 | `current_london_us` | +1.757% | 1.446% | 0.042 | 66.7% | 93.9% | FRAGILE |
| 3 | `soft_london` | +1.627% | 1.245% | 0.044 | 66.7% | 94.5% | FRAGILE |
| 6 | `current_all` | +1.524% | 1.445% | 0.036 | 100.0% | 90.1% | FRAGILE |

Closest-to-stable candidates by fold concentration:

| Candidate | Return | Fold Positive | Median Active Fold | Fold Concentration |
|---|---:|---:|---:|---:|
| `current_asia` | +0.922% | 100.0% | 0.153% | 82.5% |
| `soft_asia` | +0.876% | 100.0% | 0.120% | 86.1% |
| `sol_heavy_lowrev_asia` | +0.837% | 100.0% | 0.100% | 89.2% |

## Interpretation

The search did not find a crypto-only sizing fix.

Useful observations:

- London keeps the best return.
- Asia improves fold distribution but gives up about half the return.
- All-day trading improves fold positivity but still has too much contribution
  from the strongest fold.
- Reducing crypto exposure alone does not solve the concentration problem.

This supports the earlier fold diagnostic: the dominant positive fold is a
portfolio event, not only a crypto event. Metals contributed more realized P&L
than crypto in that fold, so the next stability search should tune asset-class
exposure across crypto, metals, and FX together.

## Decision

Keep `current_london` as the best high-return paper candidate, but do not promote
it live.

Add a next optimizer that can vary asset-class multipliers, especially metals
and crypto together, to see whether we can keep enough return while pushing fold
concentration below 80%.
