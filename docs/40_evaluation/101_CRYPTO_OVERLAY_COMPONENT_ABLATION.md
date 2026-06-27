# Crypto Overlay Component Ablation

This pass asks which parts of the current top paper candidate are actually
carrying the return.

Candidate under test:

```text
btc075_sol100_reversion075_london
crypto entry hours UTC: 7-16
BARUSD=crypto_mean_reversion at 0.75x
BTCUSD=macd_momentum at 0.75x
ETHUSD=crypto_mean_reversion at 0.75x
SOLUSD=macd_momentum at 1.00x
XRPUSD=crypto_mean_reversion at 0.75x
official FX/metals=macd_momentum at 1.00x
```

Run:

```bash
PYTHONPATH=src .venv/bin/python -c 'from quanthack.cli.crypto_overlay_component_ablation import main; main()' \
  --config configs/competition.toml \
  --price-csv data/mixed_official_crypto_proxy_overlap_prices.csv \
  --quote-csv data/mixed_official_crypto_proxy_overlap_quotes.csv \
  --train-size 96 \
  --test-size 48 \
  --step-size 48 \
  --output outputs/research/crypto_overlay_component_ablation.csv
```

Gate:

```bash
PYTHONPATH=src .venv/bin/python -c 'from quanthack.cli.research_candidate_gate import main; main()' \
  --source path=outputs/research/crypto_overlay_component_ablation.csv,data_source=mixed_proxy \
  --output outputs/research/research_candidate_gate_crypto_overlay_component_ablation.csv
```

## Result

| Scenario | Return | Return Delta | Retention | Max DD | Sharpe15 | Trades | Fold Concentration |
|---|---:|---:|---:|---:|---:|---:|---:|
| `full` | +1.852% | 0.000% | 100.0% | 1.446% | 0.044 | 53 | 93.9% |
| `no_crypto` | +0.718% | -1.134% | 38.8% | 0.727% | 0.032 | 14 | 100.0% |
| `no_metals` | +0.908% | -0.943% | 49.1% | 0.894% | 0.038 | 41 | 88.3% |
| `no_fx` | +0.948% | -0.904% | 51.2% | 0.424% | 0.045 | 30 | 85.5% |
| `no_btc` | +1.382% | -0.469% | 74.7% | 1.194% | 0.040 | 47 | 92.4% |
| `no_sol` | +1.113% | -0.738% | 60.1% | 0.983% | 0.037 | 36 | 100.0% |
| `no_btc_sol` | +0.643% | -1.208% | 34.7% | 0.801% | 0.027 | 30 | 100.0% |
| `no_crypto_reversion` | +1.041% | -0.811% | 56.2% | 1.455% | 0.029 | 18 | 100.0% |
| `crypto_only` | +0.645% | -1.206% | 34.9% | 0.132% | 0.060 | 24 | 79.6% |
| `fx_only` | +0.000% | -1.851% | 0.0% | 0.214% | 0.000 | 8 | 100.0% |

`metals_only` went flat in this portfolio-level ablation. Do not read that as
"metals have no alpha"; the portfolio allocator has a minimum diversification
preference, and the test intentionally measures this candidate inside the current
portfolio construction rules.

## Interpretation

The candidate is genuinely multi-component:

- Removing all crypto drops retention to 38.8%.
- Removing metals drops retention to 49.1%.
- Removing BTC+SOL drops retention to 34.7%.
- Removing only BTC keeps 74.7%, while removing only SOL keeps 60.1%; SOL is the
  stronger single crypto contributor in this sample.
- Removing BAR/ETH/XRP reversion keeps only 56.2%, despite those names losing in
  the strongest fold. That means the reversion sleeve still improves the whole
  short-window portfolio through coverage/activity.

The `crypto_only` row has the best raw Sharpe15, but it has only 24 trades and
much lower return. It is useful evidence that crypto can be smooth, not evidence
to replace the diversified book.

## Decision

Keep the full London-hours sized overlay as the highest-return paper candidate,
but treat it as a portfolio interaction candidate, not a pure crypto strategy.

The next improvement should not be more optimization on this short mixed-proxy
window. The high leverage action is official MT5 crypto data collection and
rerunning:

1. `crypto-overlay-sizing-compare`
2. `crypto-overlay-component-ablation`
3. `crypto-overlay-fold-diagnostic`

Only promote if the same component structure survives official data and the fold
concentration falls.
