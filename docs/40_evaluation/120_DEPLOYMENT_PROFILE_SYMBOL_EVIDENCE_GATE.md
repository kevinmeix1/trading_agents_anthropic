# Deployment Profile Symbol Evidence Gate

Purpose: test a live-safe adaptive guard for fragile symbols instead of a static
multiplier edit.

The `symbol_refined` profile depends too much on `USDCAD`, but static removal or
replacement destroys too much return. This experiment adds a targeted online
symbol-evidence gate:

- Apply the gate only to fragile symbols, currently `USDCAD`.
- If the symbol has no closed-trade history yet, allow only a small probe entry.
- If the probe closes profitably, future entries can pass the normal evidence
  checks.
- If recent closed evidence is poor, block new/increased exposure while still
  allowing reductions and exits.

The generic portfolio commands now support the same knobs:

```bash
--symbol-evidence-gate \
--symbol-evidence-target-symbol USDCAD \
--symbol-evidence-block-without-history \
--symbol-evidence-no-history-multiplier 0.25 \
--symbol-evidence-failed-multiplier 0.0
```

Default deployment-profile sweep:

```bash
PYTHONPATH=src .venv/bin/python -c 'from quanthack.cli.deployment_profile_symbol_evidence_refine import main; main()' \
  --config configs/competition.toml \
  --profile-pack-json outputs/research/deployment_profile_symbol_gated_pack.json \
  --slot symbol_refined \
  --robustness-csv outputs/research/deployment_profile_symbol_refined_robustness.csv \
  --price-csv data/mixed_official_crypto_proxy_overlap_prices.csv \
  --quote-csv data/mixed_official_crypto_proxy_overlap_quotes.csv \
  --output outputs/research/deployment_profile_symbol_evidence_gate.csv \
  --recommendation-json outputs/research/deployment_profile_symbol_evidence_gate.json
```

## Current Real-Data Result

Input profile: `symbol_refined`

Targeted fragile symbol:

- `USDCAD`

Results:

- Baseline: return `0.619%`, drawdown `0.389%`, Sharpe15 `0.062`, risk
  `100/100`, dependency loss `0.446%`.
- `USDCAD` no-history probe at `0.25x`: return `0.612%`, retention `98.8%`,
  drawdown improves to `0.329%`, Sharpe15 improves to `0.068`, dependency loss
  reduction only `1.6%`.
- `USDCAD` no-history probe at `0.50x`: return `0.615%`, retention `99.2%`,
  dependency loss reduction `1.1%`.
- `USDCAD` no-history probe at `0.75x`: return `0.617%`, retention `99.6%`,
  dependency loss reduction `0.5%`.

Conclusion: the probe gate is useful as a live safety mechanism because it lets
the bot start fragile symbols smaller until they earn trust. It does not solve
the historical `USDCAD` dependency in this data window, because the profile only
took one `USDCAD` round trip and it closed slightly profitable. The next alpha
research should focus on new independent signals, while live operation can still
use the probe gate as a conservative safety overlay.
