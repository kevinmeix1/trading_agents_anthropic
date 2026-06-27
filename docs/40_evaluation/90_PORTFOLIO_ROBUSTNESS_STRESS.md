# Portfolio Robustness Stress

Date: 2026-06-22

This adds a leave-one-symbol-out robustness check for portfolio candidates.

Command:

```bash
python -c 'from quanthack.cli.portfolio_robustness import main; main()' \
  --config configs/competition.toml \
  --clock-open-at 2026-05-11T00:00:00+00:00 \
  --output outputs/research/champion_leave_one_symbol_out_robustness.csv
```

The command runs the baseline portfolio, then reruns the same strategy after excluding each symbol one at a time. It reports official-style return, drawdown, Sharpe, trade count, risk score, and the return delta versus baseline.

## Champion Ensemble Result

Baseline:

- Return: 3.709%
- Max drawdown: 3.575%
- Official 15-minute Sharpe: 0.020
- Trades: 190
- Risk discipline: 100/100

Leave-one-symbol-out ranking by return impact:

| Excluded | Return | Delta vs baseline | Max DD | Note |
| --- | ---: | ---: | ---: | --- |
| `USDCHF` | -3.171% | -6.880% | 3.458% | fragile |
| `XAGUSD` | 0.389% | -3.320% | 2.977% | fragile |
| `GBPUSD` | 3.512% | -0.197% | 3.541% | weaker |
| `XAUUSD` | 3.533% | -0.176% | 2.524% | weaker |
| `AUDUSD` | 3.613% | -0.096% | 3.235% | weaker |
| `EURGBP` | 3.813% | +0.105% | 3.539% | improved |
| `EURCHF` | 3.973% | +0.264% | 3.324% | improved |
| `USDCAD` | 4.113% | +0.404% | 3.343% | improved |
| `EURUSD` | 4.121% | +0.412% | 3.029% | improved |
| `USDJPY` | 4.403% | +0.694% | 2.940% | improved |

## Interpretation

The original concern was simple metals dependence because `XAGUSD` contributed most of the baseline P&L. The stress result confirms `XAGUSD` is fragile, but it also finds a less obvious dependency: excluding `USDCHF` flips the strategy negative even though standalone USDCHF P&L was small in the baseline attribution.

That means the portfolio result depends on allocation path interactions, not only direct symbol P&L. Removing one symbol changes gross exposure, concentration scaling, entry timing, and risk trimming across the remaining book.

## Next Research Direction

Do not blindly trade all official symbols just because data exists.

Next candidates to test:

- a stricter symbol eligibility map that keeps `XAGUSD` but reduces metals concentration;
- a no-weak-FX map excluding symbols whose removal improved the sample, tested by walk-forward rather than same-window return;
- a pair/group robustness stress for metals, USD pairs, and all improving/removing symbols;
- portfolio allocator changes that preserve diversification without allowing one symbol or hedge interaction to dominate the outcome.

## Reduced-Symbol Follow-Up

Same-window subset tests looked attractive:

| Candidate | Symbols | Return | Max DD | Sharpe 15m | Fills | Risk |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `baseline` | all 10 official symbols | 3.709% | 3.575% | 0.020 | 190 | 100 |
| `no_usdjpy` | all except `USDJPY` | 4.403% | 2.940% | 0.023 | 182 | 100 |
| `no_usdjpy_eurusd` | excludes `USDJPY`, `EURUSD` | 4.820% | 2.390% | 0.027 | 154 | 100 |
| `no_usdjpy_eurusd_usdcad` | excludes `USDJPY`, `EURUSD`, `USDCAD` | 5.278% | 2.070% | 0.030 | 130 | 100 |
| `fragility_core` | `AUDUSD`, `GBPUSD`, `USDCHF`, `XAGUSD`, `XAUUSD` | 5.806% | 1.777% | 0.033 | 90 | 100 |
| `metals_usdchf_core` | `USDCHF`, `XAGUSD`, `XAUUSD` | 5.511% | 1.456% | 0.036 | 42 | 100 |

Candidate scorecard ranking on the same window:

1. `fragility_core`: 92.5
2. `metals_usdchf_core`: 83.5
3. `no_usdjpy_eurusd_usdcad`: 59.5
4. `no_usdjpy_eurusd`: 40.5
5. `no_usdjpy`: 21.5
6. `baseline`: 2.5

However, the fixed-warmup walk-forward check rejected all three tested candidates:

| Candidate | Positive folds | Active positive folds | Non-negative folds | Median active return | Worst DD | Fills | Promotion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `baseline` | 16.7% | 16.7% | 16.7% | -0.192% | 1.315% | 106 | REJECT |
| `fragility_core` | 16.7% | 25.0% | 50.0% | -0.057% | 1.072% | 42 | REJECT |
| `metals_usdchf_core` | 33.3% | 50.0% | 66.7% | -0.058% | 0.901% | 22 | REJECT |

Conclusion: the reduced-symbol books are useful paper/research candidates, but not live-promotion candidates yet. The best next alpha work is not simply shrinking the symbol set. We need a regime-aware layer that can decide when the metals/USDCHF sleeve is active and when it should stand down.

## Session Gate Follow-Up

A portfolio-level session gate was added after this test. It blocks only new or larger exposure outside configured UTC hours; exits, reductions, and flattening remain allowed. Reversals outside the session are converted to flatten-only behavior so the system does not accidentally open new opposite exposure during a blocked window.

The gate can be tested with:

```bash
python -c 'from quanthack.cli.portfolio_backtest import main; main()' \
  --config configs/competition.toml \
  --clock-open-at 2026-05-11T00:00:00+00:00 \
  --symbol USDCHF --symbol XAGUSD --symbol XAUUSD \
  --entry-utc-hours '16|17|18|19|20'
```

And fold-validated with:

```bash
python -c 'from quanthack.cli.portfolio_fixed_warmup_walk_forward import main; main()' \
  --config configs/competition.toml \
  --symbol USDCHF --symbol XAGUSD --symbol XAUUSD \
  --entry-utc-hours '16|17|18|19|20' \
  --summary-output outputs/research/wf_core_session_h16_20_summary.csv \
  --folds-output outputs/research/wf_core_session_h16_20_folds.csv
```

Session-search result on the metals/USDCHF core:

| Variant | Full return | Max DD | Trades | Non-negative folds | Median active return | Worst fold DD | Promotion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Plain core | 5.511% | 1.456% | 42 | 66.7% | -0.058% | 0.901% | REJECT |
| UTC 16-20 | 0.515% | 1.260% | 10 | 83.3% | 0.258% | 1.260% | PAPER_ONLY |
| UTC 16-20 + vol target 0.0010 | 0.001% | 0.229% | 24 | 83.3% | 0.001% | 0.229% | PAPER_ONLY |
| UTC 16-20 + regime + vol target 0.0010 | 0.014% | 0.221% | 32 | 83.3% | 0.007% | 0.217% | PAPER_ONLY |

Interpretation:

- The session gate is useful as a safety/research tool because it improves non-negative fold stability.
- It is not yet a winning alpha layer because the return and trade count collapse.
- Volatility targeting with the session gate makes drawdown tiny and clears the 30-trade threshold only when regime tilt is also enabled, but return is effectively flat.

The next useful alpha work should identify a broader entry condition than only `16-20 UTC`: enough activity to clear trade-count and return requirements, but selective enough to avoid the early losing folds.
