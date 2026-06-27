# Research Candidate Gate

The research candidate gate turns comparison CSV rows into a stricter candidate
decision table:

- `LIVE_READY`: official data, positive return, stable folds, and clean risk.
- `PAPER_ONLY`: useful research evidence, but not enough for live promotion.
- `REJECT`: negative return, unstable folds, failed walk-forward promotion, or
  weak risk discipline.

This matters because some comparison tools can produce a high-ranked candidate
on proxy data. That can be useful for research, but it should not become a live
MT5 candidate until official competition data confirms it.

## Command

```bash
PYTHONPATH=src .venv/bin/python -c 'from quanthack.cli.research_candidate_gate import main; main()' \
  --source path=outputs/research/crypto_overlay_comparison.csv,data_source=mixed_proxy \
  --output outputs/research/research_candidate_gate_crypto_overlay.csv
```

Supported data sources:

- `official`
- `proxy`
- `mixed_proxy`
- `synthetic`

Only `official` evidence can become `LIVE_READY`.

## Latest Crypto Overlay Gate

Output:

```text
outputs/research/research_candidate_gate_crypto_overlay.csv
```

Ranked gate result:

| Rank | Candidate | Readiness | Return | Max DD | Risk | Reason |
|---:|---|---|---:|---:|---:|---|
| 1 | `crypto_aggressive_btc_sol_overlay` | PAPER_ONLY | +1.377% | 1.531% | 100/100 | mixed proxy data and concentrated positive fold |
| 2 | `all_symbols_base` | PAPER_ONLY | +0.970% | 2.439% | 100/100 | mixed proxy data and selective live gate miss |
| 3 | `official_only_base` | PAPER_ONLY | +0.718% | 0.727% | 100/100 | mixed proxy comparison file and fold concentration |
| 4 | `crypto_robust_sol_overlay` | REJECT | +0.138% | 1.936% | 100/100 | non-negative fold fraction below 70% |
| 5 | `crypto_all_reversion_overlay` | REJECT | -0.490% | 1.852% | 100/100 | negative return and weak fold stability |

## Decision

The gate keeps the current crypto conclusion disciplined:

- BTC+SOL overlay is the best full-portfolio crypto research candidate.
- It is not live-ready because the evidence uses mixed official/proxy data.
- SOL-only crypto allocation is no longer the leading full-portfolio candidate,
  despite looking good in the isolated crypto allocation test.

Use this gate after each future optimizer/comparison pass so the project keeps
separating research alpha from competition-ready alpha.
