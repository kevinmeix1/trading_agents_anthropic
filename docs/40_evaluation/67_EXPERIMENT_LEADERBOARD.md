# Experiment Leaderboard

`experiment-leaderboard` ranks walk-forward summary CSVs from
`outputs/backtests`.

Why it exists:

```text
Strategy research now produces many CSVs.
The leaderboard gives one quick view of which experiments are cleanest by
positive folds, active-positive folds, non-negative folds, compounded
out-of-sample return, drawdown, median active return, fills, and risk
discipline.
```

Command:

```bash
quanthack experiment-leaderboard \
  --input 'outputs/backtests/*summary.csv' \
  --output outputs/backtests/experiment_leaderboard.csv \
  --limit 20
```

2026-06-23 adaptive selector read:

```text
output:
  outputs/research/adaptive_selector_variant_leaderboard.csv

1. adaptive_current_top_recheck
   score: 1.016
   compounded OOS return: 5.245%
   active positive folds: 77.8%
   non-negative folds: 88.2%
   worst drawdown: 0.812%

2. adaptive_plus_squeeze
   score: 0.989
   compounded OOS return: 4.845%
   active positive folds: 83.3%
   non-negative folds: 94.1%

3. adaptive_plus_macd_squeeze
   score: 0.989
   compounded OOS return: 5.238%
   active positive folds: 75.0%
   non-negative folds: 88.2%

4. adaptive_current_top_cash_fallback_pos10bp
   score: 0.957
   compounded OOS return: 4.841%
   active positive folds: 80.0%
   non-negative folds: 94.1%

5. adaptive_current_top_cash_fallback
   score: 0.937
   compounded OOS return: 5.530%
   active positive folds: 62.5%
   non-negative folds: 82.4%
```

Verdict: the current adaptive stack remains the paper leader. The squeeze
variants are useful research sleeves, but they do not justify promotion because
one gives up return and the other is a near tie with slightly weaker fold
quality. Cash fallback is a real challenger but not the default: a zero-threshold
cash fallback earned more but reduced fold consistency, while a tiny positive
training gate improved safety but became too idle. Stability-preference selector
variants and broad idle-sleeve scans were materially worse and should remain
diagnostics only.

Earlier read:

```text
The conservative MACD basket ranks as the cleanest validation profile.
The seven-symbol adaptive candidate remains the stronger broader paper candidate.
Rejected add-ons such as USD pressure, broad multi-horizon momentum, and session
breakout do not improve the main candidate.
```

Use this before changing defaults. If an experiment does not beat the current
leaderboard profile on both return quality and fold stability, keep it as a
research sleeve instead of promoting it.
