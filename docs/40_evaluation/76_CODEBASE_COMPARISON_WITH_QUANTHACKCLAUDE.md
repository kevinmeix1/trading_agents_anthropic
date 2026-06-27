# QuanHack vs QuanHackClaude Comparison

Generated from side-by-side tests on the shared Desktop folders:

- `/Users/kaiwenmei/Desktop/quanthack`
- `/Users/kaiwenmei/Desktop/quanthackclaude`

## Verdict

Use `/Users/kaiwenmei/Desktop/quanthack` as the primary competition codebase.

The deployable documented candidates are tied because both folders contain the
same current adaptive strategy stack. The newer `quanthack` folder is still the
better base because it adds more competition-specific readiness tooling, more
tests, corrected router walk-forward warmup behavior, portfolio volatility and
regime overlays, position-risk work, and stronger router alpha research via
MACD/Kalman router sleeves.

`quanthackclaude` still has useful engineering pieces to copy later: GitHub CI,
dev tooling config, `configs/competition.toml`, `IMPROVEMENTS.md`, and
`RESEARCH_LOG.md`.

## Test Health

```text
quanthack:
  unit tests: 549 passed

quanthackclaude:
  unit tests: 511 passed
```

Interpretation: both are internally consistent. `quanthack` has a larger tested
surface, and the added tests pass.

## Shared Main Candidate

Command profile:

```text
adaptive selector
strategies: kalman_trend, champion_ensemble, macd_momentum
symbols: XAGUSD, XAUUSD, USDCHF, AUDUSD, GBPUSD, EURUSD, EURGBP
train/test/step: 480 / 96 / 96
loss cooldown: 1 fold
```

Both folders produced identical results:

```text
folds: 17
positive fold fraction: 47.1%
active fold fraction: 64.7%
active positive fold fraction: 72.7%
non-negative fold fraction: 82.4%
median active test return: 0.033%
worst test drawdown: 0.071%
risk discipline: 100/100
evaluation fills: 86
stitched OOS final equity: $1,004,225.57
promotion: PAPER_ONLY
```

Interpretation: neither codebase has an advantage on the current documented
paper candidate. The strategy is still paper-only because the total positive
fold fraction is below the stricter live gate.

## Shared Static Backup

Command profile:

```text
XAGUSD=champion_ensemble
XAUUSD=macd_momentum
AUDUSD=macd_momentum
USDCHF=macd_momentum
EURUSD=macd_momentum
train/test/step: 480 / 96 / 96
```

Both folders produced identical results:

```text
folds: 17
positive fold fraction: 29.4%
active fold fraction: 47.1%
active positive fold fraction: 62.5%
non-negative fold fraction: 82.4%
median active test return: 0.019%
worst test drawdown: 0.100%
risk discipline: 100/100
evaluation fills: 54
promotion: PAPER_ONLY
```

Interpretation: good conservative fallback, but not better than the adaptive
candidate.

## Router Research Comparison

Full-sample router optimizer:

```text
quanthack:
  best: Kalman-only router sleeve
  return: 0.394%
  max drawdown: 0.125%
  Sharpe 15m: 0.035
  final equity: $1,003,938.13
  trades: 156

quanthackclaude:
  best: dual-squeeze-only router sleeve
  return: 0.021%
  max drawdown: 0.020%
  Sharpe 15m: 0.010
  final equity: $1,000,205.34
  trades: 30
```

Small router walk-forward check with `train-size=480`, `test-size=96`,
`step-size=480`:

```text
quanthack:
  folds: 4
  eligible: true
  promotion: PAPER_ONLY
  stable fold fraction: 50.0%
  selected was test-best: 25.0%
  median test return: 0.001%
  worst test drawdown: 0.012%
  most selected: Kalman-only router sleeve

quanthackclaude:
  folds: 4
  eligible: false
  promotion: REJECT
  stable fold fraction: 0.0%
  selected was test-best: 0.0%
  median test return: -0.009%
  worst test drawdown: 0.058%
  most selected: dual-squeeze-only router sleeve
```

Interpretation: the newer router work is better research material, but still
not a live replacement for the adaptive paper candidate. It gives us a promising
Kalman/MACD path to continue refining.

## Competition Readiness

`quanthack` has a dedicated readiness command. Current output:

```text
overall: FAIL
data coverage: 10/15 official instruments
missing asset class: CRYPTO
candidate: PAPER_ONLY
missing symbols: BARUSD, BTCUSD, ETHUSD, SOLUSD, XRPUSD
risk limits: PASS
```

Interpretation: this is exactly aligned with the hackathon risks. The biggest
gap is not Python architecture; it is missing crypto data/live coverage and a
paper candidate that has not cleared the live promotion gate.

## Codebase Differences That Matter

`quanthack` advantages:

- More tests: 549 versus 511.
- More strategy work: quality trend, conditional seasonality, range expansion,
  portfolio volatility targeting, regime tilt, position risk, and readiness
  reporting.
- Better router research: MACD/Kalman router sleeves and behavior profiles.
- Better router walk-forward methodology: uses train+test warmup while
  evaluating only the test region, which avoids unfairly starving slow
  indicators of history.
- More explicit hackathon readiness checks for official symbols, missing
  crypto, risk limits, and candidate promotion status.

`quanthackclaude` advantages:

- Has `.github/workflows/ci.yml`.
- Has dev-tooling config in `pyproject.toml` for ruff, mypy, and pytest.
- Has `configs/competition.toml`.
- Has concise top-level research notes: `IMPROVEMENTS.md` and `RESEARCH_LOG.md`.

## Recommended Next Moves

1. Keep building from `/Users/kaiwenmei/Desktop/quanthack`.
2. Copy the useful engineering hygiene from `quanthackclaude`: CI, dev extras,
   ruff/mypy/pytest config, and competition config.
3. Do not replace the current paper candidate with the router yet. Treat the
   newer Kalman/MACD router as research.
4. Fix the biggest competition gap: crypto data/live MT5 quote capture for
   BARUSD, BTCUSD, ETHUSD, SOLUSD, and XRPUSD.
5. Continue improving regime diversification because the main candidate still
   has too many flat or non-positive folds for automatic live promotion.
