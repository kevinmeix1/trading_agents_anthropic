# QuanHack vs QuanHackClaude Backtest Comparison

Date: 2026-06-22

Compared folders:

- Main: `/Users/kaiwenmei/Desktop/quanthack`
- Claude: `/Users/kaiwenmei/Desktop/quanthackclaude`

Verdict: **continue development in main `quanthack`**.

The current main repo is now both safer and stronger. It beats
`quanthackclaude` on the stronger all-MACD candidate with higher return, lower
drawdown, better active-fold consistency, fewer fills, and lower turnover.
Claude remains useful as a historical reference, but it is no longer the better
research or deployment base.

## Test Health

| Repo | Test command | Result |
| --- | --- | --- |
| Main `quanthack` | `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests` | **569 passed** |
| `quanthackclaude` | `PYTHONPATH=src /Users/kaiwenmei/Desktop/quanthack/.venv/bin/python -m unittest discover -s tests` | **504 passed, 2 failed, 5 errors** |

Claude has no local `.venv`, so it was tested with the main repo's Python 3.11
venv. The system Python in the Claude folder is Python 3.9.6, below the project
requirement of Python 3.11.

Claude's failing tests are meaningful:

- `LiveDryRunEngine` is missing the `continue_on_error` interface expected by
  resilience tests.
- Router walk-forward promotion tests are out of sync with the
  `most_selected_behavior` summary field.
- Router optimizer CSV tests expect an older schema.

For the hackathon, the live-loop resilience gap matters most. A transient
quote/API failure should be recorded and skipped, not crash a multi-day run.

## Structural Comparison

| Area | Main `quanthack` | `quanthackclaude` | Interpretation |
| --- | ---: | ---: | --- |
| Python source files | 140 | 129 | Main has broader functionality. |
| Test files | 63 | 58 | Main has more validation coverage. |
| Test result | 569 passing | 504 passing with failures | Main is healthier. |
| Main-only Python modules | 11 | 0 | Claude has no Python module missing from main. |

Important main-only modules:

- `src/quanthack/backtesting/fold_complement.py`
- `src/quanthack/backtesting/portfolio_regime.py`
- `src/quanthack/backtesting/portfolio_volatility.py`
- `src/quanthack/backtesting/sizing_frontier.py`
- `src/quanthack/cli/fold_complement.py`
- `src/quanthack/cli/hackathon_readiness.py`
- `src/quanthack/cli/sizing_frontier.py`
- `src/quanthack/reporting/hackathon_readiness.py`
- `src/quanthack/trading/position_risk.py`

This matters because the main repo has more direct support for competition
readiness: sizing frontiers, fold-complement analysis, regime tilt, volatility
targeting, readiness checks, and position-risk tooling.

## Backtest Data

Both repos were tested on the same local 15-minute CSV data:

- `data/full_20gb_15m_prices.csv`
- `data/full_20gb_15m_quotes.csv`
- 21,953 data rows per file, excluding the header
- Symbols: `AUDUSD`, `EURCHF`, `EURGBP`, `EURUSD`, `GBPUSD`, `USDCAD`,
  `USDCHF`, `USDJPY`, `XAGUSD`, `XAUUSD`

This covers 10 of the 15 official instruments. The missing crypto symbols
remain the biggest competition gap: `BARUSD`, `BTCUSD`, `ETHUSD`, `SOLUSD`,
and `XRPUSD`.

## Configured Champion Portfolio

Command shape:

```bash
PYTHONPATH=src python -c "from quanthack.cli.portfolio_backtest import main; main()" \
  --config configs/competition.toml \
  --price-csv data/full_20gb_15m_prices.csv \
  --quote-csv data/full_20gb_15m_quotes.csv
```

| Metric | Main | Claude |
| --- | ---: | ---: |
| Strategy | champion_ensemble | champion_ensemble |
| Final equity | $1,037,088.81 | $1,037,088.81 |
| Total return | 3.709% | 3.709% |
| Official 15m Sharpe | 0.020 | 0.020 |
| Max drawdown | 3.575% | 3.575% |
| Fills | 190 | 190 |
| Worst leverage | 5.61x | 5.61x |
| Worst net directional exposure | 70.9% | 70.9% |
| Risk discipline score | 100/100 | 100/100 |

The configured champion book is identical in both repos.

## All-MACD Portfolio

Command shape:

```bash
PYTHONPATH=src python -c "from quanthack.cli.portfolio_backtest import main; main()" \
  --config configs/competition.toml \
  --strategy-map AUDUSD=macd_momentum \
  --strategy-map EURCHF=macd_momentum \
  --strategy-map EURGBP=macd_momentum \
  --strategy-map EURUSD=macd_momentum \
  --strategy-map GBPUSD=macd_momentum \
  --strategy-map USDCAD=macd_momentum \
  --strategy-map USDCHF=macd_momentum \
  --strategy-map USDJPY=macd_momentum \
  --strategy-map XAGUSD=macd_momentum \
  --strategy-map XAUUSD=macd_momentum \
  --price-csv data/full_20gb_15m_prices.csv \
  --quote-csv data/full_20gb_15m_quotes.csv
```

| Metric | Main | Claude |
| --- | ---: | ---: |
| Final equity | $1,060,008.54 | $1,058,958.08 |
| Total return | 6.001% | 5.896% |
| Official 15m Sharpe | 0.038 | 0.035 |
| Max drawdown | 0.855% | 1.694% |
| Fills | 84 | 110 |
| Turnover | $59,131,162.79 | $77,231,330.27 |
| Worst leverage | 5.81x | 5.98x |
| Worst net directional exposure | 64.5% | 66.0% |
| Worst largest-symbol concentration | 50.2% | 50.1% |
| Risk discipline score | 100/100 | 100/100 |

Main is ahead by $1,050.46 on the full sample and has about half the max
drawdown. The improvement came from two later refinements:

- `macd_momentum.min_histogram_bps` was tightened from `2.0` to `2.5`, reducing
  low-quality MACD trades.
- Position stops became asset-class specific: FX `1%`, metals `2%`, crypto
  `2.5%`.

Symbol-level notes:

| Symbol | Main P&L | Claude P&L | Better |
| --- | ---: | ---: | --- |
| XAGUSD | $19,994.98 | $20,088.70 | Claude by $93.72 |
| XAUUSD | $25,567.83 | $23,243.74 | Main by $2,324.09 |
| AUDUSD | $8,502.11 | $10,331.91 | Claude by $1,829.80 |
| USDCHF | $3,466.91 | $2,350.35 | Main by $1,116.56 |

Claude's old `XAGUSD` advantage has effectively disappeared. Main kept the
metal upside while removing more noisy trades elsewhere.

## All-MACD Fixed-Warmup Walk-Forward

Command shape:

```bash
PYTHONPATH=src python -c "from quanthack.cli.portfolio_fixed_warmup_walk_forward import main; main()" \
  --config configs/competition.toml \
  --train-size 480 \
  --test-size 96 \
  --step-size 96 \
  --strategy-map AUDUSD=macd_momentum \
  --strategy-map EURCHF=macd_momentum \
  --strategy-map EURGBP=macd_momentum \
  --strategy-map EURUSD=macd_momentum \
  --strategy-map GBPUSD=macd_momentum \
  --strategy-map USDCAD=macd_momentum \
  --strategy-map USDCHF=macd_momentum \
  --strategy-map USDJPY=macd_momentum \
  --strategy-map XAGUSD=macd_momentum \
  --strategy-map XAUUSD=macd_momentum
```

| Metric | Main | Claude |
| --- | ---: | ---: |
| Folds | 17 | 17 |
| Positive fold fraction | 35.3% | 35.3% |
| Active fold fraction | 52.9% | 64.7% |
| Active positive fold fraction | 66.7% | 54.5% |
| Non-negative fold fraction | 82.4% | 70.6% |
| Median test return | 0.000% | 0.000% |
| Median active test return | 0.641% | 0.336% |
| Median test Sharpe 15m | 0.000 | 0.000 |
| Worst test drawdown | 0.824% | 0.926% |
| Risk discipline | 100.0/100 | 100.0/100 |
| Evaluation fills | 84 | 110 |
| Promotion | PAPER_ONLY | PAPER_ONLY |

The robustness view favors main. Both are still paper-only because total
positive folds are only `35.3%`, but main is materially better when it trades:
`66.7%` active-positive folds versus Claude's `54.5%`, and `82.4%`
non-negative folds versus `70.6%`.

## Hackathon Readiness

Main is the better repo, but neither result is live-promotion quality yet. The
largest remaining score gap is not code organization; it is alpha coverage:

- Crypto coverage is still missing from the downloaded data.
- Total positive fold fraction is still too low.
- Weekend/flat FX windows need a complementary return sleeve.
- MT5 deployment should wait until the strategy candidate is stronger.

## Recommendation

Use **main `quanthack`** as the active project.

Why:

1. It has every Claude Python module plus newer modules Claude lacks.
2. It has more tests and all 569 pass.
3. It has the better all-MACD result: `6.001%` return, `0.855%` drawdown,
   `0.038` official 15m Sharpe, and `100/100` risk discipline.
4. Its walk-forward behavior is better: fewer active windows, but much cleaner
   active windows.
5. It is safer for eventual MT5 use because it has live-loop resilience and
   stronger readiness/risk tooling.

Next priorities:

1. Import or capture crypto data for `BTCUSD`, `ETHUSD`, `SOLUSD`, `XRPUSD`,
   and `BARUSD`.
2. Build a low-turnover complement that can help the flat MACD folds without
   destroying the non-negative fold rate.
3. Rerun strategy-map optimization with crypto included.
4. Use main's readiness checker as the go/no-go gate before MT5 automation.
