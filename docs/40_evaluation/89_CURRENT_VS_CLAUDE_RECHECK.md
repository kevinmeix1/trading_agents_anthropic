# Current QuanHack Vs QuanHackClaude Recheck

Date: 2026-06-22

Folders compared:

- Current: `/Users/kaiwenmei/Desktop/quanthack`
- Claude: `/Users/kaiwenmei/Desktop/quanthackclaude`

## Test Health

| Repo | Runtime | Tests |
| --- | --- | --- |
| Current QuanHack | Python 3.11.14 | 592 passed |
| QuanHackClaude | Python 3.11 venv created for comparison | 511 passed, 5 skipped |

Claude did not already have a local `.venv`, so a Python 3.11 virtual environment was created before running its suite.

## Fair Backtest Setup

Both repos were run on the same official 10-symbol, 15-minute CSV data:

- `data/full_20gb_15m_prices.csv`
- `data/full_20gb_15m_quotes.csv`

Symbols:

`AUDUSD, EURCHF, EURGBP, EURUSD, GBPUSD, USDCAD, USDCHF, USDJPY, XAGUSD, XAUUSD`

The historical backtest data runs from 2026-05-11 to 2026-06-10, while the configured competition clock opens on 2026-06-21. For a fair replay, the current repo used `--clock-open-at 2026-05-11T00:00:00+00:00`; the Claude repo used an equivalent temporary config with the same historical open time.

## Backtest Result

Both repos produced the same champion-ensemble result on the same data:

| Repo | Strategy | Return | Max DD | Official 15m Sharpe | Fills | Risk score | Final equity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Current QuanHack | `champion_ensemble` | 3.709% | 3.575% | 0.020 | 190 | 100/100 | $1,037,088.81 |
| QuanHackClaude | `champion_ensemble` | 3.709% | 3.575% | 0.020 | 190 | 100/100 | $1,037,088.81 |

P&L concentration:

- `XAGUSD`: +$35,873.54
- `XAUUSD`: +$2,992.21
- Most FX pairs were small positive or negative contributors.

This means the shared champion-ensemble alpha path is effectively identical for the official 10-symbol dataset.

## Codebase Delta

Current QuanHack is ahead as the active hackathon branch:

- 377 tracked/untracked project files visible via `rg --files` versus 322 in Claude.
- 592 tests versus 511 in Claude.
- 69 registered CLI commands versus 60 in Claude.
- 28 registered strategy names versus 24 in Claude.
- Current adds crypto proxy data tooling, official/crypto mixed-data merging, asset-class attribution, candidate scorecards, sizing frontier, archive coverage, fold-complement diagnostics, position-risk stops, portfolio volatility targeting, and regime tilt controls.

Claude has a few files current does not need:

- `qh.py`
- `RESEARCH_LOG.md`
- `IMPROVEMENTS.md`

## Verdict

Use the current `/Users/kaiwenmei/Desktop/quanthack` repo as the main project.

The fair official-data backtest is a tie because both repos share the same champion-ensemble implementation for that path. Current QuanHack wins on hackathon readiness because it has more evaluation tooling, crypto-gap handling, risk monitoring, scorecard ranking, and deployment-readiness improvements.

The practical next priority is not merging Claude into current. It is to improve alpha robustness in the current repo, especially reducing dependence on `XAGUSD` and validating crypto behavior once official MT5 crypto data becomes available.
