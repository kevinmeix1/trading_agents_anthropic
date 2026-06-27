# Candidate Scorecard

This module ranks saved portfolio backtest output bundles using the same broad logic as the hackathon score: return, drawdown, Sharpe, and risk discipline.

It is useful because a raw backtest return can be misleading. A candidate that makes slightly less money but has lower drawdown, enough trades for the Sharpe prize, and a clean risk profile may be better for the competition than a more fragile high-return run.

## Command

The safest command format is key-value based:

```bash
python -m quanthack.cli.candidate_scorecard \
  --candidate label=current_mixed,equity=outputs/research/overlap_mixed15_macd_equity.csv,fills=outputs/research/overlap_mixed15_macd_fills.csv,pnl=outputs/research/overlap_mixed15_macd_pnl.csv \
  --candidate label=adaptive_mixed,equity=outputs/research/overlap_mixed15_asset_adaptive_macd_equity.csv,fills=outputs/research/overlap_mixed15_asset_adaptive_macd_fills.csv,pnl=outputs/research/overlap_mixed15_asset_adaptive_macd_pnl.csv \
  --candidate label=official10,equity=outputs/research/overlap_official10_macd_equity.csv,fills=outputs/research/overlap_official10_macd_fills.csv,pnl=outputs/research/overlap_official10_macd_pnl.csv \
  --output outputs/research/overlap_candidate_scorecard.csv
```

The older `LABEL:EQUITY_CSV:FILLS_CSV[:PNL_CSV]` form still works. The key-value form is preferred because it is friendlier to Windows paths when we later move MT5 work to a Windows machine.

## Latest Mixed Official/Crypto Proxy Result

Current ranking from `outputs/research/overlap_candidate_scorecard.csv`:

| Rank | Candidate | Return | Max DD | Sharpe 15m | Trades | Sharpe prize eligible | Risk score | Composite |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 1 | `current_mixed` | 0.970% | 2.439% | 0.018 | 32 | yes | 100 | 77.5 |
| 2 | `adaptive_mixed` | 0.932% | 2.231% | 0.018 | 24 | no | 100 | 45.0 |
| 3 | `official10` | 0.718% | 0.727% | 0.032 | 14 | no | 100 | 27.5 |

Interpretation:

- `current_mixed` is the current overlap-window leader because it has the best return and clears the 30-trade Sharpe prize threshold.
- `adaptive_mixed` is a safer fallback candidate with slightly lower drawdown and fewer trades, but it misses the Sharpe prize trade-count threshold.
- `official10` is the cleanest official-data-only comparison and has the best Sharpe/drawdown profile, but the lower return and low trade count make it less competitive on this overlap sample.

## Caveats

The crypto data is a Binance research proxy, not official competition MT5 crypto data. Use it to design and sanity-check crypto behavior, not to claim production/live validation.

This scorecard ranks only the candidates passed into it. Percentile-style ranks are relative to that small candidate set, so a score of 77.5 here is not a predicted leaderboard score. It is a disciplined way to avoid choosing a candidate based on return alone.
