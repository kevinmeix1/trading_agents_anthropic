# Relative Strength Strategy

The relative strength strategy is a portfolio-aware alpha sleeve. Instead of judging one
symbol by itself, it compares every active symbol at the same timestamp and asks:

> Which instruments are strongest or weakest relative to the rest of the basket?

## Why this matters

Forex, metals, and crypto often move in groups. A standalone momentum rule can buy a
symbol just because it rose, even if every other symbol rose more. Relative strength is
stricter: it wants the target symbol to be unusually strong or unusually weak versus the
current portfolio universe.

## Signal Logic

For each symbol with enough history:

1. Take the recent log-return move over `lookback` prices.
2. Convert that move into basis points.
3. Estimate realized volatility over the same window.
4. Score the symbol as:

```text
score = move_bps / max(realized_volatility_bps, volatility_floor_bps)
```

Then the strategy ranks all symbols by score.

- High positive relative z-score: long target symbol.
- High negative relative z-score: short target symbol.
- Neutral relative z-score: no new trade, or exit an existing position if the edge fades.

## Risk And Cost Filters

The strategy still has to pass normal strategy-level filters before the allocator and risk
engine see the target:

- `min_component_symbols`: avoids ranking a basket that is too small.
- `min_abs_move_bps`: ignores tiny raw moves.
- `entry_zscore` and `exit_zscore`: provide hysteresis.
- `require_asset_class_confirmation`: optionally requires the target to also rank
  well within its own asset class.
- `asset_class_entry_zscore`: same-asset confirmation threshold.
- `asset_class_min_symbols`: minimum same-asset basket size before confirmation
  can be calculated.
- `require_metal_trend_confirmation`: optionally requires XAG/XAU trades to pass
  a simple target-symbol trend check.
- `metal_trend_min_move_bps`: minimum aligned metal move.
- `metal_trend_min_efficiency`: minimum directional path efficiency for metal entries.
- `max_spread_bps`: blocks poor quotes.
- `slippage_bps`, `fee_bps`, and `cost_buffer`: require enough edge to clear estimated costs.
- `max_target_notional_usd`: caps position size before portfolio allocation.

After this, the portfolio allocator can still trim exposure for concentration, leverage,
asset class, and directional exposure.

## Alpha Router Sleeve

`alpha_router` can now include relative strength as an opt-in vote:

```toml
[strategy.alpha_router]
relative_strength_weight = 0.0
```

Keep the default at zero until router walk-forward validates a nonzero weight.
The router also requires enough cross-sectional dispersion and an efficient
target-symbol move before it accepts the relative-strength vote; this keeps the
router from churning on weak rank changes.
To research it, append relative strength as the seventh router candidate value:

```bash
quanthack router-optimize \
  --candidate 0.20,0.10,0.10,0.40,0.15,0.05,0.10
```

The order is:

```text
momentum, moving-average, breakout, mean-reversion, session-breakout, cross-rate, relative-strength
```

## Current Result

On `data/downloaded_scan_prices.csv` and `data/downloaded_scan_quotes.csv`, the initial
version ranked first in the full portfolio comparison:

- Final equity: `$1,003,865.90`
- Return: `0.387%`
- Max drawdown: `0.112%`
- Official 15m Sharpe: `0.128`
- Risk discipline: `100/100`
- Trades: `142`

The walk-forward result was promising but not yet stable enough:

- Stable fold fraction: `29.4%`
- Median test return: `0.000%`
- Eligible: `False`

## Asset-Class Confirmation Experiment

The strategy now supports optional same-asset confirmation. This was added because the
first profitable result was concentrated in metals, so we needed to test whether a symbol
should also be strong versus its own group.

On the downloaded scan data, strict confirmation reduced full-sample return and reduced
walk-forward stability:

- Default confirmation disabled: final equity `$1,003,865.90`, stable fold fraction `52.9%`
- Strict confirmation at `0.35`: final equity `$1,003,374.37`, stable fold fraction `41.2%`

Conclusion: keep the confirmation logic available for future tuning, but leave it
disabled by default for now.

## Parameter Optimization

Use the optimizer when changing `relative_strength` knobs:

```bash
python scripts/evaluation/relative_strength_optimize.py \
  --price-csv data/downloaded_scan_prices.csv \
  --quote-csv data/downloaded_scan_quotes.csv \
  --output outputs/experiments/relative_strength_parameter_optimization.csv
```

For the more important stability check:

```bash
python scripts/evaluation/relative_strength_optimize.py \
  --walk-forward \
  --train-size 40 \
  --test-size 16 \
  --step-size 8 \
  --max-baskets 10 \
  --price-csv data/downloaded_scan_prices.csv \
  --quote-csv data/downloaded_scan_quotes.csv \
  --output outputs/experiments/relative_strength_parameter_optimization_walk_forward.csv
```

Latest result:

| Candidate | Full Return | Stable Folds | Eligible |
| --- | ---: | ---: | --- |
| `baseline_l12_z0_75` | `0.387%` | `64.7%` | `True` |
| `asset_confirm_l12_z0_75` | `0.337%` | `41.2%` | `False` |
| `permissive_l12_z0_50` | `0.383%` | `41.2%` | `False` |
| `selective_l12_z1_00` | `0.346%` | `41.2%` | `False` |
| `fast_l8_z0_75` | `0.163%` | `35.3%` | `False` |
| `metal_trend_l12_z0_75` | `0.453%` | `29.4%` | `False` |
| `slow_l16_z0_75` | `0.296%` | `0.0%` | `False` |

Conclusion: the default `lookback=12`, `entry_zscore=0.75`, `exit_zscore=0.25`
setting remains the best current candidate because it is the only preset that passes the
walk-forward stability gate.

## Metals Trend Confirmation Experiment

Because most of the early P&L came from XAGUSD and XAUUSD, the strategy also supports an
optional metal-only trend confirmation gate. It checks that a proposed metals trade has:

- an aligned target-symbol move, and
- enough trend efficiency to avoid pure whipsaw.

On the downloaded scan data, the best full-sample version improved final equity:

- Default disabled: `$1,003,865.90`
- Metal move `2.0` bps, efficiency `0.20`: `$1,004,531.93`

But the same setting reduced walk-forward stability:

- Default disabled: stable fold fraction `52.9%`, eligible `True`
- Metal trend confirmation: stable fold fraction `29.4%`, eligible `False`

Conclusion: keep this gate disabled for now. It may be useful later inside a
walk-forward optimizer, but it is not robust enough to promote as the default.
