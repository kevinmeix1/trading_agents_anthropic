# Alpha Router

Step 27 adds a multi-strategy decision layer:

```text
alpha_router
```

The goal is not to place one trade per strategy. The goal is to let strategies
produce opinions, then combine those opinions into one final target position.

## Mental Model

Old shape:

```text
one strategy -> one trade request
```

Router shape:

```text
momentum signal
moving-average crossover signal
breakout signal
session breakout signal
mean-reversion signal
FX cross-rate reversion signal
ML alpha signal
  -> alpha router
  -> one StrategyDecision
  -> risk engine
```

## StrategySignal

Each sleeve now can be represented as a normalized signal:

```text
strategy_name
symbol
direction: LONG / SHORT / FLAT
confidence: 0.0 to 1.0
expected_edge_bps
cost_bps
weight
horizon
reason
diagnostics
```

The router combines weighted signed scores:

```text
LONG  = positive score
SHORT = negative score
FLAT  = zero score
```

## Conflict Handling

If momentum says `LONG` and mean reversion says `SHORT`, the router does not
blindly average them and trade anyway. It applies a conflict penalty.

This is why the router may stay flat even when one sleeve looks excited.

## Primary Signal Override

The router can also avoid being too passive.

If there is no signal conflict and one sleeve has high confidence plus enough
edge after estimated costs, the router may enter from that primary signal even
when the combined weighted score is diluted by inactive sleeves.

This is useful for breakout-heavy FX/metals windows where the standalone
breakout strategy survives walk-forward, but the router would otherwise stay
flat because momentum, moving average, or mean reversion are inactive.

## Config

In `configs/default.toml`:

```toml
[strategy.alpha_router]
symbol = "EURUSD"
target_notional_usd = 50000.0
max_target_notional_usd = 75000.0
min_trade_notional_usd = 1000.0
entry_score = 0.35
exit_score = 0.15
min_signal_confidence = 0.20
cost_buffer = 1.20
max_spread_bps = 10.0
momentum_weight = 0.30
moving_average_weight = 0.15
breakout_weight = 0.15
session_breakout_weight = 0.25
volatility_squeeze_weight = 0.0
dual_squeeze_weight = 0.0
mean_reversion_weight = 0.35
relative_strength_weight = 0.0
cross_rate_weight = 0.0
conflict_penalty = 0.50
primary_signal_override_enabled = true
primary_signal_min_confidence = 0.90
primary_signal_min_edge_bps = 4.0
adaptive_weighting_enabled = true
adaptive_regime_lookback = 80
chop_mean_reversion_multiplier = 1.20
chop_trend_signal_multiplier = 0.75
trend_aligned_signal_multiplier = 1.20
trend_counter_signal_multiplier = 0.65
metal_mean_reversion_multiplier = 1.25
metal_raw_breakout_multiplier = 0.60
relative_strength_min_score_dispersion = 0.75
relative_strength_min_target_trend_efficiency = 0.20
```

## Session Breakout Sleeve

The router now includes `session_breakout` as a separate low-weight sleeve. This
is not the same as raw Donchian `breakout`:

- `breakout` asks whether price broke the channel.
- `session_breakout` asks whether the breakout also passes session, volatility,
  spread, expected-edge, and optional regime filters.

The previous 20GB run showed that raw breakouts overtraded, while session
breakout reduced churn but was not stable enough to trade alone. Keeping it as a
soft router vote lets it shape conviction without becoming the whole strategy.

## Squeeze Sleeves

The router can also listen to `volatility_squeeze` and `dual_squeeze`, but both
default weights are `0.0`. Current research favors running `dual_squeeze`
standalone because it is low turnover and less noisy than the older router
mixture. Keep these router sleeves off unless a walk-forward/optimizer run shows
the blend is better than standalone squeeze.

After the fast signal-diagnostics screen, the default router weights were tilted
away from raw breakout and toward session breakout plus mean reversion. The
reason is practical: raw breakout had negative forward-return diagnostics across
several focus symbols, while session breakout and mean reversion were stronger
on GBPUSD, EURUSD, USDJPY, XAUUSD, and XAGUSD.

## FX Cross-Rate Sleeve

The router can include `cross_rate_reversion` as an opt-in FX-only sleeve via
`cross_rate_weight`.

This sleeve compares the target pair to a synthetic cross rate from other FX
pairs. For example, `EURGBP` can be checked against `EURUSD / GBPUSD`. If the
target is rich versus the synthetic value, the sleeve votes short; if it is
cheap, it votes long.

It is disabled by default because realistic spread and slippage assumptions made
standalone cross-rate trades rare in the first 15-minute diagnostics run. As a
router vote, though, it can still be useful confirmation for selected FX pairs.

The allocator-aware router optimizer can now test cross-rate and relative
strength as part of the weight grid without enabling either by default:

```bash
quanthack router-optimize \
  --candidate 0.20,0.10,0.10,0.40,0.15,0.05,0.10
```

That tuple means:

```text
momentum, moving-average crossover, breakout, mean-reversion, session-breakout, cross-rate, relative-strength
```

Older four-number candidates still work; they keep session-breakout at its
default research weight and keep cross-rate and relative-strength at zero.

When relative-strength has nonzero weight, the router also applies two extra
guards before accepting the vote:

```text
score dispersion >= 0.75
target trend efficiency >= 0.20
```

Those guards exist because early full-window smoke tests showed a loose
relative-strength vote increased turnover sharply.

## Adaptive Weighting

The router can adjust weights before a full backtest:

- In choppy regimes, mean reversion gets more weight and trend-style signals get
  less weight.
- In trend regimes, aligned momentum/session-breakout signals get more weight,
  while counter-trend signals get less weight.
- For metals, mean reversion gets extra weight and raw breakout is damped.

This does not create a new trade signal by itself. It changes how much each
existing sleeve can influence the final router score. The goal is to encode what
the previous diagnostics showed without overfitting directly to one full
backtest.

## Run In VS Code Terminal

Inspect the signals:

```bash
python scripts/evaluation/strategy_demo.py --strategy alpha_router --scenario up
```

Backtest the router:

```bash
python scripts/evaluation/run_backtest.py --strategy alpha_router
```

Compare all strategies:

```bash
python scripts/evaluation/compare_strategies.py
```

Inspect sleeve-level diagnostics:

```bash
python scripts/evaluation/signal_diagnostics.py --strategy alpha_router --horizon-bars 4
```

Run walk-forward evaluation:

```bash
python scripts/evaluation/run_walk_forward.py
```

Run tests:

```bash
python -m unittest tests.test_strategy
```

## Current Sample Result

On the tiny sample CSV, `alpha_router` trades less than pure momentum. It is
currently more conservative, not more profitable.

That is acceptable for this step. The important architecture is now in place:

```text
many opinions -> one controlled decision
```

Next useful sleeve: volatility breakout or Bollinger-band mean reversion.
