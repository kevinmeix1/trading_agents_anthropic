# Strategy Registry

Step 17 turns strategy code into a small plug-in system.

Before this step, the project only knew about `SimpleMomentumStrategy`. That was
fine for learning, but weak for a hackathon because we need to compare ideas.

Now the common rule is:

```text
strategy.generate_request(prices) -> TradeRequest | None
```

That means a strategy can propose a trade, but it still cannot place a trade.
Market quality, risk, and dry-run execution remain separate.

## Strategies Available

### `simple_momentum`

Momentum says:

```text
If price moved up enough, buy.
If price moved down enough, sell.
```

### `multi_horizon_momentum`

Multi-horizon momentum says:

```text
If fast and slow momentum agree, and volatility is in a usable regime, trade with the trend.
```

It is a stricter trend sleeve than `simple_momentum`. The broad seven-symbol
version was rejected in walk-forward, but the `AUDUSD / USDCHF / XAUUSD` top-3
basket is a paper backup candidate.

### `autocorrelation_regime`

Autocorrelation regime says:

```text
If short-term returns are positively autocorrelated, follow the move.
If short-term returns are negatively autocorrelated and price is stretched, fade it.
```

It is a research diagnostic for serial-correlation regimes in FX/metals/crypto.
The first seven-symbol and asset-split tests were negative, so it should not be
added to champion, adaptive selection, or MT5 routing without new evidence.

### `conditional_seasonality`

Conditional seasonality says:

```text
Look at prior same-time-of-day examples.
Keep only examples whose recent momentum condition matches the current bar.
Trade the historical forward drift, or fade it in reversal mode.
```

This was built from the hourly research scans as a stricter alternative to raw
intraday seasonality. It is leakage-aware: each historical example uses only a
past forward return that would already be known at the current bar.

The initial direct-drift version churned badly. The default is therefore a
conservative reversal-mode research profile:

```text
signal_mode: reversal
min_samples: 3
entry_threshold_bps: 10.0
min_abs_tstat: 1.50
min_consistency: 0.67
```

Latest seven-symbol fixed-warmup evidence:

```text
positive fold fraction: 5.9%
active positive fold fraction: 25.0%
non-negative fold fraction: 82.4%
worst drawdown: 0.026%
fills: 20
promotion: REJECT
```

Verdict: useful as an example of a leakage-safe time-of-day research pipeline,
but not useful enough for the current adaptive or live candidate stack.

### `ma_crossover`

Moving-average crossover says:

```text
If the fast average is meaningfully above the slow average, buy.
If the fast average is meaningfully below the slow average, sell.
```

It is a smoother trend signal than raw momentum. The implementation requires the
average separation to clear a basis-point threshold and estimated trading costs.

### `macd_momentum`

MACD momentum says:

```text
If fast EMA momentum pulls away from slow EMA momentum and the histogram clears costs, trade with acceleration.
```

It uses a fast EMA, slow EMA, MACD signal line, and histogram. The optimized
default is a short-horizon 6/18/5 profile. It passed a cleaner standalone
walk-forward screen than the first baseline MACD settings, but it is not enabled
inside the champion ensemble by default.

### `breakout`

Breakout says:

```text
If price breaks above the recent prior range, buy.
If price breaks below the recent prior range, sell.
```

It uses prior prices for the channel, so the latest price is not included in the
range it is trying to break.

### `volatility_squeeze`

Volatility squeeze says:

```text
If volatility compressed first and price then breaks above the band, buy.
If volatility compressed first and price then breaks below the band, sell.
```

It is stricter than raw breakout because it checks `recent_volatility /
prior_volatility` before allowing an entry. This makes it a research sleeve for
testing whether selective breakouts can reduce churn.

### `dual_squeeze`

Dual squeeze says:

```text
Use a fast volatility squeeze for the entry.
Require a slower squeeze/bias confirmation before opening a new position.
```

This is a lower-turnover version of `volatility_squeeze`. The current default
uses a 14-price fast squeeze and a 24-price confirmation window. It improved the
full-sample portfolio backtest, but remains paper/research until longer
walk-forward evidence confirms enough trade activity.

### `asset_adaptive_dual_squeeze`

Asset-adaptive dual squeeze says:

```text
Use the current dual_squeeze profile for FX/crypto.
Use a faster, looser dual_squeeze profile for metals.
```

This captures the latest research finding that the faster profile helped
`XAGUSD` and `XAUUSD` but hurt several FX pairs. It improved the eligible-basket
full-sample backtest, but warmup walk-forward remains sparse, so it stays a
research sleeve rather than the live default.

### `range_expansion_trend`

Range expansion trend says:

```text
If the latest short impulse breaks outside the prior range
and is unusually large versus baseline volatility, trade with the break.
```

It is a stricter breakout-continuation sleeve than raw breakout or raw momentum.
The first loose version traded too often and was rejected. The default now uses
a stricter `10 bps` trigger move, `3 bps` range break, `2.5` expansion z-score,
and `0.65` trend-efficiency gate.

Latest top-4 eligible basket:

```text
XAGUSD
XAUUSD
USDCHF
AUDUSD
```

Fixed-warmup evidence:

```text
active positive folds: 57.1%
non-negative folds: 82.4%
median active return: 0.005%
worst drawdown: 0.053%
fills: 44
promotion: PAPER_ONLY
```

Adding this sleeve to adaptive selection did not beat the current best
`kalman_trend / champion_ensemble / macd_momentum` candidate, so keep it as a
paper sleeve and diagnostic rather than a live/default component.

### `trend_pullback`

Trend pullback says:

```text
If a larger trend pauses in a controlled pullback and then resumes, trade with the trend.
```

It is less aggressive than raw momentum because it waits for a pullback and a
resume bar. It is still research-only until walk-forward confirms that it adds
return without adding too much churn.

### `exhaustion_reversal`

Exhaustion reversal says:

```text
If price makes a large fast shock and then prints a reversal bar, fade the shock.
```

It checks shock size, shock z-score, shock path efficiency, reversal size, spread,
session, and estimated costs. Current backtests did not support promoting it, so
it stays as a documented research/rejection sleeve.

### `fixing_reversal`

Fixing reversal says:

```text
If price moves strongly into a configured fixing/session window
and then prints an opposite confirmation bar, fade the pre-window move.
```

This is an FX/metals intraday research sleeve. It is disabled for crypto by
default, exits quickly, and passes through the same allocator/risk checks as the
other portfolio strategies. The current tuned profile is tiny-positive
full-sample but mixed in walk-forward, so it is documented as research-only.

### `kalman_trend`

Kalman trend says:

```text
Smooth recent prices with a Kalman-style filter.
Trade with the smoothed trend only when slope and path efficiency confirm it.
Exit when the regime becomes chop or too volatile.
```

This is the first standalone advanced time-series strategy. It is currently the
strongest research candidate on the positive-attribution basket
`XAGUSD XAUUSD USDCHF AUDUSD GBPUSD`, but it still needs more validation before
automatic MT5 execution.

### `quality_trend`

Quality trend says:

```text
Trade only when MACD momentum and Kalman trend agree on direction.
Use the weaker signal confidence for position sizing.
Exit when agreement fades, reverses, or the holding cap is reached.
```

This is a conservative trend-quality sleeve built after the adaptive selector
showed that raw trend candidates can be too noisy. The revised default uses the
same 10-14 UTC FX/metals window as the stronger MACD runs, with lower confidence
and edge gates than the first prototype.

Latest seven-symbol fixed-warmup evidence:

```text
positive fold fraction: 17.6%
active positive fold fraction: 75.0%
non-negative fold fraction: 94.1%
median active return: 0.023%
worst drawdown: 0.055%
fills: 30
risk discipline: 100/100
promotion: PAPER_ONLY
```

It is clean when it trades but too sparse to replace the current adaptive paper
candidate. Adding it to adaptive selection reduced stitched OOS equity, so keep
it as a diagnostic/conservative sleeve until more data supports it.

### `mean_reversion`

Mean reversion says:

```text
If price is unusually high versus its recent average, sell.
If price is unusually low versus its recent average, buy.
```

It measures "unusually" with a z-score:

```text
zscore = (last_price - mean_price) / stdev_price
```

### `regime_switch`

The regime selector says:

```text
If the path is smooth and directional, use momentum.
If the baseline is stable and the latest price is stretched, use mean reversion.
If conditions are ambiguous or expensive, stay flat.
```

It chooses one of:

```text
MOMENTUM
MEAN_REVERSION
FLAT
```

### `alpha_router`

The alpha router says:

```text
Let every sleeve produce a normalized opinion.
Combine those opinions into one target position.
```

It currently listens to:

```text
momentum
ma_crossover
breakout
session_breakout
volatility_squeeze
mean_reversion
relative_strength
cross_rate_reversion
ml_alpha
```

If they agree, the router can enter. If they conflict, the router applies a
conflict penalty and may stay flat.

### `champion_ensemble`

The champion ensemble is the current research router for the best tested
sleeves:

```text
kalman_trend
asset_adaptive_dual_squeeze
fixing_reversal
macd_momentum
```

It is intentionally narrower than `alpha_router`. Kalman is the main trend
driver, and asset-adaptive squeeze is used as a confirmation/filter sleeve. The
default profile does not let asset-adaptive squeeze, fixing reversal, or MACD
trade alone.

### `usd_pressure_router`

The USD pressure router wraps `alpha_router`, then checks whether the broader USD
FX basket confirms the trade direction.

```text
positive pressure = broad USD strength
negative pressure = broad USD weakness
```

It is most useful in portfolio backtests and live dry-run monitoring because it
needs multiple symbols at the same timestamp.

## Config

In `configs/default.toml`:

```toml
[strategy]
active = "simple_momentum"

[strategy.simple_momentum]
symbol = "EURUSD"
lookback = 5
threshold_bps = 8.0
target_notional_usd = 50000.0

[strategy.ma_crossover]
symbol = "EURUSD"
fast_window = 3
slow_window = 8
min_separation_bps = 2.0
target_notional_usd = 50000.0

[strategy.mean_reversion]
symbol = "EURUSD"
lookback = 5
entry_zscore = 1.0
target_notional_usd = 50000.0

[strategy.breakout]
symbol = "EURUSD"
lookback = 8
breakout_buffer_bps = 2.0
target_notional_usd = 50000.0

[strategy.volatility_squeeze]
symbol = "EURUSD"
lookback = 24
squeeze_window = 8
breakout_buffer_bps = 2.5
max_squeeze_ratio = 0.50
forex_allowed_utc_hours = [11, 12, 13, 14, 15, 16, 17, 18, 19]
metal_allowed_utc_hours = [11, 12, 13, 14, 15, 16, 17, 18, 19]
target_notional_usd = 50000.0

[strategy.regime_switch]
symbol = "EURUSD"
lookback = 10
hysteresis_bars = 2

[strategy.alpha_router]
symbol = "EURUSD"
entry_score = 0.35
exit_score = 0.15
momentum_weight = 0.40
moving_average_weight = 0.20
breakout_weight = 0.35
mean_reversion_weight = 0.25

[strategy.champion_ensemble]
symbol = "EURUSD"
entry_score = 0.50
strong_lead_score = 0.50
kalman_trend_weight = 0.70
asset_adaptive_dual_squeeze_weight = 0.30
dual_squeeze_weight = 0.0
trend_pullback_weight = 0.0
conflict_penalty = 0.70

[strategy.usd_pressure]
lookback = 8
pressure_threshold_bps = 2.0
min_target_volatility_bps = 0.0
min_component_symbols = 3
min_confirming_symbols = 2
```

## Run In VS Code Terminal

Show the two strategy behaviors:

```bash
python scripts/evaluation/strategy_demo.py --strategy simple_momentum --scenario up
python scripts/evaluation/strategy_demo.py --strategy ma_crossover --scenario up
python scripts/evaluation/strategy_demo.py --strategy breakout --scenario up
python scripts/evaluation/strategy_demo.py --strategy volatility_squeeze --lookback 10 --prices 1.0000,1.0010,0.9990,1.0010,0.9990,1.0000,1.00005,0.99995,1.0000,1.0025
python scripts/evaluation/strategy_demo.py --strategy dual_squeeze --lookback 10 --prices 1.0000,1.0005,0.9995,1.0002,0.9998,1.0001,0.9999,1.0000,1.0001,1.0040
python scripts/evaluation/strategy_demo.py --strategy mean_reversion --scenario spike_up
python scripts/evaluation/strategy_demo.py --strategy mean_reversion --scenario spike_down
python scripts/evaluation/strategy_demo.py --strategy regime_switch --scenario up
python scripts/evaluation/strategy_demo.py --strategy alpha_router --scenario up
python scripts/evaluation/strategy_demo.py --strategy champion_ensemble --scenario up
python scripts/evaluation/strategy_demo.py --strategy usd_pressure_router --scenario up
```

Run a backtest with each strategy:

```bash
python scripts/evaluation/run_backtest.py --strategy simple_momentum
python scripts/evaluation/run_backtest.py --strategy ma_crossover
python scripts/evaluation/run_backtest.py --strategy breakout
python scripts/evaluation/run_backtest.py --strategy volatility_squeeze
python scripts/evaluation/run_backtest.py --strategy dual_squeeze
python scripts/evaluation/run_backtest.py --strategy mean_reversion
python scripts/evaluation/run_backtest.py --strategy regime_switch
python scripts/evaluation/run_backtest.py --strategy alpha_router
python scripts/evaluation/run_backtest.py --strategy champion_ensemble
python scripts/evaluation/run_backtest.py --strategy usd_pressure_router
```

Run tests:

```bash
python -m unittest discover -s tests
```

## What Changed In The Code

- `Strategy` is a protocol: anything with `generate_request(prices)` can be used.
- `build_strategy(...)` chooses a strategy by name.
- `BacktestEngine` accepts the protocol instead of a concrete strategy class.
- Config stores both strategy parameter sets.
- `regime_switch` acts as a controller over the base strategies.
- `alpha_router` combines normalized strategy signals into one final decision.
- `champion_ensemble` combines only the current best research sleeves.

This is the foundation for adding more strategy sleeves without rewriting the
risk engine.
