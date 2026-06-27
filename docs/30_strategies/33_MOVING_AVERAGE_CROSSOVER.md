# Moving Average Crossover

This step adds:

```text
ma_crossover
```

It is a trend-following strategy for FX and crypto.

## Why It Fits FX And Crypto

FX and crypto can both trend when markets react to macro news, liquidity shifts,
risk-on/risk-off behavior, or crypto momentum cycles.

The idea is simple:

```text
fast average above slow average -> bullish trend pressure
fast average below slow average -> bearish trend pressure
```

The important warning is that moving averages can overtrade in sideways markets.
That is why this implementation requires:

- A minimum fast/slow separation in basis points.
- A trend-efficiency filter.
- Spread/slippage cost checks.
- An exit band so small wiggles do not cause constant reversals.

## What The Strategy Reads

For each decision it computes:

```text
fast_average
slow_average
separation_bps
previous_separation_bps
crossed_direction
trend_efficiency
realized_volatility
```

The strategy can trade even if the exact cross happened before the bot started,
as long as the current fast/slow separation is strong enough. The fresh cross is
still recorded as a diagnostic.

## Config

In `configs/default.toml`:

```toml
[strategy.ma_crossover]
symbol = "EURUSD"
fast_window = 3
slow_window = 8
min_separation_bps = 2.0
exit_separation_bps = 0.5
min_trend_efficiency = 0.20
target_notional_usd = 50000.0
max_spread_bps = 10.0
```

Inside `alpha_router`, it is a weighted sleeve:

```toml
[strategy.alpha_router]
moving_average_weight = 0.20
```

That means the router can listen to the crossover, but it still combines it with
momentum, breakout, mean reversion, and ML alpha.

## Run In VS Code Terminal

Inspect the strategy:

```bash
quanthack strategy-demo --strategy ma_crossover --scenario up
```

Backtest it:

```bash
quanthack backtest --strategy ma_crossover
```

Tune it with walk-forward evaluation:

```bash
quanthack walk-forward --strategy ma_crossover
```

This chooses `fast_window`, `slow_window`, and `min_separation_bps` on each
training window, then evaluates the chosen parameters on the next unseen test
window.

Compare it with the existing simple momentum strategy:

```bash
quanthack compare --strategy ma_crossover --strategy simple_momentum
```

Inspect how the router sees it:

```bash
quanthack strategy-demo --strategy alpha_router --scenario up
```

Run tests:

```bash
python -m unittest discover -s tests
```

## How To Interpret It

If `ma_crossover` performs worse than simple momentum, that does not mean the
code is bad. It means this sample market path was too choppy, too costly, or the
windows were not a good fit.

For the hackathon, the useful question is:

```text
Does this strategy improve out-of-sample Sharpe or reduce bad trades when used
inside the router?
```

The fold output shows which parameters were selected:

```text
fast_window=2;slow_window=5;min_separation_bps=1.0
```
