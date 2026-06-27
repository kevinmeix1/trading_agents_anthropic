# Multi-Horizon Momentum

`multi_horizon_momentum` is a volatility-managed trend sleeve for FX, metals,
and crypto.

The idea:

```text
Trade only when fast momentum and slower momentum agree.
Reject the signal when volatility is too quiet, too unstable, or too expensive.
Size by realized volatility and route through the same allocator/risk stack.
```

Why it exists:

```text
The current best candidates are selective trend/momentum profiles.
This sleeve tries to improve signal quality by requiring trend agreement across
two horizons and by avoiding poor volatility regimes.
```

Current default:

```text
fast_lookback: 6 bars
slow_lookback: 24 bars
volatility_lookback: 12 bars
baseline_volatility_lookback: 48 bars
minimum fast move: 2.0 bps
minimum slow move: 5.0 bps
minimum trend efficiency: 0.25
volatility ratio band: 0.35-2.50
FX/metals UTC hours: 10, 11, 12, 13, 14
```

## Evidence

Seven-symbol broad run:

```text
symbols: XAGUSD, XAUUSD, USDCHF, AUDUSD, GBPUSD, EURUSD, EURGBP
full-sample return: 0.122%
max drawdown: 0.184%
official 15m Sharpe: 0.010
fills: 266

fixed-warmup validation:
  active positive fold fraction: 40.0%
  non-negative fold fraction: 47.1%
  median active return: -0.010%
  verdict: REJECT
```

Best subset found by symbol eligibility:

```text
symbols: AUDUSD, USDCHF, XAUUSD
full-sample return: 0.205%
max drawdown: 0.061%
official 15m Sharpe: 0.034
fills: 80

fixed-warmup validation:
  positive fold fraction: 35.3%
  active fold fraction: 58.8%
  active positive fold fraction: 60.0%
  non-negative fold fraction: 82.4%
  median active return: 0.009%
  worst drawdown: 0.039%
  promotion: PAPER_ONLY
```

Adaptive recipe check:

```text
candidate: multi_horizon_top3 recipe
symbols: AUDUSD, USDCHF, XAUUSD
stitched OOS final equity: $1,003,145.74
active positive folds: 54.5%
non-negative folds: 76.5%
median active return: 0.013%
promotion: PAPER_ONLY
```

Verdict:

```text
Keep as a research sleeve and conservative paper backup. Do not add it to the
main seven-symbol adaptive candidate yet; it lowers fold quality when used too
broadly.
```

## Commands

Optimize parameters:

```bash
quanthack multi-horizon-momentum-optimize \
  --symbol XAGUSD --symbol XAUUSD --symbol USDCHF --symbol AUDUSD \
  --symbol GBPUSD --symbol EURUSD --symbol EURGBP \
  --price-csv data/full_20gb_15m_prices.csv \
  --quote-csv data/full_20gb_15m_quotes.csv \
  --include-walk-forward \
  --train-size 480 \
  --test-size 96 \
  --step-size 96 \
  --output outputs/backtests/multi_horizon_momentum_7_optimization_wf.csv
```

Validate the top-3 subset:

```bash
quanthack portfolio-fixed-warmup-walk-forward \
  --strategy multi_horizon_momentum \
  --symbol AUDUSD --symbol USDCHF --symbol XAUUSD \
  --price-csv data/full_20gb_15m_prices.csv \
  --quote-csv data/full_20gb_15m_quotes.csv \
  --train-size 480 \
  --test-size 96 \
  --step-size 96 \
  --summary-output outputs/backtests/multi_horizon_top3_fixed_warmup_summary.csv \
  --folds-output outputs/backtests/multi_horizon_top3_fixed_warmup_folds.csv
```
