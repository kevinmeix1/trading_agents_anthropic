# MACD Momentum

`macd_momentum` is a momentum-acceleration sleeve for FX, metals, and crypto.

The idea:

```text
EMA fast line - EMA slow line = MACD line
MACD line - EMA signal line = histogram
Trade only when the histogram and MACD line agree and clear costs.
```

Why it exists:

```text
Raw momentum says price moved.
MACD asks whether shorter-term momentum is pulling away from slower momentum.
That can catch acceleration without needing volume data.
```

The implementation is conservative:

- Trades only configured UTC hours.
- Requires the MACD histogram, MACD line, trend efficiency, and estimated costs to pass.
- Uses volatility sizing with a max notional cap.
- Has minimum and maximum holding periods.
- Routes through allocator, market quality, slippage, and risk checks.

Current optimized default:

```text
fast_window: 6
slow_window: 18
signal_window: 5
min_histogram_bps: 2.0
min_macd_bps: 1.0
min_histogram_slope_bps: 0.0
min_trend_efficiency: 0.20
max_holding_period: 12 bars
forex/metals UTC hours: 10, 11, 12, 13, 14
```

## Evidence

Session attribution showed weak late-session rows around UTC 17-18 and stronger
rows around UTC 12-14. A follow-up basket optimization found that starting one
hour earlier improved the conservative MACD basket, so the default profile now
uses the tighter 10-14 UTC window for FX and metals.

Seven-symbol session-filter evidence:

```text
symbols: XAGUSD, XAUUSD, USDCHF, AUDUSD, GBPUSD, EURUSD, EURGBP
full-sample return: 0.271%
max drawdown: 0.127%
official 15m Sharpe: 0.026
fills: 78
risk discipline: 100/100

short fixed-warmup validation:
  active fold fraction: 58.8%
  active positive fold fraction: 60.0%
  non-negative fold fraction: 76.5%
  median active test return: 0.017%
  promotion: PAPER_ONLY
```

Older symbol-eligible basket before the session filter:

```text
symbols: XAUUSD, AUDUSD, XAGUSD, EURGBP, EURUSD, USDCHF
full-sample return: 0.199%
max drawdown: 0.121%
official 15m Sharpe: 0.019
fills: 72
risk discipline: 100/100
```

Older short fixed-warmup walk-forward:

```text
folds: 17
positive fold fraction: 35.3%
active fold fraction: 52.9%
active positive fold fraction: 66.7%
non-negative fold fraction: 82.4%
median active test return: 0.026%
promotion: PAPER_ONLY
```

Conservative MACD basket after the 10-14 UTC update:

```text
symbols: AUDUSD, EURCHF, EURUSD, USDCAD, USDJPY, XAGUSD, XAUUSD
active fold fraction: 41.2%
active positive fold fraction: 85.7%
non-negative fold fraction: 94.1%
median active test return: 0.072%
worst test drawdown: 0.052%
fills: 62
promotion: PAPER_ONLY
```

Verdict:

```text
Useful research sleeve and adaptive-selector candidate. The session-filtered
profile is cleaner than the initial baseline, but it remains paper-only until
it proves stronger total positive fold coverage.
```

Champion blend result:

```text
Adding MACD at 20% weight improved full-sample champion return from 0.436% to
0.557%, but wider walk-forward active median worsened. Keep champion MACD
weight at 0.0 until stronger validation appears.
```

Histogram slope scan:

```text
On the conservative MACD basket, min_histogram_slope_bps=0.20 slightly improved
full-sample return while preserving active-fold validation. On the main adaptive
candidate, the same default reduced stitched OOS equity and total positive fold
fraction. Keep slope as an optimizer option rather than the global default.
```

## Commands

Optimize MACD parameters:

```bash
quanthack macd-momentum-optimize \
  --symbol XAGUSD --symbol XAUUSD --symbol USDCHF --symbol AUDUSD --symbol GBPUSD \
  --price-csv data/full_20gb_15m_prices.csv \
  --quote-csv data/full_20gb_15m_quotes.csv \
  --include-walk-forward \
  --train-size 480 \
  --test-size 96 \
  --step-size 96 \
  --output outputs/backtests/macd_momentum_optimization_short_wf.csv
```

Optimize MACD session windows:

```bash
quanthack macd-momentum-optimize \
  --symbol XAGUSD --symbol XAUUSD --symbol USDCHF --symbol AUDUSD \
  --symbol GBPUSD --symbol EURUSD --symbol EURGBP \
  --price-csv data/full_20gb_15m_prices.csv \
  --quote-csv data/full_20gb_15m_quotes.csv \
  --include-walk-forward \
  --train-size 480 \
  --test-size 96 \
  --step-size 96 \
  --candidate 'early_10_14,6,18,5,2.0,1.0,0.20,12,10|11|12|13|14' \
  --output outputs/backtests/macd_momentum_session_optimization_wf.csv
```

Backtest the current optimized profile:

```bash
quanthack portfolio-backtest \
  --strategy macd_momentum \
  --symbol XAGUSD --symbol XAUUSD --symbol USDCHF --symbol AUDUSD --symbol GBPUSD \
  --price-csv data/full_20gb_15m_prices.csv \
  --quote-csv data/full_20gb_15m_quotes.csv
```

Run the symbol eligibility scan:

```bash
quanthack symbol-eligibility-optimize \
  --strategy macd_momentum \
  --price-csv data/full_20gb_15m_prices.csv \
  --quote-csv data/full_20gb_15m_quotes.csv \
  --include-walk-forward \
  --train-size 480 \
  --test-size 96 \
  --step-size 96 \
  --output outputs/backtests/macd_momentum_symbol_eligibility_wf.csv \
  --attribution-output outputs/backtests/macd_momentum_symbol_attribution_wf.csv
```
