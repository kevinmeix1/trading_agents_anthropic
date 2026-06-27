# Portfolio Volatility Targeting

This overlay scales the whole requested portfolio before allocation. It does not
predict direction. Its job is to smooth the equity curve:

- if the intended book has high recent realized volatility, scale targets down;
- if the intended book is calm, optionally allow a small scale-up;
- keep the allocator and risk engine as the final guardrails.

Use it as a Sharpe/drawdown lever, not as an alpha source.

## Command

```bash
quanthack portfolio-backtest \
  --strategy champion_ensemble \
  --strategy-map XAUUSD=macd_momentum \
  --strategy-map AUDUSD=macd_momentum \
  --strategy-map USDCHF=macd_momentum \
  --strategy-map EURUSD=macd_momentum \
  --symbol XAGUSD --symbol XAUUSD --symbol USDCHF --symbol AUDUSD \
  --symbol GBPUSD --symbol EURUSD --symbol EURGBP \
  --price-csv data/full_20gb_15m_prices.csv \
  --quote-csv data/full_20gb_15m_quotes.csv \
  --vol-target-bar-volatility 0.00035 \
  --vol-target-lookback 32 \
  --vol-target-min-observations 12 \
  --vol-target-min-scale 0.25 \
  --vol-target-max-scale 1.00
```

The extra report is written to:

```text
outputs/backtests/portfolio_volatility_targeting.csv
```

## First Full-Window Check

Current paper-style map on the converted full sample:

| Mode | Return | Max DD | Official 15m Sharpe | Fills | Turnover |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.297% | 0.118% | 0.027 | 88 | $5.36M |
| Vol target 0.00075, max 1.10 | 0.295% | 0.171% | 0.020 | 345 | $8.30M |
| Vol target 0.00035, max 1.00 | 0.258% | 0.106% | 0.027 | 98 | $5.17M |
| Vol target 0.00025, max 1.00 | 0.214% | 0.089% | 0.027 | 98 | $4.88M |
| Vol target 0.00015, max 1.00 | 0.177% | 0.069% | 0.030 | 134 | $4.28M |

Interpretation:

- loose targeting scaled the book up and hurt smoothness;
- tighter targeting reduced drawdown and turnover;
- the tightest target improved 15m Sharpe, but gave up too much return for a
  return-heavy competition score;
- keep the baseline as the current paper candidate;
- keep volatility targeting available as a defensive mode for unusually volatile
  live conditions.

## Hackathon Use

Use volatility targeting when the live feed shows regime risk rising, especially
before checkpoints or around metals/crypto volatility spikes. Do not enable it
blindly just because it sounds sophisticated. On our current data, it is a risk
discipline tool more than a score maximizer.
