# Asset-Adaptive MACD Test

Date: 2026-06-22

## Why This Test

Previous mixed-universe tests showed that crypto MACD was the only tested crypto
sleeve with positive contribution, but it added drawdown. A crypto-only MACD
parameter sweep then showed a stricter threshold set had lower return but much
lower drawdown and slightly better 15m Sharpe.

I fixed one optimizer issue first: MACD candidate `allowed_utc_hours` now updates
crypto hours as well as FX and metals.

## Crypto Proxy 14-Day Result

Current crypto MACD:

- return: `+0.913%`
- max drawdown: `0.350%`
- official 15m Sharpe view: `0.029`
- fills: `24`
- risk score: `100/100`

Asset-adaptive crypto MACD:

- return: `+0.543%`
- max drawdown: `0.051%`
- official 15m Sharpe view: `0.031`
- fills: `10`
- risk score: `100/100`

Interpretation: the adaptive version gives up return and trade count, but it is
much smoother on the research proxy.

## Mixed 15-Symbol Overlap

Current mixed MACD:

- return: `+0.970%`
- max drawdown: `2.439%`
- fills: `32`
- risk score: `100/100`

Asset-adaptive MACD:

- return: `+0.932%`
- max drawdown: `2.231%`
- fills: `24`
- risk score: `100/100`
- allocation statuses: `OK` only

Asset-class attribution for adaptive MACD:

- `METAL`: `+$6,406`, 6 fills
- `CRYPTO`: `+$2,910`, 10 fills
- `FOREX`: roughly flat

## Decision

Do not replace the current research mixed MACD yet. The adaptive version is a
safer crypto variant, but the fill count falls below the 30-trade Sharpe-prize
threshold and return is slightly lower.

Keep `asset_adaptive_macd` as a risk-controlled alternative for live/MT5 crypto
validation. If official crypto quotes show higher noise or larger drawdowns than
the Binance proxy, this is the first fallback to test.
