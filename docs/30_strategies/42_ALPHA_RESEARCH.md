# Alpha Research Notes

This note is the current alpha research view before adding more strategy code.

## Competition Constraint

The project should not simply maximize raw return. The local research loop must
balance:

```text
return rank + drawdown rank + 15-minute Sharpe rank + risk discipline
```

The practical target is therefore:

```text
small repeatable edge * enough trades * controlled drawdown
```

## External Research Takeaways

FX and metals:

- Currency momentum has academic support. BIS Working Paper No. 366 finds broad
  evidence for profitable currency momentum strategies, though transaction costs
  and implementation details matter.
- Time-series momentum has support across asset classes including currencies and
  commodities. Moskowitz, Ooi, and Pedersen document return persistence in liquid
  futures markets, including currency and commodity futures.
- Value and momentum effects appear across markets and can diversify each other,
  but value/carry is harder to implement in this hackathon because we mostly have
  spot/tick data, not reliable interest-rate or valuation inputs.
- FX market structure is fragmented and electronic. This favors simple,
  liquidity-aware intraday rules over fragile models that assume one centralized
  order book.

Crypto:

- Crypto research finds both intraday momentum and reversal. The sign can depend
  on large jumps, liquidity, announcement windows, and market regime.
- Cross-crypto lead-lag can exist, but our downloaded archive currently does not
  include the official crypto symbols. Crypto alpha should wait until MT5/live or
  official crypto backtest data is available.

Macro/event context:

- Official central-bank calendars matter for FX. During the hackathon week, recent
  Fed, ECB, and Bank of England decisions can affect USD, EUR, and GBP volatility.
  We should treat scheduled macro windows as volatility/regime filters, not as
  unbounded news bets.

## Current Local Evidence

The larger imported sample currently uses:

```text
AUDUSD, EURUSD, GBPUSD, USDCAD, USDJPY, XAGUSD, XAUUSD
```

It covers two 15-minute-bar days from the downloaded archive:

```text
2026-05-11 to 2026-05-12
```

The full archive is richer:

```text
531 Parquet files, 22 symbols, mostly 2026-05-11 to 2026-06-10
```

Official hackathon symbols covered by the archive:

```text
AUDUSD, EURCHF, EURGBP, EURUSD, GBPUSD, USDCAD, USDCHF, USDJPY,
XAGUSD, XAUUSD
```

Official crypto symbols are not present in this downloaded archive.

Empirical clues from the imported sample:

- Metals dominate volatility. `XAGUSD` had much larger 15-minute volatility than
  FX pairs in the sample.
- Median spreads are small enough for intraday backtesting on the imported
  symbols, especially major FX and gold.
- Simple 8-bar breakout had positive one-bar follow-through on several symbols:
  strongest on `XAUUSD`, `GBPUSD`, and some EUR/USD windows.
- Mean reversion was weaker on the imported sample and harmful on several
  symbols.
- 12:00-15:00 UTC had much higher average absolute return than quiet hours in the
  sample, suggesting session/time filters may matter.
- Portfolio walk-forward selected `breakout` most often, not `alpha_router`.
- After adding primary-signal override, alpha-router became active and improved,
  but median test return was still slightly negative.

## Ranked Alpha Ideas

### 1. Session-Aware Volatility Breakout

Build first.

Logic:

- Trade only during high-opportunity sessions.
- Require channel breakout plus volatility expansion.
- Size by recent volatility.
- Use spread and quote-quality filters.
- Avoid immediately after extreme one-bar spikes if reversal risk is high.

Why:

- Local sample favors breakout more than mean reversion.
- Literature supports trend/momentum in currencies and commodities.
- Session filter can reduce overtrading in quiet hours.

### 2. Cross-Symbol USD Pressure Signal

Build second.

Logic:

- Estimate broad USD pressure from EURUSD, GBPUSD, AUDUSD, USDCAD, USDJPY, and
  USDCHF when available.
- Trade only when the target pair agrees with the broad USD basket.
- Example: EURUSD up, GBPUSD up, AUDUSD up, USDJPY down implies USD weakness.

Why:

- FX pairs are not independent.
- A shared USD factor can filter false single-symbol breakouts.
- This is implementable with the current archive.

### 3. Metal Momentum With FX Risk Filter

Build third.

Logic:

- Use XAUUSD and XAGUSD breakout/momentum.
- Reduce size when spread widens or volatility spikes too far.
- Optionally confirm with USD pressure.

Why:

- Metals have enough volatility to matter for return rank.
- The allocator can cap concentration so metals do not dominate risk discipline.

### 4. Intraday Reversal After Exhaustion

Build fourth.

Logic:

- Only trade reversal after unusually large moves.
- Require price to fail continuation, not just be statistically extended.
- Use tight max holding period.

Why:

- Crypto literature suggests intraday reversal can exist, but our FX/metals sample
  currently favors breakout more than naive reversion.
- This should be secondary, not the main alpha.

### 5. Event-Window Regime Filter

Build fifth.

Logic:

- Mark scheduled macro or central-bank windows.
- Before event: reduce or avoid new positions.
- After event: allow breakout logic to engage if spread and volatility stabilize.

Why:

- FX reacts strongly around macro events.
- This is safer than trying to predict the event direction.

## Not Recommended Yet

- Pure ML without stronger features. Current ML alpha is useful as a sleeve but
  not enough by itself.
- Carry/value. These need interest-rate and valuation inputs we do not currently
  trust in the downloaded tick archive.
- Crypto-specific models before crypto data is connected through MT5 or another
  official source.

## Next Build

Build `session_breakout` as a new strategy sleeve.

Minimum useful features:

```text
lookback channel
breakout buffer
realized volatility percentile
session allowlist
spread cap
volatility-scaled sizing
cooldown after failed breakout
```

Then add it to:

```text
strategy registry
portfolio compare
portfolio walk-forward
alpha-router signal mix
```

## Sources

- BIS Working Paper No. 366, Currency Momentum Strategies:
  https://www.bis.org/publ/work366.pdf
- Moskowitz, Ooi, Pedersen, Time Series Momentum:
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2089463
- Asness, Moskowitz, Pedersen, Value and Momentum Everywhere:
  https://pages.stern.nyu.edu/~lpederse/papers/ValMomEverywhere.pdf
- Wen, Bouri, Xu, Zhao, Intraday Return Predictability in Cryptocurrency Markets:
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4080253
- Federal Reserve FOMC calendars:
  https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
- ECB meeting calendar:
  https://www.ecb.europa.eu/press/calendars/mgcgc/html/index.en.html
- Bank of England MPC dates:
  https://www.bankofengland.co.uk/monetary-policy/upcoming-mpc-dates
