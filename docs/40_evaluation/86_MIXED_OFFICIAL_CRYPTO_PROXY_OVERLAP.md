# Mixed Official And Crypto Proxy Overlap

Date: 2026-06-22

## Question

Can the crypto research sleeve improve the current official-data MACD candidate
when tested together with FX/metals over the same timestamps?

The downloaded official archive covers `2026-05-11` through `2026-06-10`.
The research crypto proxy covers `2026-06-07` through `2026-06-21`.
The shared window is therefore small:

- `2026-06-07T23:45:00+00:00` to `2026-06-10T23:45:00+00:00`

This is not enough to promote a live candidate by itself, but it is enough to
test whether mixed-universe plumbing and risk behavior work.

## New Merge Tool

Added:

- `src/quanthack/market/merge_market_data.py`
- `src/quanthack/cli/merge_market_data.py`
- console script: `quanthack-merge-market-data`

Example:

```bash
PYTHONPATH=src python -c "from quanthack.cli.merge_market_data import main; main()" \
  --price-input data/full_20gb_15m_prices.csv \
  --price-input data/research_crypto_proxy_14d_prices.csv \
  --quote-input data/full_20gb_15m_quotes.csv \
  --quote-input data/research_crypto_proxy_14d_quotes.csv \
  --price-output data/mixed_official_crypto_proxy_overlap_prices.csv \
  --quote-output data/mixed_official_crypto_proxy_overlap_quotes.csv \
  --common-window
```

Output:

- symbols: all 15 official instruments
- price rows: `4,317`
- quote rows: `4,317`
- window: `2026-06-07T23:45:00+00:00` to `2026-06-10T23:45:00+00:00`

Data validation status was `WARN`, not `FAIL`. The warnings were small CHF
spread touches and expected metal session gaps.

## Apples-To-Apples Backtest

Official 10-symbol MACD over the overlap:

- return: `+0.718%`
- max drawdown: `0.727%`
- official 15m Sharpe view: `0.032`
- fills: `14`
- risk discipline score: `100/100`

Mixed 15-symbol MACD over the same overlap:

- return: `+0.970%`
- max drawdown: `2.439%`
- official 15m Sharpe view: `0.018`
- fills: `32`
- risk discipline score: `100/100`

Mixed 15-symbol MACD with volatility targeting:

- return: `-0.296%`
- max drawdown: `0.526%`
- official 15m Sharpe view: `-0.033`
- fills: `61`
- risk discipline score: `100/100`

## Interpretation

Adding crypto improved raw return and pushed the run over the 30-trade Sharpe
eligibility threshold, but it worsened drawdown and short-window Sharpe. The
vol-targeted version reduced drawdown but overreacted on this short sample and
lost money.

## Asset-Class Attribution

I added an asset-class attribution report:

```bash
PYTHONPATH=src python -c "from quanthack.cli.asset_class_attribution import main; main()" \
  --pnl-csv outputs/research/overlap_mixed15_macd_pnl.csv \
  --output outputs/research/overlap_mixed15_macd_asset_class_attribution.csv
```

For the mixed 15-symbol MACD run:

- `METAL`: `+$6,410`, 6 fills, 2 winners / 0 losers, `66.1%` of net P&L
- `CRYPTO`: `+$3,287`, 18 fills, 3 winners / 2 losers, `33.9%` of net P&L
- `FOREX`: `+$4`, 8 fills, 2 winners / 2 losers, roughly flat net P&L

The absolute-impact view is also important: crypto contributed `49.2%` of gross
absolute P&L movement, so it is not a tiny add-on. It materially changes the
portfolio's risk/return profile.

## Crypto Strategy Variants

I tested keeping FX/metals on MACD while changing crypto only:

- crypto `macd_momentum`: `+0.970%` return, `2.439%` max drawdown, 32 fills,
  risk score `100/100`
- crypto `quality_trend`: `+0.503%` return, `1.408%` max drawdown, 22 fills,
  risk score `100/100`
- crypto `multi_horizon_momentum`: `-0.012%` return, `2.420%` max drawdown,
  50 fills, risk score `100/100`
- crypto `champion_ensemble`: `+0.047%` return, `1.841%` max drawdown, 22
  fills, risk score `100/100`

Asset-class attribution shows why MACD remains the best tested crypto sleeve in
this narrow overlap:

- quality-trend crypto P&L: `-$2,143`
- multi-horizon crypto P&L: `-$3,402`
- champion crypto P&L: `-$6,701`
- MACD crypto P&L: `+$3,287`

This supports a cautious view:

- keep strict all-MACD as the best official-data candidate;
- keep crypto as a research sleeve for flat/weekend coverage;
- do not promote mixed crypto exposure until it is validated on MT5-captured
  competition quotes or a longer official crypto history;
- avoid volatility targeting on this sleeve unless future walk-forward evidence
  shows it improves score, not just smoothness.
