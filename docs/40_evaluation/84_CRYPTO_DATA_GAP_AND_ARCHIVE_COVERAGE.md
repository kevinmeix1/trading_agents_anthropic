# Crypto Data Gap And Archive Coverage

Date: 2026-06-22

## Why This Matters

The current strongest candidate, strict all-MACD, performs well when it trades
but has many flat walk-forward folds. The competition includes crypto symbols,
which can trade through weekends and quiet FX windows. Crypto data is therefore
the highest-priority data gap for improving positive round frequency.

## New Tool

Added a fast archive coverage checker:

```bash
PYTHONPATH=src python -c "from quanthack.cli.archive_data_coverage import main; main()" \
  --input /Users/kaiwenmei/Downloads/pricer-output-2026-05-11_2026-06-10.zip \
  --competition-symbols \
  --output outputs/research/archive_competition_coverage.csv
```

It inspects zip filenames only, so it does not need `pyarrow` and does not read
large parquet payloads.

## Actual Downloaded Archive

Input:

- `/Users/kaiwenmei/Downloads/pricer-output-2026-05-11_2026-06-10.zip`

Result:

- parquet files: `531`
- available symbols: `22`
- official competition symbols expected: `15`
- official symbols present: `10`
- official symbols missing: `5`

Present official symbols:

- `AUDUSD`
- `EURCHF`
- `EURGBP`
- `EURUSD`
- `GBPUSD`
- `USDCAD`
- `USDCHF`
- `USDJPY`
- `XAGUSD`
- `XAUUSD`

Missing official crypto symbols:

- `BARUSD`
- `BTCUSD`
- `ETHUSD`
- `SOLUSD`
- `XRPUSD`

The archive also contains extra non-official or currently unsupported symbols
such as `AUDJPY`, `AUDNZD`, `EURJPY`, `NZDUSD`, oil contracts, and non-USD gold
crosses. Those may be useful for research ideas, but they cannot be assumed to
be tradable/scored unless the official competition universe allows them.

## Conclusion

The crypto gap is not caused by the importer. The source archive does not
contain the official crypto instruments. To backtest crypto alpha, we need one
of:

1. a new organizer/pricer download that includes `BARUSD`, `BTCUSD`, `ETHUSD`,
   `SOLUSD`, and `XRPUSD`;
2. MT5 read-only capture of crypto quotes once the Windows/MT5 environment is
   ready;
3. a separate verified historical source, clearly marked as research-only if it
   does not match the competition feed.

Until then, strict all-MACD remains the best validated paper candidate on the
available official data, but it is incomplete for the full hackathon universe.
