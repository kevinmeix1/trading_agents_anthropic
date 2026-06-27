# Downloaded Backtest Data

The downloaded competition backtest file in `~/Downloads` is a large zip archive
of tick-level Parquet files. Each file is one symbol-day, for example:

```text
EURUSD_2026_05_11.parquet
XAUUSD_2026_05_11.parquet
```

The Parquet rows contain tick quotes:

```text
time, sym, bid, ask, ...
```

The QuanHack backtest engine uses normalized bar CSVs:

```text
timestamp,symbol,close
timestamp,symbol,bid,ask
```

So the workflow is:

```text
downloaded Parquet zip -> import-backtest-data -> normalized CSV -> backtest
```

## Install Data Support

In VS Code terminal:

```bash
cd ~/Desktop/quanthack
source .venv/bin/activate
python -m pip install -e ".[data]"
```

This installs `pyarrow`, which reads Parquet files.

## Quick Sample Import

Use only a couple of symbol-days first. This proves the pipeline without waiting
for the full 20GB archive to be processed.

```bash
quanthack import-backtest-data \
  --input ~/Downloads/pricer-output-2026-05-11_2026-06-10.zip \
  --symbol EURUSD \
  --max-files-per-symbol 2 \
  --progress-every-files 1 \
  --price-output data/downloaded_backtest_prices.csv \
  --quote-output data/downloaded_backtest_quotes.csv
```

Then run a backtest on the imported bars:

```bash
quanthack backtest \
  --strategy simple_momentum \
  --symbol EURUSD \
  --price-csv data/downloaded_backtest_prices.csv \
  --quote-csv data/downloaded_backtest_quotes.csv
```

Validate 15-minute imported bars with a matching gap threshold:

```bash
quanthack validate-data \
  --price-csv data/downloaded_backtest_prices.csv \
  --quote-csv data/downloaded_backtest_quotes.csv \
  --symbol EURUSD \
  --max-gap-seconds 900
```

## Full Symbol Import

After the sample works, remove `--max-files-per-symbol`:

```bash
quanthack import-backtest-data \
  --input ~/Downloads/pricer-output-2026-05-11_2026-06-10.zip \
  --symbol EURUSD \
  --progress-every-files 25 \
  --price-output data/downloaded_backtest_prices.csv \
  --quote-output data/downloaded_backtest_quotes.csv
```

You can import more than one symbol into the same output CSV:

```bash
quanthack import-backtest-data \
  --input ~/Downloads/pricer-output-2026-05-11_2026-06-10.zip \
  --symbol EURUSD \
  --symbol XAUUSD \
  --symbol USDJPY
```

Then run portfolio workflows using those same CSVs:

```bash
quanthack portfolio-backtest \
  --strategy alpha_router \
  --symbol EURUSD \
  --symbol XAUUSD \
  --symbol USDJPY \
  --price-csv data/downloaded_backtest_prices.csv \
  --quote-csv data/downloaded_backtest_quotes.csv
```

Before committing to one detailed portfolio backtest, scan diversified baskets:

```bash
quanthack portfolio-universe-scan \
  --strategy alpha_router \
  --strategy ma_crossover \
  --price-csv data/downloaded_backtest_prices.csv \
  --quote-csv data/downloaded_backtest_quotes.csv
```

Then validate whether train-window selections survive later test windows:

```bash
quanthack portfolio-walk-forward \
  --strategy alpha_router \
  --strategy ma_crossover \
  --price-csv data/downloaded_backtest_prices.csv \
  --quote-csv data/downloaded_backtest_quotes.csv \
  --train-size 96 \
  --test-size 48 \
  --step-size 48
```

After choosing a basket, tune alpha-router weights inside walk-forward:

```bash
quanthack portfolio-router-walk-forward \
  --symbol EURUSD --symbol USDJPY --symbol XAUUSD \
  --candidate 0.30,0.15,0.15,0.35,0.25,0.00 \
  --candidate 0.20,0.10,0.10,0.40,0.15,0.05,0.10 \
  --price-csv data/downloaded_backtest_prices.csv \
  --quote-csv data/downloaded_backtest_quotes.csv \
  --train-size 96 \
  --test-size 48 \
  --step-size 48
```

## Notes

- The importer defaults to 15-minute bars: `--bar-seconds 900`.
- Raw tick timestamps are treated as UTC by default: `--source-timezone UTC`.
- The importer reads selected symbols only and processes one Parquet file at a
  time, so it does not load the 20GB archive into memory.
- The importer skips rows with missing or invalid bid/ask quotes.
- Use `--progress-every-files 25` for large imports so the terminal reports file,
  tick, and bar counts while it works. Use `--progress-every-files 0` to silence
  progress output.
- Generated downloaded-data CSVs are ignored by git.
