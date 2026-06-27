from __future__ import annotations

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, skipUnless
from zipfile import ZipFile

from quanthack.market.backtest_import import import_pricer_zip_to_backtest_csv
from quanthack.market.backtest_import import (
    inspect_pricer_zip_archive,
    write_pricer_archive_coverage_csv,
)
from quanthack.market.market_data import load_price_history, load_quote_history


HAS_PYARROW = importlib.util.find_spec("pyarrow") is not None


class PricerArchiveCoverageTest(TestCase):
    def test_inspect_pricer_zip_reports_missing_expected_symbols(self) -> None:
        with TemporaryDirectory() as tmpdir:
            archive = Path(tmpdir) / "pricer-output-sample.zip"
            _write_filename_only_pricer_zip(
                archive,
                (
                    "EURUSD_2026_05_11.parquet",
                    "EURUSD_2026_05_12.parquet",
                    "XAUUSD_2026_05_11.parquet",
                ),
            )

            summary = inspect_pricer_zip_archive(
                archive,
                expected_symbols=("EURUSD", "BTCUSD"),
            )

        self.assertEqual(summary.files_seen, 3)
        self.assertEqual(summary.available_symbols, ("EURUSD", "XAUUSD"))
        self.assertEqual(summary.present_expected_symbols, ("EURUSD",))
        self.assertEqual(summary.missing_symbols, ("BTCUSD",))
        self.assertEqual(summary.extra_symbols, ("XAUUSD",))
        self.assertFalse(summary.complete)

    def test_write_pricer_archive_coverage_csv(self) -> None:
        with TemporaryDirectory() as tmpdir:
            archive = Path(tmpdir) / "pricer-output-sample.zip"
            output = Path(tmpdir) / "coverage.csv"
            _write_filename_only_pricer_zip(
                archive,
                ("EURUSD_2026_05_11.parquet",),
            )
            summary = inspect_pricer_zip_archive(
                archive,
                expected_symbols=("EURUSD", "BTCUSD"),
            )

            write_pricer_archive_coverage_csv(summary, output)
            text = output.read_text(encoding="utf-8")

        self.assertIn("symbol,status,file_count,first_file,last_file", text)
        self.assertIn("EURUSD,EXPECTED,1,EURUSD_2026_05_11.parquet", text)
        self.assertIn("BTCUSD,MISSING,0,,", text)


@skipUnless(HAS_PYARROW, "pyarrow is needed for Parquet importer tests")
class BacktestImportTest(TestCase):
    def test_import_pricer_zip_resamples_ticks_to_backtest_csvs(self) -> None:
        with TemporaryDirectory() as tmpdir:
            archive = Path(tmpdir) / "pricer-output-sample.zip"
            price_output = Path(tmpdir) / "prices.csv"
            quote_output = Path(tmpdir) / "quotes.csv"
            _write_sample_pricer_zip(archive)

            summary = import_pricer_zip_to_backtest_csv(
                input_path=archive,
                price_output=price_output,
                quote_output=quote_output,
                symbols=("EURUSD",),
                bar_seconds=900,
            )
            prices = load_price_history(price_output)
            quotes = load_quote_history(quote_output)

        self.assertEqual(summary.symbols, ("EURUSD",))
        self.assertEqual(summary.files_seen, 2)
        self.assertEqual(summary.files_imported, 1)
        self.assertEqual(summary.ticks_seen, 4)
        self.assertEqual(summary.bars_written, 3)
        self.assertEqual(len(prices.for_symbol("EURUSD").bars), 3)
        self.assertEqual(len(quotes.for_symbol("EURUSD").quotes), 3)
        self.assertAlmostEqual(prices.close_prices(symbol="EURUSD")[0], 1.10015)

    def test_import_pricer_zip_skips_null_bid_or_ask_rows(self) -> None:
        with TemporaryDirectory() as tmpdir:
            archive = Path(tmpdir) / "pricer-output-null-rows.zip"
            price_output = Path(tmpdir) / "prices.csv"
            quote_output = Path(tmpdir) / "quotes.csv"
            _write_null_quote_pricer_zip(archive)

            summary = import_pricer_zip_to_backtest_csv(
                input_path=archive,
                price_output=price_output,
                quote_output=quote_output,
                symbols=("EURUSD",),
                bar_seconds=900,
            )
            prices = load_price_history(price_output)

        self.assertEqual(summary.ticks_seen, 1)
        self.assertEqual(summary.bars_written, 1)
        self.assertEqual(prices.close_prices(symbol="EURUSD"), [1.1003])

    def test_import_pricer_zip_reports_progress(self) -> None:
        with TemporaryDirectory() as tmpdir:
            archive = Path(tmpdir) / "pricer-output-sample.zip"
            price_output = Path(tmpdir) / "prices.csv"
            quote_output = Path(tmpdir) / "quotes.csv"
            messages: list[str] = []
            _write_sample_pricer_zip(archive)

            summary = import_pricer_zip_to_backtest_csv(
                input_path=archive,
                price_output=price_output,
                quote_output=quote_output,
                symbols=("EURUSD",),
                bar_seconds=900,
                progress_every_files=1,
                progress_callback=messages.append,
            )

        self.assertEqual(summary.files_imported, 1)
        self.assertTrue(any("archive contains" in message for message in messages))
        self.assertTrue(any("imported 1 files" in message for message in messages))
        self.assertTrue(any("finished import" in message for message in messages))

    def test_import_pricer_zip_rejects_empty_archive_file(self) -> None:
        with TemporaryDirectory() as tmpdir:
            archive = Path(tmpdir) / "empty.zip"
            archive.write_bytes(b"")

            with self.assertRaisesRegex(ValueError, "empty"):
                import_pricer_zip_to_backtest_csv(
                    input_path=archive,
                    price_output=Path(tmpdir) / "prices.csv",
                    quote_output=Path(tmpdir) / "quotes.csv",
                    symbols=("EURUSD",),
                )


def _write_sample_pricer_zip(path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    eurusd = pa.table(
        {
            "time": [
                "2026-05-11 00:01:00.000000",
                "2026-05-11 00:05:00.000000",
                "2026-05-11 00:16:00.000000",
                "2026-05-11 00:31:00.000000",
            ],
            "sym": ["EURUSD"] * 4,
            "bid": [1.1000, 1.1001, 1.1010, 1.1020],
            "ask": [1.1002, 1.1002, 1.1012, 1.1022],
        }
    )
    xauusd = pa.table(
        {
            "time": ["2026-05-11 00:01:00.000000"],
            "sym": ["XAUUSD"],
            "bid": [2320.0],
            "ask": [2320.2],
        }
    )
    eurusd_path = path.parent / "EURUSD_2026_05_11.parquet"
    xauusd_path = path.parent / "XAUUSD_2026_05_11.parquet"
    pq.write_table(eurusd, eurusd_path)
    pq.write_table(xauusd, xauusd_path)
    with ZipFile(path, "w") as archive:
        archive.write(eurusd_path, eurusd_path.name)
        archive.write(xauusd_path, xauusd_path.name)


def _write_null_quote_pricer_zip(path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table(
        {
            "time": [
                "2026-05-11 00:01:00.000000",
                "2026-05-11 00:02:00.000000",
                "2026-05-11 00:03:00.000000",
            ],
            "sym": ["EURUSD"] * 3,
            "bid": [None, 1.1000, 1.1002],
            "ask": [1.1002, None, 1.1004],
        }
    )
    parquet_path = path.parent / "EURUSD_2026_05_11.parquet"
    pq.write_table(table, parquet_path)
    with ZipFile(path, "w") as archive:
        archive.write(parquet_path, parquet_path.name)


def _write_filename_only_pricer_zip(path: Path, names: tuple[str, ...]) -> None:
    with ZipFile(path, "w") as archive:
        for name in names:
            archive.writestr(name, b"not actually parquet")
