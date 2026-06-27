from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from zoneinfo import ZoneInfo

from quanthack.core.config import load_config
from quanthack.core.instruments import DEFAULT_INSTRUMENTS
from quanthack.market.market_data import PriceBar, PriceHistory, QuoteHistory, QuoteSnapshot
from quanthack.reporting.hackathon_readiness import (
    ReadinessStatus,
    build_hackathon_readiness_report,
    write_hackathon_readiness_markdown,
)


LONDON = ZoneInfo("Europe/London")


class HackathonReadinessTest(TestCase):
    def test_full_data_and_promoted_candidate_can_pass(self) -> None:
        config = load_config("configs/default.toml")
        with TemporaryDirectory() as tmpdir:
            promotion_path = Path(tmpdir) / "promotion.csv"
            summary_path = Path(tmpdir) / "summary.csv"
            _write_promotion_csv(promotion_path, status="PROMOTE", live_ready="yes")
            _write_summary_csv(summary_path)

            report = build_hackathon_readiness_report(
                config=config,
                prices=_history_for_symbols(tuple(i.symbol for i in DEFAULT_INSTRUMENTS)),
                quotes=_quotes_for_symbols(tuple(i.symbol for i in DEFAULT_INSTRUMENTS)),
                promotion_csv=promotion_path,
                summary_csv=summary_path,
            )

        self.assertEqual(report.overall_status, ReadinessStatus.PASS)
        self.assertTrue(report.ready_for_live)
        self.assertEqual(report.coverage.missing_common_symbols, ())
        self.assertEqual(report.coverage.missing_asset_classes, ())
        self.assertEqual(report.promotion.status if report.promotion else "", "PROMOTE")

    def test_missing_crypto_data_blocks_full_hackathon_readiness(self) -> None:
        config = load_config("configs/default.toml")
        symbols = ("EURUSD", "XAUUSD")

        report = build_hackathon_readiness_report(
            config=config,
            prices=_history_for_symbols(symbols),
            quotes=_quotes_for_symbols(symbols),
        )

        self.assertEqual(report.overall_status, ReadinessStatus.FAIL)
        self.assertIn("CRYPTO", report.summary_lines()[3])
        self.assertTrue(
            any(
                check.name == "asset-class coverage"
                and check.status == ReadinessStatus.FAIL
                for check in report.checks
            )
        )

    def test_paper_only_candidate_is_warning_when_data_is_complete(self) -> None:
        config = load_config("configs/default.toml")
        with TemporaryDirectory() as tmpdir:
            promotion_path = Path(tmpdir) / "promotion.csv"
            summary_path = Path(tmpdir) / "summary.csv"
            _write_promotion_csv(promotion_path, status="PAPER_ONLY", live_ready="no")
            _write_summary_csv(summary_path)

            report = build_hackathon_readiness_report(
                config=config,
                prices=_history_for_symbols(tuple(i.symbol for i in DEFAULT_INSTRUMENTS)),
                quotes=_quotes_for_symbols(tuple(i.symbol for i in DEFAULT_INSTRUMENTS)),
                promotion_csv=promotion_path,
                summary_csv=summary_path,
            )

        self.assertEqual(report.overall_status, ReadinessStatus.WARN)
        self.assertFalse(report.ready_for_live)
        self.assertTrue(
            any(
                check.name == "candidate promotion"
                and check.status == ReadinessStatus.WARN
                for check in report.checks
            )
        )

    def test_writes_markdown_report(self) -> None:
        config = load_config("configs/default.toml")
        report = build_hackathon_readiness_report(
            config=config,
            prices=_history_for_symbols(("EURUSD", "XAUUSD")),
            quotes=_quotes_for_symbols(("EURUSD", "XAUUSD")),
        )
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "readiness.md"
            write_hackathon_readiness_markdown(report, output_path)
            text = output_path.read_text(encoding="utf-8")

        self.assertIn("# Hackathon Readiness Report", text)
        self.assertIn("Overall status:", text)
        self.assertIn("Missing asset classes", text)


def _history_for_symbols(symbols: tuple[str, ...]) -> PriceHistory:
    bars = []
    for symbol_index, symbol in enumerate(symbols):
        base = 1.0 + (symbol_index * 0.01)
        for index in range(2):
            bars.append(
                PriceBar(
                    timestamp=_time(index),
                    symbol=symbol,
                    close=base + (index * 0.001),
                )
            )
    return PriceHistory(tuple(bars))


def _quotes_for_symbols(symbols: tuple[str, ...]) -> QuoteHistory:
    quotes = []
    for symbol_index, symbol in enumerate(symbols):
        base = 1.0 + (symbol_index * 0.01)
        for index in range(2):
            mid = base + (index * 0.001)
            quotes.append(
                QuoteSnapshot(
                    timestamp=_time(index),
                    symbol=symbol,
                    bid=mid - 0.0001,
                    ask=mid + 0.0001,
                )
            )
    return QuoteHistory(tuple(quotes))


def _time(index: int) -> datetime:
    return datetime(2026, 6, 22, 10, 0, tzinfo=LONDON) + timedelta(minutes=15 * index)


def _write_promotion_csv(path: Path, *, status: str, live_ready: str) -> None:
    path.write_text(
        "status,live_ready,decision_reason,gate_id,category,passed,value,threshold,comparator,gap,details\n"
        f"{status},{live_ready},test decision,folds_present,research,yes,1,1,>=,0,ok\n",
        encoding="utf-8",
    )


def _write_summary_csv(path: Path) -> None:
    path.write_text(
        "strategies,symbols,folds,positive_fold_fraction,active_positive_fold_fraction,"
        "non_negative_fold_fraction,median_active_test_return_pct,worst_test_drawdown_pct,"
        "average_risk_discipline_score,total_evaluation_fills\n"
        "test,EURUSD,4,0.75,0.75,1.0,0.001,0.01,100,40\n",
        encoding="utf-8",
    )
