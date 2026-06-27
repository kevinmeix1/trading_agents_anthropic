from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.backtesting.backtest import BacktestEngine, FillModel
from quanthack.core.config import load_config
from quanthack.market.market_data import load_price_history, load_quote_history
from quanthack.reporting.router_report import build_router_attribution_report, write_router_attribution_csv


class RouterAttributionReportTest(TestCase):
    def test_alpha_router_backtest_fills_have_signal_attribution(self) -> None:
        result = _alpha_router_backtest()

        self.assertGreater(len(result.fills), 0)
        self.assertTrue(all(fill.primary_signal for fill in result.fills))

    def test_router_report_groups_by_primary_signal(self) -> None:
        result = _alpha_router_backtest()
        report = build_router_attribution_report(result)

        self.assertEqual(report.total_fills, len(result.fills))
        self.assertGreater(len(report.rows), 0)
        self.assertEqual(
            report.total_turnover_notional_usd,
            sum(fill.turnover_notional_usd for fill in result.fills),
        )

    def test_router_report_csv_is_written(self) -> None:
        report = build_router_attribution_report(_alpha_router_backtest())

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "router.csv"
            write_router_attribution_csv(report, path)
            text = path.read_text(encoding="utf-8")

        self.assertIn("primary_signal,fills,realized_events", text)


def _alpha_router_backtest():
    config = load_config("configs/default.toml")
    engine = BacktestEngine(
        strategy=config.build_strategy("alpha_router"),
        risk_limits=config.risk,
        quality_limits=config.market_quality,
        clock=config.competition.to_clock(),
        fill_model=FillModel(slippage_bps=config.backtest.slippage_bps),
        periods_per_year=config.backtest.periods_per_year,
    )
    return engine.run(
        prices=load_price_history(config.backtest.price_csv),
        quotes=load_quote_history(config.backtest.quote_csv),
        symbol=config.alpha_router.symbol,
        starting_equity=config.competition.starting_equity,
    )
