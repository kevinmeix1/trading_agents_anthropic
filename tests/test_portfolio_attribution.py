from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.backtesting.backtest import BacktestFill
from quanthack.backtesting.metrics import PerformanceMetrics
from quanthack.backtesting.pnl import build_pnl_ledger
from quanthack.backtesting.portfolio_backtest import PortfolioBacktestResult, SymbolPnlLedger
from quanthack.reporting.portfolio_attribution import (
    build_portfolio_attribution_report,
    write_portfolio_attribution_csv,
)
from quanthack.trading.risk import Side


class PortfolioAttributionReportTest(TestCase):
    def test_groups_portfolio_realized_pnl_by_symbol_signal_hour_and_side(self) -> None:
        result = _portfolio_result(
            (
                _fill("EURUSD", "2026-06-22T10:00:00+00:00", Side.BUY, 100.0, 10.0),
                _fill("EURUSD", "2026-06-22T11:00:00+00:00", Side.SELL, 105.0, -10.0),
                _fill("XAUUSD", "2026-06-22T11:30:00+00:00", Side.SELL, 200.0, -2.0),
                _fill("XAUUSD", "2026-06-22T12:00:00+00:00", Side.BUY, 190.0, 2.0),
            )
        )

        report = build_portfolio_attribution_report(result)
        rows = {
            (row.symbol, row.utc_hour, row.side): row
            for row in report.rows
        }

        self.assertEqual(report.total_fills, 4)
        self.assertEqual(report.total_realized_pnl_usd, 70.0)
        self.assertEqual(rows[("EURUSD", 10, "BUY")].realized_pnl_usd, 50.0)
        self.assertEqual(rows[("XAUUSD", 11, "SELL")].realized_pnl_usd, 20.0)

    def test_writes_portfolio_attribution_csv(self) -> None:
        report = build_portfolio_attribution_report(
            _portfolio_result(
                (
                    _fill("EURUSD", "2026-06-22T10:00:00+00:00", Side.BUY, 100.0, 10.0),
                    _fill("EURUSD", "2026-06-22T11:00:00+00:00", Side.SELL, 101.0, -10.0),
                )
            )
        )

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "portfolio_attribution.csv"
            write_portfolio_attribution_csv(report, path)
            text = path.read_text(encoding="utf-8")

        self.assertIn("symbol,primary_signal,utc_hour,side", text)
        self.assertIn("EURUSD", text)


def _portfolio_result(fills: tuple[BacktestFill, ...]) -> PortfolioBacktestResult:
    symbols = tuple(dict.fromkeys(fill.symbol for fill in fills))
    return PortfolioBacktestResult(
        symbols=symbols,
        equity_curve=(),
        fills=fills,
        metrics=PerformanceMetrics(
            observations=0,
            final_equity=1_000_000.0,
            total_return_pct=0.0,
            sharpe_ratio=0.0,
            max_drawdown_pct=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            turnover_notional=sum(fill.turnover_notional_usd for fill in fills),
        ),
        pnl_by_symbol=tuple(
            SymbolPnlLedger(
                symbol=symbol,
                ledger=build_pnl_ledger(
                    tuple(fill for fill in fills if fill.symbol == symbol)
                ),
            )
            for symbol in symbols
        ),
    )


def _fill(
    symbol: str,
    timestamp: str,
    side: Side,
    price: float,
    units: float,
) -> BacktestFill:
    return BacktestFill(
        timestamp=timestamp,
        symbol=symbol,
        side=side,
        fill_price=price,
        trade_units=units,
        requested_notional_usd=abs(price * units),
        adjusted_notional_usd=abs(price * units),
        risk_reason="approved",
        primary_signal="test_signal",
    )
