from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.backtesting.signal_diagnostics import (
    evaluate_signal_diagnostics,
    write_signal_diagnostics_csv,
)
from quanthack.core.config import load_config
from quanthack.market.market_data import PriceBar, PriceHistory, QuoteHistory, QuoteSnapshot
from quanthack.strategies.strategy import (
    AlphaRouterConfig,
    BreakoutConfig,
    CrossRateReversionConfig,
    MeanReversionConfig,
    MomentumConfig,
    MovingAverageCrossoverConfig,
    SessionBreakoutConfig,
)


class SignalDiagnosticsTest(TestCase):
    def test_evaluates_router_signal_forward_returns_without_backtest(self) -> None:
        config = _diagnostic_config()
        prices, quotes = _upward_market()

        report = evaluate_signal_diagnostics(
            config=config,
            prices=prices,
            quotes=quotes,
            strategy_name="alpha_router",
            symbols=("EURUSD",),
            horizon_bars=1,
            min_confidence=0.0,
        )

        rows = {(row.symbol, row.signal_name): row for row in report.rows}
        self.assertIn(("EURUSD", "momentum"), rows)
        self.assertIn(("EURUSD", "session_breakout"), rows)
        self.assertGreater(rows[("EURUSD", "momentum")].active_count, 0)
        self.assertGreater(
            rows[("EURUSD", "momentum")].average_signed_forward_return_bps,
            0.0,
        )

    def test_writes_signal_diagnostics_csv(self) -> None:
        report = evaluate_signal_diagnostics(
            config=_diagnostic_config(),
            prices=_upward_market()[0],
            quotes=_upward_market()[1],
            strategy_name="alpha_router",
            symbols=("EURUSD",),
            horizon_bars=1,
            min_confidence=0.0,
        )

        with TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "signals.csv"
            write_signal_diagnostics_csv(report, output)
            text = output.read_text(encoding="utf-8")

        self.assertIn("strategy,horizon_bars,symbol,signal", text)
        self.assertIn("average_weight", text)
        self.assertIn("alpha_router,1,EURUSD,momentum", text)

    def test_cross_rate_diagnostics_receives_portfolio_context(self) -> None:
        config = replace(
            load_config("configs/default.toml"),
            cross_rate_reversion=CrossRateReversionConfig(
                symbol="EURGBP",
                lookback=4,
                entry_zscore=1.0,
                min_abs_deviation_bps=1.0,
                max_abs_deviation_bps=400.0,
                slippage_bps=0.0,
                max_spread_bps=100.0,
            ),
        )
        prices, quotes = _cross_rate_market()

        report = evaluate_signal_diagnostics(
            config=config,
            prices=prices,
            quotes=quotes,
            strategy_name="cross_rate_reversion",
            symbols=("EURGBP",),
            horizon_bars=1,
            min_confidence=0.0,
        )

        row = report.rows[0]
        self.assertEqual(row.signal_name, "cross_rate_reversion")
        self.assertGreater(row.active_count, 0)
        self.assertGreater(row.average_signed_forward_return_bps, 0.0)


def _diagnostic_config():
    config = load_config("configs/default.toml")
    return replace(
        config,
        simple_momentum=MomentumConfig(
            lookback=3,
            threshold_bps=1.0,
            min_trend_efficiency=0.1,
            slippage_bps=0.0,
        ),
        ma_crossover=MovingAverageCrossoverConfig(
            fast_window=2,
            slow_window=4,
            min_separation_bps=999.0,
            slippage_bps=0.0,
        ),
        breakout=BreakoutConfig(
            lookback=4,
            breakout_buffer_bps=1.0,
            min_channel_width_bps=0.0,
            slippage_bps=0.0,
        ),
        session_breakout=SessionBreakoutConfig(
            lookback=4,
            breakout_buffer_bps=1.0,
            min_expected_edge_bps=1.0,
            min_channel_width_bps=0.0,
            min_realized_volatility_bps=0.01,
            allowed_utc_hours=(12,),
            slippage_bps=0.0,
            max_spread_bps=100.0,
        ),
        mean_reversion=MeanReversionConfig(
            entry_zscore=10.0,
            stop_zscore=20.0,
            slippage_bps=0.0,
        ),
        alpha_router=AlphaRouterConfig(
            entry_score=0.10,
            exit_score=0.02,
            min_signal_confidence=0.0,
            cost_buffer=1.0,
            max_spread_bps=100.0,
            momentum_weight=0.40,
            moving_average_weight=0.0,
            breakout_weight=0.20,
            session_breakout_weight=0.20,
            mean_reversion_weight=0.0,
            primary_signal_override_enabled=False,
        ),
    )


def _upward_market() -> tuple[PriceHistory, QuoteHistory]:
    start = datetime(2026, 6, 22, 12, 0, tzinfo=UTC)
    bars: list[PriceBar] = []
    quotes: list[QuoteSnapshot] = []
    for index in range(12):
        timestamp = start + timedelta(minutes=15 * index)
        close = 1.1000 + (index * 0.0004)
        bars.append(PriceBar(timestamp=timestamp, symbol="EURUSD", close=close))
        quotes.append(
            QuoteSnapshot(
                timestamp=timestamp,
                symbol="EURUSD",
                bid=close - 0.00001,
                ask=close + 0.00001,
            )
        )
    return PriceHistory(tuple(bars)), QuoteHistory(tuple(quotes))


def _cross_rate_market() -> tuple[PriceHistory, QuoteHistory]:
    start = datetime(2026, 6, 22, 12, 0, tzinfo=UTC)
    series = {
        "EURUSD": [1.1000, 1.1000, 1.1000, 1.1000, 1.1000, 1.1000],
        "GBPUSD": [1.2500, 1.2500, 1.2500, 1.2500, 1.2500, 1.2500],
        "EURGBP": [0.8798, 0.8801, 0.8799, 0.9000, 0.8850, 0.8800],
    }
    bars: list[PriceBar] = []
    quotes: list[QuoteSnapshot] = []
    for index in range(6):
        timestamp = start + timedelta(minutes=15 * index)
        for symbol, closes in series.items():
            close = closes[index]
            bars.append(PriceBar(timestamp=timestamp, symbol=symbol, close=close))
            quotes.append(
                QuoteSnapshot(
                    timestamp=timestamp,
                    symbol=symbol,
                    bid=close - 0.00001,
                    ask=close + 0.00001,
                )
            )
    return PriceHistory(tuple(bars)), QuoteHistory(tuple(quotes))
