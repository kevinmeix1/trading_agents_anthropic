from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from quanthack.backtesting.backtest import BacktestEngine, FillModel, write_equity_curve_csv
from quanthack.core.clock import CompetitionClock
from quanthack.core.config import load_config
from quanthack.market.market_data import (
    PriceBar,
    PriceHistory,
    QuoteHistory,
    QuoteSnapshot,
    load_price_history,
    load_quote_history,
)
from quanthack.market.market_quality import MarketQualityLimits
from quanthack.backtesting.pnl import write_pnl_ledger_csv
from quanthack.strategies.strategy import (
    SimpleMomentumStrategy,
    StrategyAction,
    StrategyDecision,
)
from quanthack.trading.risk import RiskLimits


LONDON = ZoneInfo("Europe/London")


class BacktestTest(TestCase):
    def test_backtest_runs_and_generates_metrics(self) -> None:
        config = load_config("configs/default.toml")
        engine = BacktestEngine(
            strategy=SimpleMomentumStrategy(config.simple_momentum),
            risk_limits=config.risk,
            quality_limits=config.market_quality,
            clock=config.competition.to_clock(),
            fill_model=FillModel(slippage_bps=config.backtest.slippage_bps),
            periods_per_year=config.backtest.periods_per_year,
        )

        result = engine.run(
            prices=load_price_history(config.backtest.price_csv),
            quotes=load_quote_history(config.backtest.quote_csv),
            symbol=config.simple_momentum.symbol,
            starting_equity=config.competition.starting_equity,
        )

        self.assertEqual(result.symbol, "EURUSD")
        self.assertEqual(len(result.equity_curve), 20)
        self.assertGreater(len(result.fills), 0)
        self.assertEqual(result.metrics.observations, 20)
        self.assertGreater(result.metrics.turnover_notional, 0)
        self.assertEqual(len(result.pnl_ledger.events), len(result.fills))
        self.assertAlmostEqual(
            result.pnl_ledger.total_pnl_usd,
            result.metrics.final_equity - config.competition.starting_equity,
            places=6,
        )

    def test_equity_curve_csv_is_written(self) -> None:
        config = load_config("configs/default.toml")
        engine = BacktestEngine(
            strategy=SimpleMomentumStrategy(config.simple_momentum),
            risk_limits=config.risk,
            quality_limits=config.market_quality,
            clock=config.competition.to_clock(),
        )
        result = engine.run(
            prices=load_price_history(config.backtest.price_csv),
            quotes=load_quote_history(config.backtest.quote_csv),
            symbol=config.simple_momentum.symbol,
            starting_equity=config.competition.starting_equity,
        )

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "equity.csv"
            write_equity_curve_csv(result, path)

            text = path.read_text(encoding="utf-8")

        self.assertIn("timestamp,close,equity", text)
        self.assertIn("position_notional_usd", text)

    def test_pnl_ledger_csv_is_written(self) -> None:
        config = load_config("configs/default.toml")
        engine = BacktestEngine(
            strategy=SimpleMomentumStrategy(config.simple_momentum),
            risk_limits=config.risk,
            quality_limits=config.market_quality,
            clock=config.competition.to_clock(),
        )
        result = engine.run(
            prices=load_price_history(config.backtest.price_csv),
            quotes=load_quote_history(config.backtest.quote_csv),
            symbol=config.simple_momentum.symbol,
            starting_equity=config.competition.starting_equity,
        )

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pnl.csv"
            write_pnl_ledger_csv(result.pnl_ledger, path)
            text = path.read_text(encoding="utf-8")

        self.assertIn("realized_pnl_usd", text)
        self.assertIn("position_units_after", text)

    def test_backtest_accepts_strategy_from_registry(self) -> None:
        config = load_config("configs/default.toml")
        engine = BacktestEngine(
            strategy=config.build_strategy("mean_reversion"),
            risk_limits=config.risk,
            quality_limits=config.market_quality,
            clock=config.competition.to_clock(),
            fill_model=FillModel(slippage_bps=config.backtest.slippage_bps),
            periods_per_year=config.backtest.periods_per_year,
        )

        result = engine.run(
            prices=load_price_history(config.backtest.price_csv),
            quotes=load_quote_history(config.backtest.quote_csv),
            symbol=config.strategy_symbol("mean_reversion"),
            starting_equity=config.competition.starting_equity,
        )

        self.assertEqual(result.symbol, "EURUSD")
        self.assertEqual(result.metrics.observations, 20)

    def test_missing_quote_fails_loudly(self) -> None:
        config = load_config("configs/default.toml")
        engine = BacktestEngine(
            strategy=SimpleMomentumStrategy(config.simple_momentum),
            risk_limits=config.risk,
            quality_limits=config.market_quality,
            clock=config.competition.to_clock(),
        )

        with self.assertRaisesRegex(ValueError, "missing quote"):
            engine.run(
                prices=load_price_history(config.backtest.price_csv),
                quotes=load_quote_history("data/sample_quotes.csv"),
                symbol=config.simple_momentum.symbol,
                starting_equity=config.competition.starting_equity,
            )

    def test_position_stop_exits_losing_position(self) -> None:
        engine = BacktestEngine(
            strategy=EnterOnceStrategy("EURUSD", target_notional_usd=50_000),
            risk_limits=RiskLimits(
                max_position_loss_pct=0.002,
                max_symbol_notional_pct=1.0,
            ),
            quality_limits=MarketQualityLimits(max_spread_bps=10, max_quote_age_seconds=5),
            clock=CompetitionClock(),
            fill_model=FillModel(slippage_bps=0.0),
        )

        result = engine.run(
            prices=_falling_prices(),
            quotes=_falling_quotes(),
            symbol="EURUSD",
            starting_equity=1_000_000,
        )

        stop_fills = [
            fill for fill in result.fills if fill.primary_signal == "position_stop"
        ]
        self.assertEqual(len(stop_fills), 1)
        self.assertEqual(stop_fills[0].side.value, "SELL")
        self.assertAlmostEqual(result.equity_curve[-1].position_units, 0.0)
        self.assertIn("stop-loss", stop_fills[0].risk_reason)

    def test_asset_class_position_stop_override_keeps_metal_trade_open(self) -> None:
        engine = BacktestEngine(
            strategy=EnterOnceStrategy("XAGUSD", target_notional_usd=50_000),
            risk_limits=RiskLimits(
                max_position_loss_pct=0.002,
                max_metal_position_loss_pct=0.02,
                max_symbol_notional_pct=1.0,
            ),
            quality_limits=MarketQualityLimits(max_spread_bps=10, max_quote_age_seconds=5),
            clock=CompetitionClock(),
            fill_model=FillModel(slippage_bps=0.0),
        )

        result = engine.run(
            prices=_falling_prices("XAGUSD"),
            quotes=_falling_quotes("XAGUSD"),
            symbol="XAGUSD",
            starting_equity=1_000_000,
        )

        stop_fills = [
            fill for fill in result.fills if fill.primary_signal == "position_stop"
        ]
        self.assertEqual(stop_fills, [])
        self.assertNotAlmostEqual(result.equity_curve[-1].position_units, 0.0)


class EnterOnceStrategy:
    def __init__(self, symbol: str, target_notional_usd: float) -> None:
        self.symbol = symbol
        self.target_notional_usd = target_notional_usd
        self.entered = False

    def generate_decision(self, prices, **kwargs) -> StrategyDecision:
        if not self.entered:
            self.entered = True
            return StrategyDecision(
                action=StrategyAction.ENTER,
                symbol=self.symbol,
                target_notional_usd=self.target_notional_usd,
                reason="test initial entry",
            )
        return StrategyDecision(
            action=StrategyAction.HOLD,
            symbol=self.symbol,
            target_notional_usd=kwargs.get("current_notional_usd", 0.0),
            reason="test hold",
        )


def _falling_prices(symbol: str = "EURUSD") -> PriceHistory:
    closes = [1.0000, 1.0002, 0.9950, 0.9940]
    return PriceHistory(
        tuple(
            PriceBar(timestamp=_time(index), symbol=symbol, close=close)
            for index, close in enumerate(closes)
        )
    )


def _falling_quotes(symbol: str = "EURUSD") -> QuoteHistory:
    quotes = []
    for bar in _falling_prices(symbol).bars:
        spread = 0.00010
        quotes.append(
            QuoteSnapshot(
                timestamp=bar.timestamp,
                symbol=bar.symbol,
                bid=bar.close - spread / 2,
                ask=bar.close + spread / 2,
            )
        )
    return QuoteHistory(tuple(quotes))


def _time(index: int) -> datetime:
    return datetime(2026, 6, 22, 10, 0, tzinfo=LONDON) + timedelta(minutes=5 * index)
