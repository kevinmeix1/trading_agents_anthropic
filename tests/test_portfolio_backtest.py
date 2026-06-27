from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from zoneinfo import ZoneInfo

from quanthack.core.clock import CompetitionClock
from quanthack.market.market_data import PriceBar, PriceHistory, QuoteHistory, QuoteSnapshot
from quanthack.market.market_quality import MarketQualityLimits
from quanthack.backtesting.portfolio_backtest import (
    PortfolioBacktestEngine,
    write_portfolio_equity_curve_csv,
    write_portfolio_fills_csv,
    write_portfolio_pnl_summary_csv,
)
from quanthack.backtesting.portfolio_allocator import AllocationPolicy
from quanthack.backtesting.portfolio_regime import RegimeTiltPolicy
from quanthack.backtesting.portfolio_symbol_evidence import SymbolEvidenceGatePolicy
from quanthack.backtesting.portfolio_volatility import VolatilityTargetingPolicy
from quanthack.backtesting.warmup import evaluate_portfolio_after_warmup
from quanthack.trading.risk import RiskLimits
from quanthack.strategies.strategy import (
    MomentumConfig,
    SimpleMomentumStrategy,
    StrategyAction,
    StrategyDecision,
)


LONDON = ZoneInfo("Europe/London")


class PortfolioBacktestTest(TestCase):
    def test_portfolio_backtest_runs_multiple_symbols_with_shared_equity(self) -> None:
        engine = _engine()

        result = engine.run(
            prices=_prices(),
            quotes=_quotes(),
            starting_equity=1_000_000,
        )

        self.assertEqual(result.symbols, ("EURUSD", "GBPUSD"))
        self.assertEqual(result.metrics.observations, 6)
        self.assertGreaterEqual(len(result.fills), 2)
        self.assertEqual({fill.symbol for fill in result.fills}, {"EURUSD", "GBPUSD"})
        self.assertEqual(len(result.pnl_by_symbol), 2)
        self.assertEqual(len(result.allocation_reports), 6)
        self.assertTrue(any(report.trimmed_targets for report in result.allocation_reports))
        self.assertAlmostEqual(
            result.total_pnl_usd,
            result.metrics.final_equity - 1_000_000,
            places=6,
        )

    def test_portfolio_backtest_fails_when_symbol_quote_is_missing(self) -> None:
        engine = _engine()
        quotes = QuoteHistory(
            tuple(
                quote
                for quote in _quotes().quotes
                if not (quote.symbol == "GBPUSD" and quote.timestamp == _time(3))
            )
        )

        with self.assertRaisesRegex(ValueError, "missing quote for GBPUSD"):
            engine.run(
                prices=_prices(),
                quotes=quotes,
                starting_equity=1_000_000,
            )

    def test_portfolio_backtest_csv_outputs(self) -> None:
        result = _engine().run(
            prices=_prices(),
            quotes=_quotes(),
            starting_equity=1_000_000,
        )

        with TemporaryDirectory() as tmpdir:
            equity_path = Path(tmpdir) / "portfolio_equity.csv"
            pnl_path = Path(tmpdir) / "portfolio_pnl.csv"
            fills_path = Path(tmpdir) / "portfolio_fills.csv"
            allocation_path = Path(tmpdir) / "portfolio_allocation.csv"
            write_portfolio_equity_curve_csv(result, equity_path)
            write_portfolio_pnl_summary_csv(result, pnl_path)
            write_portfolio_fills_csv(result, fills_path)
            from quanthack.backtesting.portfolio_allocator import write_allocation_report_csv

            write_allocation_report_csv(result.allocation_reports, allocation_path)
            equity_text = equity_path.read_text(encoding="utf-8")
            pnl_text = pnl_path.read_text(encoding="utf-8")
            fills_text = fills_path.read_text(encoding="utf-8")
            allocation_text = allocation_path.read_text(encoding="utf-8")

        self.assertIn("timestamp,equity,cash,gross_notional_usd", equity_text)
        self.assertIn("positions", equity_text)
        self.assertIn("symbol,fills,realized_pnl_usd", pnl_text)
        self.assertIn("PORTFOLIO", pnl_text)
        self.assertIn("timestamp,symbol,side,fill_price", fills_text)
        self.assertIn("turnover_notional_usd", fills_text)
        self.assertIn("primary_signal", fills_text)
        self.assertIn("estimated_risk_status", allocation_text)

    def test_portfolio_backtest_accepts_symbol_specific_quality_limits(self) -> None:
        engine = PortfolioBacktestEngine(
            strategies={
                "EURUSD": SimpleMomentumStrategy(_momentum("EURUSD")),
                "GBPUSD": SimpleMomentumStrategy(_momentum("GBPUSD")),
            },
            risk_limits=RiskLimits(),
            quality_limits=MarketQualityLimits(max_spread_bps=0.01, max_quote_age_seconds=5),
            quality_limits_by_symbol={
                "EURUSD": MarketQualityLimits(max_spread_bps=10, max_quote_age_seconds=5),
                "GBPUSD": MarketQualityLimits(max_spread_bps=10, max_quote_age_seconds=5),
            },
            clock=CompetitionClock(),
        )

        result = engine.run(
            prices=_prices(),
            quotes=_quotes(),
            starting_equity=1_000_000,
        )

        self.assertGreater(len(result.fills), 0)

    def test_warmup_evaluation_scores_only_after_start_timestamp(self) -> None:
        result = _engine().run(
            prices=_prices(),
            quotes=_quotes(),
            starting_equity=1_000_000,
        )

        evaluation = evaluate_portfolio_after_warmup(
            result,
            evaluation_start=_time(3),
        )

        self.assertEqual(evaluation.evaluation_start, _time(3).isoformat())
        self.assertEqual(evaluation.equity_curve[0].timestamp, _time(3).isoformat())
        self.assertTrue(
            all(
                datetime.fromisoformat(fill.timestamp) >= _time(3)
                for fill in evaluation.fills
            )
        )
        self.assertEqual(
            evaluation.competition_metrics.starting_equity,
            evaluation.equity_curve[0].equity,
        )

    def test_warmup_evaluation_requires_timezone_aware_start(self) -> None:
        result = _engine().run(
            prices=_prices(),
            quotes=_quotes(),
            starting_equity=1_000_000,
        )

        with self.assertRaisesRegex(ValueError, "timezone"):
            evaluate_portfolio_after_warmup(
                result,
                evaluation_start=datetime(2026, 6, 22, 10, 15),
            )

    def test_portfolio_backtest_updates_context_aware_strategies(self) -> None:
        eur_strategy = ContextAwareNoTradeStrategy("EURUSD")
        gbp_strategy = ContextAwareNoTradeStrategy("GBPUSD")
        engine = PortfolioBacktestEngine(
            strategies={
                "EURUSD": eur_strategy,
                "GBPUSD": gbp_strategy,
            },
            risk_limits=RiskLimits(),
            quality_limits=MarketQualityLimits(max_spread_bps=10, max_quote_age_seconds=5),
            clock=CompetitionClock(),
        )

        engine.run(
            prices=_prices(),
            quotes=_quotes(),
            starting_equity=1_000_000,
        )

        self.assertEqual(eur_strategy.context_call_count, 6)
        self.assertIn("GBPUSD", eur_strategy.latest_closes_by_symbol)
        self.assertEqual(len(eur_strategy.latest_closes_by_symbol["GBPUSD"]), 6)

    def test_portfolio_backtest_position_stop_exits_losing_symbol(self) -> None:
        engine = PortfolioBacktestEngine(
            strategies={"EURUSD": EnterOnceStrategy("EURUSD", target_notional_usd=50_000)},
            risk_limits=RiskLimits(
                max_position_loss_pct=0.002,
                max_symbol_notional_pct=1.0,
            ),
            quality_limits=MarketQualityLimits(max_spread_bps=10, max_quote_age_seconds=5),
            allocation_policy=AllocationPolicy(
                max_symbol_gross_pct=1.0,
                max_net_directional_pct=1.0,
                max_forex_gross_pct=1.0,
                min_active_symbols=1,
                apply_diversification_scale=False,
            ),
            clock=CompetitionClock(),
        )

        result = engine.run(
            prices=_falling_prices(),
            quotes=_falling_quotes(),
            starting_equity=1_000_000,
        )

        stop_fills = [
            fill for fill in result.fills if fill.primary_signal == "position_stop"
        ]
        self.assertEqual(len(stop_fills), 1)
        self.assertEqual(stop_fills[0].side.value, "SELL")
        self.assertEqual(result.equity_curve[-1].positions, ())
        self.assertIn("stop-loss", stop_fills[0].risk_reason)

    def test_portfolio_backtest_target_notional_multiplier_scales_entries(self) -> None:
        engine = PortfolioBacktestEngine(
            strategies={"EURUSD": EnterOnceStrategy("EURUSD", target_notional_usd=50_000)},
            risk_limits=RiskLimits(max_symbol_notional_pct=1.0),
            quality_limits=MarketQualityLimits(max_spread_bps=10, max_quote_age_seconds=5),
            allocation_policy=AllocationPolicy(
                max_symbol_gross_pct=1.0,
                max_net_directional_pct=1.0,
                max_forex_gross_pct=1.0,
                min_active_symbols=1,
                apply_diversification_scale=False,
            ),
            clock=CompetitionClock(),
            target_notional_multiplier=0.5,
        )

        result = engine.run(
            prices=_rising_prices(),
            quotes=_rising_quotes(),
            starting_equity=1_000_000,
        )

        self.assertGreaterEqual(len(result.fills), 1)
        self.assertAlmostEqual(result.fills[0].requested_notional_usd, 25_000)

        with self.assertRaisesRegex(ValueError, "target_notional_multiplier"):
            PortfolioBacktestEngine(
                strategies={"EURUSD": EnterOnceStrategy("EURUSD", 50_000)},
                risk_limits=RiskLimits(),
                quality_limits=MarketQualityLimits(max_spread_bps=10, max_quote_age_seconds=5),
                clock=CompetitionClock(),
                target_notional_multiplier=0.0,
            )

    def test_portfolio_backtest_symbol_target_multipliers_scale_entries(self) -> None:
        engine = PortfolioBacktestEngine(
            strategies={
                "EURUSD": EnterOnceStrategy("EURUSD", target_notional_usd=50_000),
                "GBPUSD": EnterOnceStrategy("GBPUSD", target_notional_usd=50_000),
            },
            risk_limits=RiskLimits(max_symbol_notional_pct=1.0),
            quality_limits=MarketQualityLimits(max_spread_bps=10, max_quote_age_seconds=5),
            allocation_policy=AllocationPolicy(
                max_symbol_gross_pct=1.0,
                max_net_directional_pct=1.0,
                max_forex_gross_pct=1.0,
                min_active_symbols=1,
                apply_diversification_scale=False,
            ),
            clock=CompetitionClock(),
            target_notional_multipliers_by_symbol={"GBPUSD": 0.5},
        )

        result = engine.run(
            prices=_prices(),
            quotes=_quotes(),
            starting_equity=1_000_000,
        )

        requested_by_symbol = {
            fill.symbol: fill.requested_notional_usd
            for fill in result.fills
            if fill.side.value == "BUY"
        }
        self.assertAlmostEqual(requested_by_symbol["EURUSD"], 50_000)
        self.assertAlmostEqual(requested_by_symbol["GBPUSD"], 25_000)

        with self.assertRaisesRegex(ValueError, "target_notional_multipliers_by_symbol"):
            PortfolioBacktestEngine(
                strategies={"EURUSD": EnterOnceStrategy("EURUSD", 50_000)},
                risk_limits=RiskLimits(),
                quality_limits=MarketQualityLimits(max_spread_bps=10, max_quote_age_seconds=5),
                clock=CompetitionClock(),
                target_notional_multipliers_by_symbol={"EURUSD": 1.5},
            )

    def test_portfolio_backtest_can_apply_portfolio_volatility_targeting(self) -> None:
        engine = PortfolioBacktestEngine(
            strategies={"EURUSD": AlwaysTargetStrategy("EURUSD", target_notional_usd=100_000)},
            risk_limits=RiskLimits(max_symbol_notional_pct=1.0),
            quality_limits=MarketQualityLimits(max_spread_bps=10, max_quote_age_seconds=5),
            allocation_policy=AllocationPolicy(
                max_symbol_gross_pct=1.0,
                max_net_directional_pct=1.0,
                max_forex_gross_pct=1.0,
                min_active_symbols=1,
                apply_diversification_scale=False,
            ),
            clock=CompetitionClock(),
            volatility_targeting_policy=VolatilityTargetingPolicy(
                lookback=3,
                min_observations=3,
                target_bar_volatility=0.00025,
                min_scale=0.20,
                max_scale=1.0,
            ),
        )

        result = engine.run(
            prices=_volatile_prices(),
            quotes=_volatile_quotes(),
            starting_equity=1_000_000,
        )
        scaled_targets = [
            target
            for allocation in result.allocation_reports
            for target in allocation.targets
            if abs(target.requested_notional_usd) < 100_000
        ]

        self.assertEqual(len(result.volatility_reports), len(result.equity_curve))
        self.assertTrue(any(report.applied and report.scale < 1.0 for report in result.volatility_reports))
        self.assertTrue(scaled_targets)
        self.assertTrue(
            any(
                "portfolio_vol_target_scale" in "|".join(target.supporting_signals)
                for target in scaled_targets
            )
        )

    def test_portfolio_backtest_can_apply_regime_tilt_to_trend_sleeve(self) -> None:
        engine = PortfolioBacktestEngine(
            strategies={
                "EURUSD": AlwaysTargetStrategy(
                    "EURUSD",
                    target_notional_usd=100_000,
                    primary_signal="macd_momentum",
                )
            },
            risk_limits=RiskLimits(max_symbol_notional_pct=1.0),
            quality_limits=MarketQualityLimits(max_spread_bps=10, max_quote_age_seconds=5),
            allocation_policy=AllocationPolicy(
                max_symbol_gross_pct=1.0,
                max_net_directional_pct=1.0,
                max_forex_gross_pct=1.0,
                min_active_symbols=1,
                apply_diversification_scale=False,
            ),
            clock=CompetitionClock(),
            regime_tilt_policy=RegimeTiltPolicy(
                lookback=5,
                chop_trend_scale=0.50,
                chop_reversion_scale=1.00,
                trend_aligned_scale=1.00,
                trend_counter_scale=0.50,
                trend_reversion_scale=1.00,
                high_volatility_scale=0.50,
            ),
        )

        result = engine.run(
            prices=_choppy_prices(),
            quotes=_choppy_quotes(),
            starting_equity=1_000_000,
        )
        requested_targets = [
            target.requested_notional_usd
            for allocation in result.allocation_reports
            for target in allocation.targets
        ]

        self.assertEqual(len(result.regime_reports), len(result.equity_curve))
        self.assertTrue(
            any(
                report.applied and report.regime == "CHOP" and report.scale == 0.50
                for report in result.regime_reports
            )
        )
        self.assertIn(50_000.0, requested_targets)

    def test_portfolio_backtest_symbol_evidence_gate_blocks_reentry_after_loss(self) -> None:
        engine = PortfolioBacktestEngine(
            strategies={"EURUSD": ReenterAfterExitStrategy("EURUSD", 100_000)},
            risk_limits=RiskLimits(max_symbol_notional_pct=1.0),
            quality_limits=MarketQualityLimits(max_spread_bps=10, max_quote_age_seconds=5),
            allocation_policy=AllocationPolicy(
                max_symbol_gross_pct=1.0,
                max_net_directional_pct=1.0,
                max_forex_gross_pct=1.0,
                min_active_symbols=1,
                apply_diversification_scale=False,
            ),
            clock=CompetitionClock(),
            symbol_evidence_gate_policy=SymbolEvidenceGatePolicy(
                lookback_closed_events=1,
                min_realized_pnl_usd=0.0,
            ),
        )

        result = engine.run(
            prices=_evidence_gate_prices(),
            quotes=_evidence_gate_quotes(),
            starting_equity=1_000_000,
        )

        self.assertEqual(len(result.fills), 2)
        self.assertTrue(
            any(report.applied and report.symbol == "EURUSD" for report in result.symbol_evidence_reports)
        )
        self.assertTrue(
            any(
                target.primary_signal == "symbol_evidence_gate"
                for allocation in result.allocation_reports
                for target in allocation.targets
            )
        )


def _engine() -> PortfolioBacktestEngine:
    return PortfolioBacktestEngine(
        strategies={
            "EURUSD": SimpleMomentumStrategy(_momentum("EURUSD")),
            "GBPUSD": SimpleMomentumStrategy(_momentum("GBPUSD")),
        },
        risk_limits=RiskLimits(),
        quality_limits=MarketQualityLimits(max_spread_bps=10, max_quote_age_seconds=5),
        clock=CompetitionClock(),
    )


class ContextAwareNoTradeStrategy:
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self.context_call_count = 0
        self.latest_closes_by_symbol: dict[str, tuple[float, ...]] = {}

    def update_portfolio_context(self, *, closes_by_symbol, quotes_by_symbol=None) -> None:
        self.context_call_count += 1
        self.latest_closes_by_symbol = {
            symbol: tuple(closes)
            for symbol, closes in closes_by_symbol.items()
        }

    def generate_decision(self, prices, **kwargs) -> StrategyDecision:
        return StrategyDecision(
            action=StrategyAction.NO_ACTION,
            symbol=self.symbol,
            target_notional_usd=0.0,
            reason="test strategy no trade",
        )


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


class AlwaysTargetStrategy:
    def __init__(
        self,
        symbol: str,
        target_notional_usd: float,
        primary_signal: str = "strategy",
    ) -> None:
        self.symbol = symbol
        self.target_notional_usd = target_notional_usd
        self.primary_signal = primary_signal

    def generate_decision(self, prices, **kwargs) -> StrategyDecision:
        return StrategyDecision(
            action=StrategyAction.ENTER,
            symbol=self.symbol,
            target_notional_usd=self.target_notional_usd,
            reason="test persistent target",
            primary_signal=self.primary_signal,
        )


class ReenterAfterExitStrategy:
    def __init__(self, symbol: str, target_notional_usd: float) -> None:
        self.symbol = symbol
        self.target_notional_usd = target_notional_usd
        self.calls = 0

    def generate_decision(self, prices, **kwargs) -> StrategyDecision:
        self.calls += 1
        if self.calls == 1:
            return StrategyDecision(
                action=StrategyAction.ENTER,
                symbol=self.symbol,
                target_notional_usd=self.target_notional_usd,
                reason="initial test entry",
            )
        if self.calls == 2:
            return StrategyDecision(
                action=StrategyAction.EXIT,
                symbol=self.symbol,
                target_notional_usd=0.0,
                reason="test exit",
            )
        return StrategyDecision(
            action=StrategyAction.ENTER,
            symbol=self.symbol,
            target_notional_usd=self.target_notional_usd,
            reason="test reentry",
        )


def _momentum(symbol: str) -> MomentumConfig:
    return MomentumConfig(
        symbol=symbol,
        lookback=3,
        threshold_bps=4.0,
        exit_threshold_bps=1.0,
        target_notional_usd=50_000,
        max_target_notional_usd=50_000,
    )


def _prices() -> PriceHistory:
    eur = [1.1000, 1.1004, 1.1009, 1.1013, 1.1018, 1.1022]
    gbp = [1.3000, 1.2994, 1.2989, 1.2984, 1.2978, 1.2972]
    bars = [
        PriceBar(timestamp=_time(index), symbol="EURUSD", close=price)
        for index, price in enumerate(eur)
    ]
    bars.extend(
        PriceBar(timestamp=_time(index), symbol="GBPUSD", close=price)
        for index, price in enumerate(gbp)
    )
    return PriceHistory(tuple(bars))


def _quotes() -> QuoteHistory:
    quotes = []
    for bar in _prices().bars:
        spread = 0.00010 if bar.symbol == "EURUSD" else 0.00012
        quotes.append(
            QuoteSnapshot(
                timestamp=bar.timestamp,
                symbol=bar.symbol,
                bid=bar.close - spread / 2,
                ask=bar.close + spread / 2,
            )
        )
    return QuoteHistory(tuple(quotes))


def _falling_prices() -> PriceHistory:
    closes = [1.0000, 1.0002, 0.9950, 0.9940]
    return PriceHistory(
        tuple(
            PriceBar(timestamp=_time(index), symbol="EURUSD", close=close)
            for index, close in enumerate(closes)
        )
    )


def _falling_quotes() -> QuoteHistory:
    quotes = []
    for bar in _falling_prices().bars:
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


def _rising_prices() -> PriceHistory:
    closes = [1.0000, 1.0002, 1.0004, 1.0006]
    return PriceHistory(
        tuple(
            PriceBar(timestamp=_time(index), symbol="EURUSD", close=close)
            for index, close in enumerate(closes)
        )
    )


def _rising_quotes() -> QuoteHistory:
    quotes = []
    for bar in _rising_prices().bars:
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


def _volatile_prices() -> PriceHistory:
    closes = [1.0000, 1.0200, 0.9800, 1.0300, 0.9700, 1.0400, 0.9600]
    return PriceHistory(
        tuple(
            PriceBar(timestamp=_time(index), symbol="EURUSD", close=close)
            for index, close in enumerate(closes)
        )
    )


def _volatile_quotes() -> QuoteHistory:
    quotes = []
    for bar in _volatile_prices().bars:
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


def _choppy_prices() -> PriceHistory:
    closes = [1.0000, 1.0010, 0.9990, 1.0010, 0.9990, 1.0010, 0.9990]
    return PriceHistory(
        tuple(
            PriceBar(timestamp=_time(index), symbol="EURUSD", close=close)
            for index, close in enumerate(closes)
        )
    )


def _choppy_quotes() -> QuoteHistory:
    quotes = []
    for bar in _choppy_prices().bars:
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


def _evidence_gate_prices() -> PriceHistory:
    closes = [1.0000, 0.9900, 1.0000, 1.0100]
    return PriceHistory(
        tuple(
            PriceBar(timestamp=_time(index), symbol="EURUSD", close=close)
            for index, close in enumerate(closes)
        )
    )


def _evidence_gate_quotes() -> QuoteHistory:
    quotes = []
    for bar in _evidence_gate_prices().bars:
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
