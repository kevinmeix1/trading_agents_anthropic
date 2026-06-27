from datetime import datetime
from math import exp, log
from unittest import TestCase
from zoneinfo import ZoneInfo

from quanthack.market.market_data import QuoteSnapshot
from quanthack.trading.risk import Side
from quanthack.strategies.strategy import (
    AlphaRouterConfig,
    AlphaRouterStrategy,
    AutocorrelationRegimeConfig,
    AutocorrelationRegimeStrategy,
    AssetAdaptiveDualSqueezeConfig,
    AssetAdaptiveDualSqueezeStrategy,
    BreakoutConfig,
    BreakoutStrategy,
    ChampionEnsembleConfig,
    ChampionEnsembleStrategy,
    ConditionalSeasonalityConfig,
    ConditionalSeasonalityStrategy,
    CrossRateReversionConfig,
    CrossRateReversionStrategy,
    DualSqueezeConfig,
    DualSqueezeStrategy,
    ExhaustionReversalConfig,
    ExhaustionReversalStrategy,
    FixingReversalConfig,
    FixingReversalStrategy,
    IntradaySeasonalityConfig,
    IntradaySeasonalityStrategy,
    KalmanTrendStrategy,
    KalmanTrendStrategyConfig,
    LiquiditySweepReversalConfig,
    LiquiditySweepReversalStrategy,
    MacdConditionalFallbackConfig,
    MacdConditionalFallbackStrategy,
    MacdMomentumConfig,
    MacdMomentumStrategy,
    MacdSqueezeComplementConfig,
    MacdSqueezeComplementStrategy,
    MeanReversionConfig,
    MeanReversionStrategy,
    MomentumConfig,
    MultiHorizonMomentumConfig,
    MultiHorizonMomentumStrategy,
    MovingAverageCrossoverConfig,
    MovingAverageCrossoverStrategy,
    QualityTrendConfig,
    QualityTrendStrategy,
    RangeExpansionTrendConfig,
    RangeExpansionTrendStrategy,
    RegimeConfig,
    RegimeState,
    RegimeSwitchingStrategy,
    RelativeStrengthConfig,
    RelativeStrengthStrategy,
    SessionBreakoutConfig,
    SessionBreakoutStrategy,
    SimpleMomentumStrategy,
    StrategyAction,
    StrategyDecision,
    SignalDirection,
    TrendPullbackConfig,
    TrendPullbackStrategy,
    UsdPressureConfig,
    UsdPressureRouterStrategy,
    VolatilitySqueezeConfig,
    VolatilitySqueezeStrategy,
    build_strategy,
    normalize_strategy_name,
)


LONDON = ZoneInfo("Europe/London")


class _StaticDecisionStrategy:
    def __init__(self, decision: StrategyDecision) -> None:
        self.decision = decision

    def generate_decision(self, *args, **kwargs) -> StrategyDecision:
        return self.decision


def quote(*, bid: float = 1.09995, ask: float = 1.10005) -> QuoteSnapshot:
    return QuoteSnapshot(
        timestamp=datetime(2026, 6, 22, 10, 0, tzinfo=LONDON),
        symbol="EURUSD",
        bid=bid,
        ask=ask,
    )


def _prices_from_log_returns(log_returns: list[float]) -> list[float]:
    price = 1.0
    prices = [price]
    for log_return in log_returns:
        price *= exp(log_return)
        prices.append(price)
    return prices


def _fixed_decision(
    action: StrategyAction,
    *,
    symbol: str,
    target_notional_usd: float,
    reason: str,
) -> StrategyDecision:
    return StrategyDecision(
        action=action,
        symbol=symbol,
        target_notional_usd=target_notional_usd,
        reason=reason,
    )


class _FixedDecisionStrategy:
    def __init__(self, decision: StrategyDecision) -> None:
        self.decision = decision

    def generate_decision(
        self,
        prices,
        *,
        current_notional_usd: float = 0.0,
        holding_period: int = 0,
        quote: QuoteSnapshot | None = None,
    ) -> StrategyDecision:
        return self.decision


class SimpleMomentumStrategyTest(TestCase):
    def test_up_move_generates_buy_request(self) -> None:
        strategy = SimpleMomentumStrategy(MomentumConfig(threshold_bps=5.0))

        request = strategy.generate_request([1.1000, 1.1002, 1.1004, 1.1007, 1.1010])

        self.assertIsNotNone(request)
        self.assertEqual(request.side, Side.BUY)
        self.assertEqual(request.symbol, "EURUSD")

    def test_down_move_generates_sell_request(self) -> None:
        strategy = SimpleMomentumStrategy(MomentumConfig(threshold_bps=5.0))

        request = strategy.generate_request([1.1000, 1.0998, 1.0996, 1.0993, 1.0990])

        self.assertIsNotNone(request)
        self.assertEqual(request.side, Side.SELL)

    def test_flat_move_generates_no_request(self) -> None:
        strategy = SimpleMomentumStrategy(MomentumConfig(threshold_bps=8.0))

        request = strategy.generate_request([1.1000, 1.1001, 1.1000, 1.1001, 1.1000])

        self.assertIsNone(request)

    def test_momentum_reading_uses_log_return(self) -> None:
        strategy = SimpleMomentumStrategy(MomentumConfig(lookback=2))

        reading = strategy.read_momentum([100.0, 105.0])

        self.assertIsNotNone(reading)
        assert reading is not None
        self.assertAlmostEqual(reading.cumulative_log_return, log(1.05))
        self.assertAlmostEqual(reading.move_bps, log(1.05) * 10_000)

    def test_noisy_momentum_path_can_fail_efficiency_filter(self) -> None:
        strategy = SimpleMomentumStrategy(
            MomentumConfig(threshold_bps=5.0, min_trend_efficiency=0.80)
        )

        decision = strategy.generate_decision([1.0000, 1.0200, 0.9800, 1.0200, 1.0100])

        self.assertEqual(decision.action, StrategyAction.NO_ACTION)
        self.assertIn("trend efficiency", decision.reason)

    def test_momentum_exit_hysteresis_exits_existing_position(self) -> None:
        strategy = SimpleMomentumStrategy(
            MomentumConfig(threshold_bps=8.0, exit_threshold_bps=3.0)
        )

        decision = strategy.generate_decision(
            [1.1000, 1.1001, 1.1002, 1.1002, 1.1002],
            current_notional_usd=50_000,
        )

        self.assertEqual(decision.action, StrategyAction.EXIT)
        self.assertEqual(decision.target_notional_usd, 0.0)

    def test_momentum_cost_filter_blocks_wide_spread(self) -> None:
        strategy = SimpleMomentumStrategy(
            MomentumConfig(
                threshold_bps=5.0,
                min_trend_efficiency=0.0,
                max_spread_bps=5.0,
            )
        )

        decision = strategy.generate_decision(
            [1.1000, 1.1010, 1.1020, 1.1030, 1.1040],
            quote=quote(bid=1.0900, ask=1.1100),
        )

        self.assertEqual(decision.action, StrategyAction.NO_ACTION)
        self.assertIn("spread", decision.reason)

    def test_momentum_session_filter_blocks_new_entry_outside_hours(self) -> None:
        strategy = SimpleMomentumStrategy(
            MomentumConfig(
                threshold_bps=5.0,
                min_trend_efficiency=0.0,
                forex_allowed_utc_hours=(17,),
            )
        )

        decision = strategy.generate_decision(
            [1.1000, 1.1010, 1.1020, 1.1030, 1.1040],
            quote=QuoteSnapshot(
                timestamp=datetime(2026, 6, 22, 10, 0, tzinfo=ZoneInfo("UTC")),
                symbol="EURUSD",
                bid=1.10395,
                ask=1.10405,
            ),
        )

        self.assertEqual(decision.action, StrategyAction.NO_ACTION)
        self.assertIn("outside momentum UTC hours", decision.reason)

    def test_momentum_session_filter_exits_existing_position_outside_hours(self) -> None:
        strategy = SimpleMomentumStrategy(
            MomentumConfig(
                threshold_bps=5.0,
                min_trend_efficiency=0.0,
                forex_allowed_utc_hours=(17,),
            )
        )

        decision = strategy.generate_decision(
            [1.1000, 1.1010, 1.1020, 1.1030, 1.1040],
            current_notional_usd=50_000,
            quote=QuoteSnapshot(
                timestamp=datetime(2026, 6, 22, 10, 0, tzinfo=ZoneInfo("UTC")),
                symbol="EURUSD",
                bid=1.10395,
                ask=1.10405,
            ),
        )

        self.assertEqual(decision.action, StrategyAction.EXIT)
        self.assertEqual(decision.target_notional_usd, 0.0)

    def test_momentum_session_filter_allows_entry_inside_hours(self) -> None:
        strategy = SimpleMomentumStrategy(
            MomentumConfig(
                threshold_bps=5.0,
                min_trend_efficiency=0.0,
                forex_allowed_utc_hours=(17,),
            )
        )

        decision = strategy.generate_decision(
            [1.1000, 1.1010, 1.1020, 1.1030, 1.1040],
            quote=QuoteSnapshot(
                timestamp=datetime(2026, 6, 22, 17, 0, tzinfo=ZoneInfo("UTC")),
                symbol="EURUSD",
                bid=1.10395,
                ask=1.10405,
            ),
        )

        self.assertEqual(decision.action, StrategyAction.ENTER)

    def test_momentum_config_rejects_invalid_session_hour(self) -> None:
        with self.assertRaisesRegex(ValueError, "forex_allowed_utc_hours"):
            MomentumConfig(forex_allowed_utc_hours=(24,))

    def test_volatility_sized_momentum_respects_notional_cap(self) -> None:
        strategy = SimpleMomentumStrategy(
            MomentumConfig(
                threshold_bps=5.0,
                position_sizing="volatility",
                target_volatility=0.01,
                max_target_notional_usd=60_000,
            )
        )

        decision = strategy.generate_decision([1.1000, 1.1005, 1.1010, 1.1015, 1.1020])

        self.assertEqual(decision.action, StrategyAction.ENTER)
        self.assertEqual(decision.target_notional_usd, 60_000)

    def test_not_enough_prices_generates_no_request(self) -> None:
        strategy = SimpleMomentumStrategy(MomentumConfig(lookback=5))

        request = strategy.generate_request([1.1000, 1.1001, 1.1002])

        self.assertIsNone(request)

    def test_bad_price_is_rejected(self) -> None:
        strategy = SimpleMomentumStrategy()

        with self.assertRaisesRegex(ValueError, "positive finite"):
            strategy.generate_request([1.1000, 1.1001, 0.0, 1.1002, 1.1003])

    def test_config_rejects_lookback_below_two(self) -> None:
        with self.assertRaisesRegex(ValueError, "lookback"):
            MomentumConfig(lookback=1)


class MultiHorizonMomentumStrategyTest(TestCase):
    def test_enters_long_when_fast_and_slow_momentum_align(self) -> None:
        strategy = MultiHorizonMomentumStrategy(
            MultiHorizonMomentumConfig(
                fast_lookback=6,
                slow_lookback=24,
                baseline_volatility_lookback=48,
                min_fast_move_bps=1.0,
                min_slow_move_bps=4.0,
                min_trend_efficiency=0.20,
                min_volatility_ratio=0.0,
                min_realized_volatility_bps=0.0,
                position_sizing="fixed",
                forex_allowed_utc_hours=(12,),
                max_spread_bps=5.0,
            )
        )
        prices = [
            1.0000 + (index * 0.0002) + (0.00002 if index % 2 else 0.0)
            for index in range(49)
        ]

        decision = strategy.generate_decision(
            prices,
            quote=QuoteSnapshot(
                timestamp=datetime(2026, 6, 22, 12, 0, tzinfo=ZoneInfo("UTC")),
                symbol="EURUSD",
                bid=1.00955,
                ask=1.00965,
            ),
        )

        self.assertEqual(decision.action, StrategyAction.ENTER)
        self.assertGreater(decision.target_notional_usd, 0)
        self.assertEqual(decision.primary_signal, "multi_horizon_momentum")
        self.assertIn("long multi-horizon momentum", decision.reason)

    def test_blocks_when_fast_and_slow_momentum_disagree(self) -> None:
        strategy = MultiHorizonMomentumStrategy(
            MultiHorizonMomentumConfig(
                fast_lookback=6,
                slow_lookback=24,
                baseline_volatility_lookback=48,
                min_fast_move_bps=1.0,
                min_slow_move_bps=4.0,
                min_trend_efficiency=0.0,
                min_volatility_ratio=0.0,
                min_realized_volatility_bps=0.0,
                max_spread_bps=5.0,
            )
        )
        prices = [1.0000 + (index * 0.0002) for index in range(43)]
        prices.extend([1.0086, 1.0084, 1.0082, 1.0080, 1.0078, 1.0076])

        decision = strategy.generate_decision(
            prices,
            quote=QuoteSnapshot(
                timestamp=datetime(2026, 6, 22, 12, 0, tzinfo=ZoneInfo("UTC")),
                symbol="EURUSD",
                bid=1.00755,
                ask=1.00765,
            ),
        )

        self.assertEqual(decision.action, StrategyAction.NO_ACTION)
        self.assertIn("fast and slow momentum are not aligned", decision.reason)

    def test_blocks_new_entry_outside_session(self) -> None:
        strategy = MultiHorizonMomentumStrategy(
            MultiHorizonMomentumConfig(
                fast_lookback=6,
                slow_lookback=24,
                baseline_volatility_lookback=48,
                min_fast_move_bps=1.0,
                min_slow_move_bps=4.0,
                min_trend_efficiency=0.0,
                min_volatility_ratio=0.0,
                min_realized_volatility_bps=0.0,
                forex_allowed_utc_hours=(12,),
            )
        )

        decision = strategy.generate_decision(
            [
                1.0000 + (index * 0.0002) + (0.00002 if index % 2 else 0.0)
                for index in range(49)
            ],
            quote=QuoteSnapshot(
                timestamp=datetime(2026, 6, 22, 18, 0, tzinfo=ZoneInfo("UTC")),
                symbol="EURUSD",
                bid=1.00955,
                ask=1.00965,
            ),
        )

        self.assertEqual(decision.action, StrategyAction.NO_ACTION)
        self.assertIn("outside multi-horizon momentum UTC hours", decision.reason)

    def test_config_rejects_invalid_volatility_ratio(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_volatility_ratio"):
            MultiHorizonMomentumConfig(
                min_volatility_ratio=1.0,
                max_volatility_ratio=1.0,
            )


class IntradaySeasonalityStrategyTest(TestCase):
    def test_same_slot_positive_bias_generates_buy_decision(self) -> None:
        strategy = IntradaySeasonalityStrategy(
            IntradaySeasonalityConfig(
                period_bars=2,
                lookback_periods=2,
                entry_threshold_bps=5.0,
                min_consistency=0.5,
                position_sizing="fixed",
            )
        )

        decision = strategy.generate_decision([100.0, 101.0, 100.0, 102.0, 100.0, 103.0])

        self.assertEqual(decision.action, StrategyAction.ENTER)
        self.assertGreater(decision.target_notional_usd, 0.0)

    def test_same_slot_low_consistency_blocks_entry(self) -> None:
        strategy = IntradaySeasonalityStrategy(
            IntradaySeasonalityConfig(
                period_bars=2,
                lookback_periods=2,
                entry_threshold_bps=5.0,
                min_consistency=1.0,
                position_sizing="fixed",
            )
        )

        decision = strategy.generate_decision([100.0, 102.0, 100.0, 99.0, 100.0, 103.0])

        self.assertEqual(decision.action, StrategyAction.NO_ACTION)
        self.assertIn("consistency", decision.reason)

    def test_same_slot_session_filter_exits_existing_position(self) -> None:
        strategy = IntradaySeasonalityStrategy(
            IntradaySeasonalityConfig(
                period_bars=2,
                lookback_periods=2,
                entry_threshold_bps=5.0,
                min_consistency=0.5,
                forex_allowed_utc_hours=(17,),
            )
        )

        decision = strategy.generate_decision(
            [100.0, 101.0, 100.0, 102.0, 100.0, 103.0],
            current_notional_usd=50_000,
            quote=QuoteSnapshot(
                timestamp=datetime(2026, 6, 22, 10, 0, tzinfo=ZoneInfo("UTC")),
                symbol="EURUSD",
                bid=103.0,
                ask=103.1,
            ),
        )

        self.assertEqual(decision.action, StrategyAction.EXIT)
        self.assertIn("outside intraday seasonality UTC hours", decision.reason)

    def test_intraday_seasonality_config_rejects_bad_period(self) -> None:
        with self.assertRaisesRegex(ValueError, "period_bars"):
            IntradaySeasonalityConfig(period_bars=1)

    def test_intraday_seasonality_config_rejects_bad_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "signal_mode"):
            IntradaySeasonalityConfig(signal_mode="coinflip")


class ConditionalSeasonalityStrategyTest(TestCase):
    def _config(self) -> ConditionalSeasonalityConfig:
        return ConditionalSeasonalityConfig(
            period_bars=8,
            lookback_periods=3,
            horizon_bars=2,
            momentum_lookback=2,
            momentum_threshold_bps=0.5,
            signal_mode="momentum",
            min_samples=2,
            entry_threshold_bps=1.0,
            min_abs_tstat=0.0,
            position_sizing="fixed",
            slippage_bps=0.0,
            max_spread_bps=5.0,
        )

    def _repeating_prices(self) -> list[float]:
        prices = [1.0]
        for cycle in range(4):
            base = prices[-1]
            moves = [0.0000, 0.0001, 0.0001, 0.0001, 0.0001, 0.0004, 0.0004, 0.0005]
            for move in moves:
                prices.append(base + move + cycle * 0.0001)
        return prices

    def test_matching_same_slot_condition_generates_long_decision(self) -> None:
        prices = self._repeating_prices()
        strategy = ConditionalSeasonalityStrategy(self._config())

        decision = strategy.generate_decision(
            prices,
            quote=QuoteSnapshot(
                timestamp=datetime(2026, 6, 22, 12, 0, tzinfo=ZoneInfo("UTC")),
                symbol="EURUSD",
                bid=prices[-1] - 0.00005,
                ask=prices[-1] + 0.00005,
            ),
        )

        self.assertEqual(decision.action, StrategyAction.ENTER)
        self.assertGreater(decision.target_notional_usd, 0.0)
        self.assertEqual(decision.primary_signal, "conditional_seasonality")
        self.assertIn("conditional seasonality", decision.reason)

    def test_insufficient_matching_samples_blocks_entry(self) -> None:
        config = ConditionalSeasonalityConfig(
            period_bars=8,
            lookback_periods=3,
            horizon_bars=2,
            momentum_lookback=2,
            min_samples=3,
        )
        strategy = ConditionalSeasonalityStrategy(config)

        decision = strategy.generate_decision([1.0 + index * 0.0001 for index in range(20)])

        self.assertEqual(decision.action, StrategyAction.NO_ACTION)
        self.assertIn("not enough matching", decision.reason)

    def test_horizon_exit_closes_existing_position(self) -> None:
        prices = self._repeating_prices()
        strategy = ConditionalSeasonalityStrategy(self._config())

        decision = strategy.generate_decision(
            prices,
            current_notional_usd=50_000,
            holding_period=4,
        )

        self.assertEqual(decision.action, StrategyAction.EXIT)
        self.assertIn("horizon reached", decision.reason)

    def test_config_rejects_horizon_at_or_above_period(self) -> None:
        with self.assertRaisesRegex(ValueError, "horizon_bars"):
            ConditionalSeasonalityConfig(period_bars=4, horizon_bars=4)

    def test_config_rejects_invalid_signal_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "signal_mode"):
            ConditionalSeasonalityConfig(signal_mode="coinflip")


class BreakoutStrategyTest(TestCase):
    def test_upper_breakout_generates_buy_request(self) -> None:
        strategy = BreakoutStrategy(BreakoutConfig(lookback=5, breakout_buffer_bps=1.0))

        request = strategy.generate_request([1.1000, 1.1002, 1.1001, 1.1003, 1.1010])

        self.assertIsNotNone(request)
        assert request is not None
        self.assertEqual(request.side, Side.BUY)

    def test_lower_breakout_generates_sell_request(self) -> None:
        strategy = BreakoutStrategy(BreakoutConfig(lookback=5, breakout_buffer_bps=1.0))

        request = strategy.generate_request([1.1000, 1.0998, 1.0999, 1.0997, 1.0990])

        self.assertIsNotNone(request)
        assert request is not None
        self.assertEqual(request.side, Side.SELL)

    def test_no_breakout_generates_no_request(self) -> None:
        strategy = BreakoutStrategy(BreakoutConfig(lookback=5, breakout_buffer_bps=1.0))

        request = strategy.generate_request([1.1000, 1.1002, 1.1001, 1.1003, 1.1002])

        self.assertIsNone(request)

    def test_breakout_channel_excludes_latest_price(self) -> None:
        strategy = BreakoutStrategy(BreakoutConfig(lookback=5))

        reading = strategy.read_breakout([1.1000, 1.1002, 1.1001, 1.1003, 1.1010])

        self.assertIsNotNone(reading)
        assert reading is not None
        self.assertEqual(reading.upper_band, 1.1003)
        self.assertEqual(reading.last_price, 1.1010)

    def test_breakout_cost_filter_blocks_wide_spread(self) -> None:
        strategy = BreakoutStrategy(
            BreakoutConfig(lookback=5, breakout_buffer_bps=1.0, max_spread_bps=5.0)
        )

        decision = strategy.generate_decision(
            [1.1000, 1.1002, 1.1001, 1.1003, 1.1010],
            quote=quote(bid=1.0900, ask=1.1100),
        )

        self.assertEqual(decision.action, StrategyAction.NO_ACTION)
        self.assertIn("spread", decision.reason)

    def test_breakout_config_rejects_short_lookback(self) -> None:
        with self.assertRaisesRegex(ValueError, "lookback"):
            BreakoutConfig(lookback=2)


class VolatilitySqueezeStrategyTest(TestCase):
    def test_upper_squeeze_breakout_generates_buy_request(self) -> None:
        strategy = VolatilitySqueezeStrategy(
            VolatilitySqueezeConfig(
                lookback=10,
                squeeze_window=3,
                band_stdev_multiplier=1.0,
                breakout_buffer_bps=1.0,
                max_squeeze_ratio=0.5,
                min_prior_volatility_bps=0.1,
                min_band_width_bps=0.0,
                position_sizing="fixed",
            )
        )

        request = strategy.generate_request(
            [
                1.0000,
                1.0010,
                0.9990,
                1.0010,
                0.9990,
                1.0000,
                1.00005,
                0.99995,
                1.0000,
                1.0025,
            ]
        )

        self.assertIsNotNone(request)
        assert request is not None
        self.assertEqual(request.side, Side.BUY)

    def test_squeeze_reading_measures_compression(self) -> None:
        strategy = VolatilitySqueezeStrategy(
            VolatilitySqueezeConfig(
                lookback=10,
                squeeze_window=3,
                band_stdev_multiplier=1.0,
                min_prior_volatility_bps=0.1,
                min_band_width_bps=0.0,
            )
        )

        reading = strategy.read_squeeze(
            [
                1.0000,
                1.0010,
                0.9990,
                1.0010,
                0.9990,
                1.0000,
                1.00005,
                0.99995,
                1.0000,
                1.0025,
            ]
        )

        self.assertIsNotNone(reading)
        assert reading is not None
        self.assertGreater(reading.breakout_bps, 0.0)
        self.assertLess(reading.squeeze_ratio, 0.5)
        self.assertGreater(reading.prior_volatility_bps, reading.recent_volatility_bps)

    def test_blocks_when_market_has_not_compressed(self) -> None:
        strategy = VolatilitySqueezeStrategy(
            VolatilitySqueezeConfig(
                lookback=10,
                squeeze_window=3,
                band_stdev_multiplier=1.0,
                breakout_buffer_bps=1.0,
                max_squeeze_ratio=0.5,
                min_prior_volatility_bps=0.0,
                min_band_width_bps=0.0,
                position_sizing="fixed",
            )
        )

        decision = strategy.generate_decision(
            [
                1.0000,
                1.0001,
                1.0000,
                1.0001,
                1.0000,
                1.0010,
                0.9990,
                1.0010,
                0.9990,
                1.0025,
            ]
        )

        self.assertEqual(decision.action, StrategyAction.NO_ACTION)
        self.assertIn("squeeze ratio", decision.reason)

    def test_max_holding_period_exits_existing_position(self) -> None:
        strategy = VolatilitySqueezeStrategy(
            VolatilitySqueezeConfig(
                lookback=10,
                squeeze_window=3,
                band_stdev_multiplier=1.0,
                min_prior_volatility_bps=0.1,
                min_band_width_bps=0.0,
                max_holding_period=3,
            )
        )

        decision = strategy.generate_decision(
            [
                1.0000,
                1.0010,
                0.9990,
                1.0010,
                0.9990,
                1.0000,
                1.00005,
                0.99995,
                1.0000,
                1.0025,
            ],
            current_notional_usd=50_000,
            holding_period=3,
        )

        self.assertEqual(decision.action, StrategyAction.EXIT)
        self.assertIn("max holding period", decision.reason)

    def test_cost_filter_blocks_wide_spread(self) -> None:
        strategy = VolatilitySqueezeStrategy(
            VolatilitySqueezeConfig(
                lookback=10,
                squeeze_window=3,
                band_stdev_multiplier=1.0,
                breakout_buffer_bps=1.0,
                max_squeeze_ratio=0.5,
                min_prior_volatility_bps=0.1,
                min_band_width_bps=0.0,
                max_spread_bps=5.0,
            )
        )

        decision = strategy.generate_decision(
            [
                1.0000,
                1.0010,
                0.9990,
                1.0010,
                0.9990,
                1.0000,
                1.00005,
                0.99995,
                1.0000,
                1.0025,
            ],
            quote=QuoteSnapshot(
                timestamp=datetime(2026, 6, 22, 12, 0, tzinfo=ZoneInfo("UTC")),
                symbol="EURUSD",
                bid=0.9900,
                ask=1.0100,
            ),
        )

        self.assertEqual(decision.action, StrategyAction.NO_ACTION)
        self.assertIn("spread", decision.reason)

    def test_blocks_new_fx_entry_outside_allowed_utc_hours(self) -> None:
        strategy = VolatilitySqueezeStrategy(
            VolatilitySqueezeConfig(
                lookback=10,
                squeeze_window=3,
                band_stdev_multiplier=1.0,
                breakout_buffer_bps=1.0,
                max_squeeze_ratio=0.5,
                min_prior_volatility_bps=0.1,
                min_band_width_bps=0.0,
                position_sizing="fixed",
                forex_allowed_utc_hours=(12,),
            )
        )

        decision = strategy.generate_decision(
            [
                1.0000,
                1.0010,
                0.9990,
                1.0010,
                0.9990,
                1.0000,
                1.00005,
                0.99995,
                1.0000,
                1.0025,
            ],
            quote=QuoteSnapshot(
                timestamp=datetime(2026, 6, 22, 9, 0, tzinfo=ZoneInfo("UTC")),
                symbol="EURUSD",
                bid=1.00245,
                ask=1.00255,
            ),
        )

        self.assertEqual(decision.action, StrategyAction.NO_ACTION)
        self.assertIn("outside volatility squeeze UTC session", decision.reason)

    def test_metal_uses_metal_specific_allowed_utc_hours(self) -> None:
        strategy = VolatilitySqueezeStrategy(
            VolatilitySqueezeConfig(
                symbol="XAUUSD",
                lookback=10,
                squeeze_window=3,
                band_stdev_multiplier=1.0,
                breakout_buffer_bps=1.0,
                max_squeeze_ratio=0.5,
                min_prior_volatility_bps=0.1,
                min_band_width_bps=0.0,
                position_sizing="fixed",
                forex_allowed_utc_hours=(9,),
                metal_allowed_utc_hours=(18,),
            )
        )

        decision = strategy.generate_decision(
            [
                2300.0,
                2304.0,
                2296.0,
                2304.0,
                2296.0,
                2300.0,
                2300.1,
                2299.9,
                2300.0,
                2308.0,
            ],
            quote=QuoteSnapshot(
                timestamp=datetime(2026, 6, 22, 18, 0, tzinfo=ZoneInfo("UTC")),
                symbol="XAUUSD",
                bid=2307.9,
                ask=2308.1,
            ),
        )

        self.assertEqual(decision.action, StrategyAction.ENTER)

    def test_config_rejects_too_short_squeeze_window(self) -> None:
        with self.assertRaisesRegex(ValueError, "squeeze_window"):
            VolatilitySqueezeConfig(lookback=10, squeeze_window=1)

    def test_config_rejects_lookback_without_prior_window(self) -> None:
        with self.assertRaisesRegex(ValueError, "prior returns"):
            VolatilitySqueezeConfig(lookback=8, squeeze_window=5)

    def test_config_rejects_invalid_allowed_hour(self) -> None:
        with self.assertRaisesRegex(ValueError, "forex_allowed_utc_hours"):
            VolatilitySqueezeConfig(forex_allowed_utc_hours=(24,))


class SessionBreakoutStrategyTest(TestCase):
    def test_enters_during_allowed_utc_hour(self) -> None:
        strategy = SessionBreakoutStrategy(
            SessionBreakoutConfig(
                lookback=8,
                breakout_buffer_bps=2.0,
                min_realized_volatility_bps=0.1,
                allowed_utc_hours=(12,),
                max_spread_bps=5.0,
            )
        )
        market_quote = QuoteSnapshot(
            timestamp=datetime(2026, 6, 22, 12, 0, tzinfo=ZoneInfo("UTC")),
            symbol="EURUSD",
            bid=1.10115,
            ask=1.10125,
        )

        decision = strategy.generate_decision(
            [1.1000, 1.1001, 1.1000, 1.1002, 1.1001, 1.1004, 1.1003, 1.1012],
            quote=market_quote,
        )

        self.assertEqual(decision.action, StrategyAction.ENTER)
        self.assertGreater(decision.target_notional_usd, 0.0)
        self.assertEqual(decision.primary_signal, "session_breakout")

    def test_blocks_new_entry_outside_allowed_utc_hour(self) -> None:
        strategy = SessionBreakoutStrategy(
            SessionBreakoutConfig(
                lookback=8,
                breakout_buffer_bps=2.0,
                min_realized_volatility_bps=0.1,
                allowed_utc_hours=(12,),
            )
        )
        market_quote = QuoteSnapshot(
            timestamp=datetime(2026, 6, 22, 4, 0, tzinfo=ZoneInfo("UTC")),
            symbol="EURUSD",
            bid=1.10115,
            ask=1.10125,
        )

        decision = strategy.generate_decision(
            [1.1000, 1.1001, 1.1000, 1.1002, 1.1001, 1.1004, 1.1003, 1.1012],
            quote=market_quote,
        )

        self.assertEqual(decision.action, StrategyAction.NO_ACTION)
        self.assertIn("outside allowed UTC session", decision.reason)

    def test_blocks_low_realized_volatility(self) -> None:
        strategy = SessionBreakoutStrategy(
            SessionBreakoutConfig(
                lookback=8,
                breakout_buffer_bps=0.5,
                min_channel_width_bps=0.0,
                min_realized_volatility_bps=50.0,
                allowed_utc_hours=(12,),
            )
        )
        market_quote = QuoteSnapshot(
            timestamp=datetime(2026, 6, 22, 12, 0, tzinfo=ZoneInfo("UTC")),
            symbol="EURUSD",
            bid=1.10027,
            ask=1.10037,
        )

        decision = strategy.generate_decision(
            [1.10000, 1.10005, 1.10002, 1.10008, 1.10004, 1.10010, 1.10006, 1.10032],
            quote=market_quote,
        )

        self.assertEqual(decision.action, StrategyAction.NO_ACTION)
        self.assertIn("realized volatility", decision.reason)

    def test_blocks_session_breakout_below_required_edge(self) -> None:
        strategy = SessionBreakoutStrategy(
            SessionBreakoutConfig(
                lookback=8,
                breakout_buffer_bps=2.0,
                min_expected_edge_bps=20.0,
                min_realized_volatility_bps=0.1,
                allowed_utc_hours=(12,),
            )
        )
        market_quote = QuoteSnapshot(
            timestamp=datetime(2026, 6, 22, 12, 0, tzinfo=ZoneInfo("UTC")),
            symbol="EURUSD",
            bid=1.10115,
            ask=1.10125,
        )

        decision = strategy.generate_decision(
            [1.1000, 1.1001, 1.1000, 1.1002, 1.1001, 1.1004, 1.1003, 1.1012],
            quote=market_quote,
        )

        self.assertEqual(decision.action, StrategyAction.NO_ACTION)
        self.assertIn("required edge", decision.reason)

    def test_uses_metal_specific_session_hours(self) -> None:
        strategy = SessionBreakoutStrategy(
            SessionBreakoutConfig(
                symbol="XAUUSD",
                lookback=8,
                breakout_buffer_bps=2.0,
                min_realized_volatility_bps=0.1,
                allowed_utc_hours=(12,),
                metal_allowed_utc_hours=(8,),
                max_spread_bps=5.0,
            )
        )
        prices = [2320.0, 2320.3, 2320.1, 2320.4, 2320.2, 2320.6, 2320.5, 2322.0]

        blocked = strategy.generate_decision(
            prices,
            quote=QuoteSnapshot(
                timestamp=datetime(2026, 6, 22, 12, 0, tzinfo=ZoneInfo("UTC")),
                symbol="XAUUSD",
                bid=2321.9,
                ask=2322.1,
            ),
        )
        allowed = strategy.generate_decision(
            prices,
            quote=QuoteSnapshot(
                timestamp=datetime(2026, 6, 22, 8, 0, tzinfo=ZoneInfo("UTC")),
                symbol="XAUUSD",
                bid=2321.9,
                ask=2322.1,
            ),
        )

        self.assertEqual(blocked.action, StrategyAction.NO_ACTION)
        self.assertIn("outside allowed UTC session", blocked.reason)
        self.assertEqual(allowed.action, StrategyAction.ENTER)

    def test_regime_confirmation_blocks_without_enough_history(self) -> None:
        strategy = SessionBreakoutStrategy(
            SessionBreakoutConfig(
                lookback=8,
                breakout_buffer_bps=2.0,
                min_realized_volatility_bps=0.1,
                allowed_utc_hours=(12,),
                require_regime_confirmation=True,
                regime_lookback=12,
            )
        )

        decision = strategy.generate_decision(
            [1.1000, 1.1001, 1.1000, 1.1002, 1.1001, 1.1004, 1.1003, 1.1012],
            quote=QuoteSnapshot(
                timestamp=datetime(2026, 6, 22, 12, 0, tzinfo=ZoneInfo("UTC")),
                symbol="EURUSD",
                bid=1.10115,
                ask=1.10125,
            ),
        )

        self.assertEqual(decision.action, StrategyAction.NO_ACTION)
        self.assertIn("no regime reading", decision.reason)

    def test_regime_confirmation_allows_confirmed_trend(self) -> None:
        strategy = SessionBreakoutStrategy(
            SessionBreakoutConfig(
                lookback=8,
                breakout_buffer_bps=2.0,
                min_realized_volatility_bps=0.1,
                allowed_utc_hours=(12,),
                require_regime_confirmation=True,
                regime_lookback=12,
                regime_min_abs_slope_bps=0.01,
                regime_min_trend_efficiency=0.05,
                regime_max_realized_volatility_bps=5000.0,
            )
        )

        decision = strategy.generate_decision(
            [
                1.0000,
                1.0200,
                1.0400,
                1.0600,
                1.1000,
                1.1001,
                1.1000,
                1.1002,
                1.1001,
                1.1004,
                1.1003,
                1.1012,
            ],
            quote=QuoteSnapshot(
                timestamp=datetime(2026, 6, 22, 12, 0, tzinfo=ZoneInfo("UTC")),
                symbol="EURUSD",
                bid=1.10115,
                ask=1.10125,
            ),
        )

        self.assertEqual(decision.action, StrategyAction.ENTER)

    def test_minimum_holding_period_blocks_churn_exit(self) -> None:
        strategy = SessionBreakoutStrategy(
            SessionBreakoutConfig(
                lookback=8,
                breakout_buffer_bps=2.0,
                exit_buffer_bps=1.0,
                min_channel_width_bps=0.1,
                min_realized_volatility_bps=0.1,
                min_holding_period=3,
                allowed_utc_hours=(12,),
            )
        )

        decision = strategy.generate_decision(
            [1.1000, 1.1005, 1.1008, 1.1010, 1.1009, 1.1010, 1.10095, 1.10075],
            current_notional_usd=50_000,
            holding_period=1,
            quote=QuoteSnapshot(
                timestamp=datetime(2026, 6, 22, 12, 0, tzinfo=ZoneInfo("UTC")),
                symbol="EURUSD",
                bid=1.10070,
                ask=1.10080,
            ),
        )

        self.assertEqual(decision.action, StrategyAction.HOLD)
        self.assertIn("minimum holding period", decision.reason)

    def test_config_rejects_invalid_session_hour(self) -> None:
        with self.assertRaisesRegex(ValueError, "allowed_utc_hours"):
            SessionBreakoutConfig(allowed_utc_hours=(24,))

    def test_config_rejects_invalid_metal_session_hour(self) -> None:
        with self.assertRaisesRegex(ValueError, "metal_allowed_utc_hours"):
            SessionBreakoutConfig(metal_allowed_utc_hours=(-1,))


class AutocorrelationRegimeStrategyTest(TestCase):
    def test_positive_autocorrelation_enters_momentum_long(self) -> None:
        strategy = AutocorrelationRegimeStrategy(
            AutocorrelationRegimeConfig(
                lookback=8,
                signal_lookback=3,
                min_abs_autocorrelation=0.10,
                min_momentum_bps=2.0,
                min_expected_edge_bps=1.0,
                position_sizing="fixed",
                forex_allowed_utc_hours=(12,),
                max_spread_bps=5.0,
            )
        )

        decision = strategy.generate_decision(
            _prices_from_log_returns(
                [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
            ),
            quote=QuoteSnapshot(
                timestamp=datetime(2026, 6, 22, 12, 0, tzinfo=ZoneInfo("UTC")),
                symbol="EURUSD",
                bid=1.00275,
                ask=1.00285,
            ),
        )

        self.assertEqual(decision.action, StrategyAction.ENTER)
        self.assertGreater(decision.target_notional_usd, 0)
        self.assertEqual(decision.primary_signal, "autocorrelation_regime")
        self.assertIn("momentum", decision.reason)

    def test_negative_autocorrelation_fades_upward_stretch(self) -> None:
        strategy = AutocorrelationRegimeStrategy(
            AutocorrelationRegimeConfig(
                lookback=8,
                signal_lookback=3,
                min_abs_autocorrelation=0.10,
                min_reversion_zscore=0.80,
                min_reversion_move_bps=2.0,
                min_expected_edge_bps=1.0,
                position_sizing="fixed",
                forex_allowed_utc_hours=(12,),
                max_spread_bps=5.0,
            )
        )

        decision = strategy.generate_decision(
            _prices_from_log_returns(
                [0.0008, -0.0005, 0.0008, -0.0005, 0.0008, -0.0005, 0.0008]
            ),
            quote=QuoteSnapshot(
                timestamp=datetime(2026, 6, 22, 12, 0, tzinfo=ZoneInfo("UTC")),
                symbol="EURUSD",
                bid=1.00165,
                ask=1.00175,
            ),
        )

        self.assertEqual(decision.action, StrategyAction.ENTER)
        self.assertLess(decision.target_notional_usd, 0)
        self.assertIn("mean_reversion", decision.reason)

    def test_rejects_invalid_autocorrelation_threshold(self) -> None:
        with self.assertRaisesRegex(ValueError, "min_abs_autocorrelation"):
            AutocorrelationRegimeConfig(min_abs_autocorrelation=1.5)


class TrendPullbackStrategyTest(TestCase):
    def test_enters_long_after_orderly_pullback_and_resume(self) -> None:
        strategy = TrendPullbackStrategy(
            TrendPullbackConfig(
                lookback=8,
                pullback_window=2,
                min_trend_bps=10.0,
                min_pullback_bps=2.0,
                max_pullback_bps=20.0,
                min_resume_bps=2.0,
                min_expected_edge_bps=3.0,
                position_sizing="fixed",
                forex_allowed_utc_hours=(12,),
                max_spread_bps=5.0,
            )
        )

        decision = strategy.generate_decision(
            [1.0000, 1.0020, 1.0040, 1.0060, 1.0080, 1.0100, 1.0085, 1.0095],
            quote=QuoteSnapshot(
                timestamp=datetime(2026, 6, 22, 12, 0, tzinfo=ZoneInfo("UTC")),
                symbol="EURUSD",
                bid=1.00945,
                ask=1.00955,
            ),
        )

        self.assertEqual(decision.action, StrategyAction.ENTER)
        self.assertGreater(decision.target_notional_usd, 0)
        self.assertEqual(decision.primary_signal, "trend_pullback")
        self.assertIn("long trend pullback", decision.reason)

    def test_enters_short_after_orderly_pullback_and_resume(self) -> None:
        strategy = TrendPullbackStrategy(
            TrendPullbackConfig(
                lookback=8,
                pullback_window=2,
                min_trend_bps=10.0,
                min_pullback_bps=2.0,
                max_pullback_bps=20.0,
                min_resume_bps=2.0,
                min_expected_edge_bps=3.0,
                position_sizing="fixed",
                forex_allowed_utc_hours=(12,),
                max_spread_bps=5.0,
            )
        )

        decision = strategy.generate_decision(
            [1.0100, 1.0080, 1.0060, 1.0040, 1.0020, 1.0000, 1.0015, 1.0005],
            quote=QuoteSnapshot(
                timestamp=datetime(2026, 6, 22, 12, 0, tzinfo=ZoneInfo("UTC")),
                symbol="EURUSD",
                bid=1.00045,
                ask=1.00055,
            ),
        )

        self.assertEqual(decision.action, StrategyAction.ENTER)
        self.assertLess(decision.target_notional_usd, 0)
        self.assertIn("short trend pullback", decision.reason)

    def test_blocks_new_entry_outside_session(self) -> None:
        strategy = TrendPullbackStrategy(
            TrendPullbackConfig(
                lookback=8,
                pullback_window=2,
                min_trend_bps=10.0,
                min_pullback_bps=2.0,
                max_pullback_bps=20.0,
                min_resume_bps=2.0,
                forex_allowed_utc_hours=(12,),
            )
        )

        decision = strategy.generate_decision(
            [1.0000, 1.0020, 1.0040, 1.0060, 1.0080, 1.0100, 1.0085, 1.0095],
            quote=QuoteSnapshot(
                timestamp=datetime(2026, 6, 22, 3, 0, tzinfo=ZoneInfo("UTC")),
                symbol="EURUSD",
                bid=1.00945,
                ask=1.00955,
            ),
        )

        self.assertEqual(decision.action, StrategyAction.NO_ACTION)
        self.assertIn("outside trend pullback UTC hours", decision.reason)

    def test_config_rejects_invalid_pullback_hours(self) -> None:
        with self.assertRaisesRegex(ValueError, "forex_allowed_utc_hours"):
            TrendPullbackConfig(forex_allowed_utc_hours=(24,))


class DualSqueezeStrategyTest(TestCase):
    def test_enters_when_fast_breakout_has_slow_squeeze_bias_confirmation(self) -> None:
        strategy = DualSqueezeStrategy(
            DualSqueezeConfig(
                lookback=8,
                squeeze_window=2,
                band_stdev_multiplier=1.0,
                breakout_buffer_bps=1.0,
                max_squeeze_ratio=1.50,
                min_prior_volatility_bps=0.0,
                confirmation_lookback=10,
                confirmation_squeeze_window=2,
                confirmation_band_stdev_multiplier=2.0,
                confirmation_max_squeeze_ratio=1.50,
                confirmation_mode="squeeze_bias",
                position_sizing="fixed",
                forex_allowed_utc_hours=(12,),
                max_spread_bps=5.0,
            )
        )

        decision = strategy.generate_decision(
            [1.0000, 1.0005, 0.9995, 1.0002, 0.9998, 1.0001, 0.9999, 1.0000, 1.0001, 1.0040],
            quote=QuoteSnapshot(
                timestamp=datetime(2026, 6, 22, 12, 0, tzinfo=ZoneInfo("UTC")),
                symbol="EURUSD",
                bid=1.00395,
                ask=1.00405,
            ),
        )

        self.assertEqual(decision.action, StrategyAction.ENTER)
        self.assertGreater(decision.target_notional_usd, 0)
        self.assertEqual(decision.primary_signal, "dual_squeeze")
        self.assertIn("dual squeeze confirmed", decision.reason)

    def test_blocks_when_confirmation_bias_disagrees(self) -> None:
        strategy = DualSqueezeStrategy(
            DualSqueezeConfig(
                lookback=8,
                squeeze_window=2,
                band_stdev_multiplier=1.0,
                breakout_buffer_bps=1.0,
                max_squeeze_ratio=1.50,
                min_prior_volatility_bps=0.0,
                confirmation_lookback=10,
                confirmation_squeeze_window=2,
                confirmation_band_stdev_multiplier=4.0,
                confirmation_max_squeeze_ratio=1.50,
                confirmation_mode="bias",
                position_sizing="fixed",
                forex_allowed_utc_hours=(12,),
                max_spread_bps=5.0,
            )
        )

        decision = strategy.generate_decision(
            [1.0200, 1.0200, 1.0000, 1.0002, 0.9998, 1.0001, 0.9999, 1.0000, 1.0001, 1.0040],
            quote=QuoteSnapshot(
                timestamp=datetime(2026, 6, 22, 12, 0, tzinfo=ZoneInfo("UTC")),
                symbol="EURUSD",
                bid=1.00395,
                ask=1.00405,
            ),
        )

        self.assertEqual(decision.action, StrategyAction.NO_ACTION)
        self.assertIn("dual squeeze blocked", decision.reason)

    def test_config_rejects_invalid_confirmation_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "confirmation_mode"):
            DualSqueezeConfig(confirmation_mode="maybe")


class AssetAdaptiveDualSqueezeStrategyTest(TestCase):
    def test_uses_metal_fast_profile_for_metals(self) -> None:
        strategy = AssetAdaptiveDualSqueezeStrategy(
            AssetAdaptiveDualSqueezeConfig(symbol="XAUUSD")
        )

        self.assertEqual(strategy.selected_profile, "metal_fast")
        self.assertEqual(strategy.inner_config.lookback, 12)
        self.assertEqual(strategy.inner_config.breakout_buffer_bps, 2.0)
        self.assertEqual(strategy.inner_config.confirmation_lookback, 20)

    def test_uses_base_profile_for_fx(self) -> None:
        strategy = AssetAdaptiveDualSqueezeStrategy(
            AssetAdaptiveDualSqueezeConfig(symbol="EURUSD")
        )

        self.assertEqual(strategy.selected_profile, "base")
        self.assertEqual(strategy.inner_config.lookback, 14)
        self.assertEqual(strategy.inner_config.breakout_buffer_bps, 2.5)
        self.assertEqual(strategy.inner_config.confirmation_lookback, 24)

    def test_relabels_inner_decision_primary_signal(self) -> None:
        strategy = AssetAdaptiveDualSqueezeStrategy(
            AssetAdaptiveDualSqueezeConfig(
                symbol="EURUSD",
                base_lookback=8,
                base_squeeze_window=2,
                base_band_stdev_multiplier=1.0,
                base_breakout_buffer_bps=1.0,
                base_max_squeeze_ratio=1.50,
                base_confirmation_lookback=10,
                base_confirmation_squeeze_window=2,
                base_confirmation_band_stdev_multiplier=2.0,
                base_confirmation_max_squeeze_ratio=1.50,
                min_prior_volatility_bps=0.0,
                position_sizing="fixed",
                forex_allowed_utc_hours=(12,),
                max_spread_bps=5.0,
            )
        )

        decision = strategy.generate_decision(
            [
                1.0000,
                1.0005,
                0.9995,
                1.0002,
                0.9998,
                1.0001,
                0.9999,
                1.0000,
                1.0001,
                1.0040,
            ],
            quote=QuoteSnapshot(
                timestamp=datetime(2026, 6, 22, 12, 0, tzinfo=ZoneInfo("UTC")),
                symbol="EURUSD",
                bid=1.00395,
                ask=1.00405,
            ),
        )

        self.assertEqual(decision.action, StrategyAction.ENTER)
        self.assertEqual(decision.primary_signal, "asset_adaptive_dual_squeeze")
        self.assertIn("dual_squeeze", decision.supporting_signals)
        self.assertIn("asset-adaptive dual squeeze", decision.reason)


class ExhaustionReversalStrategyTest(TestCase):
    def test_enters_long_after_down_shock_and_reversal(self) -> None:
        strategy = ExhaustionReversalStrategy(
            ExhaustionReversalConfig(
                lookback=8,
                shock_window=3,
                min_shock_bps=10.0,
                min_reversal_bps=2.0,
                min_shock_zscore=0.10,
                min_shock_efficiency=0.50,
                min_expected_edge_bps=3.0,
                position_sizing="fixed",
                forex_allowed_utc_hours=(12,),
                max_spread_bps=5.0,
            )
        )

        decision = strategy.generate_decision(
            [1.0000, 1.0002, 1.0001, 1.0000, 0.9980, 0.9960, 0.9940, 0.9950],
            quote=QuoteSnapshot(
                timestamp=datetime(2026, 6, 22, 12, 0, tzinfo=ZoneInfo("UTC")),
                symbol="EURUSD",
                bid=0.99495,
                ask=0.99505,
            ),
        )

        self.assertEqual(decision.action, StrategyAction.ENTER)
        self.assertGreater(decision.target_notional_usd, 0)
        self.assertEqual(decision.primary_signal, "exhaustion_reversal")
        self.assertIn("long exhaustion reversal", decision.reason)

    def test_enters_short_after_up_shock_and_reversal(self) -> None:
        strategy = ExhaustionReversalStrategy(
            ExhaustionReversalConfig(
                lookback=8,
                shock_window=3,
                min_shock_bps=10.0,
                min_reversal_bps=2.0,
                min_shock_zscore=0.10,
                min_shock_efficiency=0.50,
                min_expected_edge_bps=3.0,
                position_sizing="fixed",
                forex_allowed_utc_hours=(12,),
                max_spread_bps=5.0,
            )
        )

        decision = strategy.generate_decision(
            [1.0000, 0.9998, 0.9999, 1.0000, 1.0020, 1.0040, 1.0060, 1.0050],
            quote=QuoteSnapshot(
                timestamp=datetime(2026, 6, 22, 12, 0, tzinfo=ZoneInfo("UTC")),
                symbol="EURUSD",
                bid=1.00495,
                ask=1.00505,
            ),
        )

        self.assertEqual(decision.action, StrategyAction.ENTER)
        self.assertLess(decision.target_notional_usd, 0)
        self.assertIn("short exhaustion reversal", decision.reason)

    def test_blocks_new_entry_outside_session(self) -> None:
        strategy = ExhaustionReversalStrategy(
            ExhaustionReversalConfig(
                lookback=8,
                shock_window=3,
                min_shock_bps=10.0,
                min_reversal_bps=2.0,
                min_shock_zscore=0.10,
                min_shock_efficiency=0.50,
                forex_allowed_utc_hours=(12,),
            )
        )

        decision = strategy.generate_decision(
            [1.0000, 1.0002, 1.0001, 1.0000, 0.9980, 0.9960, 0.9940, 0.9950],
            quote=QuoteSnapshot(
                timestamp=datetime(2026, 6, 22, 3, 0, tzinfo=ZoneInfo("UTC")),
                symbol="EURUSD",
                bid=0.99495,
                ask=0.99505,
            ),
        )

        self.assertEqual(decision.action, StrategyAction.NO_ACTION)
        self.assertIn("outside exhaustion reversal UTC hours", decision.reason)

    def test_config_rejects_invalid_exhaustion_hours(self) -> None:
        with self.assertRaisesRegex(ValueError, "forex_allowed_utc_hours"):
            ExhaustionReversalConfig(forex_allowed_utc_hours=(24,))

    def test_config_rejects_too_short_baseline(self) -> None:
        with self.assertRaisesRegex(ValueError, "pre-shock baseline"):
            ExhaustionReversalConfig(lookback=8, shock_window=4)


class LiquiditySweepReversalStrategyTest(TestCase):
    def test_enters_short_after_high_sweep_closes_back_inside_range(self) -> None:
        strategy = LiquiditySweepReversalStrategy(
            LiquiditySweepReversalConfig(
                lookback=6,
                min_sweep_bps=2.0,
                reentry_buffer_bps=0.2,
                min_range_width_bps=5.0,
                max_trend_efficiency=1.0,
                min_expected_edge_bps=1.0,
                position_sizing="fixed",
                forex_allowed_utc_hours=(12,),
                slippage_bps=0.1,
                max_spread_bps=5.0,
            )
        )

        decision = strategy.generate_decision(
            [1.0000, 1.0010, 1.0005, 1.0008, 1.0015, 1.0009],
            quote=QuoteSnapshot(
                timestamp=datetime(2026, 6, 22, 12, 0, tzinfo=ZoneInfo("UTC")),
                symbol="EURUSD",
                bid=1.00085,
                ask=1.00095,
            ),
        )

        self.assertEqual(decision.action, StrategyAction.ENTER)
        self.assertLess(decision.target_notional_usd, 0)
        self.assertEqual(decision.primary_signal, "liquidity_sweep_reversal")
        self.assertIn("short liquidity sweep reversal", decision.reason)
        self.assertEqual(
            dict(decision.diagnostics)["liquidity_sweep_signal_direction"],
            "SHORT",
        )

    def test_enters_long_after_low_sweep_closes_back_inside_range(self) -> None:
        strategy = LiquiditySweepReversalStrategy(
            LiquiditySweepReversalConfig(
                lookback=6,
                min_sweep_bps=2.0,
                reentry_buffer_bps=0.2,
                min_range_width_bps=5.0,
                max_trend_efficiency=1.0,
                min_expected_edge_bps=1.0,
                position_sizing="fixed",
                forex_allowed_utc_hours=(12,),
                slippage_bps=0.1,
                max_spread_bps=5.0,
            )
        )

        decision = strategy.generate_decision(
            [1.0000, 1.0010, 1.0005, 1.0008, 0.9995, 1.0001],
            quote=QuoteSnapshot(
                timestamp=datetime(2026, 6, 22, 12, 0, tzinfo=ZoneInfo("UTC")),
                symbol="EURUSD",
                bid=1.00005,
                ask=1.00015,
            ),
        )

        self.assertEqual(decision.action, StrategyAction.ENTER)
        self.assertGreater(decision.target_notional_usd, 0)
        self.assertIn("long liquidity sweep reversal", decision.reason)

    def test_exits_short_when_range_midpoint_is_reached(self) -> None:
        strategy = LiquiditySweepReversalStrategy(
            LiquiditySweepReversalConfig(
                lookback=6,
                min_sweep_bps=2.0,
                min_expected_edge_bps=1.0,
                position_sizing="fixed",
            )
        )

        decision = strategy.generate_decision(
            [1.0000, 1.0010, 1.0005, 1.0008, 1.0009, 1.0004],
            current_notional_usd=-50_000,
            holding_period=2,
        )

        self.assertEqual(decision.action, StrategyAction.EXIT)
        self.assertEqual(decision.target_notional_usd, 0.0)
        self.assertIn("range midpoint", decision.reason)

    def test_blocks_fade_when_baseline_trend_is_too_efficient(self) -> None:
        strategy = LiquiditySweepReversalStrategy(
            LiquiditySweepReversalConfig(
                lookback=6,
                min_sweep_bps=2.0,
                reentry_buffer_bps=0.2,
                min_range_width_bps=5.0,
                max_trend_efficiency=0.50,
                min_expected_edge_bps=1.0,
                position_sizing="fixed",
                forex_allowed_utc_hours=(12,),
                slippage_bps=0.1,
                max_spread_bps=5.0,
            )
        )

        decision = strategy.generate_decision(
            [1.0000, 1.0010, 1.0020, 1.0030, 1.0035, 1.0029],
            quote=QuoteSnapshot(
                timestamp=datetime(2026, 6, 22, 12, 0, tzinfo=ZoneInfo("UTC")),
                symbol="EURUSD",
                bid=1.00285,
                ask=1.00295,
            ),
        )

        self.assertEqual(decision.action, StrategyAction.NO_ACTION)
        self.assertIn("trend efficiency", decision.reason)

    def test_config_rejects_invalid_sweep_bounds(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_sweep_bps"):
            LiquiditySweepReversalConfig(min_sweep_bps=5.0, max_sweep_bps=5.0)


class FixingReversalStrategyTest(TestCase):
    def test_enters_short_after_up_pre_fix_move_and_reversal_confirmation(self) -> None:
        strategy = FixingReversalStrategy(
            FixingReversalConfig(
                pre_fix_lookback=4,
                min_pre_fix_move_bps=5.0,
                min_reversal_confirmation_bps=0.5,
                min_pre_fix_efficiency=0.40,
                min_realized_volatility_bps=0.0,
                min_expected_edge_bps=2.0,
                position_sizing="fixed",
                forex_allowed_utc_hours=(16,),
                max_spread_bps=5.0,
            )
        )

        decision = strategy.generate_decision(
            [1.0000, 1.0010, 1.0020, 1.0030, 1.0040, 1.0035],
            quote=QuoteSnapshot(
                timestamp=datetime(2026, 6, 22, 16, 0, tzinfo=ZoneInfo("UTC")),
                symbol="EURUSD",
                bid=1.00345,
                ask=1.00355,
            ),
        )

        self.assertEqual(decision.action, StrategyAction.ENTER)
        self.assertLess(decision.target_notional_usd, 0)
        self.assertEqual(decision.primary_signal, "fixing_reversal")
        self.assertIn("short fixing reversal", decision.reason)

    def test_enters_long_after_down_pre_fix_move_and_reversal_confirmation(self) -> None:
        strategy = FixingReversalStrategy(
            FixingReversalConfig(
                pre_fix_lookback=4,
                min_pre_fix_move_bps=5.0,
                min_reversal_confirmation_bps=0.5,
                min_pre_fix_efficiency=0.40,
                min_realized_volatility_bps=0.0,
                min_expected_edge_bps=2.0,
                position_sizing="fixed",
                forex_allowed_utc_hours=(16,),
                max_spread_bps=5.0,
            )
        )

        decision = strategy.generate_decision(
            [1.0040, 1.0030, 1.0020, 1.0010, 1.0000, 1.0005],
            quote=QuoteSnapshot(
                timestamp=datetime(2026, 6, 22, 16, 0, tzinfo=ZoneInfo("UTC")),
                symbol="EURUSD",
                bid=1.00045,
                ask=1.00055,
            ),
        )

        self.assertEqual(decision.action, StrategyAction.ENTER)
        self.assertGreater(decision.target_notional_usd, 0)
        self.assertIn("long fixing reversal", decision.reason)

    def test_blocks_new_entry_outside_fixing_window(self) -> None:
        strategy = FixingReversalStrategy(
            FixingReversalConfig(
                pre_fix_lookback=4,
                min_pre_fix_move_bps=5.0,
                min_realized_volatility_bps=0.0,
                forex_allowed_utc_hours=(16,),
            )
        )

        decision = strategy.generate_decision(
            [1.0000, 1.0010, 1.0020, 1.0030, 1.0040, 1.0035],
            quote=QuoteSnapshot(
                timestamp=datetime(2026, 6, 22, 15, 0, tzinfo=ZoneInfo("UTC")),
                symbol="EURUSD",
                bid=1.00345,
                ask=1.00355,
            ),
        )

        self.assertEqual(decision.action, StrategyAction.NO_ACTION)
        self.assertIn("outside fixing reversal UTC hours", decision.reason)

    def test_exits_existing_position_after_fixing_window(self) -> None:
        strategy = FixingReversalStrategy(
            FixingReversalConfig(
                pre_fix_lookback=4,
                min_realized_volatility_bps=0.0,
                forex_allowed_utc_hours=(16,),
            )
        )

        decision = strategy.generate_decision(
            [1.0000, 1.0010, 1.0020, 1.0030, 1.0040, 1.0035],
            current_notional_usd=-50_000,
            holding_period=2,
            quote=QuoteSnapshot(
                timestamp=datetime(2026, 6, 22, 18, 0, tzinfo=ZoneInfo("UTC")),
                symbol="EURUSD",
                bid=1.00345,
                ask=1.00355,
            ),
        )

        self.assertEqual(decision.action, StrategyAction.EXIT)
        self.assertEqual(decision.target_notional_usd, 0.0)

    def test_crypto_is_disabled_by_default(self) -> None:
        strategy = FixingReversalStrategy(
            FixingReversalConfig(
                symbol="BTCUSD",
                pre_fix_lookback=4,
                min_pre_fix_move_bps=5.0,
                min_realized_volatility_bps=0.0,
            )
        )

        decision = strategy.generate_decision(
            [100.0, 101.0, 102.0, 103.0, 104.0, 103.5],
            quote=QuoteSnapshot(
                timestamp=datetime(2026, 6, 22, 16, 0, tzinfo=ZoneInfo("UTC")),
                symbol="BTCUSD",
                bid=103.4,
                ask=103.6,
            ),
        )

        self.assertEqual(decision.action, StrategyAction.NO_ACTION)
        self.assertIn("outside fixing reversal UTC hours", decision.reason)

    def test_config_rejects_invalid_fixing_bounds(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_pre_fix_move_bps"):
            FixingReversalConfig(
                min_pre_fix_move_bps=5.0,
                max_pre_fix_move_bps=5.0,
            )


class KalmanTrendStrategyTest(TestCase):
    def test_enters_long_on_clean_uptrend(self) -> None:
        strategy = KalmanTrendStrategy(
            KalmanTrendStrategyConfig(
                lookback=20,
                min_abs_slope_bps=0.1,
                min_trend_efficiency=0.20,
                min_expected_edge_bps=1.0,
                position_sizing="fixed",
                max_spread_bps=5.0,
            )
        )

        decision = strategy.generate_decision(
            [1.0000 + index * 0.0010 for index in range(30)],
            quote=QuoteSnapshot(
                timestamp=datetime(2026, 6, 22, 12, 0, tzinfo=ZoneInfo("UTC")),
                symbol="EURUSD",
                bid=1.02895,
                ask=1.02905,
            ),
        )

        self.assertEqual(decision.action, StrategyAction.ENTER)
        self.assertGreater(decision.target_notional_usd, 0)
        self.assertEqual(decision.primary_signal, "kalman_trend")
        self.assertIn("long Kalman trend", decision.reason)

    def test_enters_short_on_clean_downtrend(self) -> None:
        strategy = KalmanTrendStrategy(
            KalmanTrendStrategyConfig(
                lookback=20,
                min_abs_slope_bps=0.1,
                min_trend_efficiency=0.20,
                min_expected_edge_bps=1.0,
                position_sizing="fixed",
                max_spread_bps=5.0,
            )
        )

        decision = strategy.generate_decision(
            [1.0300 - index * 0.0010 for index in range(30)],
            quote=QuoteSnapshot(
                timestamp=datetime(2026, 6, 22, 12, 0, tzinfo=ZoneInfo("UTC")),
                symbol="EURUSD",
                bid=1.00095,
                ask=1.00105,
            ),
        )

        self.assertEqual(decision.action, StrategyAction.ENTER)
        self.assertLess(decision.target_notional_usd, 0)
        self.assertIn("short Kalman trend", decision.reason)

    def test_chop_exits_existing_position(self) -> None:
        strategy = KalmanTrendStrategy(
            KalmanTrendStrategyConfig(
                lookback=20,
                min_abs_slope_bps=5.0,
                min_trend_efficiency=0.80,
                min_expected_edge_bps=1.0,
            )
        )
        prices = [1.0 + (0.0001 if index % 2 == 0 else -0.0001) for index in range(30)]

        decision = strategy.generate_decision(
            prices,
            current_notional_usd=50_000,
        )

        self.assertEqual(decision.action, StrategyAction.EXIT)
        self.assertIn("no longer trends", decision.reason)

    def test_config_rejects_invalid_kalman_hours(self) -> None:
        with self.assertRaisesRegex(ValueError, "forex_allowed_utc_hours"):
            KalmanTrendStrategyConfig(forex_allowed_utc_hours=(24,))


class QualityTrendStrategyTest(TestCase):
    def _config(self) -> QualityTrendConfig:
        return QualityTrendConfig(
            kalman_lookback=50,
            kalman_min_abs_slope_bps=0.01,
            kalman_min_trend_efficiency=0.01,
            kalman_min_expected_edge_bps=0.1,
            macd_min_histogram_bps=0.01,
            macd_exit_histogram_bps=0.001,
            macd_min_macd_bps=0.01,
            macd_min_trend_efficiency=0.0,
            min_combined_confidence=0.01,
            exit_combined_confidence=0.0,
            min_expected_edge_bps=0.01,
            position_sizing="fixed",
            max_spread_bps=5.0,
        )

    def test_enters_long_when_macd_and_kalman_align(self) -> None:
        strategy = QualityTrendStrategy(self._config())
        prices = [1.0800 - index * 0.0005 for index in range(65)] + [
            1.0475 + index * 0.0012 for index in range(15)
        ]

        decision = strategy.generate_decision(
            prices,
            quote=QuoteSnapshot(
                timestamp=datetime(2026, 6, 22, 12, 0, tzinfo=ZoneInfo("UTC")),
                symbol="EURUSD",
                bid=1.06425,
                ask=1.06435,
            ),
        )

        self.assertEqual(decision.action, StrategyAction.ENTER)
        self.assertGreater(decision.target_notional_usd, 0)
        self.assertEqual(decision.primary_signal, "quality_trend")
        self.assertIn("long quality trend", decision.reason)

    def test_blocks_when_component_gate_fails(self) -> None:
        strategy = QualityTrendStrategy(self._config())
        prices = [1.0000 + index * 0.0005 for index in range(78)] + [
            1.0390,
            1.0386,
        ]

        decision = strategy.generate_decision(
            prices,
            quote=QuoteSnapshot(
                timestamp=datetime(2026, 6, 22, 12, 0, tzinfo=ZoneInfo("UTC")),
                symbol="EURUSD",
                bid=1.03855,
                ask=1.03865,
            ),
        )

        self.assertEqual(decision.action, StrategyAction.NO_ACTION)
        self.assertIn("MACD gate failed", decision.reason)

    def test_exits_when_alignment_fades(self) -> None:
        strategy = QualityTrendStrategy(self._config())
        prices = [1.0 + (0.0001 if index % 2 == 0 else -0.0001) for index in range(80)]

        decision = strategy.generate_decision(
            prices,
            current_notional_usd=50_000,
            holding_period=2,
        )

        self.assertEqual(decision.action, StrategyAction.EXIT)
        self.assertIn("opposite quality trend", decision.reason)

    def test_config_rejects_invalid_confidence_band(self) -> None:
        with self.assertRaisesRegex(ValueError, "exit_combined_confidence"):
            QualityTrendConfig(
                min_combined_confidence=0.50,
                exit_combined_confidence=0.50,
            )


class RangeExpansionTrendStrategyTest(TestCase):
    def _config(self) -> RangeExpansionTrendConfig:
        return RangeExpansionTrendConfig(
            lookback=12,
            trigger_window=3,
            min_trigger_move_bps=4.0,
            exit_trigger_move_bps=1.0,
            min_range_break_bps=1.0,
            min_expansion_zscore=0.5,
            max_expansion_zscore=100.0,
            min_trend_efficiency=0.60,
            min_baseline_volatility_bps=0.01,
            min_expected_edge_bps=1.0,
            position_sizing="fixed",
            max_spread_bps=5.0,
        )

    def test_enters_long_on_clean_range_expansion(self) -> None:
        strategy = RangeExpansionTrendStrategy(self._config())
        prices = [
            1.0000,
            1.0001,
            0.9999,
            1.0002,
            1.0000,
            1.00015,
            0.99995,
            1.0001,
            1.0002,
            1.0008,
            1.0014,
            1.0020,
        ]

        decision = strategy.generate_decision(
            prices,
            quote=QuoteSnapshot(
                timestamp=datetime(2026, 6, 22, 12, 0, tzinfo=ZoneInfo("UTC")),
                symbol="EURUSD",
                bid=1.00195,
                ask=1.00205,
            ),
        )

        self.assertEqual(decision.action, StrategyAction.ENTER)
        self.assertGreater(decision.target_notional_usd, 0)
        self.assertEqual(decision.primary_signal, "range_expansion_trend")
        self.assertIn("long range expansion", decision.reason)

    def test_blocks_when_breakout_stays_inside_range(self) -> None:
        strategy = RangeExpansionTrendStrategy(self._config())
        prices = [
            1.0000,
            1.0001,
            0.9999,
            1.0002,
            1.0000,
            1.00015,
            0.99995,
            1.0001,
            1.0002,
            1.0003,
            1.0005,
            1.0001,
        ]

        decision = strategy.generate_decision(
            prices,
            quote=QuoteSnapshot(
                timestamp=datetime(2026, 6, 22, 12, 0, tzinfo=ZoneInfo("UTC")),
                symbol="EURUSD",
                bid=1.00005,
                ask=1.00015,
            ),
        )

        self.assertEqual(decision.action, StrategyAction.NO_ACTION)
        self.assertIn("not aligned", decision.reason)

    def test_exits_when_long_fades_back_inside_prior_range(self) -> None:
        strategy = RangeExpansionTrendStrategy(self._config())
        prices = [
            1.0000,
            1.0001,
            0.9999,
            1.0002,
            1.0000,
            1.00015,
            0.99995,
            1.0001,
            1.0002,
            1.0009,
            1.0005,
            1.0001,
        ]

        decision = strategy.generate_decision(
            prices,
            current_notional_usd=50_000,
            holding_period=2,
            quote=QuoteSnapshot(
                timestamp=datetime(2026, 6, 22, 12, 0, tzinfo=ZoneInfo("UTC")),
                symbol="EURUSD",
                bid=1.00005,
                ask=1.00015,
            ),
        )

        self.assertEqual(decision.action, StrategyAction.EXIT)
        self.assertIn("faded back", decision.reason)

    def test_config_rejects_invalid_trigger_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "trigger_window"):
            RangeExpansionTrendConfig(trigger_window=1)
        with self.assertRaisesRegex(ValueError, "baseline prices"):
            RangeExpansionTrendConfig(lookback=8, trigger_window=4)


class ChampionEnsembleStrategyTest(TestCase):
    def test_enters_on_kalman_lead_signal(self) -> None:
        strategy = ChampionEnsembleStrategy(
            config=ChampionEnsembleConfig(entry_score=0.50),
            kalman_trend_strategy=_FixedDecisionStrategy(
                _fixed_decision(
                    StrategyAction.ENTER,
                    symbol="EURUSD",
                    target_notional_usd=75_000,
                    reason="kalman trend up",
                )
            ),
            asset_adaptive_dual_squeeze_strategy=_FixedDecisionStrategy(
                _fixed_decision(
                    StrategyAction.NO_ACTION,
                    symbol="EURUSD",
                    target_notional_usd=0,
                    reason="no squeeze",
                )
            ),
            dual_squeeze_strategy=_FixedDecisionStrategy(
                _fixed_decision(
                    StrategyAction.NO_ACTION,
                    symbol="EURUSD",
                    target_notional_usd=0,
                    reason="no dual squeeze",
                )
            ),
            trend_pullback_strategy=_FixedDecisionStrategy(
                _fixed_decision(
                    StrategyAction.NO_ACTION,
                    symbol="EURUSD",
                    target_notional_usd=0,
                    reason="no pullback",
                )
            ),
        )

        decision = strategy.generate_decision([1.0, 1.1, 1.2, 1.3, 1.4])

        self.assertEqual(decision.action, StrategyAction.ENTER)
        self.assertGreater(decision.target_notional_usd, 0)
        self.assertEqual(decision.primary_signal, "kalman_trend")
        self.assertIn("kalman_trend", decision.supporting_signals)

    def test_allows_asset_adaptive_squeeze_as_strong_standalone_lead(self) -> None:
        strategy = ChampionEnsembleStrategy(
            config=ChampionEnsembleConfig(strong_lead_score=0.25),
            kalman_trend_strategy=_FixedDecisionStrategy(
                _fixed_decision(
                    StrategyAction.NO_ACTION,
                    symbol="EURUSD",
                    target_notional_usd=0,
                    reason="no trend",
                )
            ),
            asset_adaptive_dual_squeeze_strategy=_FixedDecisionStrategy(
                _fixed_decision(
                    StrategyAction.ENTER,
                    symbol="EURUSD",
                    target_notional_usd=-50_000,
                    reason="metal squeeze style signal",
                )
            ),
            dual_squeeze_strategy=_FixedDecisionStrategy(
                _fixed_decision(
                    StrategyAction.NO_ACTION,
                    symbol="EURUSD",
                    target_notional_usd=0,
                    reason="no dual squeeze",
                )
            ),
            trend_pullback_strategy=_FixedDecisionStrategy(
                _fixed_decision(
                    StrategyAction.NO_ACTION,
                    symbol="EURUSD",
                    target_notional_usd=0,
                    reason="no pullback",
                )
            ),
        )

        decision = strategy.generate_decision([1.0, 1.1, 1.2, 1.3, 1.4])

        self.assertEqual(decision.action, StrategyAction.ENTER)
        self.assertLess(decision.target_notional_usd, 0)
        self.assertEqual(decision.primary_signal, "asset_adaptive_dual_squeeze")

    def test_allows_fixing_reversal_as_weighted_component(self) -> None:
        strategy = ChampionEnsembleStrategy(
            config=ChampionEnsembleConfig(
                kalman_trend_weight=0.0,
                asset_adaptive_dual_squeeze_weight=0.0,
                dual_squeeze_weight=0.0,
                trend_pullback_weight=0.0,
                fixing_reversal_weight=1.0,
                entry_score=0.50,
            ),
            fixing_reversal_strategy=_FixedDecisionStrategy(
                _fixed_decision(
                    StrategyAction.ENTER,
                    symbol="EURUSD",
                    target_notional_usd=50_000,
                    reason="fixing reversal long",
                )
            ),
        )

        decision = strategy.generate_decision([1.0, 1.1, 1.2, 1.3, 1.4])

        self.assertEqual(decision.action, StrategyAction.ENTER)
        self.assertGreater(decision.target_notional_usd, 0)
        self.assertEqual(decision.primary_signal, "fixing_reversal")
        self.assertIn("fixing_reversal", decision.supporting_signals)

    def test_conflict_blocks_new_position(self) -> None:
        strategy = ChampionEnsembleStrategy(
            kalman_trend_strategy=_FixedDecisionStrategy(
                _fixed_decision(
                    StrategyAction.ENTER,
                    symbol="EURUSD",
                    target_notional_usd=75_000,
                    reason="kalman long",
                )
            ),
            asset_adaptive_dual_squeeze_strategy=_FixedDecisionStrategy(
                _fixed_decision(
                    StrategyAction.ENTER,
                    symbol="EURUSD",
                    target_notional_usd=-50_000,
                    reason="squeeze short",
                )
            ),
            dual_squeeze_strategy=_FixedDecisionStrategy(
                _fixed_decision(
                    StrategyAction.NO_ACTION,
                    symbol="EURUSD",
                    target_notional_usd=0,
                    reason="no dual squeeze",
                )
            ),
            trend_pullback_strategy=_FixedDecisionStrategy(
                _fixed_decision(
                    StrategyAction.NO_ACTION,
                    symbol="EURUSD",
                    target_notional_usd=0,
                    reason="no pullback",
                )
            ),
        )

        decision = strategy.generate_decision([1.0, 1.1, 1.2, 1.3, 1.4])

        self.assertEqual(decision.action, StrategyAction.NO_ACTION)
        self.assertEqual(decision.primary_signal, "kalman_trend")
        self.assertIn("asset_adaptive_dual_squeeze", decision.conflicting_signals)

    def test_config_rejects_invalid_weight(self) -> None:
        with self.assertRaisesRegex(ValueError, "kalman_trend_weight"):
            ChampionEnsembleConfig(kalman_trend_weight=-0.1)
        with self.assertRaisesRegex(ValueError, "fixing_reversal_weight"):
            ChampionEnsembleConfig(fixing_reversal_weight=-0.1)
        with self.assertRaisesRegex(ValueError, "macd_momentum_weight"):
            ChampionEnsembleConfig(macd_momentum_weight=-0.1)


class MovingAverageCrossoverStrategyTest(TestCase):
    def test_fast_average_above_slow_generates_buy_request(self) -> None:
        strategy = MovingAverageCrossoverStrategy(
            MovingAverageCrossoverConfig(
                fast_window=2,
                slow_window=5,
                min_separation_bps=1.0,
            )
        )

        request = strategy.generate_request([1.1000, 1.1001, 1.1002, 1.1005, 1.1010])

        self.assertIsNotNone(request)
        assert request is not None
        self.assertEqual(request.side, Side.BUY)

    def test_fast_average_below_slow_generates_sell_request(self) -> None:
        strategy = MovingAverageCrossoverStrategy(
            MovingAverageCrossoverConfig(
                fast_window=2,
                slow_window=5,
                min_separation_bps=1.0,
            )
        )

        request = strategy.generate_request([1.1000, 1.0999, 1.0998, 1.0995, 1.0990])

        self.assertIsNotNone(request)
        assert request is not None
        self.assertEqual(request.side, Side.SELL)

    def test_small_average_separation_generates_no_request(self) -> None:
        strategy = MovingAverageCrossoverStrategy(
            MovingAverageCrossoverConfig(
                fast_window=2,
                slow_window=5,
                min_separation_bps=2.0,
            )
        )

        request = strategy.generate_request([1.1000, 1.1001, 1.1000, 1.1001, 1.1000])

        self.assertIsNone(request)

    def test_reading_detects_fresh_long_cross(self) -> None:
        strategy = MovingAverageCrossoverStrategy(
            MovingAverageCrossoverConfig(
                fast_window=2,
                slow_window=5,
                min_separation_bps=1.0,
            )
        )

        reading = strategy.read_crossover(
            [1.1000, 1.1002, 1.1001, 1.1000, 1.1001, 1.1010]
        )

        self.assertIsNotNone(reading)
        assert reading is not None
        self.assertEqual(reading.crossed_direction, SignalDirection.LONG)
        self.assertGreater(reading.separation_bps, 0.0)
        self.assertLess(reading.previous_separation_bps or 0.0, 0.0)

    def test_crossover_exit_hysteresis_exits_when_averages_converge(self) -> None:
        strategy = MovingAverageCrossoverStrategy(
            MovingAverageCrossoverConfig(
                fast_window=2,
                slow_window=5,
                min_separation_bps=2.0,
                exit_separation_bps=0.5,
            )
        )

        decision = strategy.generate_decision(
            [1.1000, 1.1001, 1.1000, 1.1001, 1.1000],
            current_notional_usd=50_000,
        )

        self.assertEqual(decision.action, StrategyAction.EXIT)

    def test_crossover_cost_filter_blocks_wide_spread(self) -> None:
        strategy = MovingAverageCrossoverStrategy(
            MovingAverageCrossoverConfig(
                fast_window=2,
                slow_window=5,
                min_separation_bps=1.0,
                max_spread_bps=5.0,
            )
        )

        decision = strategy.generate_decision(
            [1.1000, 1.1001, 1.1002, 1.1005, 1.1010],
            quote=quote(bid=1.0900, ask=1.1100),
        )

        self.assertEqual(decision.action, StrategyAction.NO_ACTION)
        self.assertIn("spread", decision.reason)

    def test_crossover_config_rejects_fast_window_at_or_above_slow_window(self) -> None:
        with self.assertRaisesRegex(ValueError, "slow_window"):
            MovingAverageCrossoverConfig(fast_window=5, slow_window=5)


class MacdMomentumStrategyTest(TestCase):
    def test_rising_histogram_generates_buy_request(self) -> None:
        strategy = MacdMomentumStrategy(
            MacdMomentumConfig(
                fast_window=3,
                slow_window=6,
                signal_window=3,
                min_histogram_bps=0.01,
                exit_histogram_bps=0.001,
                min_macd_bps=0.01,
                min_trend_efficiency=0.0,
                position_sizing="fixed",
            )
        )

        request = strategy.generate_request(
            [1.1000, 1.1001, 1.1002, 1.1004, 1.1008, 1.1014, 1.1022, 1.1032, 1.1044, 1.1058, 1.1074, 1.1092, 1.1112, 1.1134, 1.1158, 1.1184, 1.1212]
        )

        self.assertIsNotNone(request)
        assert request is not None
        self.assertEqual(request.side, Side.BUY)

    def test_falling_histogram_generates_sell_request(self) -> None:
        strategy = MacdMomentumStrategy(
            MacdMomentumConfig(
                fast_window=3,
                slow_window=6,
                signal_window=3,
                min_histogram_bps=0.01,
                exit_histogram_bps=0.001,
                min_macd_bps=0.01,
                min_trend_efficiency=0.0,
                position_sizing="fixed",
            )
        )

        request = strategy.generate_request(
            [1.1212, 1.1210, 1.1208, 1.1204, 1.1198, 1.1190, 1.1180, 1.1168, 1.1154, 1.1138, 1.1120, 1.1100, 1.1078, 1.1054, 1.1028, 1.1000, 1.0970]
        )

        self.assertIsNotNone(request)
        assert request is not None
        self.assertEqual(request.side, Side.SELL)

    def test_session_filter_blocks_new_entry(self) -> None:
        strategy = MacdMomentumStrategy(
            MacdMomentumConfig(
                fast_window=3,
                slow_window=6,
                signal_window=3,
                min_histogram_bps=0.01,
                exit_histogram_bps=0.001,
                min_macd_bps=0.01,
                min_trend_efficiency=0.0,
                forex_allowed_utc_hours=(23,),
            )
        )

        decision = strategy.generate_decision(
            [1.1000, 1.1001, 1.1002, 1.1004, 1.1008, 1.1014, 1.1022, 1.1032, 1.1044, 1.1058, 1.1074, 1.1092, 1.1112, 1.1134, 1.1158, 1.1184, 1.1212],
            quote=quote(),
        )

        self.assertEqual(decision.action, StrategyAction.NO_ACTION)
        self.assertIn("outside MACD momentum UTC hours", decision.reason)

    def test_histogram_exit_band_exits_current_position(self) -> None:
        strategy = MacdMomentumStrategy(
            MacdMomentumConfig(
                fast_window=3,
                slow_window=6,
                signal_window=3,
                min_histogram_bps=0.5,
                exit_histogram_bps=0.3,
                min_macd_bps=0.01,
                min_trend_efficiency=0.0,
                min_holding_period=0,
            )
        )

        decision = strategy.generate_decision(
            [1.1000, 1.1001, 1.1002, 1.1003, 1.1004, 1.1005, 1.1006, 1.1007, 1.1008, 1.1009, 1.1010, 1.1010, 1.1010, 1.1010, 1.1010, 1.1010, 1.1010],
            current_notional_usd=50_000,
            holding_period=3,
        )

        self.assertEqual(decision.action, StrategyAction.EXIT)

    def test_config_rejects_fast_window_at_or_above_slow_window(self) -> None:
        with self.assertRaisesRegex(ValueError, "slow_window"):
            MacdMomentumConfig(fast_window=10, slow_window=10)


class MeanReversionStrategyTest(TestCase):
    def test_price_spike_generates_sell_request(self) -> None:
        strategy = MeanReversionStrategy(MeanReversionConfig(entry_zscore=1.0))

        request = strategy.generate_request([1.1000, 1.1001, 1.1000, 1.1001, 1.1030])

        self.assertIsNotNone(request)
        self.assertEqual(request.side, Side.SELL)

    def test_price_drop_generates_buy_request(self) -> None:
        strategy = MeanReversionStrategy(MeanReversionConfig(entry_zscore=1.0))

        request = strategy.generate_request([1.1000, 1.0999, 1.1000, 1.0999, 1.0970])

        self.assertIsNotNone(request)
        self.assertEqual(request.side, Side.BUY)

    def test_flat_prices_generate_no_request(self) -> None:
        strategy = MeanReversionStrategy(MeanReversionConfig(entry_zscore=1.0))

        request = strategy.generate_request([1.1000, 1.1000, 1.1000, 1.1000, 1.1000])

        self.assertIsNone(request)

    def test_reversion_baseline_excludes_latest_price(self) -> None:
        strategy = MeanReversionStrategy(MeanReversionConfig())

        reading = strategy.read_reversion([1.0000, 1.0100, 0.9900, 1.0000, 1.2000])

        self.assertIsNotNone(reading)
        assert reading is not None
        self.assertAlmostEqual(reading.mean_price, 1.0000)
        self.assertEqual(reading.last_price, 1.2000)

    def test_reversion_trend_filter_blocks_strong_directional_baseline(self) -> None:
        strategy = MeanReversionStrategy(
            MeanReversionConfig(entry_zscore=1.0, max_trend_bps=5.0)
        )

        decision = strategy.generate_decision([1.0000, 1.0100, 1.0200, 1.0300, 1.0000])

        self.assertEqual(decision.action, StrategyAction.NO_ACTION)
        self.assertIn("trend filter", decision.reason)

    def test_reversion_exit_hysteresis_exits_near_baseline(self) -> None:
        strategy = MeanReversionStrategy(
            MeanReversionConfig(entry_zscore=1.0, exit_zscore=0.25)
        )

        decision = strategy.generate_decision(
            [1.0000, 1.0100, 0.9900, 1.0000, 1.0001],
            current_notional_usd=50_000,
        )

        self.assertEqual(decision.action, StrategyAction.EXIT)
        self.assertEqual(decision.target_notional_usd, 0.0)

    def test_reversion_max_holding_period_exits(self) -> None:
        strategy = MeanReversionStrategy(
            MeanReversionConfig(entry_zscore=1.0, max_holding_period=2)
        )

        decision = strategy.generate_decision(
            [1.0000, 1.0100, 0.9900, 1.0000, 0.9800],
            current_notional_usd=50_000,
            holding_period=2,
        )

        self.assertEqual(decision.action, StrategyAction.EXIT)
        self.assertIn("max holding period", decision.reason)

    def test_config_rejects_non_positive_entry_zscore(self) -> None:
        with self.assertRaisesRegex(ValueError, "entry_zscore"):
            MeanReversionConfig(entry_zscore=0.0)

    def test_config_rejects_invalid_hysteresis(self) -> None:
        with self.assertRaisesRegex(ValueError, "entry_zscore"):
            MeanReversionConfig(entry_zscore=1.0, exit_zscore=1.0)


class RegimeSwitchingStrategyTest(TestCase):
    def test_selects_momentum_regime_for_smooth_trend(self) -> None:
        strategy = RegimeSwitchingStrategy(
            config=RegimeConfig(lookback=5, hysteresis_bars=1)
        )

        decision = strategy.generate_decision([1.1000, 1.1003, 1.1006, 1.1009, 1.1012])

        self.assertEqual(strategy.active_regime, RegimeState.MOMENTUM)
        self.assertEqual(decision.action, StrategyAction.ENTER)
        self.assertIn("regime MOMENTUM", decision.reason)

    def test_selects_mean_reversion_for_stable_residual(self) -> None:
        strategy = RegimeSwitchingStrategy(
            config=RegimeConfig(
                lookback=5,
                hysteresis_bars=1,
                momentum_min_efficiency=0.95,
                mean_reversion_max_trend_bps=20.0,
                mean_reversion_max_efficiency=0.9,
            )
        )

        decision = strategy.generate_decision([1.1000, 1.1002, 1.0998, 1.1000, 1.1020])

        self.assertEqual(strategy.active_regime, RegimeState.MEAN_REVERSION)
        self.assertEqual(decision.action, StrategyAction.ENTER)
        self.assertIn("regime MEAN_REVERSION", decision.reason)

    def test_selects_flat_for_ambiguous_market(self) -> None:
        strategy = RegimeSwitchingStrategy(
            config=RegimeConfig(lookback=5, hysteresis_bars=1)
        )

        decision = strategy.generate_decision([1.1000, 1.1001, 1.1000, 1.1001, 1.1000])

        self.assertEqual(strategy.active_regime, RegimeState.FLAT)
        self.assertEqual(decision.action, StrategyAction.NO_ACTION)

    def test_wide_spread_forces_flat(self) -> None:
        strategy = RegimeSwitchingStrategy(
            config=RegimeConfig(lookback=5, hysteresis_bars=1, max_spread_bps=5.0)
        )

        decision = strategy.generate_decision(
            [1.1000, 1.1003, 1.1006, 1.1009, 1.1012],
            quote=quote(bid=1.0900, ask=1.1100),
        )

        self.assertEqual(strategy.active_regime, RegimeState.FLAT)
        self.assertEqual(decision.action, StrategyAction.NO_ACTION)
        self.assertIn("spread", decision.reason)

    def test_hysteresis_waits_before_switching_regime(self) -> None:
        strategy = RegimeSwitchingStrategy(
            config=RegimeConfig(lookback=5, hysteresis_bars=2)
        )

        first = strategy.generate_decision([1.1000, 1.1003, 1.1006, 1.1009, 1.1012])
        second = strategy.generate_decision([1.1001, 1.1004, 1.1007, 1.1010, 1.1013])

        self.assertEqual(first.action, StrategyAction.NO_ACTION)
        self.assertEqual(strategy.active_regime, RegimeState.MOMENTUM)
        self.assertEqual(second.action, StrategyAction.ENTER)

    def test_config_rejects_invalid_hysteresis(self) -> None:
        with self.assertRaisesRegex(ValueError, "hysteresis"):
            RegimeConfig(hysteresis_bars=0)


class FixedDecisionStrategy:
    def __init__(self, decision):
        self.decision = decision

    def generate_decision(self, prices, **kwargs):
        return self.decision


def _relative_strength_context() -> dict[str, list[float]]:
    return {
        "EURUSD": [1.0000, 1.0100, 1.0200, 1.0300],
        "GBPUSD": [1.0000, 1.0000, 1.0000, 1.0000],
        "AUDUSD": [1.0000, 1.0000, 1.0000, 1.0000],
        "USDJPY": [100.0, 99.0, 98.0, 97.0],
    }


def _global_only_relative_strength_context() -> dict[str, list[float]]:
    return {
        "EURUSD": [1.0000, 1.0100, 1.0200, 1.0300],
        "GBPUSD": [1.0000, 1.0100, 1.0200, 1.0300],
        "AUDUSD": [1.0000, 1.0100, 1.0200, 1.0300],
        "USDJPY": [100.0, 101.0, 102.0, 103.0],
        "XAGUSD": [20.0, 20.0, 20.0, 20.0],
        "XAUUSD": [2000.0, 1980.0, 1960.0, 1940.0],
    }


def _metal_relative_strength_context(xag_prices: list[float]) -> dict[str, list[float]]:
    return {
        "XAGUSD": xag_prices,
        "XAUUSD": [2000.0, 1980.0, 1960.0, 1940.0],
        "EURUSD": [1.0, 1.0, 1.0, 1.0],
        "GBPUSD": [1.0, 1.0, 1.0, 1.0],
        "USDJPY": [100.0, 100.0, 100.0, 100.0],
    }


def _cross_rate_context(*, eurgbp: list[float]) -> dict[str, list[float]]:
    return {
        "EURGBP": eurgbp,
        "EURUSD": [1.1000, 1.1000, 1.1000, 1.1000],
        "GBPUSD": [1.2500, 1.2500, 1.2500, 1.2500],
    }


class UsdPressureRouterStrategyTest(TestCase):
    def test_confirms_trade_when_usd_basket_agrees(self) -> None:
        base_decision = _fixed_decision(
            StrategyAction.ENTER,
            symbol="EURUSD",
            target_notional_usd=50_000,
            reason="base long EURUSD",
        )
        strategy = UsdPressureRouterStrategy(
            config=UsdPressureConfig(
                symbol="EURUSD",
                lookback=4,
                pressure_threshold_bps=2.0,
                component_threshold_bps=0.5,
                min_target_volatility_bps=0.0,
                min_component_symbols=2,
                min_confirming_symbols=2,
            ),
            base_strategy=FixedDecisionStrategy(base_decision),
        )
        strategy.update_portfolio_context(
            closes_by_symbol={
                "EURUSD": [1.1000, 1.1002, 1.1004, 1.1006],
                "GBPUSD": [1.3000, 1.3004, 1.3008, 1.3012],
                "AUDUSD": [0.6600, 0.6602, 0.6604, 0.6606],
                "USDJPY": [156.00, 155.95, 155.90, 155.85],
            }
        )

        decision = strategy.generate_decision([1.1000, 1.1002, 1.1004, 1.1006])

        self.assertEqual(decision.action, StrategyAction.ENTER)
        self.assertEqual(decision.primary_signal, "usd_pressure_router")
        self.assertIn("USD pressure confirmed", decision.reason)

    def test_blocks_trade_when_usd_basket_conflicts(self) -> None:
        base_decision = _fixed_decision(
            StrategyAction.ENTER,
            symbol="EURUSD",
            target_notional_usd=50_000,
            reason="base long EURUSD",
        )
        strategy = UsdPressureRouterStrategy(
            config=UsdPressureConfig(
                symbol="EURUSD",
                lookback=4,
                pressure_threshold_bps=2.0,
                component_threshold_bps=0.5,
                min_target_volatility_bps=0.0,
                min_component_symbols=2,
                min_confirming_symbols=2,
            ),
            base_strategy=FixedDecisionStrategy(base_decision),
        )
        strategy.update_portfolio_context(
            closes_by_symbol={
                "GBPUSD": [1.3000, 1.2996, 1.2992, 1.2988],
                "AUDUSD": [0.6600, 0.6598, 0.6595, 0.6592],
                "USDJPY": [156.00, 156.05, 156.10, 156.15],
            }
        )

        decision = strategy.generate_decision([1.1000, 1.1002, 1.1004, 1.1006])

        self.assertEqual(decision.action, StrategyAction.NO_ACTION)
        self.assertIn("USD pressure filter blocked", decision.reason)

    def test_exits_existing_trade_when_usd_pressure_conflicts(self) -> None:
        base_decision = _fixed_decision(
            StrategyAction.HOLD,
            symbol="EURUSD",
            target_notional_usd=50_000,
            reason="base holds",
        )
        strategy = UsdPressureRouterStrategy(
            config=UsdPressureConfig(
                symbol="EURUSD",
                lookback=4,
                pressure_threshold_bps=2.0,
                component_threshold_bps=0.5,
                min_target_volatility_bps=0.0,
                min_component_symbols=2,
                min_confirming_symbols=2,
                exit_on_conflict=True,
            ),
            base_strategy=FixedDecisionStrategy(base_decision),
        )
        strategy.update_portfolio_context(
            closes_by_symbol={
                "GBPUSD": [1.3000, 1.2996, 1.2992, 1.2988],
                "AUDUSD": [0.6600, 0.6598, 0.6595, 0.6592],
                "USDJPY": [156.00, 156.05, 156.10, 156.15],
            }
        )

        decision = strategy.generate_decision(
            [1.1000, 1.1002, 1.1004, 1.1006],
            current_notional_usd=50_000,
        )

        self.assertEqual(decision.action, StrategyAction.EXIT)
        self.assertIn("conflicts with held position", decision.reason)

    def test_target_volatility_floor_blocks_low_volatility_entry(self) -> None:
        base_decision = _fixed_decision(
            StrategyAction.ENTER,
            symbol="EURUSD",
            target_notional_usd=50_000,
            reason="base long EURUSD",
        )
        strategy = UsdPressureRouterStrategy(
            config=UsdPressureConfig(
                symbol="EURUSD",
                lookback=4,
                pressure_threshold_bps=2.0,
                component_threshold_bps=0.5,
                min_target_volatility_bps=3.0,
                min_component_symbols=2,
                min_confirming_symbols=2,
            ),
            base_strategy=FixedDecisionStrategy(base_decision),
        )
        strategy.update_portfolio_context(
            closes_by_symbol={
                "GBPUSD": [1.3000, 1.3004, 1.3008, 1.3012],
                "AUDUSD": [0.6600, 0.6602, 0.6604, 0.6606],
                "USDJPY": [156.00, 155.95, 155.90, 155.85],
            }
        )

        decision = strategy.generate_decision([1.1000, 1.10001, 1.10002, 1.10003])

        self.assertEqual(decision.action, StrategyAction.NO_ACTION)
        self.assertIn("target volatility filter blocked entry", decision.reason)
        self.assertIn(
            "target_realized_volatility_bps",
            {name for name, _ in decision.diagnostics},
        )

    def test_target_volatility_floor_does_not_block_base_exit(self) -> None:
        base_decision = _fixed_decision(
            StrategyAction.EXIT,
            symbol="EURUSD",
            target_notional_usd=0.0,
            reason="base exit",
        )
        strategy = UsdPressureRouterStrategy(
            config=UsdPressureConfig(
                symbol="EURUSD",
                lookback=4,
                min_target_volatility_bps=10.0,
            ),
            base_strategy=FixedDecisionStrategy(base_decision),
        )

        decision = strategy.generate_decision(
            [1.1000, 1.10001, 1.10002, 1.10003],
            current_notional_usd=50_000,
        )

        self.assertEqual(decision.action, StrategyAction.EXIT)
        self.assertEqual(decision.target_notional_usd, 0.0)
        self.assertEqual(decision.reason, "base exit")

    def test_config_rejects_impossible_confirmation_rule(self) -> None:
        with self.assertRaisesRegex(ValueError, "min_confirming_symbols"):
            UsdPressureConfig(min_component_symbols=2, min_confirming_symbols=3)

    def test_config_rejects_negative_target_volatility_floor(self) -> None:
        with self.assertRaisesRegex(ValueError, "min_target_volatility_bps"):
            UsdPressureConfig(min_target_volatility_bps=-0.1)


class RelativeStrengthStrategyTest(TestCase):
    def test_long_when_target_is_strongest_symbol(self) -> None:
        strategy = RelativeStrengthStrategy(
            RelativeStrengthConfig(
                symbol="EURUSD",
                lookback=4,
                entry_zscore=0.75,
                slippage_bps=0.0,
                max_spread_bps=10.0,
            )
        )
        strategy.update_portfolio_context(closes_by_symbol=_relative_strength_context())

        decision = strategy.generate_decision(
            [1.0000, 1.0100, 1.0200, 1.0300],
            quote=quote(),
        )

        self.assertEqual(decision.action, StrategyAction.ENTER)
        self.assertGreater(decision.target_notional_usd, 0.0)
        self.assertEqual(decision.primary_signal, "relative_strength")
        self.assertIn("rank 1/4", decision.reason)

    def test_asset_class_confirmation_allows_same_asset_leader(self) -> None:
        strategy = RelativeStrengthStrategy(
            RelativeStrengthConfig(
                symbol="EURUSD",
                lookback=4,
                entry_zscore=0.75,
                require_asset_class_confirmation=True,
                asset_class_entry_zscore=0.35,
                slippage_bps=0.0,
                max_spread_bps=10.0,
            )
        )
        strategy.update_portfolio_context(closes_by_symbol=_relative_strength_context())

        decision = strategy.generate_decision(
            [1.0000, 1.0100, 1.0200, 1.0300],
            quote=quote(),
        )

        self.assertEqual(decision.action, StrategyAction.ENTER)
        self.assertIn("same-asset z-score", decision.reason)
        self.assertIn("relative_strength_asset_class_zscore", dict(decision.diagnostics))

    def test_asset_class_confirmation_blocks_global_only_move(self) -> None:
        strategy = RelativeStrengthStrategy(
            RelativeStrengthConfig(
                symbol="EURUSD",
                lookback=4,
                entry_zscore=0.50,
                require_asset_class_confirmation=True,
                asset_class_entry_zscore=0.35,
            )
        )
        context = _global_only_relative_strength_context()
        strategy.update_portfolio_context(closes_by_symbol=context)

        decision = strategy.generate_decision(context["EURUSD"])

        self.assertEqual(decision.action, StrategyAction.NO_ACTION)
        self.assertIn("asset-class confirmation blocked entry", decision.reason)

    def test_metal_trend_confirmation_allows_smooth_metal_leader(self) -> None:
        strategy = RelativeStrengthStrategy(
            RelativeStrengthConfig(
                symbol="XAGUSD",
                lookback=4,
                entry_zscore=0.10,
                exit_zscore=0.05,
                require_metal_trend_confirmation=True,
                metal_trend_min_move_bps=2.0,
                metal_trend_min_efficiency=0.20,
                slippage_bps=0.0,
                max_spread_bps=40.0,
            )
        )
        context = _metal_relative_strength_context([20.0, 20.2, 20.4, 20.6])
        strategy.update_portfolio_context(closes_by_symbol=context)

        decision = strategy.generate_decision(context["XAGUSD"], quote=quote())

        self.assertEqual(decision.action, StrategyAction.ENTER)
        self.assertIn("metal trend move", decision.reason)
        self.assertIn("relative_strength_trend_efficiency", dict(decision.diagnostics))

    def test_metal_trend_confirmation_blocks_noisy_metal_move(self) -> None:
        strategy = RelativeStrengthStrategy(
            RelativeStrengthConfig(
                symbol="XAGUSD",
                lookback=4,
                entry_zscore=0.10,
                exit_zscore=0.05,
                require_metal_trend_confirmation=True,
                metal_trend_min_move_bps=2.0,
                metal_trend_min_efficiency=0.20,
                slippage_bps=0.0,
                max_spread_bps=40.0,
            )
        )
        context = _metal_relative_strength_context([20.0, 22.0, 18.0, 21.0])
        strategy.update_portfolio_context(closes_by_symbol=context)

        decision = strategy.generate_decision(context["XAGUSD"], quote=quote())

        self.assertEqual(decision.action, StrategyAction.NO_ACTION)
        self.assertIn("metal trend confirmation blocked entry", decision.reason)

    def test_regime_gate_blocks_low_score_dispersion(self) -> None:
        strategy = RelativeStrengthStrategy(
            RelativeStrengthConfig(
                symbol="EURUSD",
                lookback=4,
                entry_zscore=0.50,
                min_score_dispersion=10_000.0,
            )
        )
        strategy.update_portfolio_context(closes_by_symbol=_relative_strength_context())

        decision = strategy.generate_decision([1.0000, 1.0100, 1.0200, 1.0300])

        self.assertEqual(decision.action, StrategyAction.NO_ACTION)
        self.assertIn("regime gate blocked entry", decision.reason)
        self.assertIn("score dispersion", decision.reason)
        self.assertIn("relative_strength_score_dispersion", dict(decision.diagnostics))

    def test_regime_gate_blocks_noisy_target_trend(self) -> None:
        strategy = RelativeStrengthStrategy(
            RelativeStrengthConfig(
                symbol="EURUSD",
                lookback=4,
                entry_zscore=0.10,
                exit_zscore=0.05,
                min_target_trend_efficiency=0.50,
            )
        )
        context = {
            "EURUSD": [1.0, 1.2, 0.8, 1.1],
            "GBPUSD": [1.0, 1.0, 1.0, 1.0],
            "AUDUSD": [1.0, 1.0, 1.0, 1.0],
            "USDJPY": [100.0, 99.0, 98.0, 97.0],
        }
        strategy.update_portfolio_context(closes_by_symbol=context)

        decision = strategy.generate_decision(context["EURUSD"])

        self.assertEqual(decision.action, StrategyAction.NO_ACTION)
        self.assertIn("target trend efficiency", decision.reason)

    def test_short_when_target_is_weakest_symbol(self) -> None:
        strategy = RelativeStrengthStrategy(
            RelativeStrengthConfig(
                symbol="USDJPY",
                lookback=4,
                entry_zscore=0.75,
                slippage_bps=0.0,
                max_spread_bps=10.0,
            )
        )
        context = _relative_strength_context()
        strategy.update_portfolio_context(closes_by_symbol=context)

        decision = strategy.generate_decision(context["USDJPY"], quote=quote())

        self.assertEqual(decision.action, StrategyAction.ENTER)
        self.assertLess(decision.target_notional_usd, 0.0)
        self.assertIn("rank 4/4", decision.reason)

    def test_middle_rank_generates_no_action(self) -> None:
        strategy = RelativeStrengthStrategy(
            RelativeStrengthConfig(symbol="GBPUSD", lookback=4, entry_zscore=0.75)
        )
        context = _relative_strength_context()
        strategy.update_portfolio_context(closes_by_symbol=context)

        decision = strategy.generate_decision(context["GBPUSD"])

        self.assertEqual(decision.action, StrategyAction.NO_ACTION)
        self.assertEqual(decision.target_notional_usd, 0.0)
        self.assertIn("inside neutral band", decision.reason)

    def test_existing_position_exits_when_relative_edge_fades(self) -> None:
        strategy = RelativeStrengthStrategy(
            RelativeStrengthConfig(symbol="EURUSD", lookback=4, entry_zscore=0.75)
        )
        flat_context = {
            "EURUSD": [1.0, 1.0, 1.0, 1.0],
            "GBPUSD": [1.0, 1.0, 1.0, 1.0],
            "AUDUSD": [1.0, 1.0, 1.0, 1.0],
            "USDJPY": [100.0, 100.0, 100.0, 100.0],
        }
        strategy.update_portfolio_context(closes_by_symbol=flat_context)

        decision = strategy.generate_decision(
            flat_context["EURUSD"],
            current_notional_usd=50_000,
        )

        self.assertEqual(decision.action, StrategyAction.EXIT)
        self.assertEqual(decision.target_notional_usd, 0.0)
        self.assertIn("faded below exit threshold", decision.reason)

    def test_cost_filter_blocks_wide_spread_entry(self) -> None:
        strategy = RelativeStrengthStrategy(
            RelativeStrengthConfig(
                symbol="EURUSD",
                lookback=4,
                entry_zscore=0.75,
                max_spread_bps=5.0,
            )
        )
        strategy.update_portfolio_context(closes_by_symbol=_relative_strength_context())

        decision = strategy.generate_decision(
            [1.0000, 1.0100, 1.0200, 1.0300],
            quote=quote(bid=1.0000, ask=1.0100),
        )

        self.assertEqual(decision.action, StrategyAction.NO_ACTION)
        self.assertIn("spread", decision.reason)

    def test_config_rejects_invalid_entry_exit_order(self) -> None:
        with self.assertRaisesRegex(ValueError, "entry_zscore"):
            RelativeStrengthConfig(entry_zscore=0.25, exit_zscore=0.25)

    def test_config_rejects_invalid_asset_class_min_symbols(self) -> None:
        with self.assertRaisesRegex(ValueError, "asset_class_min_symbols"):
            RelativeStrengthConfig(asset_class_min_symbols=1)

    def test_config_rejects_invalid_metal_trend_efficiency(self) -> None:
        with self.assertRaisesRegex(ValueError, "metal_trend_min_efficiency"):
            RelativeStrengthConfig(metal_trend_min_efficiency=1.1)

    def test_config_rejects_invalid_target_trend_efficiency(self) -> None:
        with self.assertRaisesRegex(ValueError, "min_target_trend_efficiency"):
            RelativeStrengthConfig(min_target_trend_efficiency=1.1)


class CrossRateReversionStrategyTest(TestCase):
    def test_short_when_target_is_rich_versus_synthetic_cross(self) -> None:
        strategy = CrossRateReversionStrategy(
            CrossRateReversionConfig(
                symbol="EURGBP",
                lookback=4,
                entry_zscore=1.0,
                min_abs_deviation_bps=1.0,
                max_abs_deviation_bps=400.0,
                position_sizing="fixed",
                slippage_bps=0.0,
                max_spread_bps=100.0,
            )
        )
        context = _cross_rate_context(
            eurgbp=[0.8798, 0.8801, 0.8799, 0.9000],
        )
        strategy.update_portfolio_context(closes_by_symbol=context)

        decision = strategy.generate_decision(context["EURGBP"], quote=quote())

        self.assertEqual(decision.action, StrategyAction.ENTER)
        self.assertLess(decision.target_notional_usd, 0.0)
        self.assertEqual(decision.primary_signal, "cross_rate_reversion")
        self.assertIn("rich vs synthetic", decision.reason)
        self.assertIn("EURUSD/GBPUSD", decision.reason)

    def test_long_when_target_is_cheap_versus_synthetic_cross(self) -> None:
        strategy = CrossRateReversionStrategy(
            CrossRateReversionConfig(
                symbol="EURGBP",
                lookback=4,
                entry_zscore=1.0,
                min_abs_deviation_bps=1.0,
                max_abs_deviation_bps=400.0,
                position_sizing="fixed",
                slippage_bps=0.0,
                max_spread_bps=100.0,
            )
        )
        context = _cross_rate_context(
            eurgbp=[0.8798, 0.8801, 0.8799, 0.8600],
        )
        strategy.update_portfolio_context(closes_by_symbol=context)

        decision = strategy.generate_decision(context["EURGBP"], quote=quote())

        self.assertEqual(decision.action, StrategyAction.ENTER)
        self.assertGreater(decision.target_notional_usd, 0.0)
        self.assertIn("cheap vs synthetic", decision.reason)

    def test_signal_api_exposes_cross_rate_reversion_signal(self) -> None:
        strategy = CrossRateReversionStrategy(
            CrossRateReversionConfig(
                symbol="EURGBP",
                lookback=4,
                entry_zscore=1.0,
                min_abs_deviation_bps=1.0,
                max_abs_deviation_bps=400.0,
                slippage_bps=0.0,
                max_spread_bps=100.0,
            )
        )
        context = _cross_rate_context(
            eurgbp=[0.8798, 0.8801, 0.8799, 0.9000],
        )
        strategy.update_portfolio_context(closes_by_symbol=context)

        signals = strategy.generate_signals(context["EURGBP"], quote=quote())

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].strategy_name, "cross_rate_reversion")
        self.assertEqual(signals[0].direction, SignalDirection.SHORT)
        self.assertGreater(signals[0].expected_edge_bps, 0.0)

    def test_no_action_without_synthetic_path(self) -> None:
        strategy = CrossRateReversionStrategy(
            CrossRateReversionConfig(symbol="USDCAD", lookback=4)
        )
        context = {
            "USDCAD": [1.3500, 1.3501, 1.3502, 1.3503],
            "EURUSD": [1.1000, 1.1000, 1.1000, 1.1000],
        }
        strategy.update_portfolio_context(closes_by_symbol=context)

        decision = strategy.generate_decision(context["USDCAD"])

        self.assertEqual(decision.action, StrategyAction.NO_ACTION)
        self.assertIn("not enough FX cross-rate context", decision.reason)

    def test_allowlist_blocks_unapproved_cross_symbol(self) -> None:
        strategy = CrossRateReversionStrategy(
            CrossRateReversionConfig(
                symbol="GBPUSD",
                allowed_symbols=("EURGBP", "EURUSD"),
                lookback=4,
            )
        )
        context = _cross_rate_context(
            eurgbp=[0.8798, 0.8801, 0.8799, 0.9000],
        )
        strategy.update_portfolio_context(closes_by_symbol=context)

        decision = strategy.generate_decision(context["GBPUSD"])

        self.assertEqual(decision.action, StrategyAction.NO_ACTION)
        self.assertIn("outside cross-rate allowlist", decision.reason)

    def test_non_fx_symbol_fails_closed_without_crashing(self) -> None:
        strategy = CrossRateReversionStrategy(
            CrossRateReversionConfig(symbol="BTCUSD", lookback=4)
        )

        decision = strategy.generate_decision([65_000, 65_100, 65_200, 65_300])

        self.assertEqual(decision.action, StrategyAction.NO_ACTION)
        self.assertIn("only applies to FX", decision.reason)

    def test_config_rejects_invalid_entry_exit_order(self) -> None:
        with self.assertRaisesRegex(ValueError, "entry_zscore"):
            CrossRateReversionConfig(entry_zscore=0.25, exit_zscore=0.25)

    def test_config_rejects_non_fx_allowlist_symbol(self) -> None:
        with self.assertRaisesRegex(ValueError, "allowed_symbols"):
            CrossRateReversionConfig(allowed_symbols=("BTCUSD",))


class AlphaRouterStrategyTest(TestCase):
    def test_router_enters_when_momentum_signal_is_strong(self) -> None:
        strategy = AlphaRouterStrategy(
            config=AlphaRouterConfig(entry_score=0.30),
            momentum=MomentumConfig(threshold_bps=5.0),
            mean_reversion=MeanReversionConfig(entry_zscore=3.0),
        )

        decision = strategy.generate_decision([1.1000, 1.1003, 1.1006, 1.1009, 1.1012])

        self.assertEqual(decision.action, StrategyAction.ENTER)
        self.assertGreater(decision.target_notional_usd, 0)
        self.assertEqual(decision.primary_signal, "momentum")
        self.assertIn("momentum", decision.reason)

    def test_router_outputs_one_decision_from_multiple_signals(self) -> None:
        strategy = AlphaRouterStrategy(
            config=AlphaRouterConfig(entry_score=0.30),
            momentum=MomentumConfig(threshold_bps=5.0),
            moving_average=MovingAverageCrossoverConfig(fast_window=2, slow_window=5),
            breakout=BreakoutConfig(lookback=5, breakout_buffer_bps=1.0),
            mean_reversion=MeanReversionConfig(entry_zscore=1.0),
        )

        signals = strategy.generate_signals([1.1000, 1.1003, 1.1006, 1.1009, 1.1012])
        decision = strategy.generate_decision([1.1000, 1.1003, 1.1006, 1.1009, 1.1012])

        self.assertEqual(len(signals), 5)
        self.assertEqual(signals[0].direction, SignalDirection.LONG)
        self.assertEqual(signals[1].strategy_name, "ma_crossover")
        self.assertEqual(signals[2].strategy_name, "breakout")
        self.assertEqual(signals[3].strategy_name, "session_breakout")
        self.assertIn(decision.action, {StrategyAction.ENTER, StrategyAction.NO_ACTION})

    def test_router_can_use_session_breakout_quality_signal(self) -> None:
        strategy = AlphaRouterStrategy(
            config=AlphaRouterConfig(
                entry_score=0.10,
                exit_score=0.02,
                momentum_weight=0.0,
                moving_average_weight=0.0,
                breakout_weight=0.0,
                session_breakout_weight=1.0,
                mean_reversion_weight=0.0,
            ),
            momentum=MomentumConfig(threshold_bps=999.0),
            moving_average=MovingAverageCrossoverConfig(
                fast_window=2,
                slow_window=5,
                min_separation_bps=999.0,
            ),
            breakout=BreakoutConfig(breakout_buffer_bps=999.0),
            session_breakout=SessionBreakoutConfig(
                lookback=8,
                breakout_buffer_bps=2.0,
                min_expected_edge_bps=3.0,
                min_realized_volatility_bps=0.1,
                allowed_utc_hours=(12,),
                max_spread_bps=5.0,
            ),
            mean_reversion=MeanReversionConfig(entry_zscore=10.0, stop_zscore=20.0),
        )

        decision = strategy.generate_decision(
            [1.1000, 1.1001, 1.1000, 1.1002, 1.1001, 1.1004, 1.1003, 1.1012],
            quote=QuoteSnapshot(
                timestamp=datetime(2026, 6, 22, 12, 0, tzinfo=ZoneInfo("UTC")),
                symbol="EURUSD",
                bid=1.10115,
                ask=1.10125,
            ),
        )

        self.assertEqual(decision.action, StrategyAction.ENTER)
        self.assertEqual(decision.primary_signal, "session_breakout")
        self.assertIn("session_breakout=LONG", decision.reason)

    def test_router_can_use_volatility_squeeze_signal(self) -> None:
        strategy = AlphaRouterStrategy(
            config=AlphaRouterConfig(
                entry_score=0.10,
                exit_score=0.02,
                min_signal_confidence=0.0,
                momentum_weight=0.0,
                moving_average_weight=0.0,
                breakout_weight=0.0,
                session_breakout_weight=0.0,
                volatility_squeeze_weight=1.0,
                mean_reversion_weight=0.0,
            ),
            momentum=MomentumConfig(threshold_bps=999.0),
            moving_average=MovingAverageCrossoverConfig(
                fast_window=2,
                slow_window=5,
                min_separation_bps=999.0,
            ),
            breakout=BreakoutConfig(breakout_buffer_bps=999.0),
            volatility_squeeze=VolatilitySqueezeConfig(
                lookback=10,
                squeeze_window=3,
                band_stdev_multiplier=1.0,
                breakout_buffer_bps=1.0,
                max_squeeze_ratio=0.5,
                min_prior_volatility_bps=0.1,
                min_band_width_bps=0.0,
                position_sizing="fixed",
                forex_allowed_utc_hours=(12,),
            ),
            session_breakout=SessionBreakoutConfig(breakout_buffer_bps=999.0),
            mean_reversion=MeanReversionConfig(entry_zscore=10.0, stop_zscore=20.0),
        )

        prices = [
            1.0000,
            1.0010,
            0.9990,
            1.0010,
            0.9990,
            1.0000,
            1.00005,
            0.99995,
            1.0000,
            1.0025,
        ]
        decision = strategy.generate_decision(
            prices,
            quote=QuoteSnapshot(
                timestamp=datetime(2026, 6, 22, 12, 0, tzinfo=ZoneInfo("UTC")),
                symbol="EURUSD",
                bid=1.00245,
                ask=1.00255,
            ),
        )

        self.assertEqual(decision.action, StrategyAction.ENTER)
        self.assertEqual(decision.primary_signal, "volatility_squeeze")
        self.assertIn("volatility_squeeze=LONG", decision.reason)

    def test_router_can_use_moving_average_crossover_signal(self) -> None:
        strategy = AlphaRouterStrategy(
            config=AlphaRouterConfig(
                entry_score=0.15,
                exit_score=0.05,
                momentum_weight=0.0,
                moving_average_weight=1.0,
                breakout_weight=0.0,
                session_breakout_weight=0.0,
                mean_reversion_weight=0.0,
            ),
            momentum=MomentumConfig(threshold_bps=999.0),
            moving_average=MovingAverageCrossoverConfig(
                fast_window=2,
                slow_window=5,
                min_separation_bps=1.0,
            ),
            breakout=BreakoutConfig(breakout_buffer_bps=999.0),
            mean_reversion=MeanReversionConfig(entry_zscore=10.0, stop_zscore=20.0),
        )

        decision = strategy.generate_decision([1.1000, 1.1001, 1.1002, 1.1005, 1.1010])

        self.assertEqual(decision.action, StrategyAction.ENTER)
        self.assertGreater(decision.target_notional_usd, 0.0)
        self.assertEqual(decision.primary_signal, "ma_crossover")
        self.assertIn("ma_crossover=LONG", decision.reason)

    def test_router_can_use_macd_momentum_signal(self) -> None:
        strategy = AlphaRouterStrategy(
            config=AlphaRouterConfig(
                entry_score=0.05,
                exit_score=0.01,
                min_signal_confidence=0.0,
                momentum_weight=0.0,
                moving_average_weight=0.0,
                breakout_weight=0.0,
                session_breakout_weight=0.0,
                macd_momentum_weight=1.0,
                mean_reversion_weight=0.0,
            ),
            momentum=MomentumConfig(threshold_bps=999.0),
            moving_average=MovingAverageCrossoverConfig(min_separation_bps=999.0),
            breakout=BreakoutConfig(breakout_buffer_bps=999.0),
            session_breakout=SessionBreakoutConfig(breakout_buffer_bps=999.0),
            macd_momentum=MacdMomentumConfig(
                fast_window=2,
                slow_window=5,
                signal_window=3,
                min_histogram_bps=0.01,
                exit_histogram_bps=0.001,
                min_macd_bps=0.0,
                min_trend_efficiency=0.0,
                forex_allowed_utc_hours=(12,),
                position_sizing="fixed",
            ),
            mean_reversion=MeanReversionConfig(entry_zscore=10.0, stop_zscore=20.0),
        )

        decision = strategy.generate_decision(
            [
                1.0000,
                1.0001,
                1.0002,
                1.0004,
                1.0008,
                1.0015,
                1.0025,
                1.0040,
                1.0060,
                1.0090,
            ],
            quote=QuoteSnapshot(
                timestamp=datetime(2026, 6, 22, 12, 0, tzinfo=ZoneInfo("UTC")),
                symbol="EURUSD",
                bid=1.00895,
                ask=1.00905,
            ),
        )

        self.assertEqual(decision.action, StrategyAction.ENTER)
        self.assertGreater(decision.target_notional_usd, 0.0)
        self.assertEqual(decision.primary_signal, "macd_momentum")
        self.assertIn("macd_momentum=LONG", decision.reason)

    def test_router_can_use_kalman_trend_signal(self) -> None:
        strategy = AlphaRouterStrategy(
            config=AlphaRouterConfig(
                entry_score=0.05,
                exit_score=0.01,
                min_signal_confidence=0.0,
                momentum_weight=0.0,
                moving_average_weight=0.0,
                breakout_weight=0.0,
                session_breakout_weight=0.0,
                kalman_trend_weight=1.0,
                mean_reversion_weight=0.0,
            ),
            momentum=MomentumConfig(threshold_bps=999.0),
            moving_average=MovingAverageCrossoverConfig(min_separation_bps=999.0),
            breakout=BreakoutConfig(breakout_buffer_bps=999.0),
            session_breakout=SessionBreakoutConfig(breakout_buffer_bps=999.0),
            kalman_trend=KalmanTrendStrategyConfig(
                lookback=6,
                min_abs_slope_bps=0.01,
                min_trend_efficiency=0.0,
                min_expected_edge_bps=0.01,
                expected_holding_bars=4,
                forex_allowed_utc_hours=(12,),
                position_sizing="fixed",
            ),
            mean_reversion=MeanReversionConfig(entry_zscore=10.0, stop_zscore=20.0),
        )

        decision = strategy.generate_decision(
            [1.0000, 1.0004, 1.0009, 1.0015, 1.0022, 1.0030],
            quote=QuoteSnapshot(
                timestamp=datetime(2026, 6, 22, 12, 0, tzinfo=ZoneInfo("UTC")),
                symbol="EURUSD",
                bid=1.00295,
                ask=1.00305,
            ),
        )

        self.assertEqual(decision.action, StrategyAction.ENTER)
        self.assertGreater(decision.target_notional_usd, 0.0)
        self.assertEqual(decision.primary_signal, "kalman_trend")
        self.assertIn("kalman_trend=LONG", decision.reason)

    def test_router_can_use_cross_rate_reversion_signal(self) -> None:
        strategy = AlphaRouterStrategy(
            config=AlphaRouterConfig(
                symbol="EURGBP",
                entry_score=0.10,
                exit_score=0.02,
                min_signal_confidence=0.0,
                momentum_weight=0.0,
                moving_average_weight=0.0,
                breakout_weight=0.0,
                session_breakout_weight=0.0,
                mean_reversion_weight=0.0,
                cross_rate_weight=1.0,
            ),
            momentum=MomentumConfig(symbol="EURGBP", threshold_bps=999.0),
            moving_average=MovingAverageCrossoverConfig(
                symbol="EURGBP",
                fast_window=2,
                slow_window=5,
                min_separation_bps=999.0,
            ),
            breakout=BreakoutConfig(symbol="EURGBP", breakout_buffer_bps=999.0),
            session_breakout=SessionBreakoutConfig(
                symbol="EURGBP",
                breakout_buffer_bps=999.0,
            ),
            mean_reversion=MeanReversionConfig(
                symbol="EURGBP",
                entry_zscore=10.0,
                stop_zscore=20.0,
            ),
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
        context = _cross_rate_context(
            eurgbp=[0.8798, 0.8801, 0.8799, 0.9000],
        )
        strategy.update_portfolio_context(closes_by_symbol=context)

        signals = strategy.generate_signals(context["EURGBP"], quote=quote())
        decision = strategy.generate_decision(context["EURGBP"], quote=quote())

        self.assertEqual(len(signals), 4)
        self.assertEqual(signals[-1].strategy_name, "cross_rate_reversion")
        self.assertEqual(signals[-1].direction, SignalDirection.SHORT)
        self.assertEqual(signals[-1].weight, 1.0)
        self.assertEqual(decision.action, StrategyAction.ENTER)
        self.assertLess(decision.target_notional_usd, 0.0)
        self.assertEqual(decision.primary_signal, "cross_rate_reversion")

    def test_router_can_use_relative_strength_signal(self) -> None:
        strategy = AlphaRouterStrategy(
            config=AlphaRouterConfig(
                symbol="EURUSD",
                entry_score=0.10,
                exit_score=0.02,
                min_signal_confidence=0.0,
                momentum_weight=0.0,
                moving_average_weight=0.0,
                breakout_weight=0.0,
                session_breakout_weight=0.0,
                mean_reversion_weight=0.0,
                relative_strength_weight=1.0,
                cross_rate_weight=0.0,
            ),
            momentum=MomentumConfig(symbol="EURUSD", threshold_bps=999.0),
            moving_average=MovingAverageCrossoverConfig(
                symbol="EURUSD",
                fast_window=2,
                slow_window=5,
                min_separation_bps=999.0,
            ),
            breakout=BreakoutConfig(symbol="EURUSD", breakout_buffer_bps=999.0),
            session_breakout=SessionBreakoutConfig(
                symbol="EURUSD",
                breakout_buffer_bps=999.0,
            ),
            mean_reversion=MeanReversionConfig(
                symbol="EURUSD",
                entry_zscore=10.0,
                stop_zscore=20.0,
            ),
            relative_strength=RelativeStrengthConfig(
                symbol="EURUSD",
                lookback=4,
                entry_zscore=0.50,
                min_component_symbols=4,
                min_abs_move_bps=0.0,
                slippage_bps=0.0,
                max_spread_bps=100.0,
            ),
        )
        context = _relative_strength_context()
        strategy.update_portfolio_context(closes_by_symbol=context)

        signals = strategy.generate_signals(context["EURUSD"], quote=quote())
        decision = strategy.generate_decision(context["EURUSD"], quote=quote())

        relative_signal = signals[-1]
        self.assertEqual(relative_signal.strategy_name, "relative_strength")
        self.assertEqual(relative_signal.direction, SignalDirection.LONG)
        self.assertEqual(relative_signal.weight, 1.0)
        self.assertEqual(decision.action, StrategyAction.ENTER)
        self.assertGreater(decision.target_notional_usd, 0.0)
        self.assertEqual(decision.primary_signal, "relative_strength")
        self.assertIn("relative_strength=LONG", decision.reason)

    def test_router_relative_strength_gate_blocks_low_dispersion(self) -> None:
        strategy = AlphaRouterStrategy(
            config=AlphaRouterConfig(
                symbol="EURUSD",
                momentum_weight=0.0,
                moving_average_weight=0.0,
                breakout_weight=0.0,
                session_breakout_weight=0.0,
                mean_reversion_weight=0.0,
                relative_strength_weight=1.0,
                relative_strength_min_score_dispersion=1_000.0,
            ),
            relative_strength=RelativeStrengthConfig(
                symbol="EURUSD",
                lookback=4,
                entry_zscore=0.50,
                min_component_symbols=4,
                min_abs_move_bps=0.0,
            ),
        )
        context = _relative_strength_context()
        strategy.update_portfolio_context(closes_by_symbol=context)

        signal = strategy.generate_signals(context["EURUSD"])[-1]

        self.assertEqual(signal.strategy_name, "relative_strength")
        self.assertEqual(signal.direction, SignalDirection.FLAT)
        self.assertIn("low dispersion", signal.reason)

    def test_router_adaptive_weighting_tilts_metals_toward_mean_reversion(self) -> None:
        strategy = AlphaRouterStrategy(
            config=AlphaRouterConfig(
                symbol="XAUUSD",
                momentum_weight=0.30,
                moving_average_weight=0.0,
                breakout_weight=0.15,
                session_breakout_weight=0.25,
                mean_reversion_weight=0.35,
                adaptive_weighting_enabled=True,
                metal_mean_reversion_multiplier=1.25,
                metal_raw_breakout_multiplier=0.60,
            ),
            momentum=MomentumConfig(symbol="XAUUSD", threshold_bps=999.0),
            breakout=BreakoutConfig(symbol="XAUUSD", lookback=5, breakout_buffer_bps=1.0),
            session_breakout=SessionBreakoutConfig(
                symbol="XAUUSD",
                lookback=5,
                breakout_buffer_bps=1.0,
                allowed_utc_hours=(10,),
            ),
            mean_reversion=MeanReversionConfig(symbol="XAUUSD", entry_zscore=1.0),
        )

        signals = {
            signal.strategy_name: signal
            for signal in strategy.generate_signals([2320.0, 2321.0, 2319.0, 2320.0, 2325.0])
        }

        self.assertAlmostEqual(signals["mean_reversion"].weight, 0.35 * 1.25)
        self.assertAlmostEqual(signals["breakout"].weight, 0.15 * 0.60)
        self.assertIn(
            ("adaptive_router_asset_class", "METAL"),
            signals["mean_reversion"].diagnostics,
        )

    def test_router_adaptive_weighting_switches_on_high_volatility(self) -> None:
        strategy = AlphaRouterStrategy(
            config=AlphaRouterConfig(
                momentum_weight=0.30,
                moving_average_weight=0.0,
                breakout_weight=0.20,
                session_breakout_weight=0.0,
                mean_reversion_weight=0.40,
                adaptive_weighting_enabled=True,
                adaptive_regime_lookback=12,
                chop_mean_reversion_multiplier=1.0,
                chop_trend_signal_multiplier=1.0,
                trend_aligned_signal_multiplier=1.0,
                trend_counter_signal_multiplier=1.0,
                volatility_regime_lookback=4,
                high_volatility_ratio=1.01,
                high_volatility_reversion_multiplier=2.0,
                high_volatility_trend_multiplier=0.50,
            ),
            momentum=MomentumConfig(threshold_bps=999.0),
            breakout=BreakoutConfig(lookback=5, breakout_buffer_bps=1.0),
            mean_reversion=MeanReversionConfig(entry_zscore=1.0),
        )

        prices = [
            1.00000,
            1.00005,
            1.00000,
            1.00005,
            1.00000,
            1.00005,
            1.00000,
            1.00005,
            1.00000,
            1.00005,
            1.00000,
            1.02000,
            0.98000,
            1.03000,
            0.97000,
            1.04000,
        ]
        signals = {signal.strategy_name: signal for signal in strategy.generate_signals(prices)}

        self.assertAlmostEqual(signals["mean_reversion"].weight, 0.40 * 2.0)
        self.assertAlmostEqual(signals["breakout"].weight, 0.20 * 0.50)
        self.assertIn(
            ("adaptive_router_volatility_regime", "HIGH_VOL"),
            signals["mean_reversion"].diagnostics,
        )

    def test_router_adaptive_weighting_can_be_disabled(self) -> None:
        strategy = AlphaRouterStrategy(
            config=AlphaRouterConfig(
                symbol="XAUUSD",
                moving_average_weight=0.0,
                breakout_weight=0.15,
                session_breakout_weight=0.25,
                mean_reversion_weight=0.35,
                adaptive_weighting_enabled=False,
                metal_mean_reversion_multiplier=1.25,
                metal_raw_breakout_multiplier=0.60,
            ),
            momentum=MomentumConfig(symbol="XAUUSD", threshold_bps=999.0),
            breakout=BreakoutConfig(symbol="XAUUSD", lookback=5, breakout_buffer_bps=1.0),
            mean_reversion=MeanReversionConfig(symbol="XAUUSD", entry_zscore=1.0),
        )

        signals = {
            signal.strategy_name: signal
            for signal in strategy.generate_signals([2320.0, 2321.0, 2319.0, 2320.0, 2325.0])
        }

        self.assertAlmostEqual(signals["mean_reversion"].weight, 0.35)
        self.assertAlmostEqual(signals["breakout"].weight, 0.15)

    def test_router_primary_override_can_follow_strong_breakout(self) -> None:
        strategy = AlphaRouterStrategy(
            config=AlphaRouterConfig(entry_score=0.35),
            momentum=MomentumConfig(threshold_bps=999.0),
            moving_average=MovingAverageCrossoverConfig(
                fast_window=2,
                slow_window=5,
                min_separation_bps=999.0,
            ),
            breakout=BreakoutConfig(lookback=5, breakout_buffer_bps=2.0),
            mean_reversion=MeanReversionConfig(entry_zscore=10.0, stop_zscore=20.0),
        )

        decision = strategy.generate_decision([1.1000, 1.1001, 1.1002, 1.1003, 1.1012])

        self.assertEqual(decision.action, StrategyAction.ENTER)
        self.assertGreater(decision.target_notional_usd, 0.0)
        self.assertEqual(decision.primary_signal, "breakout")
        self.assertIn("primary signal override", decision.reason)

    def test_router_primary_override_can_be_disabled(self) -> None:
        strategy = AlphaRouterStrategy(
            config=AlphaRouterConfig(
                entry_score=0.35,
                primary_signal_override_enabled=False,
                session_breakout_weight=0.0,
            ),
            momentum=MomentumConfig(threshold_bps=999.0),
            moving_average=MovingAverageCrossoverConfig(
                fast_window=2,
                slow_window=5,
                min_separation_bps=999.0,
            ),
            breakout=BreakoutConfig(lookback=5, breakout_buffer_bps=2.0),
            mean_reversion=MeanReversionConfig(entry_zscore=10.0, stop_zscore=20.0),
        )

        decision = strategy.generate_decision([1.1000, 1.1001, 1.1002, 1.1003, 1.1012])

        self.assertEqual(decision.action, StrategyAction.NO_ACTION)
        self.assertIn("score below entry threshold", decision.reason)

    def test_router_holds_existing_position_instead_of_duplicate_entry(self) -> None:
        strategy = AlphaRouterStrategy(
            config=AlphaRouterConfig(entry_score=0.30),
            momentum=MomentumConfig(threshold_bps=5.0),
            mean_reversion=MeanReversionConfig(entry_zscore=3.0),
        )

        decision = strategy.generate_decision(
            [1.1000, 1.1003, 1.1006, 1.1009, 1.1012],
            current_notional_usd=50_000,
        )

        self.assertEqual(decision.action, StrategyAction.HOLD)

    def test_router_exits_when_score_fades(self) -> None:
        strategy = AlphaRouterStrategy()

        decision = strategy.generate_decision(
            [1.1000, 1.1001, 1.1000, 1.1001, 1.1000],
            current_notional_usd=50_000,
        )

        self.assertEqual(decision.action, StrategyAction.EXIT)

    def test_router_blocks_wide_spread(self) -> None:
        strategy = AlphaRouterStrategy(
            config=AlphaRouterConfig(max_spread_bps=5.0),
            momentum=MomentumConfig(threshold_bps=5.0),
        )

        decision = strategy.generate_decision(
            [1.1000, 1.1003, 1.1006, 1.1009, 1.1012],
            quote=quote(bid=1.0900, ask=1.1100),
        )

        self.assertEqual(decision.action, StrategyAction.NO_ACTION)
        self.assertIn("spread above router limit", decision.reason)

    def test_router_can_use_ml_alpha_long_signal(self) -> None:
        strategy = _ml_only_router()
        prices = [1.0000 + (index * 0.0010) for index in range(30)]

        signals = strategy.generate_signals(prices)
        ml_signal = signals[-1]
        decision = strategy.generate_decision(prices)

        self.assertEqual(ml_signal.strategy_name, "ml_alpha")
        self.assertEqual(ml_signal.direction, SignalDirection.LONG)
        self.assertGreater(ml_signal.confidence, 0.0)
        self.assertEqual(decision.action, StrategyAction.ENTER)
        self.assertGreater(decision.target_notional_usd, 0.0)
        self.assertEqual(decision.primary_signal, "ml_alpha")
        self.assertIn("ml_alpha", decision.supporting_signals)
        self.assertIn("ml_alpha=LONG", decision.reason)

    def test_router_can_use_ml_alpha_short_signal(self) -> None:
        strategy = _ml_only_router()
        prices = [1.0000 - (index * 0.0010) for index in range(30)]

        signals = strategy.generate_signals(prices)
        ml_signal = signals[-1]
        decision = strategy.generate_decision(prices)

        self.assertEqual(ml_signal.strategy_name, "ml_alpha")
        self.assertEqual(ml_signal.direction, SignalDirection.SHORT)
        self.assertEqual(decision.action, StrategyAction.ENTER)
        self.assertLess(decision.target_notional_usd, 0.0)

    def test_ml_guardrail_blocks_when_sample_count_is_too_low(self) -> None:
        strategy = AlphaRouterStrategy(
            config=AlphaRouterConfig(
                ml_enabled=True,
                ml_weight=1.0,
                ml_lookback=4,
                ml_min_train_samples=4,
                ml_min_samples_for_trade=100,
                ml_entry_probability=0.55,
                ml_label_threshold_bps=0.1,
                momentum_weight=0.0,
                moving_average_weight=0.0,
                breakout_weight=0.0,
                session_breakout_weight=0.0,
                mean_reversion_weight=0.0,
            ),
            momentum=MomentumConfig(threshold_bps=999.0),
            breakout=BreakoutConfig(breakout_buffer_bps=999.0),
            mean_reversion=MeanReversionConfig(entry_zscore=10.0, stop_zscore=20.0),
        )

        ml_signal = strategy.generate_signals(
            [1.0000 + (index * 0.0010) for index in range(30)]
        )[-1]

        self.assertEqual(ml_signal.strategy_name, "ml_alpha")
        self.assertEqual(ml_signal.direction, SignalDirection.FLAT)
        self.assertIn("sample count", ml_signal.reason)

    def test_ml_alpha_stays_flat_without_enough_training_samples(self) -> None:
        strategy = _ml_only_router()

        signals = strategy.generate_signals([1.0000, 1.0010, 1.0020, 1.0030, 1.0040])
        ml_signal = signals[-1]

        self.assertEqual(ml_signal.strategy_name, "ml_alpha")
        self.assertEqual(ml_signal.direction, SignalDirection.FLAT)
        self.assertIn("not enough labeled history", ml_signal.reason)

    def test_router_config_rejects_invalid_scores(self) -> None:
        with self.assertRaisesRegex(ValueError, "entry_score"):
            AlphaRouterConfig(entry_score=0.1, exit_score=0.1)

    def test_router_config_rejects_invalid_ml_probability(self) -> None:
        with self.assertRaisesRegex(ValueError, "ml_entry_probability"):
            AlphaRouterConfig(ml_enabled=True, ml_entry_probability=0.5)

    def test_router_config_rejects_invalid_ml_accuracy_gate(self) -> None:
        with self.assertRaisesRegex(ValueError, "ml_min_training_accuracy"):
            AlphaRouterConfig(ml_enabled=True, ml_min_training_accuracy=1.1)

    def test_router_config_rejects_invalid_adaptive_multiplier(self) -> None:
        with self.assertRaisesRegex(ValueError, "metal_mean_reversion_multiplier"):
            AlphaRouterConfig(metal_mean_reversion_multiplier=-0.1)
        with self.assertRaisesRegex(ValueError, "high_volatility_ratio"):
            AlphaRouterConfig(high_volatility_ratio=0.0)
        with self.assertRaisesRegex(ValueError, "low_volatility_ratio"):
            AlphaRouterConfig(low_volatility_ratio=2.0, high_volatility_ratio=1.5)

    def test_router_config_rejects_invalid_cross_rate_weight(self) -> None:
        with self.assertRaisesRegex(ValueError, "cross_rate_weight"):
            AlphaRouterConfig(cross_rate_weight=-0.1)

    def test_router_config_rejects_invalid_relative_strength_weight(self) -> None:
        with self.assertRaisesRegex(ValueError, "relative_strength_weight"):
            AlphaRouterConfig(relative_strength_weight=-0.1)

    def test_router_config_rejects_invalid_volatility_squeeze_weight(self) -> None:
        with self.assertRaisesRegex(ValueError, "volatility_squeeze_weight"):
            AlphaRouterConfig(volatility_squeeze_weight=-0.1)

    def test_router_config_rejects_invalid_dual_squeeze_weight(self) -> None:
        with self.assertRaisesRegex(ValueError, "dual_squeeze_weight"):
            AlphaRouterConfig(dual_squeeze_weight=-0.1)

    def test_router_config_rejects_invalid_relative_strength_gate(self) -> None:
        with self.assertRaisesRegex(ValueError, "relative_strength_min_score_dispersion"):
            AlphaRouterConfig(relative_strength_min_score_dispersion=-0.1)
        with self.assertRaisesRegex(
            ValueError,
            "relative_strength_min_target_trend_efficiency",
        ):
            AlphaRouterConfig(relative_strength_min_target_trend_efficiency=1.1)


class StrategyRegistryTest(TestCase):
    def test_aliases_are_normalized(self) -> None:
        self.assertEqual(normalize_strategy_name("momentum"), "simple_momentum")
        self.assertEqual(normalize_strategy_name("late-momentum"), "session_momentum")
        self.assertEqual(
            normalize_strategy_name("volatility-managed-momentum"),
            "multi_horizon_momentum",
        )
        self.assertEqual(
            normalize_strategy_name("dual-horizon-momentum"),
            "multi_horizon_momentum",
        )
        self.assertEqual(normalize_strategy_name("rho-regime"), "autocorrelation_regime")
        self.assertEqual(normalize_strategy_name("same-time"), "intraday_seasonality")
        self.assertEqual(normalize_strategy_name("hourly-drift"), "conditional_seasonality")
        self.assertEqual(normalize_strategy_name("moving-average"), "ma_crossover")
        self.assertEqual(normalize_strategy_name("crossover"), "ma_crossover")
        self.assertEqual(normalize_strategy_name("macd"), "macd_momentum")
        self.assertEqual(normalize_strategy_name("macd-histogram"), "macd_momentum")
        self.assertEqual(normalize_strategy_name("crypto-macd"), "asset_adaptive_macd")
        self.assertEqual(
            normalize_strategy_name("macd-squeeze"),
            "macd_squeeze_complement",
        )
        self.assertEqual(
            normalize_strategy_name("crypto-strict-macd"),
            "asset_adaptive_macd",
        )
        self.assertEqual(normalize_strategy_name("donchian"), "breakout")
        self.assertEqual(normalize_strategy_name("session-breakout"), "session_breakout")
        self.assertEqual(normalize_strategy_name("volatility-breakout"), "session_breakout")
        self.assertEqual(normalize_strategy_name("squeeze-breakout"), "volatility_squeeze")
        self.assertEqual(normalize_strategy_name("vol-squeeze"), "volatility_squeeze")
        self.assertEqual(normalize_strategy_name("dual-squeeze"), "dual_squeeze")
        self.assertEqual(normalize_strategy_name("confirmed-squeeze"), "dual_squeeze")
        self.assertEqual(
            normalize_strategy_name("adaptive-squeeze"),
            "asset_adaptive_dual_squeeze",
        )
        self.assertEqual(
            normalize_strategy_name("metal-fast-squeeze"),
            "asset_adaptive_dual_squeeze",
        )
        self.assertEqual(
            normalize_strategy_name("range-expansion"),
            "range_expansion_trend",
        )
        self.assertEqual(normalize_strategy_name("vol-expansion"), "range_expansion_trend")
        self.assertEqual(normalize_strategy_name("pullback"), "trend_pullback")
        self.assertEqual(normalize_strategy_name("momentum-pullback"), "trend_pullback")
        self.assertEqual(normalize_strategy_name("exhaustion"), "exhaustion_reversal")
        self.assertEqual(normalize_strategy_name("shock-reversal"), "exhaustion_reversal")
        self.assertEqual(normalize_strategy_name("false-breakout"), "liquidity_sweep_reversal")
        self.assertEqual(normalize_strategy_name("stop-hunt"), "liquidity_sweep_reversal")
        self.assertEqual(normalize_strategy_name("london-fix"), "fixing_reversal")
        self.assertEqual(normalize_strategy_name("post-fix-reversal"), "fixing_reversal")
        self.assertEqual(normalize_strategy_name("kalman"), "kalman_trend")
        self.assertEqual(normalize_strategy_name("time-series-trend"), "kalman_trend")
        self.assertEqual(normalize_strategy_name("confirmed-trend"), "quality_trend")
        self.assertEqual(normalize_strategy_name("macd-kalman"), "quality_trend")
        self.assertEqual(normalize_strategy_name("champion"), "champion_ensemble")
        self.assertEqual(normalize_strategy_name("champion-router"), "champion_ensemble")
        self.assertEqual(normalize_strategy_name("mean-reversion"), "mean_reversion")
        self.assertEqual(
            normalize_strategy_name("crypto-reversion"),
            "crypto_mean_reversion",
        )
        self.assertEqual(
            normalize_strategy_name("low-turnover-reversion"),
            "crypto_mean_reversion",
        )
        self.assertEqual(normalize_strategy_name("regime"), "regime_switch")
        self.assertEqual(normalize_strategy_name("router"), "alpha_router")
        self.assertEqual(
            normalize_strategy_name("crypto-blend"),
            "crypto_trend_reversion",
        )
        self.assertEqual(
            normalize_strategy_name("macd-reversion"),
            "crypto_trend_reversion",
        )
        self.assertEqual(normalize_strategy_name("usd-pressure"), "usd_pressure_router")
        self.assertEqual(normalize_strategy_name("dollar-pressure"), "usd_pressure_router")
        self.assertEqual(normalize_strategy_name("relative-strength"), "relative_strength")
        self.assertEqual(normalize_strategy_name("xs-momentum"), "relative_strength")
        self.assertEqual(normalize_strategy_name("cross-rate"), "cross_rate_reversion")
        self.assertEqual(normalize_strategy_name("triangular"), "cross_rate_reversion")

    def test_build_strategy_returns_selected_strategy(self) -> None:
        strategy = build_strategy(
            "mean_reversion",
            simple_momentum=MomentumConfig(),
            mean_reversion=MeanReversionConfig(),
        )

        self.assertIsInstance(strategy, MeanReversionStrategy)

    def test_build_strategy_returns_session_momentum_strategy(self) -> None:
        strategy = build_strategy(
            "session_momentum",
            simple_momentum=MomentumConfig(),
            session_momentum=MomentumConfig(forex_allowed_utc_hours=(17,)),
            mean_reversion=MeanReversionConfig(),
        )

        self.assertIsInstance(strategy, SimpleMomentumStrategy)
        self.assertEqual(strategy.config.forex_allowed_utc_hours, (17,))

    def test_build_strategy_returns_multi_horizon_momentum_strategy(self) -> None:
        strategy = build_strategy(
            "multi_horizon",
            simple_momentum=MomentumConfig(),
            multi_horizon_momentum=MultiHorizonMomentumConfig(fast_lookback=4),
            mean_reversion=MeanReversionConfig(),
        )

        self.assertIsInstance(strategy, MultiHorizonMomentumStrategy)
        self.assertEqual(strategy.config.fast_lookback, 4)

    def test_build_strategy_returns_autocorrelation_regime_strategy(self) -> None:
        strategy = build_strategy(
            "rho_regime",
            simple_momentum=MomentumConfig(),
            autocorrelation_regime=AutocorrelationRegimeConfig(),
            mean_reversion=MeanReversionConfig(),
        )

        self.assertIsInstance(strategy, AutocorrelationRegimeStrategy)
        self.assertEqual(strategy.config.lookback, 32)

    def test_build_strategy_returns_intraday_seasonality_strategy(self) -> None:
        strategy = build_strategy(
            "same_slot",
            simple_momentum=MomentumConfig(),
            intraday_seasonality=IntradaySeasonalityConfig(period_bars=24),
            mean_reversion=MeanReversionConfig(),
        )

        self.assertIsInstance(strategy, IntradaySeasonalityStrategy)
        self.assertEqual(strategy.config.period_bars, 24)

    def test_build_strategy_returns_conditional_seasonality_strategy(self) -> None:
        strategy = build_strategy(
            "hourly_drift",
            simple_momentum=MomentumConfig(),
            conditional_seasonality=ConditionalSeasonalityConfig(period_bars=24),
            mean_reversion=MeanReversionConfig(),
        )

        self.assertIsInstance(strategy, ConditionalSeasonalityStrategy)
        self.assertEqual(strategy.config.period_bars, 24)

    def test_build_strategy_returns_moving_average_crossover_strategy(self) -> None:
        strategy = build_strategy(
            "ma_crossover",
            simple_momentum=MomentumConfig(),
            ma_crossover=MovingAverageCrossoverConfig(),
            mean_reversion=MeanReversionConfig(),
        )

        self.assertIsInstance(strategy, MovingAverageCrossoverStrategy)

    def test_build_strategy_returns_macd_momentum_strategy(self) -> None:
        strategy = build_strategy(
            "macd",
            simple_momentum=MomentumConfig(),
            macd_momentum=MacdMomentumConfig(),
            mean_reversion=MeanReversionConfig(),
        )

        self.assertIsInstance(strategy, MacdMomentumStrategy)

    def test_asset_adaptive_macd_tightens_crypto_only(self) -> None:
        forex_strategy = build_strategy(
            "asset_adaptive_macd",
            simple_momentum=MomentumConfig(),
            macd_momentum=MacdMomentumConfig(
                symbol="EURUSD",
                min_histogram_bps=2.5,
                min_macd_bps=1.0,
                min_trend_efficiency=0.20,
                max_holding_period=12,
            ),
            mean_reversion=MeanReversionConfig(),
        )
        crypto_strategy = build_strategy(
            "asset_adaptive_macd",
            simple_momentum=MomentumConfig(),
            macd_momentum=MacdMomentumConfig(
                symbol="BTCUSD",
                min_histogram_bps=2.5,
                min_macd_bps=1.0,
                min_trend_efficiency=0.20,
                max_holding_period=12,
            ),
            mean_reversion=MeanReversionConfig(),
        )

        self.assertIsInstance(forex_strategy, MacdMomentumStrategy)
        self.assertEqual(forex_strategy.config.min_histogram_bps, 2.5)
        self.assertEqual(forex_strategy.config.min_macd_bps, 1.0)
        self.assertEqual(crypto_strategy.config.symbol, "BTCUSD")
        self.assertEqual(crypto_strategy.config.min_histogram_bps, 5.0)
        self.assertEqual(crypto_strategy.config.min_macd_bps, 2.0)
        self.assertEqual(crypto_strategy.config.min_trend_efficiency, 0.25)
        self.assertEqual(crypto_strategy.config.max_holding_period, 10)

    def test_build_strategy_returns_macd_conditional_fallback_strategy(self) -> None:
        strategy = build_strategy(
            "macd_fallback",
            simple_momentum=MomentumConfig(),
            macd_conditional_fallback=MacdConditionalFallbackConfig(
                conditional_notional_multiplier=0.2
            ),
            macd_momentum=MacdMomentumConfig(min_histogram_bps=2.0),
            conditional_seasonality=ConditionalSeasonalityConfig(
                target_notional_usd=200_000,
                max_target_notional_usd=200_000,
                position_sizing="fixed",
            ),
            mean_reversion=MeanReversionConfig(),
            symbol="XAUUSD",
        )

        self.assertIsInstance(strategy, MacdConditionalFallbackStrategy)
        self.assertEqual(strategy.config.symbol, "XAUUSD")
        self.assertEqual(strategy.macd.config.symbol, "XAUUSD")
        self.assertEqual(strategy.conditional.config.symbol, "XAUUSD")

    def test_macd_conditional_fallback_scales_conditional_entry(self) -> None:
        strategy = MacdConditionalFallbackStrategy(
            MacdConditionalFallbackConfig(conditional_notional_multiplier=0.25)
        )
        strategy.macd = _StaticDecisionStrategy(
            StrategyDecision(
                StrategyAction.NO_ACTION,
                "EURUSD",
                0.0,
                "MACD below threshold",
                primary_signal="macd_momentum",
            )
        )
        strategy.conditional = _StaticDecisionStrategy(
            StrategyDecision(
                StrategyAction.ENTER,
                "EURUSD",
                200_000.0,
                "same-slot edge",
                primary_signal="conditional_seasonality",
            )
        )

        decision = strategy.generate_decision([1.0, 1.01])

        self.assertEqual(decision.action, StrategyAction.ENTER)
        self.assertEqual(decision.target_notional_usd, 50_000.0)
        self.assertEqual(decision.primary_signal, "macd_conditional_fallback")
        self.assertIn("conditional fallback", decision.reason)

    def test_macd_conditional_fallback_respects_reason_gate(self) -> None:
        strategy = MacdConditionalFallbackStrategy(
            MacdConditionalFallbackConfig(macd_inactive_reason_keywords=("below",))
        )
        strategy.macd = _StaticDecisionStrategy(
            StrategyDecision(
                StrategyAction.NO_ACTION,
                "EURUSD",
                0.0,
                "outside MACD momentum UTC hours",
                primary_signal="macd_momentum",
            )
        )
        strategy.conditional = _StaticDecisionStrategy(
            StrategyDecision(
                StrategyAction.ENTER,
                "EURUSD",
                200_000.0,
                "same-slot edge",
                primary_signal="conditional_seasonality",
            )
        )

        decision = strategy.generate_decision([1.0, 1.01])

        self.assertEqual(decision.action, StrategyAction.NO_ACTION)
        self.assertEqual(decision.primary_signal, "macd_momentum")

    def test_macd_conditional_fallback_config_validates_multiplier(self) -> None:
        with self.assertRaisesRegex(ValueError, "conditional_notional_multiplier"):
            MacdConditionalFallbackConfig(conditional_notional_multiplier=1.01)

    def test_macd_conditional_fallback_config_validates_keywords(self) -> None:
        with self.assertRaisesRegex(ValueError, "macd_inactive_reason_keywords"):
            MacdConditionalFallbackConfig(macd_inactive_reason_keywords=("",))

    def test_build_strategy_returns_macd_squeeze_complement_strategy(self) -> None:
        strategy = build_strategy(
            "macd_squeeze",
            simple_momentum=MomentumConfig(),
            macd_squeeze_complement=MacdSqueezeComplementConfig(
                squeeze_notional_multiplier=0.5
            ),
            macd_momentum=MacdMomentumConfig(min_histogram_bps=2.5),
            volatility_squeeze=VolatilitySqueezeConfig(
                target_notional_usd=200_000,
                max_target_notional_usd=200_000,
                position_sizing="fixed",
            ),
            mean_reversion=MeanReversionConfig(),
            symbol="XAUUSD",
        )

        self.assertIsInstance(strategy, MacdSqueezeComplementStrategy)
        self.assertEqual(strategy.config.symbol, "XAUUSD")
        self.assertEqual(strategy.macd.config.symbol, "XAUUSD")
        self.assertEqual(strategy.squeeze.config.symbol, "XAUUSD")

    def test_macd_squeeze_complement_scales_squeeze_entry(self) -> None:
        strategy = MacdSqueezeComplementStrategy(
            MacdSqueezeComplementConfig(squeeze_notional_multiplier=0.25)
        )
        strategy.macd = _StaticDecisionStrategy(
            StrategyDecision(
                StrategyAction.NO_ACTION,
                "EURUSD",
                0.0,
                "MACD below threshold",
                primary_signal="macd_momentum",
            )
        )
        strategy.squeeze = _StaticDecisionStrategy(
            StrategyDecision(
                StrategyAction.ENTER,
                "EURUSD",
                200_000.0,
                "squeeze breakout",
                primary_signal="volatility_squeeze",
            )
        )

        decision = strategy.generate_decision([1.0, 1.01])

        self.assertEqual(decision.action, StrategyAction.ENTER)
        self.assertEqual(decision.target_notional_usd, 50_000.0)
        self.assertEqual(decision.primary_signal, "macd_squeeze_complement")
        self.assertIn("squeeze complement", decision.reason)

    def test_macd_squeeze_complement_preserves_macd_priority(self) -> None:
        strategy = MacdSqueezeComplementStrategy()
        strategy.macd = _StaticDecisionStrategy(
            StrategyDecision(
                StrategyAction.ENTER,
                "EURUSD",
                100_000.0,
                "MACD momentum long",
                primary_signal="macd_momentum",
            )
        )
        strategy.squeeze = _StaticDecisionStrategy(
            StrategyDecision(
                StrategyAction.ENTER,
                "EURUSD",
                -200_000.0,
                "squeeze short",
                primary_signal="volatility_squeeze",
            )
        )

        decision = strategy.generate_decision([1.0, 1.01])

        self.assertEqual(decision.target_notional_usd, 100_000.0)
        self.assertEqual(decision.primary_signal, "macd_momentum")

    def test_macd_squeeze_complement_respects_optional_reason_gate(self) -> None:
        strategy = MacdSqueezeComplementStrategy(
            MacdSqueezeComplementConfig(macd_inactive_reason_keywords=("below",))
        )
        strategy.macd = _StaticDecisionStrategy(
            StrategyDecision(
                StrategyAction.NO_ACTION,
                "EURUSD",
                0.0,
                "outside MACD momentum UTC hours",
                primary_signal="macd_momentum",
            )
        )
        strategy.squeeze = _StaticDecisionStrategy(
            StrategyDecision(
                StrategyAction.ENTER,
                "EURUSD",
                200_000.0,
                "squeeze breakout",
                primary_signal="volatility_squeeze",
            )
        )

        decision = strategy.generate_decision([1.0, 1.01])

        self.assertEqual(decision.action, StrategyAction.NO_ACTION)
        self.assertEqual(decision.primary_signal, "macd_momentum")

    def test_macd_squeeze_complement_config_validates_multiplier(self) -> None:
        with self.assertRaisesRegex(ValueError, "squeeze_notional_multiplier"):
            MacdSqueezeComplementConfig(squeeze_notional_multiplier=1.01)

    def test_build_strategy_returns_breakout_strategy(self) -> None:
        strategy = build_strategy(
            "breakout",
            simple_momentum=MomentumConfig(),
            breakout=BreakoutConfig(),
            mean_reversion=MeanReversionConfig(),
        )

        self.assertIsInstance(strategy, BreakoutStrategy)

    def test_build_strategy_returns_session_breakout_strategy(self) -> None:
        strategy = build_strategy(
            "session_breakout",
            simple_momentum=MomentumConfig(),
            breakout=BreakoutConfig(),
            session_breakout=SessionBreakoutConfig(),
            mean_reversion=MeanReversionConfig(),
        )

        self.assertIsInstance(strategy, SessionBreakoutStrategy)

    def test_build_strategy_returns_volatility_squeeze_strategy(self) -> None:
        strategy = build_strategy(
            "squeeze",
            simple_momentum=MomentumConfig(),
            volatility_squeeze=VolatilitySqueezeConfig(),
            mean_reversion=MeanReversionConfig(),
        )

        self.assertIsInstance(strategy, VolatilitySqueezeStrategy)

    def test_build_strategy_returns_trend_pullback_strategy(self) -> None:
        strategy = build_strategy(
            "pullback",
            simple_momentum=MomentumConfig(),
            trend_pullback=TrendPullbackConfig(),
            mean_reversion=MeanReversionConfig(),
        )

        self.assertIsInstance(strategy, TrendPullbackStrategy)

    def test_build_strategy_returns_dual_squeeze_strategy(self) -> None:
        strategy = build_strategy(
            "confirmed_squeeze",
            simple_momentum=MomentumConfig(),
            dual_squeeze=DualSqueezeConfig(),
            mean_reversion=MeanReversionConfig(),
        )

        self.assertIsInstance(strategy, DualSqueezeStrategy)

    def test_build_strategy_returns_asset_adaptive_dual_squeeze_strategy(self) -> None:
        strategy = build_strategy(
            "adaptive_squeeze",
            simple_momentum=MomentumConfig(),
            asset_adaptive_dual_squeeze=AssetAdaptiveDualSqueezeConfig(),
            mean_reversion=MeanReversionConfig(),
        )

        self.assertIsInstance(strategy, AssetAdaptiveDualSqueezeStrategy)

    def test_build_strategy_returns_range_expansion_trend_strategy(self) -> None:
        strategy = build_strategy(
            "vol_expansion",
            simple_momentum=MomentumConfig(),
            range_expansion_trend=RangeExpansionTrendConfig(trigger_window=3),
            mean_reversion=MeanReversionConfig(),
        )

        self.assertIsInstance(strategy, RangeExpansionTrendStrategy)
        self.assertEqual(strategy.config.trigger_window, 3)

    def test_build_strategy_returns_exhaustion_reversal_strategy(self) -> None:
        strategy = build_strategy(
            "shock_reversal",
            simple_momentum=MomentumConfig(),
            exhaustion_reversal=ExhaustionReversalConfig(),
            mean_reversion=MeanReversionConfig(),
        )

        self.assertIsInstance(strategy, ExhaustionReversalStrategy)

    def test_build_strategy_returns_liquidity_sweep_reversal_strategy(self) -> None:
        strategy = build_strategy(
            "false_breakout",
            simple_momentum=MomentumConfig(),
            liquidity_sweep_reversal=LiquiditySweepReversalConfig(lookback=10),
            mean_reversion=MeanReversionConfig(),
        )

        self.assertIsInstance(strategy, LiquiditySweepReversalStrategy)
        self.assertEqual(strategy.config.lookback, 10)

    def test_build_strategy_returns_fixing_reversal_strategy(self) -> None:
        strategy = build_strategy(
            "london_fix",
            simple_momentum=MomentumConfig(),
            fixing_reversal=FixingReversalConfig(),
            mean_reversion=MeanReversionConfig(),
        )

        self.assertIsInstance(strategy, FixingReversalStrategy)

    def test_build_strategy_returns_kalman_trend_strategy(self) -> None:
        strategy = build_strategy(
            "time_series_trend",
            simple_momentum=MomentumConfig(),
            kalman_trend=KalmanTrendStrategyConfig(),
            mean_reversion=MeanReversionConfig(),
        )

        self.assertIsInstance(strategy, KalmanTrendStrategy)

    def test_build_strategy_returns_champion_ensemble_strategy(self) -> None:
        strategy = build_strategy(
            "champion_router",
            simple_momentum=MomentumConfig(),
            asset_adaptive_dual_squeeze=AssetAdaptiveDualSqueezeConfig(),
            dual_squeeze=DualSqueezeConfig(),
            trend_pullback=TrendPullbackConfig(),
            kalman_trend=KalmanTrendStrategyConfig(),
            champion_ensemble=ChampionEnsembleConfig(),
            mean_reversion=MeanReversionConfig(),
        )

        self.assertIsInstance(strategy, ChampionEnsembleStrategy)

    def test_build_strategy_returns_quality_trend_strategy(self) -> None:
        strategy = build_strategy(
            "confirmed_trend",
            simple_momentum=MomentumConfig(),
            quality_trend=QualityTrendConfig(kalman_lookback=20),
            mean_reversion=MeanReversionConfig(),
        )

        self.assertIsInstance(strategy, QualityTrendStrategy)
        self.assertEqual(strategy.config.kalman_lookback, 20)

    def test_build_strategy_returns_regime_switching_strategy(self) -> None:
        strategy = build_strategy(
            "regime_switch",
            simple_momentum=MomentumConfig(),
            mean_reversion=MeanReversionConfig(),
            regime_switch=RegimeConfig(),
        )

        self.assertIsInstance(strategy, RegimeSwitchingStrategy)

    def test_build_strategy_returns_crypto_mean_reversion_profile(self) -> None:
        strategy = build_strategy(
            "crypto_reversion",
            simple_momentum=MomentumConfig(),
            mean_reversion=MeanReversionConfig(
                symbol="BTCUSD",
                lookback=5,
                entry_zscore=1.0,
                target_notional_usd=50_000,
                position_sizing="fixed",
                max_target_notional_usd=50_000,
                max_holding_period=20,
            ),
        )

        self.assertIsInstance(strategy, MeanReversionStrategy)
        self.assertEqual(strategy.config.symbol, "BTCUSD")
        self.assertEqual(strategy.config.lookback, 16)
        self.assertEqual(strategy.config.entry_zscore, 1.0)
        self.assertEqual(strategy.config.position_sizing, "volatility")
        self.assertEqual(strategy.config.target_notional_usd, 500_000)
        self.assertEqual(strategy.config.max_target_notional_usd, 150_000)
        self.assertEqual(strategy.config.max_holding_period, 20)
        self.assertEqual(strategy.config.cost_buffer, 1.0)

    def test_build_strategy_returns_alpha_router_strategy(self) -> None:
        strategy = build_strategy(
            "alpha_router",
            simple_momentum=MomentumConfig(),
            mean_reversion=MeanReversionConfig(),
            regime_switch=RegimeConfig(),
            alpha_router=AlphaRouterConfig(),
        )

        self.assertIsInstance(strategy, AlphaRouterStrategy)

    def test_build_strategy_returns_crypto_trend_reversion_profile(self) -> None:
        strategy = build_strategy(
            "crypto_blend",
            simple_momentum=MomentumConfig(),
            macd_momentum=MacdMomentumConfig(
                symbol="BTCUSD",
                target_notional_usd=800_000,
                max_target_notional_usd=800_000,
            ),
            mean_reversion=MeanReversionConfig(symbol="BTCUSD"),
            alpha_router=AlphaRouterConfig(symbol="BTCUSD"),
        )

        self.assertIsInstance(strategy, AlphaRouterStrategy)
        self.assertEqual(strategy.config.symbol, "BTCUSD")
        self.assertEqual(strategy.config.macd_momentum_weight, 0.70)
        self.assertEqual(strategy.config.mean_reversion_weight, 0.30)
        self.assertEqual(strategy.config.momentum_weight, 0.0)
        self.assertEqual(strategy.config.target_notional_usd, 800_000)

    def test_build_strategy_returns_usd_pressure_router_strategy(self) -> None:
        strategy = build_strategy(
            "usd_pressure_router",
            simple_momentum=MomentumConfig(),
            mean_reversion=MeanReversionConfig(),
            regime_switch=RegimeConfig(),
            alpha_router=AlphaRouterConfig(),
            usd_pressure=UsdPressureConfig(),
        )

        self.assertIsInstance(strategy, UsdPressureRouterStrategy)

    def test_build_strategy_returns_relative_strength_strategy(self) -> None:
        strategy = build_strategy(
            "relative_strength",
            simple_momentum=MomentumConfig(),
            mean_reversion=MeanReversionConfig(),
            relative_strength=RelativeStrengthConfig(),
        )

        self.assertIsInstance(strategy, RelativeStrengthStrategy)

    def test_build_strategy_returns_cross_rate_reversion_strategy(self) -> None:
        strategy = build_strategy(
            "cross_rate_reversion",
            simple_momentum=MomentumConfig(),
            mean_reversion=MeanReversionConfig(),
            cross_rate_reversion=CrossRateReversionConfig(),
        )

        self.assertIsInstance(strategy, CrossRateReversionStrategy)

    def test_unknown_strategy_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown strategy"):
            normalize_strategy_name("coin_flip")


def _ml_only_router() -> AlphaRouterStrategy:
    return AlphaRouterStrategy(
        config=AlphaRouterConfig(
            entry_score=0.20,
            ml_enabled=True,
            ml_weight=1.0,
            ml_lookback=4,
            ml_min_train_samples=8,
            ml_entry_probability=0.55,
            ml_label_threshold_bps=0.1,
            momentum_weight=0.0,
            moving_average_weight=0.0,
            breakout_weight=0.0,
            session_breakout_weight=0.0,
            mean_reversion_weight=0.0,
        ),
        momentum=MomentumConfig(threshold_bps=999.0),
        breakout=BreakoutConfig(breakout_buffer_bps=999.0),
        mean_reversion=MeanReversionConfig(entry_zscore=10.0, stop_zscore=20.0),
    )
