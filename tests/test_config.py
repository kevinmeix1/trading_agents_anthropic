from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.core.clock import CompetitionMode, LONDON
from quanthack.core.config import load_config
from quanthack.trading.risk import AccountSnapshot, PortfolioSnapshot, RiskEngine
from quanthack.strategies.strategy import (
    AlphaRouterStrategy,
    AutocorrelationRegimeStrategy,
    AssetAdaptiveDualSqueezeStrategy,
    BreakoutStrategy,
    ChampionEnsembleStrategy,
    ConditionalSeasonalityStrategy,
    CrossRateReversionStrategy,
    DualSqueezeStrategy,
    ExhaustionReversalStrategy,
    FixingReversalStrategy,
    IntradaySeasonalityStrategy,
    KalmanTrendStrategy,
    LiquiditySweepReversalStrategy,
    MacdMomentumStrategy,
    MacdConditionalFallbackStrategy,
    MacdSqueezeComplementStrategy,
    MeanReversionStrategy,
    MovingAverageCrossoverStrategy,
    QualityTrendStrategy,
    RangeExpansionTrendStrategy,
    RegimeSwitchingStrategy,
    RelativeStrengthStrategy,
    SessionBreakoutStrategy,
    SimpleMomentumStrategy,
    TrendPullbackStrategy,
    UsdPressureRouterStrategy,
    VolatilitySqueezeStrategy,
)


class ConfigTest(TestCase):
    def test_default_config_loads_hackathon_defaults(self) -> None:
        config = load_config("configs/default.toml")

        self.assertEqual(config.competition.timezone, "Europe/London")
        self.assertEqual(config.competition.starting_equity, 1_000_000)
        self.assertEqual(config.risk.max_gross_leverage, 2.0)
        self.assertEqual(config.risk.max_daily_loss_pct, 0.025)
        self.assertEqual(config.risk.max_position_loss_pct, 0.01)
        self.assertIsNone(config.risk.max_metal_position_loss_pct)
        self.assertEqual(config.market_quality.max_spread_bps, 10.0)
        self.assertEqual(config.market_quality.max_quote_age_seconds, 5.0)
        self.assertEqual(config.active_strategy, "simple_momentum")
        self.assertEqual(config.simple_momentum.symbol, "EURUSD")
        self.assertEqual(config.simple_momentum.exit_threshold_bps, 3.0)
        self.assertEqual(config.simple_momentum.min_trend_efficiency, 0.4)
        self.assertEqual(config.simple_momentum.position_sizing, "fixed")
        self.assertEqual(config.session_momentum.forex_allowed_utc_hours, (17, 18, 19, 20, 21))
        self.assertEqual(config.session_momentum.metal_allowed_utc_hours, (17, 18, 19, 20, 21))
        self.assertEqual(config.session_momentum.crypto_allowed_utc_hours, ())
        self.assertEqual(config.session_momentum.position_sizing, "volatility")
        self.assertEqual(config.multi_horizon_momentum.fast_lookback, 6)
        self.assertEqual(config.multi_horizon_momentum.slow_lookback, 24)
        self.assertEqual(
            config.multi_horizon_momentum.forex_allowed_utc_hours,
            (10, 11, 12, 13, 14),
        )
        self.assertEqual(config.multi_horizon_momentum.position_sizing, "volatility")
        self.assertEqual(config.autocorrelation_regime.lookback, 32)
        self.assertEqual(config.autocorrelation_regime.signal_lookback, 6)
        self.assertEqual(config.autocorrelation_regime.min_abs_autocorrelation, 0.18)
        self.assertEqual(config.autocorrelation_regime.position_sizing, "volatility")
        self.assertEqual(config.intraday_seasonality.period_bars, 96)
        self.assertEqual(config.intraday_seasonality.lookback_periods, 5)
        self.assertEqual(config.intraday_seasonality.signal_mode, "reversal")
        self.assertEqual(config.intraday_seasonality.entry_threshold_bps, 0.5)
        self.assertIsNone(config.intraday_seasonality.forex_allowed_utc_hours)
        self.assertEqual(config.intraday_seasonality.position_sizing, "volatility")
        self.assertEqual(config.conditional_seasonality.period_bars, 96)
        self.assertEqual(config.conditional_seasonality.horizon_bars, 4)
        self.assertEqual(config.conditional_seasonality.momentum_lookback, 4)
        self.assertEqual(config.conditional_seasonality.signal_mode, "reversal")
        self.assertEqual(config.conditional_seasonality.min_samples, 3)
        self.assertEqual(config.conditional_seasonality.entry_threshold_bps, 10.0)
        self.assertEqual(config.conditional_seasonality.position_sizing, "volatility")
        self.assertEqual(config.ma_crossover.fast_window, 3)
        self.assertEqual(config.ma_crossover.slow_window, 8)
        self.assertEqual(config.ma_crossover.min_separation_bps, 2.0)
        self.assertEqual(config.breakout.lookback, 8)
        self.assertEqual(config.breakout.breakout_buffer_bps, 2.0)
        self.assertEqual(config.volatility_squeeze.lookback, 24)
        self.assertEqual(config.volatility_squeeze.squeeze_window, 8)
        self.assertEqual(config.volatility_squeeze.breakout_buffer_bps, 2.5)
        self.assertEqual(config.volatility_squeeze.max_squeeze_ratio, 0.50)
        self.assertEqual(
            config.volatility_squeeze.forex_allowed_utc_hours,
            (11, 12, 13, 14, 15, 16, 17, 18, 19),
        )
        self.assertEqual(
            config.volatility_squeeze.metal_allowed_utc_hours,
            (11, 12, 13, 14, 15, 16, 17, 18, 19),
        )
        self.assertIsNone(config.volatility_squeeze.crypto_allowed_utc_hours)
        self.assertEqual(config.volatility_squeeze.position_sizing, "volatility")
        self.assertEqual(config.volatility_squeeze.max_holding_period, 24)
        self.assertEqual(config.dual_squeeze.lookback, 14)
        self.assertEqual(config.dual_squeeze.squeeze_window, 4)
        self.assertEqual(config.dual_squeeze.confirmation_lookback, 24)
        self.assertEqual(config.dual_squeeze.confirmation_squeeze_window, 8)
        self.assertEqual(config.dual_squeeze.confirmation_mode, "squeeze_bias")
        self.assertEqual(config.dual_squeeze.max_holding_period, 12)
        self.assertEqual(config.asset_adaptive_dual_squeeze.base_lookback, 14)
        self.assertEqual(config.asset_adaptive_dual_squeeze.metal_lookback, 12)
        self.assertEqual(
            config.asset_adaptive_dual_squeeze.metal_confirmation_lookback,
            20,
        )
        self.assertEqual(config.range_expansion_trend.lookback, 40)
        self.assertEqual(config.range_expansion_trend.trigger_window, 4)
        self.assertEqual(config.range_expansion_trend.min_trigger_move_bps, 10.0)
        self.assertEqual(config.range_expansion_trend.min_range_break_bps, 3.0)
        self.assertEqual(config.range_expansion_trend.min_expansion_zscore, 2.5)
        self.assertEqual(config.range_expansion_trend.min_trend_efficiency, 0.65)
        self.assertEqual(
            config.range_expansion_trend.forex_allowed_utc_hours,
            (10, 11, 12, 13, 14),
        )
        self.assertEqual(config.range_expansion_trend.position_sizing, "volatility")
        self.assertEqual(config.range_expansion_trend.max_holding_period, 6)
        self.assertEqual(config.session_breakout.allowed_utc_hours, (12, 13, 14, 15))
        self.assertEqual(config.session_breakout.metal_allowed_utc_hours, (12, 13, 14, 15, 16, 17))
        self.assertEqual(config.session_breakout.min_expected_edge_bps, 4.0)
        self.assertEqual(config.session_breakout.min_holding_period, 2)
        self.assertFalse(config.session_breakout.require_regime_confirmation)
        self.assertEqual(config.session_breakout.regime_lookback, 80)
        self.assertEqual(config.session_breakout.position_sizing, "volatility")
        self.assertEqual(config.trend_pullback.lookback, 32)
        self.assertEqual(config.trend_pullback.pullback_window, 4)
        self.assertEqual(config.trend_pullback.min_expected_edge_bps, 3.0)
        self.assertEqual(
            config.trend_pullback.forex_allowed_utc_hours,
            (11, 12, 13, 14, 15, 16, 17, 18, 19),
        )
        self.assertEqual(config.trend_pullback.position_sizing, "volatility")
        self.assertEqual(config.exhaustion_reversal.lookback, 32)
        self.assertEqual(config.exhaustion_reversal.shock_window, 4)
        self.assertEqual(config.exhaustion_reversal.min_expected_edge_bps, 3.0)
        self.assertEqual(
            config.exhaustion_reversal.forex_allowed_utc_hours,
            (11, 12, 13, 14, 15, 16, 17, 18, 19),
        )
        self.assertEqual(config.exhaustion_reversal.position_sizing, "volatility")
        self.assertEqual(config.liquidity_sweep_reversal.lookback, 32)
        self.assertEqual(config.liquidity_sweep_reversal.min_sweep_bps, 2.0)
        self.assertEqual(config.liquidity_sweep_reversal.min_expected_edge_bps, 2.0)
        self.assertEqual(
            config.liquidity_sweep_reversal.forex_allowed_utc_hours,
            (10, 11, 12, 13, 14, 15, 16, 17, 18, 19),
        )
        self.assertEqual(config.liquidity_sweep_reversal.position_sizing, "volatility")
        self.assertEqual(config.fixing_reversal.pre_fix_lookback, 4)
        self.assertEqual(config.fixing_reversal.min_pre_fix_move_bps, 8.0)
        self.assertEqual(config.fixing_reversal.min_reversal_confirmation_bps, 1.5)
        self.assertEqual(config.fixing_reversal.forex_allowed_utc_hours, (14,))
        self.assertEqual(config.fixing_reversal.crypto_allowed_utc_hours, ())
        self.assertEqual(config.fixing_reversal.max_holding_period, 4)
        self.assertEqual(config.fixing_reversal.position_sizing, "volatility")
        self.assertEqual(config.macd_momentum.fast_window, 6)
        self.assertEqual(config.macd_momentum.slow_window, 18)
        self.assertEqual(config.macd_momentum.signal_window, 5)
        self.assertEqual(config.macd_momentum.min_histogram_bps, 2.0)
        self.assertEqual(config.macd_momentum.min_histogram_slope_bps, 0.0)
        self.assertEqual(config.macd_momentum.forex_allowed_utc_hours, (10, 11, 12, 13, 14))
        self.assertEqual(config.macd_momentum.metal_allowed_utc_hours, (10, 11, 12, 13, 14))
        self.assertEqual(config.macd_momentum.position_sizing, "volatility")
        self.assertEqual(config.kalman_trend.lookback, 80)
        self.assertEqual(config.kalman_trend.min_abs_slope_bps, 0.25)
        self.assertEqual(config.kalman_trend.min_trend_efficiency, 0.20)
        self.assertEqual(config.kalman_trend.expected_holding_bars, 6)
        self.assertEqual(config.kalman_trend.min_expected_edge_bps, 5.0)
        self.assertEqual(config.kalman_trend.max_holding_period, 32)
        self.assertEqual(config.kalman_trend.position_sizing, "volatility")
        self.assertEqual(config.quality_trend.kalman_lookback, 80)
        self.assertEqual(config.quality_trend.macd_fast_window, 6)
        self.assertEqual(config.quality_trend.min_combined_confidence, 0.30)
        self.assertEqual(config.quality_trend.forex_allowed_utc_hours, (10, 11, 12, 13, 14))
        self.assertEqual(config.quality_trend.max_holding_period, 16)
        self.assertEqual(config.champion_ensemble.entry_score, 0.50)
        self.assertEqual(config.champion_ensemble.strong_lead_score, 0.50)
        self.assertEqual(config.champion_ensemble.kalman_trend_weight, 0.70)
        self.assertEqual(
            config.champion_ensemble.asset_adaptive_dual_squeeze_weight,
            0.30,
        )
        self.assertEqual(config.champion_ensemble.fixing_reversal_weight, 0.0)
        self.assertEqual(config.champion_ensemble.macd_momentum_weight, 0.0)
        self.assertEqual(config.mean_reversion.entry_zscore, 1.0)
        self.assertEqual(config.mean_reversion.exit_zscore, 0.25)
        self.assertEqual(config.mean_reversion.max_holding_period, 20)
        self.assertEqual(config.regime_switch.lookback, 10)
        self.assertEqual(config.regime_switch.hysteresis_bars, 2)
        self.assertEqual(config.alpha_router.entry_score, 0.35)
        self.assertEqual(config.alpha_router.momentum_weight, 0.30)
        self.assertEqual(config.alpha_router.moving_average_weight, 0.15)
        self.assertEqual(config.alpha_router.breakout_weight, 0.15)
        self.assertEqual(config.alpha_router.session_breakout_weight, 0.25)
        self.assertEqual(config.alpha_router.macd_momentum_weight, 0.0)
        self.assertEqual(config.alpha_router.kalman_trend_weight, 0.0)
        self.assertEqual(config.alpha_router.volatility_squeeze_weight, 0.0)
        self.assertEqual(config.alpha_router.dual_squeeze_weight, 0.0)
        self.assertEqual(config.alpha_router.exhaustion_reversal_weight, 0.0)
        self.assertEqual(config.alpha_router.mean_reversion_weight, 0.35)
        self.assertEqual(config.alpha_router.relative_strength_weight, 0.0)
        self.assertEqual(config.alpha_router.cross_rate_weight, 0.0)
        self.assertTrue(config.alpha_router.adaptive_weighting_enabled)
        self.assertEqual(config.alpha_router.adaptive_regime_lookback, 80)
        self.assertEqual(config.alpha_router.metal_mean_reversion_multiplier, 1.25)
        self.assertEqual(config.alpha_router.metal_raw_breakout_multiplier, 0.60)
        self.assertTrue(config.alpha_router.volatility_regime_enabled)
        self.assertEqual(config.alpha_router.volatility_regime_lookback, 24)
        self.assertEqual(config.alpha_router.high_volatility_ratio, 1.50)
        self.assertEqual(config.alpha_router.low_volatility_ratio, 0.60)
        self.assertEqual(
            config.alpha_router.high_volatility_reversion_multiplier,
            1.15,
        )
        self.assertEqual(config.alpha_router.high_volatility_trend_multiplier, 0.90)
        self.assertEqual(config.alpha_router.relative_strength_min_score_dispersion, 0.75)
        self.assertEqual(
            config.alpha_router.relative_strength_min_target_trend_efficiency,
            0.20,
        )
        self.assertFalse(config.alpha_router.ml_enabled)
        self.assertEqual(config.alpha_router.ml_lookback, 5)
        self.assertEqual(config.alpha_router.ml_min_train_samples, 8)
        self.assertEqual(config.alpha_router.ml_min_samples_for_trade, 12)
        self.assertEqual(config.alpha_router.ml_min_training_accuracy, 0.55)
        self.assertEqual(config.usd_pressure.lookback, 8)
        self.assertEqual(config.usd_pressure.min_target_volatility_bps, 0.0)
        self.assertEqual(config.usd_pressure.min_component_symbols, 3)
        self.assertTrue(config.usd_pressure.exit_on_conflict)
        self.assertEqual(config.relative_strength.lookback, 12)
        self.assertEqual(config.relative_strength.entry_zscore, 0.75)
        self.assertEqual(config.relative_strength.min_component_symbols, 4)
        self.assertFalse(config.relative_strength.require_asset_class_confirmation)
        self.assertEqual(config.relative_strength.asset_class_entry_zscore, 0.35)
        self.assertEqual(config.relative_strength.asset_class_min_symbols, 2)
        self.assertFalse(config.relative_strength.require_metal_trend_confirmation)
        self.assertEqual(config.relative_strength.metal_trend_min_move_bps, 2.0)
        self.assertEqual(config.relative_strength.metal_trend_min_efficiency, 0.20)
        self.assertEqual(config.relative_strength.min_score_dispersion, 0.0)
        self.assertEqual(config.relative_strength.min_target_trend_efficiency, 0.0)
        self.assertEqual(config.cross_rate_reversion.symbol, "EURUSD")
        self.assertEqual(config.cross_rate_reversion.allowed_symbols, ())
        self.assertEqual(config.cross_rate_reversion.lookback, 12)
        self.assertEqual(config.cross_rate_reversion.entry_zscore, 1.0)
        self.assertEqual(config.cross_rate_reversion.position_sizing, "volatility")
        self.assertEqual(config.cross_rate_reversion.max_holding_period, 24)
        self.assertEqual(config.market_data.price_csv, "data/sample_prices.csv")
        self.assertEqual(config.market_data.quote_csv, "data/sample_quotes.csv")
        self.assertEqual(config.execution.route, "dry_run")
        self.assertEqual(config.live_dry_run.adapter, "csv")
        self.assertEqual(config.live_dry_run.timeframe, "M1")
        self.assertEqual(config.live_dry_run.bars, 120)
        self.assertEqual(config.live_dry_run.iterations, 1)
        self.assertEqual(config.live_dry_run.poll_seconds, 0.0)
        self.assertEqual(config.live_dry_run.journal_path, "outputs/live_dry_run_journal.jsonl")
        self.assertEqual(
            config.live_dry_run.monitor_csv,
            "outputs/live_competition_monitor.csv",
        )
        self.assertEqual(config.backtest.pnl_ledger_csv, "outputs/backtests/pnl_ledger.csv")
        self.assertEqual(config.sweep.lookbacks, (3, 5, 7))
        self.assertEqual(config.sweep.threshold_bps, (4.0, 8.0, 12.0))
        self.assertEqual(config.walk_forward.train_size, 10)
        self.assertEqual(config.walk_forward.cost_multipliers, (1.5, 2.0))
        self.assertEqual(config.walk_forward.ma_fast_windows, (2, 3))
        self.assertEqual(config.walk_forward.ma_slow_windows, (5, 8))
        self.assertEqual(config.walk_forward.ma_min_separation_bps, (1.0, 2.0))

    def test_config_builds_selected_strategy(self) -> None:
        config = load_config("configs/default.toml")

        strategy = config.build_strategy("mean_reversion")

        self.assertIsInstance(strategy, MeanReversionStrategy)
        self.assertEqual(config.strategy_symbol("mean_reversion"), "EURUSD")

        session_momentum_strategy = config.build_strategy("late_momentum")
        self.assertIsInstance(session_momentum_strategy, SimpleMomentumStrategy)
        self.assertEqual(config.strategy_symbol("session_momentum"), "EURUSD")

        intraday_strategy = config.build_strategy("same_time")
        self.assertIsInstance(intraday_strategy, IntradaySeasonalityStrategy)
        self.assertEqual(config.strategy_symbol("intraday_seasonality"), "EURUSD")

        conditional_strategy = config.build_strategy("hourly_drift")
        self.assertIsInstance(conditional_strategy, ConditionalSeasonalityStrategy)
        self.assertEqual(config.strategy_symbol("conditional_seasonality"), "EURUSD")

        ma_strategy = config.build_strategy("ma_crossover")
        self.assertIsInstance(ma_strategy, MovingAverageCrossoverStrategy)
        self.assertEqual(config.strategy_symbol("ma_crossover"), "EURUSD")

        adaptive_macd_strategy = config.build_strategy("asset_adaptive_macd")
        self.assertIsInstance(adaptive_macd_strategy, MacdMomentumStrategy)
        self.assertEqual(config.strategy_symbol("asset_adaptive_macd"), "EURUSD")

        breakout_strategy = config.build_strategy("breakout")
        self.assertIsInstance(breakout_strategy, BreakoutStrategy)
        self.assertEqual(config.strategy_symbol("breakout"), "EURUSD")

        squeeze_strategy = config.build_strategy("squeeze")
        self.assertIsInstance(squeeze_strategy, VolatilitySqueezeStrategy)
        self.assertEqual(config.strategy_symbol("volatility_squeeze"), "EURUSD")

        dual_squeeze_strategy = config.build_strategy("dual-squeeze")
        self.assertIsInstance(dual_squeeze_strategy, DualSqueezeStrategy)
        self.assertEqual(config.strategy_symbol("dual_squeeze"), "EURUSD")

        adaptive_squeeze_strategy = config.build_strategy("adaptive-squeeze")
        self.assertIsInstance(
            adaptive_squeeze_strategy,
            AssetAdaptiveDualSqueezeStrategy,
        )
        self.assertEqual(config.strategy_symbol("asset_adaptive_dual_squeeze"), "EURUSD")

        range_expansion_strategy = config.build_strategy("range-expansion")
        self.assertIsInstance(range_expansion_strategy, RangeExpansionTrendStrategy)
        self.assertEqual(config.strategy_symbol("range_expansion_trend"), "EURUSD")

        session_breakout_strategy = config.build_strategy("session")
        self.assertIsInstance(session_breakout_strategy, SessionBreakoutStrategy)
        self.assertEqual(config.strategy_symbol("session_breakout"), "EURUSD")

        autocorrelation_strategy = config.build_strategy("rho_regime")
        self.assertIsInstance(autocorrelation_strategy, AutocorrelationRegimeStrategy)
        self.assertEqual(config.strategy_symbol("autocorrelation_regime"), "EURUSD")

        pullback_strategy = config.build_strategy("pullback")
        self.assertIsInstance(pullback_strategy, TrendPullbackStrategy)
        self.assertEqual(config.strategy_symbol("trend_pullback"), "EURUSD")

        exhaustion_strategy = config.build_strategy("exhaustion")
        self.assertIsInstance(exhaustion_strategy, ExhaustionReversalStrategy)
        self.assertEqual(config.strategy_symbol("exhaustion_reversal"), "EURUSD")

        liquidity_strategy = config.build_strategy("false_breakout")
        self.assertIsInstance(liquidity_strategy, LiquiditySweepReversalStrategy)
        self.assertEqual(config.strategy_symbol("liquidity_sweep_reversal"), "EURUSD")

        fixing_strategy = config.build_strategy("london_fix")
        self.assertIsInstance(fixing_strategy, FixingReversalStrategy)
        self.assertEqual(config.strategy_symbol("fixing_reversal"), "EURUSD")

        kalman_strategy = config.build_strategy("kalman")
        self.assertIsInstance(kalman_strategy, KalmanTrendStrategy)
        self.assertEqual(config.strategy_symbol("kalman_trend"), "EURUSD")

        quality_strategy = config.build_strategy("confirmed_trend")
        self.assertIsInstance(quality_strategy, QualityTrendStrategy)
        self.assertEqual(config.strategy_symbol("quality_trend"), "EURUSD")

        champion_strategy = config.build_strategy("champion")
        self.assertIsInstance(champion_strategy, ChampionEnsembleStrategy)
        self.assertEqual(config.strategy_symbol("champion_ensemble"), "EURUSD")

        regime_strategy = config.build_strategy("regime")
        self.assertIsInstance(regime_strategy, RegimeSwitchingStrategy)
        self.assertEqual(config.strategy_symbol("regime_switch"), "EURUSD")

        router_strategy = config.build_strategy("router")
        self.assertIsInstance(router_strategy, AlphaRouterStrategy)
        self.assertEqual(config.strategy_symbol("alpha_router"), "EURUSD")

        usd_strategy = config.build_strategy("usd_pressure")
        self.assertIsInstance(usd_strategy, UsdPressureRouterStrategy)
        self.assertEqual(config.strategy_symbol("usd_pressure_router"), "EURUSD")

        relative_strategy = config.build_strategy("relative")
        self.assertIsInstance(relative_strategy, RelativeStrengthStrategy)
        self.assertEqual(config.strategy_symbol("relative_strength"), "EURUSD")

        cross_rate_strategy = config.build_strategy("cross-rate")
        self.assertIsInstance(cross_rate_strategy, CrossRateReversionStrategy)
        self.assertEqual(config.strategy_symbol("cross_rate_reversion"), "EURUSD")

    def test_config_builds_symbol_with_instrument_spread_limits(self) -> None:
        config = load_config("configs/default.toml")

        strategy = config.build_strategy("alpha_router", symbol="BTCUSD")

        self.assertIsInstance(strategy, AlphaRouterStrategy)
        self.assertEqual(strategy.config.symbol, "BTCUSD")
        self.assertEqual(strategy.config.max_spread_bps, 60.0)
        self.assertEqual(strategy.momentum.config.max_spread_bps, 60.0)
        self.assertEqual(strategy.breakout.config.max_spread_bps, 60.0)

        adaptive_macd_strategy = config.build_strategy(
            "asset_adaptive_macd",
            symbol="BTCUSD",
        )
        self.assertIsInstance(adaptive_macd_strategy, MacdMomentumStrategy)
        self.assertEqual(adaptive_macd_strategy.config.symbol, "BTCUSD")
        self.assertEqual(adaptive_macd_strategy.config.max_spread_bps, 60.0)
        self.assertEqual(adaptive_macd_strategy.config.min_histogram_bps, 5.0)
        self.assertEqual(adaptive_macd_strategy.config.min_macd_bps, 2.0)

        session_momentum_strategy = config.build_strategy(
            "session_momentum",
            symbol="BTCUSD",
        )
        self.assertIsInstance(session_momentum_strategy, SimpleMomentumStrategy)
        self.assertEqual(session_momentum_strategy.config.symbol, "BTCUSD")
        self.assertEqual(session_momentum_strategy.config.max_spread_bps, 60.0)

        intraday_strategy = config.build_strategy(
            "intraday_seasonality",
            symbol="BTCUSD",
        )
        self.assertIsInstance(intraday_strategy, IntradaySeasonalityStrategy)
        self.assertEqual(intraday_strategy.config.symbol, "BTCUSD")
        self.assertEqual(intraday_strategy.config.max_spread_bps, 60.0)

        conditional_strategy = config.build_strategy(
            "conditional_seasonality",
            symbol="BTCUSD",
        )
        self.assertIsInstance(conditional_strategy, ConditionalSeasonalityStrategy)
        self.assertEqual(conditional_strategy.config.symbol, "BTCUSD")
        self.assertEqual(conditional_strategy.config.max_spread_bps, 60.0)

        squeeze_strategy = config.build_strategy("volatility_squeeze", symbol="BTCUSD")
        self.assertIsInstance(squeeze_strategy, VolatilitySqueezeStrategy)
        self.assertEqual(squeeze_strategy.config.symbol, "BTCUSD")
        self.assertEqual(squeeze_strategy.config.max_spread_bps, 60.0)

        dual_squeeze_strategy = config.build_strategy("dual_squeeze", symbol="BTCUSD")
        self.assertIsInstance(dual_squeeze_strategy, DualSqueezeStrategy)
        self.assertEqual(dual_squeeze_strategy.config.symbol, "BTCUSD")
        self.assertEqual(dual_squeeze_strategy.config.max_spread_bps, 60.0)

        adaptive_squeeze_strategy = config.build_strategy(
            "asset_adaptive_dual_squeeze",
            symbol="XAUUSD",
        )
        self.assertIsInstance(
            adaptive_squeeze_strategy,
            AssetAdaptiveDualSqueezeStrategy,
        )
        self.assertEqual(adaptive_squeeze_strategy.config.symbol, "XAUUSD")
        self.assertEqual(adaptive_squeeze_strategy.config.max_spread_bps, 25.0)
        self.assertEqual(adaptive_squeeze_strategy.selected_profile, "metal_fast")

        range_expansion_strategy = config.build_strategy(
            "range_expansion_trend",
            symbol="BTCUSD",
        )
        self.assertIsInstance(range_expansion_strategy, RangeExpansionTrendStrategy)
        self.assertEqual(range_expansion_strategy.config.symbol, "BTCUSD")
        self.assertEqual(range_expansion_strategy.config.max_spread_bps, 60.0)

        session_strategy = config.build_strategy("session_breakout", symbol="BTCUSD")
        self.assertIsInstance(session_strategy, SessionBreakoutStrategy)
        self.assertEqual(session_strategy.config.symbol, "BTCUSD")
        self.assertEqual(session_strategy.config.max_spread_bps, 60.0)

        autocorrelation_strategy = config.build_strategy(
            "autocorrelation_regime",
            symbol="BTCUSD",
        )
        self.assertIsInstance(autocorrelation_strategy, AutocorrelationRegimeStrategy)
        self.assertEqual(autocorrelation_strategy.config.symbol, "BTCUSD")
        self.assertEqual(autocorrelation_strategy.config.max_spread_bps, 60.0)

        pullback_strategy = config.build_strategy("trend_pullback", symbol="BTCUSD")
        self.assertIsInstance(pullback_strategy, TrendPullbackStrategy)
        self.assertEqual(pullback_strategy.config.symbol, "BTCUSD")
        self.assertEqual(pullback_strategy.config.max_spread_bps, 60.0)

        exhaustion_strategy = config.build_strategy("exhaustion_reversal", symbol="BTCUSD")
        self.assertIsInstance(exhaustion_strategy, ExhaustionReversalStrategy)
        self.assertEqual(exhaustion_strategy.config.symbol, "BTCUSD")
        self.assertEqual(exhaustion_strategy.config.max_spread_bps, 60.0)

        liquidity_strategy = config.build_strategy(
            "liquidity_sweep_reversal",
            symbol="BTCUSD",
        )
        self.assertIsInstance(liquidity_strategy, LiquiditySweepReversalStrategy)
        self.assertEqual(liquidity_strategy.config.symbol, "BTCUSD")
        self.assertEqual(liquidity_strategy.config.max_spread_bps, 60.0)

        fixing_strategy = config.build_strategy("fixing_reversal", symbol="XAUUSD")
        self.assertIsInstance(fixing_strategy, FixingReversalStrategy)
        self.assertEqual(fixing_strategy.config.symbol, "XAUUSD")
        self.assertEqual(fixing_strategy.config.max_spread_bps, 25.0)

        kalman_strategy = config.build_strategy("kalman_trend", symbol="BTCUSD")
        self.assertIsInstance(kalman_strategy, KalmanTrendStrategy)
        self.assertEqual(kalman_strategy.config.symbol, "BTCUSD")
        self.assertEqual(kalman_strategy.config.max_spread_bps, 60.0)

        quality_strategy = config.build_strategy("quality_trend", symbol="BTCUSD")
        self.assertIsInstance(quality_strategy, QualityTrendStrategy)
        self.assertEqual(quality_strategy.config.symbol, "BTCUSD")
        self.assertEqual(quality_strategy.config.max_spread_bps, 60.0)

        champion_strategy = config.build_strategy("champion_ensemble", symbol="BTCUSD")
        self.assertIsInstance(champion_strategy, ChampionEnsembleStrategy)
        self.assertEqual(champion_strategy.config.symbol, "BTCUSD")
        self.assertEqual(champion_strategy.config.max_spread_bps, 60.0)

        usd_strategy = config.build_strategy("usd_pressure_router", symbol="BTCUSD")
        self.assertIsInstance(usd_strategy, UsdPressureRouterStrategy)
        self.assertEqual(usd_strategy.config.symbol, "BTCUSD")
        self.assertEqual(usd_strategy.config.max_spread_bps, 60.0)

        relative_strategy = config.build_strategy("relative_strength", symbol="BTCUSD")
        self.assertIsInstance(relative_strategy, RelativeStrengthStrategy)
        self.assertEqual(relative_strategy.config.symbol, "BTCUSD")
        self.assertEqual(relative_strategy.config.max_spread_bps, 60.0)

        cross_rate_strategy = config.build_strategy("cross_rate_reversion", symbol="BTCUSD")
        self.assertIsInstance(cross_rate_strategy, CrossRateReversionStrategy)
        self.assertEqual(cross_rate_strategy.config.symbol, "BTCUSD")
        self.assertEqual(cross_rate_strategy.config.max_spread_bps, 60.0)

    def test_config_creates_clock(self) -> None:
        config = load_config("configs/default.toml")
        clock = config.competition.to_clock()
        now = datetime(2026, 6, 22, 21, 15, tzinfo=LONDON)

        self.assertEqual(clock.mode_at(now), CompetitionMode.CHECKPOINT_PROTECT)

    def test_config_feeds_strategy_and_risk(self) -> None:
        config = load_config("configs/default.toml")
        strategy = SimpleMomentumStrategy(config.simple_momentum)
        request = strategy.generate_request([1.1000, 1.1002, 1.1004, 1.1007, 1.1010])

        self.assertIsNotNone(request)

        decision = RiskEngine(config.risk).evaluate(
            account=AccountSnapshot(equity=1_000_000, margin_level_pct=2_000),
            portfolio=PortfolioSnapshot(),
            request=request,
            mode=CompetitionMode.QUALIFY,
        )

        self.assertTrue(decision.approved)
        self.assertEqual(decision.adjusted_notional_usd, 50_000)

    def test_competition_config_loads_larger_research_profile(self) -> None:
        config = load_config("configs/competition.toml")

        config.validate_symbols()
        self.assertEqual(config.active_strategy, "champion_ensemble")
        self.assertEqual(config.risk.max_gross_leverage, 6.0)
        self.assertEqual(config.risk.max_symbol_notional_pct, 0.80)
        self.assertEqual(config.risk.reduce_only_margin_level_pct, 500.0)
        self.assertEqual(config.risk.drawdown_derisk_start_pct, 0.04)
        self.assertEqual(config.risk.drawdown_derisk_full_pct, 0.10)
        self.assertEqual(config.risk.max_position_loss_for_symbol("EURUSD"), 0.01)
        self.assertEqual(config.risk.max_position_loss_for_symbol("XAUUSD"), 0.02)
        self.assertEqual(config.risk.max_position_loss_for_symbol("BTCUSD"), 0.025)
        self.assertEqual(config.macd_momentum.target_notional_usd, 800_000)
        self.assertEqual(config.macd_momentum.max_target_notional_usd, 800_000)
        self.assertEqual(config.macd_momentum.crypto_allowed_utc_hours, tuple(range(24)))
        self.assertEqual(config.multi_horizon_momentum.crypto_allowed_utc_hours, tuple(range(24)))
        self.assertEqual(config.quality_trend.crypto_allowed_utc_hours, tuple(range(24)))
        self.assertEqual(
            config.macd_conditional_fallback.conditional_notional_multiplier,
            0.25,
        )
        self.assertEqual(
            config.macd_squeeze_complement.squeeze_notional_multiplier,
            1.0,
        )
        self.assertEqual(
            config.macd_squeeze_complement.macd_inactive_reason_keywords,
            ("below",),
        )
        self.assertEqual(config.quality_trend.max_target_notional_usd, 800_000)

        strategy = config.build_strategy("macd_momentum", symbol="XAUUSD")

        self.assertEqual(strategy.config.symbol, "XAUUSD")
        self.assertEqual(strategy.config.max_spread_bps, 25.0)

        fallback = config.build_strategy("macd_conditional_fallback", symbol="XAUUSD")

        self.assertIsInstance(fallback, MacdConditionalFallbackStrategy)
        self.assertEqual(fallback.config.symbol, "XAUUSD")
        self.assertEqual(fallback.config.max_spread_bps, 25.0)

        complement = config.build_strategy("macd_squeeze_complement", symbol="XAUUSD")

        self.assertIsInstance(complement, MacdSqueezeComplementStrategy)
        self.assertEqual(complement.config.symbol, "XAUUSD")
        self.assertEqual(complement.config.max_spread_bps, 25.0)

    def test_config_rejects_naive_datetime(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.toml"
            path.write_text(
                """
[competition]
timezone = "Europe/London"
starting_equity = 1000000.0
open_at = "2026-06-21T22:00:00"
checkpoints = ["2026-06-22T22:00:00+01:00"]
protect_minutes_before = 90.0
protect_minutes_after = 5.0

[risk]
max_gross_leverage = 2.0
max_symbol_notional_pct = 0.25
max_daily_loss_pct = 0.025
max_drawdown_pct = 0.06
checkpoint_risk_multiplier = 0.5
min_margin_level_pct = 300.0

[strategy.simple_momentum]
symbol = "EURUSD"
lookback = 5
threshold_bps = 8.0
target_notional_usd = 50000.0

[execution]
route = "dry_run"
journal_path = "outputs/dry_run_journal.jsonl"
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "timezone"):
                load_config(path)
