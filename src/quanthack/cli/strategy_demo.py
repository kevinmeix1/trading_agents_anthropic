from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.strategies.strategy import (
    AlphaRouterConfig,
    AlphaRouterStrategy,
    AssetAdaptiveDualSqueezeConfig,
    AssetAdaptiveDualSqueezeStrategy,
    BreakoutConfig,
    BreakoutStrategy,
    ChampionEnsembleConfig,
    ChampionEnsembleStrategy,
    DualSqueezeConfig,
    DualSqueezeStrategy,
    ExhaustionReversalConfig,
    ExhaustionReversalStrategy,
    FixingReversalConfig,
    FixingReversalStrategy,
    KalmanTrendStrategy,
    KalmanTrendStrategyConfig,
    MeanReversionConfig,
    MeanReversionStrategy,
    MomentumConfig,
    MovingAverageCrossoverConfig,
    MovingAverageCrossoverStrategy,
    RangeExpansionTrendConfig,
    RangeExpansionTrendStrategy,
    RegimeConfig,
    RegimeSwitchingStrategy,
    RelativeStrengthConfig,
    RelativeStrengthStrategy,
    STRATEGY_NAMES,
    SessionBreakoutConfig,
    SessionBreakoutStrategy,
    SimpleMomentumStrategy,
    TrendPullbackConfig,
    TrendPullbackStrategy,
    UsdPressureConfig,
    UsdPressureRouterStrategy,
    VolatilitySqueezeConfig,
    VolatilitySqueezeStrategy,
)


SCENARIOS = {
    "up": [1.1000, 1.1002, 1.1004, 1.1007, 1.1010],
    "down": [1.1000, 1.0998, 1.0996, 1.0993, 1.0990],
    "flat": [1.1000, 1.1001, 1.1000, 1.1001, 1.1000],
    "spike_up": [1.1000, 1.1001, 1.1000, 1.1001, 1.1030],
    "spike_down": [1.1000, 1.0999, 1.1000, 1.0999, 1.0970],
}

NO_TRADE_ALREADY_EXPLAINED = object()


def parse_prices(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Show what a strategy proposes.")
    parser.add_argument("--strategy", choices=STRATEGY_NAMES, default="simple_momentum")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="up")
    parser.add_argument("--prices", type=parse_prices, default=None)
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--lookback", type=int, default=5)
    parser.add_argument("--fast-window", type=int, default=3)
    parser.add_argument("--slow-window", type=int, default=None)
    parser.add_argument("--threshold-bps", type=float, default=8.0)
    parser.add_argument("--entry-zscore", type=float, default=1.0)
    parser.add_argument("--target-notional", type=float, default=50_000)
    return parser


def run(args: argparse.Namespace) -> None:
    prices = args.prices or SCENARIOS[args.scenario]
    print(f"Strategy: {args.strategy}")
    print(f"Prices: {prices}")

    if args.strategy == "simple_momentum":
        request = _print_momentum(args, prices)
    elif args.strategy == "ma_crossover":
        request = _print_ma_crossover(args, prices)
    elif args.strategy == "breakout":
        request = _print_breakout(args, prices)
    elif args.strategy == "session_breakout":
        request = _print_session_breakout(args, prices)
    elif args.strategy == "volatility_squeeze":
        request = _print_volatility_squeeze(args, prices)
    elif args.strategy == "dual_squeeze":
        request = _print_dual_squeeze(args, prices)
    elif args.strategy == "asset_adaptive_dual_squeeze":
        request = _print_asset_adaptive_dual_squeeze(args, prices)
    elif args.strategy == "range_expansion_trend":
        request = _print_range_expansion_trend(args, prices)
    elif args.strategy == "trend_pullback":
        request = _print_trend_pullback(args, prices)
    elif args.strategy == "exhaustion_reversal":
        request = _print_exhaustion_reversal(args, prices)
    elif args.strategy == "fixing_reversal":
        request = _print_fixing_reversal(args, prices)
    elif args.strategy == "kalman_trend":
        request = _print_kalman_trend(args, prices)
    elif args.strategy == "champion_ensemble":
        request = _print_champion_ensemble(args, prices)
    elif args.strategy == "mean_reversion":
        request = _print_mean_reversion(args, prices)
    elif args.strategy == "regime_switch":
        request = _print_regime_switch(args, prices)
    elif args.strategy == "alpha_router":
        request = _print_alpha_router(args, prices)
    elif args.strategy == "relative_strength":
        request = _print_relative_strength(args, prices)
    elif args.strategy == "usd_pressure_router":
        request = _print_usd_pressure_router(args, prices)
    else:
        request = _print_cross_rate_reversion(args)

    if request is NO_TRADE_ALREADY_EXPLAINED:
        return

    if request is None:
        print("Strategy output: NO TRADE")
        print("Reason: signal did not cross the entry threshold")
        return

    print(f"Strategy output: {request.side.value} {request.symbol}")
    print(f"Target notional: ${request.target_notional_usd:,.0f}")
    print(f"Reason: {request.reason}")


def _print_momentum(args: argparse.Namespace, prices: list[float]):
    config = MomentumConfig(
        symbol=args.symbol,
        lookback=args.lookback,
        threshold_bps=args.threshold_bps,
        target_notional_usd=args.target_notional,
    )
    strategy = SimpleMomentumStrategy(config)
    reading = strategy.read_momentum(prices)
    request = strategy.generate_request(prices)

    if reading is None:
        print("Momentum: not enough prices")
        print("Strategy output: NO TRADE")
        return NO_TRADE_ALREADY_EXPLAINED

    print(f"Momentum: {reading.move_bps:.1f} bps")
    print(f"Trend efficiency: {reading.trend_efficiency:.2f}")
    print(f"Realized volatility: {reading.realized_volatility:.6f}")
    print(f"Normalized momentum: {reading.normalized_momentum:.2f}")
    print(f"Threshold: {config.threshold_bps:.1f} bps")
    return request


def _ma_windows(args: argparse.Namespace, prices: list[float]) -> tuple[int, int]:
    slow_window = args.slow_window or min(max(3, args.lookback), len(prices))
    slow_window = max(3, min(slow_window, len(prices)))
    fast_window = max(2, min(args.fast_window, slow_window - 1))
    return fast_window, slow_window


def _print_ma_crossover(args: argparse.Namespace, prices: list[float]):
    fast_window, slow_window = _ma_windows(args, prices)
    config = MovingAverageCrossoverConfig(
        symbol=args.symbol,
        fast_window=fast_window,
        slow_window=slow_window,
        min_separation_bps=min(args.threshold_bps, 2.0),
        target_notional_usd=args.target_notional,
    )
    strategy = MovingAverageCrossoverStrategy(config)
    reading = strategy.read_crossover(prices)
    request = strategy.generate_request(prices)

    if reading is None:
        print("Moving average crossover: not enough prices")
        print("Strategy output: NO TRADE")
        return NO_TRADE_ALREADY_EXPLAINED

    previous = (
        "n/a"
        if reading.previous_separation_bps is None
        else f"{reading.previous_separation_bps:.1f} bps"
    )
    print(f"Fast average ({config.fast_window}): {reading.fast_average:.5f}")
    print(f"Slow average ({config.slow_window}): {reading.slow_average:.5f}")
    print(f"Separation: {reading.separation_bps:.1f} bps")
    print(f"Previous separation: {previous}")
    print(f"Crossed direction: {reading.crossed_direction.value}")
    print(f"Trend efficiency: {reading.trend_efficiency:.2f}")
    print(f"Entry separation: {config.min_separation_bps:.1f} bps")
    return request


def _print_breakout(args: argparse.Namespace, prices: list[float]):
    config = BreakoutConfig(
        symbol=args.symbol,
        lookback=args.lookback,
        target_notional_usd=args.target_notional,
    )
    strategy = BreakoutStrategy(config)
    reading = strategy.read_breakout(prices)
    request = strategy.generate_request(prices)

    if reading is None:
        print("Breakout: not enough prices")
        print("Strategy output: NO TRADE")
        return NO_TRADE_ALREADY_EXPLAINED

    print(f"Upper band: {reading.upper_band:.5f}")
    print(f"Lower band: {reading.lower_band:.5f}")
    print(f"Last price: {reading.last_price:.5f}")
    print(f"Channel width: {reading.channel_width_bps:.1f} bps")
    print(f"Breakout: {reading.breakout_bps:.1f} bps")
    print(f"Position in channel: {reading.position_in_channel:.2f}")
    return request


def _print_session_breakout(args: argparse.Namespace, prices: list[float]):
    config = SessionBreakoutConfig(
        symbol=args.symbol,
        lookback=args.lookback,
        target_notional_usd=args.target_notional,
    )
    strategy = SessionBreakoutStrategy(config)
    reading = strategy.read_session_breakout(prices)
    request = strategy.generate_request(prices)

    if reading is None:
        print("Session breakout: not enough prices")
        print("Strategy output: NO TRADE")
        return NO_TRADE_ALREADY_EXPLAINED

    breakout = reading.breakout
    print(f"Upper band: {breakout.upper_band:.5f}")
    print(f"Lower band: {breakout.lower_band:.5f}")
    print(f"Last price: {breakout.last_price:.5f}")
    print(f"Channel width: {breakout.channel_width_bps:.1f} bps")
    print(f"Breakout: {breakout.breakout_bps:.1f} bps")
    print(f"Realized volatility: {reading.realized_volatility_bps:.1f} bps")
    print(f"Allowed UTC hours: {','.join(str(hour) for hour in config.allowed_utc_hours)}")
    return request


def _print_volatility_squeeze(args: argparse.Namespace, prices: list[float]):
    lookback = max(6, min(args.lookback, len(prices)))
    squeeze_window = max(2, min(3, lookback - 4))
    config = VolatilitySqueezeConfig(
        symbol=args.symbol,
        lookback=lookback,
        squeeze_window=squeeze_window,
        breakout_buffer_bps=min(args.threshold_bps, 2.0),
        min_prior_volatility_bps=0.1,
        min_band_width_bps=0.0,
        target_notional_usd=args.target_notional,
    )
    strategy = VolatilitySqueezeStrategy(config)
    reading = strategy.read_squeeze(prices)
    request = strategy.generate_request(prices)

    if reading is None:
        print("Volatility squeeze: not enough prices")
        print("Strategy output: NO TRADE")
        return NO_TRADE_ALREADY_EXPLAINED

    print(f"Mean price: {reading.mean_price:.5f}")
    print(f"Upper band: {reading.upper_band:.5f}")
    print(f"Lower band: {reading.lower_band:.5f}")
    print(f"Last price: {reading.last_price:.5f}")
    print(f"Band width: {reading.band_width_bps:.1f} bps")
    print(f"Breakout: {reading.breakout_bps:.1f} bps")
    print(f"Recent volatility: {reading.recent_volatility_bps:.1f} bps")
    print(f"Prior volatility: {reading.prior_volatility_bps:.1f} bps")
    print(f"Squeeze ratio: {reading.squeeze_ratio:.2f}")
    return request


def _print_range_expansion_trend(args: argparse.Namespace, prices: list[float]):
    lookback = max(8, min(args.lookback, len(prices)))
    trigger_window = max(2, min(3, lookback - 5))
    config = RangeExpansionTrendConfig(
        symbol=args.symbol,
        lookback=lookback,
        trigger_window=trigger_window,
        min_trigger_move_bps=min(args.threshold_bps, 4.0),
        min_range_break_bps=1.0,
        min_expansion_zscore=0.5,
        max_expansion_zscore=100.0,
        min_trend_efficiency=0.50,
        min_baseline_volatility_bps=0.01,
        min_expected_edge_bps=1.0,
        target_notional_usd=args.target_notional,
    )
    strategy = RangeExpansionTrendStrategy(config)
    reading = strategy.read_range_expansion(prices)
    request = strategy.generate_request(prices)

    if reading is None:
        print("Range expansion trend: not enough prices")
        print("Strategy output: NO TRADE")
        return NO_TRADE_ALREADY_EXPLAINED

    print(f"Baseline high: {reading.baseline_high:.5f}")
    print(f"Baseline low: {reading.baseline_low:.5f}")
    print(f"Trigger move: {reading.trigger_move_bps:.1f} bps")
    print(f"Range break: {reading.range_break_bps:.1f} bps")
    print(f"Expansion z-score: {reading.expansion_zscore:.2f}")
    print(f"Trend efficiency: {reading.trend_efficiency:.2f}")
    print(f"Expected edge: {reading.expected_edge_bps:.1f} bps")
    print(f"Signal direction: {reading.signal_direction.value}")
    return request


def _print_dual_squeeze(args: argparse.Namespace, prices: list[float]):
    lookback = max(10, min(args.lookback, len(prices)))
    confirmation_lookback = max(lookback, min(len(prices), max(12, lookback + 4)))
    squeeze_window = max(2, min(3, lookback - 4))
    confirmation_window = max(2, min(4, confirmation_lookback - 4))
    config = DualSqueezeConfig(
        symbol=args.symbol,
        lookback=lookback,
        squeeze_window=squeeze_window,
        confirmation_lookback=confirmation_lookback,
        confirmation_squeeze_window=confirmation_window,
        breakout_buffer_bps=min(args.threshold_bps, 2.5),
        target_notional_usd=args.target_notional,
    )
    strategy = DualSqueezeStrategy(config)
    reading = strategy.read_dual_squeeze(prices)
    request = strategy.generate_request(prices)

    if reading is None:
        print("Dual squeeze: not enough prices")
        print("Strategy output: NO TRADE")
        return NO_TRADE_ALREADY_EXPLAINED

    print(f"Fast breakout: {reading.fast.breakout_bps:.1f} bps")
    print(f"Fast squeeze ratio: {reading.fast.squeeze_ratio:.2f}")
    print(f"Confirmation breakout: {reading.confirmation.breakout_bps:.1f} bps")
    print(f"Confirmation squeeze ratio: {reading.confirmation.squeeze_ratio:.2f}")
    print(f"Confirmation passed: {'yes' if reading.confirmation_passed else 'no'}")
    print(f"Confirmation reason: {reading.confirmation_reason}")
    return request


def _print_asset_adaptive_dual_squeeze(args: argparse.Namespace, prices: list[float]):
    config = AssetAdaptiveDualSqueezeConfig(
        symbol=args.symbol,
        target_notional_usd=args.target_notional,
    )
    strategy = AssetAdaptiveDualSqueezeStrategy(config)
    decision = strategy.generate_decision(prices)
    inner = strategy.inner_config

    print(f"Selected profile: {strategy.selected_profile}")
    print(f"Fast lookback: {inner.lookback}")
    print(f"Fast squeeze window: {inner.squeeze_window}")
    print(f"Confirmation lookback: {inner.confirmation_lookback}")
    print(f"Confirmation squeeze window: {inner.confirmation_squeeze_window}")
    print(f"Decision: {decision.action.value}")
    print(f"Reason: {decision.reason}")
    return decision.to_trade_request()


def _print_trend_pullback(args: argparse.Namespace, prices: list[float]):
    lookback = max(8, min(args.lookback, len(prices)))
    pullback_window = max(2, min(2, lookback - 4))
    config = TrendPullbackConfig(
        symbol=args.symbol,
        lookback=lookback,
        pullback_window=pullback_window,
        min_trend_bps=min(args.threshold_bps, 8.0),
        max_pullback_bps=20.0,
        min_expected_edge_bps=2.0,
        target_notional_usd=args.target_notional,
    )
    strategy = TrendPullbackStrategy(config)
    reading = strategy.read_pullback(prices)
    request = strategy.generate_request(prices)

    if reading is None:
        print("Trend pullback: not enough prices")
        print("Strategy output: NO TRADE")
        return NO_TRADE_ALREADY_EXPLAINED

    print(f"Anchor price: {reading.anchor_price:.5f}")
    print(f"Previous price: {reading.previous_price:.5f}")
    print(f"Last price: {reading.last_price:.5f}")
    print(f"Trend move: {reading.trend_move_bps:.1f} bps")
    print(f"Pullback: {reading.pullback_bps:.1f} bps")
    print(f"Resume move: {reading.resume_bps:.1f} bps")
    print(f"Expected edge: {reading.expected_edge_bps:.1f} bps")
    print(f"Trend efficiency: {reading.trend_efficiency:.2f}")
    print(f"Signal direction: {reading.signal_direction.value}")
    return request


def _print_exhaustion_reversal(args: argparse.Namespace, prices: list[float]):
    lookback = max(8, min(args.lookback, len(prices)))
    shock_window = max(2, min(3, lookback - 5))
    config = ExhaustionReversalConfig(
        symbol=args.symbol,
        lookback=lookback,
        shock_window=shock_window,
        min_shock_bps=min(args.threshold_bps, 12.0),
        min_shock_zscore=0.50,
        min_shock_efficiency=0.40,
        min_expected_edge_bps=2.0,
        target_notional_usd=args.target_notional,
    )
    strategy = ExhaustionReversalStrategy(config)
    reading = strategy.read_exhaustion(prices)
    request = strategy.generate_request(prices)

    if reading is None:
        print("Exhaustion reversal: not enough prices")
        print("Strategy output: NO TRADE")
        return NO_TRADE_ALREADY_EXPLAINED

    print(f"Shock start price: {reading.shock_start_price:.5f}")
    print(f"Previous price: {reading.previous_price:.5f}")
    print(f"Last price: {reading.last_price:.5f}")
    print(f"Shock move: {reading.shock_move_bps:.1f} bps")
    print(f"Reversal move: {reading.reversal_bps:.1f} bps")
    print(f"Shock z-score: {reading.shock_zscore:.2f}")
    print(f"Shock efficiency: {reading.shock_efficiency:.2f}")
    print(f"Baseline volatility: {reading.baseline_volatility_bps:.1f} bps")
    print(f"Expected edge: {reading.expected_edge_bps:.1f} bps")
    print(f"Signal direction: {reading.signal_direction.value}")
    return request


def _print_fixing_reversal(args: argparse.Namespace, prices: list[float]):
    pre_fix_lookback = max(2, min(args.lookback, max(2, len(prices) - 2)))
    config = FixingReversalConfig(
        symbol=args.symbol,
        pre_fix_lookback=pre_fix_lookback,
        min_pre_fix_move_bps=min(args.threshold_bps, 4.0),
        min_pre_fix_efficiency=0.20,
        min_expected_edge_bps=2.0,
        min_realized_volatility_bps=0.0,
        target_notional_usd=args.target_notional,
    )
    strategy = FixingReversalStrategy(config)
    reading = strategy.read_fixing_reversal(prices)
    request = strategy.generate_request(prices)

    if reading is None:
        print("Fixing reversal: not enough prices")
        print("Strategy output: NO TRADE")
        return NO_TRADE_ALREADY_EXPLAINED

    print(f"Anchor price: {reading.anchor_price:.5f}")
    print(f"Previous price: {reading.previous_price:.5f}")
    print(f"Last price: {reading.last_price:.5f}")
    print(f"Pre-fix move: {reading.pre_fix_move_bps:.1f} bps")
    print(f"Confirmation move: {reading.confirmation_bps:.1f} bps")
    print(f"Pre-fix efficiency: {reading.pre_fix_efficiency:.2f}")
    print(f"Realized volatility: {reading.realized_volatility_bps:.1f} bps")
    print(f"Expected edge: {reading.expected_edge_bps:.1f} bps")
    print(f"Signal direction: {reading.signal_direction.value}")
    return request


def _print_kalman_trend(args: argparse.Namespace, prices: list[float]):
    lookback = max(5, min(args.lookback, len(prices)))
    config = KalmanTrendStrategyConfig(
        symbol=args.symbol,
        lookback=lookback,
        min_abs_slope_bps=min(args.threshold_bps, 0.75),
        min_trend_efficiency=0.10,
        min_expected_edge_bps=1.0,
        target_notional_usd=args.target_notional,
    )
    strategy = KalmanTrendStrategy(config)
    reading = strategy.read_kalman_trend(prices)
    decision = strategy.generate_decision(prices)

    if reading is None:
        print("Kalman trend: not enough prices")
        print("Strategy output: NO TRADE")
        return NO_TRADE_ALREADY_EXPLAINED

    regime = reading.regime_reading
    print(f"Regime: {regime.regime.value}")
    print(f"Kalman level: {regime.kalman_level:.5f}")
    print(f"Slope: {regime.kalman_slope_bps:.2f} bps")
    print(f"Trend efficiency: {regime.trend_efficiency:.2f}")
    print(f"Realized volatility: {regime.realized_volatility_bps:.2f} bps")
    print(f"Trend confidence: {regime.trend_confidence:.2f}")
    print(f"Expected edge: {reading.expected_edge_bps:.1f} bps")
    print(f"Signal direction: {reading.signal_direction.value}")
    print(f"Decision: {decision.action.value}")
    print(f"Reason: {decision.reason}")
    return decision.to_trade_request()


def _print_champion_ensemble(args: argparse.Namespace, prices: list[float]):
    lookback = max(5, min(args.lookback, len(prices)))
    config = ChampionEnsembleConfig(
        symbol=args.symbol,
        lookback=lookback,
        target_notional_usd=args.target_notional,
    )
    strategy = ChampionEnsembleStrategy(
        config=config,
        kalman_trend=KalmanTrendStrategyConfig(
            symbol=args.symbol,
            lookback=lookback,
            min_abs_slope_bps=min(args.threshold_bps, 0.75),
            min_trend_efficiency=0.10,
            min_expected_edge_bps=1.0,
            target_notional_usd=args.target_notional,
        ),
        asset_adaptive_dual_squeeze=AssetAdaptiveDualSqueezeConfig(
            symbol=args.symbol,
            base_lookback=6,
            base_squeeze_window=2,
            base_confirmation_lookback=6,
            base_confirmation_squeeze_window=2,
            target_notional_usd=args.target_notional,
        ),
        dual_squeeze=DualSqueezeConfig(
            symbol=args.symbol,
            lookback=6,
            squeeze_window=2,
            confirmation_lookback=6,
            confirmation_squeeze_window=2,
            target_notional_usd=args.target_notional,
        ),
        trend_pullback=TrendPullbackConfig(
            symbol=args.symbol,
            lookback=lookback,
            pullback_window=2,
            target_notional_usd=args.target_notional,
        ),
    )
    signals = strategy.generate_signals(prices)
    decision = strategy.generate_decision(prices)

    for signal in signals:
        print(
            f"{signal.strategy_name}: {signal.direction.value} "
            f"weight={signal.weight:.2f} score={signal.signed_score:.2f}"
        )
    print(f"Decision: {decision.action.value}")
    print(f"Reason: {decision.reason}")
    return decision.to_trade_request()


def _print_mean_reversion(args: argparse.Namespace, prices: list[float]):
    config = MeanReversionConfig(
        symbol=args.symbol,
        lookback=args.lookback,
        entry_zscore=args.entry_zscore,
        target_notional_usd=args.target_notional,
    )
    strategy = MeanReversionStrategy(config)
    reading = strategy.read_reversion(prices)
    request = strategy.generate_request(prices)

    if reading is None:
        print("Mean reversion: not enough prices")
        print("Strategy output: NO TRADE")
        return NO_TRADE_ALREADY_EXPLAINED

    print(f"Mean price: {reading.mean_price:.5f}")
    print(f"Last price: {reading.last_price:.5f}")
    print(f"Residual: {reading.residual:.5f}")
    print(f"Z-score: {reading.zscore:.2f}")
    print(f"Deviation: {reading.deviation_bps:.1f} bps")
    print(f"Trend strength: {reading.trend_strength_bps:.1f} bps")
    print(f"Entry z-score: {config.entry_zscore:.2f}")
    return request


def _print_regime_switch(args: argparse.Namespace, prices: list[float]):
    config = RegimeConfig(
        symbol=args.symbol,
        lookback=args.lookback,
        hysteresis_bars=1,
    )
    strategy = RegimeSwitchingStrategy(
        config=config,
        momentum=MomentumConfig(
            symbol=args.symbol,
            lookback=min(args.lookback, len(prices)),
            threshold_bps=args.threshold_bps,
            target_notional_usd=args.target_notional,
        ),
        mean_reversion=MeanReversionConfig(
            symbol=args.symbol,
            lookback=min(args.lookback, len(prices)),
            entry_zscore=args.entry_zscore,
            target_notional_usd=args.target_notional,
        ),
    )
    reading = strategy.read_regime(prices)
    request = strategy.generate_request(prices)

    if reading is None:
        print("Regime: not enough prices")
        print("Strategy output: NO TRADE")
        return NO_TRADE_ALREADY_EXPLAINED

    print(f"Candidate regime: {reading.candidate.value}")
    print(f"Reason: {reading.reason}")
    print(f"Momentum move: {reading.momentum_move_bps:.1f} bps")
    print(f"Momentum efficiency: {reading.momentum_efficiency:.2f}")
    print(f"Mean-reversion z-score: {reading.reversion_zscore:.2f}")
    print(f"Mean-reversion trend: {reading.reversion_trend_bps:.1f} bps")
    return request


def _print_alpha_router(args: argparse.Namespace, prices: list[float]):
    config = AlphaRouterConfig(
        symbol=args.symbol,
        target_notional_usd=args.target_notional,
        ml_enabled=True,
        ml_lookback=max(3, min(args.lookback, max(3, len(prices) - 2))),
        ml_min_train_samples=1,
        ml_min_samples_for_trade=1,
        ml_label_threshold_bps=0.1,
    )
    fast_window, slow_window = _ma_windows(args, prices)
    strategy = AlphaRouterStrategy(
        config=config,
        momentum=MomentumConfig(
            symbol=args.symbol,
            lookback=min(args.lookback, len(prices)),
            threshold_bps=args.threshold_bps,
            target_notional_usd=args.target_notional,
        ),
        moving_average=MovingAverageCrossoverConfig(
            symbol=args.symbol,
            fast_window=fast_window,
            slow_window=slow_window,
            min_separation_bps=min(args.threshold_bps, 2.0),
            target_notional_usd=args.target_notional,
        ),
        breakout=BreakoutConfig(
            symbol=args.symbol,
            lookback=min(args.lookback, len(prices)),
            target_notional_usd=args.target_notional,
        ),
        mean_reversion=MeanReversionConfig(
            symbol=args.symbol,
            lookback=min(args.lookback, len(prices)),
            entry_zscore=args.entry_zscore,
            target_notional_usd=args.target_notional,
        ),
    )
    signals = strategy.generate_signals(prices)
    decision = strategy.generate_decision(prices)
    request = decision.to_trade_request()

    print("Signals:")
    for signal in signals:
        print(
            f"  {signal.strategy_name}: {signal.direction.value} "
            f"confidence={signal.confidence:.2f} "
            f"edge={signal.expected_edge_bps:.1f} bps "
            f"cost={signal.cost_bps:.1f} bps"
        )
        print(f"    Reason: {signal.reason}")
    print(f"Router decision: {decision.action.value}")
    print(f"Router reason: {decision.reason}")
    return request


def _print_usd_pressure_router(args: argparse.Namespace, prices: list[float]):
    config = UsdPressureConfig(
        symbol=args.symbol,
        lookback=max(3, min(args.lookback, len(prices))),
        min_component_symbols=1,
        min_confirming_symbols=1,
    )
    fast_window, slow_window = _ma_windows(args, prices)
    strategy = UsdPressureRouterStrategy(
        config=config,
        base_strategy=AlphaRouterStrategy(
            config=AlphaRouterConfig(symbol=args.symbol, target_notional_usd=args.target_notional),
            momentum=MomentumConfig(
                symbol=args.symbol,
                lookback=min(args.lookback, len(prices)),
                threshold_bps=args.threshold_bps,
                target_notional_usd=args.target_notional,
            ),
            moving_average=MovingAverageCrossoverConfig(
                symbol=args.symbol,
                fast_window=fast_window,
                slow_window=slow_window,
                min_separation_bps=min(args.threshold_bps, 2.0),
                target_notional_usd=args.target_notional,
            ),
            breakout=BreakoutConfig(
                symbol=args.symbol,
                lookback=min(args.lookback, len(prices)),
                target_notional_usd=args.target_notional,
            ),
            mean_reversion=MeanReversionConfig(
                symbol=args.symbol,
                lookback=min(args.lookback, len(prices)),
                entry_zscore=args.entry_zscore,
                target_notional_usd=args.target_notional,
            ),
        ),
    )
    strategy.update_portfolio_context(
        closes_by_symbol={
            args.symbol: prices,
            "GBPUSD": [1.3000, 1.3003, 1.3007, 1.3010, 1.3014],
            "AUDUSD": [0.6600, 0.6602, 0.6605, 0.6608, 0.6610],
            "USDJPY": [156.00, 155.95, 155.90, 155.86, 155.82],
        }
    )
    reading = strategy.read_usd_pressure()
    decision = strategy.generate_decision(prices)
    request = decision.to_trade_request()

    if reading is None:
        print("USD pressure: not enough basket context")
    else:
        print(f"USD pressure: {reading.pressure_bps:.1f} bps")
        print(f"Components: {reading.component_count}")
        print(f"Confirming symbols: {reading.confirming_symbols}")
    print(f"Router decision: {decision.action.value}")
    print(f"Router reason: {decision.reason}")
    return request


def _print_relative_strength(args: argparse.Namespace, prices: list[float]):
    config = RelativeStrengthConfig(
        symbol=args.symbol,
        lookback=max(3, min(args.lookback, len(prices))),
        target_notional_usd=args.target_notional,
        min_component_symbols=4,
        entry_zscore=0.75,
    )
    strategy = RelativeStrengthStrategy(config)
    strategy.update_portfolio_context(
        closes_by_symbol={
            args.symbol: prices,
            "GBPUSD": [1.3000, 1.3003, 1.3007, 1.3010, 1.3014],
            "AUDUSD": [0.6600, 0.6602, 0.6605, 0.6608, 0.6610],
            "USDJPY": [156.00, 155.95, 155.90, 155.86, 155.82],
        }
    )
    reading = strategy.read_relative_strength(prices)
    decision = strategy.generate_decision(prices)
    request = decision.to_trade_request()

    if reading is None:
        print("Relative strength: not enough basket context")
    else:
        print(f"Relative z-score: {reading.relative_zscore:.2f}")
        print(f"Rank: {reading.target_rank}/{reading.component_count}")
        print(f"Target score: {reading.target_score:.2f}")
        print(f"Move: {reading.move_bps:.1f} bps")
        print(f"Strongest: {reading.strongest_symbol}")
        print(f"Weakest: {reading.weakest_symbol}")
    print(f"Relative-strength decision: {decision.action.value}")
    print(f"Relative-strength reason: {decision.reason}")
    return request


def _print_cross_rate_reversion(args: argparse.Namespace):
    print("FX cross-rate reversion requires multi-symbol portfolio context.")
    print("Use portfolio-backtest or signal-diagnostics for this strategy.")
    print("Strategy output: NO TRADE")
    return NO_TRADE_ALREADY_EXPLAINED


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
