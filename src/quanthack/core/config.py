from __future__ import annotations

import tomllib
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from quanthack.core.clock import CompetitionClock
from quanthack.core.instruments import instrument_for
from quanthack.market.market_quality import MarketQualityLimits
from quanthack.trading.risk import RiskLimits
from quanthack.strategies.strategy import (
    AlphaRouterConfig,
    AutocorrelationRegimeConfig,
    AssetAdaptiveDualSqueezeConfig,
    BreakoutConfig,
    ChampionEnsembleConfig,
    ConditionalSeasonalityConfig,
    CrossRateReversionConfig,
    DualSqueezeConfig,
    ExhaustionReversalConfig,
    FixingReversalConfig,
    IntradaySeasonalityConfig,
    KalmanTrendStrategyConfig,
    LiquiditySweepReversalConfig,
    MacdConditionalFallbackConfig,
    MacdMomentumConfig,
    MacdSqueezeComplementConfig,
    MeanReversionConfig,
    MomentumConfig,
    MultiHorizonMomentumConfig,
    MovingAverageCrossoverConfig,
    QualityTrendConfig,
    RangeExpansionTrendConfig,
    RegimeConfig,
    RelativeStrengthConfig,
    SessionBreakoutConfig,
    Strategy,
    StrategyConfig,
    UsdPressureConfig,
    TrendPullbackConfig,
    VolatilitySqueezeConfig,
    build_strategy,
    normalize_strategy_name,
)


@dataclass(frozen=True)
class CompetitionSettings:
    timezone: str
    starting_equity: float
    open_at: datetime
    checkpoints: tuple[datetime, ...]
    protect_minutes_before: float
    protect_minutes_after: float

    def to_clock(self) -> CompetitionClock:
        return CompetitionClock(
            open_at=self.open_at,
            checkpoints=self.checkpoints,
            protect_minutes_before=self.protect_minutes_before,
            protect_minutes_after=self.protect_minutes_after,
        )


@dataclass(frozen=True)
class ExecutionSettings:
    route: str
    journal_path: str


@dataclass(frozen=True)
class MarketDataSettings:
    price_csv: str
    quote_csv: str


@dataclass(frozen=True)
class BacktestSettings:
    price_csv: str
    quote_csv: str
    slippage_bps: float
    periods_per_year: float
    equity_curve_csv: str
    pnl_ledger_csv: str = "outputs/backtests/pnl_ledger.csv"


@dataclass(frozen=True)
class LiveDryRunConfig:
    adapter: str
    timeframe: str
    bars: int
    iterations: int
    poll_seconds: float
    journal_path: str
    monitor_csv: str

    def __post_init__(self) -> None:
        if self.adapter not in {"csv", "mt5"}:
            raise ValueError("live_dry_run adapter must be csv or mt5")
        if self.bars < 2:
            raise ValueError("live_dry_run bars must be at least 2")
        if self.iterations < 1:
            raise ValueError("live_dry_run iterations must be at least 1")
        if self.poll_seconds < 0:
            raise ValueError("live_dry_run poll_seconds cannot be negative")


@dataclass(frozen=True)
class SweepSettings:
    lookbacks: tuple[int, ...]
    threshold_bps: tuple[float, ...]
    train_fraction: float
    results_csv: str

    def __post_init__(self) -> None:
        if not self.lookbacks:
            raise ValueError("sweep lookbacks cannot be empty")
        if not self.threshold_bps:
            raise ValueError("sweep threshold_bps cannot be empty")
        if not 0 < self.train_fraction < 1:
            raise ValueError("sweep train_fraction must be between 0 and 1")


@dataclass(frozen=True)
class WalkForwardSettings:
    train_size: int
    test_size: int
    step_size: int
    min_total_fills: int
    min_profitable_fold_fraction: float
    max_worst_drawdown_pct: float
    cost_multipliers: tuple[float, ...]
    ma_fast_windows: tuple[int, ...]
    ma_slow_windows: tuple[int, ...]
    ma_min_separation_bps: tuple[float, ...]
    summary_csv: str
    folds_csv: str

    def __post_init__(self) -> None:
        if self.train_size < 2:
            raise ValueError("walk_forward train_size must be at least 2")
        if self.test_size < 1:
            raise ValueError("walk_forward test_size must be at least 1")
        if self.step_size < 1:
            raise ValueError("walk_forward step_size must be at least 1")
        if self.min_total_fills < 0:
            raise ValueError("walk_forward min_total_fills cannot be negative")
        if not 0 <= self.min_profitable_fold_fraction <= 1:
            raise ValueError(
                "walk_forward min_profitable_fold_fraction must be between 0 and 1"
            )
        if not 0 <= self.max_worst_drawdown_pct <= 1:
            raise ValueError("walk_forward max_worst_drawdown_pct must be between 0 and 1")
        if not self.cost_multipliers:
            raise ValueError("walk_forward cost_multipliers cannot be empty")
        if any(value <= 0 for value in self.cost_multipliers):
            raise ValueError("walk_forward cost_multipliers must be positive")
        if not self.ma_fast_windows:
            raise ValueError("walk_forward ma_fast_windows cannot be empty")
        if not self.ma_slow_windows:
            raise ValueError("walk_forward ma_slow_windows cannot be empty")
        if not self.ma_min_separation_bps:
            raise ValueError("walk_forward ma_min_separation_bps cannot be empty")
        if any(value < 2 for value in self.ma_fast_windows):
            raise ValueError("walk_forward ma_fast_windows must be at least 2")
        if any(value < 3 for value in self.ma_slow_windows):
            raise ValueError("walk_forward ma_slow_windows must be at least 3")
        if any(value <= 0 for value in self.ma_min_separation_bps):
            raise ValueError("walk_forward ma_min_separation_bps must be positive")
        if not any(
            fast_window < slow_window
            for fast_window in self.ma_fast_windows
            for slow_window in self.ma_slow_windows
        ):
            raise ValueError(
                "walk_forward moving-average grid needs at least one fast < slow pair"
            )


@dataclass(frozen=True)
class AppConfig:
    competition: CompetitionSettings
    risk: RiskLimits
    market_quality: MarketQualityLimits
    active_strategy: str
    simple_momentum: MomentumConfig
    session_momentum: MomentumConfig
    multi_horizon_momentum: MultiHorizonMomentumConfig
    autocorrelation_regime: AutocorrelationRegimeConfig
    intraday_seasonality: IntradaySeasonalityConfig
    conditional_seasonality: ConditionalSeasonalityConfig
    ma_crossover: MovingAverageCrossoverConfig
    macd_momentum: MacdMomentumConfig
    macd_conditional_fallback: MacdConditionalFallbackConfig
    macd_squeeze_complement: MacdSqueezeComplementConfig
    breakout: BreakoutConfig
    volatility_squeeze: VolatilitySqueezeConfig
    dual_squeeze: DualSqueezeConfig
    asset_adaptive_dual_squeeze: AssetAdaptiveDualSqueezeConfig
    range_expansion_trend: RangeExpansionTrendConfig
    session_breakout: SessionBreakoutConfig
    trend_pullback: TrendPullbackConfig
    exhaustion_reversal: ExhaustionReversalConfig
    liquidity_sweep_reversal: LiquiditySweepReversalConfig
    fixing_reversal: FixingReversalConfig
    kalman_trend: KalmanTrendStrategyConfig
    quality_trend: QualityTrendConfig
    champion_ensemble: ChampionEnsembleConfig
    mean_reversion: MeanReversionConfig
    regime_switch: RegimeConfig
    alpha_router: AlphaRouterConfig
    usd_pressure: UsdPressureConfig
    relative_strength: RelativeStrengthConfig
    cross_rate_reversion: CrossRateReversionConfig
    market_data: MarketDataSettings
    execution: ExecutionSettings
    backtest: BacktestSettings
    live_dry_run: LiveDryRunConfig
    sweep: SweepSettings
    walk_forward: WalkForwardSettings

    def validate_symbols(self) -> None:
        """Validate configured strategy symbols against the official registry."""
        from dataclasses import fields

        for field_def in fields(self):
            value = getattr(self, field_def.name)
            symbol = getattr(value, "symbol", None)
            if isinstance(symbol, str) and symbol:
                instrument_for(symbol)

    def strategy_config_for(self, name: str | None = None) -> StrategyConfig:
        strategy_name = normalize_strategy_name(name or self.active_strategy)
        if strategy_name == "simple_momentum":
            return self.simple_momentum
        if strategy_name == "session_momentum":
            return self.session_momentum
        if strategy_name == "multi_horizon_momentum":
            return self.multi_horizon_momentum
        if strategy_name == "autocorrelation_regime":
            return self.autocorrelation_regime
        if strategy_name == "intraday_seasonality":
            return self.intraday_seasonality
        if strategy_name == "conditional_seasonality":
            return self.conditional_seasonality
        if strategy_name == "ma_crossover":
            return self.ma_crossover
        if strategy_name == "macd_momentum":
            return self.macd_momentum
        if strategy_name == "asset_adaptive_macd":
            return self.macd_momentum
        if strategy_name == "macd_conditional_fallback":
            return self.macd_conditional_fallback
        if strategy_name == "macd_squeeze_complement":
            return self.macd_squeeze_complement
        if strategy_name == "breakout":
            return self.breakout
        if strategy_name == "volatility_squeeze":
            return self.volatility_squeeze
        if strategy_name == "dual_squeeze":
            return self.dual_squeeze
        if strategy_name == "asset_adaptive_dual_squeeze":
            return self.asset_adaptive_dual_squeeze
        if strategy_name == "range_expansion_trend":
            return self.range_expansion_trend
        if strategy_name == "session_breakout":
            return self.session_breakout
        if strategy_name == "trend_pullback":
            return self.trend_pullback
        if strategy_name == "exhaustion_reversal":
            return self.exhaustion_reversal
        if strategy_name == "liquidity_sweep_reversal":
            return self.liquidity_sweep_reversal
        if strategy_name == "fixing_reversal":
            return self.fixing_reversal
        if strategy_name == "kalman_trend":
            return self.kalman_trend
        if strategy_name == "quality_trend":
            return self.quality_trend
        if strategy_name == "champion_ensemble":
            return self.champion_ensemble
        if strategy_name == "mean_reversion":
            return self.mean_reversion
        if strategy_name == "crypto_mean_reversion":
            return self.mean_reversion
        if strategy_name == "regime_switch":
            return self.regime_switch
        if strategy_name == "alpha_router":
            return self.alpha_router
        if strategy_name == "usd_pressure_router":
            return self.usd_pressure
        if strategy_name == "relative_strength":
            return self.relative_strength
        if strategy_name == "cross_rate_reversion":
            return self.cross_rate_reversion
        raise ValueError(f"unsupported strategy {strategy_name!r}")

    def strategy_symbol(self, name: str | None = None) -> str:
        return self.strategy_config_for(name).symbol

    def strategy_lookback(self, name: str | None = None) -> int:
        return self.strategy_config_for(name).lookback

    def build_strategy(self, name: str | None = None, *, symbol: str | None = None) -> Strategy:
        if symbol is not None:
            return build_strategy(
                name or self.active_strategy,
                simple_momentum=_with_symbol_and_spread(self.simple_momentum, symbol),
                session_momentum=_with_symbol_and_spread(
                    self.session_momentum,
                    symbol,
                ),
                multi_horizon_momentum=_with_symbol_and_spread(
                    self.multi_horizon_momentum,
                    symbol,
                ),
                autocorrelation_regime=_with_symbol_and_spread(
                    self.autocorrelation_regime,
                    symbol,
                ),
                intraday_seasonality=_with_symbol_and_spread(
                    self.intraday_seasonality,
                    symbol,
                ),
                conditional_seasonality=_with_symbol_and_spread(
                    self.conditional_seasonality,
                    symbol,
                ),
                ma_crossover=_with_symbol_and_spread(self.ma_crossover, symbol),
                macd_momentum=_with_symbol_and_spread(self.macd_momentum, symbol),
                macd_conditional_fallback=_with_symbol_and_spread(
                    self.macd_conditional_fallback,
                    symbol,
                ),
                macd_squeeze_complement=_with_symbol_and_spread(
                    self.macd_squeeze_complement,
                    symbol,
                ),
                breakout=_with_symbol_and_spread(self.breakout, symbol),
                volatility_squeeze=_with_symbol_and_spread(
                    self.volatility_squeeze,
                    symbol,
                ),
                dual_squeeze=_with_symbol_and_spread(self.dual_squeeze, symbol),
                asset_adaptive_dual_squeeze=_with_symbol_and_spread(
                    self.asset_adaptive_dual_squeeze,
                    symbol,
                ),
                range_expansion_trend=_with_symbol_and_spread(
                    self.range_expansion_trend,
                    symbol,
                ),
                session_breakout=_with_symbol_and_spread(self.session_breakout, symbol),
                trend_pullback=_with_symbol_and_spread(self.trend_pullback, symbol),
                exhaustion_reversal=_with_symbol_and_spread(
                    self.exhaustion_reversal,
                    symbol,
                ),
                liquidity_sweep_reversal=_with_symbol_and_spread(
                    self.liquidity_sweep_reversal,
                    symbol,
                ),
                fixing_reversal=_with_symbol_and_spread(self.fixing_reversal, symbol),
                kalman_trend=_with_symbol_and_spread(self.kalman_trend, symbol),
                quality_trend=_with_symbol_and_spread(self.quality_trend, symbol),
                champion_ensemble=_with_symbol_and_spread(
                    self.champion_ensemble,
                    symbol,
                ),
                mean_reversion=_with_symbol_and_spread(self.mean_reversion, symbol),
                regime_switch=_with_symbol_and_spread(self.regime_switch, symbol),
                alpha_router=_with_symbol_and_spread(self.alpha_router, symbol),
                usd_pressure=_with_symbol_and_spread(self.usd_pressure, symbol),
                relative_strength=_with_symbol_and_spread(self.relative_strength, symbol),
                cross_rate_reversion=_with_symbol_and_spread(
                    self.cross_rate_reversion,
                    symbol,
                ),
            )

        return build_strategy(
            name or self.active_strategy,
            simple_momentum=self.simple_momentum,
            session_momentum=self.session_momentum,
            multi_horizon_momentum=self.multi_horizon_momentum,
            autocorrelation_regime=self.autocorrelation_regime,
            intraday_seasonality=self.intraday_seasonality,
            conditional_seasonality=self.conditional_seasonality,
            ma_crossover=self.ma_crossover,
            macd_momentum=self.macd_momentum,
            macd_conditional_fallback=self.macd_conditional_fallback,
            macd_squeeze_complement=self.macd_squeeze_complement,
            breakout=self.breakout,
            volatility_squeeze=self.volatility_squeeze,
            dual_squeeze=self.dual_squeeze,
            asset_adaptive_dual_squeeze=self.asset_adaptive_dual_squeeze,
            range_expansion_trend=self.range_expansion_trend,
            session_breakout=self.session_breakout,
            trend_pullback=self.trend_pullback,
            exhaustion_reversal=self.exhaustion_reversal,
            liquidity_sweep_reversal=self.liquidity_sweep_reversal,
            fixing_reversal=self.fixing_reversal,
            kalman_trend=self.kalman_trend,
            quality_trend=self.quality_trend,
            champion_ensemble=self.champion_ensemble,
            mean_reversion=self.mean_reversion,
            regime_switch=self.regime_switch,
            alpha_router=self.alpha_router,
            usd_pressure=self.usd_pressure,
            relative_strength=self.relative_strength,
            cross_rate_reversion=self.cross_rate_reversion,
            symbol=symbol,
        )


def load_config(path: str | Path = "configs/default.toml") -> AppConfig:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    competition = raw["competition"]
    strategy = raw["strategy"]
    timezone = competition["timezone"]
    ZoneInfo(timezone)

    walk_forward = raw.get("walk_forward", {})
    live_dry_run = raw.get("live_dry_run", {})

    return AppConfig(
        competition=CompetitionSettings(
            timezone=timezone,
            starting_equity=float(competition["starting_equity"]),
            open_at=_parse_datetime(competition["open_at"]),
            checkpoints=tuple(_parse_datetime(value) for value in competition["checkpoints"]),
            protect_minutes_before=float(competition["protect_minutes_before"]),
            protect_minutes_after=float(competition["protect_minutes_after"]),
        ),
        risk=RiskLimits(**raw["risk"]),
        market_quality=MarketQualityLimits(**raw["market_quality"]),
        active_strategy=normalize_strategy_name(str(strategy.get("active", "simple_momentum"))),
        simple_momentum=MomentumConfig(**strategy["simple_momentum"]),
        session_momentum=MomentumConfig(**strategy.get("session_momentum", {})),
        multi_horizon_momentum=MultiHorizonMomentumConfig(
            **strategy.get("multi_horizon_momentum", {})
        ),
        autocorrelation_regime=AutocorrelationRegimeConfig(
            **strategy.get("autocorrelation_regime", {})
        ),
        intraday_seasonality=IntradaySeasonalityConfig(
            **strategy.get("intraday_seasonality", {})
        ),
        conditional_seasonality=ConditionalSeasonalityConfig(
            **strategy.get("conditional_seasonality", {})
        ),
        ma_crossover=MovingAverageCrossoverConfig(**strategy.get("ma_crossover", {})),
        macd_momentum=MacdMomentumConfig(**strategy.get("macd_momentum", {})),
        macd_conditional_fallback=MacdConditionalFallbackConfig(
            **strategy.get("macd_conditional_fallback", {})
        ),
        macd_squeeze_complement=MacdSqueezeComplementConfig(
            **strategy.get("macd_squeeze_complement", {})
        ),
        breakout=BreakoutConfig(**strategy.get("breakout", {})),
        volatility_squeeze=VolatilitySqueezeConfig(
            **strategy.get("volatility_squeeze", {})
        ),
        dual_squeeze=DualSqueezeConfig(**strategy.get("dual_squeeze", {})),
        asset_adaptive_dual_squeeze=AssetAdaptiveDualSqueezeConfig(
            **strategy.get("asset_adaptive_dual_squeeze", {})
        ),
        range_expansion_trend=RangeExpansionTrendConfig(
            **strategy.get("range_expansion_trend", {})
        ),
        session_breakout=SessionBreakoutConfig(**strategy.get("session_breakout", {})),
        trend_pullback=TrendPullbackConfig(**strategy.get("trend_pullback", {})),
        exhaustion_reversal=ExhaustionReversalConfig(
            **strategy.get("exhaustion_reversal", {})
        ),
        liquidity_sweep_reversal=LiquiditySweepReversalConfig(
            **strategy.get("liquidity_sweep_reversal", {})
        ),
        fixing_reversal=FixingReversalConfig(
            **strategy.get("fixing_reversal", {})
        ),
        kalman_trend=KalmanTrendStrategyConfig(**strategy.get("kalman_trend", {})),
        quality_trend=QualityTrendConfig(**strategy.get("quality_trend", {})),
        champion_ensemble=ChampionEnsembleConfig(
            **strategy.get("champion_ensemble", {})
        ),
        mean_reversion=MeanReversionConfig(**strategy.get("mean_reversion", {})),
        regime_switch=RegimeConfig(**strategy.get("regime_switch", {})),
        alpha_router=AlphaRouterConfig(**strategy.get("alpha_router", {})),
        usd_pressure=UsdPressureConfig(**strategy.get("usd_pressure", {})),
        relative_strength=RelativeStrengthConfig(
            **strategy.get("relative_strength", {})
        ),
        cross_rate_reversion=CrossRateReversionConfig(
            **strategy.get("cross_rate_reversion", {})
        ),
        market_data=MarketDataSettings(**raw["market_data"]),
        execution=ExecutionSettings(**raw["execution"]),
        backtest=BacktestSettings(**raw["backtest"]),
        live_dry_run=LiveDryRunConfig(
            adapter=str(live_dry_run.get("adapter", "csv")),
            timeframe=str(live_dry_run.get("timeframe", "M1")),
            bars=int(live_dry_run.get("bars", 120)),
            iterations=int(live_dry_run.get("iterations", 1)),
            poll_seconds=float(live_dry_run.get("poll_seconds", 0.0)),
            journal_path=str(
                live_dry_run.get(
                    "journal_path",
                    "outputs/live_dry_run_journal.jsonl",
                )
            ),
            monitor_csv=str(
                live_dry_run.get(
                    "monitor_csv",
                    "outputs/live_competition_monitor.csv",
                )
            ),
        ),
        sweep=SweepSettings(
            lookbacks=tuple(int(value) for value in raw["sweep"]["lookbacks"]),
            threshold_bps=tuple(float(value) for value in raw["sweep"]["threshold_bps"]),
            train_fraction=float(raw["sweep"]["train_fraction"]),
            results_csv=str(raw["sweep"]["results_csv"]),
        ),
        walk_forward=WalkForwardSettings(
            train_size=int(walk_forward.get("train_size", 10)),
            test_size=int(walk_forward.get("test_size", 5)),
            step_size=int(walk_forward.get("step_size", 5)),
            min_total_fills=int(walk_forward.get("min_total_fills", 1)),
            min_profitable_fold_fraction=float(
                walk_forward.get("min_profitable_fold_fraction", 0.0)
            ),
            max_worst_drawdown_pct=float(walk_forward.get("max_worst_drawdown_pct", 1.0)),
            cost_multipliers=tuple(
                float(value) for value in walk_forward.get("cost_multipliers", [1.5, 2.0])
            ),
            ma_fast_windows=tuple(
                int(value) for value in walk_forward.get("ma_fast_windows", [2, 3])
            ),
            ma_slow_windows=tuple(
                int(value) for value in walk_forward.get("ma_slow_windows", [5, 8])
            ),
            ma_min_separation_bps=tuple(
                float(value)
                for value in walk_forward.get("ma_min_separation_bps", [1.0, 2.0])
            ),
            summary_csv=str(
                walk_forward.get(
                    "summary_csv",
                    "outputs/backtests/walk_forward_summary.csv",
                )
            ),
            folds_csv=str(
                walk_forward.get(
                    "folds_csv",
                    "outputs/backtests/walk_forward_folds.csv",
                )
            ),
        ),
    )


def _parse_datetime(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(value)

    if parsed.tzinfo is None:
        raise ValueError("configured datetimes must include timezone offsets")
    return parsed


def _with_symbol_and_spread(config: StrategyConfig, symbol: str) -> StrategyConfig:
    instrument = instrument_for(symbol)
    return replace(
        config,
        symbol=instrument.symbol,
        max_spread_bps=instrument.max_spread_bps,
    )
