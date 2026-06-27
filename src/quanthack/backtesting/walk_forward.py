from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import median

from quanthack.backtesting.backtest import BacktestEngine, BacktestResult, FillModel
from quanthack.core.config import AppConfig
from quanthack.market.market_data import PriceBar, PriceHistory, QuoteHistory
from quanthack.strategies.strategy import (
    AlphaRouterConfig,
    AlphaRouterStrategy,
    BreakoutConfig,
    BreakoutStrategy,
    MeanReversionConfig,
    MeanReversionStrategy,
    MomentumConfig,
    MovingAverageCrossoverConfig,
    MovingAverageCrossoverStrategy,
    RegimeConfig,
    RegimeSwitchingStrategy,
    SessionBreakoutConfig,
    SessionBreakoutStrategy,
    SimpleMomentumStrategy,
    UsdPressureConfig,
    UsdPressureRouterStrategy,
    normalize_strategy_name,
)


@dataclass(frozen=True)
class WalkForwardFold:
    fold_index: int
    strategy_name: str
    symbol: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    selected_parameters: str
    train: BacktestResult
    test: BacktestResult
    stressed_test_returns: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class WalkForwardSummary:
    strategy_name: str
    folds: tuple[WalkForwardFold, ...]
    median_test_sharpe: float
    lower_quartile_test_return: float
    worst_test_drawdown: float
    profitable_fold_fraction: float
    total_test_fills: int
    median_test_fills: float
    total_test_turnover: float
    cost_stress_return_1_5x: float | None
    cost_stress_return_2_0x: float | None
    eligible: bool

    @property
    def rank_key(self) -> tuple[int, float, float, float, float, float]:
        return (
            1 if self.eligible else 0,
            self.median_test_sharpe,
            self.lower_quartile_test_return,
            -self.worst_test_drawdown,
            self.profitable_fold_fraction,
            -self.total_test_turnover,
        )


@dataclass(frozen=True)
class WalkForwardResult:
    summaries: tuple[WalkForwardSummary, ...]

    @property
    def best(self) -> WalkForwardSummary | None:
        if not self.summaries:
            return None
        return self.summaries[0]


def run_walk_forward(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    strategy_names: tuple[str, ...],
    symbol: str,
    train_size: int,
    test_size: int,
    step_size: int,
    momentum_lookbacks: tuple[int, ...],
    momentum_threshold_bps: tuple[float, ...],
    ma_fast_windows: tuple[int, ...] = (2, 3),
    ma_slow_windows: tuple[int, ...] = (5, 8),
    ma_min_separation_bps: tuple[float, ...] = (1.0, 2.0),
    cost_multipliers: tuple[float, ...] = (1.5, 2.0),
    min_total_fills: int = 1,
    min_profitable_fold_fraction: float = 0.0,
    max_worst_drawdown_pct: float = 1.0,
) -> WalkForwardResult:
    _validate_walk_forward_inputs(
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
        min_total_fills=min_total_fills,
        min_profitable_fold_fraction=min_profitable_fold_fraction,
        max_worst_drawdown_pct=max_worst_drawdown_pct,
    )
    _validate_ma_crossover_grid(
        ma_fast_windows=ma_fast_windows,
        ma_slow_windows=ma_slow_windows,
        ma_min_separation_bps=ma_min_separation_bps,
    )
    normalized_names = _normalize_unique(strategy_names)
    bars = prices.for_symbol(symbol).bars
    if len(bars) < train_size + test_size:
        raise ValueError("not enough bars for one walk-forward fold")

    summaries: list[WalkForwardSummary] = []
    for strategy_name in normalized_names:
        folds = tuple(
            _run_fold(
                config=config,
                all_prices=prices,
                all_quotes=quotes,
                bars=bars,
                symbol=symbol,
                strategy_name=strategy_name,
                fold_index=fold_index,
                train_start=start,
                train_size=train_size,
                test_size=test_size,
                momentum_lookbacks=momentum_lookbacks,
                momentum_threshold_bps=momentum_threshold_bps,
                ma_fast_windows=ma_fast_windows,
                ma_slow_windows=ma_slow_windows,
                ma_min_separation_bps=ma_min_separation_bps,
                cost_multipliers=cost_multipliers,
            )
            for fold_index, start in enumerate(
                range(0, len(bars) - train_size - test_size + 1, step_size),
                start=1,
            )
        )
        summaries.append(
            _summarize_strategy(
                strategy_name=strategy_name,
                folds=folds,
                min_total_fills=min_total_fills,
                min_profitable_fold_fraction=min_profitable_fold_fraction,
                max_worst_drawdown_pct=max_worst_drawdown_pct,
            )
        )

    summaries.sort(key=lambda summary: summary.rank_key, reverse=True)
    return WalkForwardResult(tuple(summaries))


def write_walk_forward_summary_csv(result: WalkForwardResult, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "strategy",
                "eligible",
                "folds",
                "median_test_sharpe",
                "lower_quartile_test_return",
                "worst_test_drawdown",
                "profitable_fold_fraction",
                "total_test_fills",
                "median_test_fills",
                "total_test_turnover",
                "cost_stress_return_1_5x",
                "cost_stress_return_2_0x",
            ],
        )
        writer.writeheader()
        for rank, summary in enumerate(result.summaries, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "strategy": summary.strategy_name,
                    "eligible": summary.eligible,
                    "folds": len(summary.folds),
                    "median_test_sharpe": summary.median_test_sharpe,
                    "lower_quartile_test_return": summary.lower_quartile_test_return,
                    "worst_test_drawdown": summary.worst_test_drawdown,
                    "profitable_fold_fraction": summary.profitable_fold_fraction,
                    "total_test_fills": summary.total_test_fills,
                    "median_test_fills": summary.median_test_fills,
                    "total_test_turnover": summary.total_test_turnover,
                    "cost_stress_return_1_5x": summary.cost_stress_return_1_5x,
                    "cost_stress_return_2_0x": summary.cost_stress_return_2_0x,
                }
            )


def write_walk_forward_folds_csv(result: WalkForwardResult, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "strategy",
                "fold",
                "symbol",
                "train_start",
                "train_end",
                "test_start",
                "test_end",
                "selected_parameters",
                "train_return",
                "train_sharpe",
                "train_drawdown",
                "train_fills",
                "test_return",
                "test_sharpe",
                "test_drawdown",
                "test_fills",
                "test_turnover",
            ],
        )
        writer.writeheader()
        for summary in result.summaries:
            for fold in summary.folds:
                writer.writerow(
                    {
                        "strategy": fold.strategy_name,
                        "fold": fold.fold_index,
                        "symbol": fold.symbol,
                        "train_start": fold.train_start,
                        "train_end": fold.train_end,
                        "test_start": fold.test_start,
                        "test_end": fold.test_end,
                        "selected_parameters": fold.selected_parameters,
                        "train_return": fold.train.metrics.total_return_pct,
                        "train_sharpe": fold.train.metrics.sharpe_ratio,
                        "train_drawdown": fold.train.metrics.max_drawdown_pct,
                        "train_fills": len(fold.train.fills),
                        "test_return": fold.test.metrics.total_return_pct,
                        "test_sharpe": fold.test.metrics.sharpe_ratio,
                        "test_drawdown": fold.test.metrics.max_drawdown_pct,
                        "test_fills": len(fold.test.fills),
                        "test_turnover": fold.test.metrics.turnover_notional,
                    }
                )


def _run_fold(
    *,
    config: AppConfig,
    all_prices: PriceHistory,
    all_quotes: QuoteHistory,
    bars: tuple[PriceBar, ...],
    symbol: str,
    strategy_name: str,
    fold_index: int,
    train_start: int,
    train_size: int,
    test_size: int,
    momentum_lookbacks: tuple[int, ...],
    momentum_threshold_bps: tuple[float, ...],
    ma_fast_windows: tuple[int, ...],
    ma_slow_windows: tuple[int, ...],
    ma_min_separation_bps: tuple[float, ...],
    cost_multipliers: tuple[float, ...],
) -> WalkForwardFold:
    train_end = train_start + train_size
    test_end = train_end + test_size
    train_prices = PriceHistory(bars[train_start:train_end])
    test_prices = PriceHistory(bars[train_end:test_end])
    train_quotes = _quotes_for_bars(all_quotes, symbol=symbol, bars=train_prices.bars)
    test_quotes = _quotes_for_bars(all_quotes, symbol=symbol, bars=test_prices.bars)

    strategy_config, selected_parameters, train = _select_strategy_on_train(
        config=config,
        strategy_name=strategy_name,
        symbol=symbol,
        train_prices=train_prices,
        train_quotes=train_quotes,
        momentum_lookbacks=momentum_lookbacks,
        momentum_threshold_bps=momentum_threshold_bps,
        ma_fast_windows=ma_fast_windows,
        ma_slow_windows=ma_slow_windows,
        ma_min_separation_bps=ma_min_separation_bps,
    )
    test = _run_strategy_config(
        config=config,
        strategy_name=strategy_name,
        strategy_config=strategy_config,
        prices=test_prices,
        quotes=test_quotes,
        symbol=symbol,
        fill_model=FillModel(slippage_bps=config.backtest.slippage_bps),
    )
    stressed_returns = tuple(
        (
            multiplier,
            _run_strategy_config(
                config=config,
                strategy_name=strategy_name,
                strategy_config=_stress_strategy_costs(strategy_config, multiplier),
                prices=test_prices,
                quotes=test_quotes,
                symbol=symbol,
                fill_model=FillModel(slippage_bps=config.backtest.slippage_bps * multiplier),
            ).metrics.total_return_pct,
        )
        for multiplier in cost_multipliers
    )

    return WalkForwardFold(
        fold_index=fold_index,
        strategy_name=strategy_name,
        symbol=symbol,
        train_start=train_prices.bars[0].timestamp.isoformat(timespec="seconds"),
        train_end=train_prices.bars[-1].timestamp.isoformat(timespec="seconds"),
        test_start=test_prices.bars[0].timestamp.isoformat(timespec="seconds"),
        test_end=test_prices.bars[-1].timestamp.isoformat(timespec="seconds"),
        selected_parameters=selected_parameters,
        train=train,
        test=test,
        stressed_test_returns=stressed_returns,
    )


def _select_strategy_on_train(
    *,
    config: AppConfig,
    strategy_name: str,
    symbol: str,
    train_prices: PriceHistory,
    train_quotes: QuoteHistory,
    momentum_lookbacks: tuple[int, ...],
    momentum_threshold_bps: tuple[float, ...],
    ma_fast_windows: tuple[int, ...],
    ma_slow_windows: tuple[int, ...],
    ma_min_separation_bps: tuple[float, ...],
) -> tuple[
    MomentumConfig
    | MovingAverageCrossoverConfig
    | BreakoutConfig
    | SessionBreakoutConfig
    | MeanReversionConfig
    | RegimeConfig
    | AlphaRouterConfig
    | UsdPressureConfig,
    str,
    BacktestResult,
]:
    if strategy_name == "simple_momentum":
        candidates: list[tuple[tuple[float, float, float], MomentumConfig, BacktestResult]] = []
        for lookback in momentum_lookbacks:
            for threshold in momentum_threshold_bps:
                if len(train_prices.bars) < lookback:
                    continue
                strategy_config = replace(
                    config.simple_momentum,
                    symbol=symbol,
                    lookback=lookback,
                    threshold_bps=threshold,
                    exit_threshold_bps=min(config.simple_momentum.exit_threshold_bps, threshold),
                )
                train = _run_strategy_config(
                    config=config,
                    strategy_name=strategy_name,
                    strategy_config=strategy_config,
                    prices=train_prices,
                    quotes=train_quotes,
                    symbol=symbol,
                    fill_model=FillModel(slippage_bps=config.backtest.slippage_bps),
                )
                candidates.append(
                    (
                        (
                            train.metrics.sharpe_ratio,
                            train.metrics.total_return_pct,
                            -train.metrics.max_drawdown_pct,
                        ),
                        strategy_config,
                        train,
                    )
                )
        if not candidates:
            raise ValueError("no valid momentum candidates for walk-forward training")
        candidates.sort(key=lambda item: item[0], reverse=True)
        selected_config = candidates[0][1]
        return (
            selected_config,
            f"lookback={selected_config.lookback};threshold_bps={selected_config.threshold_bps}",
            candidates[0][2],
        )

    if strategy_name == "ma_crossover":
        candidates: list[
            tuple[tuple[float, float, float], MovingAverageCrossoverConfig, BacktestResult]
        ] = []
        for fast_window in ma_fast_windows:
            for slow_window in ma_slow_windows:
                if fast_window >= slow_window or len(train_prices.bars) < slow_window:
                    continue
                for min_separation_bps in ma_min_separation_bps:
                    exit_separation_bps = min(
                        config.ma_crossover.exit_separation_bps,
                        min_separation_bps * 0.5,
                    )
                    strategy_config = replace(
                        config.ma_crossover,
                        symbol=symbol,
                        fast_window=fast_window,
                        slow_window=slow_window,
                        min_separation_bps=min_separation_bps,
                        exit_separation_bps=exit_separation_bps,
                    )
                    train = _run_strategy_config(
                        config=config,
                        strategy_name=strategy_name,
                        strategy_config=strategy_config,
                        prices=train_prices,
                        quotes=train_quotes,
                        symbol=symbol,
                        fill_model=FillModel(slippage_bps=config.backtest.slippage_bps),
                    )
                    candidates.append(
                        (
                            (
                                train.metrics.sharpe_ratio,
                                train.metrics.total_return_pct,
                                -train.metrics.max_drawdown_pct,
                            ),
                            strategy_config,
                            train,
                        )
                    )
        if not candidates:
            raise ValueError(
                "no valid moving-average crossover candidates for walk-forward training"
            )
        candidates.sort(key=lambda item: item[0], reverse=True)
        selected_config = candidates[0][1]
        return (
            selected_config,
            (
                f"fast_window={selected_config.fast_window};"
                f"slow_window={selected_config.slow_window};"
                f"min_separation_bps={selected_config.min_separation_bps};"
                f"exit_separation_bps={selected_config.exit_separation_bps}"
            ),
            candidates[0][2],
        )

    if strategy_name == "breakout":
        strategy_config = replace(config.breakout, symbol=symbol)
        train = _run_strategy_config(
            config=config,
            strategy_name=strategy_name,
            strategy_config=strategy_config,
            prices=train_prices,
            quotes=train_quotes,
            symbol=symbol,
            fill_model=FillModel(slippage_bps=config.backtest.slippage_bps),
        )
        return (
            strategy_config,
            (
                f"lookback={strategy_config.lookback};"
                f"breakout_buffer_bps={strategy_config.breakout_buffer_bps}"
            ),
            train,
        )

    if strategy_name == "session_breakout":
        strategy_config = replace(config.session_breakout, symbol=symbol)
        train = _run_strategy_config(
            config=config,
            strategy_name=strategy_name,
            strategy_config=strategy_config,
            prices=train_prices,
            quotes=train_quotes,
            symbol=symbol,
            fill_model=FillModel(slippage_bps=config.backtest.slippage_bps),
        )
        return (
            strategy_config,
            (
                f"lookback={strategy_config.lookback};"
                f"breakout_buffer_bps={strategy_config.breakout_buffer_bps};"
                f"utc_hours={','.join(str(hour) for hour in strategy_config.allowed_utc_hours)};"
                f"min_vol_bps={strategy_config.min_realized_volatility_bps}"
            ),
            train,
        )

    if strategy_name == "mean_reversion":
        strategy_config = replace(config.mean_reversion, symbol=symbol)
        train = _run_strategy_config(
            config=config,
            strategy_name=strategy_name,
            strategy_config=strategy_config,
            prices=train_prices,
            quotes=train_quotes,
            symbol=symbol,
            fill_model=FillModel(slippage_bps=config.backtest.slippage_bps),
        )
        return (
            strategy_config,
            (
                f"lookback={strategy_config.lookback};"
                f"entry_zscore={strategy_config.entry_zscore};"
                f"exit_zscore={strategy_config.exit_zscore}"
            ),
            train,
        )

    if strategy_name == "regime_switch":
        strategy_config = replace(config.regime_switch, symbol=symbol)
        train = _run_strategy_config(
            config=config,
            strategy_name=strategy_name,
            strategy_config=strategy_config,
            prices=train_prices,
            quotes=train_quotes,
            symbol=symbol,
            fill_model=FillModel(slippage_bps=config.backtest.slippage_bps),
        )
        return (
            strategy_config,
            (
                f"lookback={strategy_config.lookback};"
                f"hysteresis_bars={strategy_config.hysteresis_bars}"
            ),
            train,
        )

    if strategy_name == "alpha_router":
        strategy_config = replace(config.alpha_router, symbol=symbol)
        train = _run_strategy_config(
            config=config,
            strategy_name=strategy_name,
            strategy_config=strategy_config,
            prices=train_prices,
            quotes=train_quotes,
            symbol=symbol,
            fill_model=FillModel(slippage_bps=config.backtest.slippage_bps),
        )
        return (
            strategy_config,
            (
                f"entry_score={strategy_config.entry_score};"
                f"momentum_weight={strategy_config.momentum_weight};"
                f"mean_reversion_weight={strategy_config.mean_reversion_weight}"
            ),
            train,
        )

    if strategy_name == "usd_pressure_router":
        strategy_config = replace(config.usd_pressure, symbol=symbol)
        train = _run_strategy_config(
            config=config,
            strategy_name=strategy_name,
            strategy_config=strategy_config,
            prices=train_prices,
            quotes=train_quotes,
            symbol=symbol,
            fill_model=FillModel(slippage_bps=config.backtest.slippage_bps),
        )
        return (
            strategy_config,
            (
                f"lookback={strategy_config.lookback};"
                f"pressure_threshold_bps={strategy_config.pressure_threshold_bps};"
                f"min_components={strategy_config.min_component_symbols}"
            ),
            train,
        )

    raise ValueError(f"unsupported strategy {strategy_name!r}")


def _run_strategy_config(
    *,
    config: AppConfig,
    strategy_name: str,
    strategy_config: (
        MomentumConfig
        | MovingAverageCrossoverConfig
        | BreakoutConfig
        | SessionBreakoutConfig
        | MeanReversionConfig
        | RegimeConfig
        | AlphaRouterConfig
        | UsdPressureConfig
    ),
    prices: PriceHistory,
    quotes: QuoteHistory,
    symbol: str,
    fill_model: FillModel,
) -> BacktestResult:
    if strategy_name == "simple_momentum":
        strategy = SimpleMomentumStrategy(strategy_config)  # type: ignore[arg-type]
    elif strategy_name == "ma_crossover":
        strategy = MovingAverageCrossoverStrategy(strategy_config)  # type: ignore[arg-type]
    elif strategy_name == "breakout":
        strategy = BreakoutStrategy(strategy_config)  # type: ignore[arg-type]
    elif strategy_name == "session_breakout":
        strategy = SessionBreakoutStrategy(strategy_config)  # type: ignore[arg-type]
    elif strategy_name == "mean_reversion":
        strategy = MeanReversionStrategy(strategy_config)  # type: ignore[arg-type]
    elif strategy_name == "regime_switch":
        strategy = RegimeSwitchingStrategy(
            config=strategy_config,  # type: ignore[arg-type]
            momentum=config.simple_momentum,
            mean_reversion=config.mean_reversion,
        )
    elif strategy_name == "alpha_router":
        strategy = AlphaRouterStrategy(
            config=strategy_config,  # type: ignore[arg-type]
            momentum=config.simple_momentum,
            moving_average=config.ma_crossover,
            breakout=config.breakout,
            mean_reversion=config.mean_reversion,
        )
    elif strategy_name == "usd_pressure_router":
        strategy = UsdPressureRouterStrategy(
            config=strategy_config,  # type: ignore[arg-type]
            base_strategy=AlphaRouterStrategy(
                config=replace(config.alpha_router, symbol=symbol),
                momentum=config.simple_momentum,
                moving_average=config.ma_crossover,
                breakout=config.breakout,
                mean_reversion=config.mean_reversion,
            ),
        )
    else:
        raise ValueError(f"unsupported strategy {strategy_name!r}")

    engine = BacktestEngine(
        strategy=strategy,
        risk_limits=config.risk,
        quality_limits=config.market_quality,
        clock=config.competition.to_clock(),
        fill_model=fill_model,
        periods_per_year=config.backtest.periods_per_year,
    )
    return engine.run(
        prices=prices,
        quotes=quotes,
        symbol=symbol,
        starting_equity=config.competition.starting_equity,
    )


def _summarize_strategy(
    *,
    strategy_name: str,
    folds: tuple[WalkForwardFold, ...],
    min_total_fills: int,
    min_profitable_fold_fraction: float,
    max_worst_drawdown_pct: float,
) -> WalkForwardSummary:
    test_sharpes = [fold.test.metrics.sharpe_ratio for fold in folds]
    test_returns = [fold.test.metrics.total_return_pct for fold in folds]
    test_drawdowns = [fold.test.metrics.max_drawdown_pct for fold in folds]
    test_fills = [len(fold.test.fills) for fold in folds]
    total_turnover = sum(fold.test.metrics.turnover_notional for fold in folds)
    profitable_fraction = (
        sum(1 for value in test_returns if value > 0) / len(test_returns)
        if test_returns
        else 0.0
    )
    total_fills = sum(test_fills)
    worst_drawdown = max(test_drawdowns, default=0.0)
    stress_1_5x = _stress_return(folds, 1.5)
    stress_2_0x = _stress_return(folds, 2.0)
    eligible = (
        total_fills >= min_total_fills
        and profitable_fraction >= min_profitable_fold_fraction
        and worst_drawdown <= max_worst_drawdown_pct
    )
    return WalkForwardSummary(
        strategy_name=strategy_name,
        folds=folds,
        median_test_sharpe=median(test_sharpes) if test_sharpes else 0.0,
        lower_quartile_test_return=_lower_quartile(test_returns),
        worst_test_drawdown=worst_drawdown,
        profitable_fold_fraction=profitable_fraction,
        total_test_fills=total_fills,
        median_test_fills=median(test_fills) if test_fills else 0.0,
        total_test_turnover=total_turnover,
        cost_stress_return_1_5x=stress_1_5x,
        cost_stress_return_2_0x=stress_2_0x,
        eligible=eligible,
    )


def _quotes_for_bars(
    quotes: QuoteHistory,
    *,
    symbol: str,
    bars: tuple[PriceBar, ...],
) -> QuoteHistory:
    timestamps = {bar.timestamp for bar in bars}
    return QuoteHistory(
        tuple(
            quote
            for quote in quotes.for_symbol(symbol).quotes
            if quote.timestamp in timestamps
        )
    )


def _stress_strategy_costs(
    strategy_config: (
        MomentumConfig
        | MovingAverageCrossoverConfig
        | BreakoutConfig
        | SessionBreakoutConfig
        | MeanReversionConfig
        | RegimeConfig
        | AlphaRouterConfig
        | UsdPressureConfig
    ),
    multiplier: float,
) -> (
    MomentumConfig
    | MovingAverageCrossoverConfig
    | BreakoutConfig
    | SessionBreakoutConfig
    | MeanReversionConfig
    | RegimeConfig
    | AlphaRouterConfig
    | UsdPressureConfig
):
    if isinstance(strategy_config, RegimeConfig | AlphaRouterConfig | UsdPressureConfig):
        return strategy_config
    return replace(
        strategy_config,
        slippage_bps=strategy_config.slippage_bps * multiplier,
        fee_bps=strategy_config.fee_bps * multiplier,
    )


def _stress_return(folds: tuple[WalkForwardFold, ...], multiplier: float) -> float | None:
    returns = [
        value
        for fold in folds
        for stress_multiplier, value in fold.stressed_test_returns
        if stress_multiplier == multiplier
    ]
    if not returns:
        return None
    return sum(returns)


def _lower_quartile(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = max(0, int(len(sorted_values) * 0.25) - 1)
    return sorted_values[index]


def _normalize_unique(strategy_names: tuple[str, ...]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for name in strategy_names:
        strategy_name = normalize_strategy_name(name)
        if strategy_name not in seen:
            normalized.append(strategy_name)
            seen.add(strategy_name)
    return tuple(normalized)


def _validate_walk_forward_inputs(
    *,
    train_size: int,
    test_size: int,
    step_size: int,
    min_total_fills: int,
    min_profitable_fold_fraction: float,
    max_worst_drawdown_pct: float,
) -> None:
    if train_size < 2:
        raise ValueError("train_size must be at least 2")
    if test_size < 1:
        raise ValueError("test_size must be at least 1")
    if step_size < 1:
        raise ValueError("step_size must be at least 1")
    if min_total_fills < 0:
        raise ValueError("min_total_fills cannot be negative")
    if not 0 <= min_profitable_fold_fraction <= 1:
        raise ValueError("min_profitable_fold_fraction must be between 0 and 1")
    if not 0 <= max_worst_drawdown_pct <= 1:
        raise ValueError("max_worst_drawdown_pct must be between 0 and 1")


def _validate_ma_crossover_grid(
    *,
    ma_fast_windows: tuple[int, ...],
    ma_slow_windows: tuple[int, ...],
    ma_min_separation_bps: tuple[float, ...],
) -> None:
    if not ma_fast_windows:
        raise ValueError("ma_fast_windows cannot be empty")
    if not ma_slow_windows:
        raise ValueError("ma_slow_windows cannot be empty")
    if not ma_min_separation_bps:
        raise ValueError("ma_min_separation_bps cannot be empty")
    if any(value < 2 for value in ma_fast_windows):
        raise ValueError("ma_fast_windows must be at least 2")
    if any(value < 3 for value in ma_slow_windows):
        raise ValueError("ma_slow_windows must be at least 3")
    if any(value <= 0 for value in ma_min_separation_bps):
        raise ValueError("ma_min_separation_bps must be positive")
    if not any(
        fast_window < slow_window
        for fast_window in ma_fast_windows
        for slow_window in ma_slow_windows
    ):
        raise ValueError("moving-average grid needs at least one fast < slow pair")
