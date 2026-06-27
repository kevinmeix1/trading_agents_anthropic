from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path

from quanthack.backtesting.backtest import FillModel
from quanthack.backtesting.competition_score import (
    CompetitionMetrics,
    RiskDisciplineReport,
    build_competition_metrics,
    build_risk_discipline_report,
    risk_samples_from_portfolio_equity,
)
from quanthack.backtesting.portfolio_backtest import (
    PortfolioBacktestEngine,
    PortfolioBacktestResult,
)
from quanthack.backtesting.portfolio_fixed_warmup_walk_forward import (
    FixedWarmupPortfolioWalkForwardResult,
    run_fixed_warmup_portfolio_walk_forward,
)
from quanthack.core.config import AppConfig
from quanthack.core.instruments import instrument_for
from quanthack.market.market_data import PriceHistory, QuoteHistory
from quanthack.strategies.strategy import normalize_strategy_name


@dataclass(frozen=True)
class SizingFrontierPoint:
    symbol_notional_pct: float
    max_gross_leverage: float
    result: PortfolioBacktestResult
    competition_metrics: CompetitionMetrics
    risk_discipline: RiskDisciplineReport
    worst_leverage: float
    worst_net_directional_exposure: float
    worst_largest_symbol_concentration: float
    walk_forward: FixedWarmupPortfolioWalkForwardResult | None = None

    @property
    def rank_key(self) -> tuple[float, float, float, float, int]:
        wf = self.walk_forward
        wf_score = 0.0 if wf is None else wf.non_negative_fold_fraction
        return (
            wf_score,
            self.competition_metrics.return_pct,
            self.competition_metrics.sharpe_15m,
            -self.competition_metrics.max_drawdown_pct,
            self.risk_discipline.score,
        )


@dataclass(frozen=True)
class SizingFrontierResult:
    strategy_by_symbol: tuple[tuple[str, str], ...]
    points: tuple[SizingFrontierPoint, ...]

    @property
    def best_full_sample(self) -> SizingFrontierPoint | None:
        if not self.points:
            return None
        return max(
            self.points,
            key=lambda point: (
                point.risk_discipline.score,
                point.competition_metrics.return_pct,
                point.competition_metrics.sharpe_15m,
                -point.competition_metrics.max_drawdown_pct,
            ),
        )

    @property
    def best_walk_forward(self) -> SizingFrontierPoint | None:
        points = [point for point in self.points if point.walk_forward is not None]
        if not points:
            return None
        return max(points, key=lambda point: point.rank_key)


def evaluate_sizing_frontier(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    strategy_by_symbol: dict[str, str],
    symbol_notional_pcts: tuple[float, ...],
    max_gross_leverage: float | None = None,
    include_walk_forward: bool = False,
    train_size: int = 480,
    test_size: int = 96,
    step_size: int = 96,
) -> SizingFrontierResult:
    normalized_map = _normalize_strategy_map(strategy_by_symbol)
    _validate_symbol_notional_pcts(symbol_notional_pcts)
    leverage_cap = (
        config.risk.max_gross_leverage
        if max_gross_leverage is None
        else max_gross_leverage
    )
    if leverage_cap <= 0:
        raise ValueError("max_gross_leverage must be positive")

    points: list[SizingFrontierPoint] = []
    for symbol_notional_pct in symbol_notional_pcts:
        point_config = replace(
            config,
            risk=replace(
                config.risk,
                max_gross_leverage=leverage_cap,
                max_symbol_notional_pct=symbol_notional_pct,
            ),
        )
        result = _run_strategy_map(
            config=point_config,
            prices=prices,
            quotes=quotes,
            strategy_by_symbol=normalized_map,
        )
        metrics = build_competition_metrics(
            equity_points=result.equity_curve,
            fills=result.fills,
        )
        risk_samples = risk_samples_from_portfolio_equity(result.equity_curve)
        risk_discipline = build_risk_discipline_report(risk_samples)
        walk_forward = (
            run_fixed_warmup_portfolio_walk_forward(
                config=point_config,
                prices=prices,
                quotes=quotes,
                strategy_name=next(iter(normalized_map.values())),
                symbols=tuple(normalized_map),
                strategy_by_symbol=normalized_map,
                train_size=train_size,
                test_size=test_size,
                step_size=step_size,
            )
            if include_walk_forward
            else None
        )
        points.append(
            SizingFrontierPoint(
                symbol_notional_pct=symbol_notional_pct,
                max_gross_leverage=leverage_cap,
                result=result,
                competition_metrics=metrics,
                risk_discipline=risk_discipline,
                worst_leverage=max(
                    (sample.leverage for sample in risk_samples),
                    default=0.0,
                ),
                worst_net_directional_exposure=max(
                    (sample.net_directional_exposure for sample in risk_samples),
                    default=0.0,
                ),
                worst_largest_symbol_concentration=max(
                    (sample.single_instrument_exposure for sample in risk_samples),
                    default=0.0,
                ),
                walk_forward=walk_forward,
            )
        )

    return SizingFrontierResult(
        strategy_by_symbol=tuple(sorted(normalized_map.items())),
        points=tuple(sorted(points, key=lambda point: point.symbol_notional_pct)),
    )


def write_sizing_frontier_csv(result: SizingFrontierResult, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "symbol_notional_pct",
                "max_gross_leverage",
                "final_equity",
                "return_pct",
                "max_drawdown_pct",
                "sharpe_15m",
                "risk_discipline_score",
                "trade_count",
                "fills",
                "turnover_notional",
                "worst_leverage",
                "worst_net_directional_exposure",
                "worst_largest_symbol_concentration",
                "wf_positive_fold_fraction",
                "wf_active_positive_fold_fraction",
                "wf_non_negative_fold_fraction",
                "wf_median_active_test_return_pct",
                "wf_worst_test_drawdown_pct",
                "wf_total_evaluation_fills",
            ],
        )
        writer.writeheader()
        for point in result.points:
            metrics = point.competition_metrics
            walk_forward = point.walk_forward
            writer.writerow(
                {
                    "symbol_notional_pct": point.symbol_notional_pct,
                    "max_gross_leverage": point.max_gross_leverage,
                    "final_equity": metrics.final_equity,
                    "return_pct": metrics.return_pct,
                    "max_drawdown_pct": metrics.max_drawdown_pct,
                    "sharpe_15m": metrics.sharpe_15m,
                    "risk_discipline_score": point.risk_discipline.score,
                    "trade_count": metrics.trade_count,
                    "fills": len(point.result.fills),
                    "turnover_notional": point.result.metrics.turnover_notional,
                    "worst_leverage": point.worst_leverage,
                    "worst_net_directional_exposure": (
                        point.worst_net_directional_exposure
                    ),
                    "worst_largest_symbol_concentration": (
                        point.worst_largest_symbol_concentration
                    ),
                    "wf_positive_fold_fraction": (
                        "" if walk_forward is None else walk_forward.positive_fold_fraction
                    ),
                    "wf_active_positive_fold_fraction": (
                        ""
                        if walk_forward is None
                        else walk_forward.active_positive_fold_fraction
                    ),
                    "wf_non_negative_fold_fraction": (
                        ""
                        if walk_forward is None
                        else walk_forward.non_negative_fold_fraction
                    ),
                    "wf_median_active_test_return_pct": (
                        ""
                        if walk_forward is None
                        else walk_forward.median_active_test_return_pct
                    ),
                    "wf_worst_test_drawdown_pct": (
                        ""
                        if walk_forward is None
                        else walk_forward.worst_test_drawdown_pct
                    ),
                    "wf_total_evaluation_fills": (
                        "" if walk_forward is None else walk_forward.total_evaluation_fills
                    ),
                }
            )


def _run_strategy_map(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    strategy_by_symbol: dict[str, str],
) -> PortfolioBacktestResult:
    engine = PortfolioBacktestEngine(
        strategies={
            symbol: config.build_strategy(strategy_name, symbol=symbol)
            for symbol, strategy_name in strategy_by_symbol.items()
        },
        risk_limits=config.risk,
        quality_limits=config.market_quality,
        quality_limits_by_symbol={
            symbol: replace(
                config.market_quality,
                max_spread_bps=instrument_for(symbol).max_spread_bps,
            )
            for symbol in strategy_by_symbol
        },
        clock=config.competition.to_clock(),
        fill_model=FillModel(slippage_bps=config.backtest.slippage_bps),
        periods_per_year=config.backtest.periods_per_year,
    )
    return engine.run(
        prices=prices,
        quotes=quotes,
        starting_equity=config.competition.starting_equity,
    )


def _normalize_strategy_map(strategy_by_symbol: dict[str, str]) -> dict[str, str]:
    if not strategy_by_symbol:
        raise ValueError("strategy map cannot be empty")
    return {
        instrument_for(symbol).symbol: normalize_strategy_name(strategy_name)
        for symbol, strategy_name in strategy_by_symbol.items()
    }


def _validate_symbol_notional_pcts(values: tuple[float, ...]) -> None:
    if not values:
        raise ValueError("symbol_notional_pcts cannot be empty")
    for value in values:
        if not 0 < value <= 1:
            raise ValueError("symbol_notional_pcts must be in (0, 1]")
