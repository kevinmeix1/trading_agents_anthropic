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
from quanthack.strategies.strategy import STRATEGY_NAMES, normalize_strategy_name


@dataclass(frozen=True)
class SymbolStrategyScore:
    symbol: str
    strategy_name: str
    total_pnl_usd: float
    return_pct: float
    max_drawdown_pct: float
    sharpe_15m: float
    fills: int
    risk_discipline_score: int

    @property
    def rank_key(self) -> tuple[float, int, float, float, float, int]:
        return (
            self.total_pnl_usd,
            self.risk_discipline_score,
            self.return_pct,
            self.sharpe_15m,
            -self.max_drawdown_pct,
            -self.fills,
        )


@dataclass(frozen=True)
class StrategyMapCandidate:
    label: str
    fallback_strategy: str
    strategy_by_symbol: tuple[tuple[str, str], ...]
    result: PortfolioBacktestResult
    competition_metrics: CompetitionMetrics
    risk_discipline: RiskDisciplineReport
    walk_forward: FixedWarmupPortfolioWalkForwardResult | None = None

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(symbol for symbol, _ in self.strategy_by_symbol)

    @property
    def strategy_map_text(self) -> str:
        return " ".join(
            f"{symbol}={strategy}" for symbol, strategy in self.strategy_by_symbol
        )

    @property
    def rank_key(self) -> tuple[float, ...]:
        if self.walk_forward is not None:
            return (
                _coverage_adjusted_active_score(self.walk_forward),
                self.walk_forward.active_positive_fold_fraction,
                self.walk_forward.non_negative_fold_fraction,
                self.walk_forward.median_active_test_return_pct,
                self.walk_forward.positive_fold_fraction,
                -self.walk_forward.losing_fold_fraction,
                self.risk_discipline.score,
                self.competition_metrics.return_pct,
                self.competition_metrics.sharpe_15m,
                -self.competition_metrics.max_drawdown_pct,
            )
        return (
            self.risk_discipline.score,
            self.competition_metrics.return_pct,
            self.competition_metrics.sharpe_15m,
            -self.competition_metrics.max_drawdown_pct,
            -float(self.competition_metrics.trade_count),
        )


@dataclass(frozen=True)
class StrategyMapOptimizationResult:
    available_symbols: tuple[str, ...]
    strategy_names: tuple[str, ...]
    symbol_scores: tuple[SymbolStrategyScore, ...]
    candidates: tuple[StrategyMapCandidate, ...]

    @property
    def best(self) -> StrategyMapCandidate | None:
        if not self.candidates:
            return None
        return self.candidates[0]


def optimize_strategy_map(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    strategy_names: tuple[str, ...],
    symbols: tuple[str, ...] | None = None,
    include_walk_forward: bool = False,
    train_size: int = 960,
    test_size: int = 192,
    step_size: int = 192,
    min_positive_pnl_usd: float = 0.0,
    top_symbol_counts: tuple[int, ...] = (3, 4, 5, 6),
) -> StrategyMapOptimizationResult:
    selected_symbols = _selected_symbols(prices=prices, quotes=quotes, symbols=symbols)
    normalized_strategies = _normalize_unique_strategy_names(strategy_names)
    if not normalized_strategies:
        raise ValueError("strategy-map optimizer needs at least one strategy")

    baseline_results = {
        strategy_name: _evaluate_strategy_map(
            config=config,
            prices=prices,
            quotes=quotes,
            strategy_by_symbol={
                symbol: strategy_name for symbol in selected_symbols
            },
        )
        for strategy_name in normalized_strategies
    }
    symbol_scores = _symbol_scores_from_baselines(
        baseline_results=baseline_results,
        starting_equity=config.competition.starting_equity,
    )
    candidate_specs = _candidate_specs(
        symbols=selected_symbols,
        strategy_names=normalized_strategies,
        symbol_scores=symbol_scores,
        min_positive_pnl_usd=min_positive_pnl_usd,
        top_symbol_counts=top_symbol_counts,
    )

    candidates: list[StrategyMapCandidate] = []
    seen_maps: set[tuple[tuple[str, str], ...]] = set()
    for label, strategy_by_symbol in candidate_specs:
        strategy_map = tuple(sorted(strategy_by_symbol.items()))
        if not strategy_map or strategy_map in seen_maps:
            continue
        seen_maps.add(strategy_map)
        fallback_strategy = strategy_map[0][1]
        evaluation = baseline_results.get(fallback_strategy)
        if (
            evaluation is None
            or tuple(sorted((symbol, fallback_strategy) for symbol in selected_symbols))
            != strategy_map
        ):
            evaluation = _evaluate_strategy_map(
                config=config,
                prices=prices,
                quotes=quotes,
                strategy_by_symbol=dict(strategy_map),
            )
        result = evaluation.result
        competition_metrics = evaluation.competition_metrics
        risk_discipline = evaluation.risk_discipline
        walk_forward = (
            run_fixed_warmup_portfolio_walk_forward(
                config=config,
                prices=prices,
                quotes=quotes,
                strategy_name=fallback_strategy,
                symbols=tuple(symbol for symbol, _ in strategy_map),
                strategy_by_symbol=dict(strategy_map),
                train_size=train_size,
                test_size=test_size,
                step_size=step_size,
            )
            if include_walk_forward
            else None
        )
        candidates.append(
            StrategyMapCandidate(
                label=label,
                fallback_strategy=fallback_strategy,
                strategy_by_symbol=strategy_map,
                result=result,
                competition_metrics=competition_metrics,
                risk_discipline=risk_discipline,
                walk_forward=walk_forward,
            )
        )

    candidates.sort(key=lambda candidate: candidate.rank_key, reverse=True)
    return StrategyMapOptimizationResult(
        available_symbols=selected_symbols,
        strategy_names=normalized_strategies,
        symbol_scores=tuple(sorted(symbol_scores, key=lambda row: row.rank_key, reverse=True)),
        candidates=tuple(candidates),
    )


@dataclass(frozen=True)
class _StrategyMapEvaluation:
    result: PortfolioBacktestResult
    competition_metrics: CompetitionMetrics
    risk_discipline: RiskDisciplineReport


def write_strategy_map_optimization_csv(
    result: StrategyMapOptimizationResult,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "label",
                "symbols",
                "strategy_map",
                "final_equity",
                "return_pct",
                "max_drawdown_pct",
                "sharpe_15m",
                "risk_discipline_score",
                "trade_count",
                "fills",
                "turnover_notional",
                "total_pnl_usd",
                "wf_positive_fold_fraction",
                "wf_active_fold_fraction",
                "wf_active_positive_fold_fraction",
                "wf_non_negative_fold_fraction",
                "wf_losing_fold_fraction",
                "wf_median_test_return_pct",
                "wf_median_active_test_return_pct",
                "wf_worst_test_drawdown_pct",
                "wf_total_evaluation_fills",
            ],
        )
        writer.writeheader()
        for rank, candidate in enumerate(result.candidates, start=1):
            metrics = candidate.competition_metrics
            walk_forward = candidate.walk_forward
            writer.writerow(
                {
                    "rank": rank,
                    "label": candidate.label,
                    "symbols": " ".join(candidate.symbols),
                    "strategy_map": candidate.strategy_map_text,
                    "final_equity": metrics.final_equity,
                    "return_pct": metrics.return_pct,
                    "max_drawdown_pct": metrics.max_drawdown_pct,
                    "sharpe_15m": metrics.sharpe_15m,
                    "risk_discipline_score": candidate.risk_discipline.score,
                    "trade_count": metrics.trade_count,
                    "fills": len(candidate.result.fills),
                    "turnover_notional": candidate.result.metrics.turnover_notional,
                    "total_pnl_usd": candidate.result.total_pnl_usd,
                    "wf_positive_fold_fraction": (
                        "" if walk_forward is None else walk_forward.positive_fold_fraction
                    ),
                    "wf_active_fold_fraction": (
                        "" if walk_forward is None else walk_forward.active_fold_fraction
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
                    "wf_losing_fold_fraction": (
                        "" if walk_forward is None else walk_forward.losing_fold_fraction
                    ),
                    "wf_median_test_return_pct": (
                        "" if walk_forward is None else walk_forward.median_test_return_pct
                    ),
                    "wf_median_active_test_return_pct": (
                        ""
                        if walk_forward is None
                        else walk_forward.median_active_test_return_pct
                    ),
                    "wf_worst_test_drawdown_pct": (
                        "" if walk_forward is None else walk_forward.worst_test_drawdown_pct
                    ),
                    "wf_total_evaluation_fills": (
                        "" if walk_forward is None else walk_forward.total_evaluation_fills
                    ),
                }
            )


def write_symbol_strategy_scores_csv(
    result: StrategyMapOptimizationResult,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "symbol",
                "strategy",
                "total_pnl_usd",
                "return_pct",
                "max_drawdown_pct",
                "sharpe_15m",
                "fills",
                "risk_discipline_score",
            ],
        )
        writer.writeheader()
        for rank, score in enumerate(result.symbol_scores, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "symbol": score.symbol,
                    "strategy": score.strategy_name,
                    "total_pnl_usd": score.total_pnl_usd,
                    "return_pct": score.return_pct,
                    "max_drawdown_pct": score.max_drawdown_pct,
                    "sharpe_15m": score.sharpe_15m,
                    "fills": score.fills,
                    "risk_discipline_score": score.risk_discipline_score,
                }
            )


def _symbol_scores_from_baselines(
    *,
    baseline_results: dict[str, _StrategyMapEvaluation],
    starting_equity: float,
) -> tuple[SymbolStrategyScore, ...]:
    scores: list[SymbolStrategyScore] = []
    for strategy_name, evaluation in baseline_results.items():
        metrics = evaluation.competition_metrics
        fills_by_symbol = {
            symbol: len(
                [fill for fill in evaluation.result.fills if fill.symbol == symbol]
            )
            for symbol in evaluation.result.symbols
        }
        for pnl_row in evaluation.result.pnl_by_symbol:
            scores.append(
                SymbolStrategyScore(
                    symbol=pnl_row.symbol,
                    strategy_name=strategy_name,
                    total_pnl_usd=pnl_row.ledger.total_pnl_usd,
                    return_pct=pnl_row.ledger.total_pnl_usd / starting_equity,
                    max_drawdown_pct=metrics.max_drawdown_pct,
                    sharpe_15m=metrics.sharpe_15m,
                    fills=fills_by_symbol.get(pnl_row.symbol, 0),
                    risk_discipline_score=evaluation.risk_discipline.score,
                )
            )
    return tuple(scores)


def _evaluate_strategy_map(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    strategy_by_symbol: dict[str, str],
) -> _StrategyMapEvaluation:
    result = _run_strategy_map(
        config=config,
        prices=prices,
        quotes=quotes,
        strategy_by_symbol=strategy_by_symbol,
    )
    competition_metrics = build_competition_metrics(
        equity_points=result.equity_curve,
        fills=result.fills,
    )
    risk_discipline = build_risk_discipline_report(
        risk_samples_from_portfolio_equity(result.equity_curve)
    )
    return _StrategyMapEvaluation(
        result=result,
        competition_metrics=competition_metrics,
        risk_discipline=risk_discipline,
    )


def _run_strategy_map(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    strategy_by_symbol: dict[str, str],
) -> PortfolioBacktestResult:
    selected_symbols = tuple(strategy_by_symbol)
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
            for symbol in selected_symbols
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


def _candidate_specs(
    *,
    symbols: tuple[str, ...],
    strategy_names: tuple[str, ...],
    symbol_scores: tuple[SymbolStrategyScore, ...],
    min_positive_pnl_usd: float,
    top_symbol_counts: tuple[int, ...],
) -> tuple[tuple[str, dict[str, str]], ...]:
    best_by_symbol = _best_score_by_symbol(symbol_scores)
    specs: list[tuple[str, dict[str, str]]] = []

    for strategy_name in strategy_names:
        specs.append(
            (
                f"all_{strategy_name}",
                {symbol: strategy_name for symbol in symbols},
            )
        )

    specs.append(
        (
            "best_per_symbol_all",
            {
                symbol: best_by_symbol[symbol].strategy_name
                for symbol in symbols
                if symbol in best_by_symbol
            },
        )
    )
    positive_best = {
        symbol: score.strategy_name
        for symbol, score in best_by_symbol.items()
        if score.total_pnl_usd > min_positive_pnl_usd
    }
    specs.append(("best_per_symbol_positive_only", positive_best))

    ranked_positive_symbols = [
        score.symbol
        for score in sorted(best_by_symbol.values(), key=lambda row: row.rank_key, reverse=True)
        if score.total_pnl_usd > min_positive_pnl_usd
    ]
    for count in top_symbol_counts:
        if count < 1:
            continue
        selected = tuple(ranked_positive_symbols[:count])
        if not selected:
            continue
        specs.append(
            (
                f"top_{len(selected)}_best_symbol_strategies",
                {symbol: best_by_symbol[symbol].strategy_name for symbol in selected},
            )
        )

    return tuple(specs)


def _best_score_by_symbol(
    symbol_scores: tuple[SymbolStrategyScore, ...],
) -> dict[str, SymbolStrategyScore]:
    best: dict[str, SymbolStrategyScore] = {}
    for score in symbol_scores:
        current = best.get(score.symbol)
        if current is None or score.rank_key > current.rank_key:
            best[score.symbol] = score
    return best


def _selected_symbols(
    *,
    prices: PriceHistory,
    quotes: QuoteHistory,
    symbols: tuple[str, ...] | None,
) -> tuple[str, ...]:
    if symbols:
        selected = tuple(instrument_for(symbol).symbol for symbol in symbols)
    else:
        selected = tuple(sorted(set(prices.symbols()) & set(quotes.symbols())))
    if not selected:
        raise ValueError("no symbols found in both price and quote data")
    return selected


def _normalize_unique_strategy_names(strategy_names: tuple[str, ...]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_name in strategy_names:
        strategy_name = normalize_strategy_name(raw_name)
        if strategy_name in seen:
            continue
        if strategy_name not in STRATEGY_NAMES:
            raise ValueError(f"unsupported strategy {raw_name!r}")
        normalized.append(strategy_name)
        seen.add(strategy_name)
    return tuple(normalized)


def _coverage_adjusted_active_score(
    walk_forward: FixedWarmupPortfolioWalkForwardResult,
    *,
    target_active_fold_fraction: float = 0.35,
) -> float:
    if target_active_fold_fraction <= 0:
        return walk_forward.active_positive_fold_fraction
    coverage = min(walk_forward.active_fold_fraction / target_active_fold_fraction, 1.0)
    return coverage * walk_forward.active_positive_fold_fraction
