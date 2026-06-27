from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path

from quanthack.backtesting.competition_score import official_composite_score
from quanthack.backtesting.portfolio_fixed_warmup_walk_forward import (
    FixedWarmupPortfolioWalkForwardResult,
    run_fixed_warmup_portfolio_walk_forward,
)
from quanthack.backtesting.portfolio_strategy_compare import (
    PortfolioStrategyComparisonRow,
    compare_portfolio_strategies,
)
from quanthack.backtesting.strategy_attribution import (
    StrategyAttributionRow,
    run_strategy_attribution,
)
from quanthack.core.config import AppConfig
from quanthack.core.instruments import instrument_for
from quanthack.market.market_data import PriceHistory, QuoteHistory
from quanthack.strategies.strategy import normalize_strategy_name


@dataclass(frozen=True)
class SymbolEligibilityCandidate:
    name: str
    strategy_name: str
    symbols: tuple[str, ...]
    excluded_symbols: tuple[str, ...]
    reason: str
    comparison_row: PortfolioStrategyComparisonRow
    return_rank: float = 0.0
    drawdown_rank: float = 0.0
    sharpe_rank: float = 0.0
    proxy_score: float = 0.0
    walk_forward: FixedWarmupPortfolioWalkForwardResult | None = None
    walk_forward_error: str = ""

    @property
    def rank_key(self) -> tuple[float, ...]:
        metrics = self.comparison_row.competition_metrics
        if self.walk_forward is not None:
            active_quality = _coverage_adjusted_active_score(self.walk_forward)
            return (
                1.0,
                active_quality,
                self.walk_forward.active_positive_fold_fraction,
                self.walk_forward.median_active_test_return_pct,
                self.walk_forward.non_negative_fold_fraction,
                self.walk_forward.active_fold_fraction,
                self.walk_forward.positive_fold_fraction,
                self.walk_forward.median_test_return_pct,
                self.walk_forward.median_test_sharpe_15m,
                -self.walk_forward.losing_fold_fraction,
                -self.walk_forward.largest_positive_fold_contribution,
                -self.walk_forward.worst_test_drawdown_pct,
                self.walk_forward.average_risk_discipline_score,
                self.proxy_score,
                self.comparison_row.risk_discipline.score,
                metrics.return_pct,
                metrics.sharpe_15m,
                -metrics.max_drawdown_pct,
                len(self.symbols),
            )
        if self.walk_forward_error:
            return (
                0.0,
                self.proxy_score,
                self.comparison_row.risk_discipline.score,
                metrics.return_pct,
                metrics.sharpe_15m,
                -metrics.max_drawdown_pct,
                len(self.symbols),
            )
        return (
            self.proxy_score,
            self.comparison_row.risk_discipline.score,
            metrics.return_pct,
            metrics.sharpe_15m,
            -metrics.max_drawdown_pct,
            len(self.symbols),
        )


@dataclass(frozen=True)
class SymbolEligibilityOptimization:
    strategy_name: str
    available_symbols: tuple[str, ...]
    attribution_rows: tuple[StrategyAttributionRow, ...]
    candidates: tuple[SymbolEligibilityCandidate, ...]

    @property
    def best(self) -> SymbolEligibilityCandidate | None:
        if not self.candidates:
            return None
        return self.candidates[0]


def optimize_symbol_eligibility(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    strategy_name: str,
    symbols: tuple[str, ...] | None = None,
    min_symbols: int = 3,
    max_symbols: int | None = None,
    min_fills: int = 1,
    min_symbol_pnl_usd: float = 0.0,
    min_profit_factor: float = 1.0,
    max_exclusions: int = 3,
    include_flat_symbols: bool = True,
    include_walk_forward: bool = False,
    train_size: int = 960,
    test_size: int = 192,
    step_size: int = 192,
    include_combinations: bool = False,
    combination_pool_size: int = 0,
    max_combinations: int = 200,
) -> SymbolEligibilityOptimization:
    if min_symbols < 1:
        raise ValueError("min_symbols must be at least 1")
    if max_symbols is not None and max_symbols < min_symbols:
        raise ValueError("max_symbols must be at least min_symbols")
    if min_fills < 0:
        raise ValueError("min_fills cannot be negative")
    if max_exclusions < 0:
        raise ValueError("max_exclusions cannot be negative")
    if combination_pool_size < 0:
        raise ValueError("combination_pool_size cannot be negative")
    if max_combinations < 1:
        raise ValueError("max_combinations must be at least 1")

    normalized_strategy = normalize_strategy_name(strategy_name)
    attribution = run_strategy_attribution(
        config=config,
        prices=prices,
        quotes=quotes,
        strategy_names=(normalized_strategy,),
        symbols=symbols,
    )
    available_symbols = attribution.symbols
    effective_max_symbols = max_symbols or len(available_symbols)
    if effective_max_symbols > len(available_symbols):
        effective_max_symbols = len(available_symbols)

    symbol_rows = tuple(
        row
        for row in attribution.rows
        if row.strategy_name == normalized_strategy and row.symbol != "PORTFOLIO"
    )
    row_by_symbol = {row.symbol: row for row in symbol_rows}
    candidate_sets = _candidate_symbol_sets(
        available_symbols=available_symbols,
        rows=symbol_rows,
        min_symbols=min_symbols,
        max_symbols=effective_max_symbols,
        min_fills=min_fills,
        min_symbol_pnl_usd=min_symbol_pnl_usd,
        min_profit_factor=min_profit_factor,
        max_exclusions=max_exclusions,
        include_flat_symbols=include_flat_symbols,
        include_combinations=include_combinations,
        combination_pool_size=combination_pool_size,
        max_combinations=max_combinations,
    )

    candidates: list[SymbolEligibilityCandidate] = []
    for name, candidate_symbols, reason in candidate_sets:
        comparison = compare_portfolio_strategies(
            config=config,
            prices=prices,
            quotes=quotes,
            strategy_names=(normalized_strategy,),
            symbols=candidate_symbols,
        )
        if comparison.best is None:
            continue
        walk_forward: FixedWarmupPortfolioWalkForwardResult | None = None
        walk_forward_error = ""
        if include_walk_forward:
            try:
                walk_forward = run_fixed_warmup_portfolio_walk_forward(
                    config=config,
                    prices=prices,
                    quotes=quotes,
                    strategy_name=normalized_strategy,
                    symbols=comparison.symbols,
                    train_size=train_size,
                    test_size=test_size,
                    step_size=step_size,
                )
            except ValueError as exc:
                walk_forward_error = str(exc)
        candidates.append(
            SymbolEligibilityCandidate(
                name=name,
                strategy_name=normalized_strategy,
                symbols=comparison.symbols,
                excluded_symbols=tuple(
                    symbol for symbol in available_symbols if symbol not in comparison.symbols
                ),
                reason=reason,
                comparison_row=comparison.best,
                walk_forward=walk_forward,
                walk_forward_error=walk_forward_error,
            )
        )

    ranked = _attach_proxy_scores(tuple(candidates))
    ranked = tuple(sorted(ranked, key=lambda candidate: candidate.rank_key, reverse=True))
    return SymbolEligibilityOptimization(
        strategy_name=normalized_strategy,
        available_symbols=available_symbols,
        attribution_rows=tuple(
            sorted(
                symbol_rows,
                key=lambda row: (
                    row.total_pnl_usd,
                    row.profit_factor,
                    row.win_rate,
                ),
                reverse=True,
            )
        ),
        candidates=ranked,
    )


def write_symbol_eligibility_csv(
    result: SymbolEligibilityOptimization,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "candidate",
                "strategy",
                "symbols",
                "excluded_symbols",
                "reason",
                "proxy_score",
                "return_rank",
                "drawdown_rank",
                "sharpe_rank",
                "risk_discipline_score",
                "final_equity",
                "official_return_pct",
                "official_max_drawdown_pct",
                "official_15m_sharpe",
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
                "wf_median_test_sharpe_15m",
                "wf_worst_test_drawdown_pct",
                "wf_average_risk_discipline_score",
                "wf_total_evaluation_fills",
                "wf_largest_positive_fold_contribution",
                "wf_error",
            ],
        )
        writer.writeheader()
        for rank, candidate in enumerate(result.candidates, start=1):
            row = candidate.comparison_row
            metrics = row.competition_metrics
            walk_forward = candidate.walk_forward
            writer.writerow(
                {
                    "rank": rank,
                    "candidate": candidate.name,
                    "strategy": candidate.strategy_name,
                    "symbols": " ".join(candidate.symbols),
                    "excluded_symbols": " ".join(candidate.excluded_symbols),
                    "reason": candidate.reason,
                    "proxy_score": candidate.proxy_score,
                    "return_rank": candidate.return_rank,
                    "drawdown_rank": candidate.drawdown_rank,
                    "sharpe_rank": candidate.sharpe_rank,
                    "risk_discipline_score": row.risk_discipline.score,
                    "final_equity": metrics.final_equity,
                    "official_return_pct": metrics.return_pct,
                    "official_max_drawdown_pct": metrics.max_drawdown_pct,
                    "official_15m_sharpe": metrics.sharpe_15m,
                    "trade_count": metrics.trade_count,
                    "fills": len(row.result.fills),
                    "turnover_notional": row.result.metrics.turnover_notional,
                    "total_pnl_usd": row.result.total_pnl_usd,
                    "wf_positive_fold_fraction": (
                        ""
                        if walk_forward is None
                        else walk_forward.positive_fold_fraction
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
                    "wf_median_test_sharpe_15m": (
                        "" if walk_forward is None else walk_forward.median_test_sharpe_15m
                    ),
                    "wf_worst_test_drawdown_pct": (
                        "" if walk_forward is None else walk_forward.worst_test_drawdown_pct
                    ),
                    "wf_average_risk_discipline_score": (
                        ""
                        if walk_forward is None
                        else walk_forward.average_risk_discipline_score
                    ),
                    "wf_total_evaluation_fills": (
                        "" if walk_forward is None else walk_forward.total_evaluation_fills
                    ),
                    "wf_largest_positive_fold_contribution": (
                        ""
                        if walk_forward is None
                        else walk_forward.largest_positive_fold_contribution
                    ),
                    "wf_error": candidate.walk_forward_error,
                }
            )


def write_symbol_attribution_rank_csv(
    result: SymbolEligibilityOptimization,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "strategy",
                "symbol",
                "fills",
                "total_pnl_usd",
                "closed_event_count",
                "win_rate",
                "profit_factor",
            ],
        )
        writer.writeheader()
        for rank, row in enumerate(result.attribution_rows, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "strategy": row.strategy_name,
                    "symbol": row.symbol,
                    "fills": row.fills,
                    "total_pnl_usd": row.total_pnl_usd,
                    "closed_event_count": row.closed_event_count,
                    "win_rate": row.win_rate,
                    "profit_factor": row.profit_factor,
                }
            )


def _candidate_symbol_sets(
    *,
    available_symbols: tuple[str, ...],
    rows: tuple[StrategyAttributionRow, ...],
    min_symbols: int,
    max_symbols: int,
    min_fills: int,
    min_symbol_pnl_usd: float,
    min_profit_factor: float,
    max_exclusions: int,
    include_flat_symbols: bool,
    include_combinations: bool,
    combination_pool_size: int,
    max_combinations: int,
) -> tuple[tuple[str, tuple[str, ...], str], ...]:
    row_by_symbol = {row.symbol: row for row in rows}
    ranked_rows = tuple(
        sorted(
            rows,
            key=lambda row: (
                row.total_pnl_usd,
                row.profit_factor,
                row.win_rate,
            ),
            reverse=True,
        )
    )
    active_rows = tuple(row for row in ranked_rows if row.fills >= min_fills)
    flat_symbols = tuple(
        symbol for symbol in available_symbols if row_by_symbol[symbol].fills < min_fills
    )

    raw_sets: list[tuple[str, tuple[str, ...], str]] = [
        (
            "all_symbols",
            available_symbols,
            "baseline using every symbol available in price and quote data",
        )
    ]

    positive_symbols = tuple(
        row.symbol
        for row in active_rows
        if row.total_pnl_usd > min_symbol_pnl_usd
    )
    if include_flat_symbols:
        positive_with_flat = _ordered_unique(positive_symbols + flat_symbols)
        raw_sets.append(
            (
                "positive_plus_flat",
                positive_with_flat,
                "symbols with positive attribution plus symbols that did not trade",
            )
        )
    raw_sets.append(
        (
            "positive_active",
            positive_symbols,
            "symbols with positive attribution and enough fills",
        )
    )

    pf_symbols = tuple(
        row.symbol
        for row in active_rows
        if row.total_pnl_usd > min_symbol_pnl_usd
        and row.profit_factor >= min_profit_factor
    )
    raw_sets.append(
        (
            "positive_profit_factor",
            pf_symbols,
            (
                f"positive attribution with profit factor >= "
                f"{min_profit_factor:.2f}"
            ),
        )
    )

    worst_active = tuple(sorted(active_rows, key=lambda row: row.total_pnl_usd))
    for exclusion_count in range(1, min(max_exclusions, len(worst_active)) + 1):
        excluded = {row.symbol for row in worst_active[:exclusion_count]}
        symbols = tuple(symbol for symbol in available_symbols if symbol not in excluded)
        raw_sets.append(
            (
                f"drop_worst_{exclusion_count}",
                symbols,
                "drop lowest-P&L attributed symbols from the baseline universe",
            )
        )

    ranked_symbols = tuple(row.symbol for row in ranked_rows)
    for count in range(min_symbols, min(max_symbols, len(ranked_symbols)) + 1):
        raw_sets.append(
            (
                f"top_{count}_pnl",
                ranked_symbols[:count],
                "top symbols by attribution P&L",
            )
        )

    if include_combinations:
        pool_size = combination_pool_size or len(ranked_symbols)
        pool = ranked_symbols[: min(pool_size, len(ranked_symbols))]
        combination_count = 0
        for count in range(min_symbols, min(max_symbols, len(pool)) + 1):
            for symbols in combinations(pool, count):
                raw_sets.append(
                    (
                        f"combo_top{len(pool)}_{count}_{combination_count + 1}",
                        tuple(symbols),
                        (
                            "combinational basket search over top attributed "
                            f"{len(pool)} symbols"
                        ),
                    )
                )
                combination_count += 1
                if combination_count >= max_combinations:
                    break
            if combination_count >= max_combinations:
                break

    return _dedupe_candidate_sets(
        raw_sets,
        min_symbols=min_symbols,
        max_symbols=max_symbols,
    )


def _dedupe_candidate_sets(
    raw_sets: list[tuple[str, tuple[str, ...], str]],
    *,
    min_symbols: int,
    max_symbols: int,
) -> tuple[tuple[str, tuple[str, ...], str], ...]:
    seen: set[tuple[str, ...]] = set()
    candidates: list[tuple[str, tuple[str, ...], str]] = []
    for name, raw_symbols, reason in raw_sets:
        symbols = tuple(instrument_for(symbol).symbol for symbol in raw_symbols)
        symbols = _ordered_unique(symbols)
        if not min_symbols <= len(symbols) <= max_symbols:
            continue
        key = tuple(sorted(symbols))
        if key in seen:
            continue
        seen.add(key)
        candidates.append((name, symbols, reason))
    return tuple(candidates)


def _ordered_unique(symbols: tuple[str, ...]) -> tuple[str, ...]:
    selected: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        normalized = instrument_for(symbol).symbol
        if normalized in seen:
            continue
        selected.append(normalized)
        seen.add(normalized)
    return tuple(selected)


def _attach_proxy_scores(
    candidates: tuple[SymbolEligibilityCandidate, ...],
) -> tuple[SymbolEligibilityCandidate, ...]:
    return_ranks = _percentile_scores(
        [candidate.comparison_row.competition_metrics.return_pct for candidate in candidates],
        higher_is_better=True,
    )
    drawdown_ranks = _percentile_scores(
        [
            candidate.comparison_row.competition_metrics.max_drawdown_pct
            for candidate in candidates
        ],
        higher_is_better=False,
    )
    sharpe_ranks = _percentile_scores(
        [candidate.comparison_row.competition_metrics.sharpe_15m for candidate in candidates],
        higher_is_better=True,
    )

    ranked: list[SymbolEligibilityCandidate] = []
    for candidate, return_rank, drawdown_rank, sharpe_rank in zip(
        candidates,
        return_ranks,
        drawdown_ranks,
        sharpe_ranks,
        strict=True,
    ):
        metrics = candidate.comparison_row.competition_metrics
        proxy_score = official_composite_score(
            return_rank=return_rank,
            drawdown_rank=drawdown_rank,
            sharpe_rank=sharpe_rank,
            risk_discipline_score=candidate.comparison_row.risk_discipline.score,
            sharpe_rank_cap=metrics.sharpe_rank_cap,
        )
        ranked.append(
            replace(
                candidate,
                return_rank=return_rank,
                drawdown_rank=drawdown_rank,
                sharpe_rank=sharpe_rank,
                proxy_score=proxy_score,
            )
        )
    return tuple(ranked)


def _coverage_adjusted_active_score(
    walk_forward: FixedWarmupPortfolioWalkForwardResult,
    *,
    target_active_fold_fraction: float = 0.35,
) -> float:
    if target_active_fold_fraction <= 0:
        return walk_forward.active_positive_fold_fraction
    coverage = min(
        walk_forward.active_fold_fraction / target_active_fold_fraction,
        1.0,
    )
    return walk_forward.active_positive_fold_fraction * coverage


def _percentile_scores(
    values: list[float],
    *,
    higher_is_better: bool,
) -> tuple[float, ...]:
    if not values:
        return ()
    if len(values) == 1 or len(set(values)) == 1:
        return tuple(100.0 for _ in values)

    denominator = len(values) - 1
    scores: list[float] = []
    for value in values:
        if higher_is_better:
            worse = sum(1 for other in values if other < value)
        else:
            worse = sum(1 for other in values if other > value)
        scores.append((worse / denominator) * 100.0)
    return tuple(scores)
