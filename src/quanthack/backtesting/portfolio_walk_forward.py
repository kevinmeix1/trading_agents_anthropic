from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median

from quanthack.backtesting.portfolio_universe_scan import (
    PortfolioUniverseScan,
    PortfolioUniverseScanRow,
    UniverseBasket,
    scan_portfolio_universes,
)
from quanthack.core.config import AppConfig
from quanthack.core.instruments import enabled_symbols
from quanthack.market.market_data import PriceHistory, QuoteHistory


@dataclass(frozen=True)
class PortfolioWalkForwardFold:
    fold_index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    selected_basket: UniverseBasket
    selected_strategy: str
    train_row: PortfolioUniverseScanRow
    test_row: PortfolioUniverseScanRow
    test_best_row: PortfolioUniverseScanRow
    stable_candidate: bool

    @property
    def selected_symbols_text(self) -> str:
        return " ".join(self.selected_basket.symbols)

    @property
    def selected_candidate_text(self) -> str:
        return f"{self.selected_basket.name}/{self.selected_strategy}"

    @property
    def test_best_candidate_text(self) -> str:
        return f"{self.test_best_row.basket.name}/{self.test_best_row.strategy_name}"

    @property
    def selected_was_test_best(self) -> bool:
        return (
            self.selected_basket.symbols == self.test_best_row.basket.symbols
            and self.selected_strategy == self.test_best_row.strategy_name
        )


@dataclass(frozen=True)
class PortfolioWalkForwardSummary:
    folds: tuple[PortfolioWalkForwardFold, ...]
    stable_fold_fraction: float
    median_test_proxy_score: float
    median_test_return_pct: float
    lower_quartile_test_return_pct: float
    median_test_sharpe_15m: float
    worst_test_drawdown_pct: float
    average_risk_discipline_score: float
    total_test_fills: int
    total_test_turnover: float
    most_selected_basket: str
    most_selected_strategy: str
    eligible: bool

    @property
    def rank_key(self) -> tuple[int, float, float, float, float, float]:
        return (
            1 if self.eligible else 0,
            self.stable_fold_fraction,
            self.median_test_proxy_score,
            self.median_test_return_pct,
            self.median_test_sharpe_15m,
            -self.worst_test_drawdown_pct,
        )


@dataclass(frozen=True)
class PortfolioPromotionDecision:
    status: str
    live_ready: bool
    reason: str


@dataclass(frozen=True)
class PortfolioWalkForwardResult:
    available_symbols: tuple[str, ...]
    strategies: tuple[str, ...]
    folds: tuple[PortfolioWalkForwardFold, ...]
    summary: PortfolioWalkForwardSummary


def decide_portfolio_promotion(
    summary: PortfolioWalkForwardSummary,
    *,
    min_median_return_pct: float = 0.0,
    min_lower_quartile_return_pct: float = 0.0,
) -> PortfolioPromotionDecision:
    if not summary.folds:
        return PortfolioPromotionDecision(
            status="REJECT",
            live_ready=False,
            reason="no walk-forward folds were produced",
        )
    if not summary.eligible:
        return PortfolioPromotionDecision(
            status="REJECT",
            live_ready=False,
            reason=(
                "walk-forward eligibility failed; do not promote full-sample "
                "backtest winners directly"
            ),
        )
    if summary.median_test_return_pct <= min_median_return_pct:
        return PortfolioPromotionDecision(
            status="REJECT",
            live_ready=False,
            reason=(
                f"median test return {summary.median_test_return_pct:.3%} is not "
                f"above {min_median_return_pct:.3%}"
            ),
        )
    if summary.lower_quartile_test_return_pct < min_lower_quartile_return_pct:
        return PortfolioPromotionDecision(
            status="PAPER_ONLY",
            live_ready=False,
            reason=(
                f"lower-quartile test return {summary.lower_quartile_test_return_pct:.3%} "
                f"is below {min_lower_quartile_return_pct:.3%}; keep testing"
            ),
        )
    return PortfolioPromotionDecision(
        status="PROMOTE",
        live_ready=True,
        reason="walk-forward stability, return, drawdown, and risk gates passed",
    )


def run_portfolio_walk_forward(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    strategy_names: tuple[str, ...] = ("alpha_router",),
    baskets: tuple[UniverseBasket, ...] | None = None,
    min_symbols: int = 3,
    max_symbols: int = 5,
    max_baskets: int = 25,
    train_size: int = 96,
    test_size: int = 48,
    step_size: int = 48,
    min_test_fills: int = 1,
    min_stable_fold_fraction: float = 0.50,
    max_test_drawdown_pct: float = 0.05,
    min_risk_discipline_score: int = 80,
) -> PortfolioWalkForwardResult:
    _validate_inputs(
        min_symbols=min_symbols,
        max_symbols=max_symbols,
        max_baskets=max_baskets,
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
        min_test_fills=min_test_fills,
        min_stable_fold_fraction=min_stable_fold_fraction,
        max_test_drawdown_pct=max_test_drawdown_pct,
        min_risk_discipline_score=min_risk_discipline_score,
    )
    available_symbols, timestamps = _common_supported_timestamps(prices, quotes)
    if len(timestamps) < train_size + test_size:
        raise ValueError("not enough aligned timestamps for one portfolio walk-forward fold")

    folds: list[PortfolioWalkForwardFold] = []
    for fold_index, start in enumerate(
        range(0, len(timestamps) - train_size - test_size + 1, step_size),
        start=1,
    ):
        train_timestamps = timestamps[start : start + train_size]
        test_timestamps = timestamps[start + train_size : start + train_size + test_size]
        train_prices = _slice_prices(
            prices,
            symbols=available_symbols,
            timestamps=train_timestamps,
        )
        train_quotes = _slice_quotes(
            quotes,
            symbols=available_symbols,
            timestamps=train_timestamps,
        )
        test_prices = _slice_prices(
            prices,
            symbols=available_symbols,
            timestamps=test_timestamps,
        )
        test_quotes = _slice_quotes(
            quotes,
            symbols=available_symbols,
            timestamps=test_timestamps,
        )

        train_scan = scan_portfolio_universes(
            config=config,
            prices=train_prices,
            quotes=train_quotes,
            strategy_names=strategy_names,
            baskets=baskets,
            min_symbols=min_symbols,
            max_symbols=max_symbols,
            max_baskets=max_baskets,
        )
        if train_scan.best is None:
            raise ValueError("portfolio walk-forward train scan returned no candidates")

        test_scan = scan_portfolio_universes(
            config=config,
            prices=test_prices,
            quotes=test_quotes,
            strategy_names=train_scan.strategies,
            baskets=train_scan.baskets,
            min_symbols=min_symbols,
            max_symbols=max_symbols,
            max_baskets=max_baskets,
        )
        if test_scan.best is None:
            raise ValueError("portfolio walk-forward test scan returned no candidates")

        selected_train_row = train_scan.best
        selected_test_row = _matching_test_row(test_scan, selected_train_row)
        stable_candidate = _is_stable_candidate(
            selected_test_row,
            min_test_fills=min_test_fills,
            max_test_drawdown_pct=max_test_drawdown_pct,
            min_risk_discipline_score=min_risk_discipline_score,
        )
        folds.append(
            PortfolioWalkForwardFold(
                fold_index=fold_index,
                train_start=train_timestamps[0].isoformat(timespec="seconds"),
                train_end=train_timestamps[-1].isoformat(timespec="seconds"),
                test_start=test_timestamps[0].isoformat(timespec="seconds"),
                test_end=test_timestamps[-1].isoformat(timespec="seconds"),
                selected_basket=selected_train_row.basket,
                selected_strategy=selected_train_row.strategy_name,
                train_row=selected_train_row,
                test_row=selected_test_row,
                test_best_row=test_scan.best,
                stable_candidate=stable_candidate,
            )
        )

    summary = _summarize(
        tuple(folds),
        min_test_fills=min_test_fills,
        min_stable_fold_fraction=min_stable_fold_fraction,
        max_test_drawdown_pct=max_test_drawdown_pct,
        min_risk_discipline_score=min_risk_discipline_score,
    )
    strategies = tuple(dict.fromkeys(fold.selected_strategy for fold in folds))
    return PortfolioWalkForwardResult(
        available_symbols=available_symbols,
        strategies=strategies,
        folds=tuple(folds),
        summary=summary,
    )


def write_portfolio_walk_forward_summary_csv(
    result: PortfolioWalkForwardResult,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = result.summary
    promotion = decide_portfolio_promotion(summary)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "eligible",
                "folds",
                "available_symbols",
                "promotion_status",
                "promotion_live_ready",
                "promotion_reason",
                "most_selected_basket",
                "most_selected_strategy",
                "stable_fold_fraction",
                "median_test_proxy_score",
                "median_test_return_pct",
                "lower_quartile_test_return_pct",
                "median_test_sharpe_15m",
                "worst_test_drawdown_pct",
                "average_risk_discipline_score",
                "total_test_fills",
                "total_test_turnover",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "eligible": summary.eligible,
                "folds": len(summary.folds),
                "available_symbols": " ".join(result.available_symbols),
                "promotion_status": promotion.status,
                "promotion_live_ready": promotion.live_ready,
                "promotion_reason": promotion.reason,
                "most_selected_basket": summary.most_selected_basket,
                "most_selected_strategy": summary.most_selected_strategy,
                "stable_fold_fraction": summary.stable_fold_fraction,
                "median_test_proxy_score": summary.median_test_proxy_score,
                "median_test_return_pct": summary.median_test_return_pct,
                "lower_quartile_test_return_pct": summary.lower_quartile_test_return_pct,
                "median_test_sharpe_15m": summary.median_test_sharpe_15m,
                "worst_test_drawdown_pct": summary.worst_test_drawdown_pct,
                "average_risk_discipline_score": summary.average_risk_discipline_score,
                "total_test_fills": summary.total_test_fills,
                "total_test_turnover": summary.total_test_turnover,
            }
        )


def write_portfolio_walk_forward_folds_csv(
    result: PortfolioWalkForwardResult,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "fold",
                "train_start",
                "train_end",
                "test_start",
                "test_end",
                "selected_basket",
                "selected_symbols",
                "selected_strategy",
                "train_proxy_score",
                "test_proxy_score",
                "test_best_candidate",
                "selected_was_test_best",
                "stable_candidate",
                "train_return_pct",
                "train_drawdown_pct",
                "train_sharpe_15m",
                "train_risk_discipline_score",
                "test_return_pct",
                "test_drawdown_pct",
                "test_sharpe_15m",
                "test_risk_discipline_score",
                "test_final_equity",
                "test_trade_count",
                "test_fills",
                "test_turnover_notional",
            ],
        )
        writer.writeheader()
        for fold in result.folds:
            train_metrics = fold.train_row.competition_metrics
            test_metrics = fold.test_row.competition_metrics
            writer.writerow(
                {
                    "fold": fold.fold_index,
                    "train_start": fold.train_start,
                    "train_end": fold.train_end,
                    "test_start": fold.test_start,
                    "test_end": fold.test_end,
                    "selected_basket": fold.selected_basket.name,
                    "selected_symbols": fold.selected_symbols_text,
                    "selected_strategy": fold.selected_strategy,
                    "train_proxy_score": fold.train_row.proxy_score,
                    "test_proxy_score": fold.test_row.proxy_score,
                    "test_best_candidate": fold.test_best_candidate_text,
                    "selected_was_test_best": fold.selected_was_test_best,
                    "stable_candidate": fold.stable_candidate,
                    "train_return_pct": train_metrics.return_pct,
                    "train_drawdown_pct": train_metrics.max_drawdown_pct,
                    "train_sharpe_15m": train_metrics.sharpe_15m,
                    "train_risk_discipline_score": fold.train_row.risk_discipline.score,
                    "test_return_pct": test_metrics.return_pct,
                    "test_drawdown_pct": test_metrics.max_drawdown_pct,
                    "test_sharpe_15m": test_metrics.sharpe_15m,
                    "test_risk_discipline_score": fold.test_row.risk_discipline.score,
                    "test_final_equity": test_metrics.final_equity,
                    "test_trade_count": test_metrics.trade_count,
                    "test_fills": len(fold.test_row.result.fills),
                    "test_turnover_notional": (
                        fold.test_row.result.metrics.turnover_notional
                    ),
                }
            )


def _common_supported_timestamps(
    prices: PriceHistory,
    quotes: QuoteHistory,
) -> tuple[tuple[str, ...], tuple[datetime, ...]]:
    symbols: list[str] = []
    common_timestamps: set | None = None
    for symbol in enabled_symbols():
        price_timestamps = {bar.timestamp for bar in prices.for_symbol(symbol).bars}
        quote_timestamps = {quote.timestamp for quote in quotes.for_symbol(symbol).quotes}
        timestamps = price_timestamps & quote_timestamps
        if not timestamps:
            continue
        symbols.append(symbol)
        common_timestamps = (
            timestamps
            if common_timestamps is None
            else common_timestamps & timestamps
        )

    if not symbols or not common_timestamps:
        raise ValueError("no supported symbols have aligned price and quote timestamps")
    return tuple(symbols), tuple(sorted(common_timestamps))


def _slice_prices(
    prices: PriceHistory,
    *,
    symbols: tuple[str, ...],
    timestamps: tuple[datetime, ...],
) -> PriceHistory:
    timestamp_set = set(timestamps)
    return PriceHistory(
        tuple(
            bar
            for bar in prices.bars
            if bar.symbol in symbols and bar.timestamp in timestamp_set
        )
    )


def _slice_quotes(
    quotes: QuoteHistory,
    *,
    symbols: tuple[str, ...],
    timestamps: tuple[datetime, ...],
) -> QuoteHistory:
    timestamp_set = set(timestamps)
    return QuoteHistory(
        tuple(
            quote
            for quote in quotes.quotes
            if quote.symbol in symbols and quote.timestamp in timestamp_set
        )
    )


def _matching_test_row(
    test_scan: PortfolioUniverseScan,
    train_row: PortfolioUniverseScanRow,
) -> PortfolioUniverseScanRow:
    for row in test_scan.rows:
        if (
            row.strategy_name == train_row.strategy_name
            and row.basket.symbols == train_row.basket.symbols
        ):
            return row
    raise ValueError(
        f"selected train candidate {train_row.basket.name}/{train_row.strategy_name} "
        "was not found in the test scan"
    )


def _is_stable_candidate(
    row: PortfolioUniverseScanRow,
    *,
    min_test_fills: int,
    max_test_drawdown_pct: float,
    min_risk_discipline_score: int,
) -> bool:
    metrics = row.competition_metrics
    return (
        metrics.return_pct > 0
        and metrics.max_drawdown_pct <= max_test_drawdown_pct
        and row.risk_discipline.score >= min_risk_discipline_score
        and len(row.result.fills) >= min_test_fills
    )


def _summarize(
    folds: tuple[PortfolioWalkForwardFold, ...],
    *,
    min_test_fills: int,
    min_stable_fold_fraction: float,
    max_test_drawdown_pct: float,
    min_risk_discipline_score: int,
) -> PortfolioWalkForwardSummary:
    test_returns = [fold.test_row.competition_metrics.return_pct for fold in folds]
    test_drawdowns = [
        fold.test_row.competition_metrics.max_drawdown_pct for fold in folds
    ]
    test_sharpes = [fold.test_row.competition_metrics.sharpe_15m for fold in folds]
    test_proxy_scores = [fold.test_row.proxy_score for fold in folds]
    risk_scores = [fold.test_row.risk_discipline.score for fold in folds]
    stable_fraction = (
        sum(1 for fold in folds if fold.stable_candidate) / len(folds)
        if folds
        else 0.0
    )
    total_fills = sum(len(fold.test_row.result.fills) for fold in folds)
    total_turnover = sum(
        fold.test_row.result.metrics.turnover_notional for fold in folds
    )
    basket_counter = Counter(fold.selected_basket.name for fold in folds)
    strategy_counter = Counter(fold.selected_strategy for fold in folds)
    worst_drawdown = max(test_drawdowns, default=0.0)
    average_risk_score = sum(risk_scores) / len(risk_scores) if risk_scores else 0.0
    eligible = (
        total_fills >= min_test_fills
        and stable_fraction >= min_stable_fold_fraction
        and worst_drawdown <= max_test_drawdown_pct
        and average_risk_score >= min_risk_discipline_score
    )

    return PortfolioWalkForwardSummary(
        folds=folds,
        stable_fold_fraction=stable_fraction,
        median_test_proxy_score=median(test_proxy_scores) if test_proxy_scores else 0.0,
        median_test_return_pct=median(test_returns) if test_returns else 0.0,
        lower_quartile_test_return_pct=_lower_quartile(test_returns),
        median_test_sharpe_15m=median(test_sharpes) if test_sharpes else 0.0,
        worst_test_drawdown_pct=worst_drawdown,
        average_risk_discipline_score=average_risk_score,
        total_test_fills=total_fills,
        total_test_turnover=total_turnover,
        most_selected_basket=basket_counter.most_common(1)[0][0] if basket_counter else "",
        most_selected_strategy=(
            strategy_counter.most_common(1)[0][0] if strategy_counter else ""
        ),
        eligible=eligible,
    )


def _lower_quartile(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = max(0, int(len(sorted_values) * 0.25) - 1)
    return sorted_values[index]


def _validate_inputs(
    *,
    min_symbols: int,
    max_symbols: int,
    max_baskets: int,
    train_size: int,
    test_size: int,
    step_size: int,
    min_test_fills: int,
    min_stable_fold_fraction: float,
    max_test_drawdown_pct: float,
    min_risk_discipline_score: int,
) -> None:
    if min_symbols < 1:
        raise ValueError("min_symbols must be at least 1")
    if max_symbols < min_symbols:
        raise ValueError("max_symbols must be at least min_symbols")
    if max_baskets < 1:
        raise ValueError("max_baskets must be at least 1")
    if train_size < 2:
        raise ValueError("train_size must be at least 2")
    if test_size < 1:
        raise ValueError("test_size must be at least 1")
    if step_size < 1:
        raise ValueError("step_size must be at least 1")
    if min_test_fills < 0:
        raise ValueError("min_test_fills cannot be negative")
    if not 0 <= min_stable_fold_fraction <= 1:
        raise ValueError("min_stable_fold_fraction must be between 0 and 1")
    if not 0 <= max_test_drawdown_pct <= 1:
        raise ValueError("max_test_drawdown_pct must be between 0 and 1")
    if not 0 <= min_risk_discipline_score <= 100:
        raise ValueError("min_risk_discipline_score must be between 0 and 100")
