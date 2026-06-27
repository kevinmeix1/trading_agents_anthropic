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
    official_composite_score,
    risk_samples_from_portfolio_equity,
)
from quanthack.core.config import AppConfig
from quanthack.core.instruments import instrument_for
from quanthack.market.market_data import PriceHistory, QuoteHistory
from quanthack.backtesting.portfolio_backtest import PortfolioBacktestEngine, PortfolioBacktestResult
from quanthack.strategies.strategy import STRATEGY_NAMES, normalize_strategy_name


@dataclass(frozen=True)
class PortfolioStrategyComparisonRow:
    strategy_name: str
    result: PortfolioBacktestResult
    competition_metrics: CompetitionMetrics
    risk_discipline: RiskDisciplineReport
    return_rank: float = 0.0
    drawdown_rank: float = 0.0
    sharpe_rank: float = 0.0
    proxy_score: float = 0.0

    @property
    def rank_key(self) -> tuple[float, int, float, float, float]:
        return (
            self.proxy_score,
            self.risk_discipline.score,
            self.competition_metrics.return_pct,
            self.competition_metrics.sharpe_15m,
            -self.competition_metrics.max_drawdown_pct,
        )


@dataclass(frozen=True)
class PortfolioStrategyComparison:
    symbols: tuple[str, ...]
    rows: tuple[PortfolioStrategyComparisonRow, ...]

    @property
    def best(self) -> PortfolioStrategyComparisonRow | None:
        if not self.rows:
            return None
        return self.rows[0]


def compare_portfolio_strategies(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    strategy_names: tuple[str, ...] = STRATEGY_NAMES,
    symbols: tuple[str, ...] | None = None,
) -> PortfolioStrategyComparison:
    selected_symbols = _selected_symbols(
        prices=prices,
        quotes=quotes,
        symbols=symbols,
    )
    rows: list[PortfolioStrategyComparisonRow] = []

    for strategy_name in _normalize_unique_strategy_names(strategy_names):
        engine = PortfolioBacktestEngine(
            strategies={
                symbol: config.build_strategy(strategy_name, symbol=symbol)
                for symbol in selected_symbols
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
        result = engine.run(
            prices=prices,
            quotes=quotes,
            starting_equity=config.competition.starting_equity,
        )
        competition_metrics = build_competition_metrics(
            equity_points=result.equity_curve,
            fills=result.fills,
        )
        risk_discipline = build_risk_discipline_report(
            risk_samples_from_portfolio_equity(result.equity_curve)
        )
        rows.append(
            PortfolioStrategyComparisonRow(
                strategy_name=strategy_name,
                result=result,
                competition_metrics=competition_metrics,
                risk_discipline=risk_discipline,
            )
        )

    ranked_rows = _attach_proxy_scores(tuple(rows))
    ranked_rows = tuple(sorted(ranked_rows, key=lambda row: row.rank_key, reverse=True))
    return PortfolioStrategyComparison(symbols=selected_symbols, rows=ranked_rows)


def write_portfolio_strategy_comparison_csv(
    comparison: PortfolioStrategyComparison,
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
                "symbols",
                "proxy_score",
                "return_rank",
                "drawdown_rank",
                "sharpe_rank",
                "risk_discipline_score",
                "compliance_review_required",
                "final_equity",
                "official_return_pct",
                "official_max_drawdown_pct",
                "official_15m_sharpe",
                "sharpe_rank_cap",
                "trade_count",
                "fills",
                "turnover_notional",
                "total_pnl_usd",
            ],
        )
        writer.writeheader()
        for rank, row in enumerate(comparison.rows, start=1):
            metrics = row.competition_metrics
            writer.writerow(
                {
                    "rank": rank,
                    "strategy": row.strategy_name,
                    "symbols": " ".join(comparison.symbols),
                    "proxy_score": row.proxy_score,
                    "return_rank": row.return_rank,
                    "drawdown_rank": row.drawdown_rank,
                    "sharpe_rank": row.sharpe_rank,
                    "risk_discipline_score": row.risk_discipline.score,
                    "compliance_review_required": (
                        row.risk_discipline.compliance_review_required
                    ),
                    "final_equity": metrics.final_equity,
                    "official_return_pct": metrics.return_pct,
                    "official_max_drawdown_pct": metrics.max_drawdown_pct,
                    "official_15m_sharpe": metrics.sharpe_15m,
                    "sharpe_rank_cap": metrics.sharpe_rank_cap,
                    "trade_count": metrics.trade_count,
                    "fills": len(row.result.fills),
                    "turnover_notional": row.result.metrics.turnover_notional,
                    "total_pnl_usd": row.result.total_pnl_usd,
                }
            )


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
        normalized.append(strategy_name)
        seen.add(strategy_name)
    return tuple(normalized)


def _attach_proxy_scores(
    rows: tuple[PortfolioStrategyComparisonRow, ...],
) -> tuple[PortfolioStrategyComparisonRow, ...]:
    return_ranks = _percentile_scores(
        [row.competition_metrics.return_pct for row in rows],
        higher_is_better=True,
    )
    drawdown_ranks = _percentile_scores(
        [row.competition_metrics.max_drawdown_pct for row in rows],
        higher_is_better=False,
    )
    sharpe_ranks = _percentile_scores(
        [row.competition_metrics.sharpe_15m for row in rows],
        higher_is_better=True,
    )

    ranked_rows: list[PortfolioStrategyComparisonRow] = []
    for row, return_rank, drawdown_rank, sharpe_rank in zip(
        rows,
        return_ranks,
        drawdown_ranks,
        sharpe_ranks,
        strict=True,
    ):
        proxy_score = official_composite_score(
            return_rank=return_rank,
            drawdown_rank=drawdown_rank,
            sharpe_rank=sharpe_rank,
            risk_discipline_score=row.risk_discipline.score,
            sharpe_rank_cap=row.competition_metrics.sharpe_rank_cap,
        )
        ranked_rows.append(
            replace(
                row,
                return_rank=return_rank,
                drawdown_rank=drawdown_rank,
                sharpe_rank=sharpe_rank,
                proxy_score=proxy_score,
            )
        )
    return tuple(ranked_rows)


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
        scores.append((worse / denominator) * 100)
    return tuple(scores)
