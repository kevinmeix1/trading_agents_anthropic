from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path

from quanthack.backtesting.backtest import FillModel
from quanthack.backtesting.competition_score import (
    RiskDisciplineReport,
    build_risk_discipline_report,
    risk_samples_from_portfolio_equity,
)
from quanthack.backtesting.metrics import PerformanceMetrics
from quanthack.backtesting.portfolio_backtest import PortfolioBacktestEngine
from quanthack.core.config import AppConfig
from quanthack.core.instruments import instrument_for
from quanthack.market.market_data import PriceHistory, QuoteHistory
from quanthack.strategies.strategy import STRATEGY_NAMES, normalize_strategy_name


@dataclass(frozen=True)
class StrategyAttributionRow:
    strategy_name: str
    symbol: str
    fills: int
    realized_pnl_usd: float
    open_pnl_usd: float
    total_pnl_usd: float
    closed_event_count: int
    win_rate: float
    profit_factor: float
    final_position_units: float
    portfolio_final_equity: float
    portfolio_return_pct: float
    portfolio_max_drawdown_pct: float
    portfolio_sharpe_ratio: float
    portfolio_risk_score: int


@dataclass(frozen=True)
class StrategyAttributionReport:
    symbols: tuple[str, ...]
    rows: tuple[StrategyAttributionRow, ...]


def run_strategy_attribution(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    strategy_names: tuple[str, ...] = STRATEGY_NAMES,
    symbols: tuple[str, ...] | None = None,
) -> StrategyAttributionReport:
    selected_symbols = _selected_symbols(prices=prices, quotes=quotes, symbols=symbols)
    rows: list[StrategyAttributionRow] = []
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
        risk = build_risk_discipline_report(
            risk_samples_from_portfolio_equity(result.equity_curve)
        )
        rows.extend(
            _rows_for_strategy(
                strategy_name=strategy_name,
                result_metrics=result.metrics,
                risk=risk,
                pnl_rows=result.pnl_by_symbol,
                fills_by_symbol={
                    symbol: len([fill for fill in result.fills if fill.symbol == symbol])
                    for symbol in selected_symbols
                },
            )
        )

    ranked_rows = tuple(
        sorted(
            rows,
            key=lambda row: (
                row.strategy_name,
                row.symbol == "PORTFOLIO",
                row.total_pnl_usd,
            ),
            reverse=True,
        )
    )
    return StrategyAttributionReport(symbols=selected_symbols, rows=ranked_rows)


def write_strategy_attribution_csv(
    report: StrategyAttributionReport,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "strategy",
                "symbol",
                "fills",
                "realized_pnl_usd",
                "open_pnl_usd",
                "total_pnl_usd",
                "closed_event_count",
                "win_rate",
                "profit_factor",
                "final_position_units",
                "portfolio_final_equity",
                "portfolio_return_pct",
                "portfolio_max_drawdown_pct",
                "portfolio_sharpe_ratio",
                "portfolio_risk_score",
            ],
        )
        writer.writeheader()
        for row in report.rows:
            writer.writerow(
                {
                    "strategy": row.strategy_name,
                    "symbol": row.symbol,
                    "fills": row.fills,
                    "realized_pnl_usd": row.realized_pnl_usd,
                    "open_pnl_usd": row.open_pnl_usd,
                    "total_pnl_usd": row.total_pnl_usd,
                    "closed_event_count": row.closed_event_count,
                    "win_rate": row.win_rate,
                    "profit_factor": row.profit_factor,
                    "final_position_units": row.final_position_units,
                    "portfolio_final_equity": row.portfolio_final_equity,
                    "portfolio_return_pct": row.portfolio_return_pct,
                    "portfolio_max_drawdown_pct": row.portfolio_max_drawdown_pct,
                    "portfolio_sharpe_ratio": row.portfolio_sharpe_ratio,
                    "portfolio_risk_score": row.portfolio_risk_score,
                }
            )


def _rows_for_strategy(
    *,
    strategy_name: str,
    result_metrics: PerformanceMetrics,
    risk: RiskDisciplineReport,
    pnl_rows,
    fills_by_symbol: dict[str, int],
) -> tuple[StrategyAttributionRow, ...]:
    rows: list[StrategyAttributionRow] = []
    for pnl_row in pnl_rows:
        ledger = pnl_row.ledger
        rows.append(
            StrategyAttributionRow(
                strategy_name=strategy_name,
                symbol=pnl_row.symbol,
                fills=fills_by_symbol.get(pnl_row.symbol, 0),
                realized_pnl_usd=ledger.realized_pnl_usd,
                open_pnl_usd=ledger.open_pnl_usd,
                total_pnl_usd=ledger.total_pnl_usd,
                closed_event_count=ledger.closed_event_count,
                win_rate=ledger.realized_win_rate,
                profit_factor=ledger.realized_profit_factor,
                final_position_units=ledger.final_position_units,
                portfolio_final_equity=result_metrics.final_equity,
                portfolio_return_pct=result_metrics.total_return_pct,
                portfolio_max_drawdown_pct=result_metrics.max_drawdown_pct,
                portfolio_sharpe_ratio=result_metrics.sharpe_ratio,
                portfolio_risk_score=risk.score,
            )
        )
    rows.append(
        StrategyAttributionRow(
            strategy_name=strategy_name,
            symbol="PORTFOLIO",
            fills=sum(fills_by_symbol.values()),
            realized_pnl_usd=sum(row.ledger.realized_pnl_usd for row in pnl_rows),
            open_pnl_usd=sum(row.ledger.open_pnl_usd for row in pnl_rows),
            total_pnl_usd=sum(row.ledger.total_pnl_usd for row in pnl_rows),
            closed_event_count=sum(row.ledger.closed_event_count for row in pnl_rows),
            win_rate=0.0,
            profit_factor=0.0,
            final_position_units=0.0,
            portfolio_final_equity=result_metrics.final_equity,
            portfolio_return_pct=result_metrics.total_return_pct,
            portfolio_max_drawdown_pct=result_metrics.max_drawdown_pct,
            portfolio_sharpe_ratio=result_metrics.sharpe_ratio,
            portfolio_risk_score=risk.score,
        )
    )
    return tuple(rows)


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
