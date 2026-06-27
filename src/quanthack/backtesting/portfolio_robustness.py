from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from quanthack.backtesting.backtest import FillModel
from quanthack.backtesting.competition_score import (
    CompetitionMetrics,
    RiskDisciplineReport,
    build_competition_metrics,
    build_risk_discipline_report,
    risk_samples_from_portfolio_equity,
)
from quanthack.backtesting.portfolio_backtest import PortfolioBacktestEngine
from quanthack.core.config import AppConfig
from quanthack.core.instruments import instrument_for
from quanthack.market.market_data import PriceHistory, QuoteHistory


@dataclass(frozen=True)
class PortfolioRobustnessRow:
    scenario: str
    excluded_symbol: str
    symbols: str
    symbol_count: int
    return_pct: float
    return_delta_pct: float
    max_drawdown_pct: float
    drawdown_delta_pct: float
    sharpe_15m: float
    sharpe_delta: float
    trade_count: int
    risk_discipline_score: int
    compliance_review_required: bool
    final_equity: float
    total_pnl_usd: float
    fragility_note: str


def evaluate_leave_one_symbol_out(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    symbols: tuple[str, ...],
    strategy_name: str,
    strategy_by_symbol: dict[str, str],
    clock_open_at: datetime | None = None,
) -> tuple[PortfolioRobustnessRow, ...]:
    if len(symbols) < 2:
        raise ValueError("leave-one-symbol-out robustness requires at least two symbols")

    normalized_symbols = tuple(instrument_for(symbol).symbol for symbol in symbols)
    baseline = _run_scenario(
        config=config,
        prices=prices,
        quotes=quotes,
        symbols=normalized_symbols,
        strategy_name=strategy_name,
        strategy_by_symbol=strategy_by_symbol,
        clock_open_at=clock_open_at,
    )
    baseline_metrics, baseline_risk, baseline_pnl = baseline
    rows = [
        _row(
            scenario="baseline",
            excluded_symbol="",
            symbols=normalized_symbols,
            metrics=baseline_metrics,
            risk=baseline_risk,
            total_pnl_usd=baseline_pnl,
            baseline_metrics=baseline_metrics,
        )
    ]

    for excluded_symbol in normalized_symbols:
        kept_symbols = tuple(symbol for symbol in normalized_symbols if symbol != excluded_symbol)
        metrics, risk, total_pnl = _run_scenario(
            config=config,
            prices=prices,
            quotes=quotes,
            symbols=kept_symbols,
            strategy_name=strategy_name,
            strategy_by_symbol=strategy_by_symbol,
            clock_open_at=clock_open_at,
        )
        rows.append(
            _row(
                scenario=f"exclude_{excluded_symbol}",
                excluded_symbol=excluded_symbol,
                symbols=kept_symbols,
                metrics=metrics,
                risk=risk,
                total_pnl_usd=total_pnl,
                baseline_metrics=baseline_metrics,
            )
        )

    return tuple(rows)


def write_portfolio_robustness_csv(
    rows: tuple[PortfolioRobustnessRow, ...],
    output: str | Path,
) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "scenario",
                "excluded_symbol",
                "symbols",
                "symbol_count",
                "return_pct",
                "return_delta_pct",
                "max_drawdown_pct",
                "drawdown_delta_pct",
                "sharpe_15m",
                "sharpe_delta",
                "trade_count",
                "risk_discipline_score",
                "compliance_review_required",
                "final_equity",
                "total_pnl_usd",
                "fragility_note",
            ),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _run_scenario(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    symbols: tuple[str, ...],
    strategy_name: str,
    strategy_by_symbol: dict[str, str],
    clock_open_at: datetime | None,
) -> tuple[CompetitionMetrics, RiskDisciplineReport, float]:
    clock = config.competition.to_clock()
    if clock_open_at is not None:
        clock = replace(clock, open_at=clock_open_at)

    engine = PortfolioBacktestEngine(
        strategies={
            symbol: config.build_strategy(
                strategy_by_symbol.get(symbol, strategy_name),
                symbol=symbol,
            )
            for symbol in symbols
        },
        risk_limits=config.risk,
        quality_limits=config.market_quality,
        quality_limits_by_symbol={
            symbol: replace(
                config.market_quality,
                max_spread_bps=instrument_for(symbol).max_spread_bps,
            )
            for symbol in symbols
        },
        clock=clock,
        fill_model=FillModel(slippage_bps=config.backtest.slippage_bps),
        periods_per_year=config.backtest.periods_per_year,
    )
    result = engine.run(
        prices=prices,
        quotes=quotes,
        starting_equity=config.competition.starting_equity,
    )
    metrics = build_competition_metrics(
        equity_points=result.equity_curve,
        fills=result.fills,
    )
    risk = build_risk_discipline_report(
        risk_samples_from_portfolio_equity(result.equity_curve)
    )
    return metrics, risk, result.total_pnl_usd


def _row(
    *,
    scenario: str,
    excluded_symbol: str,
    symbols: tuple[str, ...],
    metrics: CompetitionMetrics,
    risk: RiskDisciplineReport,
    total_pnl_usd: float,
    baseline_metrics: CompetitionMetrics,
) -> PortfolioRobustnessRow:
    return_delta = metrics.return_pct - baseline_metrics.return_pct
    drawdown_delta = metrics.max_drawdown_pct - baseline_metrics.max_drawdown_pct
    sharpe_delta = metrics.sharpe_15m - baseline_metrics.sharpe_15m
    return PortfolioRobustnessRow(
        scenario=scenario,
        excluded_symbol=excluded_symbol,
        symbols=";".join(symbols),
        symbol_count=len(symbols),
        return_pct=metrics.return_pct,
        return_delta_pct=return_delta,
        max_drawdown_pct=metrics.max_drawdown_pct,
        drawdown_delta_pct=drawdown_delta,
        sharpe_15m=metrics.sharpe_15m,
        sharpe_delta=sharpe_delta,
        trade_count=metrics.trade_count,
        risk_discipline_score=risk.score,
        compliance_review_required=risk.compliance_review_required,
        final_equity=metrics.final_equity,
        total_pnl_usd=total_pnl_usd,
        fragility_note=_fragility_note(
            excluded_symbol=excluded_symbol,
            return_delta=return_delta,
            baseline_return=baseline_metrics.return_pct,
            trade_count=metrics.trade_count,
        ),
    )


def _fragility_note(
    *,
    excluded_symbol: str,
    return_delta: float,
    baseline_return: float,
    trade_count: int,
) -> str:
    if not excluded_symbol:
        return "baseline"
    if baseline_return > 0 and return_delta <= -0.5 * baseline_return:
        return "fragile: exclusion removes at least half of baseline return"
    if baseline_return > 0 and return_delta < 0:
        return "weaker without excluded symbol"
    if trade_count < 30:
        return "under 30 trades; Sharpe prize ineligible"
    return "robust or improved"
