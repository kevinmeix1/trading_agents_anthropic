from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from quanthack.backtesting.backtest import BacktestEngine, BacktestResult, FillModel
from quanthack.core.config import AppConfig
from quanthack.market.market_data import PriceHistory, QuoteHistory
from quanthack.strategies.strategy import STRATEGY_NAMES, normalize_strategy_name


@dataclass(frozen=True)
class StrategyComparisonRow:
    strategy_name: str
    result: BacktestResult

    @property
    def rank_key(self) -> tuple[float, float, float]:
        return (
            self.result.metrics.sharpe_ratio,
            self.result.metrics.total_return_pct,
            -self.result.metrics.max_drawdown_pct,
        )


@dataclass(frozen=True)
class StrategyComparison:
    rows: tuple[StrategyComparisonRow, ...]

    @property
    def best(self) -> StrategyComparisonRow | None:
        if not self.rows:
            return None
        return self.rows[0]


def compare_strategies(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    strategy_names: tuple[str, ...] = STRATEGY_NAMES,
    symbol: str | None = None,
) -> StrategyComparison:
    rows: list[StrategyComparisonRow] = []

    for strategy_name in _normalize_unique_strategy_names(strategy_names):
        strategy_symbol = symbol or config.strategy_symbol(strategy_name)
        engine = BacktestEngine(
            strategy=config.build_strategy(strategy_name, symbol=strategy_symbol),
            risk_limits=config.risk,
            quality_limits=config.market_quality,
            clock=config.competition.to_clock(),
            fill_model=FillModel(slippage_bps=config.backtest.slippage_bps),
            periods_per_year=config.backtest.periods_per_year,
        )
        result = engine.run(
            prices=prices,
            quotes=quotes,
            symbol=strategy_symbol,
            starting_equity=config.competition.starting_equity,
        )
        rows.append(StrategyComparisonRow(strategy_name=strategy_name, result=result))

    rows.sort(key=lambda row: row.rank_key, reverse=True)
    return StrategyComparison(tuple(rows))


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


def write_strategy_comparison_csv(
    comparison: StrategyComparison,
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
                "final_equity",
                "total_return_pct",
                "sharpe_ratio",
                "max_drawdown_pct",
                "win_rate",
                "profit_factor",
                "fills",
                "turnover_notional",
            ],
        )
        writer.writeheader()
        for rank, row in enumerate(comparison.rows, start=1):
            metrics = row.result.metrics
            writer.writerow(
                {
                    "rank": rank,
                    "strategy": row.strategy_name,
                    "symbol": row.result.symbol,
                    "final_equity": metrics.final_equity,
                    "total_return_pct": metrics.total_return_pct,
                    "sharpe_ratio": metrics.sharpe_ratio,
                    "max_drawdown_pct": metrics.max_drawdown_pct,
                    "win_rate": metrics.win_rate,
                    "profit_factor": metrics.profit_factor,
                    "fills": len(row.result.fills),
                    "turnover_notional": metrics.turnover_notional,
                }
            )
