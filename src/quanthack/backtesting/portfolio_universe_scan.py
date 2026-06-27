from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path

from quanthack.backtesting.competition_score import (
    CompetitionMetrics,
    RiskDisciplineReport,
    official_composite_score,
)
from quanthack.backtesting.portfolio_backtest import PortfolioBacktestResult
from quanthack.backtesting.portfolio_strategy_compare import compare_portfolio_strategies
from quanthack.core.config import AppConfig
from quanthack.core.instruments import AssetClass, enabled_symbols, instrument_for
from quanthack.market.market_data import PriceHistory, QuoteHistory
from quanthack.strategies.strategy import STRATEGY_NAMES, normalize_strategy_name


DEFAULT_MIN_SYMBOLS = 3
DEFAULT_MAX_SYMBOLS = 5
DEFAULT_MAX_BASKETS = 25


@dataclass(frozen=True)
class UniverseBasket:
    name: str
    symbols: tuple[str, ...]

    def __post_init__(self) -> None:
        name = self.name.strip()
        if not name:
            raise ValueError("universe basket name is required")

        normalized: list[str] = []
        seen: set[str] = set()
        for raw_symbol in self.symbols:
            symbol = instrument_for(raw_symbol).symbol
            if symbol in seen:
                continue
            normalized.append(symbol)
            seen.add(symbol)

        if not normalized:
            raise ValueError("universe basket needs at least one symbol")

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "symbols", tuple(normalized))


DEFAULT_UNIVERSE_BASKETS: tuple[UniverseBasket, ...] = (
    UniverseBasket("core_fx", ("EURUSD", "GBPUSD", "USDJPY")),
    UniverseBasket("major_fx", ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD")),
    UniverseBasket("safe_fx_metals", ("EURUSD", "USDJPY", "XAUUSD", "XAGUSD")),
    UniverseBasket("balanced_fx_crypto", ("EURUSD", "USDJPY", "BTCUSD", "ETHUSD")),
    UniverseBasket("broad_balanced", ("EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "BTCUSD")),
)


@dataclass(frozen=True)
class PortfolioUniverseScanRow:
    basket: UniverseBasket
    strategy_name: str
    result: PortfolioBacktestResult
    competition_metrics: CompetitionMetrics
    risk_discipline: RiskDisciplineReport
    return_rank: float = 0.0
    drawdown_rank: float = 0.0
    sharpe_rank: float = 0.0
    proxy_score: float = 0.0

    @property
    def rank_key(self) -> tuple[float, int, float, float, float, int]:
        return (
            self.proxy_score,
            self.risk_discipline.score,
            self.competition_metrics.return_pct,
            self.competition_metrics.sharpe_15m,
            -self.competition_metrics.max_drawdown_pct,
            len(self.basket.symbols),
        )

    @property
    def asset_mix(self) -> str:
        counts = Counter(
            instrument_for(symbol).asset_class.value for symbol in self.basket.symbols
        )
        return ";".join(
            f"{asset_class.value}={counts.get(asset_class.value, 0)}"
            for asset_class in AssetClass
            if counts.get(asset_class.value, 0)
        )

    @property
    def trimmed_allocation_periods(self) -> int:
        return len(
            [
                allocation
                for allocation in self.result.allocation_reports
                if allocation.trimmed_targets
            ]
        )

    @property
    def warning_allocation_periods(self) -> int:
        return len(
            [
                allocation
                for allocation in self.result.allocation_reports
                if allocation.estimated_risk_status != "OK"
            ]
        )

    @property
    def worst_leverage(self) -> float:
        return max(
            (allocation.leverage for allocation in self.result.allocation_reports),
            default=0.0,
        )

    @property
    def worst_largest_symbol_concentration(self) -> float:
        return max(
            (
                allocation.largest_symbol_concentration
                for allocation in self.result.allocation_reports
            ),
            default=0.0,
        )

    @property
    def worst_net_directional_exposure(self) -> float:
        return max(
            (
                allocation.net_directional_exposure
                for allocation in self.result.allocation_reports
            ),
            default=0.0,
        )


@dataclass(frozen=True)
class PortfolioUniverseScan:
    available_symbols: tuple[str, ...]
    strategies: tuple[str, ...]
    baskets: tuple[UniverseBasket, ...]
    rows: tuple[PortfolioUniverseScanRow, ...]

    @property
    def best(self) -> PortfolioUniverseScanRow | None:
        if not self.rows:
            return None
        return self.rows[0]


def scan_portfolio_universes(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    strategy_names: tuple[str, ...] = ("alpha_router",),
    baskets: tuple[UniverseBasket, ...] | None = None,
    min_symbols: int = DEFAULT_MIN_SYMBOLS,
    max_symbols: int = DEFAULT_MAX_SYMBOLS,
    max_baskets: int = DEFAULT_MAX_BASKETS,
) -> PortfolioUniverseScan:
    if min_symbols < 1:
        raise ValueError("min_symbols must be at least 1")
    if max_symbols < min_symbols:
        raise ValueError("max_symbols must be at least min_symbols")
    if max_baskets < 1:
        raise ValueError("max_baskets must be at least 1")

    available_symbols = _available_supported_symbols(prices, quotes)
    selected_strategies = _normalize_unique_strategy_names(strategy_names)
    selected_baskets = _select_baskets(
        available_symbols=available_symbols,
        baskets=baskets,
        min_symbols=min_symbols,
        max_symbols=max_symbols,
        max_baskets=max_baskets,
    )

    rows: list[PortfolioUniverseScanRow] = []
    for basket in selected_baskets:
        comparison = compare_portfolio_strategies(
            config=config,
            prices=prices,
            quotes=quotes,
            strategy_names=selected_strategies,
            symbols=basket.symbols,
        )
        for comparison_row in comparison.rows:
            rows.append(
                PortfolioUniverseScanRow(
                    basket=basket,
                    strategy_name=comparison_row.strategy_name,
                    result=comparison_row.result,
                    competition_metrics=comparison_row.competition_metrics,
                    risk_discipline=comparison_row.risk_discipline,
                )
            )

    ranked = _attach_proxy_scores(tuple(rows))
    ranked = tuple(sorted(ranked, key=lambda row: row.rank_key, reverse=True))
    return PortfolioUniverseScan(
        available_symbols=available_symbols,
        strategies=selected_strategies,
        baskets=selected_baskets,
        rows=ranked,
    )


def write_portfolio_universe_scan_csv(
    scan: PortfolioUniverseScan,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "basket",
                "strategy",
                "symbols",
                "asset_mix",
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
                "trimmed_allocation_periods",
                "warning_allocation_periods",
                "worst_leverage",
                "worst_largest_symbol_concentration",
                "worst_net_directional_exposure",
            ],
        )
        writer.writeheader()
        for rank, row in enumerate(scan.rows, start=1):
            metrics = row.competition_metrics
            writer.writerow(
                {
                    "rank": rank,
                    "basket": row.basket.name,
                    "strategy": row.strategy_name,
                    "symbols": " ".join(row.basket.symbols),
                    "asset_mix": row.asset_mix,
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
                    "trimmed_allocation_periods": row.trimmed_allocation_periods,
                    "warning_allocation_periods": row.warning_allocation_periods,
                    "worst_leverage": row.worst_leverage,
                    "worst_largest_symbol_concentration": (
                        row.worst_largest_symbol_concentration
                    ),
                    "worst_net_directional_exposure": row.worst_net_directional_exposure,
                }
            )


def _available_supported_symbols(
    prices: PriceHistory,
    quotes: QuoteHistory,
) -> tuple[str, ...]:
    raw_available = set(prices.symbols()) & set(quotes.symbols())
    ordered: list[str] = []
    for symbol in enabled_symbols():
        if symbol in raw_available:
            ordered.append(symbol)
    if not ordered:
        raise ValueError("no supported symbols found in both price and quote data")
    return tuple(ordered)


def _select_baskets(
    *,
    available_symbols: tuple[str, ...],
    baskets: tuple[UniverseBasket, ...] | None,
    min_symbols: int,
    max_symbols: int,
    max_baskets: int,
) -> tuple[UniverseBasket, ...]:
    if baskets is not None:
        normalized = tuple(_validate_custom_basket(basket, available_symbols) for basket in baskets)
        return _filter_by_size(normalized, min_symbols=min_symbols, max_symbols=max_symbols)

    selected: list[UniverseBasket] = []
    seen: set[tuple[str, ...]] = set()

    for basket in DEFAULT_UNIVERSE_BASKETS:
        if all(symbol in available_symbols for symbol in basket.symbols):
            _append_unique_basket(selected, seen, basket)

    if min_symbols <= len(available_symbols) <= max_symbols:
        _append_unique_basket(
            selected,
            seen,
            UniverseBasket("all_available", available_symbols),
        )

    combo_baskets = [
        UniverseBasket(_basket_name(symbols), symbols)
        for size in range(min_symbols, min(max_symbols, len(available_symbols)) + 1)
        for symbols in combinations(available_symbols, size)
    ]
    combo_baskets.sort(key=_basket_priority, reverse=True)
    for basket in combo_baskets:
        if len(selected) >= max_baskets:
            break
        _append_unique_basket(selected, seen, basket)

    filtered = _filter_by_size(
        tuple(selected),
        min_symbols=min_symbols,
        max_symbols=max_symbols,
    )
    if not filtered:
        raise ValueError(
            f"need at least {min_symbols} supported symbols with aligned price/quote data"
        )
    return filtered[:max_baskets]


def _validate_custom_basket(
    basket: UniverseBasket,
    available_symbols: tuple[str, ...],
) -> UniverseBasket:
    missing = tuple(symbol for symbol in basket.symbols if symbol not in available_symbols)
    if missing:
        raise ValueError(
            f"basket {basket.name!r} includes symbols missing from data: "
            f"{', '.join(missing)}"
        )
    return basket


def _filter_by_size(
    baskets: tuple[UniverseBasket, ...],
    *,
    min_symbols: int,
    max_symbols: int,
) -> tuple[UniverseBasket, ...]:
    filtered = tuple(
        basket
        for basket in baskets
        if min_symbols <= len(basket.symbols) <= max_symbols
    )
    if not filtered:
        raise ValueError(
            f"no universe baskets have between {min_symbols} and {max_symbols} symbols"
        )
    return filtered


def _append_unique_basket(
    selected: list[UniverseBasket],
    seen: set[tuple[str, ...]],
    basket: UniverseBasket,
) -> None:
    key = tuple(sorted(basket.symbols))
    if key in seen:
        return
    selected.append(basket)
    seen.add(key)


def _basket_priority(basket: UniverseBasket) -> tuple[int, int, int, int, float]:
    classes = {instrument_for(symbol).asset_class for symbol in basket.symbols}
    has_metal = int(AssetClass.METAL in classes)
    has_crypto = int(AssetClass.CRYPTO in classes)
    known_order = {symbol: index for index, symbol in enumerate(enabled_symbols())}
    average_rank = sum(known_order[symbol] for symbol in basket.symbols) / len(
        basket.symbols
    )
    return (
        len(classes),
        has_metal,
        has_crypto,
        len(basket.symbols),
        -average_rank,
    )


def _basket_name(symbols: tuple[str, ...]) -> str:
    return "basket_" + "_".join(symbol.lower() for symbol in symbols)


def _normalize_unique_strategy_names(strategy_names: tuple[str, ...]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_name in strategy_names:
        strategy_name = normalize_strategy_name(raw_name)
        if strategy_name not in STRATEGY_NAMES:
            raise ValueError(f"unsupported strategy {raw_name!r}")
        if strategy_name in seen:
            continue
        normalized.append(strategy_name)
        seen.add(strategy_name)
    if not normalized:
        raise ValueError("at least one strategy is required")
    return tuple(normalized)


def _attach_proxy_scores(
    rows: tuple[PortfolioUniverseScanRow, ...],
) -> tuple[PortfolioUniverseScanRow, ...]:
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

    ranked_rows: list[PortfolioUniverseScanRow] = []
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
