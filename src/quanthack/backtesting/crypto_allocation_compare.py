from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from itertools import product
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
from quanthack.backtesting.portfolio_backtest import (
    PortfolioBacktestEngine,
    PortfolioBacktestResult,
)
from quanthack.backtesting.portfolio_fixed_warmup_walk_forward import (
    FixedWarmupPortfolioWalkForwardResult,
    FixedWarmupPromotionDecision,
    decide_fixed_warmup_promotion,
    run_fixed_warmup_portfolio_walk_forward,
)
from quanthack.core.config import AppConfig
from quanthack.core.instruments import AssetClass, instrument_for, instruments_by_asset_class
from quanthack.market.market_data import PriceHistory, QuoteHistory
from quanthack.strategies.strategy import normalize_strategy_name


DEFAULT_CRYPTO_ALLOCATION_STRATEGIES = (
    "macd_momentum",
    "crypto_mean_reversion",
)


@dataclass(frozen=True)
class CryptoAllocationCandidate:
    label: str
    strategy_by_symbol: tuple[tuple[str, str], ...]
    result: PortfolioBacktestResult
    competition_metrics: CompetitionMetrics
    risk_discipline: RiskDisciplineReport
    proxy_score: float = 0.0
    selection_score: float = 0.0
    walk_forward: FixedWarmupPortfolioWalkForwardResult | None = None
    promotion: FixedWarmupPromotionDecision | None = None

    @property
    def strategy_map_text(self) -> str:
        return " ".join(
            f"{symbol}={strategy}" for symbol, strategy in self.strategy_by_symbol
        )

    @property
    def strategy_counts_text(self) -> str:
        counts: dict[str, int] = {}
        for _, strategy in self.strategy_by_symbol:
            counts[strategy] = counts.get(strategy, 0) + 1
        return " ".join(f"{strategy}:{count}" for strategy, count in sorted(counts.items()))

    @property
    def rank_key(self) -> tuple[float, float, int, float, float]:
        return (
            self.selection_score,
            self.proxy_score,
            self.risk_discipline.score,
            self.competition_metrics.return_pct,
            -self.competition_metrics.max_drawdown_pct,
        )


@dataclass(frozen=True)
class CryptoAllocationComparison:
    symbols: tuple[str, ...]
    strategy_names: tuple[str, ...]
    candidates: tuple[CryptoAllocationCandidate, ...]
    walk_forward_enabled: bool

    @property
    def best(self) -> CryptoAllocationCandidate | None:
        if not self.candidates:
            return None
        return self.candidates[0]


def compare_crypto_allocations(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    strategy_names: tuple[str, ...] = DEFAULT_CRYPTO_ALLOCATION_STRATEGIES,
    symbols: tuple[str, ...] | None = None,
    run_walk_forward: bool = True,
    train_size: int = 480,
    test_size: int = 192,
    step_size: int = 192,
    max_maps: int = 128,
) -> CryptoAllocationComparison:
    selected_symbols = _selected_crypto_symbols(
        prices=prices,
        quotes=quotes,
        symbols=symbols,
    )
    normalized_strategies = _normalize_unique_strategy_names(strategy_names)
    if max_maps < 1:
        raise ValueError("max_maps must be at least 1")

    strategy_maps = _candidate_strategy_maps(
        symbols=selected_symbols,
        strategy_names=normalized_strategies,
    )
    if len(strategy_maps) > max_maps:
        raise ValueError(
            f"crypto allocation comparison would produce {len(strategy_maps)} maps; "
            f"increase max_maps above {max_maps}"
        )

    candidates: list[CryptoAllocationCandidate] = []
    for strategy_by_symbol in strategy_maps:
        result = _run_strategy_map(
            config=config,
            prices=prices,
            quotes=quotes,
            strategy_by_symbol=strategy_by_symbol,
        )
        metrics = build_competition_metrics(
            equity_points=result.equity_curve,
            fills=result.fills,
        )
        risk_discipline = build_risk_discipline_report(
            risk_samples_from_portfolio_equity(result.equity_curve)
        )
        walk_forward: FixedWarmupPortfolioWalkForwardResult | None = None
        promotion: FixedWarmupPromotionDecision | None = None
        if run_walk_forward:
            walk_forward = run_fixed_warmup_portfolio_walk_forward(
                config=config,
                prices=prices,
                quotes=quotes,
                strategy_name=next(iter(strategy_by_symbol.values())),
                symbols=tuple(strategy_by_symbol),
                strategy_by_symbol=strategy_by_symbol,
                train_size=train_size,
                test_size=test_size,
                step_size=step_size,
            )
            promotion = decide_fixed_warmup_promotion(walk_forward)
        strategy_map = tuple(sorted(strategy_by_symbol.items()))
        candidates.append(
            CryptoAllocationCandidate(
                label=_candidate_label(strategy_map),
                strategy_by_symbol=strategy_map,
                result=result,
                competition_metrics=metrics,
                risk_discipline=risk_discipline,
                walk_forward=walk_forward,
                promotion=promotion,
            )
        )

    scored_candidates = _attach_scores(tuple(candidates))
    return CryptoAllocationComparison(
        symbols=selected_symbols,
        strategy_names=normalized_strategies,
        candidates=tuple(
            sorted(scored_candidates, key=lambda candidate: candidate.rank_key, reverse=True)
        ),
        walk_forward_enabled=run_walk_forward,
    )


def write_crypto_allocation_comparison_csv(
    comparison: CryptoAllocationComparison,
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
                "strategies",
                "strategy_counts",
                "strategy_map",
                "selection_score",
                "proxy_score",
                "return_pct",
                "max_drawdown_pct",
                "sharpe_15m",
                "risk_discipline_score",
                "trade_count",
                "fills",
                "turnover_notional",
                "total_pnl_usd",
                "promotion_status",
                "promotion_live_ready",
                "promotion_reason",
                "wf_folds",
                "wf_positive_fold_fraction",
                "wf_active_fold_fraction",
                "wf_active_positive_fold_fraction",
                "wf_non_negative_fold_fraction",
                "wf_median_test_return_pct",
                "wf_median_active_test_return_pct",
                "wf_median_test_sharpe_15m",
                "wf_worst_test_drawdown_pct",
                "wf_average_risk_discipline_score",
                "wf_total_evaluation_fills",
                "wf_largest_positive_fold_contribution",
            ],
        )
        writer.writeheader()
        for rank, candidate in enumerate(comparison.candidates, start=1):
            metrics = candidate.competition_metrics
            writer.writerow(
                {
                    "rank": rank,
                    "label": candidate.label,
                    "symbols": " ".join(comparison.symbols),
                    "strategies": " ".join(comparison.strategy_names),
                    "strategy_counts": candidate.strategy_counts_text,
                    "strategy_map": candidate.strategy_map_text,
                    "selection_score": candidate.selection_score,
                    "proxy_score": candidate.proxy_score,
                    "return_pct": metrics.return_pct,
                    "max_drawdown_pct": metrics.max_drawdown_pct,
                    "sharpe_15m": metrics.sharpe_15m,
                    "risk_discipline_score": candidate.risk_discipline.score,
                    "trade_count": metrics.trade_count,
                    "fills": len(candidate.result.fills),
                    "turnover_notional": candidate.result.metrics.turnover_notional,
                    "total_pnl_usd": candidate.result.total_pnl_usd,
                    "promotion_status": (
                        candidate.promotion.status if candidate.promotion else ""
                    ),
                    "promotion_live_ready": (
                        candidate.promotion.live_ready if candidate.promotion else ""
                    ),
                    "promotion_reason": (
                        candidate.promotion.reason if candidate.promotion else ""
                    ),
                    **_walk_forward_columns(candidate.walk_forward),
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


def _attach_scores(
    candidates: tuple[CryptoAllocationCandidate, ...],
) -> tuple[CryptoAllocationCandidate, ...]:
    return_ranks = _percentile_scores(
        [candidate.competition_metrics.return_pct for candidate in candidates],
        higher_is_better=True,
    )
    drawdown_ranks = _percentile_scores(
        [candidate.competition_metrics.max_drawdown_pct for candidate in candidates],
        higher_is_better=False,
    )
    sharpe_ranks = _percentile_scores(
        [candidate.competition_metrics.sharpe_15m for candidate in candidates],
        higher_is_better=True,
    )

    scored: list[CryptoAllocationCandidate] = []
    for candidate, return_rank, drawdown_rank, sharpe_rank in zip(
        candidates,
        return_ranks,
        drawdown_ranks,
        sharpe_ranks,
        strict=True,
    ):
        proxy_score = official_composite_score(
            return_rank=return_rank,
            drawdown_rank=drawdown_rank,
            sharpe_rank=sharpe_rank,
            risk_discipline_score=candidate.risk_discipline.score,
            sharpe_rank_cap=candidate.competition_metrics.sharpe_rank_cap,
        )
        selection_score = _selection_score(
            proxy_score=proxy_score,
            walk_forward=candidate.walk_forward,
            promotion=candidate.promotion,
        )
        scored.append(
            replace(
                candidate,
                proxy_score=proxy_score,
                selection_score=selection_score,
            )
        )
    return tuple(scored)


def _selection_score(
    *,
    proxy_score: float,
    walk_forward: FixedWarmupPortfolioWalkForwardResult | None,
    promotion: FixedWarmupPromotionDecision | None,
) -> float:
    if walk_forward is None:
        return proxy_score
    risk_score = min(max(walk_forward.average_risk_discipline_score, 0.0), 100.0) / 100.0
    fold_stability = (
        0.30 * walk_forward.non_negative_fold_fraction
        + 0.25 * walk_forward.active_positive_fold_fraction
        + 0.15 * walk_forward.positive_fold_fraction
        + 0.15 * walk_forward.active_fold_fraction
        + 0.15 * risk_score
    )
    concentration_penalty = max(
        0.0,
        walk_forward.largest_positive_fold_contribution - 0.80,
    ) * 25.0
    promotion_bonus = 5.0 if promotion and promotion.live_ready else 0.0
    return 0.60 * proxy_score + 40.0 * fold_stability + promotion_bonus - concentration_penalty


def _candidate_strategy_maps(
    *,
    symbols: tuple[str, ...],
    strategy_names: tuple[str, ...],
) -> tuple[dict[str, str], ...]:
    maps: list[dict[str, str]] = []
    for assignments in product(strategy_names, repeat=len(symbols)):
        maps.append(
            {
                symbol: strategy_name
                for symbol, strategy_name in zip(symbols, assignments, strict=True)
            }
        )
    return tuple(maps)


def _candidate_label(strategy_map: tuple[tuple[str, str], ...]) -> str:
    counts: dict[str, int] = {}
    for _, strategy_name in strategy_map:
        counts[strategy_name] = counts.get(strategy_name, 0) + 1
    if len(counts) == 1:
        return f"all_{next(iter(counts))}"
    return "mix_" + "_".join(
        f"{strategy}_{count}" for strategy, count in sorted(counts.items())
    )


def _walk_forward_columns(
    walk_forward: FixedWarmupPortfolioWalkForwardResult | None,
) -> dict[str, float | int | str]:
    if walk_forward is None:
        return {
            "wf_folds": "",
            "wf_positive_fold_fraction": "",
            "wf_active_fold_fraction": "",
            "wf_active_positive_fold_fraction": "",
            "wf_non_negative_fold_fraction": "",
            "wf_median_test_return_pct": "",
            "wf_median_active_test_return_pct": "",
            "wf_median_test_sharpe_15m": "",
            "wf_worst_test_drawdown_pct": "",
            "wf_average_risk_discipline_score": "",
            "wf_total_evaluation_fills": "",
            "wf_largest_positive_fold_contribution": "",
        }
    return {
        "wf_folds": len(walk_forward.folds),
        "wf_positive_fold_fraction": walk_forward.positive_fold_fraction,
        "wf_active_fold_fraction": walk_forward.active_fold_fraction,
        "wf_active_positive_fold_fraction": (
            walk_forward.active_positive_fold_fraction
        ),
        "wf_non_negative_fold_fraction": walk_forward.non_negative_fold_fraction,
        "wf_median_test_return_pct": walk_forward.median_test_return_pct,
        "wf_median_active_test_return_pct": (
            walk_forward.median_active_test_return_pct
        ),
        "wf_median_test_sharpe_15m": walk_forward.median_test_sharpe_15m,
        "wf_worst_test_drawdown_pct": walk_forward.worst_test_drawdown_pct,
        "wf_average_risk_discipline_score": (
            walk_forward.average_risk_discipline_score
        ),
        "wf_total_evaluation_fills": walk_forward.total_evaluation_fills,
        "wf_largest_positive_fold_contribution": (
            walk_forward.largest_positive_fold_contribution
        ),
    }


def _selected_crypto_symbols(
    *,
    prices: PriceHistory,
    quotes: QuoteHistory,
    symbols: tuple[str, ...] | None,
) -> tuple[str, ...]:
    if symbols:
        selected = tuple(_normalize_crypto_symbol(symbol) for symbol in symbols)
    else:
        crypto_symbols = {
            instrument.symbol
            for instrument in instruments_by_asset_class(AssetClass.CRYPTO)
        }
        selected = tuple(
            sorted(set(prices.symbols()) & set(quotes.symbols()) & crypto_symbols)
        )
    if not selected:
        raise ValueError("no crypto symbols found in both price and quote data")
    missing_prices = sorted(set(selected) - set(prices.symbols()))
    missing_quotes = sorted(set(selected) - set(quotes.symbols()))
    if missing_prices:
        raise ValueError(f"crypto symbols missing price data: {', '.join(missing_prices)}")
    if missing_quotes:
        raise ValueError(f"crypto symbols missing quote data: {', '.join(missing_quotes)}")
    return selected


def _normalize_crypto_symbol(symbol: str) -> str:
    instrument = instrument_for(symbol)
    if instrument.asset_class != AssetClass.CRYPTO:
        raise ValueError(
            f"{instrument.symbol} is {instrument.asset_class.value}, not CRYPTO"
        )
    return instrument.symbol


def _normalize_unique_strategy_names(strategy_names: tuple[str, ...]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_name in strategy_names:
        strategy_name = normalize_strategy_name(raw_name)
        if strategy_name in seen:
            continue
        normalized.append(strategy_name)
        seen.add(strategy_name)
    if not normalized:
        raise ValueError("at least one strategy is required")
    return tuple(normalized)


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
