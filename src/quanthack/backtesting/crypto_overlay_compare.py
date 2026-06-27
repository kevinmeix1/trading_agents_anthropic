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
from quanthack.core.instruments import AssetClass, instrument_for
from quanthack.market.market_data import PriceHistory, QuoteHistory
from quanthack.strategies.strategy import normalize_strategy_name


ROBUST_CRYPTO_OVERLAY = {
    "BARUSD": "crypto_mean_reversion",
    "BTCUSD": "crypto_mean_reversion",
    "ETHUSD": "crypto_mean_reversion",
    "SOLUSD": "macd_momentum",
    "XRPUSD": "crypto_mean_reversion",
}
AGGRESSIVE_CRYPTO_OVERLAY = {
    "BARUSD": "crypto_mean_reversion",
    "BTCUSD": "macd_momentum",
    "ETHUSD": "crypto_mean_reversion",
    "SOLUSD": "macd_momentum",
    "XRPUSD": "crypto_mean_reversion",
}


@dataclass(frozen=True)
class CryptoOverlayCandidate:
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
    def symbols(self) -> tuple[str, ...]:
        return tuple(symbol for symbol, _ in self.strategy_by_symbol)

    @property
    def strategy_map_text(self) -> str:
        return " ".join(
            f"{symbol}={strategy}" for symbol, strategy in self.strategy_by_symbol
        )

    @property
    def crypto_map_text(self) -> str:
        return " ".join(
            f"{symbol}={strategy}"
            for symbol, strategy in self.strategy_by_symbol
            if instrument_for(symbol).asset_class == AssetClass.CRYPTO
        )

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
class CryptoOverlayComparison:
    official_symbols: tuple[str, ...]
    crypto_symbols: tuple[str, ...]
    candidates: tuple[CryptoOverlayCandidate, ...]
    base_strategy: str
    walk_forward_enabled: bool

    @property
    def best(self) -> CryptoOverlayCandidate | None:
        if not self.candidates:
            return None
        return self.candidates[0]


def compare_crypto_overlays(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    base_strategy: str = "macd_momentum",
    symbols: tuple[str, ...] | None = None,
    run_walk_forward: bool = True,
    train_size: int = 96,
    test_size: int = 48,
    step_size: int = 48,
) -> CryptoOverlayComparison:
    selected_symbols = _selected_symbols(prices=prices, quotes=quotes, symbols=symbols)
    official_symbols = tuple(
        symbol
        for symbol in selected_symbols
        if instrument_for(symbol).asset_class != AssetClass.CRYPTO
    )
    crypto_symbols = tuple(
        symbol
        for symbol in selected_symbols
        if instrument_for(symbol).asset_class == AssetClass.CRYPTO
    )
    normalized_base = normalize_strategy_name(base_strategy)
    specs = _candidate_maps(
        official_symbols=official_symbols,
        crypto_symbols=crypto_symbols,
        base_strategy=normalized_base,
    )

    candidates: list[CryptoOverlayCandidate] = []
    seen_maps: set[tuple[tuple[str, str], ...]] = set()
    for label, strategy_by_symbol in specs:
        strategy_map = tuple(sorted(strategy_by_symbol.items()))
        if not strategy_map or strategy_map in seen_maps:
            continue
        seen_maps.add(strategy_map)

        result = _run_strategy_map(
            config=config,
            prices=prices,
            quotes=quotes,
            strategy_by_symbol=dict(strategy_map),
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
                strategy_name=next(iter(dict(strategy_map).values())),
                symbols=tuple(symbol for symbol, _ in strategy_map),
                strategy_by_symbol=dict(strategy_map),
                train_size=train_size,
                test_size=test_size,
                step_size=step_size,
            )
            promotion = decide_fixed_warmup_promotion(walk_forward)
        candidates.append(
            CryptoOverlayCandidate(
                label=label,
                strategy_by_symbol=strategy_map,
                result=result,
                competition_metrics=metrics,
                risk_discipline=risk_discipline,
                walk_forward=walk_forward,
                promotion=promotion,
            )
        )

    scored_candidates = _attach_scores(tuple(candidates))
    return CryptoOverlayComparison(
        official_symbols=official_symbols,
        crypto_symbols=crypto_symbols,
        candidates=tuple(
            sorted(scored_candidates, key=lambda candidate: candidate.rank_key, reverse=True)
        ),
        base_strategy=normalized_base,
        walk_forward_enabled=run_walk_forward,
    )


def write_crypto_overlay_comparison_csv(
    comparison: CryptoOverlayComparison,
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
                "official_symbols",
                "crypto_symbols",
                "base_strategy",
                "strategy_map",
                "crypto_map",
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
                    "official_symbols": " ".join(comparison.official_symbols),
                    "crypto_symbols": " ".join(comparison.crypto_symbols),
                    "base_strategy": comparison.base_strategy,
                    "strategy_map": candidate.strategy_map_text,
                    "crypto_map": candidate.crypto_map_text,
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


def _candidate_maps(
    *,
    official_symbols: tuple[str, ...],
    crypto_symbols: tuple[str, ...],
    base_strategy: str,
) -> tuple[tuple[str, dict[str, str]], ...]:
    official_base = {symbol: base_strategy for symbol in official_symbols}
    all_symbols = official_symbols + crypto_symbols
    candidates: list[tuple[str, dict[str, str]]] = []
    if official_symbols:
        candidates.append(("official_only_base", dict(official_base)))
    if all_symbols:
        candidates.append(
            (
                "all_symbols_base",
                {symbol: base_strategy for symbol in all_symbols},
            )
        )
    if crypto_symbols:
        all_reversion = dict(official_base)
        all_reversion.update(
            {symbol: "crypto_mean_reversion" for symbol in crypto_symbols}
        )
        candidates.append(("crypto_all_reversion_overlay", all_reversion))

        robust = dict(official_base)
        robust.update(
            {
                symbol: ROBUST_CRYPTO_OVERLAY.get(symbol, "crypto_mean_reversion")
                for symbol in crypto_symbols
            }
        )
        candidates.append(("crypto_robust_sol_overlay", robust))

        aggressive = dict(official_base)
        aggressive.update(
            {
                symbol: AGGRESSIVE_CRYPTO_OVERLAY.get(symbol, "crypto_mean_reversion")
                for symbol in crypto_symbols
            }
        )
        candidates.append(("crypto_aggressive_btc_sol_overlay", aggressive))
    return tuple(candidates)


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
    candidates: tuple[CryptoOverlayCandidate, ...],
) -> tuple[CryptoOverlayCandidate, ...]:
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
    scored: list[CryptoOverlayCandidate] = []
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
        scored.append(
            replace(
                candidate,
                proxy_score=proxy_score,
                selection_score=_selection_score(
                    proxy_score=proxy_score,
                    walk_forward=candidate.walk_forward,
                    promotion=candidate.promotion,
                ),
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
    missing_prices = sorted(set(selected) - set(prices.symbols()))
    missing_quotes = sorted(set(selected) - set(quotes.symbols()))
    if missing_prices:
        raise ValueError(f"symbols missing price data: {', '.join(missing_prices)}")
    if missing_quotes:
        raise ValueError(f"symbols missing quote data: {', '.join(missing_quotes)}")
    return selected


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
