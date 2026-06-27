from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from quanthack.backtesting.portfolio_fixed_warmup_walk_forward import (
    FixedWarmupPortfolioWalkForwardResult,
    FixedWarmupPromotionDecision,
    decide_fixed_warmup_promotion,
    run_fixed_warmup_portfolio_walk_forward,
)
from quanthack.backtesting.portfolio_strategy_compare import (
    PortfolioStrategyComparisonRow,
    compare_portfolio_strategies,
)
from quanthack.core.config import AppConfig
from quanthack.core.instruments import AssetClass, instrument_for, instruments_by_asset_class
from quanthack.market.market_data import PriceHistory, QuoteHistory
from quanthack.strategies.strategy import normalize_strategy_name


DEFAULT_CRYPTO_SLEEVE_STRATEGIES = (
    "macd_momentum",
    "asset_adaptive_macd",
    "quality_trend",
    "multi_horizon_momentum",
    "volatility_squeeze",
    "dual_squeeze",
    "asset_adaptive_dual_squeeze",
    "range_expansion_trend",
    "trend_pullback",
    "mean_reversion",
    "crypto_mean_reversion",
    "crypto_trend_reversion",
    "champion_ensemble",
)


@dataclass(frozen=True)
class CryptoSleeveComparisonRow:
    strategy_name: str
    full_sample: PortfolioStrategyComparisonRow
    walk_forward: FixedWarmupPortfolioWalkForwardResult | None
    promotion: FixedWarmupPromotionDecision | None
    walk_forward_error: str | None
    selection_score: float

    @property
    def rank_key(self) -> tuple[float, float, float, float, float]:
        metrics = self.full_sample.competition_metrics
        return (
            self.selection_score,
            self.full_sample.proxy_score,
            self.full_sample.risk_discipline.score,
            metrics.return_pct,
            -metrics.max_drawdown_pct,
        )


@dataclass(frozen=True)
class CryptoSleeveComparison:
    symbols: tuple[str, ...]
    rows: tuple[CryptoSleeveComparisonRow, ...]
    walk_forward_enabled: bool

    @property
    def best(self) -> CryptoSleeveComparisonRow | None:
        if not self.rows:
            return None
        return self.rows[0]


def compare_crypto_sleeves(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    strategy_names: tuple[str, ...] = DEFAULT_CRYPTO_SLEEVE_STRATEGIES,
    symbols: tuple[str, ...] | None = None,
    run_walk_forward: bool = True,
    train_size: int = 480,
    test_size: int = 192,
    step_size: int = 192,
) -> CryptoSleeveComparison:
    selected_symbols = _selected_crypto_symbols(
        prices=prices,
        quotes=quotes,
        symbols=symbols,
    )
    normalized_strategy_names = _normalize_unique_strategy_names(strategy_names)
    full_sample = compare_portfolio_strategies(
        config=config,
        prices=prices,
        quotes=quotes,
        strategy_names=normalized_strategy_names,
        symbols=selected_symbols,
    )
    full_by_strategy = {row.strategy_name: row for row in full_sample.rows}

    rows: list[CryptoSleeveComparisonRow] = []
    for strategy_name in normalized_strategy_names:
        full_row = full_by_strategy[strategy_name]
        walk_forward: FixedWarmupPortfolioWalkForwardResult | None = None
        promotion: FixedWarmupPromotionDecision | None = None
        walk_forward_error: str | None = None
        if run_walk_forward:
            try:
                walk_forward = run_fixed_warmup_portfolio_walk_forward(
                    config=config,
                    prices=prices,
                    quotes=quotes,
                    strategy_name=strategy_name,
                    symbols=selected_symbols,
                    train_size=train_size,
                    test_size=test_size,
                    step_size=step_size,
                )
                promotion = decide_fixed_warmup_promotion(walk_forward)
            except ValueError as exc:
                walk_forward_error = str(exc)
        rows.append(
            CryptoSleeveComparisonRow(
                strategy_name=strategy_name,
                full_sample=full_row,
                walk_forward=walk_forward,
                promotion=promotion,
                walk_forward_error=walk_forward_error,
                selection_score=_selection_score(
                    full_row=full_row,
                    walk_forward=walk_forward,
                    promotion=promotion,
                ),
            )
        )

    ranked_rows = tuple(sorted(rows, key=lambda row: row.rank_key, reverse=True))
    return CryptoSleeveComparison(
        symbols=selected_symbols,
        rows=ranked_rows,
        walk_forward_enabled=run_walk_forward,
    )


def write_crypto_sleeve_comparison_csv(
    comparison: CryptoSleeveComparison,
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
                "selection_score",
                "full_proxy_score",
                "full_return_pct",
                "full_max_drawdown_pct",
                "full_sharpe_15m",
                "full_risk_discipline_score",
                "full_trade_count",
                "full_fills",
                "full_turnover_notional",
                "full_total_pnl_usd",
                "walk_forward_enabled",
                "walk_forward_error",
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
        for rank, row in enumerate(comparison.rows, start=1):
            metrics = row.full_sample.competition_metrics
            writer.writerow(
                {
                    "rank": rank,
                    "strategy": row.strategy_name,
                    "symbols": " ".join(comparison.symbols),
                    "selection_score": row.selection_score,
                    "full_proxy_score": row.full_sample.proxy_score,
                    "full_return_pct": metrics.return_pct,
                    "full_max_drawdown_pct": metrics.max_drawdown_pct,
                    "full_sharpe_15m": metrics.sharpe_15m,
                    "full_risk_discipline_score": (
                        row.full_sample.risk_discipline.score
                    ),
                    "full_trade_count": metrics.trade_count,
                    "full_fills": len(row.full_sample.result.fills),
                    "full_turnover_notional": (
                        row.full_sample.result.metrics.turnover_notional
                    ),
                    "full_total_pnl_usd": row.full_sample.result.total_pnl_usd,
                    "walk_forward_enabled": comparison.walk_forward_enabled,
                    "walk_forward_error": row.walk_forward_error or "",
                    "promotion_status": row.promotion.status if row.promotion else "",
                    "promotion_live_ready": (
                        row.promotion.live_ready if row.promotion else ""
                    ),
                    "promotion_reason": row.promotion.reason if row.promotion else "",
                    **_walk_forward_columns(row.walk_forward),
                }
            )


def _selection_score(
    *,
    full_row: PortfolioStrategyComparisonRow,
    walk_forward: FixedWarmupPortfolioWalkForwardResult | None,
    promotion: FixedWarmupPromotionDecision | None,
) -> float:
    if walk_forward is None:
        return full_row.proxy_score

    risk_score = min(max(walk_forward.average_risk_discipline_score, 0.0), 100.0) / 100.0
    fold_stability = (
        0.30 * walk_forward.non_negative_fold_fraction
        + 0.30 * walk_forward.active_positive_fold_fraction
        + 0.20 * walk_forward.positive_fold_fraction
        + 0.10 * walk_forward.active_fold_fraction
        + 0.10 * risk_score
    )
    concentration_penalty = max(
        0.0,
        walk_forward.largest_positive_fold_contribution - 0.80,
    ) * 25.0
    promotion_bonus = 5.0 if promotion and promotion.live_ready else 0.0
    return (
        0.65 * full_row.proxy_score
        + 35.0 * fold_stability
        + promotion_bonus
        - concentration_penalty
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
