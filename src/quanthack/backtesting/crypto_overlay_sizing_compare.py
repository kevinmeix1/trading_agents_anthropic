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
from quanthack.backtesting.crypto_overlay_compare import AGGRESSIVE_CRYPTO_OVERLAY
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
from quanthack.backtesting.portfolio_session import SessionGatePolicy
from quanthack.core.config import AppConfig
from quanthack.core.instruments import AssetClass, instrument_for
from quanthack.market.market_data import PriceHistory, QuoteHistory
from quanthack.strategies.strategy import normalize_strategy_name


@dataclass(frozen=True)
class CryptoOverlaySizingSpec:
    label: str
    crypto_multiplier: float = 1.0
    btc_multiplier: float | None = None
    sol_multiplier: float | None = None
    trend_crypto_multiplier: float | None = None
    reversion_crypto_multiplier: float | None = None
    crypto_allowed_utc_hours: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("sizing candidate label cannot be empty")
        for name, value in (
            ("crypto_multiplier", self.crypto_multiplier),
            ("btc_multiplier", self.btc_multiplier),
            ("sol_multiplier", self.sol_multiplier),
            ("trend_crypto_multiplier", self.trend_crypto_multiplier),
            ("reversion_crypto_multiplier", self.reversion_crypto_multiplier),
        ):
            if value is not None and not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.crypto_allowed_utc_hours is not None:
            normalized_hours = tuple(
                sorted({int(hour) for hour in self.crypto_allowed_utc_hours})
            )
            if any(hour < 0 or hour > 23 for hour in normalized_hours):
                raise ValueError("crypto_allowed_utc_hours must be between 0 and 23")
            object.__setattr__(
                self,
                "crypto_allowed_utc_hours",
                normalized_hours,
            )


@dataclass(frozen=True)
class CryptoOverlaySizingCandidate:
    label: str
    strategy_by_symbol: tuple[tuple[str, str], ...]
    multipliers_by_symbol: tuple[tuple[str, float], ...]
    crypto_allowed_utc_hours: tuple[int, ...] | None
    result: PortfolioBacktestResult
    competition_metrics: CompetitionMetrics
    risk_discipline: RiskDisciplineReport
    proxy_score: float = 0.0
    selection_score: float = 0.0
    walk_forward: FixedWarmupPortfolioWalkForwardResult | None = None
    promotion: FixedWarmupPromotionDecision | None = None

    @property
    def multiplier_map_text(self) -> str:
        return " ".join(
            f"{symbol}={multiplier:.3f}"
            for symbol, multiplier in self.multipliers_by_symbol
        )

    @property
    def crypto_allowed_utc_hours_text(self) -> str:
        if self.crypto_allowed_utc_hours is None:
            return "all"
        return "|".join(str(hour) for hour in self.crypto_allowed_utc_hours)

    @property
    def strategy_map_text(self) -> str:
        return " ".join(
            f"{symbol}={strategy}" for symbol, strategy in self.strategy_by_symbol
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
class CryptoOverlaySizingComparison:
    official_symbols: tuple[str, ...]
    crypto_symbols: tuple[str, ...]
    candidates: tuple[CryptoOverlaySizingCandidate, ...]
    base_strategy: str
    walk_forward_enabled: bool

    @property
    def best(self) -> CryptoOverlaySizingCandidate | None:
        if not self.candidates:
            return None
        return self.candidates[0]


DEFAULT_SIZING_SPECS = (
    CryptoOverlaySizingSpec(label="crypto_100", crypto_multiplier=1.0),
    CryptoOverlaySizingSpec(label="crypto_075", crypto_multiplier=0.75),
    CryptoOverlaySizingSpec(label="crypto_050", crypto_multiplier=0.50),
    CryptoOverlaySizingSpec(label="crypto_035", crypto_multiplier=0.35),
    CryptoOverlaySizingSpec(
        label="btc075_sol100_reversion075",
        crypto_multiplier=0.75,
        btc_multiplier=0.75,
        sol_multiplier=1.0,
    ),
    CryptoOverlaySizingSpec(
        label="btc050_sol100_reversion075",
        crypto_multiplier=0.75,
        btc_multiplier=0.50,
        sol_multiplier=1.0,
    ),
    CryptoOverlaySizingSpec(
        label="trend075_reversion050",
        crypto_multiplier=0.50,
        trend_crypto_multiplier=0.75,
    ),
    CryptoOverlaySizingSpec(
        label="btc075_sol100_reversion075_london_us",
        crypto_multiplier=0.75,
        btc_multiplier=0.75,
        sol_multiplier=1.0,
        crypto_allowed_utc_hours=tuple(range(8, 22)),
    ),
    CryptoOverlaySizingSpec(
        label="btc075_sol100_reversion075_us",
        crypto_multiplier=0.75,
        btc_multiplier=0.75,
        sol_multiplier=1.0,
        crypto_allowed_utc_hours=tuple(range(13, 22)),
    ),
    CryptoOverlaySizingSpec(
        label="btc075_sol100_reversion075_london",
        crypto_multiplier=0.75,
        btc_multiplier=0.75,
        sol_multiplier=1.0,
        crypto_allowed_utc_hours=tuple(range(7, 17)),
    ),
    CryptoOverlaySizingSpec(
        label="btc075_sol100_reversion075_asia",
        crypto_multiplier=0.75,
        btc_multiplier=0.75,
        sol_multiplier=1.0,
        crypto_allowed_utc_hours=tuple(range(0, 9)),
    ),
)


def compare_crypto_overlay_sizing(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    base_strategy: str = "macd_momentum",
    symbols: tuple[str, ...] | None = None,
    specs: tuple[CryptoOverlaySizingSpec, ...] | None = DEFAULT_SIZING_SPECS,
    run_walk_forward: bool = True,
    train_size: int = 96,
    test_size: int = 48,
    step_size: int = 48,
) -> CryptoOverlaySizingComparison:
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
    sizing_specs = specs or DEFAULT_SIZING_SPECS
    strategy_map = _aggressive_strategy_map(
        official_symbols=official_symbols,
        crypto_symbols=crypto_symbols,
        base_strategy=normalized_base,
    )

    candidates: list[CryptoOverlaySizingCandidate] = []
    for spec in sizing_specs:
        multipliers = _multipliers_for_spec(
            spec=spec,
            crypto_symbols=crypto_symbols,
            strategy_by_symbol=strategy_map,
        )
        session_policy = _session_policy_for_spec(spec)
        result = _run_strategy_map(
            config=config,
            prices=prices,
            quotes=quotes,
            strategy_by_symbol=strategy_map,
            multipliers_by_symbol=multipliers,
            session_gate_policy=session_policy,
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
                strategy_name=normalized_base,
                symbols=tuple(symbol for symbol, _ in sorted(strategy_map.items())),
                strategy_by_symbol=strategy_map,
                train_size=train_size,
                test_size=test_size,
                step_size=step_size,
                target_notional_multipliers_by_symbol=multipliers,
                session_gate_policy=session_policy,
            )
            promotion = decide_fixed_warmup_promotion(walk_forward)
        candidates.append(
            CryptoOverlaySizingCandidate(
                label=spec.label,
                strategy_by_symbol=tuple(sorted(strategy_map.items())),
                multipliers_by_symbol=tuple(sorted(multipliers.items())),
                crypto_allowed_utc_hours=spec.crypto_allowed_utc_hours,
                result=result,
                competition_metrics=metrics,
                risk_discipline=risk_discipline,
                walk_forward=walk_forward,
                promotion=promotion,
            )
        )

    scored = _attach_scores(tuple(candidates))
    return CryptoOverlaySizingComparison(
        official_symbols=official_symbols,
        crypto_symbols=crypto_symbols,
        candidates=tuple(sorted(scored, key=lambda candidate: candidate.rank_key, reverse=True)),
        base_strategy=normalized_base,
        walk_forward_enabled=run_walk_forward,
    )


def write_crypto_overlay_sizing_comparison_csv(
    comparison: CryptoOverlaySizingComparison,
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
                "multiplier_map",
                "crypto_allowed_utc_hours",
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
                    "multiplier_map": candidate.multiplier_map_text,
                    "crypto_allowed_utc_hours": (
                        candidate.crypto_allowed_utc_hours_text
                    ),
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


def _aggressive_strategy_map(
    *,
    official_symbols: tuple[str, ...],
    crypto_symbols: tuple[str, ...],
    base_strategy: str,
) -> dict[str, str]:
    strategy_map = {symbol: base_strategy for symbol in official_symbols}
    strategy_map.update(
        {
            symbol: AGGRESSIVE_CRYPTO_OVERLAY.get(symbol, "crypto_mean_reversion")
            for symbol in crypto_symbols
        }
    )
    return strategy_map


def _multipliers_for_spec(
    *,
    spec: CryptoOverlaySizingSpec,
    crypto_symbols: tuple[str, ...],
    strategy_by_symbol: dict[str, str],
) -> dict[str, float]:
    multipliers: dict[str, float] = {}
    for symbol in crypto_symbols:
        multiplier = spec.crypto_multiplier
        strategy_name = strategy_by_symbol[symbol]
        if (
            spec.trend_crypto_multiplier is not None
            and strategy_name == "macd_momentum"
        ):
            multiplier = spec.trend_crypto_multiplier
        if (
            spec.reversion_crypto_multiplier is not None
            and strategy_name == "crypto_mean_reversion"
        ):
            multiplier = spec.reversion_crypto_multiplier
        if symbol == "BTCUSD" and spec.btc_multiplier is not None:
            multiplier = spec.btc_multiplier
        if symbol == "SOLUSD" and spec.sol_multiplier is not None:
            multiplier = spec.sol_multiplier
        multipliers[symbol] = multiplier
    return multipliers


def _session_policy_for_spec(
    spec: CryptoOverlaySizingSpec,
) -> SessionGatePolicy | None:
    if spec.crypto_allowed_utc_hours is None:
        return None
    return SessionGatePolicy(crypto_allowed_utc_hours=spec.crypto_allowed_utc_hours)


def _run_strategy_map(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    strategy_by_symbol: dict[str, str],
    multipliers_by_symbol: dict[str, float],
    session_gate_policy: SessionGatePolicy | None,
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
        target_notional_multipliers_by_symbol=multipliers_by_symbol,
        session_gate_policy=session_gate_policy,
    )
    return engine.run(
        prices=prices,
        quotes=quotes,
        starting_equity=config.competition.starting_equity,
    )


def _attach_scores(
    candidates: tuple[CryptoOverlaySizingCandidate, ...],
) -> tuple[CryptoOverlaySizingCandidate, ...]:
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
    scored: list[CryptoOverlaySizingCandidate] = []
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
        0.35 * walk_forward.non_negative_fold_fraction
        + 0.25 * walk_forward.active_positive_fold_fraction
        + 0.15 * walk_forward.positive_fold_fraction
        + 0.10 * walk_forward.active_fold_fraction
        + 0.15 * risk_score
    )
    concentration_penalty = max(
        0.0,
        walk_forward.largest_positive_fold_contribution - 0.70,
    ) * 35.0
    promotion_bonus = 8.0 if promotion and promotion.live_ready else 0.0
    return 0.55 * proxy_score + 45.0 * fold_stability + promotion_bonus - concentration_penalty


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
