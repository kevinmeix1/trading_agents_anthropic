from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from quanthack.backtesting.crypto_overlay_sizing_compare import (
    CryptoOverlaySizingCandidate,
    CryptoOverlaySizingComparison,
    CryptoOverlaySizingSpec,
    compare_crypto_overlay_sizing,
)
from quanthack.core.config import AppConfig
from quanthack.market.market_data import PriceHistory, QuoteHistory
from quanthack.strategies.strategy import normalize_strategy_name


@dataclass(frozen=True)
class CryptoFoldStabilityCandidate:
    sizing: CryptoOverlaySizingCandidate
    return_retention: float
    stability_score: float
    stability_status: str

    @property
    def rank_key(self) -> tuple[float, ...]:
        metrics = self.sizing.competition_metrics
        walk_forward = self.sizing.walk_forward
        if walk_forward is None:
            return (
                self.stability_score,
                self.return_retention,
                metrics.return_pct,
                metrics.sharpe_15m,
                -metrics.max_drawdown_pct,
            )
        return (
            self.stability_score,
            self.return_retention,
            1.0 - walk_forward.largest_positive_fold_contribution,
            walk_forward.active_positive_fold_fraction,
            walk_forward.non_negative_fold_fraction,
            walk_forward.median_active_test_return_pct,
            metrics.return_pct,
            metrics.sharpe_15m,
            -metrics.max_drawdown_pct,
        )


@dataclass(frozen=True)
class CryptoFoldStabilityOptimization:
    comparison: CryptoOverlaySizingComparison
    candidates: tuple[CryptoFoldStabilityCandidate, ...]

    @property
    def best(self) -> CryptoFoldStabilityCandidate | None:
        if not self.candidates:
            return None
        return self.candidates[0]


DEFAULT_STABILITY_SPECS = tuple(
    CryptoOverlaySizingSpec(
        label=f"{profile_label}_{session_label}",
        crypto_multiplier=crypto_multiplier,
        btc_multiplier=btc_multiplier,
        sol_multiplier=sol_multiplier,
        trend_crypto_multiplier=trend_multiplier,
        reversion_crypto_multiplier=reversion_multiplier,
        crypto_allowed_utc_hours=session_hours,
    )
    for (
        profile_label,
        crypto_multiplier,
        btc_multiplier,
        sol_multiplier,
        trend_multiplier,
        reversion_multiplier,
    ) in (
        ("current", 0.75, 0.75, 1.00, None, None),
        ("soft", 0.50, 0.50, 0.75, None, 0.50),
        ("balanced_low", 0.35, 0.50, 0.50, None, 0.35),
        ("trend_heavy", 0.25, 0.75, 1.00, None, 0.25),
        ("sol_heavy_lowrev", 0.25, 0.50, 1.00, None, 0.25),
        ("btc_heavy_lowrev", 0.25, 1.00, 0.50, None, 0.25),
        ("trend_only", 0.00, 0.75, 1.00, 0.75, 0.00),
        ("reversion_only", 0.50, 0.00, 0.00, 0.00, 0.50),
    )
    for session_label, session_hours in (
        ("all", None),
        ("asia", tuple(range(0, 9))),
        ("london", tuple(range(7, 17))),
        ("london_us", tuple(range(8, 22))),
        ("us", tuple(range(13, 22))),
    )
)


def optimize_crypto_fold_stability(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    base_strategy: str = "macd_momentum",
    symbols: tuple[str, ...] | None = None,
    specs: tuple[CryptoOverlaySizingSpec, ...] = DEFAULT_STABILITY_SPECS,
    train_size: int = 96,
    test_size: int = 48,
    step_size: int = 48,
) -> CryptoFoldStabilityOptimization:
    if not specs:
        raise ValueError("crypto fold stability optimizer needs at least one spec")
    comparison = compare_crypto_overlay_sizing(
        config=config,
        prices=prices,
        quotes=quotes,
        base_strategy=normalize_strategy_name(base_strategy),
        symbols=symbols,
        specs=specs,
        run_walk_forward=True,
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
    )
    best_positive_return = max(
        (candidate.competition_metrics.return_pct for candidate in comparison.candidates),
        default=0.0,
    )
    candidates = tuple(
        _stability_candidate(
            candidate=candidate,
            best_positive_return=max(best_positive_return, 0.0),
        )
        for candidate in comparison.candidates
    )
    return CryptoFoldStabilityOptimization(
        comparison=comparison,
        candidates=tuple(
            sorted(candidates, key=lambda candidate: candidate.rank_key, reverse=True)
        ),
    )


def write_crypto_fold_stability_csv(
    result: CryptoFoldStabilityOptimization,
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
                "stability_status",
                "stability_score",
                "return_retention",
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
                "total_pnl_usd",
                "promotion_status",
                "promotion_reason",
                "wf_positive_fold_fraction",
                "wf_active_fold_fraction",
                "wf_active_positive_fold_fraction",
                "wf_non_negative_fold_fraction",
                "wf_median_active_test_return_pct",
                "wf_worst_test_drawdown_pct",
                "wf_largest_positive_fold_contribution",
            ],
        )
        writer.writeheader()
        for rank, candidate in enumerate(result.candidates, start=1):
            sizing = candidate.sizing
            metrics = sizing.competition_metrics
            walk_forward = sizing.walk_forward
            writer.writerow(
                {
                    "rank": rank,
                    "label": sizing.label,
                    "stability_status": candidate.stability_status,
                    "stability_score": candidate.stability_score,
                    "return_retention": candidate.return_retention,
                    "official_symbols": " ".join(result.comparison.official_symbols),
                    "crypto_symbols": " ".join(result.comparison.crypto_symbols),
                    "base_strategy": result.comparison.base_strategy,
                    "strategy_map": sizing.strategy_map_text,
                    "multiplier_map": sizing.multiplier_map_text,
                    "crypto_allowed_utc_hours": sizing.crypto_allowed_utc_hours_text,
                    "selection_score": sizing.selection_score,
                    "proxy_score": sizing.proxy_score,
                    "return_pct": metrics.return_pct,
                    "max_drawdown_pct": metrics.max_drawdown_pct,
                    "sharpe_15m": metrics.sharpe_15m,
                    "risk_discipline_score": sizing.risk_discipline.score,
                    "trade_count": metrics.trade_count,
                    "fills": len(sizing.result.fills),
                    "total_pnl_usd": sizing.result.total_pnl_usd,
                    "promotion_status": sizing.promotion.status if sizing.promotion else "",
                    "promotion_reason": (
                        sizing.promotion.reason if sizing.promotion else ""
                    ),
                    "wf_positive_fold_fraction": (
                        "" if walk_forward is None else walk_forward.positive_fold_fraction
                    ),
                    "wf_active_fold_fraction": (
                        "" if walk_forward is None else walk_forward.active_fold_fraction
                    ),
                    "wf_active_positive_fold_fraction": (
                        ""
                        if walk_forward is None
                        else walk_forward.active_positive_fold_fraction
                    ),
                    "wf_non_negative_fold_fraction": (
                        "" if walk_forward is None else walk_forward.non_negative_fold_fraction
                    ),
                    "wf_median_active_test_return_pct": (
                        ""
                        if walk_forward is None
                        else walk_forward.median_active_test_return_pct
                    ),
                    "wf_worst_test_drawdown_pct": (
                        "" if walk_forward is None else walk_forward.worst_test_drawdown_pct
                    ),
                    "wf_largest_positive_fold_contribution": (
                        ""
                        if walk_forward is None
                        else walk_forward.largest_positive_fold_contribution
                    ),
                }
            )


def _stability_candidate(
    *,
    candidate: CryptoOverlaySizingCandidate,
    best_positive_return: float,
) -> CryptoFoldStabilityCandidate:
    return_retention = (
        0.0
        if best_positive_return <= 0
        else max(candidate.competition_metrics.return_pct, 0.0) / best_positive_return
    )
    score = _stability_score(
        candidate=candidate,
        return_retention=return_retention,
    )
    return CryptoFoldStabilityCandidate(
        sizing=candidate,
        return_retention=return_retention,
        stability_score=score,
        stability_status=_stability_status(candidate),
    )


def _stability_score(
    *,
    candidate: CryptoOverlaySizingCandidate,
    return_retention: float,
) -> float:
    metrics = candidate.competition_metrics
    walk_forward = candidate.walk_forward
    risk_score = min(max(candidate.risk_discipline.score, 0.0), 100.0) / 100.0
    drawdown_score = 1.0 - min(metrics.max_drawdown_pct / 0.03, 1.0)
    trade_score = min(metrics.trade_count / 40.0, 1.0)

    if walk_forward is None:
        base = (
            0.45 * return_retention
            + 0.20 * risk_score
            + 0.20 * drawdown_score
            + 0.15 * trade_score
        )
    else:
        concentration_score = 1.0 - min(
            max(walk_forward.largest_positive_fold_contribution, 0.0),
            1.0,
        )
        base = (
            0.18 * return_retention
            + 0.20 * walk_forward.active_positive_fold_fraction
            + 0.17 * walk_forward.non_negative_fold_fraction
            + 0.20 * concentration_score
            + 0.10 * drawdown_score
            + 0.10 * risk_score
            + 0.05 * trade_score
        )

    penalty = 0.0
    if metrics.return_pct <= 0:
        penalty += 40.0
    if walk_forward is not None and walk_forward.median_active_test_return_pct <= 0:
        penalty += 20.0
    return 100.0 * base - penalty


def _stability_status(candidate: CryptoOverlaySizingCandidate) -> str:
    metrics = candidate.competition_metrics
    walk_forward = candidate.walk_forward
    if metrics.return_pct <= 0 or candidate.risk_discipline.score < 95:
        return "REJECT"
    if walk_forward is None:
        return "FULL_SAMPLE_ONLY"
    if (
        walk_forward.non_negative_fold_fraction < 0.70
        or walk_forward.median_active_test_return_pct <= 0
    ):
        return "REJECT"
    if (
        walk_forward.largest_positive_fold_contribution <= 0.80
        and walk_forward.positive_fold_fraction >= 0.67
        and walk_forward.active_positive_fold_fraction >= 0.67
    ):
        return "STABLE_PROFILE"
    return "FRAGILE_PROFILE"
