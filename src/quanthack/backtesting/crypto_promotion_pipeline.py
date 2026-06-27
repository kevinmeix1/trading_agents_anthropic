from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from quanthack.backtesting.asset_class_stability_optimizer import (
    DEFAULT_ASSET_CLASS_STABILITY_SPECS,
    AssetClassStabilityCandidate,
    AssetClassStabilityOptimization,
    AssetClassStabilitySpec,
    optimize_asset_class_stability,
    write_asset_class_stability_csv,
)
from quanthack.backtesting.crypto_overlay_component_ablation import (
    DEFAULT_COMPONENT_ABLATIONS,
    CryptoOverlayComponentAblationResult,
    CryptoOverlayComponentAblationSpec,
    compare_crypto_overlay_components,
    write_crypto_overlay_component_ablation_csv,
)
from quanthack.backtesting.crypto_overlay_fold_diagnostic import (
    CryptoOverlayFoldDiagnostic,
    build_crypto_overlay_fold_diagnostic,
    write_crypto_overlay_fold_diagnostic_summary_csv,
    write_crypto_overlay_fold_symbol_summary_csv,
)
from quanthack.backtesting.crypto_overlay_sizing_compare import (
    DEFAULT_SIZING_SPECS,
    CryptoOverlaySizingComparison,
    CryptoOverlaySizingSpec,
    compare_crypto_overlay_sizing,
    write_crypto_overlay_sizing_comparison_csv,
)
from quanthack.backtesting.research_candidate_gate import (
    ResearchCandidateGateRow,
    ResearchCandidateSource,
    ResearchDataSource,
    ResearchReadiness,
    build_research_candidate_gate,
    normalize_research_data_source,
    write_research_candidate_gate_csv,
)
from quanthack.core.config import AppConfig
from quanthack.core.instruments import DEFAULT_INSTRUMENTS, AssetClass, instrument_for
from quanthack.market.data_health import (
    DataHealthSeverity,
    MarketDataHealthReport,
    validate_market_data,
    write_market_data_health_csv,
)
from quanthack.market.market_data import PriceHistory, QuoteHistory
from quanthack.strategies.strategy import normalize_strategy_name


DEFAULT_PIPELINE_MAX_GAP_SECONDS = 960.0
DEFAULT_MAX_LIVE_FOLD_CONTRIBUTION = 0.80


@dataclass(frozen=True)
class CryptoPromotionArtifacts:
    data_health_csv: Path
    sizing_csv: Path
    sizing_gate_csv: Path
    component_ablation_csv: Path
    component_gate_csv: Path
    asset_class_stability_csv: Path
    fold_diagnostic_prefix: Path
    fold_diagnostic_summary_csv: Path
    fold_diagnostic_symbol_summary_csv: Path
    summary_csv: Path


@dataclass(frozen=True)
class CryptoPromotionSummary:
    data_source: ResearchDataSource
    price_csv: str
    quote_csv: str
    selected_symbols: tuple[str, ...]
    missing_competition_symbols: tuple[str, ...]
    crypto_symbols: tuple[str, ...]
    data_health: DataHealthSeverity
    data_health_issue_count: int
    best_sizing_label: str
    best_sizing_return_pct: float
    best_sizing_drawdown_pct: float
    best_sizing_sharpe_15m: float
    best_sizing_risk_score: float
    best_sizing_readiness: ResearchReadiness
    best_sizing_reason: str
    component_full_return_pct: float
    no_crypto_return_delta_pct: float
    no_metals_return_delta_pct: float
    no_btc_sol_return_delta_pct: float
    stable_backup_label: str
    stable_backup_return_pct: float
    stable_backup_drawdown_pct: float
    stable_backup_sharpe_15m: float
    stable_backup_fold_contribution: float
    stable_backup_return_retention: float
    stable_backup_promotion_status: str
    stable_backup_reason: str
    fold_count: int
    strongest_fold: int
    strongest_fold_return_pct: float
    largest_positive_fold_contribution: float
    promotion_readiness: ResearchReadiness
    live_ready: bool
    promotion_reason: str


@dataclass(frozen=True)
class CryptoPromotionPipelineResult:
    artifacts: CryptoPromotionArtifacts
    data_health: MarketDataHealthReport
    sizing: CryptoOverlaySizingComparison
    sizing_gate: tuple[ResearchCandidateGateRow, ...]
    components: CryptoOverlayComponentAblationResult
    component_gate: tuple[ResearchCandidateGateRow, ...]
    asset_class_stability: AssetClassStabilityOptimization
    diagnostic: CryptoOverlayFoldDiagnostic
    summary: CryptoPromotionSummary


def run_crypto_promotion_pipeline(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    price_csv: str,
    quote_csv: str,
    data_source: str | ResearchDataSource,
    output_prefix: str | Path = "outputs/research/crypto_promotion_pipeline",
    base_strategy: str = "macd_momentum",
    symbols: tuple[str, ...] | None = None,
    sizing_specs: tuple[CryptoOverlaySizingSpec, ...] | None = DEFAULT_SIZING_SPECS,
    component_specs: tuple[CryptoOverlayComponentAblationSpec, ...]
    | None = DEFAULT_COMPONENT_ABLATIONS,
    asset_class_specs: tuple[AssetClassStabilitySpec, ...]
    | None = DEFAULT_ASSET_CLASS_STABILITY_SPECS,
    train_size: int = 96,
    test_size: int = 48,
    step_size: int = 48,
    max_gap_seconds: float | None = DEFAULT_PIPELINE_MAX_GAP_SECONDS,
    max_live_fold_contribution: float = DEFAULT_MAX_LIVE_FOLD_CONTRIBUTION,
) -> CryptoPromotionPipelineResult:
    normalized_source = normalize_research_data_source(data_source)
    normalized_base_strategy = normalize_strategy_name(base_strategy)
    selected_symbols = _selected_symbols(prices=prices, quotes=quotes, symbols=symbols)
    artifacts = _artifacts_for_prefix(output_prefix)

    health = validate_market_data(
        prices=prices,
        quotes=quotes,
        symbols=selected_symbols,
        max_gap_seconds=max_gap_seconds,
        max_spread_bps=config.market_quality.max_spread_bps,
        max_spread_bps_by_symbol=_spread_limits_by_symbol(selected_symbols),
    )
    write_market_data_health_csv(health, artifacts.data_health_csv)
    if health.overall == DataHealthSeverity.FAIL:
        raise ValueError(
            "market data health failed; fix alignment/coverage before running promotion"
        )

    specs = sizing_specs or DEFAULT_SIZING_SPECS
    sizing = compare_crypto_overlay_sizing(
        config=config,
        prices=prices,
        quotes=quotes,
        base_strategy=normalized_base_strategy,
        symbols=selected_symbols,
        specs=specs,
        run_walk_forward=True,
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
    )
    write_crypto_overlay_sizing_comparison_csv(sizing, artifacts.sizing_csv)
    sizing_gate = build_research_candidate_gate(
        (
            ResearchCandidateSource(
                path=str(artifacts.sizing_csv),
                data_source=normalized_source,
            ),
        )
    )
    write_research_candidate_gate_csv(sizing_gate, artifacts.sizing_gate_csv)

    selected_spec = _selected_spec_from_sizing(
        sizing=sizing,
        specs=specs,
    )
    components = compare_crypto_overlay_components(
        config=config,
        prices=prices,
        quotes=quotes,
        base_strategy=normalized_base_strategy,
        base_spec=selected_spec,
        specs=component_specs,
        symbols=selected_symbols,
        run_walk_forward=True,
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
    )
    write_crypto_overlay_component_ablation_csv(
        components,
        artifacts.component_ablation_csv,
    )
    component_gate = build_research_candidate_gate(
        (
            ResearchCandidateSource(
                path=str(artifacts.component_ablation_csv),
                data_source=normalized_source,
            ),
        )
    )
    write_research_candidate_gate_csv(component_gate, artifacts.component_gate_csv)

    asset_class_stability = optimize_asset_class_stability(
        config=config,
        prices=prices,
        quotes=quotes,
        base_strategy=normalized_base_strategy,
        symbols=selected_symbols,
        specs=asset_class_specs or DEFAULT_ASSET_CLASS_STABILITY_SPECS,
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
    )
    write_asset_class_stability_csv(
        asset_class_stability,
        artifacts.asset_class_stability_csv,
    )

    diagnostic = build_crypto_overlay_fold_diagnostic(
        config=config,
        prices=prices,
        quotes=quotes,
        base_strategy=normalized_base_strategy,
        spec=selected_spec,
        symbols=selected_symbols,
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
        output_prefix=artifacts.fold_diagnostic_prefix,
    )
    write_crypto_overlay_fold_diagnostic_summary_csv(
        diagnostic,
        artifacts.fold_diagnostic_summary_csv,
    )
    write_crypto_overlay_fold_symbol_summary_csv(
        diagnostic,
        artifacts.fold_diagnostic_symbol_summary_csv,
    )

    summary = _build_summary(
        data_source=normalized_source,
        price_csv=price_csv,
        quote_csv=quote_csv,
        selected_symbols=selected_symbols,
        health=health,
        sizing=sizing,
        sizing_gate=sizing_gate,
        components=components,
        asset_class_stability=asset_class_stability,
        diagnostic=diagnostic,
        max_live_fold_contribution=max_live_fold_contribution,
    )
    write_crypto_promotion_summary_csv(summary, artifacts.summary_csv)

    return CryptoPromotionPipelineResult(
        artifacts=artifacts,
        data_health=health,
        sizing=sizing,
        sizing_gate=sizing_gate,
        components=components,
        component_gate=component_gate,
        asset_class_stability=asset_class_stability,
        diagnostic=diagnostic,
        summary=summary,
    )


def write_crypto_promotion_summary_csv(
    summary: CryptoPromotionSummary,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "data_source",
                "price_csv",
                "quote_csv",
                "selected_symbols",
                "missing_competition_symbols",
                "crypto_symbols",
                "data_health",
                "data_health_issue_count",
                "best_sizing_label",
                "best_sizing_return_pct",
                "best_sizing_drawdown_pct",
                "best_sizing_sharpe_15m",
                "best_sizing_risk_score",
                "best_sizing_readiness",
                "best_sizing_reason",
                "component_full_return_pct",
                "no_crypto_return_delta_pct",
                "no_metals_return_delta_pct",
                "no_btc_sol_return_delta_pct",
                "stable_backup_label",
                "stable_backup_return_pct",
                "stable_backup_drawdown_pct",
                "stable_backup_sharpe_15m",
                "stable_backup_fold_contribution",
                "stable_backup_return_retention",
                "stable_backup_promotion_status",
                "stable_backup_reason",
                "fold_count",
                "strongest_fold",
                "strongest_fold_return_pct",
                "largest_positive_fold_contribution",
                "promotion_readiness",
                "live_ready",
                "promotion_reason",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "data_source": summary.data_source.value,
                "price_csv": summary.price_csv,
                "quote_csv": summary.quote_csv,
                "selected_symbols": " ".join(summary.selected_symbols),
                "missing_competition_symbols": " ".join(
                    summary.missing_competition_symbols
                ),
                "crypto_symbols": " ".join(summary.crypto_symbols),
                "data_health": summary.data_health.value,
                "data_health_issue_count": summary.data_health_issue_count,
                "best_sizing_label": summary.best_sizing_label,
                "best_sizing_return_pct": summary.best_sizing_return_pct,
                "best_sizing_drawdown_pct": summary.best_sizing_drawdown_pct,
                "best_sizing_sharpe_15m": summary.best_sizing_sharpe_15m,
                "best_sizing_risk_score": summary.best_sizing_risk_score,
                "best_sizing_readiness": summary.best_sizing_readiness.value,
                "best_sizing_reason": summary.best_sizing_reason,
                "component_full_return_pct": summary.component_full_return_pct,
                "no_crypto_return_delta_pct": summary.no_crypto_return_delta_pct,
                "no_metals_return_delta_pct": summary.no_metals_return_delta_pct,
                "no_btc_sol_return_delta_pct": summary.no_btc_sol_return_delta_pct,
                "stable_backup_label": summary.stable_backup_label,
                "stable_backup_return_pct": summary.stable_backup_return_pct,
                "stable_backup_drawdown_pct": summary.stable_backup_drawdown_pct,
                "stable_backup_sharpe_15m": summary.stable_backup_sharpe_15m,
                "stable_backup_fold_contribution": (
                    summary.stable_backup_fold_contribution
                ),
                "stable_backup_return_retention": (
                    summary.stable_backup_return_retention
                ),
                "stable_backup_promotion_status": (
                    summary.stable_backup_promotion_status
                ),
                "stable_backup_reason": summary.stable_backup_reason,
                "fold_count": summary.fold_count,
                "strongest_fold": summary.strongest_fold,
                "strongest_fold_return_pct": summary.strongest_fold_return_pct,
                "largest_positive_fold_contribution": (
                    summary.largest_positive_fold_contribution
                ),
                "promotion_readiness": summary.promotion_readiness.value,
                "live_ready": summary.live_ready,
                "promotion_reason": summary.promotion_reason,
            }
        )


def _artifacts_for_prefix(output_prefix: str | Path) -> CryptoPromotionArtifacts:
    prefix = Path(output_prefix)
    return CryptoPromotionArtifacts(
        data_health_csv=prefix.with_name(f"{prefix.name}_data_health.csv"),
        sizing_csv=prefix.with_name(f"{prefix.name}_sizing.csv"),
        sizing_gate_csv=prefix.with_name(f"{prefix.name}_sizing_gate.csv"),
        component_ablation_csv=prefix.with_name(
            f"{prefix.name}_component_ablation.csv"
        ),
        component_gate_csv=prefix.with_name(f"{prefix.name}_component_gate.csv"),
        asset_class_stability_csv=prefix.with_name(
            f"{prefix.name}_asset_class_stability.csv"
        ),
        fold_diagnostic_prefix=prefix.with_name(f"{prefix.name}_fold_diagnostic"),
        fold_diagnostic_summary_csv=prefix.with_name(
            f"{prefix.name}_fold_diagnostic_summary.csv"
        ),
        fold_diagnostic_symbol_summary_csv=prefix.with_name(
            f"{prefix.name}_fold_diagnostic_symbol_summary.csv"
        ),
        summary_csv=prefix.with_name(f"{prefix.name}_summary.csv"),
    )


def _build_summary(
    *,
    data_source: ResearchDataSource,
    price_csv: str,
    quote_csv: str,
    selected_symbols: tuple[str, ...],
    health: MarketDataHealthReport,
    sizing: CryptoOverlaySizingComparison,
    sizing_gate: tuple[ResearchCandidateGateRow, ...],
    components: CryptoOverlayComponentAblationResult,
    asset_class_stability: AssetClassStabilityOptimization,
    diagnostic: CryptoOverlayFoldDiagnostic,
    max_live_fold_contribution: float,
) -> CryptoPromotionSummary:
    best_candidate = sizing.best
    if best_candidate is None:
        raise ValueError("sizing comparison produced no candidates")
    best_gate = _gate_row_for_label(sizing_gate, best_candidate.label)
    readiness, live_ready, reason = _promotion_decision(
        data_source=data_source,
        health=health,
        best_gate=best_gate,
        diagnostic=diagnostic,
        max_live_fold_contribution=max_live_fold_contribution,
    )
    crypto_symbols = tuple(
        symbol
        for symbol in selected_symbols
        if instrument_for(symbol).asset_class == AssetClass.CRYPTO
    )
    missing_competition_symbols = tuple(
        instrument.symbol
        for instrument in DEFAULT_INSTRUMENTS
        if instrument.symbol not in selected_symbols
    )
    stable_backup = _best_stable_backup(asset_class_stability)
    return CryptoPromotionSummary(
        data_source=data_source,
        price_csv=price_csv,
        quote_csv=quote_csv,
        selected_symbols=selected_symbols,
        missing_competition_symbols=missing_competition_symbols,
        crypto_symbols=crypto_symbols,
        data_health=health.overall,
        data_health_issue_count=len(health.issues),
        best_sizing_label=best_candidate.label,
        best_sizing_return_pct=best_candidate.competition_metrics.return_pct,
        best_sizing_drawdown_pct=best_candidate.competition_metrics.max_drawdown_pct,
        best_sizing_sharpe_15m=best_candidate.competition_metrics.sharpe_15m,
        best_sizing_risk_score=best_candidate.risk_discipline.score,
        best_sizing_readiness=best_gate.readiness,
        best_sizing_reason=best_gate.reason,
        component_full_return_pct=_component_return(components, "full"),
        no_crypto_return_delta_pct=_component_delta(components, "no_crypto"),
        no_metals_return_delta_pct=_component_delta(components, "no_metals"),
        no_btc_sol_return_delta_pct=_component_delta(components, "no_btc_sol"),
        stable_backup_label="" if stable_backup is None else stable_backup.spec.label,
        stable_backup_return_pct=(
            0.0 if stable_backup is None else stable_backup.competition_metrics.return_pct
        ),
        stable_backup_drawdown_pct=(
            0.0
            if stable_backup is None
            else stable_backup.competition_metrics.max_drawdown_pct
        ),
        stable_backup_sharpe_15m=(
            0.0 if stable_backup is None else stable_backup.competition_metrics.sharpe_15m
        ),
        stable_backup_fold_contribution=(
            0.0
            if stable_backup is None
            else stable_backup.walk_forward.largest_positive_fold_contribution
        ),
        stable_backup_return_retention=(
            0.0 if stable_backup is None else stable_backup.return_retention
        ),
        stable_backup_promotion_status=(
            "" if stable_backup is None else stable_backup.promotion.status
        ),
        stable_backup_reason=(
            "no stable asset-class backup found"
            if stable_backup is None
            else stable_backup.promotion.reason
        ),
        fold_count=len(diagnostic.walk_forward.folds),
        strongest_fold=diagnostic.strongest_fold_index,
        strongest_fold_return_pct=diagnostic.strongest_fold_return_pct,
        largest_positive_fold_contribution=(
            diagnostic.walk_forward.largest_positive_fold_contribution
        ),
        promotion_readiness=readiness,
        live_ready=live_ready,
        promotion_reason=reason,
    )


def _promotion_decision(
    *,
    data_source: ResearchDataSource,
    health: MarketDataHealthReport,
    best_gate: ResearchCandidateGateRow,
    diagnostic: CryptoOverlayFoldDiagnostic,
    max_live_fold_contribution: float,
) -> tuple[ResearchReadiness, bool, str]:
    reject_reasons: list[str] = []
    paper_reasons: list[str] = []

    if health.overall == DataHealthSeverity.FAIL:
        reject_reasons.append("market data health failed")
    if best_gate.readiness == ResearchReadiness.REJECT:
        reject_reasons.append(f"sizing gate rejected: {best_gate.reason}")
    if diagnostic.promotion.status == "REJECT":
        reject_reasons.append(
            f"fold diagnostic rejected: {diagnostic.promotion.reason}"
        )

    if reject_reasons:
        return ResearchReadiness.REJECT, False, "; ".join(reject_reasons)

    if data_source != ResearchDataSource.OFFICIAL:
        paper_reasons.append(f"{data_source.value} data cannot be live-ready")
    if health.overall == DataHealthSeverity.WARN:
        paper_reasons.append("market data has warnings")
    if best_gate.readiness != ResearchReadiness.LIVE_READY:
        paper_reasons.append(f"sizing gate is {best_gate.readiness.value}: {best_gate.reason}")
    if diagnostic.walk_forward.largest_positive_fold_contribution > max_live_fold_contribution:
        paper_reasons.append(
            "largest positive fold contribution "
            f"{diagnostic.walk_forward.largest_positive_fold_contribution:.1%} "
            f"exceeds {max_live_fold_contribution:.1%} live threshold"
        )
    if diagnostic.promotion.status != "PROMOTE":
        paper_reasons.append(
            f"fold diagnostic status {diagnostic.promotion.status}: "
            f"{diagnostic.promotion.reason}"
        )

    if paper_reasons:
        return ResearchReadiness.PAPER_ONLY, False, "; ".join(paper_reasons)
    return (
        ResearchReadiness.LIVE_READY,
        True,
        "official data, clean health, stable folds, and low fold concentration",
    )


def _selected_spec_from_sizing(
    *,
    sizing: CryptoOverlaySizingComparison,
    specs: tuple[CryptoOverlaySizingSpec, ...],
) -> CryptoOverlaySizingSpec:
    best = sizing.best
    if best is None:
        raise ValueError("sizing comparison produced no candidates")
    for spec in specs:
        if spec.label == best.label:
            return spec
    raise ValueError(f"best sizing candidate {best.label!r} was not found in specs")


def _gate_row_for_label(
    rows: tuple[ResearchCandidateGateRow, ...],
    label: str,
) -> ResearchCandidateGateRow:
    for row in rows:
        if row.label == label:
            return row
    raise ValueError(f"gate output missing candidate {label!r}")


def _component_return(
    result: CryptoOverlayComponentAblationResult,
    label: str,
) -> float:
    for row in result.rows:
        if row.label == label:
            return row.competition_metrics.return_pct
    return 0.0


def _component_delta(
    result: CryptoOverlayComponentAblationResult,
    label: str,
) -> float:
    for row in result.rows:
        if row.label == label:
            return row.return_delta_pct
    return 0.0


def _best_stable_backup(
    result: AssetClassStabilityOptimization,
) -> AssetClassStabilityCandidate | None:
    stable_candidates = tuple(
        candidate
        for candidate in result.candidates
        if candidate.stability_status == "STABLE_PROFILE"
    )
    if not stable_candidates:
        return None
    return max(
        stable_candidates,
        key=lambda candidate: (
            candidate.competition_metrics.return_pct,
            candidate.stability_score,
            -candidate.competition_metrics.max_drawdown_pct,
        ),
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
    missing_prices = sorted(set(selected) - set(prices.symbols()))
    missing_quotes = sorted(set(selected) - set(quotes.symbols()))
    if missing_prices:
        raise ValueError(f"symbols missing price data: {', '.join(missing_prices)}")
    if missing_quotes:
        raise ValueError(f"symbols missing quote data: {', '.join(missing_quotes)}")
    return selected


def _spread_limits_by_symbol(symbols: tuple[str, ...]) -> dict[str, float]:
    return {
        symbol: instrument_for(symbol).max_spread_bps
        for symbol in symbols
    }
