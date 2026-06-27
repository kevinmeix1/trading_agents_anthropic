from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from quanthack.backtesting.competition_score import (
    CompetitionMetrics,
    RiskDisciplineReport,
    build_competition_metrics,
    build_risk_discipline_report,
    risk_samples_from_portfolio_equity,
)
from quanthack.backtesting.crypto_overlay_sizing_compare import (
    CryptoOverlaySizingSpec,
    _aggressive_strategy_map,
    _multipliers_for_spec,
    _run_strategy_map,
    _selected_symbols,
    _session_policy_for_spec,
)
from quanthack.backtesting.portfolio_backtest import PortfolioBacktestResult
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


DEFAULT_COMPONENT_BASE_SPEC = CryptoOverlaySizingSpec(
    label="btc075_sol100_reversion075_london",
    crypto_multiplier=0.75,
    btc_multiplier=0.75,
    sol_multiplier=1.0,
    crypto_allowed_utc_hours=tuple(range(7, 17)),
)


@dataclass(frozen=True)
class CryptoOverlayComponentAblationSpec:
    label: str
    disabled_symbols: tuple[str, ...] = ()
    disabled_asset_classes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("component ablation label cannot be empty")
        normalized_asset_classes = tuple(
            asset_class.strip().upper()
            for asset_class in self.disabled_asset_classes
            if asset_class.strip()
        )
        invalid = [
            asset_class
            for asset_class in normalized_asset_classes
            if asset_class not in AssetClass.__members__
        ]
        if invalid:
            valid = ", ".join(AssetClass.__members__)
            raise ValueError(
                f"unknown disabled asset class {invalid[0]!r}; expected one of {valid}"
            )
        object.__setattr__(
            self,
            "disabled_symbols",
            tuple(instrument_for(symbol).symbol for symbol in self.disabled_symbols),
        )
        object.__setattr__(
            self,
            "disabled_asset_classes",
            normalized_asset_classes,
        )


DEFAULT_COMPONENT_ABLATIONS = (
    CryptoOverlayComponentAblationSpec(label="full"),
    CryptoOverlayComponentAblationSpec(label="no_crypto", disabled_asset_classes=("CRYPTO",)),
    CryptoOverlayComponentAblationSpec(label="no_metals", disabled_asset_classes=("METAL",)),
    CryptoOverlayComponentAblationSpec(label="no_fx", disabled_asset_classes=("FOREX",)),
    CryptoOverlayComponentAblationSpec(label="no_btc", disabled_symbols=("BTCUSD",)),
    CryptoOverlayComponentAblationSpec(label="no_sol", disabled_symbols=("SOLUSD",)),
    CryptoOverlayComponentAblationSpec(label="no_btc_sol", disabled_symbols=("BTCUSD", "SOLUSD")),
    CryptoOverlayComponentAblationSpec(
        label="no_crypto_reversion",
        disabled_symbols=("BARUSD", "ETHUSD", "XRPUSD"),
    ),
    CryptoOverlayComponentAblationSpec(
        label="crypto_only",
        disabled_asset_classes=("FOREX", "METAL"),
    ),
    CryptoOverlayComponentAblationSpec(
        label="metals_only",
        disabled_asset_classes=("FOREX", "CRYPTO"),
    ),
    CryptoOverlayComponentAblationSpec(
        label="fx_only",
        disabled_asset_classes=("METAL", "CRYPTO"),
    ),
)


@dataclass(frozen=True)
class CryptoOverlayComponentAblationRow:
    label: str
    disabled_symbols: tuple[str, ...]
    disabled_asset_classes: tuple[str, ...]
    active_symbols: tuple[str, ...]
    multipliers_by_symbol: tuple[tuple[str, float], ...]
    result: PortfolioBacktestResult
    competition_metrics: CompetitionMetrics
    risk_discipline: RiskDisciplineReport
    return_delta_pct: float
    drawdown_delta_pct: float
    sharpe_delta: float
    walk_forward: FixedWarmupPortfolioWalkForwardResult | None = None
    promotion: FixedWarmupPromotionDecision | None = None

    @property
    def multiplier_map_text(self) -> str:
        return " ".join(
            f"{symbol}={multiplier:.3f}"
            for symbol, multiplier in self.multipliers_by_symbol
        )

    @property
    def disabled_symbols_text(self) -> str:
        return " ".join(self.disabled_symbols)

    @property
    def disabled_asset_classes_text(self) -> str:
        return " ".join(self.disabled_asset_classes)

    @property
    def active_symbols_text(self) -> str:
        return " ".join(self.active_symbols)


@dataclass(frozen=True)
class CryptoOverlayComponentAblationResult:
    base_spec: CryptoOverlaySizingSpec
    base_strategy: str
    official_symbols: tuple[str, ...]
    crypto_symbols: tuple[str, ...]
    rows: tuple[CryptoOverlayComponentAblationRow, ...]
    walk_forward_enabled: bool

    @property
    def baseline(self) -> CryptoOverlayComponentAblationRow | None:
        if not self.rows:
            return None
        return self.rows[0]


def compare_crypto_overlay_components(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    base_strategy: str = "macd_momentum",
    base_spec: CryptoOverlaySizingSpec = DEFAULT_COMPONENT_BASE_SPEC,
    specs: tuple[CryptoOverlayComponentAblationSpec, ...] | None = DEFAULT_COMPONENT_ABLATIONS,
    symbols: tuple[str, ...] | None = None,
    run_walk_forward: bool = True,
    train_size: int = 96,
    test_size: int = 48,
    step_size: int = 48,
) -> CryptoOverlayComponentAblationResult:
    ablation_specs = specs or DEFAULT_COMPONENT_ABLATIONS
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
    strategy_map = _aggressive_strategy_map(
        official_symbols=official_symbols,
        crypto_symbols=crypto_symbols,
        base_strategy=normalized_base,
    )
    base_multipliers = _all_symbol_multipliers(
        selected_symbols=selected_symbols,
        crypto_multipliers=_multipliers_for_spec(
            spec=base_spec,
            crypto_symbols=crypto_symbols,
            strategy_by_symbol=strategy_map,
        ),
    )
    session_policy = _session_policy_for_spec(base_spec)

    raw_rows: list[
        tuple[
            CryptoOverlayComponentAblationSpec,
            dict[str, float],
            PortfolioBacktestResult,
            CompetitionMetrics,
            RiskDisciplineReport,
            FixedWarmupPortfolioWalkForwardResult | None,
            FixedWarmupPromotionDecision | None,
        ]
    ] = []
    for spec in ablation_specs:
        multipliers = _component_multipliers(
            selected_symbols=selected_symbols,
            base_multipliers=base_multipliers,
            spec=spec,
        )
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
        raw_rows.append(
            (
                spec,
                multipliers,
                result,
                metrics,
                risk_discipline,
                walk_forward,
                promotion,
            )
        )

    baseline_metrics = raw_rows[0][3]
    rows = tuple(
        CryptoOverlayComponentAblationRow(
            label=spec.label,
            disabled_symbols=spec.disabled_symbols,
            disabled_asset_classes=spec.disabled_asset_classes,
            active_symbols=tuple(
                symbol
                for symbol, multiplier in sorted(multipliers.items())
                if multiplier > 0
            ),
            multipliers_by_symbol=tuple(sorted(multipliers.items())),
            result=result,
            competition_metrics=metrics,
            risk_discipline=risk_discipline,
            return_delta_pct=metrics.return_pct - baseline_metrics.return_pct,
            drawdown_delta_pct=metrics.max_drawdown_pct
            - baseline_metrics.max_drawdown_pct,
            sharpe_delta=metrics.sharpe_15m - baseline_metrics.sharpe_15m,
            walk_forward=walk_forward,
            promotion=promotion,
        )
        for spec, multipliers, result, metrics, risk_discipline, walk_forward, promotion in raw_rows
    )
    return CryptoOverlayComponentAblationResult(
        base_spec=base_spec,
        base_strategy=normalized_base,
        official_symbols=official_symbols,
        crypto_symbols=crypto_symbols,
        rows=rows,
        walk_forward_enabled=run_walk_forward,
    )


def write_crypto_overlay_component_ablation_csv(
    result: CryptoOverlayComponentAblationResult,
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
                "disabled_symbols",
                "disabled_asset_classes",
                "active_symbols",
                "multiplier_map",
                "return_pct",
                "return_delta_pct",
                "max_drawdown_pct",
                "drawdown_delta_pct",
                "sharpe_15m",
                "sharpe_delta",
                "risk_discipline_score",
                "trade_count",
                "total_pnl_usd",
                "promotion_status",
                "promotion_reason",
                "wf_positive_fold_fraction",
                "wf_active_positive_fold_fraction",
                "wf_non_negative_fold_fraction",
                "wf_median_active_test_return_pct",
                "wf_largest_positive_fold_contribution",
            ],
        )
        writer.writeheader()
        for rank, row in enumerate(result.rows, start=1):
            metrics = row.competition_metrics
            writer.writerow(
                {
                    "rank": rank,
                    "label": row.label,
                    "disabled_symbols": row.disabled_symbols_text,
                    "disabled_asset_classes": row.disabled_asset_classes_text,
                    "active_symbols": row.active_symbols_text,
                    "multiplier_map": row.multiplier_map_text,
                    "return_pct": metrics.return_pct,
                    "return_delta_pct": row.return_delta_pct,
                    "max_drawdown_pct": metrics.max_drawdown_pct,
                    "drawdown_delta_pct": row.drawdown_delta_pct,
                    "sharpe_15m": metrics.sharpe_15m,
                    "sharpe_delta": row.sharpe_delta,
                    "risk_discipline_score": row.risk_discipline.score,
                    "trade_count": metrics.trade_count,
                    "total_pnl_usd": row.result.total_pnl_usd,
                    "promotion_status": row.promotion.status if row.promotion else "",
                    "promotion_reason": row.promotion.reason if row.promotion else "",
                    **_walk_forward_columns(row.walk_forward),
                }
            )


def _all_symbol_multipliers(
    *,
    selected_symbols: tuple[str, ...],
    crypto_multipliers: dict[str, float],
) -> dict[str, float]:
    return {
        symbol: crypto_multipliers.get(symbol, 1.0)
        for symbol in selected_symbols
    }


def _component_multipliers(
    *,
    selected_symbols: tuple[str, ...],
    base_multipliers: dict[str, float],
    spec: CryptoOverlayComponentAblationSpec,
) -> dict[str, float]:
    disabled_symbols = set(spec.disabled_symbols)
    disabled_asset_classes = set(spec.disabled_asset_classes)
    multipliers = dict(base_multipliers)
    for symbol in selected_symbols:
        asset_class = instrument_for(symbol).asset_class.name
        if symbol in disabled_symbols or asset_class in disabled_asset_classes:
            multipliers[symbol] = 0.0
    return multipliers


def _walk_forward_columns(
    walk_forward: FixedWarmupPortfolioWalkForwardResult | None,
) -> dict[str, float | str]:
    if walk_forward is None:
        return {
            "wf_positive_fold_fraction": "",
            "wf_active_positive_fold_fraction": "",
            "wf_non_negative_fold_fraction": "",
            "wf_median_active_test_return_pct": "",
            "wf_largest_positive_fold_contribution": "",
        }
    return {
        "wf_positive_fold_fraction": walk_forward.positive_fold_fraction,
        "wf_active_positive_fold_fraction": (
            walk_forward.active_positive_fold_fraction
        ),
        "wf_non_negative_fold_fraction": walk_forward.non_negative_fold_fraction,
        "wf_median_active_test_return_pct": (
            walk_forward.median_active_test_return_pct
        ),
        "wf_largest_positive_fold_contribution": (
            walk_forward.largest_positive_fold_contribution
        ),
    }
