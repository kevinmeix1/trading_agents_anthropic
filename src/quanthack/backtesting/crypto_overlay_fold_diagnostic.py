from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from quanthack.backtesting.crypto_overlay_sizing_compare import (
    CryptoOverlaySizingSpec,
    _aggressive_strategy_map,
    _multipliers_for_spec,
    _selected_symbols,
    _session_policy_for_spec,
)
from quanthack.backtesting.portfolio_fixed_warmup_walk_forward import (
    FixedWarmupPortfolioWalkForwardResult,
    FixedWarmupPromotionDecision,
    decide_fixed_warmup_promotion,
    run_fixed_warmup_portfolio_walk_forward,
    write_fixed_warmup_folds_csv,
)
from quanthack.core.config import AppConfig
from quanthack.core.instruments import AssetClass, instrument_for
from quanthack.market.market_data import PriceHistory, QuoteHistory
from quanthack.reporting.fold_trade_attribution import (
    FoldTradeAttributionReport,
    build_fold_trade_attribution_report,
    write_fold_trade_attribution_csv,
)
from quanthack.strategies.strategy import normalize_strategy_name


DEFAULT_FOLD_DIAGNOSTIC_SPEC = CryptoOverlaySizingSpec(
    label="btc075_sol100_reversion075_london",
    crypto_multiplier=0.75,
    btc_multiplier=0.75,
    sol_multiplier=1.0,
    crypto_allowed_utc_hours=tuple(range(7, 17)),
)


@dataclass(frozen=True)
class CryptoOverlayFoldDiagnostic:
    spec: CryptoOverlaySizingSpec
    base_strategy: str
    official_symbols: tuple[str, ...]
    crypto_symbols: tuple[str, ...]
    strategy_by_symbol: tuple[tuple[str, str], ...]
    multipliers_by_symbol: tuple[tuple[str, float], ...]
    walk_forward: FixedWarmupPortfolioWalkForwardResult
    promotion: FixedWarmupPromotionDecision
    attribution: FoldTradeAttributionReport

    @property
    def strongest_fold_index(self) -> int:
        if not self.walk_forward.folds:
            return 0
        strongest = max(
            self.walk_forward.folds,
            key=lambda fold: fold.metrics.return_pct,
        )
        return strongest.fold_index

    @property
    def strongest_fold_return_pct(self) -> float:
        if not self.walk_forward.folds:
            return 0.0
        strongest = max(
            self.walk_forward.folds,
            key=lambda fold: fold.metrics.return_pct,
        )
        return strongest.metrics.return_pct

    @property
    def strongest_symbol(self) -> str:
        rows = self.attribution.strongest_rows
        return rows[0].symbol if rows else ""

    @property
    def strongest_symbol_realized_pnl_usd(self) -> float:
        rows = self.attribution.strongest_rows
        return rows[0].realized_pnl_usd if rows else 0.0


def build_crypto_overlay_fold_diagnostic(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    base_strategy: str = "macd_momentum",
    spec: CryptoOverlaySizingSpec = DEFAULT_FOLD_DIAGNOSTIC_SPEC,
    symbols: tuple[str, ...] | None = None,
    train_size: int = 96,
    test_size: int = 48,
    step_size: int = 48,
    output_prefix: str | Path = "outputs/research/crypto_overlay_fold_diagnostic",
) -> CryptoOverlayFoldDiagnostic:
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
    multipliers = _multipliers_for_spec(
        spec=spec,
        crypto_symbols=crypto_symbols,
        strategy_by_symbol=strategy_map,
    )
    session_policy = _session_policy_for_spec(spec)
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
    prefix = Path(output_prefix)
    folds_path = prefix.with_name(f"{prefix.name}_folds.csv")
    fills_path = prefix.with_name(f"{prefix.name}_fills.csv")
    attribution_path = prefix.with_name(f"{prefix.name}_attribution.csv")

    write_fixed_warmup_folds_csv(walk_forward, folds_path)
    write_fixed_warmup_evaluation_fills_csv(walk_forward, fills_path)
    attribution = build_fold_trade_attribution_report(
        fills_csv=fills_path,
        folds_csv=folds_path,
    )
    write_fold_trade_attribution_csv(attribution, attribution_path)

    return CryptoOverlayFoldDiagnostic(
        spec=spec,
        base_strategy=normalized_base,
        official_symbols=official_symbols,
        crypto_symbols=crypto_symbols,
        strategy_by_symbol=tuple(sorted(strategy_map.items())),
        multipliers_by_symbol=tuple(sorted(multipliers.items())),
        walk_forward=walk_forward,
        promotion=promotion,
        attribution=attribution,
    )


def write_fixed_warmup_evaluation_fills_csv(
    result: FixedWarmupPortfolioWalkForwardResult,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "timestamp",
                "symbol",
                "side",
                "fill_price",
                "trade_units",
                "turnover_notional_usd",
                "requested_notional_usd",
                "adjusted_notional_usd",
                "risk_reason",
                "primary_signal",
                "supporting_signals",
                "conflicting_signals",
            ],
        )
        writer.writeheader()
        for fold in result.folds:
            for fill in fold.evaluation.fills:
                writer.writerow(
                    {
                        "timestamp": fill.timestamp,
                        "symbol": fill.symbol,
                        "side": fill.side.value,
                        "fill_price": fill.fill_price,
                        "trade_units": fill.trade_units,
                        "turnover_notional_usd": fill.turnover_notional_usd,
                        "requested_notional_usd": fill.requested_notional_usd,
                        "adjusted_notional_usd": fill.adjusted_notional_usd,
                        "risk_reason": fill.risk_reason,
                        "primary_signal": fill.primary_signal,
                        "supporting_signals": "|".join(fill.supporting_signals),
                        "conflicting_signals": "|".join(fill.conflicting_signals),
                    }
                )


def write_crypto_overlay_fold_diagnostic_summary_csv(
    diagnostic: CryptoOverlayFoldDiagnostic,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "label",
                "base_strategy",
                "crypto_allowed_utc_hours",
                "official_symbols",
                "crypto_symbols",
                "strategy_map",
                "multiplier_map",
                "folds",
                "positive_fold_fraction",
                "active_positive_fold_fraction",
                "non_negative_fold_fraction",
                "median_active_test_return_pct",
                "largest_positive_fold_contribution",
                "strongest_fold",
                "strongest_fold_return_pct",
                "promotion_status",
                "promotion_reason",
                "strongest_attribution_symbol",
                "strongest_attribution_realized_pnl_usd",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "label": diagnostic.spec.label,
                "base_strategy": diagnostic.base_strategy,
                "crypto_allowed_utc_hours": _hours_text(
                    diagnostic.spec.crypto_allowed_utc_hours
                ),
                "official_symbols": " ".join(diagnostic.official_symbols),
                "crypto_symbols": " ".join(diagnostic.crypto_symbols),
                "strategy_map": " ".join(
                    f"{symbol}={strategy}"
                    for symbol, strategy in diagnostic.strategy_by_symbol
                ),
                "multiplier_map": " ".join(
                    f"{symbol}={multiplier:.3f}"
                    for symbol, multiplier in diagnostic.multipliers_by_symbol
                ),
                "folds": len(diagnostic.walk_forward.folds),
                "positive_fold_fraction": (
                    diagnostic.walk_forward.positive_fold_fraction
                ),
                "active_positive_fold_fraction": (
                    diagnostic.walk_forward.active_positive_fold_fraction
                ),
                "non_negative_fold_fraction": (
                    diagnostic.walk_forward.non_negative_fold_fraction
                ),
                "median_active_test_return_pct": (
                    diagnostic.walk_forward.median_active_test_return_pct
                ),
                "largest_positive_fold_contribution": (
                    diagnostic.walk_forward.largest_positive_fold_contribution
                ),
                "strongest_fold": diagnostic.strongest_fold_index,
                "strongest_fold_return_pct": diagnostic.strongest_fold_return_pct,
                "promotion_status": diagnostic.promotion.status,
                "promotion_reason": diagnostic.promotion.reason,
                "strongest_attribution_symbol": diagnostic.strongest_symbol,
                "strongest_attribution_realized_pnl_usd": (
                    diagnostic.strongest_symbol_realized_pnl_usd
                ),
            }
        )


def write_crypto_overlay_fold_symbol_summary_csv(
    diagnostic: CryptoOverlayFoldDiagnostic,
    path: str | Path,
) -> None:
    grouped: dict[tuple[int, str, str], dict[str, float | int]] = {}
    for row in diagnostic.attribution.rows:
        asset_class = instrument_for(row.symbol).asset_class.value
        key = (row.fold, row.symbol, asset_class)
        bucket = grouped.setdefault(
            key,
            {
                "fold_return_pct": row.fold_return_pct,
                "fills": 0,
                "realized_events": 0,
                "wins": 0,
                "losses": 0,
                "realized_pnl_usd": 0.0,
                "turnover_notional_usd": 0.0,
                "adjusted_notional_usd": 0.0,
            },
        )
        bucket["fills"] = int(bucket["fills"]) + row.fills
        bucket["realized_events"] = int(bucket["realized_events"]) + row.realized_events
        bucket["wins"] = int(bucket["wins"]) + row.wins
        bucket["losses"] = int(bucket["losses"]) + row.losses
        bucket["realized_pnl_usd"] = (
            float(bucket["realized_pnl_usd"]) + row.realized_pnl_usd
        )
        bucket["turnover_notional_usd"] = (
            float(bucket["turnover_notional_usd"]) + row.turnover_notional_usd
        )
        bucket["adjusted_notional_usd"] = (
            float(bucket["adjusted_notional_usd"]) + row.adjusted_notional_usd
        )

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "fold",
                "fold_return_pct",
                "symbol",
                "asset_class",
                "fills",
                "realized_events",
                "wins",
                "losses",
                "win_rate",
                "realized_pnl_usd",
                "turnover_notional_usd",
                "adjusted_notional_usd",
            ],
        )
        writer.writeheader()
        for (fold, symbol, asset_class), values in sorted(
            grouped.items(),
            key=lambda item: (item[0][0], -float(item[1]["realized_pnl_usd"])),
        ):
            realized_events = int(values["realized_events"])
            wins = int(values["wins"])
            writer.writerow(
                {
                    "fold": fold,
                    "fold_return_pct": values["fold_return_pct"],
                    "symbol": symbol,
                    "asset_class": asset_class,
                    "fills": values["fills"],
                    "realized_events": realized_events,
                    "wins": wins,
                    "losses": values["losses"],
                    "win_rate": 0.0 if realized_events == 0 else wins / realized_events,
                    "realized_pnl_usd": values["realized_pnl_usd"],
                    "turnover_notional_usd": values["turnover_notional_usd"],
                    "adjusted_notional_usd": values["adjusted_notional_usd"],
                }
            )


def _hours_text(hours: tuple[int, ...] | None) -> str:
    if hours is None:
        return "all"
    return "|".join(str(hour) for hour in hours)
