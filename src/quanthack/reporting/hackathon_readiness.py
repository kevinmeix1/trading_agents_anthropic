from __future__ import annotations

import csv
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from quanthack.core.config import AppConfig
from quanthack.core.instruments import AssetClass, DEFAULT_INSTRUMENTS
from quanthack.market.market_data import PriceHistory, QuoteHistory


class ReadinessStatus(StrEnum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True)
class ReadinessCheck:
    name: str
    status: ReadinessStatus
    details: str


@dataclass(frozen=True)
class DataCoverageSnapshot:
    official_symbols: tuple[str, ...]
    price_symbols: tuple[str, ...]
    quote_symbols: tuple[str, ...]
    common_symbols: tuple[str, ...]
    missing_price_symbols: tuple[str, ...]
    missing_quote_symbols: tuple[str, ...]
    missing_common_symbols: tuple[str, ...]
    covered_asset_classes: tuple[AssetClass, ...]
    missing_asset_classes: tuple[AssetClass, ...]

    @property
    def official_symbol_coverage_fraction(self) -> float:
        if not self.official_symbols:
            return 0.0
        covered = [
            symbol for symbol in self.official_symbols if symbol in self.common_symbols
        ]
        return len(covered) / len(self.official_symbols)


@dataclass(frozen=True)
class PromotionSnapshot:
    status: str
    live_ready: bool
    reason: str
    failed_gates: tuple[str, ...] = ()
    positive_fold_fraction: float | None = None
    active_positive_fold_fraction: float | None = None
    non_negative_fold_fraction: float | None = None
    median_active_return_pct: float | None = None
    worst_drawdown_pct: float | None = None
    risk_discipline_score: float | None = None
    total_fills: int | None = None


@dataclass(frozen=True)
class HackathonReadinessReport:
    overall_status: ReadinessStatus
    checks: tuple[ReadinessCheck, ...]
    coverage: DataCoverageSnapshot
    promotion: PromotionSnapshot | None

    @property
    def ready_for_live(self) -> bool:
        return self.overall_status == ReadinessStatus.PASS

    def summary_lines(self) -> tuple[str, ...]:
        lines = [
            "Hackathon Readiness",
            f"  Overall: {self.overall_status.value}",
            (
                "  Data coverage: "
                f"{len(self.coverage.common_symbols)}/{len(self.coverage.official_symbols)} "
                "official instruments"
            ),
        ]
        if self.coverage.missing_asset_classes:
            lines.append(
                "  Missing asset classes: "
                + ", ".join(value.value for value in self.coverage.missing_asset_classes)
            )
        if self.promotion is not None:
            lines.append(
                "  Candidate: "
                f"{self.promotion.status} "
                f"({'live-ready' if self.promotion.live_ready else 'not live-ready'})"
            )
        lines.extend(
            f"  [{check.status.value}] {check.name}: {check.details}"
            for check in self.checks
        )
        return tuple(lines)

    def to_markdown(self) -> str:
        lines = [
            "# Hackathon Readiness Report",
            "",
            f"Overall status: **{self.overall_status.value}**",
            "",
            "## Checks",
            "",
            "| Check | Status | Details |",
            "| --- | --- | --- |",
        ]
        for check in self.checks:
            lines.append(
                f"| {check.name} | {check.status.value} | {check.details} |"
            )

        lines.extend(
            [
                "",
                "## Data Coverage",
                "",
                f"- Official instruments: {len(self.coverage.official_symbols)}",
                f"- Price CSV symbols: {', '.join(self.coverage.price_symbols) or 'none'}",
                f"- Quote CSV symbols: {', '.join(self.coverage.quote_symbols) or 'none'}",
                f"- Common symbols: {', '.join(self.coverage.common_symbols) or 'none'}",
                (
                    "- Missing common symbols: "
                    f"{', '.join(self.coverage.missing_common_symbols) or 'none'}"
                ),
                (
                    "- Covered asset classes: "
                    + (
                        ", ".join(value.value for value in self.coverage.covered_asset_classes)
                        or "none"
                    )
                ),
                (
                    "- Missing asset classes: "
                    + (
                        ", ".join(value.value for value in self.coverage.missing_asset_classes)
                        or "none"
                    )
                ),
            ]
        )

        lines.extend(["", "## Candidate Promotion", ""])
        if self.promotion is None:
            lines.append("No promotion audit was loaded.")
        else:
            lines.extend(
                [
                    f"- Status: {self.promotion.status}",
                    f"- Live ready: {'yes' if self.promotion.live_ready else 'no'}",
                    f"- Reason: {self.promotion.reason}",
                    (
                        "- Failed gates: "
                        f"{', '.join(self.promotion.failed_gates) or 'none'}"
                    ),
                ]
            )
            if self.promotion.positive_fold_fraction is not None:
                lines.extend(
                    [
                        (
                            "- Positive folds: "
                            f"{self.promotion.positive_fold_fraction:.1%}"
                        ),
                        (
                            "- Active positive folds: "
                            f"{self.promotion.active_positive_fold_fraction or 0.0:.1%}"
                        ),
                        (
                            "- Non-negative folds: "
                            f"{self.promotion.non_negative_fold_fraction or 0.0:.1%}"
                        ),
                        (
                            "- Median active return: "
                            f"{self.promotion.median_active_return_pct or 0.0:.3%}"
                        ),
                        (
                            "- Worst drawdown: "
                            f"{self.promotion.worst_drawdown_pct or 0.0:.3%}"
                        ),
                        (
                            "- Risk discipline: "
                            f"{self.promotion.risk_discipline_score or 0.0:.1f}/100"
                        ),
                        f"- Evaluation fills: {self.promotion.total_fills or 0}",
                    ]
                )

        lines.extend(
            [
                "",
                "## Verdict",
                "",
                _verdict_text(self),
                "",
            ]
        )
        return "\n".join(lines)


def build_hackathon_readiness_report(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    promotion_csv: str | Path | None = None,
    summary_csv: str | Path | None = None,
) -> HackathonReadinessReport:
    coverage = build_data_coverage_snapshot(prices=prices, quotes=quotes)
    promotion = load_promotion_snapshot(
        promotion_csv=promotion_csv,
        summary_csv=summary_csv,
    )
    checks = (
        _data_presence_check(coverage),
        _asset_class_check(coverage),
        _official_symbol_coverage_check(coverage),
        _risk_limit_check(config),
        _promotion_check(promotion),
    )
    overall = _overall_status(checks)
    return HackathonReadinessReport(
        overall_status=overall,
        checks=checks,
        coverage=coverage,
        promotion=promotion,
    )


def build_data_coverage_snapshot(
    *,
    prices: PriceHistory,
    quotes: QuoteHistory,
) -> DataCoverageSnapshot:
    official_symbols = tuple(instrument.symbol for instrument in DEFAULT_INSTRUMENTS)
    price_symbols = tuple(sorted(prices.symbols()))
    quote_symbols = tuple(sorted(quotes.symbols()))
    common_symbols = tuple(
        symbol
        for symbol in official_symbols
        if symbol in price_symbols and symbol in quote_symbols
    )
    covered_asset_classes = tuple(
        asset_class
        for asset_class in AssetClass
        if any(
            instrument.symbol in common_symbols
            for instrument in DEFAULT_INSTRUMENTS
            if instrument.asset_class == asset_class
        )
    )
    missing_asset_classes = tuple(
        asset_class for asset_class in AssetClass if asset_class not in covered_asset_classes
    )
    return DataCoverageSnapshot(
        official_symbols=official_symbols,
        price_symbols=price_symbols,
        quote_symbols=quote_symbols,
        common_symbols=common_symbols,
        missing_price_symbols=tuple(
            symbol for symbol in official_symbols if symbol not in price_symbols
        ),
        missing_quote_symbols=tuple(
            symbol for symbol in official_symbols if symbol not in quote_symbols
        ),
        missing_common_symbols=tuple(
            symbol for symbol in official_symbols if symbol not in common_symbols
        ),
        covered_asset_classes=covered_asset_classes,
        missing_asset_classes=missing_asset_classes,
    )


def load_promotion_snapshot(
    *,
    promotion_csv: str | Path | None,
    summary_csv: str | Path | None,
) -> PromotionSnapshot | None:
    promotion_rows = _read_csv_rows(promotion_csv)
    summary_rows = _read_csv_rows(summary_csv)
    if not promotion_rows and not summary_rows:
        return None

    promotion_row = promotion_rows[0] if promotion_rows else {}
    summary_row = summary_rows[0] if summary_rows else {}
    failed_gates = tuple(
        row.get("gate_id", "")
        for row in promotion_rows
        if row.get("passed") == "no" and row.get("gate_id")
    )
    return PromotionSnapshot(
        status=str(promotion_row.get("status") or "UNKNOWN"),
        live_ready=str(promotion_row.get("live_ready") or "").lower() == "yes",
        reason=str(promotion_row.get("decision_reason") or "promotion audit not loaded"),
        failed_gates=failed_gates,
        positive_fold_fraction=_optional_float(summary_row, "positive_fold_fraction"),
        active_positive_fold_fraction=_optional_float(
            summary_row,
            "active_positive_fold_fraction",
        ),
        non_negative_fold_fraction=_optional_float(summary_row, "non_negative_fold_fraction"),
        median_active_return_pct=_optional_float(
            summary_row,
            "median_active_test_return_pct",
        ),
        worst_drawdown_pct=_optional_float(summary_row, "worst_test_drawdown_pct"),
        risk_discipline_score=_optional_float(
            summary_row,
            "average_risk_discipline_score",
        ),
        total_fills=_optional_int(summary_row, "total_evaluation_fills"),
    )


def write_hackathon_readiness_markdown(
    report: HackathonReadinessReport,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.to_markdown(), encoding="utf-8")


def _data_presence_check(coverage: DataCoverageSnapshot) -> ReadinessCheck:
    if not coverage.common_symbols:
        return ReadinessCheck(
            name="data presence",
            status=ReadinessStatus.FAIL,
            details="no official instruments have both price and quote data",
        )
    return ReadinessCheck(
        name="data presence",
        status=ReadinessStatus.PASS,
        details=(
            f"{len(coverage.common_symbols)} official instruments have both "
            "price and quote data"
        ),
    )


def _asset_class_check(coverage: DataCoverageSnapshot) -> ReadinessCheck:
    if coverage.missing_asset_classes:
        return ReadinessCheck(
            name="asset-class coverage",
            status=ReadinessStatus.FAIL,
            details=(
                "missing "
                + ", ".join(value.value for value in coverage.missing_asset_classes)
            ),
        )
    return ReadinessCheck(
        name="asset-class coverage",
        status=ReadinessStatus.PASS,
        details="forex, metals, and crypto all have usable local data",
    )


def _official_symbol_coverage_check(coverage: DataCoverageSnapshot) -> ReadinessCheck:
    missing_count = len(coverage.missing_common_symbols)
    if missing_count == 0:
        return ReadinessCheck(
            name="official-symbol coverage",
            status=ReadinessStatus.PASS,
            details="all 15 official instruments are present in price and quote CSVs",
        )
    status = ReadinessStatus.WARN
    if coverage.official_symbol_coverage_fraction < 0.5:
        status = ReadinessStatus.FAIL
    return ReadinessCheck(
        name="official-symbol coverage",
        status=status,
        details=(
            f"{missing_count} official instruments missing from common CSV coverage: "
            + ", ".join(coverage.missing_common_symbols)
        ),
    )


def _risk_limit_check(config: AppConfig) -> ReadinessCheck:
    limits = config.risk
    problems: list[str] = []
    warnings: list[str] = []
    if limits.max_gross_leverage > 2.0:
        warnings.append(f"gross leverage cap {limits.max_gross_leverage:.2f}x above 2.0x")
    if limits.max_gross_leverage > 28.0:
        problems.append("gross leverage cap can enter official penalty zone")
    if limits.max_symbol_notional_pct > 0.25:
        warnings.append(
            f"single-symbol cap {limits.max_symbol_notional_pct:.1%} above 25%"
        )
    if limits.max_position_loss_pct <= 0:
        warnings.append("per-position stop-loss disabled")
    if limits.max_daily_loss_pct > 0.025:
        warnings.append("daily loss cap looser than 2.5% starter guard")
    if limits.min_margin_level_pct < 300:
        problems.append("margin floor below 300% starter guard")
    if problems:
        return ReadinessCheck(
            name="risk limits",
            status=ReadinessStatus.FAIL,
            details="; ".join(problems + warnings),
        )
    if warnings:
        return ReadinessCheck(
            name="risk limits",
            status=ReadinessStatus.WARN,
            details="; ".join(warnings),
        )
    return ReadinessCheck(
        name="risk limits",
        status=ReadinessStatus.PASS,
        details="internal risk limits stay below official penalty zones",
    )


def _promotion_check(promotion: PromotionSnapshot | None) -> ReadinessCheck:
    if promotion is None:
        return ReadinessCheck(
            name="candidate promotion",
            status=ReadinessStatus.WARN,
            details="no adaptive promotion audit loaded",
        )
    if promotion.live_ready and promotion.status == "PROMOTE":
        return ReadinessCheck(
            name="candidate promotion",
            status=ReadinessStatus.PASS,
            details=promotion.reason,
        )
    if promotion.status == "PAPER_ONLY":
        return ReadinessCheck(
            name="candidate promotion",
            status=ReadinessStatus.WARN,
            details=promotion.reason,
        )
    return ReadinessCheck(
        name="candidate promotion",
        status=ReadinessStatus.FAIL,
        details=promotion.reason,
    )


def _overall_status(checks: tuple[ReadinessCheck, ...]) -> ReadinessStatus:
    if any(check.status == ReadinessStatus.FAIL for check in checks):
        return ReadinessStatus.FAIL
    if any(check.status == ReadinessStatus.WARN for check in checks):
        return ReadinessStatus.WARN
    return ReadinessStatus.PASS


def _read_csv_rows(path: str | Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    csv_path = Path(path)
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _optional_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key)
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _optional_int(row: dict[str, str], key: str) -> int | None:
    value = row.get(key)
    if value in {None, ""}:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _verdict_text(report: HackathonReadinessReport) -> str:
    if report.overall_status == ReadinessStatus.PASS:
        return (
            "Ready for a tightly controlled live dry-run or manual MT5 execution "
            "review. Continue monitoring risk and data health."
        )
    if report.overall_status == ReadinessStatus.WARN:
        return (
            "Paper-ready with warnings. Do not enable automated live execution "
            "until warnings are understood and accepted."
        )
    return (
        "Not hackathon-live ready. Keep researching/backtesting and fix failed "
        "checks before automated MT5 execution."
    )
