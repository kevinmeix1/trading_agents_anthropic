from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from quanthack.market.market_data import PriceHistory, QuoteHistory


class DataHealthSeverity(StrEnum):
    OK = "OK"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True)
class DataHealthIssue:
    severity: DataHealthSeverity
    symbol: str
    category: str
    details: str


@dataclass(frozen=True)
class SymbolDataHealth:
    symbol: str
    price_count: int
    quote_count: int
    price_start: datetime | None
    price_end: datetime | None
    quote_start: datetime | None
    quote_end: datetime | None
    duplicate_price_timestamps: int
    duplicate_quote_timestamps: int
    missing_quote_timestamps: tuple[datetime, ...]
    missing_price_timestamps: tuple[datetime, ...]
    max_price_gap_seconds: float
    max_quote_gap_seconds: float
    max_spread_bps: float
    median_spread_bps: float
    p95_spread_bps: float
    p99_spread_bps: float
    spread_limit_breach_fraction: float
    max_allowed_spread_bps: float | None = None


@dataclass(frozen=True)
class MarketDataHealthReport:
    symbols: tuple[SymbolDataHealth, ...]
    issues: tuple[DataHealthIssue, ...]

    @property
    def overall(self) -> DataHealthSeverity:
        if any(issue.severity == DataHealthSeverity.FAIL for issue in self.issues):
            return DataHealthSeverity.FAIL
        if any(issue.severity == DataHealthSeverity.WARN for issue in self.issues):
            return DataHealthSeverity.WARN
        return DataHealthSeverity.OK

    @property
    def ok(self) -> bool:
        return self.overall != DataHealthSeverity.FAIL

    def summary_lines(self) -> list[str]:
        lines = ["Market Data Validation", f"  Overall: {self.overall.value}"]
        for symbol in self.symbols:
            lines.append(
                (
                    f"  {symbol.symbol}: prices={symbol.price_count}, "
                    f"quotes={symbol.quote_count}, "
                    f"price_range={_range_label(symbol.price_start, symbol.price_end)}, "
                    f"quote_range={_range_label(symbol.quote_start, symbol.quote_end)}, "
                    f"median_spread={symbol.median_spread_bps:.2f} bps, "
                    f"p95_spread={symbol.p95_spread_bps:.2f} bps, "
                    f"max_spread={symbol.max_spread_bps:.2f} bps, "
                    f"spread_breach={symbol.spread_limit_breach_fraction:.1%}"
                    f"{_limit_label(symbol.max_allowed_spread_bps)}"
                )
            )

        if not self.issues:
            lines.append("  Issues: none")
            return lines

        lines.append("  Issues:")
        for issue in self.issues:
            lines.append(
                (
                    f"    {issue.severity.value} | {issue.symbol} | "
                    f"{issue.category}: {issue.details}"
                )
            )
        return lines


def validate_market_data(
    *,
    prices: PriceHistory,
    quotes: QuoteHistory,
    symbols: tuple[str, ...] | None = None,
    max_gap_seconds: float | None = 300.0,
    max_spread_bps: float | None = None,
    max_spread_bps_by_symbol: Mapping[str, float] | None = None,
) -> MarketDataHealthReport:
    selected_symbols = tuple(symbols or sorted(set(prices.symbols()) | set(quotes.symbols())))
    if not selected_symbols:
        return MarketDataHealthReport(
            symbols=(),
            issues=(
                DataHealthIssue(
                    severity=DataHealthSeverity.FAIL,
                    symbol="ALL",
                    category="coverage",
                    details="no symbols found in price or quote data",
                ),
            ),
        )

    symbol_reports: list[SymbolDataHealth] = []
    issues: list[DataHealthIssue] = []
    for symbol in selected_symbols:
        symbol_report, symbol_issues = _validate_symbol(
            prices=prices,
            quotes=quotes,
            symbol=symbol,
            max_gap_seconds=max_gap_seconds,
            max_spread_bps=_spread_limit_for_symbol(
                symbol=symbol,
                default_max_spread_bps=max_spread_bps,
                max_spread_bps_by_symbol=max_spread_bps_by_symbol,
            ),
        )
        symbol_reports.append(symbol_report)
        issues.extend(symbol_issues)

    return MarketDataHealthReport(
        symbols=tuple(symbol_reports),
        issues=tuple(issues),
    )


def _validate_symbol(
    *,
    prices: PriceHistory,
    quotes: QuoteHistory,
    symbol: str,
    max_gap_seconds: float | None,
    max_spread_bps: float | None,
) -> tuple[SymbolDataHealth, list[DataHealthIssue]]:
    bars = prices.for_symbol(symbol).bars
    snapshots = quotes.for_symbol(symbol).quotes
    price_times = tuple(bar.timestamp for bar in bars)
    quote_times = tuple(quote.timestamp for quote in snapshots)
    unique_price_times = tuple(sorted(set(price_times)))
    unique_quote_times = tuple(sorted(set(quote_times)))
    missing_quote_times = tuple(sorted(set(price_times) - set(quote_times)))
    missing_price_times = tuple(sorted(set(quote_times) - set(price_times)))
    duplicate_price_count = _duplicate_count(price_times)
    duplicate_quote_count = _duplicate_count(quote_times)
    spreads = tuple(sorted(quote.spread_bps for quote in snapshots))
    max_spread = max(spreads, default=0.0)
    spread_limit_breach_fraction = (
        0.0
        if max_spread_bps is None or not spreads
        else len([spread for spread in spreads if spread > max_spread_bps]) / len(spreads)
    )

    report = SymbolDataHealth(
        symbol=symbol,
        price_count=len(bars),
        quote_count=len(snapshots),
        price_start=unique_price_times[0] if unique_price_times else None,
        price_end=unique_price_times[-1] if unique_price_times else None,
        quote_start=unique_quote_times[0] if unique_quote_times else None,
        quote_end=unique_quote_times[-1] if unique_quote_times else None,
        duplicate_price_timestamps=duplicate_price_count,
        duplicate_quote_timestamps=duplicate_quote_count,
        missing_quote_timestamps=missing_quote_times,
        missing_price_timestamps=missing_price_times,
        max_price_gap_seconds=_max_gap_seconds(unique_price_times),
        max_quote_gap_seconds=_max_gap_seconds(unique_quote_times),
        max_spread_bps=max_spread,
        median_spread_bps=_percentile(spreads, 0.50),
        p95_spread_bps=_percentile(spreads, 0.95),
        p99_spread_bps=_percentile(spreads, 0.99),
        spread_limit_breach_fraction=spread_limit_breach_fraction,
        max_allowed_spread_bps=max_spread_bps,
    )

    issues: list[DataHealthIssue] = []
    if not bars:
        issues.append(_issue(DataHealthSeverity.FAIL, symbol, "coverage", "no price bars"))
    if not snapshots:
        issues.append(_issue(DataHealthSeverity.FAIL, symbol, "coverage", "no quotes"))
    if duplicate_price_count:
        issues.append(
            _issue(
                DataHealthSeverity.FAIL,
                symbol,
                "duplicates",
                f"{duplicate_price_count} duplicate price timestamp(s)",
            )
        )
    if duplicate_quote_count:
        issues.append(
            _issue(
                DataHealthSeverity.FAIL,
                symbol,
                "duplicates",
                f"{duplicate_quote_count} duplicate quote timestamp(s)",
            )
        )
    if missing_quote_times:
        issues.append(
            _issue(
                DataHealthSeverity.FAIL,
                symbol,
                "alignment",
                (
                    f"{len(missing_quote_times)} price bar timestamp(s) missing quotes: "
                    f"{_sample_times(missing_quote_times)}"
                ),
            )
        )
    if missing_price_times:
        issues.append(
            _issue(
                DataHealthSeverity.WARN,
                symbol,
                "alignment",
                (
                    f"{len(missing_price_times)} quote timestamp(s) without price bars: "
                    f"{_sample_times(missing_price_times)}"
                ),
            )
        )
    if max_gap_seconds is not None:
        if report.max_price_gap_seconds > max_gap_seconds:
            issues.append(
                _issue(
                    DataHealthSeverity.WARN,
                    symbol,
                    "gaps",
                    (
                        f"max price gap {report.max_price_gap_seconds:.1f}s "
                        f"> {max_gap_seconds:.1f}s limit"
                    ),
                )
            )
        if report.max_quote_gap_seconds > max_gap_seconds:
            issues.append(
                _issue(
                    DataHealthSeverity.WARN,
                    symbol,
                    "gaps",
                    (
                        f"max quote gap {report.max_quote_gap_seconds:.1f}s "
                        f"> {max_gap_seconds:.1f}s limit"
                    ),
                )
            )
    if max_spread_bps is not None and max_spread > max_spread_bps:
        issues.append(
            _issue(
                DataHealthSeverity.WARN,
                symbol,
                "spread",
                f"max spread {max_spread:.2f} bps > {max_spread_bps:.2f} bps limit",
            )
        )

    return report, issues


def write_market_data_health_csv(
    report: MarketDataHealthReport,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    issues_by_symbol: dict[str, list[DataHealthIssue]] = {}
    for issue in report.issues:
        issues_by_symbol.setdefault(issue.symbol, []).append(issue)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "symbol",
                "status",
                "issue_count",
                "issue_details",
                "price_count",
                "quote_count",
                "price_start",
                "price_end",
                "quote_start",
                "quote_end",
                "duplicate_price_timestamps",
                "duplicate_quote_timestamps",
                "missing_quotes",
                "missing_prices",
                "max_price_gap_seconds",
                "max_quote_gap_seconds",
                "max_spread_bps",
                "median_spread_bps",
                "p95_spread_bps",
                "p99_spread_bps",
                "spread_limit_breach_fraction",
                "max_allowed_spread_bps",
            ],
        )
        writer.writeheader()
        for symbol in report.symbols:
            symbol_issues = tuple(issues_by_symbol.get(symbol.symbol, ()))
            writer.writerow(
                {
                    "symbol": symbol.symbol,
                    "status": _symbol_status(symbol_issues).value,
                    "issue_count": len(symbol_issues),
                    "issue_details": " | ".join(
                        f"{issue.severity.value}:{issue.category}:{issue.details}"
                        for issue in symbol_issues
                    ),
                    "price_count": symbol.price_count,
                    "quote_count": symbol.quote_count,
                    "price_start": _dt(symbol.price_start),
                    "price_end": _dt(symbol.price_end),
                    "quote_start": _dt(symbol.quote_start),
                    "quote_end": _dt(symbol.quote_end),
                    "duplicate_price_timestamps": symbol.duplicate_price_timestamps,
                    "duplicate_quote_timestamps": symbol.duplicate_quote_timestamps,
                    "missing_quotes": len(symbol.missing_quote_timestamps),
                    "missing_prices": len(symbol.missing_price_timestamps),
                    "max_price_gap_seconds": f"{symbol.max_price_gap_seconds:.6f}",
                    "max_quote_gap_seconds": f"{symbol.max_quote_gap_seconds:.6f}",
                    "max_spread_bps": f"{symbol.max_spread_bps:.6f}",
                    "median_spread_bps": f"{symbol.median_spread_bps:.6f}",
                    "p95_spread_bps": f"{symbol.p95_spread_bps:.6f}",
                    "p99_spread_bps": f"{symbol.p99_spread_bps:.6f}",
                    "spread_limit_breach_fraction": (
                        f"{symbol.spread_limit_breach_fraction:.6f}"
                    ),
                    "max_allowed_spread_bps": (
                        ""
                        if symbol.max_allowed_spread_bps is None
                        else f"{symbol.max_allowed_spread_bps:.6f}"
                    ),
                }
            )


def _symbol_status(issues: tuple[DataHealthIssue, ...]) -> DataHealthSeverity:
    if any(issue.severity == DataHealthSeverity.FAIL for issue in issues):
        return DataHealthSeverity.FAIL
    if any(issue.severity == DataHealthSeverity.WARN for issue in issues):
        return DataHealthSeverity.WARN
    return DataHealthSeverity.OK


def _spread_limit_for_symbol(
    *,
    symbol: str,
    default_max_spread_bps: float | None,
    max_spread_bps_by_symbol: Mapping[str, float] | None,
) -> float | None:
    if not max_spread_bps_by_symbol:
        return default_max_spread_bps
    return max_spread_bps_by_symbol.get(symbol.upper(), default_max_spread_bps)


def _issue(
    severity: DataHealthSeverity,
    symbol: str,
    category: str,
    details: str,
) -> DataHealthIssue:
    return DataHealthIssue(
        severity=severity,
        symbol=symbol,
        category=category,
        details=details,
    )


def _duplicate_count(values: tuple[datetime, ...]) -> int:
    counts = Counter(values)
    return sum(count - 1 for count in counts.values() if count > 1)


def _percentile(values: tuple[float, ...], percentile: float) -> float:
    if not values:
        return 0.0
    if percentile <= 0:
        return values[0]
    if percentile >= 1:
        return values[-1]
    index = int(round((len(values) - 1) * percentile))
    return values[index]


def _max_gap_seconds(values: tuple[datetime, ...]) -> float:
    if len(values) < 2:
        return 0.0
    return max(
        (current - previous).total_seconds()
        for previous, current in zip(values, values[1:])
    )


def _range_label(start: datetime | None, end: datetime | None) -> str:
    if start is None or end is None:
        return "none"
    return f"{start.isoformat(timespec='seconds')} -> {end.isoformat(timespec='seconds')}"


def _limit_label(max_allowed_spread_bps: float | None) -> str:
    if max_allowed_spread_bps is None:
        return ""
    return f" limit={max_allowed_spread_bps:.2f} bps"


def _sample_times(values: tuple[datetime, ...], limit: int = 3) -> str:
    sample = ", ".join(value.isoformat(timespec="seconds") for value in values[:limit])
    if len(values) > limit:
        return f"{sample}, ..."
    return sample


def _dt(value: datetime | None) -> str:
    if value is None:
        return ""
    return value.isoformat(timespec="seconds")
