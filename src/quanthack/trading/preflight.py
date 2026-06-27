from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from zoneinfo import ZoneInfo

from quanthack.core.config import AppConfig, load_config
from quanthack.market.market_data import load_price_history, load_quote_history
from quanthack.market.market_quality import MarketQualityChecker, MarketQualityDecision


MARKET_QUALITY_WARN_FRACTION = 0.5


class PreflightStatus(StrEnum):
    OK = "OK"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    status: PreflightStatus
    details: str


@dataclass(frozen=True)
class PreflightReport:
    checks: tuple[PreflightCheck, ...]

    @property
    def overall(self) -> str:
        if any(check.status == PreflightStatus.FAIL for check in self.checks):
            return "ATTENTION_REQUIRED"
        if any(check.status == PreflightStatus.WARN for check in self.checks):
            return "READY_WITH_WARNINGS"
        return "READY_FOR_DRY_RUN"

    def summary_lines(self) -> list[str]:
        lines = ["Preflight"]
        for check in self.checks:
            lines.append(f"  {check.name}: {check.status.value} - {check.details}")
        lines.append(f"  Overall: {self.overall}")
        return lines


def run_preflight(
    *,
    config_path: str | Path = "configs/default.toml",
    now: datetime | None = None,
    quote_as_of: datetime | None = None,
) -> PreflightReport:
    checks: list[PreflightCheck] = []

    checks.append(_check_python())

    config = _load_config_check(config_path, checks)
    if config is None:
        return PreflightReport(tuple(checks))

    checks.append(_check_clock(config, now))
    checks.append(_check_prices(config))
    quote_check, quote = _check_quotes(config)
    checks.append(quote_check)
    if quote is not None:
        checks.append(_check_market_quality(config, quote=quote, quote_as_of=quote_as_of))
    checks.append(_check_risk_limits(config))
    checks.append(_check_journal_path(config))

    return PreflightReport(tuple(checks))


def _check_python() -> PreflightCheck:
    version = ".".join(str(part) for part in sys.version_info[:3])
    if sys.version_info >= (3, 11):
        return _ok("Python", f"{version}")
    return _fail("Python", f"{version}; project requires Python 3.11+")


def _load_config_check(path: str | Path, checks: list[PreflightCheck]) -> AppConfig | None:
    try:
        config = load_config(path)
    except Exception as exc:
        checks.append(_fail("Config", f"could not load {path}: {exc}"))
        return None

    checks.append(_ok("Config", f"loaded {path}"))
    return config


def _check_clock(config: AppConfig, now: datetime | None) -> PreflightCheck:
    try:
        timezone = ZoneInfo(config.competition.timezone)
        as_of = now or datetime.now(tz=timezone)
        clock = config.competition.to_clock()
        mode = clock.mode_at(as_of)
        minutes = clock.minutes_to_next_checkpoint(as_of)
    except Exception as exc:
        return _fail("Clock", str(exc))

    if minutes is None:
        return _ok("Clock", f"mode={mode.value}; no future checkpoint configured")
    return _ok("Clock", f"mode={mode.value}; next checkpoint in {minutes:.1f} minutes")


def _check_prices(config: AppConfig) -> PreflightCheck:
    symbol = config.strategy_symbol()
    lookback = config.strategy_lookback()
    try:
        history = load_price_history(config.market_data.price_csv)
        closes = history.close_prices(symbol=symbol)
    except Exception as exc:
        return _fail("Prices", str(exc))

    if len(closes) < lookback:
        return _fail(
            "Prices",
            f"{len(closes)} closes for {symbol}; need {lookback}",
        )
    return _ok("Prices", f"{len(closes)} closes for {symbol}")


def _check_quotes(config: AppConfig) -> tuple[PreflightCheck, object | None]:
    symbol = config.strategy_symbol()
    try:
        quotes = load_quote_history(config.market_data.quote_csv)
        quote = quotes.latest_quote(symbol)
    except Exception as exc:
        return _fail("Quotes", str(exc)), None

    if quote is None:
        return _fail("Quotes", f"no quote for {symbol}"), None
    return _ok("Quotes", f"latest quote for {symbol}"), quote


def _check_market_quality(
    config: AppConfig,
    *,
    quote,
    quote_as_of: datetime | None,
) -> PreflightCheck:
    try:
        as_of = quote_as_of or quote.timestamp
        decision = MarketQualityChecker(config.market_quality).evaluate(quote=quote, as_of=as_of)
    except Exception as exc:
        return _fail("Market quality", str(exc))

    details = (
        f"{decision.reason}; spread={decision.spread_bps:.2f} bps; "
        f"age={decision.quote_age_seconds:.1f}s"
    )
    if not decision.ok:
        return _fail("Market quality", details)
    warnings = _market_quality_warnings(config, decision)
    if warnings:
        return _warn("Market quality", f"{details}; warnings: {'; '.join(warnings)}")
    return _ok("Market quality", details)


def _check_risk_limits(config: AppConfig) -> PreflightCheck:
    limits = config.risk
    problems: list[str] = []
    warnings: list[str] = []

    if limits.max_gross_leverage > 3.0:
        problems.append("max_gross_leverage above 3.0x starter guard")
    elif limits.max_gross_leverage > 2.0:
        warnings.append("max_gross_leverage above 2.0x conservative target")

    if limits.max_symbol_notional_pct > 0.35:
        problems.append("max_symbol_notional_pct above 35% starter guard")
    elif limits.max_symbol_notional_pct > 0.25:
        warnings.append("max_symbol_notional_pct above 25% conservative target")

    if limits.max_daily_loss_pct > 0.03:
        problems.append("max_daily_loss_pct above 3% starter guard")
    elif limits.max_daily_loss_pct > 0.025:
        warnings.append("max_daily_loss_pct above 2.5% conservative target")

    if limits.max_drawdown_pct > 0.08:
        problems.append("max_drawdown_pct above 8% starter guard")
    elif limits.max_drawdown_pct > 0.06:
        warnings.append("max_drawdown_pct above 6% conservative target")

    if limits.min_margin_level_pct < 300:
        problems.append("min_margin_level_pct below 300% starter guard")
    elif 300 < limits.min_margin_level_pct < 400:
        warnings.append("min_margin_level_pct close to 300% starter guard")

    if problems:
        return _fail("Risk limits", "; ".join(problems))

    details = _risk_limit_summary(config)
    if warnings:
        return _warn("Risk limits", f"{details}; warnings: {'; '.join(warnings)}")

    return _ok("Risk limits", details)


def _market_quality_warnings(
    config: AppConfig,
    decision: MarketQualityDecision,
) -> list[str]:
    limits = config.market_quality
    warnings: list[str] = []
    if decision.spread_bps >= limits.max_spread_bps * MARKET_QUALITY_WARN_FRACTION:
        warnings.append(
            (
                f"spread is above {MARKET_QUALITY_WARN_FRACTION:.0%} of limit "
                f"({decision.spread_bps:.2f}/{limits.max_spread_bps:.2f} bps)"
            )
        )
    if (
        decision.quote_age_seconds
        >= limits.max_quote_age_seconds * MARKET_QUALITY_WARN_FRACTION
    ):
        warnings.append(
            (
                f"quote age is above {MARKET_QUALITY_WARN_FRACTION:.0%} of limit "
                f"({decision.quote_age_seconds:.1f}/{limits.max_quote_age_seconds:.1f}s)"
            )
        )
    return warnings


def _risk_limit_summary(config: AppConfig) -> str:
    limits = config.risk
    return (
        f"gross={limits.max_gross_leverage:.2f}x, "
        f"daily_loss={limits.max_daily_loss_pct:.1%}, "
        f"margin_floor={limits.min_margin_level_pct:.0f}%"
    )


def _check_journal_path(config: AppConfig) -> PreflightCheck:
    journal = Path(config.execution.journal_path)
    probe: Path | None = None
    try:
        journal.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            dir=journal.parent,
            prefix=".preflight_write_test_",
            delete=False,
            encoding="utf-8",
        ) as handle:
            handle.write("ok\n")
            probe = Path(handle.name)
    except Exception as exc:
        return _fail("Journal", f"{journal}: {exc}")
    finally:
        if probe is not None:
            probe.unlink(missing_ok=True)

    return _ok("Journal", f"writable: {journal}")


def _ok(name: str, details: str) -> PreflightCheck:
    return PreflightCheck(name=name, status=PreflightStatus.OK, details=details)


def _warn(name: str, details: str) -> PreflightCheck:
    return PreflightCheck(name=name, status=PreflightStatus.WARN, details=details)


def _fail(name: str, details: str) -> PreflightCheck:
    return PreflightCheck(name=name, status=PreflightStatus.FAIL, details=details)
