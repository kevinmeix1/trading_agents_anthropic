from __future__ import annotations

import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path

from quanthack.backtesting.backtest import FillModel
from quanthack.backtesting.competition_score import (
    CompetitionMetrics,
    RiskDisciplineReport,
    build_competition_metrics,
    build_risk_discipline_report,
    risk_samples_from_portfolio_equity,
)
from quanthack.backtesting.portfolio_backtest import (
    PortfolioBacktestEngine,
    PortfolioBacktestResult,
)
from quanthack.backtesting.portfolio_session import SessionGatePolicy
from quanthack.core.config import AppConfig
from quanthack.core.instruments import instrument_for
from quanthack.market.market_data import PriceHistory, QuoteHistory


@dataclass(frozen=True)
class LoadedDeploymentProfile:
    slot: str
    label: str
    evidence_status: str
    use_case: str
    reason: str
    strategy_by_symbol: tuple[tuple[str, str], ...]
    multipliers_by_symbol: tuple[tuple[str, float], ...]
    allowed_utc_hours: tuple[int, ...] | None
    forex_allowed_utc_hours: tuple[int, ...] | None
    metal_allowed_utc_hours: tuple[int, ...] | None
    crypto_allowed_utc_hours: tuple[int, ...] | None
    symbol_allowed_utc_hours: tuple[tuple[str, tuple[int, ...]], ...]
    profile_return_pct: float
    profile_drawdown_pct: float
    profile_sharpe_15m: float
    profile_fold_contribution: float

    @property
    def strategy_map_text(self) -> str:
        return " ".join(
            f"{symbol}={strategy}" for symbol, strategy in self.strategy_by_symbol
        )

    @property
    def multiplier_map_text(self) -> str:
        return " ".join(
            f"{symbol}={multiplier:.3f}"
            for symbol, multiplier in self.multipliers_by_symbol
        )

    @property
    def crypto_hours_text(self) -> str:
        return _hours_text(self.crypto_allowed_utc_hours)

    @property
    def allowed_hours_text(self) -> str:
        return _hours_text(self.allowed_utc_hours)

    @property
    def forex_hours_text(self) -> str:
        return _hours_text(self.forex_allowed_utc_hours)

    @property
    def metal_hours_text(self) -> str:
        return _hours_text(self.metal_allowed_utc_hours)

    @property
    def symbol_hours_text(self) -> str:
        if not self.symbol_allowed_utc_hours:
            return ""
        return " ".join(
            f"{symbol}={_hours_text(hours)}"
            for symbol, hours in self.symbol_allowed_utc_hours
        )


@dataclass(frozen=True)
class DeploymentProfileBacktestResult:
    profile: LoadedDeploymentProfile
    result: PortfolioBacktestResult
    competition_metrics: CompetitionMetrics
    risk_discipline: RiskDisciplineReport


def run_deployment_profile_backtest(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    profile_pack_json: str | Path,
    slot: str = "conservative",
) -> DeploymentProfileBacktestResult:
    profile = load_deployment_profile(
        profile_pack_json=profile_pack_json,
        slot=slot,
    )
    symbols = tuple(symbol for symbol, _ in profile.strategy_by_symbol)
    _validate_profile_data_symbols(prices=prices, quotes=quotes, symbols=symbols)
    engine = PortfolioBacktestEngine(
        strategies={
            symbol: config.build_strategy(strategy, symbol=symbol)
            for symbol, strategy in profile.strategy_by_symbol
        },
        risk_limits=config.risk,
        quality_limits=config.market_quality,
        quality_limits_by_symbol={
            symbol: replace(
                config.market_quality,
                max_spread_bps=instrument_for(symbol).max_spread_bps,
            )
            for symbol in symbols
        },
        clock=config.competition.to_clock(),
        fill_model=FillModel(slippage_bps=config.backtest.slippage_bps),
        periods_per_year=config.backtest.periods_per_year,
        target_notional_multipliers_by_symbol=dict(profile.multipliers_by_symbol),
        session_gate_policy=session_gate_policy_for_profile(profile),
    )
    result = engine.run(
        prices=prices,
        quotes=quotes,
        starting_equity=config.competition.starting_equity,
    )
    competition_metrics = build_competition_metrics(
        equity_points=result.equity_curve,
        fills=result.fills,
    )
    risk_discipline = build_risk_discipline_report(
        risk_samples_from_portfolio_equity(result.equity_curve)
    )
    return DeploymentProfileBacktestResult(
        profile=profile,
        result=result,
        competition_metrics=competition_metrics,
        risk_discipline=risk_discipline,
    )


def load_deployment_profile(
    *,
    profile_pack_json: str | Path,
    slot: str,
) -> LoadedDeploymentProfile:
    pack = json.loads(Path(profile_pack_json).read_text(encoding="utf-8"))
    selected_slot = _selected_slot(pack, slot)
    profiles = pack.get("profiles", ())
    for raw_profile in profiles:
        if raw_profile.get("slot") == selected_slot:
            return _profile_from_json(raw_profile)
    available = ", ".join(profile.get("slot", "") for profile in profiles)
    raise ValueError(f"profile slot {selected_slot!r} not found; available: {available}")


def write_deployment_profile_backtest_summary_csv(
    backtest: DeploymentProfileBacktestResult,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile = backtest.profile
    metrics = backtest.competition_metrics
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "slot",
                "label",
                "evidence_status",
                "use_case",
                "profile_return_pct",
                "profile_drawdown_pct",
                "profile_sharpe_15m",
                "profile_fold_contribution",
                "backtest_return_pct",
                "backtest_max_drawdown_pct",
                "backtest_sharpe_15m",
                "risk_discipline_score",
                "trade_count",
                "fills",
                "total_pnl_usd",
                "strategy_map",
                "multiplier_map",
                "allowed_utc_hours",
                "forex_allowed_utc_hours",
                "metal_allowed_utc_hours",
                "crypto_allowed_utc_hours",
                "symbol_allowed_utc_hours",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "slot": profile.slot,
                "label": profile.label,
                "evidence_status": profile.evidence_status,
                "use_case": profile.use_case,
                "profile_return_pct": profile.profile_return_pct,
                "profile_drawdown_pct": profile.profile_drawdown_pct,
                "profile_sharpe_15m": profile.profile_sharpe_15m,
                "profile_fold_contribution": profile.profile_fold_contribution,
                "backtest_return_pct": metrics.return_pct,
                "backtest_max_drawdown_pct": metrics.max_drawdown_pct,
                "backtest_sharpe_15m": metrics.sharpe_15m,
                "risk_discipline_score": backtest.risk_discipline.score,
                "trade_count": metrics.trade_count,
                "fills": len(backtest.result.fills),
                "total_pnl_usd": backtest.result.total_pnl_usd,
                "strategy_map": profile.strategy_map_text,
                "multiplier_map": profile.multiplier_map_text,
                "allowed_utc_hours": profile.allowed_hours_text,
                "forex_allowed_utc_hours": profile.forex_hours_text,
                "metal_allowed_utc_hours": profile.metal_hours_text,
                "crypto_allowed_utc_hours": profile.crypto_hours_text,
                "symbol_allowed_utc_hours": profile.symbol_hours_text,
            }
        )


def _profile_from_json(raw_profile: dict) -> LoadedDeploymentProfile:
    strategy_by_symbol = _parse_strategy_map(raw_profile.get("strategy_map", ""))
    multipliers_by_symbol = _parse_multiplier_map(
        raw_profile.get("multiplier_map", ""),
        symbols=tuple(symbol for symbol, _ in strategy_by_symbol),
    )
    return LoadedDeploymentProfile(
        slot=str(raw_profile.get("slot", "")),
        label=str(raw_profile.get("label", "")),
        evidence_status=str(raw_profile.get("evidence_status", "")),
        use_case=str(raw_profile.get("use_case", "")),
        reason=str(raw_profile.get("reason", "")),
        strategy_by_symbol=strategy_by_symbol,
        multipliers_by_symbol=multipliers_by_symbol,
        allowed_utc_hours=_parse_hours(
            raw_profile.get("allowed_utc_hours", ""),
            field_name="allowed_utc_hours",
        ),
        forex_allowed_utc_hours=_parse_hours(
            raw_profile.get("forex_allowed_utc_hours", ""),
            field_name="forex_allowed_utc_hours",
        ),
        metal_allowed_utc_hours=_parse_hours(
            raw_profile.get("metal_allowed_utc_hours", ""),
            field_name="metal_allowed_utc_hours",
        ),
        crypto_allowed_utc_hours=_parse_hours(
            raw_profile.get("crypto_allowed_utc_hours", ""),
            field_name="crypto_allowed_utc_hours",
        ),
        symbol_allowed_utc_hours=_parse_symbol_hours(
            raw_profile.get("symbol_allowed_utc_hours", "")
        ),
        profile_return_pct=float(raw_profile.get("return_pct", 0.0)),
        profile_drawdown_pct=float(raw_profile.get("max_drawdown_pct", 0.0)),
        profile_sharpe_15m=float(raw_profile.get("sharpe_15m", 0.0)),
        profile_fold_contribution=float(raw_profile.get("fold_contribution", 0.0)),
    )


def _selected_slot(pack: dict, slot: str) -> str:
    if slot != "recommended":
        return slot
    recommended = str(pack.get("recommended_slot", ""))
    profile_slots = {
        str(profile.get("slot", "")).strip()
        for profile in pack.get("profiles", ())
        if str(profile.get("slot", "")).strip()
    }
    if recommended in profile_slots:
        return recommended
    available = ", ".join(sorted(profile_slots)) or "none"
    raise ValueError(
        "profile pack recommendation is not executable; "
        f"choose one of {available} explicitly"
    )


def _parse_strategy_map(text: str) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for raw_part in text.split():
        symbol, separator, strategy = raw_part.partition("=")
        if not separator:
            raise ValueError(f"invalid strategy map item {raw_part!r}")
        pairs.append((instrument_for(symbol).symbol, strategy.strip()))
    if not pairs:
        raise ValueError("profile strategy map cannot be empty")
    return tuple(pairs)


def _parse_multiplier_map(
    text: str,
    *,
    symbols: tuple[str, ...],
) -> tuple[tuple[str, float], ...]:
    parsed: dict[str, float] = {}
    for raw_part in text.split():
        symbol, separator, raw_multiplier = raw_part.partition("=")
        if not separator:
            raise ValueError(f"invalid multiplier map item {raw_part!r}")
        parsed[instrument_for(symbol).symbol] = float(raw_multiplier)
    for symbol in symbols:
        parsed.setdefault(symbol, 1.0)
    return tuple(sorted(parsed.items()))


def _parse_hours(text: str, *, field_name: str) -> tuple[int, ...] | None:
    if not text or text == "all":
        return None
    hours = tuple(int(part) for part in text.split("|") if part)
    if any(hour < 0 or hour > 23 for hour in hours):
        raise ValueError(f"{field_name} must be between 0 and 23")
    return tuple(sorted(set(hours)))


def _parse_symbol_hours(text) -> tuple[tuple[str, tuple[int, ...]], ...]:
    if not text:
        return ()
    if isinstance(text, dict):
        items = text.items()
    else:
        items = (
            raw_part.partition("=")
            for raw_part in str(text).split()
            if raw_part.strip()
        )
    parsed: dict[str, tuple[int, ...]] = {}
    for item in items:
        if isinstance(item, tuple) and len(item) == 3:
            symbol, separator, raw_hours = item
            if not separator:
                raise ValueError(f"invalid symbol hour map item {symbol!r}")
        else:
            symbol, raw_hours = item
        normalized_symbol = instrument_for(str(symbol)).symbol
        hours = _parse_hours(str(raw_hours), field_name="symbol_allowed_utc_hours")
        if hours is None:
            raise ValueError("symbol_allowed_utc_hours cannot use 'all' or empty hours")
        parsed[normalized_symbol] = hours
    return tuple(sorted(parsed.items()))


def session_gate_policy_for_profile(
    profile: LoadedDeploymentProfile,
) -> SessionGatePolicy | None:
    if (
        profile.allowed_utc_hours is None
        and profile.forex_allowed_utc_hours is None
        and profile.metal_allowed_utc_hours is None
        and profile.crypto_allowed_utc_hours is None
        and not profile.symbol_allowed_utc_hours
    ):
        return None
    return SessionGatePolicy(
        allowed_utc_hours=profile.allowed_utc_hours,
        forex_allowed_utc_hours=profile.forex_allowed_utc_hours,
        metal_allowed_utc_hours=profile.metal_allowed_utc_hours,
        crypto_allowed_utc_hours=profile.crypto_allowed_utc_hours,
        symbol_allowed_utc_hours=profile.symbol_allowed_utc_hours,
    )


def _hours_text(hours: tuple[int, ...] | None) -> str:
    if hours is None:
        return "all"
    return "|".join(str(hour) for hour in hours)


def _validate_profile_data_symbols(
    *,
    prices: PriceHistory,
    quotes: QuoteHistory,
    symbols: tuple[str, ...],
) -> None:
    missing_prices = sorted(set(symbols) - set(prices.symbols()))
    missing_quotes = sorted(set(symbols) - set(quotes.symbols()))
    if missing_prices:
        raise ValueError(f"profile symbols missing prices: {', '.join(missing_prices)}")
    if missing_quotes:
        raise ValueError(f"profile symbols missing quotes: {', '.join(missing_quotes)}")
