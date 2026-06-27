from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from quanthack.backtesting.deployment_profile_backtest import (
    LoadedDeploymentProfile,
    load_deployment_profile,
    session_gate_policy_for_profile,
)
from quanthack.backtesting.portfolio_allocator import (
    AllocationPolicy,
    AllocatedTarget,
    PortfolioAllocation,
    PortfolioAllocator,
    SymbolIntent,
)
from quanthack.backtesting.portfolio_session import PortfolioSessionGate
from quanthack.core.clock import CompetitionMode
from quanthack.core.config import AppConfig
from quanthack.core.instruments import instrument_for
from quanthack.market.market_data import PriceBar, PriceHistory, QuoteHistory, QuoteSnapshot
from quanthack.market.market_quality import MarketQualityChecker
from quanthack.strategies.strategy import EPSILON_NOTIONAL, Strategy
from quanthack.trading.risk import (
    AccountSnapshot,
    PortfolioSnapshot,
    Position,
    RiskDecision,
    RiskEngine,
    Side,
    TradeRequest,
)


@dataclass(frozen=True)
class DeploymentProfileSignalRow:
    symbol: str
    strategy_name: str
    timestamp: str
    bid: float
    ask: float
    mid: float
    current_notional_usd: float
    raw_target_notional_usd: float
    scaled_target_notional_usd: float
    allocated_target_notional_usd: float
    change_notional_usd: float
    order_side: str
    risk_approved: bool
    risk_adjusted_notional_usd: float
    risk_reason: str
    primary_signal: str
    strategy_reason: str
    allocation_reasons: tuple[str, ...]
    supporting_signals: tuple[str, ...] = ()
    conflicting_signals: tuple[str, ...] = ()


@dataclass(frozen=True)
class DeploymentProfileSignalSnapshot:
    profile: LoadedDeploymentProfile
    timestamp: str
    account: AccountSnapshot
    allocation: PortfolioAllocation
    rows: tuple[DeploymentProfileSignalRow, ...]

    @property
    def actionable_rows(self) -> tuple[DeploymentProfileSignalRow, ...]:
        return tuple(row for row in self.rows if row.order_side != "HOLD")


def build_deployment_profile_signal_snapshot(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    profile_pack_json: str | Path,
    slot: str,
    account: AccountSnapshot,
    portfolio: PortfolioSnapshot | None = None,
    mode: CompetitionMode = CompetitionMode.QUALIFY,
    as_of: datetime | None = None,
) -> DeploymentProfileSignalSnapshot:
    profile = load_deployment_profile(
        profile_pack_json=profile_pack_json,
        slot=slot,
    )
    symbols = tuple(symbol for symbol, _ in profile.strategy_by_symbol)
    timestamp = _latest_common_timestamp(
        prices=prices,
        quotes=quotes,
        symbols=symbols,
        as_of=as_of,
    )
    latest_quotes = {
        symbol: _quote_at(quotes=quotes, symbol=symbol, timestamp=timestamp)
        for symbol in symbols
    }
    histories = {
        symbol: _bars_until(prices=prices, symbol=symbol, timestamp=timestamp)
        for symbol in symbols
    }
    portfolio_before = portfolio or PortfolioSnapshot()
    strategies = {
        symbol: config.build_strategy(strategy_name, symbol=symbol)
        for symbol, strategy_name in profile.strategy_by_symbol
    }
    _update_strategy_context(strategies=strategies, histories=histories, quotes=latest_quotes)

    raw_intents = tuple(
        _build_raw_intent(
            config=config,
            profile=profile,
            symbol=symbol,
            strategy_name=_strategy_name(profile, symbol),
            strategy=strategies[symbol],
            quote=latest_quotes[symbol],
            bars=histories[symbol],
            portfolio=portfolio_before,
        )
        for symbol in symbols
    )
    gated_intents = _apply_session_gate(
        profile=profile,
        intents=raw_intents,
        timestamp=timestamp,
    )
    allocation = PortfolioAllocator(
        AllocationPolicy(
            max_gross_leverage=config.risk.max_gross_leverage,
            max_symbol_gross_pct=config.risk.max_symbol_notional_pct,
        )
    ).allocate(
        gated_intents,
        equity=account.equity,
        timestamp=timestamp.isoformat(timespec="seconds"),
    )
    rows = _build_rows(
        profile=profile,
        timestamp=timestamp,
        quotes=latest_quotes,
        raw_intents=raw_intents,
        gated_intents=gated_intents,
        allocation=allocation,
        account=account,
        portfolio=portfolio_before,
        mode=mode,
        risk_engine=RiskEngine(config.risk),
    )
    return DeploymentProfileSignalSnapshot(
        profile=profile,
        timestamp=timestamp.isoformat(timespec="seconds"),
        account=account,
        allocation=allocation,
        rows=rows,
    )


def write_deployment_profile_signal_snapshot_csv(
    snapshot: DeploymentProfileSignalSnapshot,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "profile_slot",
                "profile_label",
                "timestamp",
                "symbol",
                "strategy_name",
                "bid",
                "ask",
                "mid",
                "current_notional_usd",
                "raw_target_notional_usd",
                "scaled_target_notional_usd",
                "allocated_target_notional_usd",
                "change_notional_usd",
                "order_side",
                "risk_approved",
                "risk_adjusted_notional_usd",
                "risk_reason",
                "primary_signal",
                "strategy_reason",
                "allocation_reasons",
                "supporting_signals",
                "conflicting_signals",
            ],
        )
        writer.writeheader()
        for row in snapshot.rows:
            writer.writerow(
                {
                    "profile_slot": snapshot.profile.slot,
                    "profile_label": snapshot.profile.label,
                    "timestamp": row.timestamp,
                    "symbol": row.symbol,
                    "strategy_name": row.strategy_name,
                    "bid": row.bid,
                    "ask": row.ask,
                    "mid": row.mid,
                    "current_notional_usd": row.current_notional_usd,
                    "raw_target_notional_usd": row.raw_target_notional_usd,
                    "scaled_target_notional_usd": row.scaled_target_notional_usd,
                    "allocated_target_notional_usd": row.allocated_target_notional_usd,
                    "change_notional_usd": row.change_notional_usd,
                    "order_side": row.order_side,
                    "risk_approved": row.risk_approved,
                    "risk_adjusted_notional_usd": row.risk_adjusted_notional_usd,
                    "risk_reason": row.risk_reason,
                    "primary_signal": row.primary_signal,
                    "strategy_reason": row.strategy_reason,
                    "allocation_reasons": "|".join(row.allocation_reasons),
                    "supporting_signals": "|".join(row.supporting_signals),
                    "conflicting_signals": "|".join(row.conflicting_signals),
                }
            )


def _build_raw_intent(
    *,
    config: AppConfig,
    profile: LoadedDeploymentProfile,
    symbol: str,
    strategy_name: str,
    strategy: Strategy,
    quote: QuoteSnapshot,
    bars: tuple[PriceBar, ...],
    portfolio: PortfolioSnapshot,
) -> SymbolIntent:
    current_notional = portfolio.notional_for_symbol(symbol)
    quality = MarketQualityChecker(
        replace(
            config.market_quality,
            max_spread_bps=instrument_for(symbol).max_spread_bps,
        )
    ).evaluate(quote=quote, as_of=quote.timestamp)
    if not quality.ok:
        return SymbolIntent(
            symbol=symbol,
            target_notional_usd=current_notional,
            current_notional_usd=current_notional,
            reason=f"market quality hold: {quality.reason}",
            primary_signal="market_quality",
        )

    closes = [bar.close for bar in bars]
    multiplier = dict(profile.multipliers_by_symbol).get(symbol, 1.0)
    if hasattr(strategy, "generate_decision"):
        decision = strategy.generate_decision(
            closes,
            current_notional_usd=current_notional,
            holding_period=0,
            quote=quote,
        )
        target = (
            decision.target_notional_usd * multiplier
            if decision.is_trade_intent
            else current_notional
        )
        return SymbolIntent(
            symbol=symbol,
            target_notional_usd=target,
            current_notional_usd=current_notional,
            reason=decision.reason,
            primary_signal=decision.primary_signal,
            supporting_signals=decision.supporting_signals,
            conflicting_signals=decision.conflicting_signals,
        )

    request = strategy.generate_request(closes)
    if request is None:
        return SymbolIntent(
            symbol=symbol,
            target_notional_usd=current_notional,
            current_notional_usd=current_notional,
            reason="no strategy request",
        )
    direction = 1 if request.side == Side.BUY else -1
    return SymbolIntent(
        symbol=symbol,
        target_notional_usd=direction * request.target_notional_usd * multiplier,
        current_notional_usd=current_notional,
        reason=request.reason,
        primary_signal="request",
    )


def _apply_session_gate(
    *,
    profile: LoadedDeploymentProfile,
    intents: tuple[SymbolIntent, ...],
    timestamp: datetime,
) -> tuple[SymbolIntent, ...]:
    policy = session_gate_policy_for_profile(profile)
    if policy is None:
        return intents
    return PortfolioSessionGate(policy).apply(intents, timestamp=timestamp)


def _build_rows(
    *,
    profile: LoadedDeploymentProfile,
    timestamp: datetime,
    quotes: dict[str, QuoteSnapshot],
    raw_intents: tuple[SymbolIntent, ...],
    gated_intents: tuple[SymbolIntent, ...],
    allocation: PortfolioAllocation,
    account: AccountSnapshot,
    portfolio: PortfolioSnapshot,
    mode: CompetitionMode,
    risk_engine: RiskEngine,
) -> tuple[DeploymentProfileSignalRow, ...]:
    raw_by_symbol = {intent.symbol: intent for intent in raw_intents}
    gated_by_symbol = {intent.symbol: intent for intent in gated_intents}
    rows: list[DeploymentProfileSignalRow] = []
    preview_portfolio = portfolio
    for target in allocation.targets:
        symbol = target.symbol
        decision = _risk_preview(
            target=target,
            account=account,
            portfolio=preview_portfolio,
            mode=mode,
            risk_engine=risk_engine,
        )
        order_side = _order_side(target.change_notional_usd)
        if decision.approved and order_side != "HOLD":
            preview_portfolio = _portfolio_after_target(
                portfolio=preview_portfolio,
                symbol=symbol,
                signed_notional=_signed_target(
                    target.adjusted_notional_usd,
                    decision.adjusted_notional_usd,
                ),
            )
        quote = quotes[symbol]
        rows.append(
            DeploymentProfileSignalRow(
                symbol=symbol,
                strategy_name=_strategy_name(profile, symbol),
                timestamp=timestamp.isoformat(timespec="seconds"),
                bid=quote.bid,
                ask=quote.ask,
                mid=quote.mid,
                current_notional_usd=target.current_notional_usd,
                raw_target_notional_usd=raw_by_symbol[symbol].target_notional_usd,
                scaled_target_notional_usd=gated_by_symbol[symbol].target_notional_usd,
                allocated_target_notional_usd=target.adjusted_notional_usd,
                change_notional_usd=target.change_notional_usd,
                order_side=order_side,
                risk_approved=decision.approved,
                risk_adjusted_notional_usd=decision.adjusted_notional_usd,
                risk_reason=decision.reason,
                primary_signal=target.primary_signal,
                strategy_reason=target.intent_reason,
                allocation_reasons=target.reasons,
                supporting_signals=target.supporting_signals,
                conflicting_signals=target.conflicting_signals,
            )
        )
    return tuple(rows)


def _risk_preview(
    *,
    target: AllocatedTarget,
    account: AccountSnapshot,
    portfolio: PortfolioSnapshot,
    mode: CompetitionMode,
    risk_engine: RiskEngine,
) -> RiskDecision:
    if abs(target.change_notional_usd) <= EPSILON_NOTIONAL:
        return RiskDecision(
            approved=True,
            reason="no allocated change",
            adjusted_notional_usd=0.0,
            state=risk_engine.state,
        )
    if abs(target.adjusted_notional_usd) <= EPSILON_NOTIONAL:
        return risk_engine.evaluate_exit(
            account=account,
            portfolio=portfolio,
            symbol=target.symbol,
            mode=mode,
            reason="allocated exit",
        )
    side = Side.BUY if target.adjusted_notional_usd > 0 else Side.SELL
    return risk_engine.evaluate(
        account=account,
        portfolio=portfolio,
        request=TradeRequest(
            symbol=target.symbol,
            side=side,
            target_notional_usd=abs(target.adjusted_notional_usd),
            reason=target.intent_reason or "deployment profile signal snapshot",
        ),
        mode=mode,
    )


def _latest_common_timestamp(
    *,
    prices: PriceHistory,
    quotes: QuoteHistory,
    symbols: tuple[str, ...],
    as_of: datetime | None = None,
) -> datetime:
    if as_of is not None and as_of.tzinfo is None:
        raise ValueError("as_of timestamp must include a timezone")
    common: set[datetime] | None = None
    for symbol in symbols:
        price_times = {bar.timestamp for bar in prices.for_symbol(symbol).bars}
        quote_times = {quote.timestamp for quote in quotes.for_symbol(symbol).quotes}
        symbol_common = price_times & quote_times
        if not symbol_common:
            raise ValueError(f"no aligned price/quote timestamps for {symbol}")
        common = symbol_common if common is None else common & symbol_common
    if not common:
        raise ValueError("no common timestamp across profile symbols")
    if as_of is not None:
        common = {timestamp for timestamp in common if timestamp <= as_of}
        if not common:
            raise ValueError(f"no common timestamp at or before {as_of.isoformat()}")
    return max(common)


def _bars_until(
    *,
    prices: PriceHistory,
    symbol: str,
    timestamp: datetime,
) -> tuple[PriceBar, ...]:
    bars = tuple(
        bar for bar in prices.for_symbol(symbol).bars if bar.timestamp <= timestamp
    )
    if not bars:
        raise ValueError(f"no price bars for {symbol} before {timestamp.isoformat()}")
    return bars


def _quote_at(
    *,
    quotes: QuoteHistory,
    symbol: str,
    timestamp: datetime,
) -> QuoteSnapshot:
    for quote in quotes.for_symbol(symbol).quotes:
        if quote.timestamp == timestamp:
            return quote
    raise ValueError(f"no quote for {symbol} at {timestamp.isoformat()}")


def _update_strategy_context(
    *,
    strategies: dict[str, Strategy],
    histories: dict[str, tuple[PriceBar, ...]],
    quotes: dict[str, QuoteSnapshot],
) -> None:
    closes_by_symbol = {
        symbol: tuple(bar.close for bar in bars)
        for symbol, bars in histories.items()
    }
    for strategy in strategies.values():
        update_context = getattr(strategy, "update_portfolio_context", None)
        if callable(update_context):
            update_context(
                closes_by_symbol=closes_by_symbol,
                quotes_by_symbol=quotes,
            )


def _strategy_name(profile: LoadedDeploymentProfile, symbol: str) -> str:
    for profile_symbol, strategy_name in profile.strategy_by_symbol:
        if profile_symbol == symbol:
            return strategy_name
    raise KeyError(f"profile missing strategy for {symbol}")


def _order_side(change_notional_usd: float) -> str:
    if change_notional_usd > EPSILON_NOTIONAL:
        return Side.BUY.value
    if change_notional_usd < -EPSILON_NOTIONAL:
        return Side.SELL.value
    return "HOLD"


def _portfolio_after_target(
    *,
    portfolio: PortfolioSnapshot,
    symbol: str,
    signed_notional: float,
) -> PortfolioSnapshot:
    positions = {position.symbol: position.notional_usd for position in portfolio.positions}
    if abs(signed_notional) <= EPSILON_NOTIONAL:
        positions.pop(symbol, None)
    else:
        positions[symbol] = signed_notional
    return PortfolioSnapshot(
        positions=tuple(
            sorted(
                (
                    Position(symbol=position_symbol, notional_usd=notional)
                    for position_symbol, notional in positions.items()
                    if abs(notional) > EPSILON_NOTIONAL
                ),
                key=lambda position: position.symbol,
            )
        )
    )


def _signed_target(requested_signed: float, adjusted_abs: float) -> float:
    if abs(adjusted_abs) <= EPSILON_NOTIONAL:
        return 0.0
    return adjusted_abs if requested_signed > 0 else -adjusted_abs
