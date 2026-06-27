from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from math import isfinite
from pathlib import Path
from time import sleep

from quanthack.backtesting.portfolio_allocator import (
    AllocationPolicy,
    PortfolioAllocation,
    PortfolioAllocator,
    SymbolIntent,
)
from quanthack.backtesting.portfolio_session import (
    PortfolioSessionGate,
    SessionGatePolicy,
)
from quanthack.core.clock import CompetitionClock, UTC
from quanthack.core.config import AppConfig
from quanthack.core.instruments import instrument_for
from quanthack.market.adapters import AccountAdapter, MarketDataAdapter
from quanthack.market.market_data import PriceBar, QuoteSnapshot
from quanthack.market.market_quality import MarketQualityChecker, MarketQualityLimits
from quanthack.strategies.strategy import (
    EPSILON_NOTIONAL,
    Strategy,
    normalize_strategy_name,
)
from quanthack.trading.competition_monitor import (
    CompetitionMonitor,
    CompetitionMonitorReport,
)
from quanthack.trading.execution import DryRunExecutor, ExecutionRecord, read_journal
from quanthack.trading.risk import (
    AccountSnapshot,
    PortfolioSnapshot,
    Position,
    RiskDecision,
    RiskEngine,
    RiskLimits,
    RiskState,
    Side,
    TradeRequest,
)


@dataclass(frozen=True)
class LiveDryRunSettings:
    symbols: tuple[str, ...]
    strategy_name: str
    strategy_by_symbol: tuple[tuple[str, str], ...] = ()
    timeframe: str = "M1"
    bars: int = 120
    iterations: int = 1
    poll_seconds: float = 0.0
    journal_path: str = "outputs/live_dry_run_journal.jsonl"
    monitor_csv: str = "outputs/live_competition_monitor.csv"
    target_notional_multipliers_by_symbol: tuple[tuple[str, float], ...] = ()
    session_gate_policy: SessionGatePolicy | None = None
    deployment_profile_slot: str | None = None
    deployment_profile_label: str | None = None

    def __post_init__(self) -> None:
        if not self.symbols:
            raise ValueError("live dry-run needs at least one symbol")
        canonical_symbols: list[str] = []
        seen: set[str] = set()
        for raw_symbol in self.symbols:
            symbol = instrument_for(raw_symbol).symbol
            if symbol in seen:
                continue
            canonical_symbols.append(symbol)
            seen.add(symbol)
        object.__setattr__(self, "symbols", tuple(canonical_symbols))
        object.__setattr__(
            self,
            "strategy_name",
            normalize_strategy_name(self.strategy_name),
        )
        strategy_overrides: list[tuple[str, str]] = []
        seen_override_symbols: set[str] = set()
        for raw_symbol, raw_strategy in self.strategy_by_symbol:
            symbol = instrument_for(raw_symbol).symbol
            if symbol not in self.symbols:
                raise ValueError(
                    f"strategy override symbol {symbol} is not in live dry-run symbols"
                )
            strategy = normalize_strategy_name(raw_strategy)
            if symbol in seen_override_symbols:
                strategy_overrides = [
                    item for item in strategy_overrides if item[0] != symbol
                ]
            strategy_overrides.append((symbol, strategy))
            seen_override_symbols.add(symbol)
        object.__setattr__(
            self,
            "strategy_by_symbol",
            tuple(sorted(strategy_overrides)),
        )
        multipliers: list[tuple[str, float]] = []
        seen_multiplier_symbols: set[str] = set()
        for raw_symbol, raw_multiplier in self.target_notional_multipliers_by_symbol:
            symbol = instrument_for(raw_symbol).symbol
            if symbol not in self.symbols:
                raise ValueError(
                    f"multiplier symbol {symbol} is not in live dry-run symbols"
                )
            multiplier = float(raw_multiplier)
            if not isfinite(multiplier) or multiplier < 0:
                raise ValueError("target notional multipliers must be finite and non-negative")
            if symbol in seen_multiplier_symbols:
                multipliers = [item for item in multipliers if item[0] != symbol]
            multipliers.append((symbol, multiplier))
            seen_multiplier_symbols.add(symbol)
        object.__setattr__(
            self,
            "target_notional_multipliers_by_symbol",
            tuple(sorted(multipliers)),
        )
        if self.bars < 2:
            raise ValueError("live dry-run bars must be at least 2")
        if self.iterations < 1:
            raise ValueError("live dry-run iterations must be at least 1")
        if self.poll_seconds < 0:
            raise ValueError("live dry-run poll_seconds cannot be negative")

    def strategy_for_symbol(self, symbol: str) -> str:
        canonical = instrument_for(symbol).symbol
        for override_symbol, strategy_name in self.strategy_by_symbol:
            if override_symbol == canonical:
                return strategy_name
        return self.strategy_name

    def target_multiplier_for_symbol(self, symbol: str) -> float:
        canonical = instrument_for(symbol).symbol
        for multiplier_symbol, multiplier in self.target_notional_multipliers_by_symbol:
            if multiplier_symbol == canonical:
                return multiplier
        return 1.0


@dataclass(frozen=True)
class LiveDryRunIteration:
    timestamp: datetime
    account: AccountSnapshot
    portfolio_before: PortfolioSnapshot
    allocation: PortfolioAllocation
    records: tuple[ExecutionRecord, ...]
    error: str | None = None


@dataclass(frozen=True)
class LiveDryRunResult:
    iterations: tuple[LiveDryRunIteration, ...]
    monitor_report: CompetitionMonitorReport

    @property
    def records(self) -> tuple[ExecutionRecord, ...]:
        return tuple(record for item in self.iterations for record in item.records)


class LiveDryRunEngine:
    def __init__(
        self,
        *,
        config: AppConfig,
        settings: LiveDryRunSettings,
        market_data: MarketDataAdapter,
        account_adapter: AccountAdapter,
        risk_limits: RiskLimits | None = None,
        quality_limits: MarketQualityLimits | None = None,
        allocation_policy: AllocationPolicy | None = None,
        executor: DryRunExecutor | None = None,
        clock: CompetitionClock | None = None,
    ) -> None:
        self.config = config
        self.settings = settings
        self.market_data = market_data
        self.account_adapter = account_adapter
        self.risk_engine = RiskEngine(risk_limits or config.risk)
        self.quality_limits = quality_limits or config.market_quality
        self.quality_limits_by_symbol = {
            symbol: replace(
                self.quality_limits,
                max_spread_bps=instrument_for(symbol).max_spread_bps,
            )
            for symbol in settings.symbols
        }
        self.allocator = PortfolioAllocator(
            allocation_policy
            or AllocationPolicy(
                max_gross_leverage=(risk_limits or config.risk).max_gross_leverage,
                max_symbol_gross_pct=(risk_limits or config.risk).max_symbol_notional_pct,
            )
        )
        self.executor = executor or DryRunExecutor(Path(settings.journal_path))
        self.clock = clock or config.competition.to_clock()
        self.monitor = CompetitionMonitor()
        self._strategies = {
            symbol: config.build_strategy(
                settings.strategy_for_symbol(symbol),
                symbol=symbol,
            )
            for symbol in settings.symbols
        }
        self._holding_periods = {symbol: 0 for symbol in settings.symbols}
        self._peak_equity = config.competition.starting_equity
        self._day_start_equity = config.competition.starting_equity

    def run(self) -> LiveDryRunResult:
        iterations: list[LiveDryRunIteration] = []
        for index in range(self.settings.iterations):
            try:
                iterations.append(self.run_once())
            except Exception as exc:
                iterations.append(self._record_iteration_failure(exc))
            if index < self.settings.iterations - 1 and self.settings.poll_seconds > 0:
                sleep(self.settings.poll_seconds)
        return LiveDryRunResult(
            iterations=tuple(iterations),
            monitor_report=self.monitor.report(),
        )

    def run_once(self) -> LiveDryRunIteration:
        quotes = {
            symbol: self.market_data.get_latest_quote(symbol)
            for symbol in self.settings.symbols
        }
        histories = {
            symbol: self.market_data.get_recent_bars(
                symbol,
                timeframe=self.settings.timeframe,
                count=self.settings.bars,
            )
            for symbol in self.settings.symbols
        }
        self._update_strategy_context(histories=histories, quotes=quotes)
        timestamp = max(quote.timestamp for quote in quotes.values())
        account = self.account_adapter.get_account_snapshot(
            starting_equity=self.config.competition.starting_equity,
            day_start_equity=self._day_start_equity,
            peak_equity=self._peak_equity,
        )
        self._peak_equity = max(self._peak_equity, account.equity)
        portfolio_before = self.executor.current_portfolio()
        intents = tuple(
            self._build_intent(
                symbol=symbol,
                strategy=self._strategies[symbol],
                quote=quotes[symbol],
                bars=histories[symbol],
                portfolio=portfolio_before,
            )
            for symbol in self.settings.symbols
        )
        gated_intents = self._apply_session_gate(
            intents,
            timestamp=timestamp,
        )
        allocation = self.allocator.allocate(
            gated_intents,
            equity=account.equity,
            timestamp=timestamp.isoformat(timespec="seconds"),
        )
        records = self._submit_allocation(
            allocation=allocation,
            account=account,
            portfolio_before=portfolio_before,
            timestamp=timestamp,
        )
        portfolio_after = self.executor.current_portfolio()
        self._update_holding_periods(
            before=portfolio_before,
            after=portfolio_after,
            records=records,
        )
        self.monitor.record(
            timestamp=timestamp,
            account=account,
            portfolio=portfolio_after,
            accepted_trade_count=_accepted_count(self.executor.journal_path),
        )
        return LiveDryRunIteration(
            timestamp=timestamp,
            account=account,
            portfolio_before=portfolio_before,
            allocation=allocation,
            records=records,
        )

    def _record_iteration_failure(self, exc: Exception) -> LiveDryRunIteration:
        timestamp = datetime.now(tz=UTC)
        account = AccountSnapshot(
            equity=self.config.competition.starting_equity,
            starting_equity=self.config.competition.starting_equity,
            day_start_equity=self._day_start_equity,
            peak_equity=max(self._peak_equity, self.config.competition.starting_equity),
            margin_level_pct=None,
        )
        portfolio_before = self._safe_current_portfolio()
        reason = f"live dry-run polling failure: {exc.__class__.__name__}: {exc}"
        request = TradeRequest(
            symbol=self.settings.symbols[0],
            side=Side.BUY,
            target_notional_usd=1.0,
            reason=reason,
        )
        decision = RiskDecision(
            approved=False,
            reason=reason,
            adjusted_notional_usd=0.0,
            state=self.risk_engine.state,
        )
        record = self.executor.submit(
            account=account,
            request=request,
            decision=decision,
            mode=self.clock.mode_at(timestamp),
            portfolio_before=portfolio_before,
        )
        allocation = PortfolioAllocation(
            targets=(),
            policy=self.allocator.policy,
            equity=account.equity,
            timestamp=timestamp.isoformat(timespec="seconds"),
        )
        self.monitor.record(
            timestamp=timestamp,
            account=account,
            portfolio=portfolio_before,
            accepted_trade_count=_accepted_count(self.executor.journal_path),
        )
        return LiveDryRunIteration(
            timestamp=timestamp,
            account=account,
            portfolio_before=portfolio_before,
            allocation=allocation,
            records=(record,),
            error=reason,
        )

    def _safe_current_portfolio(self) -> PortfolioSnapshot:
        try:
            return self.executor.current_portfolio()
        except Exception:
            return PortfolioSnapshot()

    def _build_intent(
        self,
        *,
        symbol: str,
        strategy: Strategy,
        quote: QuoteSnapshot,
        bars: tuple[PriceBar, ...],
        portfolio: PortfolioSnapshot,
    ) -> SymbolIntent:
        canonical = instrument_for(symbol).symbol
        current_notional = portfolio.notional_for_symbol(canonical)
        quality = MarketQualityChecker(self.quality_limits_by_symbol[canonical]).evaluate(
            quote=quote,
            as_of=quote.timestamp,
        )
        if not quality.ok:
            return SymbolIntent(
                symbol=canonical,
                target_notional_usd=current_notional,
                current_notional_usd=current_notional,
                reason=f"market quality hold: {quality.reason}",
                primary_signal="market_quality",
            )
        closes = [bar.close for bar in bars]
        multiplier = self.settings.target_multiplier_for_symbol(canonical)
        if hasattr(strategy, "generate_decision"):
            decision = strategy.generate_decision(
                closes,
                current_notional_usd=current_notional,
                holding_period=self._holding_periods[canonical],
                quote=quote,
            )
            target = (
                decision.target_notional_usd * multiplier
                if decision.is_trade_intent
                else current_notional
            )
            return SymbolIntent(
                symbol=canonical,
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
                symbol=canonical,
                target_notional_usd=current_notional,
                current_notional_usd=current_notional,
                reason="no strategy request",
            )
        target = (
            request.target_notional_usd
            if request.side == Side.BUY
            else -request.target_notional_usd
        )
        return SymbolIntent(
            symbol=canonical,
            target_notional_usd=target * multiplier,
            current_notional_usd=current_notional,
            reason=request.reason,
            primary_signal="request",
        )

    def _apply_session_gate(
        self,
        intents: tuple[SymbolIntent, ...],
        *,
        timestamp: datetime,
    ) -> tuple[SymbolIntent, ...]:
        if self.settings.session_gate_policy is None:
            return intents
        return PortfolioSessionGate(self.settings.session_gate_policy).apply(
            intents,
            timestamp=timestamp,
        )

    def _update_strategy_context(
        self,
        *,
        histories: dict[str, tuple[PriceBar, ...]],
        quotes: dict[str, QuoteSnapshot],
    ) -> None:
        closes_by_symbol = {
            symbol: tuple(bar.close for bar in bars)
            for symbol, bars in histories.items()
        }
        for strategy in self._strategies.values():
            update_context = getattr(strategy, "update_portfolio_context", None)
            if callable(update_context):
                update_context(
                    closes_by_symbol=closes_by_symbol,
                    quotes_by_symbol=quotes,
                )

    def _submit_allocation(
        self,
        *,
        allocation: PortfolioAllocation,
        account: AccountSnapshot,
        portfolio_before: PortfolioSnapshot,
        timestamp: datetime,
    ) -> tuple[ExecutionRecord, ...]:
        records: list[ExecutionRecord] = []
        mode = self.clock.mode_at(timestamp)
        portfolio = portfolio_before
        for target in allocation.targets:
            if abs(target.change_notional_usd) <= EPSILON_NOTIONAL:
                if target.primary_signal == "market_quality":
                    records.append(
                        self._submit_market_quality_hold(
                            target=target,
                            account=account,
                            portfolio=portfolio,
                            mode=mode,
                        )
                    )
                continue
            request, decision = self._request_and_decision(
                target_symbol=target.symbol,
                current_notional=target.current_notional_usd,
                adjusted_target=target.adjusted_notional_usd,
                account=account,
                portfolio=portfolio,
                mode=mode,
                reason=_allocation_reason(target),
            )
            record = self.executor.submit(
                account=account,
                request=request,
                decision=decision,
                mode=mode,
                portfolio_before=portfolio,
            )
            records.append(record)
            if decision.approved:
                portfolio = _portfolio_after_target(
                    portfolio=portfolio,
                    symbol=target.symbol,
                    signed_notional=(
                        _signed_target(target.adjusted_notional_usd, decision.adjusted_notional_usd)
                    ),
                )
        return tuple(records)

    def _submit_market_quality_hold(
        self,
        *,
        target,
        account: AccountSnapshot,
        portfolio: PortfolioSnapshot,
        mode,
    ) -> ExecutionRecord:
        side = Side.SELL if target.current_notional_usd > 0 else Side.BUY
        request = TradeRequest(
            symbol=target.symbol,
            side=side,
            target_notional_usd=max(abs(target.current_notional_usd), 1.0),
            reason=_allocation_reason(target),
        )
        decision = RiskDecision(
            approved=False,
            reason=request.reason,
            adjusted_notional_usd=0.0,
            state=self.risk_engine.state,
        )
        return self.executor.submit(
            account=account,
            request=request,
            decision=decision,
            mode=mode,
            portfolio_before=portfolio,
        )

    def _request_and_decision(
        self,
        *,
        target_symbol: str,
        current_notional: float,
        adjusted_target: float,
        account: AccountSnapshot,
        portfolio: PortfolioSnapshot,
        mode,
        reason: str,
    ) -> tuple[TradeRequest, RiskDecision]:
        if abs(adjusted_target) <= EPSILON_NOTIONAL:
            side = Side.SELL if current_notional > 0 else Side.BUY
            request = TradeRequest(
                symbol=target_symbol,
                side=side,
                target_notional_usd=max(abs(current_notional), 1.0),
                reason=f"{reason}; allocated exit",
            )
            decision = self.risk_engine.evaluate_exit(
                account=account,
                portfolio=portfolio,
                symbol=target_symbol,
                mode=mode,
                reason="allocated exit",
            )
            return (
                request,
                decision,
            )

        side = Side.BUY if adjusted_target > 0 else Side.SELL
        request = TradeRequest(
            symbol=target_symbol,
            side=side,
            target_notional_usd=abs(adjusted_target),
            reason=reason,
        )
        decision = self.risk_engine.evaluate(
            account=account,
            portfolio=portfolio,
            request=request,
            mode=mode,
        )
        return request, decision

    def _update_holding_periods(
        self,
        *,
        before: PortfolioSnapshot,
        after: PortfolioSnapshot,
        records: tuple[ExecutionRecord, ...],
    ) -> None:
        traded_symbols = {record.request.symbol for record in records if record.decision.approved}
        for symbol in self.settings.symbols:
            previous_direction = _notional_direction(before.notional_for_symbol(symbol))
            current_direction = _notional_direction(after.notional_for_symbol(symbol))
            if current_direction == 0:
                self._holding_periods[symbol] = 0
            elif symbol in traded_symbols and current_direction != previous_direction:
                self._holding_periods[symbol] = 1
            else:
                self._holding_periods[symbol] += 1


def _allocation_reason(target) -> str:
    parts = []
    if target.intent_reason:
        parts.append(target.intent_reason)
    if target.reasons:
        parts.append(f"allocation: {'; '.join(target.reasons)}")
    return "; ".join(parts) if parts else "allocated target"


def _accepted_count(journal_path: Path) -> int:
    return len(
        [
            record
            for record in read_journal(journal_path)
            if record.get("status") == "DRY_RUN_ACCEPTED"
        ]
    )


def _portfolio_after_target(
    *,
    portfolio: PortfolioSnapshot,
    symbol: str,
    signed_notional: float,
) -> PortfolioSnapshot:
    positions = {
        position.symbol: position.notional_usd
        for position in portfolio.positions
    }
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
                key=lambda item: item.symbol,
            )
        )
    )


def _signed_target(requested_signed: float, adjusted_abs: float) -> float:
    if abs(adjusted_abs) <= EPSILON_NOTIONAL:
        return 0.0
    return adjusted_abs if requested_signed > 0 else -adjusted_abs


def _notional_direction(notional: float) -> int:
    if notional > EPSILON_NOTIONAL:
        return 1
    if notional < -EPSILON_NOTIONAL:
        return -1
    return 0
