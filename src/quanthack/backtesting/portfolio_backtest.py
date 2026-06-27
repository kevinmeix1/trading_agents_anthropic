from __future__ import annotations

import csv
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from quanthack.backtesting.backtest import (
    BacktestFill,
    FillModel,
    _signed_notional,
    _target_position_trade,
    _unit_direction,
)
from quanthack.core.clock import CompetitionClock
from quanthack.market.market_data import PriceBar, PriceHistory, QuoteHistory, QuoteSnapshot
from quanthack.market.market_quality import MarketQualityChecker, MarketQualityLimits
from quanthack.backtesting.metrics import PerformanceMetrics, summarize_performance
from quanthack.backtesting.pnl import PnlLedger, build_pnl_ledger
from quanthack.backtesting.portfolio_allocator import (
    AllocatedTarget,
    AllocationPolicy,
    PortfolioAllocation,
    PortfolioAllocator,
    SymbolIntent,
)
from quanthack.backtesting.portfolio_regime import (
    PortfolioRegimeTilter,
    RegimeTiltPolicy,
    RegimeTiltReport,
)
from quanthack.backtesting.portfolio_session import (
    PortfolioSessionGate,
    SessionGatePolicy,
)
from quanthack.backtesting.portfolio_symbol_evidence import (
    PortfolioSymbolEvidenceGate,
    SymbolEvidenceGatePolicy,
    SymbolEvidenceGateReport,
)
from quanthack.backtesting.portfolio_volatility import (
    PortfolioVolatilityTargeter,
    VolatilityTargetingPolicy,
    VolatilityTargetingReport,
)
from quanthack.trading.position_risk import PositionCostBasis, evaluate_position_stop
from quanthack.trading.risk import (
    AccountSnapshot,
    PortfolioSnapshot,
    Position,
    RiskEngine,
    RiskLimits,
    Side,
    TradeRequest,
)
from quanthack.strategies.strategy import EPSILON_NOTIONAL, Strategy


@dataclass(frozen=True)
class PortfolioEquityPoint:
    timestamp: str
    equity: float
    cash: float
    gross_notional_usd: float
    net_notional_usd: float
    drawdown_pct: float
    positions: tuple[Position, ...]


@dataclass(frozen=True)
class SymbolPnlLedger:
    symbol: str
    ledger: PnlLedger


@dataclass(frozen=True)
class PortfolioBacktestResult:
    symbols: tuple[str, ...]
    equity_curve: tuple[PortfolioEquityPoint, ...]
    fills: tuple[BacktestFill, ...]
    metrics: PerformanceMetrics
    pnl_by_symbol: tuple[SymbolPnlLedger, ...]
    allocation_reports: tuple[PortfolioAllocation, ...] = ()
    regime_reports: tuple[RegimeTiltReport, ...] = ()
    volatility_reports: tuple[VolatilityTargetingReport, ...] = ()
    symbol_evidence_reports: tuple[SymbolEvidenceGateReport, ...] = ()

    @property
    def realized_pnl_usd(self) -> float:
        return sum(row.ledger.realized_pnl_usd for row in self.pnl_by_symbol)

    @property
    def open_pnl_usd(self) -> float:
        return sum(row.ledger.open_pnl_usd for row in self.pnl_by_symbol)

    @property
    def total_pnl_usd(self) -> float:
        return sum(row.ledger.total_pnl_usd for row in self.pnl_by_symbol)

    def pnl_for_symbol(self, symbol: str) -> PnlLedger:
        for row in self.pnl_by_symbol:
            if row.symbol == symbol:
                return row.ledger
        raise KeyError(f"no P&L ledger for {symbol}")


@dataclass
class _SymbolState:
    closes_so_far: list[float] = field(default_factory=list)
    position_units: float = 0.0
    holding_period: int = 0
    cost_basis: PositionCostBasis = field(default_factory=PositionCostBasis)


class PortfolioBacktestEngine:
    def __init__(
        self,
        *,
        strategies: Mapping[str, Strategy],
        risk_limits: RiskLimits,
        quality_limits: MarketQualityLimits,
        quality_limits_by_symbol: Mapping[str, MarketQualityLimits] | None = None,
        allocation_policy: AllocationPolicy | None = None,
        clock: CompetitionClock,
        fill_model: FillModel | None = None,
        periods_per_year: float = 252.0,
        target_notional_multiplier: float = 1.0,
        target_notional_multipliers_by_symbol: Mapping[str, float] | None = None,
        regime_tilt_policy: RegimeTiltPolicy | None = None,
        session_gate_policy: SessionGatePolicy | None = None,
        volatility_targeting_policy: VolatilityTargetingPolicy | None = None,
        symbol_evidence_gate_policy: SymbolEvidenceGatePolicy | None = None,
    ) -> None:
        if not strategies:
            raise ValueError("portfolio backtest requires at least one strategy")
        if not 0 < target_notional_multiplier <= 1:
            raise ValueError("target_notional_multiplier must be in (0, 1]")
        target_notional_multipliers = dict(target_notional_multipliers_by_symbol or {})
        invalid_multipliers = {
            symbol: multiplier
            for symbol, multiplier in target_notional_multipliers.items()
            if not 0 <= multiplier <= 1
        }
        if invalid_multipliers:
            raise ValueError("target_notional_multipliers_by_symbol values must be in [0, 1]")

        self.strategies = dict(strategies)
        self.symbols = tuple(self.strategies.keys())
        self.risk_limits = risk_limits
        self.quality_limits = quality_limits
        self.quality_limits_by_symbol = dict(quality_limits_by_symbol or {})
        self.allocation_policy = allocation_policy or AllocationPolicy(
            max_gross_leverage=risk_limits.max_gross_leverage,
            max_symbol_gross_pct=risk_limits.max_symbol_notional_pct,
        )
        self.allocator = PortfolioAllocator(self.allocation_policy)
        self.clock = clock
        self.fill_model = fill_model or FillModel()
        self.periods_per_year = periods_per_year
        self.target_notional_multiplier = target_notional_multiplier
        self.target_notional_multipliers_by_symbol = target_notional_multipliers
        self.regime_tilter = (
            PortfolioRegimeTilter(regime_tilt_policy)
            if regime_tilt_policy is not None
            else None
        )
        self.session_gate = (
            PortfolioSessionGate(session_gate_policy)
            if session_gate_policy is not None
            else None
        )
        self.volatility_targeter = (
            PortfolioVolatilityTargeter(volatility_targeting_policy)
            if volatility_targeting_policy is not None
            else None
        )
        self.symbol_evidence_gate = (
            PortfolioSymbolEvidenceGate(symbol_evidence_gate_policy)
            if symbol_evidence_gate_policy is not None
            else None
        )

    def run(
        self,
        *,
        prices: PriceHistory,
        quotes: QuoteHistory,
        starting_equity: float,
    ) -> PortfolioBacktestResult:
        bars_by_symbol, quotes_by_symbol, timestamps = self._aligned_data(prices, quotes)
        states = {symbol: _SymbolState() for symbol in self.symbols}
        cash = starting_equity
        peak_equity = starting_equity
        risk_engine = RiskEngine(self.risk_limits)
        fills: list[BacktestFill] = []
        equity_points: list[PortfolioEquityPoint] = []
        allocation_reports: list[PortfolioAllocation] = []
        regime_reports: list[RegimeTiltReport] = []
        volatility_reports: list[VolatilityTargetingReport] = []
        symbol_evidence_reports: list[SymbolEvidenceGateReport] = []
        final_mark_prices: dict[str, float] = {}

        for timestamp in timestamps:
            current_quotes = {
                symbol: quotes_by_symbol[symbol][timestamp]
                for symbol in self.symbols
            }
            for symbol in self.symbols:
                state = states[symbol]
                quote = current_quotes[symbol]
                state.closes_so_far.append(quote.mid)
                final_mark_prices[symbol] = quote.mid
            self._update_strategy_context(states=states, quotes=current_quotes)

            equity_before = _current_equity(cash, states, current_quotes)
            peak_equity = max(peak_equity, equity_before)
            account = AccountSnapshot(
                equity=equity_before,
                starting_equity=starting_equity,
                day_start_equity=starting_equity,
                peak_equity=peak_equity,
                margin_level_pct=2_000,
            )
            portfolio = PortfolioSnapshot(
                positions=_positions(states=states, quotes=current_quotes)
            )
            intents = tuple(
                self._build_symbol_intent(
                    symbol=symbol,
                    strategy=self.strategies[symbol],
                    state=states[symbol],
                    quote=current_quotes[symbol],
                    bar=bars_by_symbol[symbol][timestamp],
                )
                for symbol in self.symbols
            )
            closes_by_symbol = {
                symbol: tuple(state.closes_so_far)
                for symbol, state in states.items()
            }
            if self.regime_tilter is not None:
                intents, timestamp_regime_reports = self.regime_tilter.apply(
                    intents,
                    closes_by_symbol=closes_by_symbol,
                    timestamp=timestamp.isoformat(timespec="seconds"),
                )
                regime_reports.extend(timestamp_regime_reports)
            if self.volatility_targeter is not None:
                intents, volatility_report = self.volatility_targeter.apply(
                    intents,
                    closes_by_symbol=closes_by_symbol,
                    equity=equity_before,
                    timestamp=timestamp.isoformat(timespec="seconds"),
                )
                volatility_reports.append(volatility_report)
            if self.session_gate is not None:
                intents = self.session_gate.apply(
                    intents,
                    timestamp=timestamp,
                )
            if self.symbol_evidence_gate is not None:
                intents, timestamp_symbol_evidence_reports = (
                    self.symbol_evidence_gate.apply(
                        intents,
                        timestamp=timestamp.isoformat(timespec="seconds"),
                    )
                )
                symbol_evidence_reports.extend(timestamp_symbol_evidence_reports)
            allocation = self.allocator.allocate(
                intents,
                equity=equity_before,
                timestamp=timestamp.isoformat(timespec="seconds"),
            )
            allocation_reports.append(allocation)
            direction_before = {
                symbol: _unit_direction(state.position_units)
                for symbol, state in states.items()
            }
            trade_executed = {symbol: False for symbol in states}

            for target in allocation.targets:
                symbol = target.symbol
                state = states[symbol]
                quote = current_quotes[symbol]
                equity_now = _current_equity(cash, states, current_quotes)
                peak_equity = max(peak_equity, equity_now)
                account = AccountSnapshot(
                    equity=equity_now,
                    starting_equity=starting_equity,
                    day_start_equity=starting_equity,
                    peak_equity=peak_equity,
                    margin_level_pct=2_000,
                )
                portfolio = PortfolioSnapshot(
                    positions=_positions(states=states, quotes=current_quotes)
                )
                trade = self._execute_allocated_target(
                    target=target,
                    state=state,
                    quote=quote,
                    account=account,
                    portfolio=portfolio,
                    timestamp=timestamp,
                    risk_engine=risk_engine,
                )
                if trade is not None:
                    fill, target_units = trade
                    cash -= fill.trade_units * fill.fill_price
                    state.position_units = target_units
                    state.cost_basis = state.cost_basis.apply_fill(
                        trade_units=fill.trade_units,
                        fill_price=fill.fill_price,
                    )
                    trade_executed[symbol] = True
                    fills.append(fill)
                    if self.symbol_evidence_gate is not None:
                        self.symbol_evidence_gate.observe_fill(fill)

            for symbol, state in states.items():
                direction_after = _unit_direction(state.position_units)
                if direction_after == 0:
                    state.holding_period = 0
                elif (
                    trade_executed[symbol]
                    and direction_after != direction_before[symbol]
                ):
                    state.holding_period = 1
                elif direction_after != 0:
                    state.holding_period += 1

            equity_after = _current_equity(cash, states, current_quotes)
            peak_equity = max(peak_equity, equity_after)
            positions = _positions(states=states, quotes=current_quotes)
            gross_notional = sum(abs(position.notional_usd) for position in positions)
            net_notional = sum(position.notional_usd for position in positions)
            equity_points.append(
                PortfolioEquityPoint(
                    timestamp=timestamp.isoformat(timespec="seconds"),
                    equity=equity_after,
                    cash=cash,
                    gross_notional_usd=gross_notional,
                    net_notional_usd=net_notional,
                    drawdown_pct=max(0.0, 1.0 - (equity_after / peak_equity)),
                    positions=positions,
                )
            )

        pnl_by_symbol = tuple(
            SymbolPnlLedger(
                symbol=symbol,
                ledger=build_pnl_ledger(
                    [fill for fill in fills if fill.symbol == symbol],
                    final_mark_price=final_mark_prices.get(symbol),
                ),
            )
            for symbol in self.symbols
        )
        metrics = summarize_performance(
            equity_curve=[point.equity for point in equity_points],
            turnover_notional=sum(fill.turnover_notional_usd for fill in fills),
            periods_per_year=self.periods_per_year,
        )
        return PortfolioBacktestResult(
            symbols=self.symbols,
            equity_curve=tuple(equity_points),
            fills=tuple(fills),
            metrics=metrics,
            pnl_by_symbol=pnl_by_symbol,
            allocation_reports=tuple(allocation_reports),
            regime_reports=tuple(regime_reports),
            volatility_reports=tuple(volatility_reports),
            symbol_evidence_reports=tuple(symbol_evidence_reports),
        )

    def _build_symbol_intent(
        self,
        *,
        symbol: str,
        strategy: Strategy,
        state: _SymbolState,
        quote: QuoteSnapshot,
        bar: PriceBar,
    ) -> SymbolIntent:
        current_notional_usd = state.position_units * quote.mid
        stop_decision = evaluate_position_stop(
            symbol=symbol,
            cost_basis=state.cost_basis,
            mark_price=quote.mid,
            max_position_loss_pct=self.risk_limits.max_position_loss_for_symbol(symbol),
        )
        if stop_decision.triggered:
            return SymbolIntent(
                symbol=symbol,
                target_notional_usd=0.0,
                current_notional_usd=current_notional_usd,
                reason=stop_decision.reason,
                primary_signal="position_stop",
            )

        quality = MarketQualityChecker(
            self.quality_limits_by_symbol.get(symbol, self.quality_limits)
        ).evaluate(quote=quote, as_of=bar.timestamp)
        if not quality.ok:
            return SymbolIntent(
                symbol=symbol,
                target_notional_usd=current_notional_usd,
                current_notional_usd=current_notional_usd,
                reason=f"market quality hold: {quality.reason}",
                primary_signal="market_quality",
            )

        if hasattr(strategy, "generate_decision"):
            strategy_decision = strategy.generate_decision(
                state.closes_so_far,
                current_notional_usd=current_notional_usd,
                holding_period=state.holding_period,
                quote=quote,
            )
            target_notional_usd = (
                _scale_target_notional(
                    strategy_decision.target_notional_usd,
                    multiplier=self._target_notional_multiplier(symbol),
                )
                if strategy_decision.is_trade_intent
                else current_notional_usd
            )
            return SymbolIntent(
                symbol=symbol,
                target_notional_usd=target_notional_usd,
                current_notional_usd=current_notional_usd,
                reason=strategy_decision.reason,
                primary_signal=strategy_decision.primary_signal,
                supporting_signals=strategy_decision.supporting_signals,
                conflicting_signals=strategy_decision.conflicting_signals,
            )

        request = strategy.generate_request(state.closes_so_far)
        if request is None:
            return SymbolIntent(
                symbol=symbol,
                target_notional_usd=current_notional_usd,
                current_notional_usd=current_notional_usd,
                reason="no strategy request",
            )

        direction = 1 if request.side.value == "BUY" else -1
        target_notional_usd = _scale_target_notional(
            direction * request.target_notional_usd,
            multiplier=self._target_notional_multiplier(symbol),
        )
        return SymbolIntent(
            symbol=symbol,
            target_notional_usd=target_notional_usd,
            current_notional_usd=current_notional_usd,
            reason=request.reason,
            primary_signal="request",
        )

    def _target_notional_multiplier(self, symbol: str) -> float:
        return (
            self.target_notional_multiplier
            * self.target_notional_multipliers_by_symbol.get(symbol, 1.0)
        )

    def _update_strategy_context(
        self,
        *,
        states: Mapping[str, _SymbolState],
        quotes: Mapping[str, QuoteSnapshot],
    ) -> None:
        closes_by_symbol = {
            symbol: tuple(state.closes_so_far)
            for symbol, state in states.items()
        }
        for strategy in self.strategies.values():
            update_context = getattr(strategy, "update_portfolio_context", None)
            if callable(update_context):
                update_context(
                    closes_by_symbol=closes_by_symbol,
                    quotes_by_symbol=quotes,
                )

    def _execute_allocated_target(
        self,
        *,
        target: AllocatedTarget,
        state: _SymbolState,
        quote: QuoteSnapshot,
        account: AccountSnapshot,
        portfolio: PortfolioSnapshot,
        timestamp: datetime,
        risk_engine: RiskEngine,
    ) -> tuple[BacktestFill, float] | None:
        if abs(target.change_notional_usd) <= EPSILON_NOTIONAL:
            return None

        risk_reason = _allocation_reason(target)
        target_notional_usd = target.adjusted_notional_usd
        if abs(target_notional_usd) <= EPSILON_NOTIONAL:
            decision = risk_engine.evaluate_exit(
                account=account,
                portfolio=portfolio,
                symbol=target.symbol,
                mode=self.clock.mode_at(timestamp),
                reason=risk_reason,
            )
            risk_reason = f"{risk_reason}; risk: {decision.reason}"
            if not decision.approved:
                return None
            target_trade = _target_position_trade(
                target_notional_usd=0.0,
                quote=quote,
                current_units=state.position_units,
                fill_model=self.fill_model,
            )
            adjusted_abs_notional = decision.adjusted_notional_usd
        else:
            request_side = "BUY" if target_notional_usd > 0 else "SELL"
            request = _trade_request(
                symbol=target.symbol,
                side_value=request_side,
                target_abs_notional_usd=abs(target_notional_usd),
                reason=risk_reason,
            )
            decision = risk_engine.evaluate(
                account=account,
                portfolio=portfolio,
                request=request,
                mode=self.clock.mode_at(timestamp),
            )
            risk_reason = f"{risk_reason}; risk: {decision.reason}"
            if not decision.approved:
                return None
            adjusted_abs_notional = decision.adjusted_notional_usd
            target_trade = _target_position_trade(
                target_notional_usd=_signed_notional(
                    target_notional_usd,
                    adjusted_abs_notional,
                ),
                quote=quote,
                current_units=state.position_units,
                fill_model=self.fill_model,
            )

        if target_trade is None:
            return None

        return (
            BacktestFill(
                timestamp=timestamp.isoformat(timespec="seconds"),
                symbol=target.symbol,
                side=target_trade.side,
                fill_price=target_trade.fill_price,
                trade_units=target_trade.trade_units,
                requested_notional_usd=abs(target.requested_notional_usd),
                adjusted_notional_usd=adjusted_abs_notional,
                risk_reason=risk_reason,
                primary_signal=target.primary_signal,
                supporting_signals=target.supporting_signals,
                conflicting_signals=target.conflicting_signals,
            ),
            target_trade.target_units,
        )

    def _aligned_data(
        self,
        prices: PriceHistory,
        quotes: QuoteHistory,
    ) -> tuple[
        dict[str, dict[datetime, PriceBar]],
        dict[str, dict[datetime, QuoteSnapshot]],
        list[datetime],
    ]:
        bars_by_symbol: dict[str, dict[datetime, PriceBar]] = {}
        quotes_by_symbol: dict[str, dict[datetime, QuoteSnapshot]] = {}
        common_timestamps: set[datetime] | None = None

        for symbol in self.symbols:
            bars = prices.for_symbol(symbol).bars
            symbol_quotes = quotes.for_symbol(symbol).quotes
            if not bars:
                raise ValueError(f"no price bars for {symbol}")
            if not symbol_quotes:
                raise ValueError(f"no quotes for {symbol}")

            bar_by_time = {bar.timestamp: bar for bar in bars}
            quote_by_time = {quote.timestamp: quote for quote in symbol_quotes}
            missing_quotes = sorted(set(bar_by_time) - set(quote_by_time))
            if missing_quotes:
                first_missing = missing_quotes[0].isoformat()
                raise ValueError(f"missing quote for {symbol} at {first_missing}")

            bars_by_symbol[symbol] = bar_by_time
            quotes_by_symbol[symbol] = quote_by_time
            timestamps = set(bar_by_time) & set(quote_by_time)
            common_timestamps = (
                timestamps
                if common_timestamps is None
                else common_timestamps & timestamps
            )

        if not common_timestamps:
            joined_symbols = ", ".join(self.symbols)
            raise ValueError(f"no common price/quote timestamps for {joined_symbols}")

        return bars_by_symbol, quotes_by_symbol, sorted(common_timestamps)


def write_portfolio_equity_curve_csv(
    result: PortfolioBacktestResult,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "timestamp",
                "equity",
                "cash",
                "gross_notional_usd",
                "net_notional_usd",
                "drawdown_pct",
                "positions",
            ],
        )
        writer.writeheader()
        for point in result.equity_curve:
            writer.writerow(
                {
                    "timestamp": point.timestamp,
                    "equity": point.equity,
                    "cash": point.cash,
                    "gross_notional_usd": point.gross_notional_usd,
                    "net_notional_usd": point.net_notional_usd,
                    "drawdown_pct": point.drawdown_pct,
                    "positions": _positions_text(point.positions),
                }
            )


def write_portfolio_pnl_summary_csv(
    result: PortfolioBacktestResult,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "symbol",
                "fills",
                "realized_pnl_usd",
                "open_pnl_usd",
                "total_pnl_usd",
                "final_position_units",
                "average_entry_price",
                "final_mark_price",
            ],
        )
        writer.writeheader()
        for row in result.pnl_by_symbol:
            writer.writerow(
                {
                    "symbol": row.symbol,
                    "fills": len([fill for fill in result.fills if fill.symbol == row.symbol]),
                    "realized_pnl_usd": row.ledger.realized_pnl_usd,
                    "open_pnl_usd": row.ledger.open_pnl_usd,
                    "total_pnl_usd": row.ledger.total_pnl_usd,
                    "final_position_units": row.ledger.final_position_units,
                    "average_entry_price": row.ledger.average_entry_price,
                    "final_mark_price": row.ledger.final_mark_price,
                }
            )
        writer.writerow(
            {
                "symbol": "PORTFOLIO",
                "fills": len(result.fills),
                "realized_pnl_usd": result.realized_pnl_usd,
                "open_pnl_usd": result.open_pnl_usd,
                "total_pnl_usd": result.total_pnl_usd,
                "final_position_units": "",
                "average_entry_price": "",
                "final_mark_price": "",
            }
        )


def write_portfolio_fills_csv(
    result: PortfolioBacktestResult,
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
        for fill in result.fills:
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


def _current_equity(
    cash: float,
    states: Mapping[str, _SymbolState],
    quotes: Mapping[str, QuoteSnapshot],
) -> float:
    return cash + sum(
        state.position_units * quotes[symbol].mid
        for symbol, state in states.items()
    )


def _positions(
    *,
    states: Mapping[str, _SymbolState],
    quotes: Mapping[str, QuoteSnapshot],
) -> tuple[Position, ...]:
    return tuple(
        Position(symbol=symbol, notional_usd=state.position_units * quotes[symbol].mid)
        for symbol, state in states.items()
        if state.position_units != 0
    )


def _positions_text(positions: tuple[Position, ...]) -> str:
    if not positions:
        return ""
    return ";".join(
        f"{position.symbol}={position.notional_usd:.2f}"
        for position in positions
    )


def _allocation_reason(target: AllocatedTarget) -> str:
    parts = []
    if target.intent_reason:
        parts.append(target.intent_reason)
    if target.reasons:
        parts.append(f"allocation: {'; '.join(target.reasons)}")
    if not parts:
        parts.append("allocated target")
    return "; ".join(parts)


def _trade_request(
    *,
    symbol: str,
    side_value: str,
    target_abs_notional_usd: float,
    reason: str,
) -> TradeRequest:
    return TradeRequest(
        symbol=symbol,
        side=Side.BUY if side_value == "BUY" else Side.SELL,
        target_notional_usd=target_abs_notional_usd,
        reason=reason,
    )


def _scale_target_notional(
    target_notional_usd: float,
    *,
    multiplier: float,
) -> float:
    if abs(target_notional_usd) <= EPSILON_NOTIONAL:
        return 0.0
    return target_notional_usd * multiplier
