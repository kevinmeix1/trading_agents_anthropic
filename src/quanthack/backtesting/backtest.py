from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from quanthack.core.clock import CompetitionClock
from quanthack.market.market_data import PriceBar, PriceHistory, QuoteHistory, QuoteSnapshot
from quanthack.market.market_quality import MarketQualityChecker, MarketQualityLimits
from quanthack.backtesting.metrics import PerformanceMetrics, summarize_performance
from quanthack.backtesting.pnl import PnlLedger, build_pnl_ledger
from quanthack.trading.position_risk import PositionCostBasis, evaluate_position_stop
from quanthack.trading.risk import (
    AccountSnapshot,
    PortfolioSnapshot,
    Position,
    RiskEngine,
    RiskLimits,
    Side,
)
from quanthack.strategies.strategy import EPSILON_NOTIONAL, Strategy


EPSILON_UNITS = 1e-12


@dataclass(frozen=True)
class FillModel:
    slippage_bps: float = 1.0

    def fill_price(self, *, side: Side, quote: QuoteSnapshot) -> float:
        multiplier = self.slippage_bps / 10_000
        if side == Side.BUY:
            return quote.ask * (1.0 + multiplier)
        return quote.bid * (1.0 - multiplier)


@dataclass(frozen=True)
class BacktestFill:
    timestamp: str
    symbol: str
    side: Side
    fill_price: float
    trade_units: float
    requested_notional_usd: float
    adjusted_notional_usd: float
    risk_reason: str
    primary_signal: str = "unknown"
    supporting_signals: tuple[str, ...] = ()
    conflicting_signals: tuple[str, ...] = ()

    @property
    def turnover_notional_usd(self) -> float:
        return abs(self.trade_units * self.fill_price)


@dataclass(frozen=True)
class EquityPoint:
    timestamp: str
    close: float
    equity: float
    cash: float
    position_units: float
    position_notional_usd: float
    drawdown_pct: float


@dataclass(frozen=True)
class BacktestResult:
    symbol: str
    equity_curve: tuple[EquityPoint, ...]
    fills: tuple[BacktestFill, ...]
    metrics: PerformanceMetrics
    pnl_ledger: PnlLedger


@dataclass(frozen=True)
class TargetTrade:
    side: Side
    fill_price: float
    target_units: float
    trade_units: float


class BacktestEngine:
    def __init__(
        self,
        *,
        strategy: Strategy,
        risk_limits: RiskLimits,
        quality_limits: MarketQualityLimits,
        clock: CompetitionClock,
        fill_model: FillModel | None = None,
        periods_per_year: float = 252.0,
    ) -> None:
        self.strategy = strategy
        self.risk_limits = risk_limits
        self.quality_limits = quality_limits
        self.clock = clock
        self.fill_model = fill_model or FillModel()
        self.periods_per_year = periods_per_year

    def run(
        self,
        *,
        prices: PriceHistory,
        quotes: QuoteHistory,
        symbol: str,
        starting_equity: float,
    ) -> BacktestResult:
        price_bars = prices.for_symbol(symbol).bars
        quote_by_time = {quote.timestamp: quote for quote in quotes.for_symbol(symbol).quotes}

        if not price_bars:
            raise ValueError(f"no price bars for {symbol}")

        cash = starting_equity
        position_units = 0.0
        peak_equity = starting_equity
        risk_engine = RiskEngine(self.risk_limits)
        quality_checker = MarketQualityChecker(self.quality_limits)
        fills: list[BacktestFill] = []
        equity_points: list[EquityPoint] = []
        closes_so_far: list[float] = []
        final_mark_price: float | None = None
        cost_basis = PositionCostBasis()
        holding_period = 0

        for bar in price_bars:
            quote = quote_by_time.get(bar.timestamp)
            if quote is None:
                raise ValueError(f"missing quote for {symbol} at {bar.timestamp.isoformat()}")
            final_mark_price = quote.mid

            closes_so_far.append(quote.mid)
            equity_before = cash + position_units * quote.mid
            peak_equity = max(peak_equity, equity_before)
            account = AccountSnapshot(
                equity=equity_before,
                starting_equity=starting_equity,
                day_start_equity=starting_equity,
                peak_equity=peak_equity,
                margin_level_pct=2_000,
            )
            portfolio = PortfolioSnapshot(
                positions=(Position(symbol=symbol, notional_usd=position_units * quote.mid),)
            )

            quality = quality_checker.evaluate(quote=quote, as_of=bar.timestamp)
            trade_executed = False
            direction_before = _unit_direction(position_units)
            stop_decision = evaluate_position_stop(
                symbol=symbol,
                cost_basis=cost_basis,
                mark_price=quote.mid,
                max_position_loss_pct=self.risk_limits.max_position_loss_for_symbol(symbol),
            )
            if stop_decision.triggered:
                exit_decision = risk_engine.evaluate_exit(
                    account=account,
                    portfolio=portfolio,
                    symbol=symbol,
                    mode=self.clock.mode_at(bar.timestamp),
                    reason=stop_decision.reason,
                )
                target_trade = (
                    _target_position_trade(
                        target_notional_usd=0.0,
                        quote=quote,
                        current_units=position_units,
                        fill_model=self.fill_model,
                    )
                    if exit_decision.approved
                    else None
                )
                if target_trade is not None:
                    current_notional = abs(position_units * quote.mid)
                    cash -= target_trade.trade_units * target_trade.fill_price
                    position_units = target_trade.target_units
                    cost_basis = cost_basis.apply_fill(
                        trade_units=target_trade.trade_units,
                        fill_price=target_trade.fill_price,
                    )
                    trade_executed = True
                    fills.append(
                        BacktestFill(
                            timestamp=bar.timestamp.isoformat(timespec="seconds"),
                            symbol=symbol,
                            side=target_trade.side,
                            fill_price=target_trade.fill_price,
                            trade_units=target_trade.trade_units,
                            requested_notional_usd=current_notional,
                            adjusted_notional_usd=exit_decision.adjusted_notional_usd,
                            risk_reason=exit_decision.reason,
                            primary_signal="position_stop",
                        )
                    )
            elif quality.ok:
                mode = self.clock.mode_at(bar.timestamp)
                target_trade: TargetTrade | None = None
                requested_notional_usd = 0.0
                adjusted_notional_usd = 0.0
                risk_reason = "no trade"
                primary_signal = "unknown"
                supporting_signals: tuple[str, ...] = ()
                conflicting_signals: tuple[str, ...] = ()

                if hasattr(self.strategy, "generate_decision"):
                    strategy_decision = self.strategy.generate_decision(
                        closes_so_far,
                        current_notional_usd=position_units * quote.mid,
                        holding_period=holding_period,
                        quote=quote,
                    )
                    if strategy_decision.is_trade_intent:
                        primary_signal = strategy_decision.primary_signal
                        supporting_signals = strategy_decision.supporting_signals
                        conflicting_signals = strategy_decision.conflicting_signals
                        requested_notional_usd = abs(strategy_decision.target_notional_usd)
                        risk_reason = strategy_decision.reason
                        if requested_notional_usd <= EPSILON_NOTIONAL:
                            decision = risk_engine.evaluate_exit(
                                account=account,
                                portfolio=portfolio,
                                symbol=symbol,
                                mode=mode,
                                reason=strategy_decision.reason,
                            )
                            risk_reason = decision.reason
                            if decision.approved:
                                adjusted_notional_usd = decision.adjusted_notional_usd
                                target_trade = _target_position_trade(
                                    target_notional_usd=0.0,
                                    quote=quote,
                                    current_units=position_units,
                                    fill_model=self.fill_model,
                                )
                        else:
                            request = strategy_decision.to_trade_request()
                            if request is not None:
                                decision = risk_engine.evaluate(
                                    account=account,
                                    portfolio=portfolio,
                                    request=request,
                                    mode=mode,
                                )
                                risk_reason = decision.reason
                                if decision.approved:
                                    adjusted_notional_usd = decision.adjusted_notional_usd
                                    target_trade = _target_position_trade(
                                        target_notional_usd=_signed_notional(
                                            strategy_decision.target_notional_usd,
                                            decision.adjusted_notional_usd,
                                        ),
                                        quote=quote,
                                        current_units=position_units,
                                        fill_model=self.fill_model,
                                    )
                else:
                    request = self.strategy.generate_request(closes_so_far)
                    if request is not None:
                        primary_signal = "request"
                        decision = risk_engine.evaluate(
                            account=account,
                            portfolio=portfolio,
                            request=request,
                            mode=mode,
                        )
                        risk_reason = decision.reason
                        requested_notional_usd = request.target_notional_usd
                        if decision.approved:
                            adjusted_notional_usd = decision.adjusted_notional_usd
                            target_trade = _target_trade(
                                request_side=request.side,
                                notional_usd=decision.adjusted_notional_usd,
                                quote=quote,
                                current_units=position_units,
                                fill_model=self.fill_model,
                            )

                if target_trade is not None:
                    cash -= target_trade.trade_units * target_trade.fill_price
                    position_units = target_trade.target_units
                    cost_basis = cost_basis.apply_fill(
                        trade_units=target_trade.trade_units,
                        fill_price=target_trade.fill_price,
                    )
                    trade_executed = True
                    fills.append(
                        BacktestFill(
                            timestamp=bar.timestamp.isoformat(timespec="seconds"),
                            symbol=symbol,
                            side=target_trade.side,
                            fill_price=target_trade.fill_price,
                            trade_units=target_trade.trade_units,
                            requested_notional_usd=requested_notional_usd,
                            adjusted_notional_usd=adjusted_notional_usd,
                            risk_reason=risk_reason,
                            primary_signal=primary_signal,
                            supporting_signals=supporting_signals,
                            conflicting_signals=conflicting_signals,
                        )
                    )

            direction_after = _unit_direction(position_units)
            if direction_after == 0:
                holding_period = 0
            elif trade_executed and direction_after != direction_before:
                holding_period = 1
            elif direction_after != 0:
                holding_period += 1

            equity_after = cash + position_units * quote.mid
            peak_equity = max(peak_equity, equity_after)
            equity_points.append(
                EquityPoint(
                    timestamp=bar.timestamp.isoformat(timespec="seconds"),
                    close=bar.close,
                    equity=equity_after,
                    cash=cash,
                    position_units=position_units,
                    position_notional_usd=position_units * quote.mid,
                    drawdown_pct=max(0.0, 1.0 - (equity_after / peak_equity)),
                )
            )

        metrics = summarize_performance(
            equity_curve=[point.equity for point in equity_points],
            turnover_notional=sum(fill.turnover_notional_usd for fill in fills),
            periods_per_year=self.periods_per_year,
        )
        pnl_ledger = build_pnl_ledger(fills, final_mark_price=final_mark_price)
        return BacktestResult(
            symbol=symbol,
            equity_curve=tuple(equity_points),
            fills=tuple(fills),
            metrics=metrics,
            pnl_ledger=pnl_ledger,
        )


def write_equity_curve_csv(result: BacktestResult, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "timestamp",
                "close",
                "equity",
                "cash",
                "position_units",
                "position_notional_usd",
                "drawdown_pct",
            ],
        )
        writer.writeheader()
        for point in result.equity_curve:
            writer.writerow(
                {
                    "timestamp": point.timestamp,
                    "close": point.close,
                    "equity": point.equity,
                    "cash": point.cash,
                    "position_units": point.position_units,
                    "position_notional_usd": point.position_notional_usd,
                    "drawdown_pct": point.drawdown_pct,
                }
            )


def _target_trade(
    *,
    request_side: Side,
    notional_usd: float,
    quote: QuoteSnapshot,
    current_units: float,
    fill_model: FillModel,
) -> TargetTrade | None:
    direction = _side_direction(request_side)
    return _target_position_trade(
        target_notional_usd=direction * notional_usd,
        quote=quote,
        current_units=current_units,
        fill_model=fill_model,
    )


def _target_position_trade(
    *,
    target_notional_usd: float,
    quote: QuoteSnapshot,
    current_units: float,
    fill_model: FillModel,
) -> TargetTrade | None:
    estimated_target_units = target_notional_usd / quote.mid
    estimated_trade_units = estimated_target_units - current_units
    if abs(estimated_trade_units) <= EPSILON_UNITS:
        return None

    actual_side = _trade_side(estimated_trade_units)
    fill_price = fill_model.fill_price(side=actual_side, quote=quote)
    target_units = target_notional_usd / fill_price
    trade_units = target_units - current_units
    if abs(trade_units) <= EPSILON_UNITS:
        return None

    actual_side = _trade_side(trade_units)
    corrected_fill_price = fill_model.fill_price(side=actual_side, quote=quote)
    if corrected_fill_price != fill_price:
        fill_price = corrected_fill_price
        target_units = target_notional_usd / fill_price
        trade_units = target_units - current_units
        if abs(trade_units) <= EPSILON_UNITS:
            return None
        actual_side = _trade_side(trade_units)

    return TargetTrade(
        side=actual_side,
        fill_price=fill_price,
        target_units=target_units,
        trade_units=trade_units,
    )


def _side_direction(side: Side) -> int:
    if side == Side.SELL:
        return -1
    return 1


def _trade_side(trade_units: float) -> Side:
    if trade_units < 0:
        return Side.SELL
    return Side.BUY


def _signed_notional(original_target_notional_usd: float, adjusted_abs_notional_usd: float) -> float:
    if original_target_notional_usd < 0:
        return -adjusted_abs_notional_usd
    return adjusted_abs_notional_usd


def _unit_direction(units: float) -> int:
    if units > EPSILON_UNITS:
        return 1
    if units < -EPSILON_UNITS:
        return -1
    return 0
