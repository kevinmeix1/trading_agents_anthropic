from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


EPSILON_UNITS = 1e-12


@dataclass(frozen=True)
class PositionCostBasis:
    position_units: float = 0.0
    average_entry_price: float | None = None

    def __post_init__(self) -> None:
        if not isfinite(self.position_units):
            raise ValueError("position_units must be finite")
        if self.average_entry_price is not None:
            if not isfinite(self.average_entry_price) or self.average_entry_price <= 0:
                raise ValueError("average_entry_price must be positive and finite")
        if abs(self.position_units) <= EPSILON_UNITS and self.average_entry_price is not None:
            raise ValueError("flat cost basis cannot have an average entry price")
        if abs(self.position_units) > EPSILON_UNITS and self.average_entry_price is None:
            raise ValueError("open cost basis requires an average entry price")

    @property
    def is_open(self) -> bool:
        return abs(self.position_units) > EPSILON_UNITS

    def apply_fill(self, *, trade_units: float, fill_price: float) -> PositionCostBasis:
        if not isfinite(trade_units):
            raise ValueError("trade_units must be finite")
        if not isfinite(fill_price) or fill_price <= 0:
            raise ValueError("fill_price must be positive and finite")
        if abs(trade_units) <= EPSILON_UNITS:
            return self

        position_units = self.position_units
        average_entry_price = self.average_entry_price

        if abs(position_units) <= EPSILON_UNITS:
            position_units = trade_units
            average_entry_price = fill_price if abs(position_units) > EPSILON_UNITS else None
        elif _same_direction(position_units, trade_units):
            average_entry_price = _weighted_average_price(
                current_units=position_units,
                current_average_price=average_entry_price,
                trade_units=trade_units,
                trade_price=fill_price,
            )
            position_units += trade_units
        else:
            close_units = min(abs(position_units), abs(trade_units))
            position_units += trade_units
            if abs(position_units) <= EPSILON_UNITS:
                position_units = 0.0
                average_entry_price = None
            elif abs(trade_units) > close_units + EPSILON_UNITS:
                average_entry_price = fill_price

        return PositionCostBasis(
            position_units=position_units,
            average_entry_price=average_entry_price,
        )

    def open_pnl_usd(self, *, mark_price: float) -> float:
        if not isfinite(mark_price) or mark_price <= 0:
            raise ValueError("mark_price must be positive and finite")
        if not self.is_open or self.average_entry_price is None:
            return 0.0
        return (mark_price - self.average_entry_price) * self.position_units

    def entry_notional_usd(self) -> float:
        if not self.is_open or self.average_entry_price is None:
            return 0.0
        return abs(self.position_units) * self.average_entry_price

    def loss_pct(self, *, mark_price: float) -> float:
        entry_notional = self.entry_notional_usd()
        if entry_notional <= 0:
            return 0.0
        open_pnl = self.open_pnl_usd(mark_price=mark_price)
        return max(0.0, -open_pnl / entry_notional)


@dataclass(frozen=True)
class PositionStopDecision:
    triggered: bool
    reason: str
    open_pnl_usd: float
    loss_pct: float
    max_loss_pct: float


def evaluate_position_stop(
    *,
    symbol: str,
    cost_basis: PositionCostBasis,
    mark_price: float,
    max_position_loss_pct: float,
) -> PositionStopDecision:
    if not symbol:
        raise ValueError("symbol is required")
    if not isfinite(max_position_loss_pct) or max_position_loss_pct < 0:
        raise ValueError("max_position_loss_pct must be finite and non-negative")

    open_pnl = cost_basis.open_pnl_usd(mark_price=mark_price)
    loss_pct = cost_basis.loss_pct(mark_price=mark_price)
    if (
        cost_basis.is_open
        and max_position_loss_pct > 0
        and loss_pct >= max_position_loss_pct
    ):
        return PositionStopDecision(
            triggered=True,
            reason=(
                f"position stop-loss reached for {symbol}: "
                f"loss={loss_pct:.3%}, limit={max_position_loss_pct:.3%}, "
                f"open_pnl=${open_pnl:.2f}"
            ),
            open_pnl_usd=open_pnl,
            loss_pct=loss_pct,
            max_loss_pct=max_position_loss_pct,
        )

    return PositionStopDecision(
        triggered=False,
        reason="position stop not triggered",
        open_pnl_usd=open_pnl,
        loss_pct=loss_pct,
        max_loss_pct=max_position_loss_pct,
    )


def _same_direction(left_units: float, right_units: float) -> bool:
    return (left_units > 0 and right_units > 0) or (left_units < 0 and right_units < 0)


def _weighted_average_price(
    *,
    current_units: float,
    current_average_price: float | None,
    trade_units: float,
    trade_price: float,
) -> float:
    current_price = current_average_price or trade_price
    total_units = abs(current_units) + abs(trade_units)
    return (
        (current_price * abs(current_units)) + (trade_price * abs(trade_units))
    ) / total_units
