from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from math import isfinite

from quanthack.core.clock import CompetitionMode


class Side(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class RiskState(StrEnum):
    NORMAL = "NORMAL"
    REDUCE_ONLY = "REDUCE_ONLY"
    FROZEN = "FROZEN"


@dataclass(frozen=True)
class AccountSnapshot:
    equity: float
    starting_equity: float = 1_000_000.0
    day_start_equity: float = 1_000_000.0
    peak_equity: float = 1_000_000.0
    margin_level_pct: float | None = None

    def __post_init__(self) -> None:
        values = [self.equity, self.starting_equity, self.day_start_equity, self.peak_equity]
        if any(not isfinite(value) or value <= 0 for value in values):
            raise ValueError("equity values must be positive finite numbers")

        if self.margin_level_pct is not None and self.margin_level_pct <= 0:
            raise ValueError("margin_level_pct must be positive when provided")

    @property
    def total_pnl_pct(self) -> float:
        return (self.equity / self.starting_equity) - 1.0

    @property
    def daily_pnl_pct(self) -> float:
        return (self.equity / self.day_start_equity) - 1.0

    @property
    def drawdown_pct(self) -> float:
        return max(0.0, 1.0 - (self.equity / self.peak_equity))


@dataclass(frozen=True)
class Position:
    symbol: str
    notional_usd: float


@dataclass(frozen=True)
class PortfolioSnapshot:
    positions: tuple[Position, ...] = field(default_factory=tuple)

    @property
    def gross_notional_usd(self) -> float:
        return sum(abs(position.notional_usd) for position in self.positions)

    def notional_for_symbol(self, symbol: str) -> float:
        return sum(position.notional_usd for position in self.positions if position.symbol == symbol)

    def gross_leverage(self, account: AccountSnapshot) -> float:
        return self.gross_notional_usd / account.equity

    def gross_leverage_after(
        self,
        *,
        account: AccountSnapshot,
        symbol: str,
        target_abs_notional_usd: float,
    ) -> float:
        current_symbol_notional = abs(self.notional_for_symbol(symbol))
        adjusted_gross = self.gross_notional_usd - current_symbol_notional + target_abs_notional_usd
        return adjusted_gross / account.equity


@dataclass(frozen=True)
class TradeRequest:
    symbol: str
    side: Side
    target_notional_usd: float
    reason: str

    def __post_init__(self) -> None:
        if self.target_notional_usd <= 0 or not isfinite(self.target_notional_usd):
            raise ValueError("target_notional_usd must be a positive finite number")


# The competition force-liquidates when MT-style margin level reaches 30%.
# Keep internal floors comfortably above this red line.
STOP_OUT_MARGIN_LEVEL_PCT = 30.0


@dataclass(frozen=True)
class RiskLimits:
    max_gross_leverage: float = 2.0
    max_symbol_notional_pct: float = 0.25
    max_daily_loss_pct: float = 0.025
    max_drawdown_pct: float = 0.06
    checkpoint_risk_multiplier: float = 0.5
    min_margin_level_pct: float = 300.0
    max_position_loss_pct: float = 0.01
    max_forex_position_loss_pct: float | None = None
    max_metal_position_loss_pct: float | None = None
    max_crypto_position_loss_pct: float | None = None
    reduce_only_margin_level_pct: float | None = None
    drawdown_derisk_start_pct: float | None = None
    drawdown_derisk_full_pct: float | None = None
    freeze_on_daily_loss: bool = True

    def __post_init__(self) -> None:
        values = [
            self.max_gross_leverage,
            self.max_symbol_notional_pct,
            self.max_daily_loss_pct,
            self.max_drawdown_pct,
            self.checkpoint_risk_multiplier,
            self.min_margin_level_pct,
            self.max_position_loss_pct,
        ]
        if any(not isfinite(value) for value in values):
            raise ValueError("risk limits must be finite")
        optional_values = [
            self.max_forex_position_loss_pct,
            self.max_metal_position_loss_pct,
            self.max_crypto_position_loss_pct,
            self.reduce_only_margin_level_pct,
            self.drawdown_derisk_start_pct,
            self.drawdown_derisk_full_pct,
        ]
        if any(value is not None and not isfinite(value) for value in optional_values):
            raise ValueError("optional risk limits must be finite")
        if self.max_gross_leverage <= 0:
            raise ValueError("max_gross_leverage must be positive")
        if not 0 < self.max_symbol_notional_pct <= 1:
            raise ValueError("max_symbol_notional_pct must be in (0, 1]")
        if not 0 < self.max_daily_loss_pct <= 1:
            raise ValueError("max_daily_loss_pct must be in (0, 1]")
        if not 0 < self.max_drawdown_pct <= 1:
            raise ValueError("max_drawdown_pct must be in (0, 1]")
        if not 0 <= self.checkpoint_risk_multiplier <= 1:
            raise ValueError("checkpoint_risk_multiplier must be in [0, 1]")
        if self.min_margin_level_pct <= 0:
            raise ValueError("min_margin_level_pct must be positive")
        if self.max_position_loss_pct < 0:
            raise ValueError("max_position_loss_pct must be non-negative")
        for name, value in (
            ("max_forex_position_loss_pct", self.max_forex_position_loss_pct),
            ("max_metal_position_loss_pct", self.max_metal_position_loss_pct),
            ("max_crypto_position_loss_pct", self.max_crypto_position_loss_pct),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.reduce_only_margin_level_pct is not None:
            if self.reduce_only_margin_level_pct <= self.min_margin_level_pct:
                raise ValueError(
                    "reduce_only_margin_level_pct must exceed min_margin_level_pct"
                )
        start = self.drawdown_derisk_start_pct
        full = self.drawdown_derisk_full_pct
        if (start is None) != (full is None):
            raise ValueError("drawdown derisk thresholds must be set together")
        if start is not None and full is not None:
            if not (0 <= start < full <= 1):
                raise ValueError(
                    "require 0 <= drawdown_derisk_start_pct "
                    "< drawdown_derisk_full_pct <= 1"
                )

    @classmethod
    def competition_safe(
        cls,
        *,
        max_gross_leverage: float = 6.0,
        max_symbol_notional_pct: float = 0.80,
    ) -> "RiskLimits":
        """Return a tournament-oriented preset below official penalty zones."""
        return cls(
            max_gross_leverage=max_gross_leverage,
            max_symbol_notional_pct=max_symbol_notional_pct,
            max_daily_loss_pct=0.05,
            max_drawdown_pct=0.12,
            checkpoint_risk_multiplier=0.5,
            min_margin_level_pct=300.0,
            max_position_loss_pct=0.01,
            max_forex_position_loss_pct=0.01,
            max_metal_position_loss_pct=0.02,
            max_crypto_position_loss_pct=0.025,
            reduce_only_margin_level_pct=500.0,
            drawdown_derisk_start_pct=0.04,
            drawdown_derisk_full_pct=0.10,
            freeze_on_daily_loss=True,
        )

    def max_position_loss_for_symbol(self, symbol: str) -> float:
        """Return the per-trade stop for a symbol, allowing asset-class overrides."""
        from quanthack.core.instruments import AssetClass, instrument_for

        try:
            asset_class = instrument_for(symbol).asset_class
        except KeyError:
            return self.max_position_loss_pct

        if asset_class == AssetClass.FOREX:
            return (
                self.max_forex_position_loss_pct
                if self.max_forex_position_loss_pct is not None
                else self.max_position_loss_pct
            )
        if asset_class == AssetClass.METAL:
            return (
                self.max_metal_position_loss_pct
                if self.max_metal_position_loss_pct is not None
                else self.max_position_loss_pct
            )
        if asset_class == AssetClass.CRYPTO:
            return (
                self.max_crypto_position_loss_pct
                if self.max_crypto_position_loss_pct is not None
                else self.max_position_loss_pct
            )
        return self.max_position_loss_pct


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    reason: str
    adjusted_notional_usd: float
    state: RiskState


class RiskEngine:
    def __init__(self, limits: RiskLimits | None = None) -> None:
        self.limits = limits or RiskLimits()
        self.state = RiskState.NORMAL

    def evaluate(
        self,
        *,
        account: AccountSnapshot,
        portfolio: PortfolioSnapshot,
        request: TradeRequest,
        mode: CompetitionMode,
    ) -> RiskDecision:
        if self.state == RiskState.FROZEN:
            return self._block("risk engine is frozen")

        if account.margin_level_pct is not None:
            if account.margin_level_pct <= self.limits.min_margin_level_pct:
                self.state = RiskState.FROZEN
                return self._block("margin level below internal safety limit")
            if (
                self.limits.reduce_only_margin_level_pct is not None
                and account.margin_level_pct <= self.limits.reduce_only_margin_level_pct
            ):
                self.state = RiskState.REDUCE_ONLY
                return self._block("margin level in reduce-only band")

        if account.daily_pnl_pct <= -self.limits.max_daily_loss_pct:
            if self.limits.freeze_on_daily_loss:
                self.state = RiskState.FROZEN
            else:
                self.state = RiskState.REDUCE_ONLY
            return self._block("daily loss stop reached")

        if account.drawdown_pct >= self.limits.max_drawdown_pct:
            self.state = RiskState.REDUCE_ONLY
            return self._block("drawdown throttle reached")

        adjusted_notional = self._cap_symbol_notional(account, request.target_notional_usd)

        if mode == CompetitionMode.CHECKPOINT_PROTECT and account.total_pnl_pct > 0:
            adjusted_notional *= self.limits.checkpoint_risk_multiplier

        adjusted_notional *= self._drawdown_brake(account.drawdown_pct)

        leverage_after = portfolio.gross_leverage_after(
            account=account,
            symbol=request.symbol,
            target_abs_notional_usd=adjusted_notional,
        )

        if leverage_after > self.limits.max_gross_leverage:
            return self._block(
                f"gross leverage would be {leverage_after:.2f}x, "
                f"above {self.limits.max_gross_leverage:.2f}x limit"
            )

        return RiskDecision(
            approved=True,
            reason="approved",
            adjusted_notional_usd=adjusted_notional,
            state=self.state,
        )

    def evaluate_exit(
        self,
        *,
        account: AccountSnapshot,
        portfolio: PortfolioSnapshot,
        symbol: str,
        mode: CompetitionMode,
        reason: str = "allocated exit",
    ) -> RiskDecision:
        if not symbol:
            raise ValueError("symbol is required")

        state_reason = reason
        if account.margin_level_pct is not None:
            if account.margin_level_pct <= self.limits.min_margin_level_pct:
                self.state = RiskState.FROZEN
                state_reason = f"{state_reason}; margin floor breached"
            elif (
                self.limits.reduce_only_margin_level_pct is not None
                and account.margin_level_pct <= self.limits.reduce_only_margin_level_pct
            ):
                self.state = RiskState.REDUCE_ONLY
                state_reason = f"{state_reason}; margin reduce-only band active"

        if account.daily_pnl_pct <= -self.limits.max_daily_loss_pct:
            if self.limits.freeze_on_daily_loss:
                self.state = RiskState.FROZEN
            else:
                self.state = RiskState.REDUCE_ONLY
            state_reason = f"{state_reason}; daily loss stop active"
        elif account.drawdown_pct >= self.limits.max_drawdown_pct:
            self.state = RiskState.REDUCE_ONLY
            state_reason = f"{state_reason}; drawdown throttle active"

        current_notional = portfolio.notional_for_symbol(symbol)
        if abs(current_notional) <= 0:
            state_reason = f"{state_reason}; no open exposure"

        return RiskDecision(
            approved=True,
            reason=f"{state_reason}; exit approved",
            adjusted_notional_usd=0.0,
            state=self.state,
        )

    def _cap_symbol_notional(self, account: AccountSnapshot, requested: float) -> float:
        max_symbol_notional = account.equity * self.limits.max_symbol_notional_pct
        return min(requested, max_symbol_notional)

    def _drawdown_brake(self, drawdown_pct: float) -> float:
        start = self.limits.drawdown_derisk_start_pct
        full = self.limits.drawdown_derisk_full_pct
        if start is None or full is None or drawdown_pct <= start:
            return 1.0
        if drawdown_pct >= full:
            return 0.0
        return (full - drawdown_pct) / (full - start)

    def _block(self, reason: str) -> RiskDecision:
        return RiskDecision(
            approved=False,
            reason=reason,
            adjusted_notional_usd=0.0,
            state=self.state,
        )
