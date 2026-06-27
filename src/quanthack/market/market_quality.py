from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from quanthack.market.market_data import QuoteSnapshot


@dataclass(frozen=True)
class MarketQualityLimits:
    max_spread_bps: float = 10.0
    max_quote_age_seconds: float = 5.0


@dataclass(frozen=True)
class MarketQualityDecision:
    ok: bool
    reason: str
    spread_bps: float
    quote_age_seconds: float


class MarketQualityChecker:
    def __init__(self, limits: MarketQualityLimits | None = None) -> None:
        self.limits = limits or MarketQualityLimits()

    def evaluate(self, *, quote: QuoteSnapshot, as_of: datetime) -> MarketQualityDecision:
        if as_of.tzinfo is None:
            raise ValueError("market quality check requires a timezone-aware as_of datetime")

        quote_age_seconds = (as_of - quote.timestamp).total_seconds()
        if quote_age_seconds < 0:
            return self._block(
                quote=quote,
                quote_age_seconds=quote_age_seconds,
                reason="quote timestamp is after as_of time",
            )

        if quote_age_seconds > self.limits.max_quote_age_seconds:
            return self._block(
                quote=quote,
                quote_age_seconds=quote_age_seconds,
                reason=(
                    f"quote is stale: {quote_age_seconds:.1f}s old "
                    f"> {self.limits.max_quote_age_seconds:.1f}s limit"
                ),
            )

        if quote.spread_bps > self.limits.max_spread_bps:
            return self._block(
                quote=quote,
                quote_age_seconds=quote_age_seconds,
                reason=(
                    f"spread too wide: {quote.spread_bps:.2f} bps "
                    f"> {self.limits.max_spread_bps:.2f} bps limit"
                ),
            )

        return MarketQualityDecision(
            ok=True,
            reason="market quality ok",
            spread_bps=quote.spread_bps,
            quote_age_seconds=quote_age_seconds,
        )

    @staticmethod
    def _block(
        *,
        quote: QuoteSnapshot,
        quote_age_seconds: float,
        reason: str,
    ) -> MarketQualityDecision:
        return MarketQualityDecision(
            ok=False,
            reason=reason,
            spread_bps=quote.spread_bps,
            quote_age_seconds=quote_age_seconds,
        )
