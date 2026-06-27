from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from pathlib import Path


@dataclass(frozen=True)
class PriceBar:
    timestamp: datetime
    symbol: str
    close: float

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError("price bar timestamp must include a timezone")
        if not self.symbol:
            raise ValueError("price bar symbol is required")
        if self.close <= 0 or not isfinite(self.close):
            raise ValueError("price bar close must be a positive finite number")


@dataclass(frozen=True)
class PriceHistory:
    bars: tuple[PriceBar, ...]

    def symbols(self) -> list[str]:
        return sorted({bar.symbol for bar in self.bars})

    def for_symbol(self, symbol: str) -> PriceHistory:
        return PriceHistory(tuple(bar for bar in self.bars if bar.symbol == symbol))

    def close_prices(self, *, symbol: str | None = None, limit: int | None = None) -> list[float]:
        history = self.for_symbol(symbol) if symbol is not None else self
        closes = [bar.close for bar in history.bars]
        if limit is not None:
            closes = closes[-limit:]
        return closes

    def latest_bar(self, symbol: str) -> PriceBar | None:
        history = self.for_symbol(symbol)
        if not history.bars:
            return None
        return history.bars[-1]


@dataclass(frozen=True)
class QuoteSnapshot:
    timestamp: datetime
    symbol: str
    bid: float
    ask: float

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError("quote timestamp must include a timezone")
        if not self.symbol:
            raise ValueError("quote symbol is required")
        if self.bid <= 0 or self.ask <= 0 or not isfinite(self.bid) or not isfinite(self.ask):
            raise ValueError("quote bid/ask must be positive finite numbers")
        if self.ask < self.bid:
            raise ValueError("quote ask must be greater than or equal to bid")

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def spread_bps(self) -> float:
        return (self.spread / self.mid) * 10_000


@dataclass(frozen=True)
class QuoteHistory:
    quotes: tuple[QuoteSnapshot, ...]

    def symbols(self) -> list[str]:
        return sorted({quote.symbol for quote in self.quotes})

    def for_symbol(self, symbol: str) -> QuoteHistory:
        return QuoteHistory(tuple(quote for quote in self.quotes if quote.symbol == symbol))

    def latest_quote(self, symbol: str) -> QuoteSnapshot | None:
        history = self.for_symbol(symbol)
        if not history.quotes:
            return None
        return history.quotes[-1]


def load_price_history(path: str | Path) -> PriceHistory:
    csv_path = Path(path)
    bars: list[PriceBar] = []

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _validate_columns(reader.fieldnames, {"timestamp", "symbol", "close"}, "price CSV")
        for row_number, row in enumerate(reader, start=2):
            bars.append(_parse_row(row, row_number))

    bars.sort(key=lambda bar: (bar.symbol, bar.timestamp))
    return PriceHistory(tuple(bars))


def load_quote_history(path: str | Path) -> QuoteHistory:
    csv_path = Path(path)
    quotes: list[QuoteSnapshot] = []

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _validate_columns(reader.fieldnames, {"timestamp", "symbol", "bid", "ask"}, "quote CSV")
        for row_number, row in enumerate(reader, start=2):
            quotes.append(_parse_quote_row(row, row_number))

    quotes.sort(key=lambda quote: (quote.symbol, quote.timestamp))
    return QuoteHistory(tuple(quotes))


def _validate_columns(fieldnames: list[str] | None, required: set[str], label: str) -> None:
    found = set(fieldnames or [])
    missing = required - found
    if missing:
        raise ValueError(f"{label} missing required columns: {sorted(missing)}")


def _parse_row(row: dict[str, str], row_number: int) -> PriceBar:
    try:
        timestamp = datetime.fromisoformat(row["timestamp"])
        symbol = row["symbol"].strip()
        close = float(row["close"])
    except KeyError as exc:
        raise ValueError(f"row {row_number} is missing column {exc}") from exc
    except ValueError as exc:
        raise ValueError(f"row {row_number} has invalid data: {exc}") from exc

    return PriceBar(timestamp=timestamp, symbol=symbol, close=close)


def _parse_quote_row(row: dict[str, str], row_number: int) -> QuoteSnapshot:
    try:
        timestamp = datetime.fromisoformat(row["timestamp"])
        symbol = row["symbol"].strip()
        bid = float(row["bid"])
        ask = float(row["ask"])
    except KeyError as exc:
        raise ValueError(f"row {row_number} is missing column {exc}") from exc
    except ValueError as exc:
        raise ValueError(f"row {row_number} has invalid data: {exc}") from exc

    return QuoteSnapshot(timestamp=timestamp, symbol=symbol, bid=bid, ask=ask)
