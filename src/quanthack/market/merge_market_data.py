from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from quanthack.market.market_data import (
    PriceBar,
    PriceHistory,
    QuoteHistory,
    QuoteSnapshot,
    load_price_history,
    load_quote_history,
)


@dataclass(frozen=True)
class MarketDataMergeSummary:
    price_rows: int
    quote_rows: int
    symbols: tuple[str, ...]
    start: datetime
    end: datetime
    price_csv: str
    quote_csv: str


def merge_market_data_csvs(
    *,
    price_inputs: tuple[str | Path, ...],
    quote_inputs: tuple[str | Path, ...],
    price_output: str | Path,
    quote_output: str | Path,
    symbols: tuple[str, ...] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    common_window: bool = False,
) -> MarketDataMergeSummary:
    if not price_inputs:
        raise ValueError("at least one price input is required")
    if not quote_inputs:
        raise ValueError("at least one quote input is required")

    prices = _merge_prices(tuple(load_price_history(path) for path in price_inputs))
    quotes = _merge_quotes(tuple(load_quote_history(path) for path in quote_inputs))
    selected_symbols = tuple(sorted(symbols or (set(prices.symbols()) & set(quotes.symbols()))))
    if not selected_symbols:
        raise ValueError("no symbols selected")

    if common_window:
        common_start, common_end = _common_window(prices, quotes, selected_symbols)
        start = common_start if start is None else max(start, common_start)
        end = common_end if end is None else min(end, common_end)
    if start is not None and end is not None and end < start:
        raise ValueError("end must be greater than or equal to start")

    filtered_prices = tuple(
        bar
        for bar in prices.bars
        if bar.symbol in selected_symbols and _inside_window(bar.timestamp, start=start, end=end)
    )
    filtered_quotes = tuple(
        quote
        for quote in quotes.quotes
        if quote.symbol in selected_symbols
        and _inside_window(quote.timestamp, start=start, end=end)
    )
    if not filtered_prices:
        raise ValueError("merged price output would be empty")
    if not filtered_quotes:
        raise ValueError("merged quote output would be empty")

    _write_prices(filtered_prices, Path(price_output))
    _write_quotes(filtered_quotes, Path(quote_output))
    first_timestamp = min(
        min(bar.timestamp for bar in filtered_prices),
        min(quote.timestamp for quote in filtered_quotes),
    )
    last_timestamp = max(
        max(bar.timestamp for bar in filtered_prices),
        max(quote.timestamp for quote in filtered_quotes),
    )
    return MarketDataMergeSummary(
        price_rows=len(filtered_prices),
        quote_rows=len(filtered_quotes),
        symbols=selected_symbols,
        start=first_timestamp,
        end=last_timestamp,
        price_csv=str(price_output),
        quote_csv=str(quote_output),
    )


def _merge_prices(histories: tuple[PriceHistory, ...]) -> PriceHistory:
    by_key: dict[tuple[str, datetime], PriceBar] = {}
    for history in histories:
        for bar in history.bars:
            by_key[(bar.symbol, bar.timestamp)] = bar
    return PriceHistory(tuple(sorted(by_key.values(), key=lambda bar: (bar.symbol, bar.timestamp))))


def _merge_quotes(histories: tuple[QuoteHistory, ...]) -> QuoteHistory:
    by_key: dict[tuple[str, datetime], QuoteSnapshot] = {}
    for history in histories:
        for quote in history.quotes:
            by_key[(quote.symbol, quote.timestamp)] = quote
    return QuoteHistory(
        tuple(sorted(by_key.values(), key=lambda quote: (quote.symbol, quote.timestamp)))
    )


def _common_window(
    prices: PriceHistory,
    quotes: QuoteHistory,
    symbols: tuple[str, ...],
) -> tuple[datetime, datetime]:
    starts: list[datetime] = []
    ends: list[datetime] = []
    for symbol in symbols:
        symbol_prices = prices.for_symbol(symbol).bars
        symbol_quotes = quotes.for_symbol(symbol).quotes
        if not symbol_prices:
            raise ValueError(f"no price rows for {symbol}")
        if not symbol_quotes:
            raise ValueError(f"no quote rows for {symbol}")
        starts.append(max(symbol_prices[0].timestamp, symbol_quotes[0].timestamp))
        ends.append(min(symbol_prices[-1].timestamp, symbol_quotes[-1].timestamp))
    start = max(starts)
    end = min(ends)
    if end < start:
        raise ValueError("selected symbols have no overlapping timestamp window")
    return start, end


def _inside_window(
    timestamp: datetime,
    *,
    start: datetime | None,
    end: datetime | None,
) -> bool:
    if start is not None and timestamp < start:
        return False
    if end is not None and timestamp > end:
        return False
    return True


def _write_prices(bars: tuple[PriceBar, ...], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("timestamp", "symbol", "close"))
        writer.writeheader()
        for bar in sorted(bars, key=lambda item: (item.symbol, item.timestamp)):
            writer.writerow(
                {
                    "timestamp": bar.timestamp.isoformat(timespec="seconds"),
                    "symbol": bar.symbol,
                    "close": f"{bar.close:.10f}",
                }
            )


def _write_quotes(quotes: tuple[QuoteSnapshot, ...], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("timestamp", "symbol", "bid", "ask"))
        writer.writeheader()
        for quote in sorted(quotes, key=lambda item: (item.symbol, item.timestamp)):
            writer.writerow(
                {
                    "timestamp": quote.timestamp.isoformat(timespec="seconds"),
                    "symbol": quote.symbol,
                    "bid": f"{quote.bid:.10f}",
                    "ask": f"{quote.ask:.10f}",
                }
            )
