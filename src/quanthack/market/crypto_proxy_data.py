from __future__ import annotations

import csv
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

from quanthack.core.instruments import instrument_for


BINANCE_SPOT_BASE_URL = "https://api.binance.com"
DEFAULT_PROXY_SYMBOL_MAP: Mapping[str, str] = {
    "BARUSD": "BARUSDT",
    "BTCUSD": "BTCUSDT",
    "ETHUSD": "ETHUSDT",
    "SOLUSD": "SOLUSDT",
    "XRPUSD": "XRPUSDT",
}
PRICE_FIELDS = ("timestamp", "symbol", "close")
QUOTE_FIELDS = ("timestamp", "symbol", "bid", "ask")


@dataclass(frozen=True)
class CryptoProxyBar:
    timestamp: datetime
    symbol: str
    source_symbol: str
    close: float


@dataclass(frozen=True)
class CryptoProxyFetchSummary:
    source: str
    symbols: tuple[str, ...]
    source_symbols: tuple[str, ...]
    bars_written: int
    start: datetime
    end: datetime
    interval: str
    price_csv: str
    quote_csv: str
    research_only: bool = True


def fetch_crypto_proxy_to_csv(
    *,
    symbols: tuple[str, ...],
    price_output: str | Path,
    quote_output: str | Path,
    start: datetime,
    end: datetime,
    interval: str = "15m",
    source_symbol_map: Mapping[str, str] | None = None,
    base_url: str = BINANCE_SPOT_BASE_URL,
    limit: int = 1000,
    timeout_seconds: float = 20.0,
    request_json: Callable[[str, Mapping[str, object], float], object] | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> CryptoProxyFetchSummary:
    if not symbols:
        raise ValueError("at least one crypto symbol is required")
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start and end must be timezone-aware datetimes")
    if end <= start:
        raise ValueError("end must be after start")
    if limit < 1 or limit > 1000:
        raise ValueError("limit must be in [1, 1000]")

    canonical_symbols = tuple(instrument_for(symbol).symbol for symbol in symbols)
    symbol_map = dict(DEFAULT_PROXY_SYMBOL_MAP)
    if source_symbol_map is not None:
        symbol_map.update(
            {instrument_for(symbol).symbol: source for symbol, source in source_symbol_map.items()}
        )

    requester = request_json or _request_json
    bars: list[CryptoProxyBar] = []
    for symbol in canonical_symbols:
        source_symbol = symbol_map.get(symbol)
        if source_symbol is None:
            raise ValueError(f"no proxy source symbol configured for {symbol}")
        symbol_bars = _fetch_symbol_klines(
            symbol=symbol,
            source_symbol=source_symbol,
            start=start,
            end=end,
            interval=interval,
            base_url=base_url,
            limit=limit,
            timeout_seconds=timeout_seconds,
            requester=requester,
        )
        bars.extend(symbol_bars)
        if progress_callback is not None:
            progress_callback(
                f"fetched {len(symbol_bars):,} proxy bars for {symbol} from {source_symbol}"
            )

    if not bars:
        raise ValueError("no crypto proxy bars returned")

    ordered = sorted(bars, key=lambda bar: (bar.symbol, bar.timestamp))
    price_path = Path(price_output)
    quote_path = Path(quote_output)
    price_path.parent.mkdir(parents=True, exist_ok=True)
    quote_path.parent.mkdir(parents=True, exist_ok=True)
    with price_path.open("w", encoding="utf-8", newline="") as price_handle:
        writer = csv.DictWriter(price_handle, fieldnames=PRICE_FIELDS)
        writer.writeheader()
        for bar in ordered:
            writer.writerow(
                {
                    "timestamp": bar.timestamp.isoformat(timespec="seconds"),
                    "symbol": bar.symbol,
                    "close": f"{bar.close:.10f}",
                }
            )

    with quote_path.open("w", encoding="utf-8", newline="") as quote_handle:
        writer = csv.DictWriter(quote_handle, fieldnames=QUOTE_FIELDS)
        writer.writeheader()
        for bar in ordered:
            bid, ask = _synthetic_quote(bar.symbol, bar.close)
            writer.writerow(
                {
                    "timestamp": bar.timestamp.isoformat(timespec="seconds"),
                    "symbol": bar.symbol,
                    "bid": f"{bid:.10f}",
                    "ask": f"{ask:.10f}",
                }
            )

    return CryptoProxyFetchSummary(
        source="binance_spot_usdt_proxy",
        symbols=tuple(sorted({bar.symbol for bar in ordered})),
        source_symbols=tuple(sorted({bar.source_symbol for bar in ordered})),
        bars_written=len(ordered),
        start=start.astimezone(UTC),
        end=end.astimezone(UTC),
        interval=interval,
        price_csv=str(price_path),
        quote_csv=str(quote_path),
    )


def _fetch_symbol_klines(
    *,
    symbol: str,
    source_symbol: str,
    start: datetime,
    end: datetime,
    interval: str,
    base_url: str,
    limit: int,
    timeout_seconds: float,
    requester: Callable[[str, Mapping[str, object], float], object],
) -> tuple[CryptoProxyBar, ...]:
    bars: list[CryptoProxyBar] = []
    current_start_ms = _to_millis(start)
    end_ms = _to_millis(end)
    last_open_ms: int | None = None
    while current_start_ms < end_ms:
        payload = requester(
            f"{base_url.rstrip('/')}/api/v3/klines",
            {
                "symbol": source_symbol,
                "interval": interval,
                "startTime": current_start_ms,
                "endTime": end_ms,
                "limit": limit,
            },
            timeout_seconds,
        )
        if not isinstance(payload, list) or not payload:
            break
        for raw_bar in payload:
            open_ms = int(raw_bar[0])
            if last_open_ms is not None and open_ms <= last_open_ms:
                continue
            if open_ms >= end_ms:
                continue
            close = float(raw_bar[4])
            if close <= 0:
                continue
            bars.append(
                CryptoProxyBar(
                    timestamp=datetime.fromtimestamp(open_ms / 1000, tz=UTC),
                    symbol=symbol,
                    source_symbol=source_symbol,
                    close=close,
                )
            )
            last_open_ms = open_ms
        if last_open_ms is None:
            break
        next_start_ms = last_open_ms + 1
        if next_start_ms <= current_start_ms:
            break
        current_start_ms = next_start_ms
    return tuple(bars)


def _request_json(url: str, params: Mapping[str, object], timeout_seconds: float) -> object:
    query = urlencode(params)
    with urlopen(f"{url}?{query}", timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def _synthetic_quote(symbol: str, close: float) -> tuple[float, float]:
    spread_bps = instrument_for(symbol).typical_slippage_bps
    half_spread = close * (spread_bps / 20_000)
    return close - half_spread, close + half_spread


def _to_millis(value: datetime) -> int:
    return int(value.astimezone(UTC).timestamp() * 1000)


def default_crypto_window(*, days: int = 30, end: datetime | None = None) -> tuple[datetime, datetime]:
    if days < 1:
        raise ValueError("days must be at least 1")
    end_time = end.astimezone(UTC) if end is not None else datetime.now(tz=UTC)
    return end_time - timedelta(days=days), end_time
