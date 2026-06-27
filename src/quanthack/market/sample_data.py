from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import exp, pi, sin
from pathlib import Path
from random import Random
from zoneinfo import ZoneInfo

from quanthack.core.instruments import (
    AssetClass,
    enabled_symbols,
    instrument_for,
    instruments_by_asset_class,
)
from quanthack.market.market_data import PriceBar, PriceHistory, QuoteHistory, QuoteSnapshot


DEFAULT_SAMPLE_START = datetime(2026, 6, 22, 0, 0, tzinfo=ZoneInfo("Europe/London"))

_BASE_PRICES = {
    "AUDUSD": 0.6650,
    "EURCHF": 0.9600,
    "EURGBP": 0.8450,
    "EURUSD": 1.1000,
    "GBPUSD": 1.3000,
    "USDCAD": 1.3700,
    "USDCHF": 0.9000,
    "USDJPY": 155.00,
    "XAGUSD": 29.50,
    "XAUUSD": 2320.00,
    "BARUSD": 3.50,
    "BTCUSD": 65000.00,
    "ETHUSD": 3500.00,
    "SOLUSD": 145.00,
    "XRPUSD": 0.5200,
}


@dataclass(frozen=True)
class SyntheticMarketData:
    prices: PriceHistory
    quotes: QuoteHistory


@dataclass(frozen=True)
class _SyntheticProfile:
    base_price: float
    drift_bps: float
    volatility_bps: float
    cycle_bps: float
    pulse_bps: float


def generate_synthetic_market_data(
    *,
    symbols: tuple[str, ...] | None = None,
    asset_class: AssetClass | None = None,
    start: datetime = DEFAULT_SAMPLE_START,
    periods: int = 96,
    interval_minutes: int = 15,
    seed: int = 7,
) -> SyntheticMarketData:
    if start.tzinfo is None:
        raise ValueError("sample data start must include a timezone")
    if periods < 2:
        raise ValueError("periods must be at least 2")
    if interval_minutes < 1:
        raise ValueError("interval_minutes must be at least 1")

    selected_symbols = _selected_symbols(symbols=symbols, asset_class=asset_class)
    bars: list[PriceBar] = []
    quotes: list[QuoteSnapshot] = []

    for symbol in selected_symbols:
        profile = _profile_for(symbol)
        rng = Random(seed + _symbol_seed_offset(symbol))
        price = profile.base_price
        phase_offset = (_symbol_seed_offset(symbol) % 17) / 17

        for index in range(periods):
            timestamp = start + timedelta(minutes=interval_minutes * index)
            if index > 0:
                price *= exp(
                    _synthetic_return_bps(
                        index=index,
                        profile=profile,
                        rng=rng,
                        phase_offset=phase_offset,
                    )
                    / 10_000
                )

            quote = _quote_for(
                symbol=symbol,
                timestamp=timestamp,
                mid=price,
                index=index,
                rng=rng,
            )
            bars.append(PriceBar(timestamp=timestamp, symbol=symbol, close=quote.mid))
            quotes.append(quote)

    bars.sort(key=lambda bar: (bar.symbol, bar.timestamp))
    quotes.sort(key=lambda quote: (quote.symbol, quote.timestamp))
    return SyntheticMarketData(
        prices=PriceHistory(tuple(bars)),
        quotes=QuoteHistory(tuple(quotes)),
    )


def write_price_history_csv(history: PriceHistory, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp", "symbol", "close"])
        writer.writeheader()
        for bar in history.bars:
            writer.writerow(
                {
                    "timestamp": bar.timestamp.isoformat(timespec="seconds"),
                    "symbol": bar.symbol,
                    "close": f"{bar.close:.10f}",
                }
            )


def write_quote_history_csv(history: QuoteHistory, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp", "symbol", "bid", "ask"])
        writer.writeheader()
        for quote in history.quotes:
            writer.writerow(
                {
                    "timestamp": quote.timestamp.isoformat(timespec="seconds"),
                    "symbol": quote.symbol,
                    "bid": f"{quote.bid:.10f}",
                    "ask": f"{quote.ask:.10f}",
                }
            )


def _selected_symbols(
    *,
    symbols: tuple[str, ...] | None,
    asset_class: AssetClass | None,
) -> tuple[str, ...]:
    if symbols:
        raw_symbols = symbols
    elif asset_class is not None:
        raw_symbols = tuple(
            instrument.symbol for instrument in instruments_by_asset_class(asset_class)
        )
    else:
        raw_symbols = enabled_symbols()

    selected: list[str] = []
    seen: set[str] = set()
    for raw_symbol in raw_symbols:
        symbol = instrument_for(raw_symbol).symbol
        if symbol in seen:
            continue
        selected.append(symbol)
        seen.add(symbol)
    if not selected:
        raise ValueError("at least one symbol is required")
    return tuple(selected)


def _profile_for(symbol: str) -> _SyntheticProfile:
    instrument = instrument_for(symbol)
    bias = ((_symbol_seed_offset(symbol) % 13) - 6) / 6
    base_price = _BASE_PRICES.get(instrument.symbol, 1.0)

    if instrument.asset_class == AssetClass.FOREX:
        return _SyntheticProfile(
            base_price=base_price,
            drift_bps=0.10 * bias,
            volatility_bps=1.8 + abs(bias) * 0.4,
            cycle_bps=1.6,
            pulse_bps=4.0,
        )
    if instrument.asset_class == AssetClass.METAL:
        return _SyntheticProfile(
            base_price=base_price,
            drift_bps=0.20 * bias,
            volatility_bps=4.0 + abs(bias),
            cycle_bps=3.0,
            pulse_bps=8.0,
        )
    return _SyntheticProfile(
        base_price=base_price,
        drift_bps=0.35 * bias,
        volatility_bps=10.0 + abs(bias) * 3.0,
        cycle_bps=7.0,
        pulse_bps=18.0,
    )


def _synthetic_return_bps(
    *,
    index: int,
    profile: _SyntheticProfile,
    rng: Random,
    phase_offset: float,
) -> float:
    cycle = profile.cycle_bps * sin((index / 9 + phase_offset) * 2 * pi)
    pulse = 0.0
    if index % 24 == 0:
        pulse = profile.pulse_bps if (index // 24) % 2 == 0 else -profile.pulse_bps
    return profile.drift_bps + cycle + pulse + rng.gauss(0.0, profile.volatility_bps)


def _quote_for(
    *,
    symbol: str,
    timestamp: datetime,
    mid: float,
    index: int,
    rng: Random,
) -> QuoteSnapshot:
    instrument = instrument_for(symbol)
    base_spread_bps = max(
        instrument.typical_slippage_bps * 2.0,
        instrument.max_spread_bps * 0.18,
    )
    spread_wave = 1.0 + 0.12 * sin(index / 5)
    spread_noise = 1.0 + rng.random() * 0.10
    spread_bps = min(
        instrument.max_spread_bps * 0.70,
        base_spread_bps * spread_wave * spread_noise,
    )
    spread = mid * spread_bps / 10_000
    return QuoteSnapshot(
        timestamp=timestamp,
        symbol=instrument.symbol,
        bid=mid - spread / 2,
        ask=mid + spread / 2,
    )


def _symbol_seed_offset(symbol: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(symbol))
