from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class AssetClass(StrEnum):
    FOREX = "FOREX"
    METAL = "METAL"
    CRYPTO = "CRYPTO"


@dataclass(frozen=True)
class InstrumentSpec:
    symbol: str
    display_symbol: str
    asset_class: AssetClass
    base_currency: str
    quote_currency: str
    min_trade_notional_usd: float
    max_spread_bps: float
    typical_slippage_bps: float
    max_leverage: float = 30.0
    enabled_for_backtest: bool = True


DEFAULT_INSTRUMENTS: tuple[InstrumentSpec, ...] = (
    InstrumentSpec("AUDUSD", "AUD/USD", AssetClass.FOREX, "AUD", "USD", 1_000, 12.0, 1.0),
    InstrumentSpec("EURCHF", "EUR/CHF", AssetClass.FOREX, "EUR", "CHF", 1_000, 14.0, 1.2),
    InstrumentSpec("EURGBP", "EUR/GBP", AssetClass.FOREX, "EUR", "GBP", 1_000, 14.0, 1.2),
    InstrumentSpec("EURUSD", "EUR/USD", AssetClass.FOREX, "EUR", "USD", 1_000, 10.0, 1.0),
    InstrumentSpec("GBPUSD", "GBP/USD", AssetClass.FOREX, "GBP", "USD", 1_000, 12.0, 1.0),
    InstrumentSpec("USDCAD", "USD/CAD", AssetClass.FOREX, "USD", "CAD", 1_000, 14.0, 1.2),
    InstrumentSpec("USDCHF", "USD/CHF", AssetClass.FOREX, "USD", "CHF", 1_000, 14.0, 1.2),
    InstrumentSpec("USDJPY", "USD/JPY", AssetClass.FOREX, "USD", "JPY", 1_000, 12.0, 1.0),
    InstrumentSpec("XAGUSD", "XAG/USD", AssetClass.METAL, "XAG", "USD", 1_000, 35.0, 2.0),
    InstrumentSpec("XAUUSD", "XAU/USD", AssetClass.METAL, "XAU", "USD", 1_000, 25.0, 2.0),
    InstrumentSpec("BARUSD", "BAR/USD", AssetClass.CRYPTO, "BAR", "USD", 500, 120.0, 5.0),
    InstrumentSpec("BTCUSD", "BTC/USD", AssetClass.CRYPTO, "BTC", "USD", 1_000, 60.0, 3.0),
    InstrumentSpec("ETHUSD", "ETH/USD", AssetClass.CRYPTO, "ETH", "USD", 1_000, 70.0, 3.0),
    InstrumentSpec("SOLUSD", "SOL/USD", AssetClass.CRYPTO, "SOL", "USD", 500, 100.0, 4.0),
    InstrumentSpec("XRPUSD", "XRP/USD", AssetClass.CRYPTO, "XRP", "USD", 500, 120.0, 5.0),
)


def normalize_symbol(symbol: str) -> str:
    return symbol.replace("/", "").replace("-", "").replace("_", "").strip().upper()


def instrument_for(symbol: str) -> InstrumentSpec:
    normalized = normalize_symbol(symbol)
    for instrument in DEFAULT_INSTRUMENTS:
        if instrument.symbol == normalized:
            return instrument
    valid = ", ".join(instrument.display_symbol for instrument in DEFAULT_INSTRUMENTS)
    raise KeyError(f"unknown instrument {symbol!r}; expected one of: {valid}")


def instruments_by_asset_class(asset_class: AssetClass) -> tuple[InstrumentSpec, ...]:
    return tuple(
        instrument
        for instrument in DEFAULT_INSTRUMENTS
        if instrument.asset_class == asset_class
    )


def enabled_symbols() -> tuple[str, ...]:
    return tuple(
        instrument.symbol
        for instrument in DEFAULT_INSTRUMENTS
        if instrument.enabled_for_backtest
    )

