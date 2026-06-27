from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from importlib import import_module
from typing import Any, Protocol

from quanthack.core.clock import UTC
from quanthack.core.instruments import enabled_symbols, instrument_for
from quanthack.market.market_data import (
    PriceBar,
    PriceHistory,
    QuoteSnapshot,
    load_price_history,
    load_quote_history,
)
from quanthack.trading.risk import AccountSnapshot


class MarketDataAdapter(Protocol):
    def supported_symbols(self) -> tuple[str, ...]:
        """Return canonical symbols available through this adapter."""
        ...

    def get_latest_quote(self, symbol: str) -> QuoteSnapshot:
        """Return the latest quote for a canonical symbol."""
        ...

    def get_recent_bars(
        self,
        symbol: str,
        *,
        timeframe: str,
        count: int,
    ) -> tuple[PriceBar, ...]:
        """Return recent bars oldest-to-newest for a canonical symbol."""
        ...


class AccountAdapter(Protocol):
    def get_account_snapshot(
        self,
        *,
        starting_equity: float,
        day_start_equity: float,
        peak_equity: float,
    ) -> AccountSnapshot:
        """Return a project AccountSnapshot."""
        ...


@dataclass(frozen=True)
class CsvMarketDataAdapter:
    price_csv: str
    quote_csv: str
    _prices: PriceHistory = field(init=False, repr=False)
    _quotes_by_symbol: dict[str, QuoteSnapshot] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        prices = load_price_history(self.price_csv)
        quotes = load_quote_history(self.quote_csv)
        latest_quotes: dict[str, QuoteSnapshot] = {}
        for symbol in quotes.symbols():
            quote = quotes.latest_quote(symbol)
            if quote is not None:
                latest_quotes[instrument_for(symbol).symbol] = quote
        object.__setattr__(self, "_prices", prices)
        object.__setattr__(self, "_quotes_by_symbol", latest_quotes)

    def supported_symbols(self) -> tuple[str, ...]:
        return tuple(sorted(set(self._prices.symbols()) & set(self._quotes_by_symbol)))

    def get_latest_quote(self, symbol: str) -> QuoteSnapshot:
        canonical = instrument_for(symbol).symbol
        try:
            return self._quotes_by_symbol[canonical]
        except KeyError as exc:
            raise KeyError(f"no CSV quote for {canonical}") from exc

    def get_recent_bars(
        self,
        symbol: str,
        *,
        timeframe: str,
        count: int,
    ) -> tuple[PriceBar, ...]:
        if count < 1:
            raise ValueError("count must be at least 1")
        canonical = instrument_for(symbol).symbol
        bars = self._prices.for_symbol(canonical).bars
        if not bars:
            raise KeyError(f"no CSV bars for {canonical}")
        return bars[-count:]


@dataclass(frozen=True)
class StaticAccountAdapter:
    equity: float
    margin_level_pct: float = 2_000.0

    def get_account_snapshot(
        self,
        *,
        starting_equity: float,
        day_start_equity: float,
        peak_equity: float,
    ) -> AccountSnapshot:
        equity = self.equity
        return AccountSnapshot(
            equity=equity,
            starting_equity=starting_equity,
            day_start_equity=day_start_equity,
            peak_equity=max(peak_equity, equity),
            margin_level_pct=self.margin_level_pct,
        )


class MT5UnavailableError(RuntimeError):
    """Raised when the MetaTrader5 package or terminal connection is unavailable."""


@dataclass
class MT5ConnectionSettings:
    terminal_path: str | None = None
    login: int | None = None
    password: str | None = None
    server: str | None = None
    timeout_ms: int = 60_000
    portable: bool = False
    symbol_map: Mapping[str, str] = field(default_factory=dict)


class MT5MarketDataAdapter:
    def __init__(
        self,
        settings: MT5ConnectionSettings | None = None,
        *,
        mt5_module: Any | None = None,
    ) -> None:
        self.settings = settings or MT5ConnectionSettings()
        self._mt5 = mt5_module
        self._connected = False

    def connect(self) -> None:
        mt5 = self._module()
        kwargs: dict[str, Any] = {
            "timeout": self.settings.timeout_ms,
            "portable": self.settings.portable,
        }

        if self.settings.terminal_path:
            ok = mt5.initialize(self.settings.terminal_path, **kwargs)
        else:
            ok = mt5.initialize(**kwargs)
        if not ok:
            raise MT5UnavailableError(f"MT5 initialize failed: {self._last_error_text()}")
        self._connected = True
        if self.settings.login is not None:
            login = getattr(mt5, "login", None)
            if not callable(login):
                self.close()
                raise MT5UnavailableError("MT5 login function is unavailable")
            login_kwargs: dict[str, Any] = {"timeout": self.settings.timeout_ms}
            if self.settings.password is not None:
                login_kwargs["password"] = self.settings.password
            if self.settings.server is not None:
                login_kwargs["server"] = self.settings.server
            if not login(self.settings.login, **login_kwargs):
                reason = self._last_error_text()
                self.close()
                raise MT5UnavailableError(f"MT5 login failed: {reason}")

    def close(self) -> None:
        if self._connected:
            self._module().shutdown()
            self._connected = False

    def supported_symbols(self) -> tuple[str, ...]:
        if self.settings.symbol_map:
            return tuple(sorted(instrument_for(symbol).symbol for symbol in self.settings.symbol_map))
        return enabled_symbols()

    def get_latest_quote(self, symbol: str) -> QuoteSnapshot:
        self._ensure_connected()
        canonical = instrument_for(symbol).symbol
        mt5_symbol = self._mt5_symbol(canonical)
        tick = self._module().symbol_info_tick(mt5_symbol)
        if tick is None:
            raise MT5UnavailableError(
                f"MT5 returned no tick for {mt5_symbol}: {self._last_error_text()}"
            )
        bid = float(_field(tick, "bid"))
        ask = float(_field(tick, "ask"))
        timestamp = _timestamp_from_mt5_tick(tick)
        return QuoteSnapshot(timestamp=timestamp, symbol=canonical, bid=bid, ask=ask)

    def get_recent_bars(
        self,
        symbol: str,
        *,
        timeframe: str,
        count: int,
    ) -> tuple[PriceBar, ...]:
        if count < 1:
            raise ValueError("count must be at least 1")
        self._ensure_connected()
        canonical = instrument_for(symbol).symbol
        mt5_symbol = self._mt5_symbol(canonical)
        mt5_timeframe = _timeframe_value(self._module(), timeframe)
        rows = self._module().copy_rates_from_pos(mt5_symbol, mt5_timeframe, 0, count)
        if rows is None:
            raise MT5UnavailableError(
                f"MT5 returned no bars for {mt5_symbol}: {self._last_error_text()}"
            )
        bars = tuple(
            PriceBar(
                timestamp=datetime.fromtimestamp(int(_field(row, "time")), tz=UTC),
                symbol=canonical,
                close=float(_field(row, "close")),
            )
            for row in rows
        )
        return tuple(sorted(bars, key=lambda bar: bar.timestamp))

    def _ensure_connected(self) -> None:
        if not self._connected:
            self.connect()

    def _module(self) -> Any:
        if self._mt5 is None:
            try:
                self._mt5 = import_module("MetaTrader5")
            except ImportError as exc:
                raise MT5UnavailableError(
                    "MetaTrader5 package is not installed. "
                    "Install it only in the MT5 environment and keep this adapter read-only."
                ) from exc
        return self._mt5

    def _mt5_symbol(self, canonical: str) -> str:
        if canonical in self.settings.symbol_map:
            return self.settings.symbol_map[canonical]
        return canonical

    def _last_error_text(self) -> str:
        last_error = getattr(self._module(), "last_error", None)
        if not callable(last_error):
            return "last_error unavailable"
        return str(last_error())


class MT5AccountAdapter:
    def __init__(
        self,
        market_data_adapter: MT5MarketDataAdapter,
    ) -> None:
        self.market_data_adapter = market_data_adapter

    def get_account_snapshot(
        self,
        *,
        starting_equity: float,
        day_start_equity: float,
        peak_equity: float,
    ) -> AccountSnapshot:
        self.market_data_adapter._ensure_connected()
        info = self.market_data_adapter._module().account_info()
        if info is None:
            raise MT5UnavailableError(
                f"MT5 returned no account_info: {self.market_data_adapter._last_error_text()}"
            )
        equity = float(_field(info, "equity"))
        margin_level = _optional_float_field(info, "margin_level")
        if margin_level is not None and margin_level <= 0:
            margin_level = None
        return AccountSnapshot(
            equity=equity,
            starting_equity=starting_equity,
            day_start_equity=day_start_equity,
            peak_equity=max(peak_equity, equity),
            margin_level_pct=margin_level,
        )


def parse_symbol_map(values: tuple[str, ...]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for raw_value in values:
        if "=" not in raw_value:
            raise ValueError("symbol map entries must look like EURUSD=EURUSD.pro")
        left, right = raw_value.split("=", 1)
        canonical = instrument_for(left.strip()).symbol
        mt5_symbol = right.strip()
        if not mt5_symbol:
            raise ValueError("MT5 symbol map target cannot be empty")
        mapping[canonical] = mt5_symbol
    return mapping


def _timeframe_value(mt5: Any, timeframe: str) -> Any:
    normalized = timeframe.strip().upper()
    attr = f"TIMEFRAME_{normalized}"
    if not hasattr(mt5, attr):
        valid = "M1, M5, M15, M30, H1, H4, D1"
        raise ValueError(f"unsupported MT5 timeframe {timeframe!r}; common values: {valid}")
    return getattr(mt5, attr)


def _timestamp_from_mt5_tick(tick: Any) -> datetime:
    time_msc = _optional_float_field(tick, "time_msc")
    if time_msc is not None and time_msc > 0:
        return datetime.fromtimestamp(time_msc / 1_000, tz=UTC)
    return datetime.fromtimestamp(int(_field(tick, "time")), tz=UTC)


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value[name]
    try:
        return getattr(value, name)
    except AttributeError:
        return value[name]


def _optional_float_field(value: Any, name: str) -> float | None:
    try:
        raw_value = _field(value, name)
    except (AttributeError, KeyError, IndexError, TypeError):
        return None
    if raw_value is None:
        return None
    return float(raw_value)
