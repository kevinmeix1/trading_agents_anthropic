from __future__ import annotations

import csv
from dataclasses import dataclass
from math import log
from pathlib import Path

from quanthack.core.config import AppConfig
from quanthack.market.market_data import PriceBar, PriceHistory, QuoteHistory, QuoteSnapshot
from quanthack.strategies.strategy import SignalDirection, StrategySignal


@dataclass(frozen=True)
class SignalDiagnosticRow:
    symbol: str
    signal_name: str
    observations: int
    active_count: int
    long_count: int
    short_count: int
    hit_rate: float
    average_signed_forward_return_bps: float
    average_abs_forward_return_bps: float
    average_confidence: float
    average_weight: float
    average_edge_after_cost_bps: float


@dataclass(frozen=True)
class SignalDiagnosticsReport:
    strategy_name: str
    horizon_bars: int
    rows: tuple[SignalDiagnosticRow, ...]

    @property
    def ranked_rows(self) -> tuple[SignalDiagnosticRow, ...]:
        return tuple(
            sorted(
                self.rows,
                key=lambda row: (
                    row.average_signed_forward_return_bps,
                    row.hit_rate,
                    row.active_count,
                ),
                reverse=True,
            )
        )


def evaluate_signal_diagnostics(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    strategy_name: str = "alpha_router",
    symbols: tuple[str, ...] | None = None,
    horizon_bars: int = 1,
    min_confidence: float = 0.20,
    min_edge_after_cost_bps: float = 0.0,
) -> SignalDiagnosticsReport:
    if horizon_bars < 1:
        raise ValueError("horizon_bars must be at least 1")
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be between 0 and 1")

    selected_symbols = tuple(symbols or sorted(set(prices.symbols()) & set(quotes.symbols())))
    all_bars_by_symbol = {
        symbol: prices.for_symbol(symbol).bars
        for symbol in prices.symbols()
    }
    all_quotes_by_symbol = {
        symbol: quotes.for_symbol(symbol).quotes
        for symbol in quotes.symbols()
    }
    rows: list[SignalDiagnosticRow] = []
    for symbol in selected_symbols:
        strategy = config.build_strategy(strategy_name, symbol=symbol)
        generate_signals = getattr(strategy, "generate_signals", None)
        if generate_signals is None:
            raise ValueError(f"strategy {strategy_name!r} does not expose generate_signals")
        update_context = getattr(strategy, "update_portfolio_context", None)
        symbol_rows = _evaluate_symbol_signals(
            symbol=symbol,
            bars=prices.for_symbol(symbol).bars,
            quotes=quotes.for_symbol(symbol).quotes,
            all_bars_by_symbol=all_bars_by_symbol,
            all_quotes_by_symbol=all_quotes_by_symbol,
            update_context=update_context,
            generate_signals=generate_signals,
            horizon_bars=horizon_bars,
            min_confidence=min_confidence,
            min_edge_after_cost_bps=min_edge_after_cost_bps,
        )
        rows.extend(symbol_rows)

    return SignalDiagnosticsReport(
        strategy_name=strategy_name,
        horizon_bars=horizon_bars,
        rows=tuple(rows),
    )


def write_signal_diagnostics_csv(
    report: SignalDiagnosticsReport,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "strategy",
                "horizon_bars",
                "symbol",
                "signal",
                "observations",
                "active_count",
                "long_count",
                "short_count",
                "hit_rate",
                "average_signed_forward_return_bps",
                "average_abs_forward_return_bps",
                "average_confidence",
                "average_weight",
                "average_edge_after_cost_bps",
            ],
        )
        writer.writeheader()
        for row in report.ranked_rows:
            writer.writerow(
                {
                    "strategy": report.strategy_name,
                    "horizon_bars": report.horizon_bars,
                    "symbol": row.symbol,
                    "signal": row.signal_name,
                    "observations": row.observations,
                    "active_count": row.active_count,
                    "long_count": row.long_count,
                    "short_count": row.short_count,
                    "hit_rate": row.hit_rate,
                    "average_signed_forward_return_bps": row.average_signed_forward_return_bps,
                    "average_abs_forward_return_bps": row.average_abs_forward_return_bps,
                    "average_confidence": row.average_confidence,
                    "average_weight": row.average_weight,
                    "average_edge_after_cost_bps": row.average_edge_after_cost_bps,
                }
            )


def _evaluate_symbol_signals(
    *,
    symbol: str,
    bars: tuple[PriceBar, ...],
    quotes: tuple[QuoteSnapshot, ...],
    all_bars_by_symbol: dict[str, tuple[PriceBar, ...]],
    all_quotes_by_symbol: dict[str, tuple[QuoteSnapshot, ...]],
    update_context,
    generate_signals,
    horizon_bars: int,
    min_confidence: float,
    min_edge_after_cost_bps: float,
) -> tuple[SignalDiagnosticRow, ...]:
    if len(bars) <= horizon_bars:
        return ()

    quote_by_timestamp = {quote.timestamp: quote for quote in quotes}
    observations_by_signal: dict[str, int] = {}
    events_by_signal: dict[str, list[tuple[StrategySignal, float]]] = {}
    context_indexes = {context_symbol: 0 for context_symbol in all_bars_by_symbol}
    context_closes = {context_symbol: [] for context_symbol in all_bars_by_symbol}
    context_quote_indexes = {context_symbol: 0 for context_symbol in all_quotes_by_symbol}
    latest_quotes: dict[str, QuoteSnapshot] = {}

    closes: list[float] = []
    for index, bar in enumerate(bars[:-horizon_bars]):
        closes.append(bar.close)
        if update_context is not None:
            _advance_signal_context(
                timestamp=bar.timestamp,
                all_bars_by_symbol=all_bars_by_symbol,
                context_indexes=context_indexes,
                context_closes=context_closes,
                all_quotes_by_symbol=all_quotes_by_symbol,
                context_quote_indexes=context_quote_indexes,
                latest_quotes=latest_quotes,
            )
            update_context(
                closes_by_symbol={
                    context_symbol: tuple(symbol_closes)
                    for context_symbol, symbol_closes in context_closes.items()
                    if symbol_closes
                },
                quotes_by_symbol=latest_quotes,
            )
        future_close = bars[index + horizon_bars].close
        quote = quote_by_timestamp.get(bar.timestamp)
        signals = generate_signals(tuple(closes), quote=quote)
        for signal in signals:
            observations_by_signal[signal.strategy_name] = (
                observations_by_signal.get(signal.strategy_name, 0) + 1
            )
            if not _is_active_signal(
                signal,
                min_confidence=min_confidence,
                min_edge_after_cost_bps=min_edge_after_cost_bps,
            ):
                continue
            signed_forward_bps = (
                _direction_sign(signal.direction)
                * log(future_close / bar.close)
                * 10_000
            )
            events_by_signal.setdefault(signal.strategy_name, []).append(
                (signal, signed_forward_bps)
            )

    rows: list[SignalDiagnosticRow] = []
    for signal_name in sorted(observations_by_signal):
        events = events_by_signal.get(signal_name, [])
        active_count = len(events)
        long_count = sum(1 for signal, _ in events if signal.direction == SignalDirection.LONG)
        short_count = sum(1 for signal, _ in events if signal.direction == SignalDirection.SHORT)
        rows.append(
            SignalDiagnosticRow(
                symbol=symbol,
                signal_name=signal_name,
                observations=observations_by_signal[signal_name],
                active_count=active_count,
                long_count=long_count,
                short_count=short_count,
                hit_rate=(
                    sum(1 for _, signed_return in events if signed_return > 0) / active_count
                    if active_count
                    else 0.0
                ),
                average_signed_forward_return_bps=(
                    sum(signed_return for _, signed_return in events) / active_count
                    if active_count
                    else 0.0
                ),
                average_abs_forward_return_bps=(
                    sum(abs(signed_return) for _, signed_return in events) / active_count
                    if active_count
                    else 0.0
                ),
                average_confidence=(
                    sum(signal.confidence for signal, _ in events) / active_count
                    if active_count
                    else 0.0
                ),
                average_weight=(
                    sum(signal.weight for signal, _ in events) / active_count
                    if active_count
                    else 0.0
                ),
                average_edge_after_cost_bps=(
                    sum(signal.edge_after_cost_bps for signal, _ in events) / active_count
                    if active_count
                    else 0.0
                ),
            )
        )
    return tuple(rows)


def _advance_signal_context(
    *,
    timestamp,
    all_bars_by_symbol: dict[str, tuple[PriceBar, ...]],
    context_indexes: dict[str, int],
    context_closes: dict[str, list[float]],
    all_quotes_by_symbol: dict[str, tuple[QuoteSnapshot, ...]],
    context_quote_indexes: dict[str, int],
    latest_quotes: dict[str, QuoteSnapshot],
) -> None:
    for context_symbol, context_bars in all_bars_by_symbol.items():
        index = context_indexes[context_symbol]
        while index < len(context_bars) and context_bars[index].timestamp <= timestamp:
            context_closes[context_symbol].append(context_bars[index].close)
            index += 1
        context_indexes[context_symbol] = index

    for context_symbol, context_quotes in all_quotes_by_symbol.items():
        index = context_quote_indexes[context_symbol]
        while index < len(context_quotes) and context_quotes[index].timestamp <= timestamp:
            latest_quotes[context_symbol] = context_quotes[index]
            index += 1
        context_quote_indexes[context_symbol] = index


def _is_active_signal(
    signal: StrategySignal,
    *,
    min_confidence: float,
    min_edge_after_cost_bps: float,
) -> bool:
    return (
        signal.direction != SignalDirection.FLAT
        and signal.confidence >= min_confidence
        and signal.edge_after_cost_bps >= min_edge_after_cost_bps
    )


def _direction_sign(direction: SignalDirection) -> int:
    if direction == SignalDirection.LONG:
        return 1
    if direction == SignalDirection.SHORT:
        return -1
    return 0
