from __future__ import annotations

import csv
from dataclasses import dataclass
from math import floor, isfinite
from pathlib import Path

from quanthack.core.instruments import AssetClass, instrument_for
from quanthack.trading.risk import Side


DEFAULT_FX_CONTRACT_SIZE = 100_000.0
DEFAULT_VOLUME_STEP = 0.01
DEFAULT_MIN_VOLUME = 0.01
EPSILON_NOTIONAL = 1e-9


@dataclass(frozen=True)
class Mt5ContractSpec:
    symbol: str
    broker_symbol: str
    contract_size: float | None = None
    volume_step: float = DEFAULT_VOLUME_STEP
    min_volume: float = DEFAULT_MIN_VOLUME
    quote_usd_rate: float | None = None

    def __post_init__(self) -> None:
        canonical = instrument_for(self.symbol).symbol
        object.__setattr__(self, "symbol", canonical)
        if not self.broker_symbol:
            object.__setattr__(self, "broker_symbol", canonical)
        _validate_optional_positive("contract_size", self.contract_size)
        _validate_positive("volume_step", self.volume_step)
        _validate_positive("min_volume", self.min_volume)
        _validate_optional_positive("quote_usd_rate", self.quote_usd_rate)


@dataclass(frozen=True)
class Mt5TicketSheetRow:
    profile_slot: str
    profile_label: str
    timestamp: str
    symbol: str
    broker_symbol: str
    side: str
    status: str
    entry_price: float
    current_notional_usd: float
    target_notional_usd: float
    action_notional_usd: float
    risk_adjusted_final_notional_usd: float
    contract_size: float | None
    volume_step: float
    min_volume: float
    quote_usd_rate: float | None
    notional_per_lot_usd: float
    raw_lots: float
    rounded_lots: float
    rounded_notional_usd: float
    risk_reason: str
    strategy_reason: str
    conversion_note: str
    instruction: str


def load_contract_specs(path: str | Path) -> dict[str, Mt5ContractSpec]:
    specs: dict[str, Mt5ContractSpec] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _validate_columns(reader.fieldnames, {"symbol"}, "contract spec CSV")
        for row_number, row in enumerate(reader, start=2):
            try:
                spec = Mt5ContractSpec(
                    symbol=str(row.get("symbol", "")).strip(),
                    broker_symbol=str(row.get("broker_symbol", "")).strip(),
                    contract_size=_optional_float(row, "contract_size"),
                    volume_step=_optional_float(row, "volume_step") or DEFAULT_VOLUME_STEP,
                    min_volume=_optional_float(row, "min_volume") or DEFAULT_MIN_VOLUME,
                    quote_usd_rate=_optional_float(row, "quote_usd_rate"),
                )
            except (KeyError, ValueError) as exc:
                raise ValueError(f"contract spec row {row_number}: {exc}") from exc
            specs[spec.symbol] = spec
    return specs


def build_mt5_ticket_sheet_from_snapshot_csv(
    snapshot_csv: str | Path,
    *,
    contract_specs: dict[str, Mt5ContractSpec] | None = None,
    broker_symbol_by_symbol: dict[str, str] | None = None,
    include_holds: bool = False,
) -> tuple[Mt5TicketSheetRow, ...]:
    specs = contract_specs or {}
    broker_map = broker_symbol_by_symbol or {}
    rows: list[Mt5TicketSheetRow] = []
    with Path(snapshot_csv).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _validate_columns(
            reader.fieldnames,
            {
                "profile_slot",
                "profile_label",
                "timestamp",
                "symbol",
                "bid",
                "ask",
                "current_notional_usd",
                "allocated_target_notional_usd",
                "change_notional_usd",
                "order_side",
                "risk_approved",
                "risk_adjusted_notional_usd",
                "risk_reason",
                "strategy_reason",
            },
            "snapshot CSV",
        )
        for row_number, row in enumerate(reader, start=2):
            ticket = _ticket_from_snapshot_row(
                row,
                row_number=row_number,
                contract_specs=specs,
                broker_symbol_by_symbol=broker_map,
            )
            if include_holds or ticket.status != "HOLD":
                rows.append(ticket)
    return tuple(rows)


def write_mt5_ticket_sheet_csv(
    tickets: tuple[Mt5TicketSheetRow, ...],
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "profile_slot",
                "profile_label",
                "timestamp",
                "symbol",
                "broker_symbol",
                "side",
                "status",
                "entry_price",
                "current_notional_usd",
                "target_notional_usd",
                "action_notional_usd",
                "risk_adjusted_final_notional_usd",
                "contract_size",
                "volume_step",
                "min_volume",
                "quote_usd_rate",
                "notional_per_lot_usd",
                "raw_lots",
                "rounded_lots",
                "rounded_notional_usd",
                "risk_reason",
                "strategy_reason",
                "conversion_note",
                "instruction",
            ],
        )
        writer.writeheader()
        for ticket in tickets:
            writer.writerow(_ticket_to_row(ticket))


def write_contract_spec_template_from_snapshot_csv(
    snapshot_csv: str | Path,
    path: str | Path,
) -> None:
    symbols: list[str] = []
    seen: set[str] = set()
    with Path(snapshot_csv).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _validate_columns(reader.fieldnames, {"symbol"}, "snapshot CSV")
        for row in reader:
            symbol = instrument_for(str(row["symbol"]).strip()).symbol
            if symbol in seen:
                continue
            symbols.append(symbol)
            seen.add(symbol)

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "symbol",
                "broker_symbol",
                "contract_size",
                "volume_step",
                "min_volume",
                "quote_usd_rate",
            ],
        )
        writer.writeheader()
        for symbol in symbols:
            instrument = instrument_for(symbol)
            default_contract_size = (
                DEFAULT_FX_CONTRACT_SIZE
                if instrument.asset_class == AssetClass.FOREX
                else ""
            )
            writer.writerow(
                {
                    "symbol": symbol,
                    "broker_symbol": symbol,
                    "contract_size": default_contract_size,
                    "volume_step": DEFAULT_VOLUME_STEP,
                    "min_volume": DEFAULT_MIN_VOLUME,
                    "quote_usd_rate": "",
                }
            )


def _ticket_from_snapshot_row(
    row: dict[str, str],
    *,
    row_number: int,
    contract_specs: dict[str, Mt5ContractSpec],
    broker_symbol_by_symbol: dict[str, str],
) -> Mt5TicketSheetRow:
    try:
        symbol = instrument_for(row["symbol"]).symbol
        side = row["order_side"].strip().upper()
        if side not in {Side.BUY.value, Side.SELL.value, "HOLD"}:
            raise ValueError(f"unknown order_side {side!r}")
        bid = float(row["bid"])
        ask = float(row["ask"])
        current_notional = float(row["current_notional_usd"])
        target_notional = float(row["allocated_target_notional_usd"])
        change_notional = float(row["change_notional_usd"])
        risk_adjusted_final = float(row["risk_adjusted_notional_usd"])
        risk_approved = _parse_bool(row["risk_approved"])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"snapshot row {row_number}: {exc}") from exc

    spec = _resolve_spec(
        symbol=symbol,
        contract_specs=contract_specs,
        broker_symbol_by_symbol=broker_symbol_by_symbol,
    )
    entry_price = ask if side == Side.BUY.value else bid
    action_notional = abs(change_notional)
    base_kwargs = {
        "profile_slot": row["profile_slot"],
        "profile_label": row["profile_label"],
        "timestamp": row["timestamp"],
        "symbol": symbol,
        "broker_symbol": spec.broker_symbol,
        "side": side,
        "entry_price": entry_price,
        "current_notional_usd": current_notional,
        "target_notional_usd": target_notional,
        "action_notional_usd": action_notional,
        "risk_adjusted_final_notional_usd": risk_adjusted_final,
        "contract_size": spec.contract_size,
        "volume_step": spec.volume_step,
        "min_volume": spec.min_volume,
        "quote_usd_rate": spec.quote_usd_rate,
        "risk_reason": row["risk_reason"],
        "strategy_reason": row["strategy_reason"],
    }
    if side == "HOLD" or action_notional <= EPSILON_NOTIONAL:
        return _status_row(
            **base_kwargs,
            status="HOLD",
            instruction="No manual MT5 action.",
        )
    if not risk_approved:
        return _status_row(
            **base_kwargs,
            status="BLOCKED_BY_RISK",
            instruction="Do not place this trade.",
        )
    if spec.contract_size is None:
        return _status_row(
            **base_kwargs,
            status="NEEDS_CONTRACT_SPEC",
            instruction="Inspect MT5 Symbol Specification and add contract_size.",
        )

    try:
        notional_per_lot, conversion_note = _usd_notional_per_lot(
            symbol=symbol,
            price=entry_price,
            contract_size=spec.contract_size,
            quote_usd_rate=spec.quote_usd_rate,
        )
    except ValueError as exc:
        return _status_row(
            **base_kwargs,
            status="NEEDS_QUOTE_USD_RATE",
            instruction=str(exc),
        )

    raw_lots = action_notional / notional_per_lot
    rounded_lots = _round_lots_down(raw_lots, spec.volume_step)
    rounded_notional = rounded_lots * notional_per_lot
    if rounded_lots < spec.min_volume:
        return Mt5TicketSheetRow(
            **base_kwargs,
            status="BELOW_MIN_VOLUME",
            notional_per_lot_usd=notional_per_lot,
            raw_lots=raw_lots,
            rounded_lots=rounded_lots,
            rounded_notional_usd=rounded_notional,
            conversion_note=conversion_note,
            instruction="Do not place; rounded volume is below MT5 minimum volume.",
        )

    return Mt5TicketSheetRow(
        **base_kwargs,
        status="READY",
        notional_per_lot_usd=notional_per_lot,
        raw_lots=raw_lots,
        rounded_lots=rounded_lots,
        rounded_notional_usd=rounded_notional,
        conversion_note=conversion_note,
        instruction=(
            f"Open {spec.broker_symbol}; Market Execution {side}; "
            f"Volume {rounded_lots:.4f} lots."
        ),
    )


def _resolve_spec(
    *,
    symbol: str,
    contract_specs: dict[str, Mt5ContractSpec],
    broker_symbol_by_symbol: dict[str, str],
) -> Mt5ContractSpec:
    instrument = instrument_for(symbol)
    spec = contract_specs.get(symbol)
    broker_symbol = broker_symbol_by_symbol.get(
        symbol,
        spec.broker_symbol if spec is not None else symbol,
    )
    if spec is None:
        return Mt5ContractSpec(
            symbol=symbol,
            broker_symbol=broker_symbol,
            contract_size=(
                DEFAULT_FX_CONTRACT_SIZE
                if instrument.asset_class == AssetClass.FOREX
                else None
            ),
        )
    return Mt5ContractSpec(
        symbol=symbol,
        broker_symbol=broker_symbol,
        contract_size=spec.contract_size,
        volume_step=spec.volume_step,
        min_volume=spec.min_volume,
        quote_usd_rate=spec.quote_usd_rate,
    )


def _status_row(
    *,
    status: str,
    instruction: str,
    profile_slot: str,
    profile_label: str,
    timestamp: str,
    symbol: str,
    broker_symbol: str,
    side: str,
    entry_price: float,
    current_notional_usd: float,
    target_notional_usd: float,
    action_notional_usd: float,
    risk_adjusted_final_notional_usd: float,
    contract_size: float | None,
    volume_step: float,
    min_volume: float,
    quote_usd_rate: float | None,
    risk_reason: str,
    strategy_reason: str,
) -> Mt5TicketSheetRow:
    return Mt5TicketSheetRow(
        profile_slot=profile_slot,
        profile_label=profile_label,
        timestamp=timestamp,
        symbol=symbol,
        broker_symbol=broker_symbol,
        side=side,
        status=status,
        entry_price=entry_price,
        current_notional_usd=current_notional_usd,
        target_notional_usd=target_notional_usd,
        action_notional_usd=action_notional_usd,
        risk_adjusted_final_notional_usd=risk_adjusted_final_notional_usd,
        contract_size=contract_size,
        volume_step=volume_step,
        min_volume=min_volume,
        quote_usd_rate=quote_usd_rate,
        notional_per_lot_usd=0.0,
        raw_lots=0.0,
        rounded_lots=0.0,
        rounded_notional_usd=0.0,
        risk_reason=risk_reason,
        strategy_reason=strategy_reason,
        conversion_note="",
        instruction=instruction,
    )


def _usd_notional_per_lot(
    *,
    symbol: str,
    price: float,
    contract_size: float,
    quote_usd_rate: float | None,
) -> tuple[float, str]:
    instrument = instrument_for(symbol)
    _validate_positive("price", price)
    _validate_positive("contract_size", contract_size)
    if instrument.quote_currency == "USD":
        return (
            contract_size * price,
            (
                f"1 lot = {contract_size:,.0f} {instrument.base_currency}; "
                f"USD notional uses {instrument.symbol} price"
            ),
        )
    if instrument.base_currency == "USD":
        return contract_size, f"1 lot = {contract_size:,.0f} USD base units"
    if quote_usd_rate is None:
        raise ValueError(
            f"{instrument.symbol} needs quote_usd_rate for "
            f"{instrument.quote_currency}USD conversion"
        )
    _validate_positive("quote_usd_rate", quote_usd_rate)
    return (
        contract_size * price * quote_usd_rate,
        (
            f"1 lot = {contract_size:,.0f} {instrument.base_currency}; "
            f"converted through {instrument.quote_currency}USD rate {quote_usd_rate}"
        ),
    )


def _round_lots_down(raw_lots: float, volume_step: float) -> float:
    if raw_lots <= 0:
        return 0.0
    return floor((raw_lots / volume_step) + 1e-12) * volume_step


def _ticket_to_row(ticket: Mt5TicketSheetRow) -> dict[str, str | float]:
    return {
        "profile_slot": ticket.profile_slot,
        "profile_label": ticket.profile_label,
        "timestamp": ticket.timestamp,
        "symbol": ticket.symbol,
        "broker_symbol": ticket.broker_symbol,
        "side": ticket.side,
        "status": ticket.status,
        "entry_price": ticket.entry_price,
        "current_notional_usd": ticket.current_notional_usd,
        "target_notional_usd": ticket.target_notional_usd,
        "action_notional_usd": ticket.action_notional_usd,
        "risk_adjusted_final_notional_usd": ticket.risk_adjusted_final_notional_usd,
        "contract_size": "" if ticket.contract_size is None else ticket.contract_size,
        "volume_step": ticket.volume_step,
        "min_volume": ticket.min_volume,
        "quote_usd_rate": "" if ticket.quote_usd_rate is None else ticket.quote_usd_rate,
        "notional_per_lot_usd": ticket.notional_per_lot_usd,
        "raw_lots": ticket.raw_lots,
        "rounded_lots": ticket.rounded_lots,
        "rounded_notional_usd": ticket.rounded_notional_usd,
        "risk_reason": ticket.risk_reason,
        "strategy_reason": ticket.strategy_reason,
        "conversion_note": ticket.conversion_note,
        "instruction": ticket.instruction,
    }


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"invalid boolean value {value!r}")


def _optional_float(row: dict[str, str], key: str) -> float | None:
    raw_value = row.get(key, "")
    if raw_value is None or str(raw_value).strip() == "":
        return None
    return float(raw_value)


def _validate_columns(
    fieldnames: list[str] | None,
    required: set[str],
    label: str,
) -> None:
    found = set(fieldnames or [])
    missing = required - found
    if missing:
        raise ValueError(f"{label} missing required columns: {sorted(missing)}")


def _validate_positive(name: str, value: float) -> None:
    if value <= 0 or not isfinite(value):
        raise ValueError(f"{name} must be positive and finite")


def _validate_optional_positive(name: str, value: float | None) -> None:
    if value is not None:
        _validate_positive(name, value)
