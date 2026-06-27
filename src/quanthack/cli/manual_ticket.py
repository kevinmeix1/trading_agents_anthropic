from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from math import floor, isfinite
from pathlib import Path

from quanthack.cli._format import money
from quanthack.core.clock import CompetitionMode
from quanthack.core.instruments import AssetClass, instrument_for
from quanthack.trading.execution import DryRunExecutor
from quanthack.trading.risk import AccountSnapshot, RiskEngine, Side, TradeRequest


DEFAULT_FX_CONTRACT_SIZE = 100_000.0


@dataclass(frozen=True)
class ManualTicket:
    symbol: str
    broker_symbol: str
    side: Side
    requested_notional_usd: float
    adjusted_notional_usd: float
    price: float
    contract_size: float
    raw_lots: float
    rounded_lots: float
    rounded_notional_usd: float
    volume_step: float
    min_volume: float
    risk_approved: bool
    risk_reason: str
    conversion_note: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a safe manual MT5 order ticket from a USD notional target."
    )
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--broker-symbol", default=None)
    parser.add_argument("--side", choices=[side.value for side in Side], required=True)
    parser.add_argument("--target-notional", type=float, default=50_000.0)
    parser.add_argument(
        "--price",
        type=float,
        required=True,
        help="Use the current MT5 ask for BUY, or bid for SELL.",
    )
    parser.add_argument("--equity", type=float, default=1_000_000.0)
    parser.add_argument("--day-start-equity", type=float, default=1_000_000.0)
    parser.add_argument("--peak-equity", type=float, default=1_000_000.0)
    parser.add_argument("--margin-level-pct", type=float, default=2_000.0)
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in CompetitionMode],
        default=CompetitionMode.QUALIFY.value,
    )
    parser.add_argument(
        "--contract-size",
        type=float,
        default=None,
        help="From MT5 Symbol Specification. Defaults to 100000 for FX only.",
    )
    parser.add_argument("--volume-step", type=float, default=0.01)
    parser.add_argument("--min-volume", type=float, default=0.01)
    parser.add_argument(
        "--quote-usd-rate",
        type=float,
        default=None,
        help="Required for non-USD quote crosses, e.g. GBPUSD for EURGBP.",
    )
    parser.add_argument(
        "--journal",
        default="outputs/dry_run_journal.jsonl",
        help="Used only to reconstruct local dry-run positions for risk checks.",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    try:
        ticket = build_manual_ticket(
            symbol=args.symbol,
            broker_symbol=args.broker_symbol,
            side=Side(args.side),
            target_notional_usd=args.target_notional,
            price=args.price,
            equity=args.equity,
            day_start_equity=args.day_start_equity,
            peak_equity=args.peak_equity,
            margin_level_pct=args.margin_level_pct,
            mode=CompetitionMode(args.mode),
            contract_size=args.contract_size,
            volume_step=args.volume_step,
            min_volume=args.min_volume,
            quote_usd_rate=args.quote_usd_rate,
            journal=Path(args.journal),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    _print_ticket(ticket)


def build_manual_ticket(
    *,
    symbol: str,
    broker_symbol: str | None,
    side: Side,
    target_notional_usd: float,
    price: float,
    equity: float,
    day_start_equity: float,
    peak_equity: float,
    margin_level_pct: float | None,
    mode: CompetitionMode,
    contract_size: float | None,
    volume_step: float,
    min_volume: float,
    quote_usd_rate: float | None,
    journal: Path,
) -> ManualTicket:
    instrument = instrument_for(symbol)
    _validate_positive("target_notional", target_notional_usd)
    _validate_positive("price", price)
    _validate_positive("volume_step", volume_step)
    _validate_positive("min_volume", min_volume)

    account = AccountSnapshot(
        equity=equity,
        day_start_equity=day_start_equity,
        peak_equity=peak_equity,
        margin_level_pct=margin_level_pct,
    )
    request = TradeRequest(
        symbol=instrument.symbol,
        side=side,
        target_notional_usd=target_notional_usd,
        reason="manual MT5 ticket pre-check",
    )
    portfolio = DryRunExecutor(journal).current_portfolio()
    decision = RiskEngine().evaluate(
        account=account,
        portfolio=portfolio,
        request=request,
        mode=mode,
    )
    adjusted_notional = decision.adjusted_notional_usd if decision.approved else 0.0
    resolved_contract_size = _contract_size_for(
        asset_class=instrument.asset_class,
        contract_size=contract_size,
    )
    notional_per_lot, conversion_note = _usd_notional_per_lot(
        symbol=instrument.symbol,
        base_currency=instrument.base_currency,
        quote_currency=instrument.quote_currency,
        price=price,
        contract_size=resolved_contract_size,
        quote_usd_rate=quote_usd_rate,
    )
    raw_lots = adjusted_notional / notional_per_lot if adjusted_notional > 0 else 0.0
    rounded_lots = _round_lots_down(raw_lots, volume_step)
    rounded_notional = rounded_lots * notional_per_lot

    return ManualTicket(
        symbol=instrument.symbol,
        broker_symbol=broker_symbol or instrument.symbol,
        side=side,
        requested_notional_usd=target_notional_usd,
        adjusted_notional_usd=adjusted_notional,
        price=price,
        contract_size=resolved_contract_size,
        raw_lots=raw_lots,
        rounded_lots=rounded_lots,
        rounded_notional_usd=rounded_notional,
        volume_step=volume_step,
        min_volume=min_volume,
        risk_approved=decision.approved,
        risk_reason=decision.reason,
        conversion_note=conversion_note,
    )


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


def _contract_size_for(
    *,
    asset_class: AssetClass,
    contract_size: float | None,
) -> float:
    if contract_size is None:
        if asset_class == AssetClass.FOREX:
            return DEFAULT_FX_CONTRACT_SIZE
        raise ValueError(
            "pass --contract-size for metals/crypto using MT5 Symbol Specification"
        )
    _validate_positive("contract_size", contract_size)
    return contract_size


def _usd_notional_per_lot(
    *,
    symbol: str,
    base_currency: str,
    quote_currency: str,
    price: float,
    contract_size: float,
    quote_usd_rate: float | None,
) -> tuple[float, str]:
    if quote_currency == "USD":
        return (
            contract_size * price,
            f"1 lot = {contract_size:,.0f} {base_currency}; USD notional uses {symbol} price",
        )
    if base_currency == "USD":
        return (
            contract_size,
            f"1 lot = {contract_size:,.0f} USD base units",
        )
    if quote_usd_rate is None:
        raise ValueError(
            f"{symbol} is a non-USD quote cross; pass --quote-usd-rate "
            f"for {quote_currency}USD or equivalent"
        )
    _validate_positive("quote_usd_rate", quote_usd_rate)
    return (
        contract_size * price * quote_usd_rate,
        (
            f"1 lot = {contract_size:,.0f} {base_currency}; converted through "
            f"{quote_currency}USD rate {quote_usd_rate}"
        ),
    )


def _round_lots_down(raw_lots: float, volume_step: float) -> float:
    if raw_lots <= 0:
        return 0.0
    return floor((raw_lots / volume_step) + 1e-12) * volume_step


def _validate_positive(name: str, value: float) -> None:
    if value <= 0 or not isfinite(value):
        raise ValueError(f"{name} must be positive and finite")


def _print_ticket(ticket: ManualTicket) -> None:
    print("Manual MT5 Order Ticket")
    print("  Mode: manual desktop entry, no Python order_send")
    print(f"  Risk decision: {'APPROVED' if ticket.risk_approved else 'BLOCKED'}")
    print(f"  Risk reason: {ticket.risk_reason}")
    print(f"  Symbol: {ticket.symbol}")
    print(f"  Broker symbol to type/select in MT5: {ticket.broker_symbol}")
    print(f"  Order type: {ticket.side.value}")
    print(f"  Price used: {ticket.price}")
    print(f"  Requested notional: {money(ticket.requested_notional_usd)}")
    print(f"  Risk-adjusted notional: {money(ticket.adjusted_notional_usd)}")
    print(f"  Contract size used: {ticket.contract_size:g}")
    print(f"  Conversion: {ticket.conversion_note}")
    print(f"  Raw lots: {ticket.raw_lots:.6f}")
    print(f"  MT5 volume to enter: {ticket.rounded_lots:.4f} lots")
    print(f"  Estimated rounded notional: {money(ticket.rounded_notional_usd)}")
    if not ticket.risk_approved:
        print("  Action: DO NOT PLACE THIS TRADE.")
        return
    if ticket.rounded_lots < ticket.min_volume:
        print(
            f"  Action: DO NOT PLACE. Rounded volume is below MT5 minimum "
            f"{ticket.min_volume:g} lots."
        )
        return
    print("  MT5 steps:")
    print("    1. Open MT5 Market Watch and select the broker symbol above.")
    print("    2. Click New Order.")
    print("    3. Set Type to Market Execution.")
    print(f"    4. Enter Volume = {ticket.rounded_lots:.4f}.")
    print(f"    5. Click {ticket.side.value}.")
    print("    6. Confirm the Trade tab shows the expected position.")
    print("  Before clicking, verify MT5 Symbol Specification contract size, min volume, and volume step.")
