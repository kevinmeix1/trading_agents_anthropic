from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import PureWindowsPath

from quanthack.backtesting.candidate_scorecard import (
    CandidateBundle,
    build_candidate_scorecard,
    write_candidate_scorecard_csv,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rank portfolio backtest output bundles by competition-style metrics."
    )
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help=(
            "Candidate bundle as label=NAME,equity=EQUITY_CSV,fills=FILLS_CSV[,pnl=PNL_CSV]. "
            "The legacy LABEL:EQUITY_CSV:FILLS_CSV[:PNL_CSV] form is also supported. "
            "Repeat for multiple candidates."
        ),
    )
    parser.add_argument(
        "--output",
        default="outputs/backtests/candidate_scorecard.csv",
    )
    parser.add_argument("--limit", type=int, default=10)
    return parser


def run(args: argparse.Namespace) -> None:
    rows = build_candidate_scorecard(
        tuple(_parse_candidate(value) for value in args.candidate)
    )
    write_candidate_scorecard_csv(rows, args.output)

    print("Candidate Scorecard")
    print(f"  Candidates: {len(rows)}")
    print(f"  Output CSV: {args.output}")
    for rank, row in enumerate(rows[: max(args.limit, 0)], start=1):
        prize = "yes" if row.sharpe_prize_trade_count_met else "no"
        review = "yes" if row.compliance_review_required else "no"
        print(
            f"  {rank}. {row.label}: "
            f"score={row.composite_score:.1f}, "
            f"return={row.return_pct:.3%}, "
            f"drawdown={row.max_drawdown_pct:.3%}, "
            f"sharpe15={row.sharpe_15m:.3f}, "
            f"trades={row.trade_count}, "
            f"sharpe_prize={prize}, "
            f"risk={row.risk_discipline_score}/100, "
            f"review={review}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


def _parse_candidate(raw: str) -> CandidateBundle:
    if "=" in raw:
        return _parse_key_value_candidate(raw)
    return _parse_colon_candidate(raw)


def _parse_key_value_candidate(raw: str) -> CandidateBundle:
    fields: dict[str, str] = {}
    for part in raw.split(","):
        key, separator, value = part.partition("=")
        if not separator:
            raise argparse.ArgumentTypeError(
                "--candidate key-value form must use label=...,equity=...,fills=...[,pnl=...]"
            )
        normalized_key = key.strip().lower()
        if normalized_key not in {"label", "equity", "fills", "pnl"}:
            raise argparse.ArgumentTypeError(
                f"unknown --candidate field {normalized_key!r}; expected label/equity/fills/pnl"
            )
        fields[normalized_key] = value.strip()

    label = fields.get("label", "")
    equity_csv = fields.get("equity", "")
    fills_csv = fields.get("fills", "")
    pnl_csv = fields.get("pnl")
    if not label or not equity_csv or not fills_csv:
        raise argparse.ArgumentTypeError(
            "candidate label/equity/fills cannot be empty"
        )
    return CandidateBundle(
        label=label,
        equity_csv=equity_csv,
        fills_csv=fills_csv,
        pnl_csv=pnl_csv or None,
    )


def _parse_colon_candidate(raw: str) -> CandidateBundle:
    parts = raw.split(":")
    if len(parts) > 4:
        parts = _restore_windows_drive_colons(parts)
    if len(parts) not in {3, 4}:
        raise argparse.ArgumentTypeError(
            "--candidate must use label=...,equity=...,fills=...[,pnl=...] "
            "or LABEL:EQUITY_CSV:FILLS_CSV[:PNL_CSV]"
        )
    label, equity_csv, fills_csv = (part.strip() for part in parts[:3])
    pnl_csv = parts[3].strip() if len(parts) == 4 else None
    if not label or not equity_csv or not fills_csv:
        raise argparse.ArgumentTypeError("candidate label/equity/fills cannot be empty")
    return CandidateBundle(
        label=label,
        equity_csv=equity_csv,
        fills_csv=fills_csv,
        pnl_csv=pnl_csv or None,
    )


def _restore_windows_drive_colons(parts: list[str]) -> list[str]:
    restored: list[str] = []
    index = 0
    while index < len(parts):
        part = parts[index]
        if (
            len(part) == 1
            and part.isalpha()
            and index + 1 < len(parts)
            and PureWindowsPath(f"{part}:{parts[index + 1]}").drive
        ):
            restored.append(f"{part}:{parts[index + 1]}")
            index += 2
            continue
        restored.append(part)
        index += 1
    return restored
