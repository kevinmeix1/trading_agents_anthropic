from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.backtesting.research_candidate_gate import (
    ResearchCandidateSource,
    build_research_candidate_gate,
    normalize_research_data_source,
    write_research_candidate_gate_csv,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Gate research comparison CSV rows into live-ready, paper-only, "
            "or rejected candidate evidence."
        )
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help=(
            "Research comparison CSV as path=CSV,data_source=official|proxy|"
            "mixed_proxy|synthetic. Repeat for multiple files."
        ),
    )
    parser.add_argument(
        "--output",
        default="outputs/research/research_candidate_gate.csv",
    )
    parser.add_argument("--limit", type=int, default=10)
    return parser


def run(args: argparse.Namespace) -> None:
    rows = build_research_candidate_gate(
        tuple(_parse_source(value) for value in args.source)
    )
    write_research_candidate_gate_csv(rows, args.output)

    print("Research Candidate Gate")
    print(f"  Sources: {len(args.source)}")
    print(f"  Candidates: {len(rows)}")
    print(f"  Output CSV: {args.output}")
    for rank, row in enumerate(rows[: max(args.limit, 0)], start=1):
        print(
            f"  {rank}. {row.label}: "
            f"{row.readiness.value}, "
            f"score={row.decision_score:.1f}, "
            f"return={row.return_pct:.3%}, "
            f"drawdown={row.max_drawdown_pct:.3%}, "
            f"sharpe15={row.sharpe_15m:.3f}, "
            f"risk={row.risk_discipline_score:.0f}/100, "
            f"source={row.data_source.value}, "
            f"reason={row.reason}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


def _parse_source(raw: str) -> ResearchCandidateSource:
    fields: dict[str, str] = {}
    for part in raw.split(","):
        key, separator, value = part.partition("=")
        if not separator:
            raise argparse.ArgumentTypeError(
                "--source must use path=CSV,data_source=official|proxy|mixed_proxy|synthetic"
            )
        normalized_key = key.strip().lower().replace("-", "_")
        if normalized_key not in {"path", "data", "data_source"}:
            raise argparse.ArgumentTypeError(
                f"unknown --source field {normalized_key!r}; expected path/data_source"
            )
        fields[normalized_key] = value.strip()

    path = fields.get("path", "")
    data_source = fields.get("data_source") or fields.get("data") or ""
    if not path or not data_source:
        raise argparse.ArgumentTypeError("--source path and data_source are required")
    try:
        normalized_data_source = normalize_research_data_source(data_source)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return ResearchCandidateSource(path=path, data_source=normalized_data_source)
