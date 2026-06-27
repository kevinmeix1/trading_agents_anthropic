from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from quanthack.core.instruments import instrument_for


@dataclass(frozen=True)
class SymbolUniverseProfile:
    slot: str
    label: str
    evidence_status: str
    use_case: str
    reason: str
    data_source: str
    return_pct: float
    max_drawdown_pct: float
    sharpe_15m: float
    risk_discipline_score: float
    fold_contribution: float
    promotion_status: str
    promotion_reason: str
    strategy_map: str
    multiplier_map: str
    source_candidate: str
    excluded_symbols: str


@dataclass(frozen=True)
class SymbolUniverseProfilePack:
    source_symbol_eligibility_csv: str
    data_source: str
    recommended_slot: str
    recommendation_reason: str
    profiles: tuple[SymbolUniverseProfile, ...]


def build_symbol_universe_profile_pack(
    *,
    symbol_eligibility_csv: str | Path,
    candidate: str = "rank:1",
    selected_slot: str = "symbol_universe",
    selected_label: str | None = None,
    baseline_slot: str = "baseline",
    data_source: str = "research",
    include_baseline: bool = True,
) -> SymbolUniverseProfilePack:
    rows = _read_rows(symbol_eligibility_csv)
    selected = _select_row(rows, candidate)
    strategy = selected.get("strategy", "").strip()
    if not strategy:
        raise ValueError("selected candidate row is missing strategy")

    profiles: list[SymbolUniverseProfile] = []
    if include_baseline:
        baseline = _baseline_row(rows)
        if baseline is not None and baseline is not selected:
            profiles.append(
                _profile_from_row(
                    row=baseline,
                    slot=baseline_slot,
                    label=f"{strategy}_all_symbols",
                    data_source=data_source,
                    reason_prefix="Baseline symbol universe from eligibility optimizer",
                    use_case=(
                        "Reference profile using every eligible symbol from the "
                        "symbol-eligibility run."
                    ),
                )
            )

    profiles.append(
        _profile_from_row(
            row=selected,
            slot=selected_slot,
            label=selected_label or f"{strategy}_{selected.get('candidate', 'selected')}",
            data_source=data_source,
            reason_prefix="Selected symbol universe from eligibility optimizer",
            use_case=(
                "Research-only defensive symbol-universe profile; validate on fresh "
                "official or live dry-run data before MT5 execution."
            ),
        )
    )
    return SymbolUniverseProfilePack(
        source_symbol_eligibility_csv=str(symbol_eligibility_csv),
        data_source=data_source,
        recommended_slot=selected_slot,
        recommendation_reason=(
            "research-only symbol-universe refinement; rerun eligibility on fresh "
            "data before live use"
        ),
        profiles=tuple(profiles),
    )


def write_symbol_universe_profile_pack_json(
    pack: SymbolUniverseProfilePack,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_symbol_eligibility_csv": pack.source_symbol_eligibility_csv,
        "data_source": pack.data_source,
        "recommended_slot": pack.recommended_slot,
        "recommendation_reason": pack.recommendation_reason,
        "profiles": [asdict(profile) for profile in pack.profiles],
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_symbol_universe_profile_pack_csv(
    pack: SymbolUniverseProfilePack,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "slot",
        "label",
        "evidence_status",
        "use_case",
        "reason",
        "data_source",
        "return_pct",
        "max_drawdown_pct",
        "sharpe_15m",
        "risk_discipline_score",
        "fold_contribution",
        "promotion_status",
        "promotion_reason",
        "strategy_map",
        "multiplier_map",
        "source_candidate",
        "excluded_symbols",
        "recommended_slot",
        "recommendation_reason",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for profile in pack.profiles:
            row = asdict(profile)
            row["recommended_slot"] = pack.recommended_slot
            row["recommendation_reason"] = pack.recommendation_reason
            writer.writerow(row)


def _profile_from_row(
    *,
    row: dict[str, str],
    slot: str,
    label: str,
    data_source: str,
    reason_prefix: str,
    use_case: str,
) -> SymbolUniverseProfile:
    strategy = row.get("strategy", "").strip()
    symbols = _symbols(row.get("symbols", ""))
    excluded = _symbols(row.get("excluded_symbols", ""))
    if not symbols:
        raise ValueError(f"candidate {row.get('candidate', '')!r} has no symbols")
    strategy_map = " ".join(f"{symbol}={strategy}" for symbol in symbols)
    multiplier_map = " ".join(f"{symbol}=1.000" for symbol in symbols)
    candidate_name = row.get("candidate", "").strip()
    reason = (
        f"{reason_prefix}: {candidate_name or slot}; "
        f"excluded {', '.join(excluded) or 'none'}; "
        f"source reason: {row.get('reason', '').strip() or 'n/a'}"
    )
    return SymbolUniverseProfile(
        slot=slot,
        label=label,
        evidence_status="PAPER_ONLY",
        use_case=use_case,
        reason=reason,
        data_source=data_source,
        return_pct=_float(row, "official_return_pct"),
        max_drawdown_pct=_float(row, "official_max_drawdown_pct"),
        sharpe_15m=_float(row, "official_15m_sharpe"),
        risk_discipline_score=_float(row, "risk_discipline_score"),
        fold_contribution=_float(row, "wf_largest_positive_fold_contribution"),
        promotion_status="PAPER_ONLY",
        promotion_reason=(
            "Symbol-universe research pack; require fresh official or live dry-run "
            "confirmation before promotion."
        ),
        strategy_map=strategy_map,
        multiplier_map=multiplier_map,
        source_candidate=candidate_name,
        excluded_symbols=" ".join(excluded),
    )


def _read_rows(path: str | Path) -> tuple[dict[str, str], ...]:
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        rows = tuple(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{input_path} has no symbol eligibility rows")
    return rows


def _select_row(rows: tuple[dict[str, str], ...], candidate: str) -> dict[str, str]:
    if candidate.startswith("rank:"):
        rank = candidate.partition(":")[2].strip()
        for row in rows:
            if row.get("rank", "").strip() == rank:
                return row
        raise ValueError(f"candidate rank {rank!r} not found")
    for row in rows:
        if row.get("candidate", "").strip() == candidate:
            return row
    available = ", ".join(row.get("candidate", "") for row in rows)
    raise ValueError(f"candidate {candidate!r} not found; available: {available}")


def _baseline_row(rows: tuple[dict[str, str], ...]) -> dict[str, str] | None:
    for row in rows:
        if row.get("candidate", "").strip() == "all_symbols":
            return row
    return None


def _symbols(text: str) -> tuple[str, ...]:
    normalized = text.replace(",", " ")
    symbols = [
        instrument_for(part).symbol for part in normalized.split() if part.strip()
    ]
    return tuple(dict.fromkeys(symbols))


def _float(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    if value == "":
        return 0.0
    return float(value)
