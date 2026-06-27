from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class DeploymentProfile:
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
    fx_multiplier: float | None = None
    metal_multiplier: float | None = None
    crypto_profile: str = ""
    crypto_allowed_utc_hours: str = ""
    strategy_map: str = ""
    multiplier_map: str = ""


@dataclass(frozen=True)
class DeploymentProfilePack:
    source_summary_csv: str
    source_asset_class_csv: str
    data_source: str
    recommended_slot: str
    recommendation_reason: str
    profiles: tuple[DeploymentProfile, ...]


def build_deployment_profile_pack(
    *,
    promotion_summary_csv: str | Path,
    asset_class_stability_csv: str | Path,
) -> DeploymentProfilePack:
    summary_path = Path(promotion_summary_csv)
    asset_path = Path(asset_class_stability_csv)
    summary = _read_one_row(summary_path)
    asset_rows = _read_rows(asset_path)
    if not asset_rows:
        raise ValueError(f"{asset_path} produced no deployment profile rows")

    data_source = summary.get("data_source", "")
    aggressive = _aggressive_profile(summary=summary, rows=asset_rows)
    conservative = _stable_profile(
        slot="conservative",
        use_case="Prefer this when risk discipline and fold stability matter more than return.",
        reason="highest-return stable asset-class profile",
        data_source=data_source,
        rows=asset_rows,
        key=lambda row: (
            _float(row, "return_pct"),
            _float(row, "stability_score"),
            -_float(row, "max_drawdown_pct"),
        ),
    )
    survival = _stable_profile(
        slot="survival",
        use_case="Prefer this when avoiding drawdown and single-fold dependence dominates.",
        reason="lowest fold-contribution stable asset-class profile",
        data_source=data_source,
        rows=asset_rows,
        key=lambda row: (
            -_float(row, "wf_largest_positive_fold_contribution"),
            _float(row, "return_pct"),
            _float(row, "stability_score"),
        ),
    )
    profiles = tuple(
        profile for profile in (aggressive, conservative, survival) if profile is not None
    )
    recommended_slot, recommendation_reason = _recommendation(
        data_source=data_source,
        summary=summary,
        aggressive=aggressive,
        conservative=conservative,
    )
    return DeploymentProfilePack(
        source_summary_csv=str(summary_path),
        source_asset_class_csv=str(asset_path),
        data_source=data_source,
        recommended_slot=recommended_slot,
        recommendation_reason=recommendation_reason,
        profiles=profiles,
    )


def write_deployment_profile_pack_csv(
    pack: DeploymentProfilePack,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
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
                "fx_multiplier",
                "metal_multiplier",
                "crypto_profile",
                "crypto_allowed_utc_hours",
                "strategy_map",
                "multiplier_map",
                "recommended_slot",
                "recommendation_reason",
            ],
        )
        writer.writeheader()
        for profile in pack.profiles:
            row = asdict(profile)
            row["recommended_slot"] = pack.recommended_slot
            row["recommendation_reason"] = pack.recommendation_reason
            writer.writerow(row)


def write_deployment_profile_pack_json(
    pack: DeploymentProfilePack,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_summary_csv": pack.source_summary_csv,
        "source_asset_class_csv": pack.source_asset_class_csv,
        "data_source": pack.data_source,
        "recommended_slot": pack.recommended_slot,
        "recommendation_reason": pack.recommendation_reason,
        "profiles": [asdict(profile) for profile in pack.profiles],
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _aggressive_profile(
    *,
    summary: dict[str, str],
    rows: tuple[dict[str, str], ...],
) -> DeploymentProfile:
    best_row = max(
        rows,
        key=lambda row: (
            _float(row, "return_pct"),
            _float(row, "stability_score"),
            -_float(row, "max_drawdown_pct"),
        ),
    )
    data_source = summary.get("data_source", "")
    return DeploymentProfile(
        slot="aggressive",
        label=best_row.get("label", summary.get("best_sizing_label", "")),
        evidence_status=_evidence_status(
            data_source=data_source,
            promotion_status=summary.get("promotion_readiness", ""),
            fold_contribution=_float(summary, "largest_positive_fold_contribution"),
        ),
        use_case="Prefer this only when pursuing maximum return after official-data validation.",
        reason=summary.get("promotion_reason", ""),
        data_source=data_source,
        return_pct=_float(summary, "best_sizing_return_pct"),
        max_drawdown_pct=_float(summary, "best_sizing_drawdown_pct"),
        sharpe_15m=_float(summary, "best_sizing_sharpe_15m"),
        risk_discipline_score=_float(summary, "best_sizing_risk_score"),
        fold_contribution=_float(summary, "largest_positive_fold_contribution"),
        promotion_status=summary.get("promotion_readiness", ""),
        promotion_reason=summary.get("promotion_reason", ""),
        fx_multiplier=_optional_float(best_row, "fx_multiplier"),
        metal_multiplier=_optional_float(best_row, "metal_multiplier"),
        crypto_profile=best_row.get("crypto_profile", ""),
        crypto_allowed_utc_hours=best_row.get("crypto_allowed_utc_hours", ""),
        strategy_map=best_row.get("strategy_map", ""),
        multiplier_map=best_row.get("multiplier_map", ""),
    )


def _stable_profile(
    *,
    slot: str,
    use_case: str,
    reason: str,
    data_source: str,
    rows: tuple[dict[str, str], ...],
    key,
) -> DeploymentProfile | None:
    stable_rows = tuple(
        row for row in rows if row.get("stability_status") == "STABLE_PROFILE"
    )
    if not stable_rows:
        return None
    row = max(stable_rows, key=key)
    return DeploymentProfile(
        slot=slot,
        label=row.get("label", ""),
        evidence_status=_evidence_status(
            data_source=data_source,
            promotion_status=row.get("promotion_status", ""),
            fold_contribution=_float(row, "wf_largest_positive_fold_contribution"),
        ),
        use_case=use_case,
        reason=reason,
        data_source=data_source,
        return_pct=_float(row, "return_pct"),
        max_drawdown_pct=_float(row, "max_drawdown_pct"),
        sharpe_15m=_float(row, "sharpe_15m"),
        risk_discipline_score=_float(row, "risk_discipline_score"),
        fold_contribution=_float(row, "wf_largest_positive_fold_contribution"),
        promotion_status=row.get("promotion_status", ""),
        promotion_reason=row.get("promotion_reason", ""),
        fx_multiplier=_optional_float(row, "fx_multiplier"),
        metal_multiplier=_optional_float(row, "metal_multiplier"),
        crypto_profile=row.get("crypto_profile", ""),
        crypto_allowed_utc_hours=row.get("crypto_allowed_utc_hours", ""),
        strategy_map=row.get("strategy_map", ""),
        multiplier_map=row.get("multiplier_map", ""),
    )


def _recommendation(
    *,
    data_source: str,
    summary: dict[str, str],
    aggressive: DeploymentProfile,
    conservative: DeploymentProfile | None,
) -> tuple[str, str]:
    if data_source != "official":
        return (
            "paper_only",
            f"{data_source or 'unknown'} data cannot justify live deployment",
        )
    if summary.get("live_ready", "").lower() == "true":
        return "aggressive", "primary profile passed official live-ready gates"
    if conservative is not None and conservative.evidence_status == "LIVE_CANDIDATE":
        return "conservative", "primary is not live-ready; stable backup passed gates"
    return "paper_only", "no official-data profile passed live deployment gates"


def _evidence_status(
    *,
    data_source: str,
    promotion_status: str,
    fold_contribution: float,
) -> str:
    if data_source != "official":
        return "PAPER_ONLY"
    if promotion_status in {"LIVE_READY", "PROMOTE"} and fold_contribution <= 0.80:
        return "LIVE_CANDIDATE"
    if promotion_status == "REJECT":
        return "REJECT"
    return "PAPER_ONLY"


def _read_one_row(path: Path) -> dict[str, str]:
    rows = _read_rows(path)
    if not rows:
        raise ValueError(f"{path} produced no rows")
    return rows[0]


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return tuple(csv.DictReader(handle))


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    raw = row.get(key, "")
    if raw == "":
        return default
    return float(raw)


def _optional_float(row: dict[str, str], key: str) -> float | None:
    raw = row.get(key, "")
    if raw == "":
        return None
    return float(raw)
