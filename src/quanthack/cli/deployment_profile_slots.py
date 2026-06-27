from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path


def resolve_profile_slots(
    *,
    requested_slots: Sequence[str] | None,
    profile_pack_json: str | Path,
    fallback_slots: Sequence[str],
) -> tuple[str, ...]:
    if requested_slots:
        return _unique_nonblank("slots", requested_slots)
    pack_slots = _profile_slots_from_pack(profile_pack_json)
    if pack_slots:
        return pack_slots
    return _unique_nonblank("fallback_slots", fallback_slots)


def resolve_single_fallback_slot(
    *,
    requested_fallback_slot: str | None,
    profile_pack_json: str | Path,
    slots: Sequence[str],
    preferred_slots: Sequence[str],
) -> str:
    if requested_fallback_slot is not None:
        cleaned = _unique_nonblank("fallback_slot", (requested_fallback_slot,))
        if not cleaned:
            raise ValueError("fallback_slot must include a value")
        return cleaned[0]
    candidates = _default_fallback_candidates(
        profile_pack_json=profile_pack_json,
        slots=slots,
        preferred_slots=preferred_slots,
    )
    if not candidates:
        raise ValueError("cannot choose a fallback slot from an empty slot set")
    return candidates[0]


def resolve_fallback_slots(
    *,
    requested_fallback_slots: Sequence[str] | None,
    profile_pack_json: str | Path,
    slots: Sequence[str],
    preferred_slots: Sequence[str],
) -> tuple[str, ...]:
    if requested_fallback_slots:
        return _unique_nonblank("fallback_slots", requested_fallback_slots)
    candidates = _default_fallback_candidates(
        profile_pack_json=profile_pack_json,
        slots=slots,
        preferred_slots=preferred_slots,
    )
    if not candidates:
        raise ValueError("cannot choose fallback slots from an empty slot set")
    return candidates


def _default_fallback_candidates(
    *,
    profile_pack_json: str | Path,
    slots: Sequence[str],
    preferred_slots: Sequence[str],
) -> tuple[str, ...]:
    available = set(_unique_nonblank("slots", slots))
    recommended = _recommended_slot_from_pack(profile_pack_json)
    ordered_candidates = (
        recommended,
        *preferred_slots,
        next(iter(slots), ""),
    )
    selected: list[str] = []
    for raw_candidate in ordered_candidates:
        candidate = str(raw_candidate or "").strip()
        if candidate in available and candidate not in selected:
            selected.append(candidate)
    return tuple(selected)


def _profile_slots_from_pack(profile_pack_json: str | Path) -> tuple[str, ...]:
    pack = _read_pack(profile_pack_json)
    return _unique_nonblank(
        "profile slots",
        tuple(str(profile.get("slot", "")) for profile in pack.get("profiles", ())),
    )


def _recommended_slot_from_pack(profile_pack_json: str | Path) -> str:
    return str(_read_pack(profile_pack_json).get("recommended_slot", "")).strip()


def _read_pack(profile_pack_json: str | Path) -> dict:
    return json.loads(Path(profile_pack_json).read_text(encoding="utf-8"))


def _unique_nonblank(name: str, values: Sequence[str]) -> tuple[str, ...]:
    cleaned = tuple(str(value).strip() for value in values if str(value).strip())
    if not cleaned:
        return ()
    if len(cleaned) != len(set(cleaned)):
        raise ValueError(f"{name} values must be unique")
    return cleaned
