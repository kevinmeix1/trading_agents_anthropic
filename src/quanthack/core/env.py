from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: str | Path = ".env") -> dict[str, str]:
    env_path = Path(path)
    if not env_path.exists():
        return {}

    loaded: dict[str, str] = {}
    for line_number, raw_line in enumerate(env_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].strip()
        if "=" not in stripped:
            raise ValueError(f"{env_path}:{line_number} is not KEY=VALUE")
        key, raw_value = stripped.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"{env_path}:{line_number} has an empty key")
        value = _clean_env_value(raw_value.strip())
        loaded[key] = value
        os.environ.setdefault(key, value)
    return loaded


def env_str(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def env_int(name: str, default: int | None = None) -> int | None:
    value = env_str(name)
    if value is None:
        return default
    return int(value)


def env_bool(name: str, default: bool = False) -> bool:
    value = env_str(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _clean_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
