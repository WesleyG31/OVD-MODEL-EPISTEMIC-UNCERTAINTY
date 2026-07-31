from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, Mapping)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _load_recursive(path: Path, seen: set[Path]) -> dict[str, Any]:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"Cyclic YAML inheritance detected at {path}")
    seen.add(path)
    with path.open("r", encoding="utf-8") as handle:
        current = yaml.safe_load(handle) or {}
    parent_ref = current.pop("extends", None)
    if not parent_ref:
        return current
    parent = (path.parent / parent_ref).resolve()
    return _deep_merge(_load_recursive(parent, seen), current)


def find_project_root(start: Path) -> Path:
    start = start.resolve()
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError(f"Could not locate pyproject.toml above {start}")


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    config = _load_recursive(config_path, set())
    config["_meta"] = {
        "config_path": str(config_path),
        "project_root": str(find_project_root(config_path.parent)),
    }
    return config


def project_path(config: Mapping[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (Path(config["_meta"]["project_root"]) / path).resolve()

