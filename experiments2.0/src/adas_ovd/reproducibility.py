from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def seed_everything(
    seed: int,
    deterministic_algorithms: bool = False,
    deterministic_warn_only: bool = True,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic_algorithms
        torch.use_deterministic_algorithms(
            deterministic_algorithms,
            warn_only=deterministic_warn_only,
        )
    except ImportError:
        pass


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: str | Path, project_root: str | Path) -> str:
    resolved = Path(path).resolve()
    root = Path(project_root).resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.name


def stable_fingerprint(payload: Any) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def source_tree_sha256(
    project_root: str | Path,
    relative_paths: tuple[str, ...] = ("src", "RQ1/src", "scripts"),
) -> str:
    root = Path(project_root).resolve()
    digest = hashlib.sha256()
    paths: list[Path] = []
    for relative in relative_paths:
        entry = root / relative
        if entry.is_file() and entry.suffix == ".py":
            paths.append(entry)
        elif entry.is_dir():
            paths.extend(
                path for path in entry.rglob("*.py") if path.is_file()
            )
    paths.sort()
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def git_revision(project_root: str | Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(project_root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_metadata(project_root: str | Path) -> dict[str, Any]:
    names = (
        "groundingdino-py",
        "torch",
        "torchvision",
        "transformers",
        "numpy",
        "pandas",
        "scikit-learn",
        "pycocotools",
    )
    packages: dict[str, str | None] = {}
    for name in names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    cuda: dict[str, Any] = {}
    try:
        import torch

        cuda = {
            "available": torch.cuda.is_available(),
            "build": torch.version.cuda,
            "device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "cudnn": (
                torch.backends.cudnn.version()
                if torch.cuda.is_available()
                else None
            ),
            "deterministic_algorithms": (
                torch.are_deterministic_algorithms_enabled()
            ),
        }
    except ImportError:
        cuda = {"available": False, "build": None, "device": None}

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "cuda": cuda,
        "git_revision": git_revision(project_root),
    }


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
    temporary.replace(destination)
