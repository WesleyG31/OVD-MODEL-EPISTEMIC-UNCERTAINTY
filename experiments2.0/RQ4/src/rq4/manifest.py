from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from adas_ovd.config import project_path
from adas_ovd.data_preparation import require_passing_data_audit
from adas_ovd.reproducibility import sha256_file


def validate_manifest(config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    """Validate the immutable shared split without regenerating it."""
    require_passing_data_audit(config)
    specification = config["rq4"]["manifest"]
    path = project_path(config, specification["path"])
    if not path.is_file():
        raise FileNotFoundError(f"Frozen RQ4 manifest is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if int(manifest.get("schema_version", -1)) != int(
        specification["expected_schema_version"]
    ):
        raise RuntimeError("Frozen manifest schema does not match RQ4")
    if manifest.get("test_partition") != specification["expected_test_partition"]:
        raise RuntimeError("Frozen manifest test partition does not match RQ4 mode")
    diagnostic_ids = sorted(
        int(value) for value in specification["expected_diagnostic_image_ids"]
    )
    if manifest.get("diagnostic_requested_image_ids") != diagnostic_ids:
        raise RuntimeError("Diagnostic image identity differs from the RQ4 protocol")
    calibration = project_path(config, config["data"]["calibration_annotations"])
    evaluation = project_path(config, config["data"]["evaluation_annotations"])
    if manifest.get("calibration_sha256") != sha256_file(calibration):
        raise RuntimeError("Calibration annotations do not match the manifest")
    if manifest.get("evaluation_sha256") != sha256_file(evaluation):
        raise RuntimeError("Evaluation annotations do not match the manifest")
    splits = manifest.get("splits", {})
    expected_counts = {
        "train": 5600,
        "validation": 2400,
        "diagnostic_test": 8,
        "confirmatory_test": 1992,
    }
    if not {"train", "validation", "test", *expected_counts}.issubset(splits):
        raise RuntimeError("Frozen manifest is missing RQ4 partitions")
    for split, expected in expected_counts.items():
        if len(splits[split]) != expected:
            raise RuntimeError(f"Frozen {split} count changed")
    train = set(map(int, splits["train"]))
    validation = set(map(int, splits["validation"]))
    diagnostic = set(map(int, splits["diagnostic_test"]))
    confirmatory = set(map(int, splits["confirmatory_test"]))
    if train & validation or diagnostic & confirmatory:
        raise RuntimeError("Image overlap detected in the frozen RQ4 manifest")
    if any(int(value) for value in manifest.get("group_overlap_counts", {}).values()):
        raise RuntimeError("Source-group overlap detected in the RQ4 manifest")
    active = 8 if manifest["test_partition"] == "diagnostic" else 1992
    if len(splits["test"]) != active:
        raise RuntimeError("Active RQ4 test partition has an unexpected size")
    return path, manifest

