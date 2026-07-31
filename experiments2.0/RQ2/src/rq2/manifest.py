from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from adas_ovd.config import project_path
from adas_ovd.data_preparation import require_passing_data_audit
from adas_ovd.reproducibility import sha256_file


def validate_manifest(config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    """Validate the shared frozen manifest without regenerating its split."""
    require_passing_data_audit(config)
    specification = config["rq2"]["manifest"]
    path = project_path(config, specification["path"])
    if not path.is_file():
        raise FileNotFoundError(f"Frozen shared manifest is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    if int(manifest.get("schema_version", -1)) != int(
        specification["expected_schema_version"]
    ):
        raise RuntimeError("Frozen manifest schema does not match RQ2")
    if manifest.get("test_partition") != specification["expected_test_partition"]:
        raise RuntimeError("Frozen manifest test partition does not match RQ2 mode")
    requested = sorted(
        int(value) for value in specification["expected_diagnostic_image_ids"]
    )
    if manifest.get("diagnostic_requested_image_ids") != requested:
        raise RuntimeError("Diagnostic image identity differs from the frozen RQ2 protocol")

    calibration = project_path(config, config["data"]["calibration_annotations"])
    evaluation = project_path(config, config["data"]["evaluation_annotations"])
    if manifest.get("calibration_sha256") != sha256_file(calibration):
        raise RuntimeError("Calibration annotations do not match the frozen manifest")
    if manifest.get("evaluation_sha256") != sha256_file(evaluation):
        raise RuntimeError("Evaluation annotations do not match the frozen manifest")

    splits = manifest.get("splits", {})
    required = {"train", "validation", "test", "diagnostic_test", "confirmatory_test"}
    if not required.issubset(splits):
        raise RuntimeError("Frozen manifest is missing required partitions")
    train, validation = set(splits["train"]), set(splits["validation"])
    diagnostic = set(splits["diagnostic_test"])
    confirmatory = set(splits["confirmatory_test"])
    if train & validation or diagnostic & confirmatory:
        raise RuntimeError("Image overlap detected in the frozen manifest")
    overlap_counts = manifest.get("group_overlap_counts", {})
    if any(int(value) != 0 for value in overlap_counts.values()):
        raise RuntimeError("Source-group overlap detected in the frozen manifest")
    expected_counts = {
        "train": 5600,
        "validation": 2400,
        "diagnostic_test": 8,
        "confirmatory_test": 1992,
    }
    for split, expected in expected_counts.items():
        if len(splits[split]) != expected:
            raise RuntimeError(
                f"Frozen {split} count changed: {len(splits[split])} != {expected}"
            )
    expected_test = 8 if manifest["test_partition"] == "diagnostic" else 1992
    if len(splits["test"]) != expected_test:
        raise RuntimeError("Active test partition has an unexpected size")
    return path, manifest

