from __future__ import annotations

from pathlib import Path
from typing import Any

from adas_ovd.config import project_path
from adas_ovd.data import CocoDataset, create_sequence_manifest
from adas_ovd.data_preparation import require_passing_data_audit
from adas_ovd.reproducibility import sha256_file, write_json


def _partition_evaluation(
    manifest: dict[str, Any],
    evaluation: CocoDataset,
    requested_diagnostic_ids: list[int],
    test_partition: str,
) -> dict[str, Any]:
    if test_partition not in {"confirmatory", "diagnostic"}:
        raise ValueError(
            "rq1.manifest.test_partition must be confirmatory or diagnostic"
        )
    available_ids = set(evaluation.images)
    requested = sorted({int(value) for value in requested_diagnostic_ids})
    missing = sorted(set(requested) - available_ids)
    if missing:
        raise ValueError(
            "Diagnostic image IDs are absent from evaluation annotations: "
            f"{missing}"
        )
    diagnostic_groups = {
        evaluation.images[image_id].sequence_id for image_id in requested
    }
    diagnostic_ids = sorted(
        image_id
        for image_id, record in evaluation.images.items()
        if record.sequence_id in diagnostic_groups
    )
    confirmatory_ids = sorted(available_ids - set(diagnostic_ids))
    confirmatory_groups = {
        evaluation.images[image_id].sequence_id
        for image_id in confirmatory_ids
    }
    if diagnostic_groups & confirmatory_groups:
        raise RuntimeError(
            "Diagnostic/confirmatory evaluation group leakage detected"
        )
    if not diagnostic_ids or not confirmatory_ids:
        raise ValueError(
            "Diagnostic and confirmatory evaluation partitions must be non-empty"
        )

    manifest["schema_version"] = 4
    manifest["test_partition"] = test_partition
    manifest["diagnostic_requested_image_ids"] = requested
    manifest["splits"]["diagnostic_test"] = diagnostic_ids
    manifest["splits"]["confirmatory_test"] = confirmatory_ids
    manifest["splits"]["test"] = (
        diagnostic_ids
        if test_partition == "diagnostic"
        else confirmatory_ids
    )
    manifest["sequence_counts"]["diagnostic_test"] = len(
        diagnostic_groups
    )
    manifest["sequence_counts"]["confirmatory_test"] = len(
        confirmatory_groups
    )
    manifest["sequence_counts"]["test"] = len(
        diagnostic_groups
        if test_partition == "diagnostic"
        else confirmatory_groups
    )
    manifest["group_overlap_counts"]["diagnostic_confirmatory_test"] = 0
    return manifest


def make_manifest(config: dict[str, Any], force: bool = False) -> Path:
    require_passing_data_audit(config)
    output = project_path(config, config["rq1"]["outputs"]["manifest"])
    data = config["data"]
    calibration_path = project_path(
        config, data["calibration_annotations"]
    )
    evaluation_path = project_path(config, data["evaluation_annotations"])
    manifest_config = config["rq1"]["manifest"]
    requested_diagnostic_ids = [
        int(value)
        for value in manifest_config.get(
            "diagnostic_evaluation_image_ids", []
        )
    ]
    test_partition = str(
        manifest_config.get("test_partition", "confirmatory")
    )
    if output.exists() and not force:
        import json

        with output.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        reusable = (
            int(existing.get("schema_version", 0)) >= 4
            and int(existing.get("seed", -1))
            == int(config["project"]["seed"])
            and float(existing.get("train_fraction", -1.0))
            == float(config["rq1"]["manifest"]["train_fraction"])
            and existing.get("diagnostic_requested_image_ids")
            == sorted(set(requested_diagnostic_ids))
            and existing.get("test_partition") == test_partition
            and existing.get("calibration_sha256")
            == sha256_file(calibration_path)
            and existing.get("evaluation_sha256")
            == sha256_file(evaluation_path)
        )
        if reusable:
            return output
    images_dir = project_path(config, data["images_dir"])
    calibration = CocoDataset(
        calibration_path, images_dir
    )
    evaluation = CocoDataset(
        evaluation_path, images_dir
    )
    manifest = create_sequence_manifest(
        calibration=calibration,
        evaluation=evaluation,
        train_fraction=float(config["rq1"]["manifest"]["train_fraction"]),
        seed=int(config["project"]["seed"]),
        project_root=config["_meta"]["project_root"],
    )
    manifest = _partition_evaluation(
        manifest,
        evaluation,
        requested_diagnostic_ids,
        test_partition,
    )
    write_json(output, manifest)
    return output
