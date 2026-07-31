from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from adas_ovd.config import project_path
from adas_ovd.reproducibility import (
    environment_metadata,
    sha256_file,
    source_tree_sha256,
    stable_fingerprint,
    write_json,
)
from adas_ovd.shared_extraction import (
    SharedSplit,
    ensure_shared_split,
    materialize_common,
    shared_identity,
    validate_consumer_compatibility,
)

from .features import (
    class_agnostic_localization_targets,
    detection_feature_bundle,
)
from .manifest import validate_manifest


FEATURE_SCHEMA_VERSION = 1
EXTRACTION_SOURCE_PATHS = (
    "src/adas_ovd/config.py",
    "src/adas_ovd/data.py",
    "src/adas_ovd/matching.py",
    "src/adas_ovd/mc_features.py",
    "src/adas_ovd/reproducibility.py",
    "src/adas_ovd/shared_extraction.py",
    "RQ3/src/rq3/extraction.py",
    "RQ3/src/rq3/features.py",
    "RQ3/src/rq3/manifest.py",
)

COMMON_COLUMNS = [
    "image_id",
    "file_name",
    "sequence_id",
    "timeofday",
    "weather",
    "scene",
    "detection_index",
    "query_index",
    "category_index",
    "category_id",
    "category_name",
    "score",
    "confidence_uncertainty",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "bbox_area",
    "object_size",
    "is_true_positive",
    "is_error",
    "matched_iou",
    "matched_ground_truth_index",
    "false_negatives_image",
    "mc_matches",
    "mc_passes",
]

TARGET_COLUMNS = [
    "localization_iou",
    "localization_ground_truth_index",
    "localization_class_agreement",
    "is_well_localized",
]


def spatial_feature_names(
    config: dict[str, Any], mc_passes: int | None = None
) -> list[str]:
    groups = config["rq3"]["feature_groups"]
    mc = list(groups["spatial_mc"])
    if mc_passes is not None:
        mc = [f"{name}_mc{int(mc_passes):02d}" for name in mc]
    return mc + list(groups["spatial_static"])


def nonspatial_feature_names(config: dict[str, Any]) -> list[str]:
    return list(config["rq3"]["feature_groups"]["nonspatial_control"])


def _target_threshold_columns(config: dict[str, Any]) -> list[str]:
    return [
        f"is_well_localized_{int(round(float(value) * 100)):03d}"
        for value in config["rq3"]["targets"]["localization_iou_sensitivity"]
    ]


def _all_feature_names(config: dict[str, Any]) -> list[str]:
    main = spatial_feature_names(config) + nonspatial_feature_names(config)
    sensitivity = [
        f"{name}_mc{int(count):02d}"
        for count in config["rq3"]["extraction"]["mc_sensitivity_passes"]
        for name in config["rq3"]["feature_groups"]["spatial_mc"]
    ]
    return main + sensitivity


def feature_schema(config: dict[str, Any]) -> list[str]:
    columns = (
        COMMON_COLUMNS
        + TARGET_COLUMNS
        + _target_threshold_columns(config)
        + _all_feature_names(config)
    )
    if len(columns) != len(set(columns)):
        raise RuntimeError("RQ3 feature schema contains duplicate columns")
    return columns


def _empty_frame(config: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(columns=feature_schema(config))


def _materialization_identity(
    config: dict[str, Any],
    split: str,
    *,
    annotation_sha256: str | None = None,
    shared_fingerprint: str | None = None,
    shared_namespace: str | None = None,
) -> dict[str, Any]:
    manifest_path, manifest = validate_manifest(config)
    shared = shared_identity(config)
    shared_fingerprint = shared_fingerprint or shared["configuration_fingerprint"]
    shared_namespace = shared_namespace or str(
        config["shared_extraction"]["cache_namespace"]
    )
    if annotation_sha256 is None:
        annotation_key = (
            "evaluation_annotations" if split == "test" else "calibration_annotations"
        )
        annotation_sha256 = sha256_file(
            project_path(config, config["data"][annotation_key])
        )
    source_sha256 = source_tree_sha256(
        config["_meta"]["project_root"], EXTRACTION_SOURCE_PATHS
    )
    configuration = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "rq": "rq3",
        "match_iou": config["evaluation"]["match_iou"],
        "targets": config["rq3"]["targets"],
        "feature_groups": config["rq3"]["feature_groups"],
        "mc_sensitivity_passes": config["rq3"]["extraction"][
            "mc_sensitivity_passes"
        ],
        "feature_schema": feature_schema(config),
        "shared_fingerprint": shared_fingerprint,
        "shared_namespace": shared_namespace,
        "source_tree_sha256": source_sha256,
        "manifest_specification": config["rq3"]["manifest"],
    }
    identity: dict[str, Any] = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "split": split,
        "source_tree_sha256": source_sha256,
        "configuration_fingerprint": stable_fingerprint(configuration),
        "checkpoint_sha256": shared["checkpoint_sha256"],
        "shared_fingerprint": shared_fingerprint,
        "shared_namespace": shared_namespace,
        "annotation_sha256": annotation_sha256,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_test_partition": manifest["test_partition"],
        "split_image_ids_sha256": stable_fingerprint(manifest["splits"][split]),
        "feature_schema": feature_schema(config),
    }
    identity["materialization_fingerprint"] = stable_fingerprint(identity)
    return identity


def _ground_truth_arrays(shared_split: SharedSplit, image_id: int):
    records = shared_split.dataset.ground_truth(image_id)
    boxes = np.asarray(
        [record.bbox_xyxy for record in records], dtype=np.float64
    ).reshape(-1, 4)
    categories = np.asarray(
        [record.category_id for record in records], dtype=np.int64
    )
    return records, boxes, categories


def _materialize_shared_image(
    shared_split: SharedSplit,
    image_id: int,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    started = time.perf_counter()
    common = materialize_common(shared_split, image_id, config)
    arrays = common.arrays
    rows = common.frame.to_dict(orient="records")
    image = shared_split.dataset.images[image_id]
    _, ground_truth_boxes, ground_truth_categories = _ground_truth_arrays(
        shared_split, image_id
    )
    category_ids = np.asarray(
        [
            shared_split.dataset.category_ids_by_name[name]
            for name in config["data"]["classes"]
        ],
        dtype=np.int64,
    )
    prediction_categories = category_ids[arrays["reference_category_indices"]]
    thresholds = [
        float(value)
        for value in config["rq3"]["targets"]["localization_iou_sensitivity"]
    ]
    primary_threshold = float(
        config["rq3"]["targets"]["localization_iou_threshold"]
    )
    if primary_threshold not in thresholds:
        raise RuntimeError("Primary localization threshold is absent from sensitivity")
    targets = class_agnostic_localization_targets(
        arrays["reference_boxes_xyxy"],
        prediction_categories,
        ground_truth_boxes,
        ground_truth_categories,
        thresholds,
    )
    primary_target_name = (
        f"is_well_localized_{int(round(primary_threshold * 100)):03d}"
    )
    sensitivity_passes = sorted(
        int(value)
        for value in config["rq3"]["extraction"]["mc_sensitivity_passes"]
    )
    total_passes = int(config["rq3"]["extraction"]["mc_passes"])
    if (
        not sensitivity_passes
        or any(value < 1 or value > total_passes for value in sensitivity_passes)
    ):
        raise ValueError("RQ3 MC sensitivity counts must lie in [1, mc_passes]")

    for detection_index, row in enumerate(rows):
        row["localization_iou"] = float(
            targets["localization_iou"][detection_index]
        )
        row["localization_ground_truth_index"] = int(
            targets["localization_ground_truth_index"][detection_index]
        )
        row["localization_class_agreement"] = bool(
            targets["localization_class_agreement"][detection_index]
        )
        for target_name in _target_threshold_columns(config):
            row[target_name] = int(targets[target_name][detection_index])
        row["is_well_localized"] = int(row[primary_target_name])
        bbox_area_fraction = float(row["bbox_area"]) / float(
            image.width * image.height
        )
        arguments = {
            "reference_box_cxcywh": arrays["reference_boxes_cxcywh"][
                detection_index
            ],
            "category_scores": arrays["mc_category_scores"][:, detection_index],
            "scores": arrays["mc_scores"][:, detection_index],
            "boxes_cxcywh": arrays["mc_boxes_cxcywh"][:, detection_index],
            "embeddings": arrays["mc_embeddings"][:, detection_index],
            "present": arrays["present"][:, detection_index],
            "base_category": int(
                arrays["reference_category_indices"][detection_index]
            ),
            "deterministic_reference_variance": arrays[
                "deterministic_reference_variance"
            ][detection_index],
            "deterministic_reference_step": arrays[
                "deterministic_reference_step"
            ][detection_index],
            "deterministic_hidden_step": arrays["deterministic_hidden_step"][
                detection_index
            ],
            "bbox_area_fraction": bbox_area_fraction,
        }
        row.update(detection_feature_bundle(**arguments))
        for count in sensitivity_passes:
            prefix_arguments = dict(arguments)
            for key in (
                "category_scores",
                "scores",
                "boxes_cxcywh",
                "embeddings",
                "present",
            ):
                prefix_arguments[key] = prefix_arguments[key][:count]
            prefix = detection_feature_bundle(**prefix_arguments)
            row.update(
                {
                    f"{name}_mc{count:02d}": prefix[name]
                    for name in config["rq3"]["feature_groups"]["spatial_mc"]
                }
            )

    frame = pd.DataFrame(rows) if rows else _empty_frame(config)
    expected = feature_schema(config)
    missing = sorted(set(expected) - set(frame.columns))
    if missing:
        raise RuntimeError(f"RQ3 materialization omitted columns: {missing}")
    frame = frame[expected]
    aggregation_seconds = time.perf_counter() - started
    summary = dict(common.image_summary)
    summary.update(
        {
            "well_localized_detections": int(
                targets[primary_target_name].sum()
            ),
            "mean_localization_iou": (
                float(targets["localization_iou"].mean())
                if len(targets["localization_iou"])
                else None
            ),
            "localization_iou_threshold": primary_threshold,
            "aggregation_seconds": aggregation_seconds,
            "total_seconds": float(summary["shared_inference_seconds"])
            + aggregation_seconds,
        }
    )
    return frame, summary


def _valid_feature_shard(
    detection_path: Path,
    image_summary_path: Path,
    metadata_path: Path,
    *,
    image_id: int,
    materialization_fingerprint: str,
    source_tree_sha256_value: str,
    shared_shard_sha256: str,
    expected_schema: list[str],
) -> bool:
    if not (
        detection_path.is_file()
        and image_summary_path.is_file()
        and metadata_path.is_file()
    ):
        return False
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        return bool(
            int(metadata.get("schema_version", -1)) == FEATURE_SCHEMA_VERSION
            and int(metadata.get("image_id", -1)) == int(image_id)
            and metadata.get("materialization_fingerprint")
            == materialization_fingerprint
            and metadata.get("source_tree_sha256") == source_tree_sha256_value
            and metadata.get("shared_shard_sha256") == shared_shard_sha256
            and metadata.get("feature_schema") == expected_schema
            and metadata.get("feature_shard_sha256")
            == sha256_file(detection_path)
            and metadata.get("image_summary_sha256")
            == sha256_file(image_summary_path)
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _write_feature_shard(
    detection_path: Path,
    image_summary_path: Path,
    metadata_path: Path,
    frame: pd.DataFrame,
    image_summary: dict[str, Any],
    *,
    materialization_fingerprint: str,
    source_tree_sha256_value: str,
    shared_shard_sha256: str,
    expected_schema: list[str],
) -> None:
    detection_path.parent.mkdir(parents=True, exist_ok=True)
    image_summary_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_detection = detection_path.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary_detection, index=False)
    temporary_detection.replace(detection_path)
    temporary_summary = image_summary_path.with_suffix(".parquet.tmp")
    pd.DataFrame([image_summary]).to_parquet(temporary_summary, index=False)
    temporary_summary.replace(image_summary_path)
    write_json(
        metadata_path,
        {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "image_id": int(image_summary["image_id"]),
            "rows": int(len(frame)),
            "materialization_fingerprint": materialization_fingerprint,
            "source_tree_sha256": source_tree_sha256_value,
            "shared_shard_sha256": shared_shard_sha256,
            "feature_schema": expected_schema,
            "feature_shard_sha256": sha256_file(detection_path),
            "image_summary_sha256": sha256_file(image_summary_path),
        },
    )


def extract_split(
    config: dict[str, Any],
    split: str,
    limit: int | None = None,
    output_override: str | Path | None = None,
    shared_cache_namespace: str | None = None,
) -> Path:
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unknown RQ3 split: {split}")
    validate_consumer_compatibility(config, "rq3")
    manifest_path, _ = validate_manifest(config)
    output_path = (
        Path(output_override).resolve()
        if output_override is not None
        else project_path(config, config["rq3"]["outputs"][f"{split}_features"])
    )
    image_output_path = (
        output_path.with_name(f"{output_path.stem}_images.parquet")
        if output_override is not None
        else project_path(
            config, config["rq3"]["outputs"][f"{split}_image_summary"]
        )
    )
    shared_split = ensure_shared_split(
        config,
        manifest_path=manifest_path,
        split=split,
        limit=limit,
        cache_namespace=shared_cache_namespace,
    )
    identity = _materialization_identity(
        config,
        split,
        annotation_sha256=shared_split.dataset.sha256,
        shared_fingerprint=shared_split.fingerprint,
        shared_namespace=shared_split.namespace,
    )
    fingerprint = identity["materialization_fingerprint"]
    expected_schema = feature_schema(config)
    root = output_path.parent
    detection_root = root / "shards" / split / fingerprint
    image_root = root / "image_shards" / split / fingerprint
    metadata_root = root / "shard_metadata" / split / fingerprint
    for directory in (detection_root, image_root, metadata_root):
        directory.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    reused = 0
    recomputed = 0
    for image_id in tqdm(
        shared_split.image_ids, desc=f"RQ3 feature materialization [{split}]"
    ):
        detection_path = detection_root / f"{image_id}.parquet"
        image_summary_path = image_root / f"{image_id}.parquet"
        metadata_path = metadata_root / f"{image_id}.json"
        shared_sha256 = shared_split.shard_metadata[image_id]["shard_sha256"]
        valid = _valid_feature_shard(
            detection_path,
            image_summary_path,
            metadata_path,
            image_id=image_id,
            materialization_fingerprint=fingerprint,
            source_tree_sha256_value=identity["source_tree_sha256"],
            shared_shard_sha256=shared_sha256,
            expected_schema=expected_schema,
        )
        if valid:
            reused += 1
            continue
        frame, summary = _materialize_shared_image(shared_split, image_id, config)
        _write_feature_shard(
            detection_path,
            image_summary_path,
            metadata_path,
            frame,
            summary,
            materialization_fingerprint=fingerprint,
            source_tree_sha256_value=identity["source_tree_sha256"],
            shared_shard_sha256=shared_sha256,
            expected_schema=expected_schema,
        )
        recomputed += 1

    inventory: list[dict[str, Any]] = []
    invalid: list[int] = []
    for image_id in shared_split.image_ids:
        detection_path = detection_root / f"{image_id}.parquet"
        image_summary_path = image_root / f"{image_id}.parquet"
        metadata_path = metadata_root / f"{image_id}.json"
        if not _valid_feature_shard(
            detection_path,
            image_summary_path,
            metadata_path,
            image_id=image_id,
            materialization_fingerprint=fingerprint,
            source_tree_sha256_value=identity["source_tree_sha256"],
            shared_shard_sha256=shared_split.shard_metadata[image_id][
                "shard_sha256"
            ],
            expected_schema=expected_schema,
        ):
            invalid.append(image_id)
            continue
        with metadata_path.open("r", encoding="utf-8") as handle:
            inventory.append(json.load(handle))
    if invalid:
        raise RuntimeError(f"Invalid or missing RQ3 feature shards: {invalid[:10]}")

    frames = [
        pd.read_parquet(detection_root / f"{image_id}.parquet")
        for image_id in shared_split.image_ids
    ]
    combined = pd.concat(frames, ignore_index=True) if frames else _empty_frame(config)
    image_summaries = pd.concat(
        [
            pd.read_parquet(image_root / f"{image_id}.parquet")
            for image_id in shared_split.image_ids
        ],
        ignore_index=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_suffix(".parquet.tmp")
    combined.to_parquet(temporary_output, index=False)
    temporary_output.replace(output_path)
    image_output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_images = image_output_path.with_suffix(".parquet.tmp")
    image_summaries.to_parquet(temporary_images, index=False)
    temporary_images.replace(image_output_path)

    with shared_split.request_metadata_path.open("r", encoding="utf-8") as handle:
        shared_request = json.load(handle)
    metadata = {
        **identity,
        "images_requested": len(shared_split.image_ids),
        "images_completed": len(image_summaries),
        "detections": len(combined),
        "features_sha256": sha256_file(output_path),
        "image_summary_path": str(image_output_path),
        "image_summary_sha256": sha256_file(image_output_path),
        "feature_shard_inventory_sha256": stable_fingerprint(inventory),
        "feature_shards_reused": reused,
        "feature_shards_recomputed": recomputed,
        "elapsed_seconds": time.perf_counter() - started,
        "shared_request_metadata_path": str(shared_split.request_metadata_path),
        "shared_request_metadata_sha256": sha256_file(
            shared_split.request_metadata_path
        ),
        "shared_shard_inventory_sha256": shared_request[
            "shard_inventory_sha256"
        ],
        "shared_shards_computed": shared_request["shards_computed"],
        "shared_shards_reused": shared_request["shards_reused"],
        "random_seeds": {
            "project_seed": int(config["project"]["seed"]),
            "deterministic_image_seed": "project_seed + image_id",
            "mc_seed_stride": int(config["shared_extraction"]["mc_seed_stride"]),
            "mc_pass_seed": (
                "project_seed + image_id + (pass_index + 1) * mc_seed_stride"
            ),
        },
        "timing_seconds": {
            name: float(image_summaries[name].sum())
            for name in (
                "preprocess_seconds",
                "deterministic_seconds",
                "stochastic_seconds",
                "aggregation_seconds",
            )
        },
        "peak_gpu_memory_bytes": int(
            image_summaries["peak_gpu_memory_bytes"].max()
        ),
        "stochastic_modules": (
            shared_split.shard_metadata[shared_split.image_ids[0]][
                "enabled_stochastic_modules"
            ]
            if shared_split.image_ids
            else []
        ),
        "environment": environment_metadata(config["_meta"]["project_root"]),
        "diagnostic_only": identity["manifest_test_partition"] == "diagnostic",
    }
    write_json(output_path.with_suffix(".metadata.json"), metadata)
    return output_path


def read_validated_features(
    config: dict[str, Any], path: str | Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    feature_path = Path(path).resolve()
    metadata_path = feature_path.with_suffix(".metadata.json")
    if not feature_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"RQ3 feature artifact is incomplete: {feature_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    split = metadata.get("split")
    if split not in {"train", "validation", "test"}:
        raise RuntimeError("RQ3 feature metadata has an invalid split")
    expected = _materialization_identity(config, split)
    for key, expected_value in expected.items():
        if metadata.get(key) != expected_value:
            raise RuntimeError(
                f"RQ3 feature artifact is stale ({key} changed): {feature_path}"
            )
    if metadata.get("features_sha256") != sha256_file(feature_path):
        raise RuntimeError(f"RQ3 feature SHA-256 mismatch: {feature_path}")
    receipt_value = metadata.get("shared_request_metadata_path")
    receipt_hash = metadata.get("shared_request_metadata_sha256")
    if not receipt_value or not receipt_hash:
        raise RuntimeError("RQ3 shared extraction receipt is missing")
    receipt_path = Path(receipt_value)
    if not receipt_path.is_file() or receipt_hash != sha256_file(receipt_path):
        raise RuntimeError("RQ3 shared extraction receipt integrity check failed")
    image_summary_path = Path(metadata["image_summary_path"])
    if (
        not image_summary_path.is_file()
        or metadata.get("image_summary_sha256")
        != sha256_file(image_summary_path)
    ):
        raise RuntimeError("RQ3 image-summary integrity check failed")
    frame = pd.read_parquet(feature_path)
    if list(frame.columns) != metadata.get("feature_schema"):
        raise RuntimeError("RQ3 feature schema differs from frozen metadata")
    return frame, metadata
