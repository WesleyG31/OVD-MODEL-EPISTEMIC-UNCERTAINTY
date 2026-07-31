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

from .features import criticality_descriptors, detection_feature_bundle
from .manifest import validate_manifest


FEATURE_SCHEMA_VERSION = 1
EXTRACTION_SOURCE_PATHS = (
    "src/adas_ovd/config.py",
    "src/adas_ovd/data.py",
    "src/adas_ovd/matching.py",
    "src/adas_ovd/mc_features.py",
    "src/adas_ovd/reproducibility.py",
    "src/adas_ovd/shared_extraction.py",
    "RQ5/src/rq5/extraction.py",
    "RQ5/src/rq5/features.py",
    "RQ5/src/rq5/manifest.py",
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
CRITICALITY_COLUMNS = [
    "criticality_class_severity",
    "criticality_bottomness",
    "criticality_centrality",
    "criticality_geometry_factor",
    "criticality_weight",
    "criticality_tier",
]


def uncertainty_feature_names(
    config: dict[str, Any], mc_passes: int | None = None
) -> list[str]:
    passes = int(
        config["rq5"]["extraction"]["mc_operating_passes"]
        if mc_passes is None
        else mc_passes
    )
    return [
        f"{name}_mc{passes:02d}"
        for name in config["rq5"]["feature_groups"]["uncertainty"]
    ]


def all_inference_features(config: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for count in config["rq5"]["extraction"]["mc_sensitivity_passes"]:
        names.extend(uncertainty_feature_names(config, int(count)))
    return list(dict.fromkeys(names))


def feature_schema(config: dict[str, Any]) -> list[str]:
    columns = COMMON_COLUMNS + CRITICALITY_COLUMNS + all_inference_features(config)
    if len(columns) != len(set(columns)):
        raise RuntimeError("RQ5 feature schema contains duplicate columns")
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
        key = "evaluation_annotations" if split == "test" else "calibration_annotations"
        annotation_sha256 = sha256_file(project_path(config, config["data"][key]))
    source_sha256 = source_tree_sha256(
        config["_meta"]["project_root"], EXTRACTION_SOURCE_PATHS
    )
    configuration = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "rq": "rq5",
        "match_iou": config["evaluation"]["match_iou"],
        "feature_groups": config["rq5"]["feature_groups"],
        "criticality": config["rq5"]["criticality"],
        "extraction": config["rq5"]["extraction"],
        "feature_schema": feature_schema(config),
        "shared_fingerprint": shared_fingerprint,
        "shared_namespace": shared_namespace,
        "source_tree_sha256": source_sha256,
        "manifest_specification": config["rq5"]["manifest"],
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


def _materialize_shared_image(
    shared_split: SharedSplit, image_id: int, config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    started = time.perf_counter()
    common = materialize_common(shared_split, image_id, config)
    arrays = common.arrays
    rows = common.frame.to_dict(orient="records")
    image = shared_split.dataset.images[image_id]
    sensitivity_passes = sorted(
        int(value) for value in config["rq5"]["extraction"]["mc_sensitivity_passes"]
    )
    total_passes = int(config["rq5"]["extraction"]["mc_passes"])
    if (
        not sensitivity_passes
        or any(value < 1 or value > total_passes for value in sensitivity_passes)
        or int(config["rq5"]["extraction"]["mc_operating_passes"])
        not in sensitivity_passes
    ):
        raise ValueError("RQ5 MC prefix counts are incompatible with mc_passes")

    for detection_index, row in enumerate(rows):
        box = arrays["reference_boxes_xyxy"][detection_index]
        row.update(
            criticality_descriptors(
                box,
                width=image.width,
                height=image.height,
                category_name=str(row["category_name"]),
                specification=config["rq5"]["criticality"],
            )
        )
        arguments = {
            "reference_box_cxcywh": arrays["reference_boxes_cxcywh"][detection_index],
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
            "deterministic_reference_step": arrays["deterministic_reference_step"][
                detection_index
            ],
            "deterministic_hidden_step": arrays["deterministic_hidden_step"][
                detection_index
            ],
            "bbox_area_fraction": float(row["bbox_area"])
            / float(image.width * image.height),
        }
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
            bundle = detection_feature_bundle(**prefix_arguments)
            for name, value in bundle.items():
                row[f"{name}_mc{count:02d}"] = value

    frame = pd.DataFrame(rows) if rows else _empty_frame(config)
    expected = feature_schema(config)
    missing = sorted(set(expected) - set(frame.columns))
    if missing:
        raise RuntimeError(f"RQ5 materialization omitted columns: {missing}")
    frame = frame[expected]
    aggregation_seconds = time.perf_counter() - started
    weights = frame["criticality_weight"].to_numpy(dtype=np.float64)
    summary = dict(common.image_summary)
    summary.update(
        {
            "mean_criticality_weight": (
                float(weights.mean()) if len(weights) else None
            ),
            "high_criticality_detections": int(
                (frame["criticality_tier"] == "high").sum()
            ),
            "aggregation_seconds": aggregation_seconds,
            "total_seconds": float(common.image_summary["shared_inference_seconds"])
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
            and metadata.get("feature_shard_sha256") == sha256_file(detection_path)
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
    for path in (detection_path, image_summary_path, metadata_path):
        path.parent.mkdir(parents=True, exist_ok=True)
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
        raise ValueError(f"Unknown RQ5 split: {split}")
    validate_consumer_compatibility(config, "rq5")
    manifest_path, _ = validate_manifest(config)
    output_path = (
        Path(output_override).resolve()
        if output_override is not None
        else project_path(config, config["rq5"]["outputs"][f"{split}_features"])
    )
    image_output_path = (
        output_path.with_name(f"{output_path.stem}_images.parquet")
        if output_override is not None
        else project_path(
            config, config["rq5"]["outputs"][f"{split}_image_summary"]
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
        shared_split.image_ids, desc=f"RQ5 materialization [{split}]"
    ):
        detection_path = detection_root / f"{image_id}.parquet"
        image_summary_path = image_root / f"{image_id}.parquet"
        metadata_path = metadata_root / f"{image_id}.json"
        shared_sha256 = shared_split.shard_metadata[image_id]["shard_sha256"]
        if _valid_feature_shard(
            detection_path,
            image_summary_path,
            metadata_path,
            image_id=image_id,
            materialization_fingerprint=fingerprint,
            source_tree_sha256_value=identity["source_tree_sha256"],
            shared_shard_sha256=shared_sha256,
            expected_schema=expected_schema,
        ):
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
    frames: list[pd.DataFrame] = []
    image_frames: list[pd.DataFrame] = []
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
            raise RuntimeError(f"Invalid or missing RQ5 feature shard: {image_id}")
        with metadata_path.open("r", encoding="utf-8") as handle:
            inventory.append(json.load(handle))
        frames.append(pd.read_parquet(detection_path))
        image_frames.append(pd.read_parquet(image_summary_path))

    combined = pd.concat(frames, ignore_index=True) if frames else _empty_frame(config)
    image_summaries = pd.concat(image_frames, ignore_index=True)
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
        "shared_shard_inventory_sha256": shared_request["shard_inventory_sha256"],
        "shared_shards_computed": shared_request["shards_computed"],
        "shared_shards_reused": shared_request["shards_reused"],
        "random_seeds": {
            "project_seed": int(config["project"]["seed"]),
            "deterministic_image_seed": "project_seed + image_id",
            "mc_seed_stride": int(config["shared_extraction"]["mc_seed_stride"]),
            "mc_pass_seed": "project_seed + image_id + (pass_index + 1) * mc_seed_stride",
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
        raise FileNotFoundError(f"RQ5 feature artifact is incomplete: {feature_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    split = metadata.get("split")
    if split not in {"train", "validation", "test"}:
        raise RuntimeError("RQ5 feature metadata has an invalid split")
    expected = _materialization_identity(config, split)
    for key, expected_value in expected.items():
        if metadata.get(key) != expected_value:
            raise RuntimeError(f"RQ5 feature artifact is stale ({key}): {feature_path}")
    if metadata.get("features_sha256") != sha256_file(feature_path):
        raise RuntimeError(f"RQ5 feature SHA-256 mismatch: {feature_path}")
    receipt_path = Path(metadata.get("shared_request_metadata_path", ""))
    if (
        not receipt_path.is_file()
        or metadata.get("shared_request_metadata_sha256")
        != sha256_file(receipt_path)
    ):
        raise RuntimeError("RQ5 shared extraction receipt integrity check failed")
    image_summary_path = Path(metadata["image_summary_path"])
    if (
        not image_summary_path.is_file()
        or metadata.get("image_summary_sha256") != sha256_file(image_summary_path)
    ):
        raise RuntimeError("RQ5 image-summary integrity check failed")
    frame = pd.read_parquet(feature_path)
    if list(frame.columns) != metadata.get("feature_schema"):
        raise RuntimeError("RQ5 feature schema differs from frozen metadata")
    return frame, metadata

