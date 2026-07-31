from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from adas_ovd.config import project_path
from adas_ovd.data import CocoDataset
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

from .features import calibration_targets, detection_feature_bundle, domain_descriptors
from .manifest import validate_manifest


FEATURE_SCHEMA_VERSION = 1
EXTRACTION_SOURCE_PATHS = (
    "src/adas_ovd/config.py",
    "src/adas_ovd/data.py",
    "src/adas_ovd/matching.py",
    "src/adas_ovd/mc_features.py",
    "src/adas_ovd/reproducibility.py",
    "src/adas_ovd/shared_extraction.py",
    "RQ4/src/rq4/extraction.py",
    "RQ4/src/rq4/features.py",
    "RQ4/src/rq4/manifest.py",
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
    "is_class_correct",
    "is_well_localized",
]
DOMAIN_COLUMNS = [
    "shift_timeofday",
    "shift_weather",
    "shift_scene",
    "unknown_timeofday",
    "unknown_weather",
    "unknown_scene",
    "shift_axis_count",
    "unknown_axis_count",
    "is_domain_shift",
    "domain_stratum",
]


def feature_group_names(config: dict[str, Any], group: str) -> list[str]:
    return list(config["rq4"]["feature_groups"][group])


def model_feature_names(
    config: dict[str, Any], group: str, mc_passes: int | None = None
) -> list[str]:
    names = feature_group_names(config, group)
    if mc_passes is None or group == "class":
        return names
    return [f"{name}_mc{int(mc_passes):02d}" for name in names]


def all_inference_features(config: dict[str, Any]) -> list[str]:
    main = []
    for group in ("class", "localization", "uncertainty"):
        main.extend(model_feature_names(config, group))
    sensitivity = []
    total_passes = int(config["rq4"]["extraction"]["mc_passes"])
    for count in config["rq4"]["extraction"]["mc_sensitivity_passes"]:
        if int(count) == total_passes:
            continue
        for group in ("localization", "uncertainty"):
            sensitivity.extend(model_feature_names(config, group, int(count)))
    existing = set(COMMON_COLUMNS + TARGET_COLUMNS + DOMAIN_COLUMNS)
    return [name for name in dict.fromkeys(main + sensitivity) if name not in existing]


def feature_schema(config: dict[str, Any]) -> list[str]:
    columns = COMMON_COLUMNS + TARGET_COLUMNS + DOMAIN_COLUMNS + all_inference_features(config)
    if len(columns) != len(set(columns)):
        raise RuntimeError("RQ4 feature schema contains duplicate columns")
    return columns


def _empty_frame(config: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(columns=feature_schema(config))


def configured_mini_image_ids(
    config: dict[str, Any], split: str
) -> list[int] | None:
    mini = config["rq4"].get("mini")
    if not mini:
        return None
    values = mini.get(f"{split}_image_ids")
    return [int(value) for value in values] if values is not None else None


def source_domain_image_ids(
    config: dict[str, Any], split: str, manifest_image_ids: list[int]
) -> list[int] | None:
    if split == "test":
        return None
    if str(config["rq4"]["domain_shift"].get("development_policy", "")) != "source_only":
        return None
    dataset = CocoDataset(
        project_path(config, config["data"]["calibration_annotations"]),
        project_path(config, config["data"]["images_dir"]),
    )
    reference = {
        str(key): str(value)
        for key, value in config["rq4"]["domain_shift"]["reference"].items()
    }
    selected = [
        int(image_id)
        for image_id in manifest_image_ids
        if all(
            str(dataset.images[int(image_id)].attributes.get(axis, "")) == expected
            for axis, expected in reference.items()
        )
    ]
    if not selected:
        raise ValueError(f"RQ4 {split} contains no images in the frozen source domain")
    return selected


def _materialization_identity(
    config: dict[str, Any],
    split: str,
    *,
    annotation_sha256: str | None = None,
    shared_fingerprint: str | None = None,
    shared_namespace: str | None = None,
    requested_image_ids: list[int] | None = None,
) -> dict[str, Any]:
    manifest_path, manifest = validate_manifest(config)
    shared = shared_identity(config)
    shared_fingerprint = shared_fingerprint or shared["configuration_fingerprint"]
    shared_namespace = shared_namespace or str(config["shared_extraction"]["cache_namespace"])
    if annotation_sha256 is None:
        key = "evaluation_annotations" if split == "test" else "calibration_annotations"
        annotation_sha256 = sha256_file(project_path(config, config["data"][key]))
    source_sha256 = source_tree_sha256(
        config["_meta"]["project_root"], EXTRACTION_SOURCE_PATHS
    )
    configuration = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "rq": "rq4",
        "match_iou": config["evaluation"]["match_iou"],
        "targets": config["rq4"]["targets"],
        "domain_shift": config["rq4"]["domain_shift"],
        "feature_groups": config["rq4"]["feature_groups"],
        "mc_sensitivity_passes": config["rq4"]["extraction"]["mc_sensitivity_passes"],
        "feature_schema": feature_schema(config),
        "shared_fingerprint": shared_fingerprint,
        "shared_namespace": shared_namespace,
        "source_tree_sha256": source_sha256,
        "manifest_specification": config["rq4"]["manifest"],
        "requested_image_ids": (
            requested_image_ids
            if requested_image_ids is not None
            else configured_mini_image_ids(config, split)
            or manifest["splits"][split]
        ),
    }
    active_image_ids = configuration["requested_image_ids"]
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
        "manifest_split_image_ids_sha256": stable_fingerprint(manifest["splits"][split]),
        "split_image_ids_sha256": stable_fingerprint(active_image_ids),
        "feature_schema": feature_schema(config),
    }
    identity["materialization_fingerprint"] = stable_fingerprint(identity)
    return identity


def _ground_truth_arrays(shared_split: SharedSplit, image_id: int) -> tuple[np.ndarray, np.ndarray]:
    records = shared_split.dataset.ground_truth(image_id)
    boxes = np.asarray([record.bbox_xyxy for record in records], dtype=np.float64).reshape(-1, 4)
    categories = np.asarray([record.category_id for record in records], dtype=np.int64)
    return boxes, categories


def _materialize_shared_image(
    shared_split: SharedSplit, image_id: int, config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    started = time.perf_counter()
    common = materialize_common(shared_split, image_id, config)
    arrays = common.arrays
    rows = common.frame.to_dict(orient="records")
    image = shared_split.dataset.images[image_id]
    ground_truth_boxes, ground_truth_categories = _ground_truth_arrays(shared_split, image_id)
    category_ids = np.asarray(
        [shared_split.dataset.category_ids_by_name[name] for name in config["data"]["classes"]],
        dtype=np.int64,
    )
    prediction_categories = category_ids[arrays["reference_category_indices"]]
    targets = calibration_targets(
        arrays["reference_boxes_xyxy"],
        prediction_categories,
        ground_truth_boxes,
        ground_truth_categories,
        localization_iou_threshold=float(config["rq4"]["targets"]["localization_iou_threshold"]),
        class_assignment_iou_threshold=float(config["rq4"]["targets"]["class_assignment_iou_threshold"]),
    )
    domains = domain_descriptors(
        image.attributes,
        config["rq4"]["domain_shift"]["reference"],
        config["rq4"]["domain_shift"]["undefined_values"],
    )
    sensitivity_passes = sorted(
        int(value) for value in config["rq4"]["extraction"]["mc_sensitivity_passes"]
    )
    total_passes = int(config["rq4"]["extraction"]["mc_passes"])
    if not sensitivity_passes or any(value < 1 or value > total_passes for value in sensitivity_passes):
        raise ValueError("RQ4 MC sensitivity counts must lie in [1, mc_passes]")

    feature_names = set(
        feature_group_names(config, "localization")
        + feature_group_names(config, "uncertainty")
    )
    for detection_index, row in enumerate(rows):
        for name in TARGET_COLUMNS:
            value = targets[name][detection_index]
            if name in {"localization_iou"}:
                row[name] = float(value)
            elif name == "localization_class_agreement":
                row[name] = bool(value)
            else:
                row[name] = int(value)
        row.update(domains)
        arguments = {
            "reference_box_cxcywh": arrays["reference_boxes_cxcywh"][detection_index],
            "category_scores": arrays["mc_category_scores"][:, detection_index],
            "scores": arrays["mc_scores"][:, detection_index],
            "boxes_cxcywh": arrays["mc_boxes_cxcywh"][:, detection_index],
            "embeddings": arrays["mc_embeddings"][:, detection_index],
            "present": arrays["present"][:, detection_index],
            "base_category": int(arrays["reference_category_indices"][detection_index]),
            "deterministic_reference_variance": arrays["deterministic_reference_variance"][detection_index],
            "deterministic_reference_step": arrays["deterministic_reference_step"][detection_index],
            "deterministic_hidden_step": arrays["deterministic_hidden_step"][detection_index],
            "bbox_area_fraction": float(row["bbox_area"]) / float(image.width * image.height),
        }
        bundle = detection_feature_bundle(**arguments)
        row.update({name: bundle[name] for name in feature_names})
        for count in sensitivity_passes:
            if count == total_passes:
                continue
            prefix_arguments = dict(arguments)
            for key in ("category_scores", "scores", "boxes_cxcywh", "embeddings", "present"):
                prefix_arguments[key] = prefix_arguments[key][:count]
            prefix = detection_feature_bundle(**prefix_arguments)
            for name in feature_names:
                row[f"{name}_mc{count:02d}"] = prefix[name]

    frame = pd.DataFrame(rows) if rows else _empty_frame(config)
    expected = feature_schema(config)
    missing = sorted(set(expected) - set(frame.columns))
    if missing:
        raise RuntimeError(f"RQ4 materialization omitted columns: {missing}")
    frame = frame[expected]
    aggregation_seconds = time.perf_counter() - started
    summary = dict(common.image_summary)
    summary.update(
        {
            **domains,
            "class_correct_detections": int(targets["is_class_correct"].sum()),
            "well_localized_detections": int(targets["is_well_localized"].sum()),
            "mean_localization_iou": (
                float(targets["localization_iou"].mean()) if len(targets["localization_iou"]) else None
            ),
            "aggregation_seconds": aggregation_seconds,
            "total_seconds": float(common.image_summary["shared_inference_seconds"]) + aggregation_seconds,
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
    if not (detection_path.is_file() and image_summary_path.is_file() and metadata_path.is_file()):
        return False
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        return bool(
            int(metadata.get("schema_version", -1)) == FEATURE_SCHEMA_VERSION
            and int(metadata.get("image_id", -1)) == int(image_id)
            and metadata.get("materialization_fingerprint") == materialization_fingerprint
            and metadata.get("source_tree_sha256") == source_tree_sha256_value
            and metadata.get("shared_shard_sha256") == shared_shard_sha256
            and metadata.get("feature_schema") == expected_schema
            and metadata.get("feature_shard_sha256") == sha256_file(detection_path)
            and metadata.get("image_summary_sha256") == sha256_file(image_summary_path)
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
    image_ids_override: list[int] | None = None,
) -> Path:
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unknown RQ4 split: {split}")
    validate_consumer_compatibility(config, "rq4")
    manifest_path, manifest = validate_manifest(config)
    output_path = (
        Path(output_override).resolve()
        if output_override is not None
        else project_path(config, config["rq4"]["outputs"][f"{split}_features"])
    )
    image_output_path = (
        output_path.with_name(f"{output_path.stem}_images.parquet")
        if output_override is not None
        else project_path(config, config["rq4"]["outputs"][f"{split}_image_summary"])
    )
    configured_ids = configured_mini_image_ids(config, split)
    requested_image_ids = image_ids_override if image_ids_override is not None else configured_ids
    if requested_image_ids is None:
        requested_image_ids = source_domain_image_ids(config, split, manifest["splits"][split])
    shared_limit = limit
    if requested_image_ids is not None:
        if limit is not None:
            requested_image_ids = requested_image_ids[: int(limit)]
        shared_limit = None
    shared_split = ensure_shared_split(
        config,
        manifest_path=manifest_path,
        split=split,
        limit=shared_limit,
        image_ids_override=requested_image_ids,
        cache_namespace=shared_cache_namespace,
    )
    identity = _materialization_identity(
        config,
        split,
        annotation_sha256=shared_split.dataset.sha256,
        shared_fingerprint=shared_split.fingerprint,
        shared_namespace=shared_split.namespace,
        requested_image_ids=shared_split.image_ids,
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
    pending: list[int] = []
    for image_id in shared_split.image_ids:
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
        pending.append(image_id)

    def materialize_one(image_id: int) -> tuple[int, pd.DataFrame, dict[str, Any]]:
        frame, summary = _materialize_shared_image(shared_split, image_id, config)
        return image_id, frame, summary

    workers = max(1, int(config["rq4"].get("execution", {}).get("materialization_workers", 1)))
    completed: list[tuple[int, pd.DataFrame, dict[str, Any]]] = []
    if workers == 1 or len(pending) <= 1:
        completed = [
            materialize_one(image_id)
            for image_id in tqdm(pending, desc=f"RQ4 materialization [{split}]")
        ]
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(pending))) as executor:
            futures = {executor.submit(materialize_one, image_id): image_id for image_id in pending}
            with tqdm(total=len(futures), desc=f"RQ4 materialization [{split}]") as progress:
                for future in as_completed(futures):
                    completed.append(future.result())
                    progress.update(1)
    for image_id, frame, summary in sorted(completed, key=lambda item: item[0]):
        detection_path = detection_root / f"{image_id}.parquet"
        image_summary_path = image_root / f"{image_id}.parquet"
        metadata_path = metadata_root / f"{image_id}.json"
        shared_sha256 = shared_split.shard_metadata[image_id]["shard_sha256"]
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
            shared_shard_sha256=shared_split.shard_metadata[image_id]["shard_sha256"],
            expected_schema=expected_schema,
        ):
            raise RuntimeError(f"Invalid or missing RQ4 feature shard: {image_id}")
        with metadata_path.open("r", encoding="utf-8") as handle:
            inventory.append(json.load(handle))

    frames = [pd.read_parquet(detection_root / f"{image_id}.parquet") for image_id in shared_split.image_ids]
    combined = pd.concat(frames, ignore_index=True) if frames else _empty_frame(config)
    image_summaries = pd.concat(
        [pd.read_parquet(image_root / f"{image_id}.parquet") for image_id in shared_split.image_ids],
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
        "shared_request_metadata_sha256": sha256_file(shared_split.request_metadata_path),
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
            for name in ("preprocess_seconds", "deterministic_seconds", "stochastic_seconds", "aggregation_seconds")
        },
        "peak_gpu_memory_bytes": int(image_summaries["peak_gpu_memory_bytes"].max()),
        "stochastic_modules": (
            shared_split.shard_metadata[shared_split.image_ids[0]]["enabled_stochastic_modules"]
            if shared_split.image_ids else []
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
        raise FileNotFoundError(f"RQ4 feature artifact is incomplete: {feature_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    split = metadata.get("split")
    if split not in {"train", "validation", "test"}:
        raise RuntimeError("RQ4 feature metadata has an invalid split")
    expected = _materialization_identity(
        config,
        split,
        requested_image_ids=configured_mini_image_ids(config, split),
    )
    for key, expected_value in expected.items():
        if metadata.get(key) != expected_value:
            raise RuntimeError(f"RQ4 feature artifact is stale ({key}): {feature_path}")
    if metadata.get("features_sha256") != sha256_file(feature_path):
        raise RuntimeError(f"RQ4 feature SHA-256 mismatch: {feature_path}")
    receipt_path = Path(metadata.get("shared_request_metadata_path", ""))
    if not receipt_path.is_file() or metadata.get("shared_request_metadata_sha256") != sha256_file(receipt_path):
        raise RuntimeError("RQ4 shared extraction receipt integrity check failed")
    image_summary_path = Path(metadata["image_summary_path"])
    if not image_summary_path.is_file() or metadata.get("image_summary_sha256") != sha256_file(image_summary_path):
        raise RuntimeError("RQ4 image-summary integrity check failed")
    frame = pd.read_parquet(feature_path)
    if list(frame.columns) != metadata.get("feature_schema"):
        raise RuntimeError("RQ4 feature schema differs from frozen metadata")
    return frame, metadata
