from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from adas_ovd.config import project_path
from adas_ovd.data import CocoDataset, load_manifest_ids
from adas_ovd.groundingdino_adapter import (
    GroundingDinoAdapter,
    resolve_package_resource,
)
from adas_ovd.matching import associate_detections, match_predictions_to_ground_truth
from adas_ovd.reproducibility import (
    environment_metadata,
    seed_everything,
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

from .features import decoder_trajectory_features, stochastic_features
from .manifest import validate_manifest


EXTRACTION_SOURCE_PATHS = (
    "src/adas_ovd/config.py",
    "src/adas_ovd/data.py",
    "src/adas_ovd/groundingdino_adapter.py",
    "src/adas_ovd/matching.py",
    "src/adas_ovd/mc_features.py",
    "src/adas_ovd/reproducibility.py",
    "src/adas_ovd/shared_extraction.py",
    "RQ2/src/rq2/extraction.py",
    "RQ2/src/rq2/features.py",
    "RQ2/src/rq2/manifest.py",
)

IDENTITY_COLUMNS = [
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


def _feature_names(config: dict[str, Any]) -> list[str]:
    groups = config["rq2"]["feature_groups"]
    primary = [
        feature
        for group in ("deterministic", "stochastic")
        for feature in groups[group]
    ]
    stochastic = list(groups["stochastic"])
    sensitivity = [
        f"{feature}_mc{count:02d}"
        for count in sorted(
            int(value)
            for value in config["rq2"]["extraction"]["mc_sensitivity_passes"]
        )
        for feature in stochastic
    ]
    return primary + sensitivity


def _uncertainty_feature_names(config: dict[str, Any]) -> list[str]:
    return list(config["rq2"]["feature_groups"]["confidence"]) + _feature_names(
        config
    )


def _empty_frame(config: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(columns=IDENTITY_COLUMNS + _feature_names(config))


def _dataset_for_split(config: dict[str, Any], split: str) -> CocoDataset:
    annotation_key = (
        "evaluation_annotations" if split == "test" else "calibration_annotations"
    )
    return CocoDataset(
        project_path(config, config["data"][annotation_key]),
        project_path(config, config["data"]["images_dir"]),
    )


def _category_mapping(dataset: CocoDataset, classes: list[str]) -> np.ndarray:
    missing = [name for name in classes if name not in dataset.category_ids_by_name]
    if missing:
        raise ValueError(f"Prompt categories absent from annotations: {missing}")
    return np.asarray(
        [dataset.category_ids_by_name[name] for name in classes], dtype=np.int64
    )


def _ground_truth_arrays(dataset: CocoDataset, image_id: int):
    records = dataset.ground_truth(image_id)
    boxes = np.asarray(
        [record.bbox_xyxy for record in records], dtype=np.float64
    ).reshape(-1, 4)
    categories = np.asarray(
        [record.category_id for record in records], dtype=np.int64
    )
    return records, boxes, categories


def _synchronize(adapter: GroundingDinoAdapter) -> None:
    if adapter.device.type == "cuda":
        adapter.torch.cuda.synchronize(adapter.device)


def _object_size(box_xyxy: np.ndarray) -> tuple[float, str]:
    width = max(float(box_xyxy[2] - box_xyxy[0]), 0.0)
    height = max(float(box_xyxy[3] - box_xyxy[1]), 0.0)
    area = width * height
    if area < 32.0**2:
        label = "small"
    elif area < 96.0**2:
        label = "medium"
    else:
        label = "large"
    return area, label


def _extraction_identity(
    config: dict[str, Any],
    split: str,
    dataset: CocoDataset | None = None,
    shared_fingerprint: str | None = None,
    shared_namespace: str | None = None,
) -> dict[str, Any]:
    manifest_path, manifest = validate_manifest(config)
    dataset = dataset or _dataset_for_split(config, split)
    shared = shared_identity(config)
    shared_fingerprint = shared_fingerprint or shared["configuration_fingerprint"]
    shared_namespace = shared_namespace or str(
        config["shared_extraction"]["cache_namespace"]
    )
    source_sha256 = source_tree_sha256(
        config["_meta"]["project_root"], EXTRACTION_SOURCE_PATHS
    )
    configuration = {
        "schema_version": 2,
        "rq": "rq2",
        "match_iou": config["evaluation"]["match_iou"],
        "mc_sensitivity_passes": config["rq2"]["extraction"][
            "mc_sensitivity_passes"
        ],
        "feature_groups": config["rq2"]["feature_groups"],
        "shared_fingerprint": shared_fingerprint,
        "shared_namespace": shared_namespace,
        "source_tree_sha256": source_sha256,
        "manifest_specification": config["rq2"]["manifest"],
    }
    configuration_fingerprint = stable_fingerprint(configuration)
    identity = {
        "schema_version": 2,
        "split": split,
        "source_tree_sha256": source_sha256,
        "configuration_fingerprint": configuration_fingerprint,
        "checkpoint_sha256": shared["checkpoint_sha256"],
        "shared_fingerprint": shared_fingerprint,
        "shared_namespace": shared_namespace,
        "annotation_sha256": dataset.sha256,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_test_partition": manifest["test_partition"],
        "split_image_ids_sha256": stable_fingerprint(manifest["splits"][split]),
    }
    identity["extraction_fingerprint"] = stable_fingerprint(identity)
    return identity


def extract_image(
    adapter: GroundingDinoAdapter,
    dataset: CocoDataset,
    image_id: int,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    extraction = config["rq2"]["extraction"]
    classes = list(config["data"]["classes"])
    category_ids = _category_mapping(dataset, classes)
    image = dataset.images[image_id]
    image_path = dataset.image_path(image_id)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image is missing: {image_path}")
    _, tensor = adapter.preprocess(image_path)
    ground_truth, gt_boxes, gt_categories = _ground_truth_arrays(dataset, image_id)
    common_summary: dict[str, Any] = {
        "image_id": int(image_id),
        "file_name": image.file_name,
        "sequence_id": image.sequence_id,
        "timeofday": image.attributes.get("timeofday", "unknown"),
        "weather": image.attributes.get("weather", "unknown"),
        "scene": image.attributes.get("scene", "unknown"),
        "ground_truth_objects": len(ground_truth),
    }

    seed = int(config["project"]["seed"]) + int(image_id)
    deterministic = bool(config["project"]["deterministic_algorithms"])
    warn_only = bool(config["project"]["deterministic_warn_only"])
    seed_everything(seed, deterministic, warn_only)
    _synchronize(adapter)
    started = time.perf_counter()
    reference = adapter.run(
        tensor,
        image.width,
        image.height,
        float(config["model"]["box_threshold"]),
        int(config["model"]["reference_max_detections"]),
    )
    _synchronize(adapter)
    deterministic_seconds = time.perf_counter() - started

    if len(reference.scores) == 0:
        common_summary.update(
            {
                "reference_detections": 0,
                "true_positive_detections": 0,
                "false_positive_detections": 0,
                "false_negatives": len(ground_truth),
                "deterministic_seconds": deterministic_seconds,
                "stochastic_seconds": 0.0,
                "aggregation_seconds": 0.0,
            }
        )
        return _empty_frame(config), common_summary

    coco_categories = category_ids[reference.category_indices]
    correctness = match_predictions_to_ground_truth(
        reference.boxes_xyxy,
        reference.scores,
        coco_categories,
        gt_boxes,
        gt_categories,
        float(config["evaluation"]["match_iou"]),
    )
    common_summary.update(
        {
            "reference_detections": int(len(reference.scores)),
            "true_positive_detections": int(correctness.is_true_positive.sum()),
            "false_positive_detections": int((~correctness.is_true_positive).sum()),
            "false_negatives": int(correctness.false_negatives),
        }
    )

    passes = int(extraction["mc_passes"])
    sensitivity = sorted(
        {int(value) for value in extraction["mc_sensitivity_passes"]}
    )
    if not sensitivity or any(value < 2 or value > passes for value in sensitivity):
        raise ValueError("MC sensitivity counts must lie in [2, mc_passes]")
    n_reference = len(reference.scores)
    n_classes = len(classes)
    embedding_dimension = reference.hidden_states.shape[-1]
    mc_category_scores = np.full(
        (passes, n_reference, n_classes), np.nan, dtype=np.float32
    )
    mc_scores = np.full((passes, n_reference), np.nan, dtype=np.float32)
    mc_boxes = np.full((passes, n_reference, 4), np.nan, dtype=np.float32)
    mc_embeddings = np.full(
        (passes, n_reference, embedding_dimension), np.nan, dtype=np.float32
    )
    present = np.zeros((passes, n_reference), dtype=bool)

    _synchronize(adapter)
    stochastic_started = time.perf_counter()
    with adapter.stochastic_mode():
        for pass_index in range(passes):
            pass_seed = seed + (pass_index + 1) * int(extraction["mc_seed_stride"])
            seed_everything(pass_seed, deterministic, warn_only)
            stochastic = adapter.run(
                tensor,
                image.width,
                image.height,
                float(extraction["candidate_threshold"]),
                int(config["model"]["max_detections"]),
                required_query_indices=reference.query_indices,
            )
            association = associate_detections(
                reference.boxes_xyxy,
                reference.category_indices,
                stochastic.boxes_xyxy,
                stochastic.category_indices,
                float(extraction["association_iou"]),
                float(extraction["association_class_penalty"]),
                float(extraction["unmatched_cost"]),
            )
            for reference_index, candidate_index in enumerate(association):
                if candidate_index < 0:
                    continue
                present[pass_index, reference_index] = True
                mc_category_scores[pass_index, reference_index] = (
                    stochastic.category_scores[candidate_index]
                )
                mc_scores[pass_index, reference_index] = stochastic.scores[candidate_index]
                mc_boxes[pass_index, reference_index] = (
                    stochastic.boxes_cxcywh[candidate_index]
                )
                mc_embeddings[pass_index, reference_index] = (
                    stochastic.hidden_states[-1, candidate_index]
                )
    _synchronize(adapter)
    stochastic_seconds = time.perf_counter() - stochastic_started

    aggregation_started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for detection_index in range(n_reference):
        box = reference.boxes_xyxy[detection_index]
        area, size = _object_size(box)
        row: dict[str, Any] = {
            "image_id": int(image_id),
            "file_name": image.file_name,
            "sequence_id": image.sequence_id,
            "timeofday": image.attributes.get("timeofday", "unknown"),
            "weather": image.attributes.get("weather", "unknown"),
            "scene": image.attributes.get("scene", "unknown"),
            "detection_index": int(detection_index),
            "query_index": int(reference.query_indices[detection_index]),
            "category_index": int(reference.category_indices[detection_index]),
            "category_id": int(coco_categories[detection_index]),
            "category_name": classes[reference.category_indices[detection_index]],
            "score": float(reference.scores[detection_index]),
            "confidence_uncertainty": float(1.0 - reference.scores[detection_index]),
            "bbox_x1": float(box[0]),
            "bbox_y1": float(box[1]),
            "bbox_x2": float(box[2]),
            "bbox_y2": float(box[3]),
            "bbox_area": area,
            "object_size": size,
            "is_true_positive": bool(correctness.is_true_positive[detection_index]),
            "is_error": int(not correctness.is_true_positive[detection_index]),
            "matched_iou": float(correctness.matched_iou[detection_index]),
            "matched_ground_truth_index": int(
                correctness.matched_ground_truth[detection_index]
            ),
            "false_negatives_image": int(correctness.false_negatives),
            "mc_matches": int(present[:, detection_index].sum()),
            "mc_passes": passes,
        }
        row.update(
            decoder_trajectory_features(
                reference.hidden_states[:, detection_index, :],
                reference.reference_points[:, detection_index, :],
            )
        )
        stochastic_arguments = {
            "category_scores": mc_category_scores[:, detection_index],
            "scores": mc_scores[:, detection_index],
            "boxes_cxcywh": mc_boxes[:, detection_index],
            "embeddings": mc_embeddings[:, detection_index],
            "present": present[:, detection_index],
            "base_category": int(reference.category_indices[detection_index]),
        }
        row.update(stochastic_features(**stochastic_arguments))
        for count in sensitivity:
            prefix_values = stochastic_features(
                category_scores=stochastic_arguments["category_scores"][:count],
                scores=stochastic_arguments["scores"][:count],
                boxes_cxcywh=stochastic_arguments["boxes_cxcywh"][:count],
                embeddings=stochastic_arguments["embeddings"][:count],
                present=stochastic_arguments["present"][:count],
                base_category=stochastic_arguments["base_category"],
            )
            row.update(
                {f"{name}_mc{count:02d}": value for name, value in prefix_values.items()}
            )
        rows.append(row)
    aggregation_seconds = time.perf_counter() - aggregation_started
    common_summary.update(
        {
            "deterministic_seconds": deterministic_seconds,
            "stochastic_seconds": stochastic_seconds,
            "aggregation_seconds": aggregation_seconds,
        }
    )
    frame = pd.DataFrame(rows)
    expected_columns = IDENTITY_COLUMNS + _feature_names(config)
    missing = sorted(set(expected_columns) - set(frame.columns))
    if missing:
        raise RuntimeError(f"Extraction omitted frozen columns: {missing}")
    return frame[expected_columns], common_summary


def _materialize_shared_image(
    shared_split: SharedSplit,
    image_id: int,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    started = time.perf_counter()
    common = materialize_common(shared_split, image_id, config)
    rows = common.frame.to_dict(orient="records")
    arrays = common.arrays
    sensitivity = sorted(
        int(value)
        for value in config["rq2"]["extraction"]["mc_sensitivity_passes"]
    )
    for detection_index, row in enumerate(rows):
        row["deterministic_reference_variance"] = arrays[
            "deterministic_reference_variance"
        ][detection_index]
        row["deterministic_reference_step"] = arrays[
            "deterministic_reference_step"
        ][detection_index]
        row["deterministic_hidden_step"] = arrays[
            "deterministic_hidden_step"
        ][detection_index]
        arguments = {
            "category_scores": arrays["mc_category_scores"][:, detection_index],
            "scores": arrays["mc_scores"][:, detection_index],
            "boxes_cxcywh": arrays["mc_boxes_cxcywh"][:, detection_index],
            "embeddings": arrays["mc_embeddings"][:, detection_index],
            "present": arrays["present"][:, detection_index],
            "base_category": int(
                arrays["reference_category_indices"][detection_index]
            ),
        }
        row.update(stochastic_features(**arguments))
        for count in sensitivity:
            prefix = stochastic_features(
                category_scores=arguments["category_scores"][:count],
                scores=arguments["scores"][:count],
                boxes_cxcywh=arguments["boxes_cxcywh"][:count],
                embeddings=arguments["embeddings"][:count],
                present=arguments["present"][:count],
                base_category=arguments["base_category"],
            )
            row.update(
                {
                    f"{name}_mc{count:02d}": value
                    for name, value in prefix.items()
                }
            )
    aggregation_seconds = time.perf_counter() - started
    frame = pd.DataFrame(rows)
    summary = dict(common.image_summary)
    summary["aggregation_seconds"] = aggregation_seconds
    summary["total_seconds"] = (
        float(summary["shared_inference_seconds"]) + aggregation_seconds
    )
    expected_columns = IDENTITY_COLUMNS + _feature_names(config)
    missing = sorted(set(expected_columns) - set(frame.columns))
    if missing:
        raise RuntimeError(f"RQ2 shared materialization omitted columns: {missing}")
    return frame[expected_columns], summary


def _valid_shard(
    detection_path: Path,
    image_path: Path,
    metadata_path: Path,
    image_id: int,
    extraction_fingerprint: str,
) -> bool:
    if not detection_path.is_file() or not image_path.is_file() or not metadata_path.is_file():
        return False
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        return bool(
            int(metadata.get("image_id", -1)) == int(image_id)
            and metadata.get("extraction_fingerprint") == extraction_fingerprint
            and metadata.get("detection_sha256") == sha256_file(detection_path)
            and metadata.get("image_summary_sha256") == sha256_file(image_path)
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _write_shard(
    detection_path: Path,
    image_path: Path,
    metadata_path: Path,
    frame: pd.DataFrame,
    image_summary: dict[str, Any],
    extraction_fingerprint: str,
) -> None:
    detection_temporary = detection_path.with_suffix(".parquet.tmp")
    frame.to_parquet(detection_temporary, index=False)
    detection_temporary.replace(detection_path)
    image_temporary = image_path.with_suffix(".parquet.tmp")
    pd.DataFrame([image_summary]).to_parquet(image_temporary, index=False)
    image_temporary.replace(image_path)
    write_json(
        metadata_path,
        {
            "schema_version": 1,
            "image_id": int(image_summary["image_id"]),
            "rows": int(len(frame)),
            "extraction_fingerprint": extraction_fingerprint,
            "detection_sha256": sha256_file(detection_path),
            "image_summary_sha256": sha256_file(image_path),
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
        raise ValueError(f"Unknown split: {split}")
    validate_consumer_compatibility(config, "rq2")
    manifest_path, _ = validate_manifest(config)
    output_key = f"{split}_features"
    image_output_key = f"{split}_image_summary"
    output_path = (
        Path(output_override).resolve()
        if output_override is not None
        else project_path(config, config["rq2"]["outputs"][output_key])
    )
    image_output_path = (
        output_path.with_name(f"{output_path.stem}_images.parquet")
        if output_override is not None
        else project_path(config, config["rq2"]["outputs"][image_output_key])
    )
    shared_split = ensure_shared_split(
        config,
        manifest_path=manifest_path,
        split=split,
        limit=limit,
        cache_namespace=shared_cache_namespace,
    )
    image_ids = shared_split.image_ids
    dataset = shared_split.dataset
    identity = _extraction_identity(
        config,
        split,
        dataset,
        shared_fingerprint=shared_split.fingerprint,
        shared_namespace=shared_split.namespace,
    )
    fingerprint = identity["extraction_fingerprint"]

    shard_root = output_path.parent / "shards" / split / fingerprint
    image_shard_root = output_path.parent / "image_shards" / split / fingerprint
    metadata_root = output_path.parent / "shard_metadata" / split / fingerprint
    for directory in (shard_root, image_shard_root, metadata_root):
        directory.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    reused = 0
    recomputed = 0
    for image_id in tqdm(image_ids, desc=f"RQ2 feature materialization [{split}]"):
        detection_shard = shard_root / f"{image_id}.parquet"
        image_shard = image_shard_root / f"{image_id}.parquet"
        shard_metadata = metadata_root / f"{image_id}.json"
        shared_sha = shared_split.shard_metadata[image_id]["shard_sha256"]
        if _valid_shard(
            detection_shard,
            image_shard,
            shard_metadata,
            image_id,
            fingerprint,
        ):
            with shard_metadata.open("r", encoding="utf-8") as handle:
                existing_metadata = json.load(handle)
            if existing_metadata.get("shared_shard_sha256") == shared_sha:
                reused += 1
                continue
        frame, summary = _materialize_shared_image(
            shared_split, image_id, config
        )
        _write_shard(
            detection_shard,
            image_shard,
            shard_metadata,
            frame,
            summary,
            fingerprint,
        )
        with shard_metadata.open("r", encoding="utf-8") as handle:
            feature_metadata = json.load(handle)
        feature_metadata["shared_shard_sha256"] = shared_sha
        write_json(shard_metadata, feature_metadata)
        recomputed += 1

    invalid = []
    inventory = []
    for image_id in image_ids:
        detection_shard = shard_root / f"{image_id}.parquet"
        image_shard = image_shard_root / f"{image_id}.parquet"
        shard_metadata = metadata_root / f"{image_id}.json"
        if not _valid_shard(
            detection_shard, image_shard, shard_metadata, image_id, fingerprint
        ):
            invalid.append(image_id)
            continue
        with shard_metadata.open("r", encoding="utf-8") as handle:
            inventory.append(json.load(handle))
    if invalid:
        raise RuntimeError(f"RQ2 extraction has invalid/missing shards: {invalid[:10]}")

    frames = [pd.read_parquet(shard_root / f"{image_id}.parquet") for image_id in image_ids]
    combined = pd.concat(frames, ignore_index=True) if frames else _empty_frame(config)
    image_summaries = pd.concat(
        [pd.read_parquet(image_shard_root / f"{image_id}.parquet") for image_id in image_ids],
        ignore_index=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_temporary = output_path.with_suffix(".parquet.tmp")
    combined.to_parquet(combined_temporary, index=False)
    combined_temporary.replace(output_path)
    image_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_temporary = image_output_path.with_suffix(".parquet.tmp")
    image_summaries.to_parquet(summary_temporary, index=False)
    summary_temporary.replace(image_output_path)

    timing_columns = [
        "deterministic_seconds",
        "stochastic_seconds",
        "aggregation_seconds",
    ]
    with shared_split.request_metadata_path.open("r", encoding="utf-8") as handle:
        shared_request = json.load(handle)
    metadata = {
        **identity,
        "images_requested": len(image_ids),
        "images_completed": len(image_summaries),
        "detections": len(combined),
        "feature_columns": list(combined.columns),
        "uncertainty_feature_columns": _uncertainty_feature_names(config),
        "features_sha256": sha256_file(output_path),
        "image_summary_path": str(image_output_path),
        "image_summary_sha256": sha256_file(image_output_path),
        "shard_inventory_sha256": stable_fingerprint(inventory),
        "shards_reused": reused,
        "shards_recomputed": recomputed,
        "elapsed_seconds": time.perf_counter() - started,
        "random_seeds": {
            "project_seed": int(config["project"]["seed"]),
            "deterministic_image_seed": "project_seed + image_id",
            "mc_seed_stride": int(config["shared_extraction"]["mc_seed_stride"]),
            "mc_pass_seed": (
                "project_seed + image_id + (pass_index + 1) * mc_seed_stride"
            ),
        },
        "timing_seconds": {
            name: float(image_summaries[name].sum()) for name in timing_columns
        },
        "shared_request_metadata_path": str(shared_split.request_metadata_path),
        "shared_request_metadata_sha256": sha256_file(
            shared_split.request_metadata_path
        ),
        "shared_shard_inventory_sha256": shared_request[
            "shard_inventory_sha256"
        ],
        "shared_shards_computed": shared_request["shards_computed"],
        "shared_shards_reused": shared_request["shards_reused"],
        "shared_cache_namespace": shared_split.namespace,
        "shared_fingerprint": shared_split.fingerprint,
        "checkpoint_path": str(project_path(config, config["model"]["checkpoint"])),
        "annotation_path": str(dataset.annotation_path),
        "stochastic_modules": shared_split.shard_metadata[image_ids[0]][
            "enabled_stochastic_modules"
        ],
        "environment": environment_metadata(config["_meta"]["project_root"]),
    }
    write_json(output_path.with_suffix(".metadata.json"), metadata)
    return output_path


def read_validated_features(
    config: dict[str, Any], path: str | Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    feature_path = Path(path).resolve()
    metadata_path = feature_path.with_suffix(".metadata.json")
    if not feature_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"RQ2 feature artifact is incomplete: {feature_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    split = metadata.get("split")
    if split not in {"train", "validation", "test"}:
        raise RuntimeError("RQ2 feature metadata has an invalid split")
    expected = _extraction_identity(config, split)
    for key, expected_value in expected.items():
        if metadata.get(key) != expected_value:
            raise RuntimeError(
                f"RQ2 feature artifact is stale ({key} changed): {feature_path}"
            )
    if metadata.get("features_sha256") != sha256_file(feature_path):
        raise RuntimeError(f"RQ2 feature SHA-256 mismatch: {feature_path}")
    request_path_value = metadata.get("shared_request_metadata_path")
    request_sha256 = metadata.get("shared_request_metadata_sha256")
    if not request_path_value or not request_sha256:
        raise RuntimeError("RQ2 shared extraction receipt is missing")
    request_path = Path(request_path_value)
    if (
        not request_path.is_file()
        or request_sha256 != sha256_file(request_path)
    ):
        raise RuntimeError("RQ2 shared extraction receipt integrity check failed")
    image_summary_path = Path(metadata["image_summary_path"])
    if (
        not image_summary_path.is_file()
        or metadata.get("image_summary_sha256") != sha256_file(image_summary_path)
    ):
        raise RuntimeError("RQ2 image-summary integrity check failed")
    frame = pd.read_parquet(feature_path)
    if list(frame.columns) != metadata.get("feature_columns"):
        raise RuntimeError("RQ2 feature schema differs from its metadata")
    return frame, metadata
