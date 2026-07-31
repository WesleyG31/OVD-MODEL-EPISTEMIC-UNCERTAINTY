from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

from adas_ovd.config import project_path
from adas_ovd.data import CocoDataset, load_manifest_ids
from adas_ovd.groundingdino_adapter import (
    GroundingDinoAdapter,
    PromptMapper,
    resolve_package_resource,
)
from adas_ovd.matching import (
    associate_detections,
    match_predictions_to_ground_truth,
)
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
    validate_consumer_compatibility,
)

from .features import (
    decoder_hidden_step,
    decoder_reference_features,
    geometric_features,
    representation_features,
    semantic_features,
)


EXTRACTION_SOURCE_PATHS = (
    "src/adas_ovd",
    "RQ1/src/rq1/extraction.py",
    "RQ1/src/rq1/features.py",
)


def read_validated_features(
    config: dict[str, Any], path: str | Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    feature_path = Path(path).resolve()
    metadata_path = feature_path.with_suffix(".metadata.json")
    if not feature_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"Feature artifact or metadata is missing: {feature_path}"
        )
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    expected_source = source_tree_sha256(
        config["_meta"]["project_root"], EXTRACTION_SOURCE_PATHS
    )
    if metadata.get("source_tree_sha256") != expected_source:
        raise RuntimeError(
            "Extraction source changed after this feature artifact was "
            f"created: {feature_path}. Re-run extraction."
        )
    if metadata.get("features_sha256") != sha256_file(feature_path):
        raise RuntimeError(f"Feature artifact hash mismatch: {feature_path}")
    request_path_value = metadata.get("shared_request_metadata_path")
    request_sha256 = metadata.get("shared_request_metadata_sha256")
    if not request_path_value or not request_sha256:
        raise RuntimeError(f"Shared extraction receipt is missing: {feature_path}")
    request_path = Path(request_path_value)
    if (
        not request_path.is_file()
        or request_sha256 != sha256_file(request_path)
    ):
        raise RuntimeError(
            f"Shared extraction receipt integrity check failed: {feature_path}"
        )
    return pd.read_parquet(feature_path), metadata


def _ground_truth_arrays(dataset: CocoDataset, image_id: int):
    records = dataset.ground_truth(image_id)
    boxes = np.asarray([record.bbox_xyxy for record in records], dtype=np.float64).reshape(
        -1, 4
    )
    categories = np.asarray(
        [record.category_id for record in records], dtype=np.int64
    )
    return records, boxes, categories


def _category_mapping(dataset: CocoDataset, classes: list[str]) -> np.ndarray:
    missing = [name for name in classes if name not in dataset.category_ids_by_name]
    if missing:
        raise ValueError(f"Prompt classes absent from COCO categories: {missing}")
    return np.asarray(
        [dataset.category_ids_by_name[name] for name in classes], dtype=np.int64
    )


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


def _detection_uncertainty_features(
    *,
    count: int,
    detection_index: int,
    base_category: int,
    present: np.ndarray,
    category_scores: np.ndarray,
    scores: np.ndarray,
    boxes: np.ndarray,
    embeddings: np.ndarray,
    reference_variance: np.ndarray,
    reference_step: np.ndarray,
    hidden_step: np.ndarray,
) -> dict[str, float]:
    detected = present[:count, detection_index]
    values = {
        "absence_rate": float(1.0 - detected.mean()),
    }
    values.update(
        semantic_features(
            category_scores=category_scores[
                :count, detection_index, :
            ],
            scores=scores[:count, detection_index],
            present=detected,
            base_category=base_category,
        )
    )
    values.update(
        geometric_features(
            boxes_cxcywh=boxes[:count, detection_index, :],
            present=detected,
            reference_variance=reference_variance[
                :count, detection_index
            ],
            reference_step=reference_step[:count, detection_index],
        )
    )
    values.update(
        representation_features(
            embeddings=embeddings[:count, detection_index, :],
            present=detected,
            hidden_step=hidden_step[:count, detection_index],
        )
    )
    return values


def extract_image(
    adapter: GroundingDinoAdapter,
    dataset: CocoDataset,
    image_id: int,
    config: dict[str, Any],
    image_path_override: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rq1 = config["rq1"]
    extraction = rq1["extraction"]
    model_config = config["model"]
    classes = list(config["data"]["classes"])
    category_ids = _category_mapping(dataset, classes)
    image_record = dataset.images[image_id]
    image_path = (
        Path(image_path_override).resolve()
        if image_path_override is not None
        else dataset.image_path(image_id)
    )
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if adapter.device.type == "cuda":
        adapter.torch.cuda.reset_peak_memory_stats(adapter.device)
    preprocess_started = time.perf_counter()
    _, image_tensor = adapter.preprocess(image_path)
    preprocess_seconds = time.perf_counter() - preprocess_started
    ground_truth, gt_boxes, gt_categories = _ground_truth_arrays(
        dataset, image_id
    )
    image_summary: dict[str, Any] = {
        "image_id": int(image_id),
        "file_name": image_record.file_name,
        "sequence_id": image_record.sequence_id,
        "timeofday": image_record.attributes.get("timeofday", "unknown"),
        "weather": image_record.attributes.get("weather", "unknown"),
        "scene": image_record.attributes.get("scene", "unknown"),
        "ground_truth_objects": len(ground_truth),
    }
    deterministic_seed = int(config["project"]["seed"]) + int(image_id)
    deterministic_algorithms = bool(
        config["project"]["deterministic_algorithms"]
    )
    deterministic_warn_only = bool(
        config["project"]["deterministic_warn_only"]
    )
    seed_everything(
        deterministic_seed,
        deterministic_algorithms,
        deterministic_warn_only,
    )
    _synchronize(adapter)
    deterministic_started = time.perf_counter()
    reference = adapter.run(
        image_tensor=image_tensor,
        image_width=image_record.width,
        image_height=image_record.height,
        candidate_threshold=float(model_config["box_threshold"]),
        max_detections=int(model_config["reference_max_detections"]),
    )
    _synchronize(adapter)
    deterministic_seconds = time.perf_counter() - deterministic_started
    if len(reference.scores) == 0:
        image_summary.update(
            {
                "reference_detections": 0,
                "true_positive_detections": 0,
                "false_positive_detections": 0,
                "false_negatives": len(ground_truth),
                "preprocess_seconds": preprocess_seconds,
                "deterministic_seconds": deterministic_seconds,
                "stochastic_seconds": 0.0,
                "aggregation_seconds": 0.0,
                "total_seconds": preprocess_seconds + deterministic_seconds,
                "peak_gpu_memory_bytes": (
                    int(adapter.torch.cuda.max_memory_allocated(adapter.device))
                    if adapter.device.type == "cuda"
                    else 0
                ),
            }
        )
        return pd.DataFrame(), image_summary

    reference_coco_categories = category_ids[reference.category_indices]
    correctness = match_predictions_to_ground_truth(
        prediction_boxes=reference.boxes_xyxy,
        prediction_scores=reference.scores,
        prediction_categories=reference_coco_categories,
        ground_truth_boxes=gt_boxes,
        ground_truth_categories=gt_categories,
        iou_threshold=float(config["evaluation"]["match_iou"]),
    )
    image_summary.update(
        {
            "reference_detections": len(reference.scores),
            "true_positive_detections": int(
                correctness.is_true_positive.sum()
            ),
            "false_positive_detections": int(
                (~correctness.is_true_positive).sum()
            ),
            "false_negatives": int(correctness.false_negatives),
        }
    )

    passes = int(extraction["mc_passes"])
    sensitivity_passes = sorted(
        {int(value) for value in extraction["mc_sensitivity_passes"]}
    )
    invalid_sensitivity = [
        value
        for value in sensitivity_passes
        if value < 2 or value > passes
    ]
    if invalid_sensitivity:
        raise ValueError(
            "MC sensitivity passes must be between 2 and mc_passes: "
            f"{invalid_sensitivity}"
        )
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
    reference_variance = np.full((passes, n_reference), np.nan, dtype=np.float32)
    reference_step = np.full((passes, n_reference), np.nan, dtype=np.float32)
    hidden_step = np.full((passes, n_reference), np.nan, dtype=np.float32)
    present = np.zeros((passes, n_reference), dtype=bool)

    _synchronize(adapter)
    stochastic_started = time.perf_counter()
    with adapter.stochastic_mode():
        for pass_index in range(passes):
            pass_seed = (
                deterministic_seed
                + (pass_index + 1) * int(extraction["mc_seed_stride"])
                + int(extraction.get("mc_seed_offset", 0))
            )
            seed_everything(
                pass_seed,
                deterministic_algorithms,
                deterministic_warn_only,
            )
            stochastic = adapter.run(
                image_tensor=image_tensor,
                image_width=image_record.width,
                image_height=image_record.height,
                candidate_threshold=float(extraction["candidate_threshold"]),
                max_detections=int(model_config["max_detections"]),
                required_query_indices=reference.query_indices,
            )
            association = associate_detections(
                reference_boxes=reference.boxes_xyxy,
                reference_categories=reference.category_indices,
                candidate_boxes=stochastic.boxes_xyxy,
                candidate_categories=stochastic.category_indices,
                minimum_iou=float(extraction["association_iou"]),
                class_penalty=float(extraction["association_class_penalty"]),
                unmatched_cost=float(extraction["unmatched_cost"]),
            )
            for reference_index, candidate_index in enumerate(association):
                if candidate_index < 0:
                    continue
                present[pass_index, reference_index] = True
                mc_category_scores[pass_index, reference_index] = (
                    stochastic.category_scores[candidate_index]
                )
                mc_scores[pass_index, reference_index] = stochastic.scores[
                    candidate_index
                ]
                mc_boxes[pass_index, reference_index] = stochastic.boxes_cxcywh[
                    candidate_index
                ]
                trajectory = stochastic.reference_points[:, candidate_index, :]
                representation = stochastic.hidden_states[:, candidate_index, :]
                mc_embeddings[pass_index, reference_index] = representation[-1]
                variance, step = decoder_reference_features(trajectory)
                reference_variance[pass_index, reference_index] = variance
                reference_step[pass_index, reference_index] = step
                hidden_step[pass_index, reference_index] = decoder_hidden_step(
                    representation
                )
    _synchronize(adapter)
    stochastic_seconds = time.perf_counter() - stochastic_started

    aggregation_started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for detection_index in range(n_reference):
        detected = present[:, detection_index]
        box = reference.boxes_xyxy[detection_index]
        bbox_area, object_size = _object_size(box)
        row = {
            "image_id": int(image_id),
            "file_name": image_record.file_name,
            "sequence_id": image_record.sequence_id,
            "timeofday": image_record.attributes.get("timeofday", "unknown"),
            "weather": image_record.attributes.get("weather", "unknown"),
            "scene": image_record.attributes.get("scene", "unknown"),
            "detection_index": int(detection_index),
            "query_index": int(reference.query_indices[detection_index]),
            "category_index": int(reference.category_indices[detection_index]),
            "category_id": int(reference_coco_categories[detection_index]),
            "category_name": classes[reference.category_indices[detection_index]],
            "score": float(reference.scores[detection_index]),
            "confidence_uncertainty": float(
                1.0 - reference.scores[detection_index]
            ),
            "bbox_x1": float(box[0]),
            "bbox_y1": float(box[1]),
            "bbox_x2": float(box[2]),
            "bbox_y2": float(box[3]),
            "bbox_area": bbox_area,
            "object_size": object_size,
            "is_true_positive": bool(
                correctness.is_true_positive[detection_index]
            ),
            "is_error": int(not correctness.is_true_positive[detection_index]),
            "matched_iou": float(correctness.matched_iou[detection_index]),
            "matched_ground_truth_index": int(
                correctness.matched_ground_truth[detection_index]
            ),
            "false_negatives_image": int(correctness.false_negatives),
            "mc_matches": int(detected.sum()),
            "mc_passes": passes,
        }
        feature_arguments = {
            "detection_index": detection_index,
            "base_category": int(
                reference.category_indices[detection_index]
            ),
            "present": present,
            "category_scores": mc_category_scores,
            "scores": mc_scores,
            "boxes": mc_boxes,
            "embeddings": mc_embeddings,
            "reference_variance": reference_variance,
            "reference_step": reference_step,
            "hidden_step": hidden_step,
        }
        row.update(
            _detection_uncertainty_features(
                count=passes, **feature_arguments
            )
        )
        for sensitivity_count in sensitivity_passes:
            prefix_features = _detection_uncertainty_features(
                count=sensitivity_count, **feature_arguments
            )
            suffix = f"_mc{sensitivity_count:02d}"
            row.update(
                {
                    f"{name}{suffix}": value
                    for name, value in prefix_features.items()
                }
            )
        rows.append(row)
    aggregation_seconds = time.perf_counter() - aggregation_started
    peak_gpu_memory_bytes = (
        int(adapter.torch.cuda.max_memory_allocated(adapter.device))
        if adapter.device.type == "cuda"
        else 0
    )
    image_summary.update(
        {
            "preprocess_seconds": preprocess_seconds,
            "deterministic_seconds": deterministic_seconds,
            "stochastic_seconds": stochastic_seconds,
            "aggregation_seconds": aggregation_seconds,
            "total_seconds": (
                preprocess_seconds
                + deterministic_seconds
                + stochastic_seconds
                + aggregation_seconds
            ),
            "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        }
    )
    return pd.DataFrame(rows), image_summary


def _dataset_for_split(config: dict[str, Any], split: str) -> CocoDataset:
    data = config["data"]
    annotation_key = (
        "evaluation_annotations" if split == "test" else "calibration_annotations"
    )
    return CocoDataset(
        annotation_path=project_path(config, data[annotation_key]),
        images_dir=project_path(config, data["images_dir"]),
    )


def build_adapter(
    config: dict[str, Any], prompt_phrases: list[str] | None = None
) -> GroundingDinoAdapter:
    config_path = config["model"]["config"]
    model_config_path = (
        resolve_package_resource(str(config_path))
        if str(config_path).startswith("package://")
        else project_path(config, config_path)
    )
    checkpoint_path = project_path(config, config["model"]["checkpoint"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Run "
            "experiments2.0/scripts/prepare_model.py first."
        )
    checkpoint_sha256 = sha256_file(checkpoint_path)
    expected_checkpoint_sha256 = str(
        config["model"]["checkpoint_sha256"]
    ).lower()
    if checkpoint_sha256 != expected_checkpoint_sha256:
        raise RuntimeError(
            "GroundingDINO checkpoint SHA256 mismatch: "
            f"{checkpoint_sha256} != {expected_checkpoint_sha256}"
        )
    adapter = GroundingDinoAdapter(
        config_path=model_config_path,
        checkpoint_path=checkpoint_path,
        text_encoder_path=project_path(
            config, config["model"]["text_encoder"]["local_dir"]
        ),
        classes=config["data"]["classes"],
        stochastic_module_types=config["rq1"]["extraction"][
            "stochastic_module_types"
        ],
        device=config["model"]["device"],
        amp=bool(config["model"]["amp"]),
    )
    if prompt_phrases is not None:
        if len(prompt_phrases) != len(config["data"]["classes"]):
            adapter.close()
            raise ValueError(
                "Prompt sensitivity phrases must map one-to-one to classes"
            )
        adapter.prompt = PromptMapper(adapter.model.tokenizer, prompt_phrases)
    return adapter


def _materialize_shared_image(
    shared_split: SharedSplit,
    image_id: int,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    started = time.perf_counter()
    common = materialize_common(shared_split, image_id, config)
    rows = common.frame.to_dict(orient="records")
    arrays = common.arrays
    passes = int(config["shared_extraction"]["mc_passes"])
    sensitivity_passes = sorted(
        int(value)
        for value in config["rq1"]["extraction"]["mc_sensitivity_passes"]
    )
    for detection_index, row in enumerate(rows):
        feature_arguments = {
            "detection_index": detection_index,
            "base_category": int(
                arrays["reference_category_indices"][detection_index]
            ),
            "present": arrays["present"],
            "category_scores": arrays["mc_category_scores"],
            "scores": arrays["mc_scores"],
            "boxes": arrays["mc_boxes_cxcywh"],
            "embeddings": arrays["mc_embeddings"],
            "reference_variance": arrays["mc_reference_variance"],
            "reference_step": arrays["mc_reference_step"],
            "hidden_step": arrays["mc_hidden_step"],
        }
        values = _detection_uncertainty_features(
            count=passes, **feature_arguments
        )
        row.update(values)
        for count in sensitivity_passes:
            prefix = _detection_uncertainty_features(
                count=count, **feature_arguments
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
    return frame, summary


def _feature_shard_valid(
    detection_path: Path,
    image_path: Path,
    metadata_path: Path,
    *,
    image_id: int,
    materialization_fingerprint: str,
    shared_sha256: str,
) -> bool:
    if not detection_path.is_file() or not image_path.is_file() or not metadata_path.is_file():
        return False
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        return bool(
            metadata.get("image_id") == int(image_id)
            and metadata.get("materialization_fingerprint")
            == materialization_fingerprint
            and metadata.get("shared_shard_sha256") == shared_sha256
            and metadata.get("detection_sha256") == sha256_file(detection_path)
            and metadata.get("image_summary_sha256") == sha256_file(image_path)
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def extract_split(
    config: dict[str, Any],
    split: str,
    limit: int | None = None,
    output_override: str | Path | None = None,
    shared_cache_namespace: str | None = None,
) -> Path:
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unknown split: {split}")
    validate_consumer_compatibility(config, "rq1")
    output_key = f"{split}_features"
    image_output_key = f"{split}_image_summary"
    output_path = (
        Path(output_override).resolve()
        if output_override
        else project_path(config, config["rq1"]["outputs"][output_key])
    )
    image_output_path = (
        output_path.with_name(f"{output_path.stem}_images.parquet")
        if output_override
        else project_path(config, config["rq1"]["outputs"][image_output_key])
    )
    manifest_path = project_path(config, config["rq1"]["outputs"]["manifest"])
    shared_split = ensure_shared_split(
        config,
        manifest_path=manifest_path,
        split=split,
        limit=limit,
        cache_namespace=shared_cache_namespace,
    )
    image_ids = shared_split.image_ids
    dataset = shared_split.dataset
    source_sha256 = source_tree_sha256(
        config["_meta"]["project_root"], EXTRACTION_SOURCE_PATHS
    )
    materialization_fingerprint = stable_fingerprint(
        {
            "schema_version": 3,
            "rq": "rq1",
            "source_tree_sha256": source_sha256,
            "shared_fingerprint": shared_split.fingerprint,
            "shared_namespace": shared_split.namespace,
            "annotation_sha256": dataset.sha256,
            "match_iou": config["evaluation"]["match_iou"],
            "mc_sensitivity_passes": config["rq1"]["extraction"][
                "mc_sensitivity_passes"
            ],
        }
    )
    shard_root = output_path.parent / "shards" / split / materialization_fingerprint
    image_shard_root = (
        output_path.parent / "image_shards" / split / materialization_fingerprint
    )
    metadata_root = (
        output_path.parent / "shard_metadata" / split / materialization_fingerprint
    )
    for directory in (shard_root, image_shard_root, metadata_root):
        directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    reused = 0
    recomputed = 0
    for image_id in tqdm(image_ids, desc=f"RQ1 feature materialization [{split}]"):
        detection_path = shard_root / f"{image_id}.parquet"
        summary_path = image_shard_root / f"{image_id}.parquet"
        metadata_path = metadata_root / f"{image_id}.json"
        shared_sha = shared_split.shard_metadata[image_id]["shard_sha256"]
        if _feature_shard_valid(
            detection_path,
            summary_path,
            metadata_path,
            image_id=image_id,
            materialization_fingerprint=materialization_fingerprint,
            shared_sha256=shared_sha,
        ):
            reused += 1
            continue
        frame, summary = _materialize_shared_image(
            shared_split, image_id, config
        )
        detection_temporary = detection_path.with_suffix(".parquet.tmp")
        frame.to_parquet(detection_temporary, index=False)
        detection_temporary.replace(detection_path)
        summary_temporary = summary_path.with_suffix(".parquet.tmp")
        pd.DataFrame([summary]).to_parquet(summary_temporary, index=False)
        summary_temporary.replace(summary_path)
        write_json(
            metadata_path,
            {
                "schema_version": 1,
                "image_id": int(image_id),
                "materialization_fingerprint": materialization_fingerprint,
                "shared_shard_sha256": shared_sha,
                "detection_sha256": sha256_file(detection_path),
                "image_summary_sha256": sha256_file(summary_path),
            },
        )
        recomputed += 1

    frames = [pd.read_parquet(shard_root / f"{image_id}.parquet") for image_id in image_ids]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    image_summaries = pd.concat(
        [pd.read_parquet(image_shard_root / f"{image_id}.parquet") for image_id in image_ids],
        ignore_index=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".parquet.tmp")
    combined.to_parquet(temporary, index=False)
    temporary.replace(output_path)
    image_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_temporary = image_output_path.with_suffix(".parquet.tmp")
    image_summaries.to_parquet(summary_temporary, index=False)
    summary_temporary.replace(image_output_path)
    with shared_split.request_metadata_path.open("r", encoding="utf-8") as handle:
        shared_request = json.load(handle)
    metadata = {
        "schema_version": 3,
        "split": split,
        "images_requested": len(image_ids),
        "images_completed": len(image_ids),
        "detections": len(combined),
        "features_sha256": sha256_file(output_path),
        "feature_columns": list(combined.columns),
        "ground_truth_objects": int(image_summaries["ground_truth_objects"].sum()),
        "false_negatives_at_extraction_threshold": int(
            image_summaries["false_negatives"].sum()
        ),
        "image_summary_path": str(image_output_path),
        "image_summary_sha256": sha256_file(image_output_path),
        "elapsed_seconds": time.perf_counter() - started,
        "materialization_fingerprint": materialization_fingerprint,
        "extraction_fingerprint": materialization_fingerprint,
        "source_tree_sha256": source_sha256,
        "annotation_path": str(dataset.annotation_path),
        "annotation_sha256": dataset.sha256,
        "shared_fingerprint": shared_split.fingerprint,
        "shared_cache_namespace": shared_split.namespace,
        "shared_request_metadata_path": str(shared_split.request_metadata_path),
        "shared_request_metadata_sha256": sha256_file(
            shared_split.request_metadata_path
        ),
        "shared_shard_inventory_sha256": shared_request[
            "shard_inventory_sha256"
        ],
        "shared_shards_computed": shared_request["shards_computed"],
        "shared_shards_reused": shared_request["shards_reused"],
        "feature_shards_recomputed": recomputed,
        "feature_shards_reused": reused,
        "checkpoint_sha256": shared_request["checkpoint_sha256"],
        "timing_seconds": {
            column: float(image_summaries[column].sum())
            for column in (
                "preprocess_seconds",
                "deterministic_seconds",
                "stochastic_seconds",
                "aggregation_seconds",
                "total_seconds",
            )
        },
        "peak_gpu_memory_bytes": int(image_summaries["peak_gpu_memory_bytes"].max()),
        "stochastic_modules": shared_split.shard_metadata[image_ids[0]][
            "enabled_stochastic_modules"
        ],
        "environment": environment_metadata(config["_meta"]["project_root"]),
    }
    write_json(output_path.with_suffix(".metadata.json"), metadata)
    return output_path
