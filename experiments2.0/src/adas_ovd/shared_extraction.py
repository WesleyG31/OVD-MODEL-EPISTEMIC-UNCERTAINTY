from __future__ import annotations

import importlib.metadata
import inspect
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import project_path
from .data import CocoDataset, load_manifest_ids
from .groundingdino_adapter import GroundingDinoAdapter, resolve_package_resource
from .matching import associate_detections, match_predictions_to_ground_truth
from .reproducibility import (
    environment_metadata,
    seed_everything,
    sha256_file,
    source_tree_sha256,
    stable_fingerprint,
    write_json,
)


SHARED_SOURCE_PATHS = (
    "src/adas_ovd/groundingdino_adapter.py",
    "src/adas_ovd/matching.py",
    "src/adas_ovd/reproducibility.py",
)

INFERENCE_COMPATIBILITY_KEYS = (
    "mc_passes",
    "mc_seed_stride",
    "stochastic_module_types",
    "candidate_threshold",
    "association_iou",
    "association_class_penalty",
    "unmatched_cost",
)

SUPPORTED_SCHEMA_VERSION = 1
SUPPORTED_FORMAT = "npz_compressed"

ARRAY_SCHEMA = {
    "reference_boxes_cxcywh": "float32[n_reference,4]",
    "reference_boxes_xyxy": "float64[n_reference,4]",
    "reference_category_scores": "float32[n_reference,n_classes]",
    "reference_scores": "float32[n_reference]",
    "reference_category_indices": "int64[n_reference]",
    "reference_query_indices": "int64[n_reference]",
    "deterministic_reference_variance": "float64[n_reference]",
    "deterministic_reference_step": "float64[n_reference]",
    "deterministic_hidden_step": "float64[n_reference]",
    "present": "bool[mc_passes,n_reference]",
    "mc_category_scores": "float32[mc_passes,n_reference,n_classes]",
    "mc_scores": "float32[mc_passes,n_reference]",
    "mc_boxes_cxcywh": "float32[mc_passes,n_reference,4]",
    "mc_embeddings": "float32[mc_passes,n_reference,hidden_dimension]",
    "mc_reference_variance": "float64[mc_passes,n_reference]",
    "mc_reference_step": "float64[mc_passes,n_reference]",
    "mc_hidden_step": "float64[mc_passes,n_reference]",
}


@dataclass(frozen=True)
class SharedSplit:
    split: str
    image_ids: list[int]
    dataset: CocoDataset
    fingerprint: str
    namespace: str
    shard_paths: dict[int, Path]
    shard_metadata: dict[int, dict[str, Any]]
    request_metadata_path: Path


@dataclass(frozen=True)
class CommonMaterialization:
    frame: pd.DataFrame
    image_summary: dict[str, Any]
    arrays: dict[str, np.ndarray]


def validate_shared_configuration(config: dict[str, Any]) -> None:
    """Fail before GPU work when the frozen shared contract is incomplete."""
    if "shared_extraction" not in config:
        raise RuntimeError("Configuration has no shared_extraction block")
    shared = config["shared_extraction"]
    required = {
        "schema_version",
        "artifact_root",
        "cache_namespace",
        "format",
        *INFERENCE_COMPATIBILITY_KEYS,
    }
    missing = sorted(required - set(shared))
    if missing:
        raise RuntimeError(f"Shared extraction configuration is incomplete: {missing}")
    if int(shared["schema_version"]) != SUPPORTED_SCHEMA_VERSION:
        raise RuntimeError(
            "Unsupported shared extraction schema version: "
            f"{shared['schema_version']}. Add a coexisting versioned reader/writer "
            "instead of changing schema v1 in place."
        )
    if shared["format"] != SUPPORTED_FORMAT:
        raise RuntimeError(f"Unsupported shared extraction format: {shared['format']}")
    if int(shared["mc_passes"]) < 2:
        raise RuntimeError("Shared extraction requires at least two MC passes")
    if int(shared["mc_seed_stride"]) <= 0:
        raise RuntimeError("Shared extraction mc_seed_stride must be positive")
    if not shared["stochastic_module_types"]:
        raise RuntimeError("Shared extraction requires a stochastic module type")
    if not 0.0 <= float(shared["candidate_threshold"]) <= 1.0:
        raise RuntimeError("Shared candidate_threshold must lie in [0, 1]")
    if not 0.0 <= float(shared["association_iou"]) <= 1.0:
        raise RuntimeError("Shared association_iou must lie in [0, 1]")


def _dataset_for_split(config: dict[str, Any], split: str) -> CocoDataset:
    annotation_key = (
        "evaluation_annotations" if split == "test" else "calibration_annotations"
    )
    return CocoDataset(
        project_path(config, config["data"][annotation_key]),
        project_path(config, config["data"]["images_dir"]),
    )


def validate_consumer_compatibility(config: dict[str, Any], rq_name: str) -> None:
    validate_shared_configuration(config)
    if rq_name not in config or "extraction" not in config[rq_name]:
        raise RuntimeError(
            f"{rq_name.upper()} must declare an extraction block compatible "
            "with shared_extraction"
        )
    shared = config["shared_extraction"]
    consumer = config[rq_name]["extraction"]
    mismatches = {
        key: {"shared": shared.get(key), rq_name: consumer.get(key)}
        for key in INFERENCE_COMPATIBILITY_KEYS
        if shared.get(key) != consumer.get(key)
    }
    if mismatches:
        raise RuntimeError(
            f"{rq_name.upper()} is incompatible with shared extraction: {mismatches}"
        )


def _provenance_images(config: dict[str, Any]) -> tuple[Path, dict[str, dict[str, Any]]]:
    path = project_path(config, "artifacts/data_provenance.json")
    if not path.is_file():
        raise FileNotFoundError(f"Data provenance is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    inventory = {record["file_name"]: record for record in payload["images"]}
    return path, inventory


def _trajectory_statistics(
    hidden_states: np.ndarray, reference_points: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hidden = np.asarray(hidden_states)
    references = np.asarray(reference_points)
    n_candidates = hidden.shape[1]
    if n_candidates == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty.copy(), empty.copy()
    reference_variance = np.empty(n_candidates, dtype=np.float64)
    reference_step = np.empty(n_candidates, dtype=np.float64)
    hidden_step = np.empty(n_candidates, dtype=np.float64)
    # Preserve the exact per-detection reduction order used by RQ1/RQ2. The
    # loop is negligible beside GPU inference and avoids architecture changes
    # introducing ~1e-9 differences in frozen feature artifacts.
    for index in range(n_candidates):
        # Convert each slice independently, exactly as the original RQ
        # functions do. Converting the complete 3-D tensor first leaves a
        # strided float64 view and changes reduction rounding by ~1e-9.
        trajectory = np.asarray(
            references[:, index, :], dtype=np.float64
        )
        representation = np.asarray(
            hidden[:, index, :], dtype=np.float64
        )
        reference_variance[index] = float(np.var(trajectory, axis=0).mean())
        reference_step[index] = (
            float(np.linalg.norm(np.diff(trajectory, axis=0), axis=1).mean())
            if len(trajectory) > 1
            else 0.0
        )
        if len(representation) > 1:
            normalized = representation / np.clip(
                np.linalg.norm(representation, axis=1, keepdims=True),
                1e-12,
                None,
            )
            hidden_step[index] = float(
                (1.0 - (normalized[:-1] * normalized[1:]).sum(axis=1)).mean()
            )
        else:
            hidden_step[index] = 0.0
    return reference_variance, reference_step, hidden_step


def _synchronize(adapter: GroundingDinoAdapter) -> None:
    if adapter.device.type == "cuda":
        adapter.torch.cuda.synchronize(adapter.device)


def _build_adapter(config: dict[str, Any]) -> GroundingDinoAdapter:
    model_configuration = str(config["model"]["config"])
    model_configuration_path = (
        resolve_package_resource(model_configuration)
        if model_configuration.startswith("package://")
        else project_path(config, model_configuration)
    )
    adapter = GroundingDinoAdapter(
        model_configuration_path,
        project_path(config, config["model"]["checkpoint"]),
        project_path(config, config["model"]["text_encoder"]["local_dir"]),
        config["data"]["classes"],
        config["shared_extraction"]["stochastic_module_types"],
        config["model"]["device"],
        bool(config["model"]["amp"]),
    )
    if adapter.device.type != "cuda":
        adapter.close()
        raise RuntimeError("Shared detector extraction requires CUDA")
    return adapter


def _inference_implementation_sha256() -> str:
    """Hash only code that can change canonical detector shard contents.

    Receipt handling and label-dependent materialization deliberately remain
    outside this hash, so future RQ/report work cannot trigger an unnecessary
    multi-hour GPU recomputation.
    """
    functions = (
        _dataset_for_split,
        _provenance_images,
        _trajectory_statistics,
        _synchronize,
        _build_adapter,
        _infer_image,
        _validate_array_schema,
        _shard_is_valid,
        _write_shard,
        load_shared_shard,
    )
    return stable_fingerprint(
        {
            "functions": {
                function.__name__: inspect.getsource(function)
                for function in functions
            },
            "array_schema": ARRAY_SCHEMA,
            "compatibility_keys": INFERENCE_COMPATIBILITY_KEYS,
            "supported_schema_version": SUPPORTED_SCHEMA_VERSION,
            "supported_format": SUPPORTED_FORMAT,
        }
    )


def shared_identity(config: dict[str, Any]) -> dict[str, Any]:
    validate_shared_configuration(config)
    shared = config["shared_extraction"]
    checkpoint = project_path(config, config["model"]["checkpoint"])
    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 != str(config["model"]["checkpoint_sha256"]).lower():
        raise RuntimeError("Shared extraction checkpoint SHA-256 mismatch")
    encoder_root = project_path(
        config, config["model"]["text_encoder"]["local_dir"]
    )
    encoder_hashes = {
        name: sha256_file(encoder_root / name)
        for name in config["model"]["text_encoder"]["required_files"]
    }
    provenance_path, _ = _provenance_images(config)
    external_source_sha256 = source_tree_sha256(
        config["_meta"]["project_root"], SHARED_SOURCE_PATHS
    )
    implementation_sha256 = _inference_implementation_sha256()
    source_sha256 = stable_fingerprint(
        {
            "external_source_tree_sha256": external_source_sha256,
            "inference_implementation_sha256": implementation_sha256,
        }
    )
    versions = {}
    for name in ("groundingdino-py", "torch", "torchvision", "transformers", "numpy"):
        versions[name] = importlib.metadata.version(name)
    configuration = {
        "schema_version": int(shared["schema_version"]),
        "format": shared["format"],
        "project_seed": int(config["project"]["seed"]),
        "deterministic_algorithms": bool(
            config["project"]["deterministic_algorithms"]
        ),
        "deterministic_warn_only": bool(
            config["project"]["deterministic_warn_only"]
        ),
        "model": {
            key: config["model"][key]
            for key in (
                "package",
                "package_version",
                "config",
                "device",
                "amp",
                "box_threshold",
                "reference_max_detections",
                "max_detections",
                "nms_iou",
            )
        },
        "checkpoint_sha256": checkpoint_sha256,
        "text_encoder_repository": config["model"]["text_encoder"]["repository"],
        "text_encoder_revision": config["model"]["text_encoder"]["revision"],
        "text_encoder_file_sha256": encoder_hashes,
        "classes": config["data"]["classes"],
        "shared_extraction": {
            key: shared[key] for key in INFERENCE_COMPATIBILITY_KEYS
        },
        "array_schema": ARRAY_SCHEMA,
        "runtime_versions": versions,
        "data_provenance_sha256": sha256_file(provenance_path),
        "source_tree_sha256": source_sha256,
        "external_source_tree_sha256": external_source_sha256,
        "inference_implementation_sha256": implementation_sha256,
    }
    return {
        "schema_version": int(shared["schema_version"]),
        "configuration": configuration,
        "configuration_fingerprint": stable_fingerprint(configuration),
        "source_tree_sha256": source_sha256,
        "external_source_tree_sha256": external_source_sha256,
        "inference_implementation_sha256": implementation_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "data_provenance_sha256": sha256_file(provenance_path),
    }


def _infer_image(
    adapter: GroundingDinoAdapter,
    dataset: CocoDataset,
    image_id: int,
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    shared = config["shared_extraction"]
    model = config["model"]
    image = dataset.images[image_id]
    image_path = dataset.image_path(image_id)
    if not image_path.is_file():
        raise FileNotFoundError(f"Shared extraction image is missing: {image_path}")
    if adapter.device.type == "cuda":
        adapter.torch.cuda.reset_peak_memory_stats(adapter.device)
    preprocess_started = time.perf_counter()
    _, tensor = adapter.preprocess(image_path)
    preprocess_seconds = time.perf_counter() - preprocess_started
    seed = int(config["project"]["seed"]) + int(image_id)
    deterministic = bool(config["project"]["deterministic_algorithms"])
    warn_only = bool(config["project"]["deterministic_warn_only"])
    seed_everything(seed, deterministic, warn_only)
    _synchronize(adapter)
    deterministic_started = time.perf_counter()
    reference = adapter.run(
        tensor,
        image.width,
        image.height,
        float(model["box_threshold"]),
        int(model["reference_max_detections"]),
    )
    _synchronize(adapter)
    deterministic_seconds = time.perf_counter() - deterministic_started

    n_reference = len(reference.scores)
    passes = int(shared["mc_passes"])
    n_classes = len(config["data"]["classes"])
    hidden_dimension = int(reference.hidden_states.shape[-1])
    deterministic_statistics = _trajectory_statistics(
        reference.hidden_states, reference.reference_points
    )
    present = np.zeros((passes, n_reference), dtype=bool)
    mc_category_scores = np.full(
        (passes, n_reference, n_classes), np.nan, dtype=np.float32
    )
    mc_scores = np.full((passes, n_reference), np.nan, dtype=np.float32)
    mc_boxes = np.full((passes, n_reference, 4), np.nan, dtype=np.float32)
    mc_embeddings = np.full(
        (passes, n_reference, hidden_dimension), np.nan, dtype=np.float32
    )
    mc_reference_variance = np.full(
        (passes, n_reference), np.nan, dtype=np.float64
    )
    mc_reference_step = np.full((passes, n_reference), np.nan, dtype=np.float64)
    mc_hidden_step = np.full((passes, n_reference), np.nan, dtype=np.float64)
    mc_candidates_before_cap = np.zeros(passes, dtype=np.int64)
    mc_candidates_retained = np.zeros(passes, dtype=np.int64)
    mc_protected_candidates = np.zeros(passes, dtype=np.int64)

    _synchronize(adapter)
    stochastic_started = time.perf_counter()
    with adapter.stochastic_mode():
        for pass_index in range(passes):
            pass_seed = seed + (pass_index + 1) * int(shared["mc_seed_stride"])
            seed_everything(pass_seed, deterministic, warn_only)
            stochastic = adapter.run(
                tensor,
                image.width,
                image.height,
                float(shared["candidate_threshold"]),
                int(model["max_detections"]),
                required_query_indices=reference.query_indices,
            )
            mc_candidates_before_cap[pass_index] = stochastic.candidates_before_cap
            mc_candidates_retained[pass_index] = stochastic.candidates_retained
            mc_protected_candidates[pass_index] = (
                stochastic.protected_candidates_retained
            )
            statistics = _trajectory_statistics(
                stochastic.hidden_states, stochastic.reference_points
            )
            association = associate_detections(
                reference.boxes_xyxy,
                reference.category_indices,
                stochastic.boxes_xyxy,
                stochastic.category_indices,
                float(shared["association_iou"]),
                float(shared["association_class_penalty"]),
                float(shared["unmatched_cost"]),
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
                mc_embeddings[pass_index, reference_index] = (
                    stochastic.hidden_states[-1, candidate_index]
                )
                mc_reference_variance[pass_index, reference_index] = statistics[0][
                    candidate_index
                ]
                mc_reference_step[pass_index, reference_index] = statistics[1][
                    candidate_index
                ]
                mc_hidden_step[pass_index, reference_index] = statistics[2][
                    candidate_index
                ]
    _synchronize(adapter)
    stochastic_seconds = time.perf_counter() - stochastic_started
    total_seconds = preprocess_seconds + deterministic_seconds + stochastic_seconds
    arrays = {
        "reference_boxes_cxcywh": np.asarray(
            reference.boxes_cxcywh, dtype=np.float32
        ),
        "reference_boxes_xyxy": np.asarray(reference.boxes_xyxy, dtype=np.float64),
        "reference_category_scores": np.asarray(
            reference.category_scores, dtype=np.float32
        ),
        "reference_scores": np.asarray(reference.scores, dtype=np.float32),
        "reference_category_indices": np.asarray(
            reference.category_indices, dtype=np.int64
        ),
        "reference_query_indices": np.asarray(reference.query_indices, dtype=np.int64),
        "deterministic_reference_variance": deterministic_statistics[0],
        "deterministic_reference_step": deterministic_statistics[1],
        "deterministic_hidden_step": deterministic_statistics[2],
        "present": present,
        "mc_category_scores": mc_category_scores,
        "mc_scores": mc_scores,
        "mc_boxes_cxcywh": mc_boxes,
        "mc_embeddings": mc_embeddings,
        "mc_reference_variance": mc_reference_variance,
        "mc_reference_step": mc_reference_step,
        "mc_hidden_step": mc_hidden_step,
    }
    summary = {
        "image_id": int(image_id),
        "file_name": image.file_name,
        "sequence_id": image.sequence_id,
        "reference_detections": n_reference,
        "preprocess_seconds": preprocess_seconds,
        "deterministic_seconds": deterministic_seconds,
        "stochastic_seconds": stochastic_seconds,
        "shared_inference_seconds": total_seconds,
        "peak_gpu_memory_bytes": (
            int(adapter.torch.cuda.max_memory_allocated(adapter.device))
            if adapter.device.type == "cuda"
            else 0
        ),
        "deterministic_seed": seed,
        "mc_seed_stride": int(shared["mc_seed_stride"]),
        "mc_passes": passes,
        "candidate_cap": int(model["max_detections"]),
        "reference_candidate_cap": int(model["reference_max_detections"]),
        "deterministic_candidates_before_cap": reference.candidates_before_cap,
        "deterministic_candidates_retained": reference.candidates_retained,
        "deterministic_candidate_truncated": bool(
            reference.candidates_before_cap > reference.candidates_retained
        ),
        "mc_candidates_before_cap_min": int(mc_candidates_before_cap.min()),
        "mc_candidates_before_cap_mean": float(mc_candidates_before_cap.mean()),
        "mc_candidates_before_cap_max": int(mc_candidates_before_cap.max()),
        "mc_candidates_retained_min": int(mc_candidates_retained.min()),
        "mc_candidates_retained_mean": float(mc_candidates_retained.mean()),
        "mc_candidates_retained_max": int(mc_candidates_retained.max()),
        "mc_passes_above_nominal_cap": int(
            np.count_nonzero(mc_candidates_before_cap > int(model["max_detections"]))
        ),
        "mc_protected_candidates_mean": float(mc_protected_candidates.mean()),
        "reference_query_protection": True,
    }
    return arrays, summary


def _validate_array_schema(
    arrays: dict[str, np.ndarray], config: dict[str, Any]
) -> None:
    if set(arrays) != set(ARRAY_SCHEMA):
        raise RuntimeError("Shared inference array names differ from schema")
    n_reference = len(arrays["reference_scores"])
    passes = int(config["shared_extraction"]["mc_passes"])
    n_classes = len(config["data"]["classes"])
    if arrays["reference_boxes_xyxy"].shape != (n_reference, 4):
        raise RuntimeError("Invalid shared reference box shape")
    if arrays["present"].shape != (passes, n_reference):
        raise RuntimeError("Invalid shared MC presence shape")
    if arrays["mc_category_scores"].shape != (
        passes,
        n_reference,
        n_classes,
    ):
        raise RuntimeError("Invalid shared category-score shape")
    if arrays["mc_embeddings"].shape[:2] != (passes, n_reference):
        raise RuntimeError("Invalid shared embedding shape")
    for name in (
        "reference_category_indices",
        "reference_query_indices",
    ):
        if arrays[name].dtype != np.int64:
            raise RuntimeError(f"Invalid shared dtype for {name}")
    if arrays["present"].dtype != np.bool_:
        raise RuntimeError("Invalid shared presence dtype")
    float64_names = {
        "reference_boxes_xyxy",
        "deterministic_reference_variance",
        "deterministic_reference_step",
        "deterministic_hidden_step",
        "mc_reference_variance",
        "mc_reference_step",
        "mc_hidden_step",
    }
    for name, values in arrays.items():
        if name in {"present", "reference_category_indices", "reference_query_indices"}:
            continue
        expected_dtype = np.float64 if name in float64_names else np.float32
        if values.dtype != expected_dtype:
            raise RuntimeError(f"Invalid shared dtype for {name}")


def _shard_is_valid(
    path: Path,
    metadata_path: Path,
    *,
    image_id: int,
    file_name: str,
    image_record: dict[str, Any],
    fingerprint: str,
) -> bool:
    if not path.is_file() or not metadata_path.is_file():
        return False
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        return bool(
            metadata.get("image_id") == int(image_id)
            and metadata.get("file_name") == file_name
            and metadata.get("image_sha256") == image_record["sha256"]
            and metadata.get("image_bytes") == int(image_record["bytes"])
            and metadata.get("shared_fingerprint") == fingerprint
            and metadata.get("array_schema") == ARRAY_SCHEMA
            and metadata.get("shard_sha256") == sha256_file(path)
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _write_shard(
    path: Path,
    metadata_path: Path,
    arrays: dict[str, np.ndarray],
    summary: dict[str, Any],
    image_record: dict[str, Any],
    fingerprint: str,
    enabled_modules: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    _validate_array_schema(arrays, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)
    write_json(
        metadata_path,
        {
            "schema_version": int(config["shared_extraction"]["schema_version"]),
            **summary,
            "image_sha256": image_record["sha256"],
            "image_bytes": int(image_record["bytes"]),
            "shared_fingerprint": fingerprint,
            "array_schema": ARRAY_SCHEMA,
            "enabled_stochastic_modules": enabled_modules,
            "shard_sha256": sha256_file(path),
            "shard_bytes": path.stat().st_size,
        },
    )


def load_shared_shard(
    path: Path,
    metadata: dict[str, Any],
    config: dict[str, Any],
    expected_fingerprint: str,
) -> dict[str, np.ndarray]:
    if metadata.get("shared_fingerprint") != expected_fingerprint:
        raise RuntimeError(f"Shared shard fingerprint mismatch: {path}")
    if metadata.get("shard_sha256") != sha256_file(path):
        raise RuntimeError(f"Shared shard SHA-256 mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    _validate_array_schema(arrays, config)
    return arrays


def ensure_shared_split(
    config: dict[str, Any],
    *,
    manifest_path: str | Path,
    split: str,
    limit: int | None = None,
    image_ids_override: list[int] | tuple[int, ...] | None = None,
    cache_namespace: str | None = None,
    force_recompute: bool = False,
) -> SharedSplit:
    validate_shared_configuration(config)
    receipt_created_ns = time.time_ns()
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unknown shared extraction split: {split}")
    manifest_path = Path(manifest_path).resolve()
    manifest_image_ids = load_manifest_ids(manifest_path, split)
    if image_ids_override is not None and limit is not None:
        raise ValueError("Shared extraction cannot combine limit and image override")
    if image_ids_override is not None:
        image_ids = [int(value) for value in image_ids_override]
        if not image_ids:
            raise ValueError("Shared extraction image override cannot be empty")
        if len(image_ids) != len(set(image_ids)):
            raise ValueError("Shared extraction image override contains duplicates")
        outside = sorted(set(image_ids) - set(manifest_image_ids))
        if outside:
            raise ValueError(
                f"Shared extraction image override is outside {split}: {outside[:10]}"
            )
    else:
        image_ids = manifest_image_ids
    if limit is not None:
        if int(limit) <= 0:
            raise ValueError("Shared extraction limit must be positive")
        image_ids = image_ids[: int(limit)]
    dataset = _dataset_for_split(config, split)
    identity = shared_identity(config)
    fingerprint = identity["configuration_fingerprint"]
    namespace = str(
        cache_namespace or config["shared_extraction"]["cache_namespace"]
    )
    source_partition = "evaluation" if split == "test" else "calibration"
    root = (
        project_path(config, config["shared_extraction"]["artifact_root"])
        / namespace
        / fingerprint
        / source_partition
    )
    shard_paths = {image_id: root / f"{image_id}.npz" for image_id in image_ids}
    metadata_paths = {image_id: root / f"{image_id}.json" for image_id in image_ids}
    provenance_path, inventory = _provenance_images(config)
    missing: list[int] = []
    for image_id in image_ids:
        file_name = dataset.images[image_id].file_name
        if file_name not in inventory:
            raise RuntimeError(f"Image is absent from provenance: {file_name}")
        if force_recompute or not _shard_is_valid(
            shard_paths[image_id],
            metadata_paths[image_id],
            image_id=image_id,
            file_name=file_name,
            image_record=inventory[file_name],
            fingerprint=fingerprint,
        ):
            missing.append(image_id)

    started = time.perf_counter()
    enabled_modules: list[dict[str, Any]] = []
    if missing:
        seed_everything(
            int(config["project"]["seed"]),
            bool(config["project"]["deterministic_algorithms"]),
            bool(config["project"]["deterministic_warn_only"]),
        )
        adapter = _build_adapter(config)
        try:
            for image_id in tqdm(missing, desc=f"Shared GPU inference [{split}]"):
                arrays, summary = _infer_image(adapter, dataset, image_id, config)
                enabled_modules = list(adapter.enabled_stochastic_modules)
                file_name = dataset.images[image_id].file_name
                _write_shard(
                    shard_paths[image_id],
                    metadata_paths[image_id],
                    arrays,
                    summary,
                    inventory[file_name],
                    fingerprint,
                    enabled_modules,
                    config,
                )
        finally:
            adapter.close()

    shard_metadata: dict[int, dict[str, Any]] = {}
    invalid: list[int] = []
    for image_id in image_ids:
        file_name = dataset.images[image_id].file_name
        if not _shard_is_valid(
            shard_paths[image_id],
            metadata_paths[image_id],
            image_id=image_id,
            file_name=file_name,
            image_record=inventory[file_name],
            fingerprint=fingerprint,
        ):
            invalid.append(image_id)
            continue
        with metadata_paths[image_id].open("r", encoding="utf-8") as handle:
            shard_metadata[image_id] = json.load(handle)
    if invalid:
        raise RuntimeError(f"Invalid shared inference shards: {invalid[:10]}")

    request_identity = {
        "schema_version": int(config["shared_extraction"]["schema_version"]),
        "shared_fingerprint": fingerprint,
        "namespace": namespace,
        "source_partition": source_partition,
        "split": split,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "image_ids_sha256": stable_fingerprint(image_ids),
        "images": len(image_ids),
    }
    request_name = stable_fingerprint(request_identity)
    # A consumer invocation is an immutable receipt, not mutable cache state.
    # RQ2 must never overwrite the exact receipt whose hash RQ1 recorded.
    request_path = (
        root.parent
        / "requests"
        / request_name
        / f"{receipt_created_ns}.json"
    )
    write_json(
        request_path,
        {
            **request_identity,
            "receipt_created_ns": receipt_created_ns,
            **identity,
            "array_schema": ARRAY_SCHEMA,
            "data_provenance_path": str(provenance_path),
            "shards_reused": len(image_ids) - len(missing),
            "shards_computed": len(missing),
            "elapsed_seconds": time.perf_counter() - started,
            "shard_inventory_sha256": stable_fingerprint(
                [
                    {
                        "image_id": image_id,
                        "shard_sha256": shard_metadata[image_id]["shard_sha256"],
                    }
                    for image_id in image_ids
                ]
            ),
            "total_shard_bytes": int(
                sum(shard_metadata[image_id]["shard_bytes"] for image_id in image_ids)
            ),
            "environment": environment_metadata(config["_meta"]["project_root"]),
        },
    )
    return SharedSplit(
        split=split,
        image_ids=image_ids,
        dataset=dataset,
        fingerprint=fingerprint,
        namespace=namespace,
        shard_paths=shard_paths,
        shard_metadata=shard_metadata,
        request_metadata_path=request_path,
    )


def _object_size(box_xyxy: np.ndarray) -> tuple[float, str]:
    width = max(float(box_xyxy[2] - box_xyxy[0]), 0.0)
    height = max(float(box_xyxy[3] - box_xyxy[1]), 0.0)
    area = width * height
    if area < 32.0**2:
        return area, "small"
    if area < 96.0**2:
        return area, "medium"
    return area, "large"


def materialize_common(
    shared_split: SharedSplit,
    image_id: int,
    config: dict[str, Any],
) -> CommonMaterialization:
    dataset = shared_split.dataset
    image = dataset.images[image_id]
    arrays = load_shared_shard(
        shared_split.shard_paths[image_id],
        shared_split.shard_metadata[image_id],
        config,
        shared_split.fingerprint,
    )
    classes = list(config["data"]["classes"])
    missing_classes = [
        name for name in classes if name not in dataset.category_ids_by_name
    ]
    if missing_classes:
        raise ValueError(f"Shared classes absent from annotations: {missing_classes}")
    category_ids = np.asarray(
        [dataset.category_ids_by_name[name] for name in classes], dtype=np.int64
    )
    ground_truth = dataset.ground_truth(image_id)
    gt_boxes = np.asarray(
        [record.bbox_xyxy for record in ground_truth], dtype=np.float64
    ).reshape(-1, 4)
    gt_categories = np.asarray(
        [record.category_id for record in ground_truth], dtype=np.int64
    )
    reference_categories = category_ids[arrays["reference_category_indices"]]
    correctness = match_predictions_to_ground_truth(
        arrays["reference_boxes_xyxy"],
        arrays["reference_scores"],
        reference_categories,
        gt_boxes,
        gt_categories,
        float(config["evaluation"]["match_iou"]),
    )
    rows: list[dict[str, Any]] = []
    for detection_index in range(len(arrays["reference_scores"])):
        box = arrays["reference_boxes_xyxy"][detection_index]
        area, size = _object_size(box)
        category_index = int(arrays["reference_category_indices"][detection_index])
        rows.append(
            {
                "image_id": int(image_id),
                "file_name": image.file_name,
                "sequence_id": image.sequence_id,
                "timeofday": image.attributes.get("timeofday", "unknown"),
                "weather": image.attributes.get("weather", "unknown"),
                "scene": image.attributes.get("scene", "unknown"),
                "detection_index": detection_index,
                "query_index": int(arrays["reference_query_indices"][detection_index]),
                "category_index": category_index,
                "category_id": int(reference_categories[detection_index]),
                "category_name": classes[category_index],
                "score": float(arrays["reference_scores"][detection_index]),
                "confidence_uncertainty": float(
                    1.0 - arrays["reference_scores"][detection_index]
                ),
                "bbox_x1": float(box[0]),
                "bbox_y1": float(box[1]),
                "bbox_x2": float(box[2]),
                "bbox_y2": float(box[3]),
                "bbox_area": area,
                "object_size": size,
                "is_true_positive": bool(
                    correctness.is_true_positive[detection_index]
                ),
                "is_error": int(not correctness.is_true_positive[detection_index]),
                "matched_iou": float(correctness.matched_iou[detection_index]),
                "matched_ground_truth_index": int(
                    correctness.matched_ground_truth[detection_index]
                ),
                "false_negatives_image": int(correctness.false_negatives),
                "mc_matches": int(arrays["present"][:, detection_index].sum()),
                "mc_passes": int(config["shared_extraction"]["mc_passes"]),
            }
        )
    shared_metadata = shared_split.shard_metadata[image_id]
    summary = {
        "image_id": int(image_id),
        "file_name": image.file_name,
        "sequence_id": image.sequence_id,
        "timeofday": image.attributes.get("timeofday", "unknown"),
        "weather": image.attributes.get("weather", "unknown"),
        "scene": image.attributes.get("scene", "unknown"),
        "ground_truth_objects": len(ground_truth),
        "reference_detections": len(rows),
        "true_positive_detections": int(correctness.is_true_positive.sum()),
        "false_positive_detections": int((~correctness.is_true_positive).sum()),
        "false_negatives": int(correctness.false_negatives),
        "preprocess_seconds": float(shared_metadata["preprocess_seconds"]),
        "deterministic_seconds": float(shared_metadata["deterministic_seconds"]),
        "stochastic_seconds": float(shared_metadata["stochastic_seconds"]),
        "shared_inference_seconds": float(shared_metadata["shared_inference_seconds"]),
        "peak_gpu_memory_bytes": int(shared_metadata["peak_gpu_memory_bytes"]),
        "candidate_cap": int(shared_metadata["candidate_cap"]),
        "reference_candidate_cap": int(
            shared_metadata["reference_candidate_cap"]
        ),
        "deterministic_candidates_before_cap": int(
            shared_metadata["deterministic_candidates_before_cap"]
        ),
        "deterministic_candidates_retained": int(
            shared_metadata["deterministic_candidates_retained"]
        ),
        "deterministic_candidate_truncated": bool(
            shared_metadata["deterministic_candidate_truncated"]
        ),
        "mc_candidates_before_cap_min": int(
            shared_metadata["mc_candidates_before_cap_min"]
        ),
        "mc_candidates_before_cap_mean": float(
            shared_metadata["mc_candidates_before_cap_mean"]
        ),
        "mc_candidates_before_cap_max": int(
            shared_metadata["mc_candidates_before_cap_max"]
        ),
        "mc_candidates_retained_min": int(
            shared_metadata["mc_candidates_retained_min"]
        ),
        "mc_candidates_retained_mean": float(
            shared_metadata["mc_candidates_retained_mean"]
        ),
        "mc_candidates_retained_max": int(
            shared_metadata["mc_candidates_retained_max"]
        ),
        "mc_passes_above_nominal_cap": int(
            shared_metadata["mc_passes_above_nominal_cap"]
        ),
        "mc_protected_candidates_mean": float(
            shared_metadata["mc_protected_candidates_mean"]
        ),
        "reference_query_protection": bool(
            shared_metadata["reference_query_protection"]
        ),
        "shared_shard_sha256": shared_metadata["shard_sha256"],
        "shared_fingerprint": shared_split.fingerprint,
        "shared_cache_namespace": shared_split.namespace,
    }
    return CommonMaterialization(pd.DataFrame(rows), summary, arrays)
