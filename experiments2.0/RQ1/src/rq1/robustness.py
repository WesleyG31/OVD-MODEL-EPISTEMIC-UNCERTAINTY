from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
from scipy.stats import spearmanr

from adas_ovd.config import project_path
from adas_ovd.data import load_manifest_ids
from adas_ovd.metrics import binary_uncertainty_metrics
from adas_ovd.reproducibility import (
    environment_metadata,
    sha256_file,
    source_tree_sha256,
    stable_fingerprint,
    write_json,
)

from .extraction import _dataset_for_split, build_adapter, extract_image
from .fusion import load_fusions


def _corrupt_image(
    source: Path, destination: Path, name: str, severity: float
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as opened:
        image = opened.convert("RGB")
        if name == "gaussian_blur":
            transformed = image.filter(ImageFilter.GaussianBlur(radius=severity))
        elif name == "brightness":
            transformed = ImageEnhance.Brightness(image).enhance(severity)
        else:
            raise ValueError(f"Unsupported frozen corruption: {name}")
        transformed.save(destination, format="PNG", optimize=False)


def _paired_mean_interval(
    baseline: pd.Series,
    candidate: pd.Series,
    repetitions: int,
    confidence_level: float,
    seed: int,
) -> dict[str, float | int]:
    paired = pd.concat(
        [baseline.rename("baseline"), candidate.rename("candidate")], axis=1
    ).dropna()
    differences = (
        paired["candidate"] - paired["baseline"]
    ).to_numpy(dtype=np.float64)
    if len(differences) == 0:
        return {"paired_images": 0, "mean_difference": float("nan")}
    rng = np.random.default_rng(seed)
    values = np.asarray(
        [
            rng.choice(differences, size=len(differences), replace=True).mean()
            for _ in range(int(repetitions))
        ]
    )
    alpha = (1.0 - float(confidence_level)) / 2.0
    return {
        "paired_images": len(differences),
        "mean_difference": float(differences.mean()),
        "lower": float(np.quantile(values, alpha)),
        "upper": float(np.quantile(values, 1.0 - alpha)),
    }


def _condition_summary(
    detections: pd.DataFrame,
    images: pd.DataFrame,
    threshold: float,
    annotations_path: Path,
) -> dict[str, Any]:
    operational = detections.loc[detections["score"] >= threshold].copy()
    labels = operational["is_error"].to_numpy(dtype=np.int64)
    true_positives = int((labels == 0).sum())
    false_positives = int((labels == 1).sum())
    ground_truth = int(images["ground_truth_objects"].sum())
    false_negatives = max(ground_truth - true_positives, 0)
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = true_positives / ground_truth if ground_truth else 0.0
    record: dict[str, Any] = {
        "images": int(images["image_id"].nunique()),
        "detections": len(operational),
        "error_prevalence": float(labels.mean()) if len(labels) else None,
        "mean_uncertainty": (
            float(operational["primary_uncertainty"].mean())
            if len(operational)
            else None
        ),
        "precision_at_operating_threshold": precision,
        "recall_at_operating_threshold": recall,
        "false_negatives_at_operating_threshold": false_negatives,
        "total_seconds": float(images["total_seconds"].sum()),
        "peak_gpu_memory_bytes": int(images["peak_gpu_memory_bytes"].max()),
    }
    if len(detections):
        from .evaluation import _coco_detection_metrics

        record["detector_coco"] = _coco_detection_metrics(
            annotations_path, detections
        )
    if len(np.unique(labels)) == 2:
        metrics = binary_uncertainty_metrics(
            labels, operational["primary_uncertainty"].to_numpy()
        )
        record.update(
            {
                "uncertainty_auroc": metrics["auroc"],
                "uncertainty_auprc": metrics["auprc"],
                "uncertainty_aurc": metrics["aurc"],
            }
        )
    return record


def _extract_condition(
    config: dict[str, Any],
    image_ids: list[int],
    *,
    family: str,
    condition: str,
    prompt_phrases: list[str] | None = None,
    image_overrides: dict[int, Path] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = _dataset_for_split(config, "validation")
    adapter = build_adapter(config, prompt_phrases=prompt_phrases)
    frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    try:
        for image_id in image_ids:
            frame, summary = extract_image(
                adapter,
                dataset,
                image_id,
                config,
                image_path_override=(
                    image_overrides.get(image_id)
                    if image_overrides is not None
                    else None
                ),
            )
            frame["robustness_family"] = family
            frame["robustness_condition"] = condition
            summary["robustness_family"] = family
            summary["robustness_condition"] = condition
            frames.append(frame)
            summaries.append(summary)
    finally:
        adapter.close()
    detections = (
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    )
    return detections, pd.DataFrame(summaries)


def run_robustness(config: dict[str, Any]) -> dict[str, Any]:
    settings = config["rq1"]["robustness"]
    output_root = project_path(config, config["rq1"]["outputs"]["robustness"])
    output_root.mkdir(parents=True, exist_ok=True)
    image_ids = load_manifest_ids(
        project_path(config, config["rq1"]["outputs"]["manifest"]),
        "validation",
    )[: int(settings["validation_image_limit"])]
    if not image_ids:
        raise ValueError("Robustness evaluation requires validation images")

    primary_method = str(config["rq1"]["fusion"]["primary_method"])
    primary_model = load_fusions(config)[primary_method]
    primary_model_path = project_path(
        config, config["rq1"]["outputs"]["models"]
    ) / f"{primary_method}.joblib"
    conditions: list[tuple[pd.DataFrame, pd.DataFrame]] = []

    baseline_detections, baseline_images = _extract_condition(
        config,
        image_ids,
        family="baseline",
        condition="canonical",
    )
    conditions.append((baseline_detections, baseline_images))

    for offset in settings["independent_seed_offsets"]:
        offset = int(offset)
        if offset == 0:
            continue
        condition_config = deepcopy(config)
        condition_config["rq1"]["extraction"]["mc_seed_offset"] = offset
        conditions.append(
            _extract_condition(
                condition_config,
                image_ids,
                family="mc_seed",
                condition=str(offset),
            )
        )

    primary_iou = float(config["rq1"]["extraction"]["association_iou"])
    for iou in config["rq1"]["extraction"]["association_iou_sensitivity"]:
        iou = float(iou)
        if np.isclose(iou, primary_iou):
            continue
        condition_config = deepcopy(config)
        condition_config["rq1"]["extraction"]["association_iou"] = iou
        conditions.append(
            _extract_condition(
                condition_config,
                image_ids,
                family="association_iou",
                condition=f"{iou:.2f}",
            )
        )

    canonical_prompts = list(settings["prompt_sets"]["canonical"])
    if canonical_prompts != list(config["data"]["classes"]):
        raise ValueError("Frozen canonical prompts must equal canonical classes")
    for name, phrases in settings["prompt_sets"].items():
        if name == "canonical":
            continue
        conditions.append(
            _extract_condition(
                config,
                image_ids,
                family="prompt",
                condition=str(name),
                prompt_phrases=list(phrases),
            )
        )

    dataset = _dataset_for_split(config, "validation")
    corruption_root = output_root / "corrupted_images"
    corruption_hashes: dict[str, str] = {}
    for corruption in settings["corruptions"]:
        name = str(corruption["name"])
        for severity_value in corruption["severities"]:
            severity = float(severity_value)
            key = f"{name}_{severity:g}"
            overrides: dict[int, Path] = {}
            for image_id in image_ids:
                destination = corruption_root / key / f"{image_id}.png"
                _corrupt_image(
                    dataset.image_path(image_id), destination, name, severity
                )
                overrides[image_id] = destination
                corruption_hashes[f"{key}/{image_id}.png"] = sha256_file(
                    destination
                )
            conditions.append(
                _extract_condition(
                    config,
                    image_ids,
                    family="corruption",
                    condition=key,
                    image_overrides=overrides,
                )
            )

    detection_frames: list[pd.DataFrame] = []
    image_frames: list[pd.DataFrame] = []
    for detections, images in conditions:
        if not detections.empty:
            detections["primary_uncertainty"] = primary_model.rank_score(
                detections
            )
        detection_frames.append(detections)
        image_frames.append(images)
    all_detections = pd.concat(detection_frames, ignore_index=True)
    all_images = pd.concat(image_frames, ignore_index=True)

    threshold = float(config["evaluation"]["primary_score_threshold"])
    annotations_path = project_path(
        config, config["data"]["calibration_annotations"]
    )
    summaries: dict[str, Any] = {}
    baseline_operational = all_detections.loc[
        (all_detections["robustness_family"] == "baseline")
        & (all_detections["score"] >= threshold)
    ]
    baseline_by_image = baseline_operational.groupby("image_id")[
        "primary_uncertainty"
    ].mean()
    for (family, condition), detections in all_detections.groupby(
        ["robustness_family", "robustness_condition"], sort=False
    ):
        images = all_images.loc[
            (all_images["robustness_family"] == family)
            & (all_images["robustness_condition"] == condition)
        ]
        key = f"{family}:{condition}"
        record = _condition_summary(
            detections, images, threshold, annotations_path
        )
        operational = detections.loc[detections["score"] >= threshold]
        if family != "baseline":
            paired = baseline_operational.merge(
                operational,
                on=["image_id", "query_index"],
                suffixes=("_baseline", "_candidate"),
            )
            if len(paired) >= 3:
                correlation = spearmanr(
                    paired["primary_uncertainty_baseline"],
                    paired["primary_uncertainty_candidate"],
                ).statistic
                record["rank_spearman_vs_baseline"] = float(correlation)
                record["paired_detections"] = len(paired)
            candidate_by_image = operational.groupby("image_id")[
                "primary_uncertainty"
            ].mean()
            record["paired_image_uncertainty_change"] = _paired_mean_interval(
                baseline_by_image,
                candidate_by_image,
                repetitions=int(config["evaluation"]["bootstrap_repetitions"]),
                confidence_level=float(config["evaluation"]["confidence_level"]),
                seed=int(config["project"]["seed"])
                + sum(ord(character) for character in key),
            )
        summaries[key] = record

    detections_path = output_root / "robustness_features.parquet"
    images_path = output_root / "robustness_images.parquet"
    all_detections.to_parquet(detections_path, index=False)
    all_images.to_parquet(images_path, index=False)
    result: dict[str, Any] = {
        "schema_version": 1,
        "evidence_tier": "validation_robustness_diagnostic",
        "primary_method": primary_method,
        "primary_model_sha256": sha256_file(primary_model_path),
        "validation_images": len(image_ids),
        "image_ids_fingerprint": stable_fingerprint(image_ids),
        "operating_score_threshold": threshold,
        "conditions": summaries,
        "corrupted_image_sha256": corruption_hashes,
        "features_path": str(detections_path),
        "features_sha256": sha256_file(detections_path),
        "images_path": str(images_path),
        "images_sha256": sha256_file(images_path),
        "source_tree_sha256": source_tree_sha256(
            config["_meta"]["project_root"],
            ("RQ1/src/rq1/robustness.py",),
        ),
        "environment": environment_metadata(config["_meta"]["project_root"]),
    }
    result["robustness_fingerprint"] = stable_fingerprint(
        {
            "settings": settings,
            "image_ids": image_ids,
            "features_sha256": result["features_sha256"],
            "primary_model_sha256": result["primary_model_sha256"],
            "source_tree_sha256": result["source_tree_sha256"],
        }
    )
    write_json(output_root / "robustness.json", result)
    return result
