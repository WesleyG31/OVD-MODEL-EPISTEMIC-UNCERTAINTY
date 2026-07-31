from __future__ import annotations

import contextlib
import io
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from adas_ovd.config import project_path
from adas_ovd.metrics import (
    binary_uncertainty_metrics,
    centered_bootstrap_p_value,
    clustered_bootstrap,
    clustered_bootstrap_difference,
    expected_calibration_error,
    percentile_interval,
    risk_coverage_curve,
)
from adas_ovd.reproducibility import sha256_file, write_json

from .estimators import load_estimators, load_sensitivity_estimators
from .extraction import read_validated_features
from .features import finite_feature_audit


def _aurc(labels: np.ndarray, uncertainty: np.ndarray) -> float:
    return risk_coverage_curve(labels, uncertainty).aurc


def _brier(labels: np.ndarray, probability: np.ndarray) -> float:
    return float(brier_score_loss(labels, np.clip(probability, 1e-7, 1 - 1e-7)))


def _nll(labels: np.ndarray, probability: np.ndarray) -> float:
    return float(
        log_loss(labels, np.clip(probability, 1e-7, 1 - 1e-7), labels=[0, 1])
    )


def _ece(labels: np.ndarray, probability: np.ndarray) -> float:
    return expected_calibration_error(labels, probability)


def _coco_detection_metrics(
    annotations_path: Path, detections: pd.DataFrame, image_ids: list[int]
) -> dict[str, float | int]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    records = [
        {
            "image_id": int(row.image_id),
            "category_id": int(row.category_id),
            "bbox": [
                float(row.bbox_x1),
                float(row.bbox_y1),
                float(row.bbox_x2 - row.bbox_x1),
                float(row.bbox_y2 - row.bbox_y1),
            ],
            "score": float(row.score),
        }
        for row in detections.itertuples(index=False)
    ]
    if not records:
        raise ValueError("COCO evaluation requires at least one prediction")
    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth = COCO(str(annotations_path))
        predictions = ground_truth.loadRes(records)
        evaluator = COCOeval(ground_truth, predictions, "bbox")
        evaluator.params.imgIds = sorted(int(value) for value in image_ids)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    stats = evaluator.stats
    return {
        "images": len(evaluator.params.imgIds),
        "predictions_at_extraction_threshold": len(records),
        "max_detections_per_image": int(evaluator.params.maxDets[-1]),
        "map_50_95": float(stats[0]),
        "ap_50": float(stats[1]),
        "ap_75": float(stats[2]),
        "ap_small": float(stats[3]),
        "ap_medium": float(stats[4]),
        "ap_large": float(stats[5]),
        "ar_1": float(stats[6]),
        "ar_10": float(stats[7]),
        "ar_100": float(stats[8]),
        "ar_small": float(stats[9]),
        "ar_medium": float(stats[10]),
        "ar_large": float(stats[11]),
    }


def _holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - index) * float(value)))
        adjusted[name] = running
    return adjusted


def _one_sided_bootstrap_p(values: np.ndarray, observed: float) -> float:
    return centered_bootstrap_p_value(values, observed)


def _method_metrics(
    labels: np.ndarray,
    rank: np.ndarray,
    probability: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    rank_metrics = binary_uncertainty_metrics(
        labels,
        rank,
        coverages=config["evaluation"]["risk_coverages"],
        risk_targets=config["evaluation"]["risk_targets"],
    )
    calibration = binary_uncertainty_metrics(
        labels,
        probability,
        calibrated_probability=True,
        coverages=config["evaluation"]["risk_coverages"],
        risk_targets=config["evaluation"]["risk_targets"],
    )
    rank_metrics.update(
        {name: calibration[name] for name in ("brier", "nll", "ece")}
    )
    return rank_metrics


def _validation_complementarity(
    config: dict[str, Any], model_index: dict[str, Any]
) -> dict[str, float | int]:
    model_root = project_path(config, config["rq2"]["outputs"]["models"])
    path = model_root / "validation_predictions.parquet"
    if (
        not path.is_file()
        or model_index.get("validation_predictions_sha256") != sha256_file(path)
    ):
        raise RuntimeError("RQ2 validation-prediction integrity check failed")
    frame = pd.read_parquet(path)
    deterministic = frame["rank_deterministic_fixed"].to_numpy(dtype=np.float64)
    stochastic = frame["rank_stochastic_fixed"].to_numpy(dtype=np.float64)
    correlation = spearmanr(deterministic, stochastic).statistic
    det_high = deterministic > np.median(deterministic)
    stochastic_high = stochastic > np.median(stochastic)
    return {
        "validation_rows": len(frame),
        "fixed_score_spearman": float(correlation),
        "opposite_median_quadrant_fraction": float(
            np.logical_xor(det_high, stochastic_high).mean()
        ),
    }


def evaluate_estimators(config: dict[str, Any]) -> dict[str, Any]:
    outputs = config["rq2"]["outputs"]
    test_path = project_path(config, outputs["test_features"])
    extracted, test_metadata = read_validated_features(config, test_path)
    image_summary_path = project_path(config, outputs["test_image_summary"])
    if (
        not image_summary_path.is_file()
        or test_metadata.get("image_summary_sha256") != sha256_file(image_summary_path)
    ):
        raise RuntimeError("RQ2 test image-summary integrity check failed")
    image_summaries = pd.read_parquet(image_summary_path)
    if extracted.empty:
        raise ValueError("RQ2 test feature artifact is empty")
    for feature in test_metadata["uncertainty_feature_columns"]:
        finite_feature_audit(
            extracted[feature].to_numpy(dtype=np.float64),
            allow_all_missing=False,
        )

    estimators = load_estimators(config)
    sensitivity_estimators = load_sensitivity_estimators(config)
    model_root = project_path(config, outputs["models"])
    model_index_path = model_root / "model_index.json"
    with model_index_path.open("r", encoding="utf-8") as handle:
        model_index = json.load(handle)

    prediction_columns = [
        "image_id",
        "file_name",
        "sequence_id",
        "timeofday",
        "weather",
        "scene",
        "detection_index",
        "query_index",
        "category_name",
        "object_size",
        "score",
        "is_error",
        "matched_iou",
    ]
    all_predictions = extracted[prediction_columns].copy()
    prediction_started = time.perf_counter()
    for method, estimator in {**estimators, **sensitivity_estimators}.items():
        rank = estimator.rank_score(extracted)
        probability = np.clip(estimator.error_probability(extracted), 1e-7, 1 - 1e-7)
        if not np.isfinite(rank).all() or not np.isfinite(probability).all():
            raise ValueError(f"RQ2 estimator {method} produced a non-finite output")
        all_predictions[f"rank_{method}"] = rank
        all_predictions[f"prob_error_{method}"] = probability
    prediction_seconds = time.perf_counter() - prediction_started

    threshold = float(config["evaluation"]["primary_score_threshold"])
    operating_mask = extracted["score"].to_numpy(dtype=np.float64) >= threshold
    test = extracted.loc[operating_mask].reset_index(drop=True)
    predictions = all_predictions.loc[operating_mask].reset_index(drop=True)
    if test.empty or test["is_error"].nunique() != 2:
        raise ValueError("RQ2 operating test set requires correct and error detections")
    labels = test["is_error"].to_numpy(dtype=np.int64)
    groups = test["sequence_id"].astype(str).to_numpy()
    repetitions = int(config["evaluation"]["bootstrap_repetitions"])
    confidence_level = float(config["evaluation"]["confidence_level"])
    seed = int(config["project"]["seed"])
    manifest_partition = test_metadata["manifest_test_partition"]
    result: dict[str, Any] = {
        "schema_version": 1,
        "research_question": config["rq2"]["question"],
        "test_partition": manifest_partition,
        "evidence_status": (
            "diagnostic_not_scientific_evidence"
            if manifest_partition == "diagnostic"
            else "confirmatory_requires_human_interpretation"
        ),
        "extracted_test_rows": len(extracted),
        "primary_score_threshold": threshold,
        "test_rows": len(test),
        "test_images": int(image_summaries["image_id"].nunique()),
        "test_sequences": int(image_summaries["sequence_id"].nunique()),
        "methods": {},
        "primary_inference": {},
        "comparisons_vs_confidence": {},
        "mc_pass_sensitivity": {},
        "score_threshold_sensitivity": {},
        "subgroups": {},
        "validation_complementarity": _validation_complementarity(
            config, model_index
        ),
        "detector_performance": _coco_detection_metrics(
            project_path(config, config["data"]["evaluation_annotations"]),
            extracted,
            image_summaries["image_id"].astype(int).tolist(),
        ),
    }
    bootstrap_records: list[dict[str, Any]] = []
    rank_bootstrap: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
        "auroc": roc_auc_score,
        "auprc": average_precision_score,
        "aurc": _aurc,
    }
    probability_bootstrap: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
        "brier": _brier,
        "nll": _nll,
        "ece": _ece,
    }
    for method in estimators:
        rank = predictions[f"rank_{method}"].to_numpy(dtype=np.float64)
        probability = predictions[f"prob_error_{method}"].to_numpy(dtype=np.float64)
        intervals = {}
        for metric_name, metric_function in rank_bootstrap.items():
            values = clustered_bootstrap(
                labels,
                rank,
                groups,
                metric_function,
                repetitions,
                seed + sum(ord(character) for character in method + metric_name),
            )
            lower, upper = percentile_interval(values, confidence_level)
            intervals[metric_name] = {"lower": lower, "upper": upper}
            bootstrap_records.extend(
                {
                    "analysis": "method_interval",
                    "method": method,
                    "baseline": None,
                    "metric": metric_name,
                    "repetition": index,
                    "value": float(value),
                }
                for index, value in enumerate(values)
                if np.isfinite(value)
            )
        for metric_name, metric_function in probability_bootstrap.items():
            values = clustered_bootstrap(
                labels,
                probability,
                groups,
                metric_function,
                repetitions,
                seed + sum(ord(character) for character in "prob" + method + metric_name),
            )
            lower, upper = percentile_interval(values, confidence_level)
            intervals[metric_name] = {"lower": lower, "upper": upper}
            bootstrap_records.extend(
                {
                    "analysis": "method_interval",
                    "method": method,
                    "baseline": None,
                    "metric": metric_name,
                    "repetition": index,
                    "value": float(value),
                }
                for index, value in enumerate(values)
                if np.isfinite(value)
            )
        result["methods"][method] = {
            "features": estimators[method].features,
            "family": estimators[method].family,
            "metrics": _method_metrics(labels, rank, probability, config),
            "clustered_bootstrap_interval": intervals,
        }

    primary_method = config["rq2"]["estimators"]["primary_fusion"]
    baselines = list(config["rq2"]["inference"]["comparison_baselines"])
    primary_metrics = list(config["rq2"]["inference"]["primary_metrics"])
    primary_bootstraps: dict[str, np.ndarray] = {}
    raw_p_values: dict[str, float] = {}
    comparisons: dict[str, Any] = {}
    for baseline in baselines:
        comparisons[baseline] = {}
        for metric_name in primary_metrics:
            metric_function = rank_bootstrap[metric_name]
            lower_is_better = metric_name == "aurc"
            candidate = predictions[f"rank_{primary_method}"].to_numpy()
            baseline_values = predictions[f"rank_{baseline}"].to_numpy()
            point_candidate = metric_function(labels, candidate)
            point_baseline = metric_function(labels, baseline_values)
            point = (
                point_baseline - point_candidate
                if lower_is_better
                else point_candidate - point_baseline
            )
            name = f"{metric_name}_vs_{baseline}"
            values = clustered_bootstrap_difference(
                labels,
                candidate,
                baseline_values,
                groups,
                metric_function,
                repetitions,
                seed + sum(ord(character) for character in "primary" + name),
                lower_is_better=lower_is_better,
            )
            primary_bootstraps[name] = values
            raw_p_values[name] = _one_sided_bootstrap_p(values, point)
            lower, upper = percentile_interval(values, confidence_level)
            comparisons[baseline][metric_name] = {
                "improvement": float(point),
                "lower": lower,
                "upper": upper,
                "raw_one_sided_p": raw_p_values[name],
                "positive_is_better": True,
            }
            bootstrap_records.extend(
                {
                    "analysis": "primary_difference",
                    "method": primary_method,
                    "baseline": baseline,
                    "metric": metric_name,
                    "repetition": index,
                    "value": float(value),
                }
                for index, value in enumerate(values)
                if np.isfinite(value)
            )
    adjusted = _holm_adjust(raw_p_values)
    alpha = float(config["rq2"]["inference"]["familywise_alpha"])
    successes = []
    for baseline in baselines:
        for metric_name in primary_metrics:
            name = f"{metric_name}_vs_{baseline}"
            record = comparisons[baseline][metric_name]
            record["holm_adjusted_p"] = adjusted[name]
            record["reject_no_improvement"] = bool(adjusted[name] < alpha)
            successes.append(
                record["improvement"] > 0 and record["reject_no_improvement"]
            )
    result["primary_inference"] = {
        "method": primary_method,
        "test": "null_centered_paired_cluster_bootstrap",
        "familywise_alpha": alpha,
        "correction": config["rq2"]["inference"]["correction"],
        "comparisons": comparisons,
        "success_criterion_met": bool(all(successes)),
        "interpretation_guard": (
            "Diagnostic inference is technical only."
            if manifest_partition == "diagnostic"
            else "Report negative or mixed outcomes without retuning."
        ),
    }

    confidence = predictions["rank_confidence"].to_numpy(dtype=np.float64)
    for method in estimators:
        if method == "confidence":
            continue
        result["comparisons_vs_confidence"][method] = {}
        candidate = predictions[f"rank_{method}"].to_numpy(dtype=np.float64)
        for metric_name in ("auroc", "aurc"):
            metric_function = rank_bootstrap[metric_name]
            lower_is_better = metric_name == "aurc"
            values = clustered_bootstrap_difference(
                labels,
                candidate,
                confidence,
                groups,
                metric_function,
                repetitions,
                seed + sum(ord(character) for character in "confidence" + method + metric_name),
                lower_is_better=lower_is_better,
            )
            lower, upper = percentile_interval(values, confidence_level)
            candidate_point = metric_function(labels, candidate)
            confidence_point = metric_function(labels, confidence)
            improvement = (
                confidence_point - candidate_point
                if lower_is_better
                else candidate_point - confidence_point
            )
            result["comparisons_vs_confidence"][method][metric_name] = {
                "improvement": float(improvement),
                "lower": lower,
                "upper": upper,
                "multiplicity_adjusted": False,
                "positive_is_better": True,
            }

    for method, estimator in sensitivity_estimators.items():
        rank = predictions[f"rank_{method}"].to_numpy(dtype=np.float64)
        probability = predictions[f"prob_error_{method}"].to_numpy(dtype=np.float64)
        result["mc_pass_sensitivity"][method] = {
            "mc_passes": int(method.rsplit("mc", 1)[1]),
            "features": estimator.features,
            "metrics": _method_metrics(labels, rank, probability, config),
        }

    for sensitivity_threshold in config["evaluation"]["score_threshold_sensitivity"]:
        sensitivity_threshold = float(sensitivity_threshold)
        mask = extracted["score"].to_numpy(dtype=np.float64) >= sensitivity_threshold
        threshold_labels = extracted.loc[mask, "is_error"].to_numpy(dtype=np.int64)
        record: dict[str, Any] = {
            "rows": int(mask.sum()),
            "error_prevalence": (
                float(threshold_labels.mean()) if len(threshold_labels) else None
            ),
            "methods": {},
        }
        if len(np.unique(threshold_labels)) == 2:
            for method in estimators:
                record["methods"][method] = binary_uncertainty_metrics(
                    threshold_labels,
                    all_predictions.loc[mask, f"rank_{method}"].to_numpy(),
                )
        result["score_threshold_sensitivity"][f"{sensitivity_threshold:.2f}"] = record

    minimum_rows = int(config["evaluation"]["minimum_subgroup_rows"])
    subgroup_attributes = list(config["evaluation"]["subgroup_attributes"]) + [
        "category_name",
        "object_size",
    ]
    for attribute in subgroup_attributes:
        result["subgroups"][attribute] = {}
        for value in sorted(predictions[attribute].astype(str).unique()):
            mask = predictions[attribute].astype(str).to_numpy() == value
            subgroup_labels = predictions.loc[mask, "is_error"].to_numpy(dtype=np.int64)
            record = {
                "rows": int(mask.sum()),
                "error_prevalence": float(subgroup_labels.mean()),
                "methods": {},
            }
            if mask.sum() >= minimum_rows and len(np.unique(subgroup_labels)) == 2:
                for method in estimators:
                    record["methods"][method] = binary_uncertainty_metrics(
                        subgroup_labels,
                        predictions.loc[mask, f"rank_{method}"].to_numpy(),
                    )
            result["subgroups"][attribute][value] = record

    deterministic_seconds = float(image_summaries["deterministic_seconds"].sum())
    stochastic_seconds = float(image_summaries["stochastic_seconds"].sum())
    aggregation_seconds = float(image_summaries["aggregation_seconds"].sum())
    image_count = len(image_summaries)
    full_seconds = deterministic_seconds + stochastic_seconds + aggregation_seconds
    result["computational_cost"] = {
        "images": image_count,
        "mc_passes": int(config["rq2"]["extraction"]["mc_passes"]),
        "deterministic_detector_seconds": deterministic_seconds,
        "stochastic_detector_seconds": stochastic_seconds,
        "feature_aggregation_seconds": aggregation_seconds,
        "all_estimator_prediction_seconds": prediction_seconds,
        "deterministic_proxy_images_per_second": (
            image_count / deterministic_seconds if deterministic_seconds > 0 else None
        ),
        "full_fused_path_images_per_second": (
            image_count / full_seconds if full_seconds > 0 else None
        ),
        "mc_incremental_overhead_ratio": (
            stochastic_seconds / deterministic_seconds
            if deterministic_seconds > 0
            else None
        ),
        "stochastic_modules": test_metadata["stochastic_modules"],
        "environment": test_metadata["environment"],
    }

    predictions_path = project_path(config, outputs["predictions"])
    bootstrap_path = project_path(config, outputs["bootstrap"])
    metrics_path = project_path(config, outputs["metrics"])
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(predictions_path, index=False)
    pd.DataFrame(bootstrap_records).to_parquet(bootstrap_path, index=False)
    result["artifact_integrity"] = {
        "test_features_sha256": test_metadata["features_sha256"],
        "test_image_summary_sha256": test_metadata["image_summary_sha256"],
        "model_index_sha256": sha256_file(model_index_path),
        "predictions_sha256": sha256_file(predictions_path),
        "bootstrap_sha256": sha256_file(bootstrap_path),
        "all_required_metrics_finite": bool(
            all(
                np.isfinite(value)
                for method in result["methods"].values()
                for value in method["metrics"].values()
                if isinstance(value, (int, float))
            )
        ),
    }
    if not result["artifact_integrity"]["all_required_metrics_finite"]:
        raise ValueError("RQ2 produced a non-finite required metric")
    write_json(metrics_path, result)
    return result
