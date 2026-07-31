from __future__ import annotations

import contextlib
import io
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
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

from .extraction import read_validated_features
from .features import finite_feature_audit
from .fusion import load_fusions, load_sensitivity_fusions


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


def _holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    total = len(ordered)
    running = 0.0
    adjusted: dict[str, float] = {}
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - index) * float(value)))
        adjusted[name] = running
    return adjusted


def _one_sided_bootstrap_p(values: np.ndarray, observed: float) -> float:
    return centered_bootstrap_p_value(values, observed)


def _coco_detection_metrics(
    annotations_path: Path,
    detections: pd.DataFrame,
    image_ids: list[int],
    score_column: str,
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
            "score": float(np.clip(getattr(row, score_column), 0.0, 1.0)),
        }
        for row in detections.itertuples(index=False)
    ]
    if not records:
        raise ValueError("RQ3 COCO evaluation requires predictions")
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
        "predictions": len(records),
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


def _method_metrics(
    labels: np.ndarray,
    rank: np.ndarray,
    probability: np.ndarray,
    config: dict[str, Any],
) -> dict[str, float]:
    metrics = binary_uncertainty_metrics(
        labels,
        rank,
        coverages=config["evaluation"]["risk_coverages"],
        risk_targets=config["evaluation"]["risk_targets"],
    )
    calibrated = binary_uncertainty_metrics(
        labels,
        probability,
        calibrated_probability=True,
        coverages=config["evaluation"]["risk_coverages"],
        risk_targets=config["evaluation"]["risk_targets"],
    )
    metrics.update({name: calibrated[name] for name in ("brier", "nll", "ece")})
    return metrics


def _bootstrap_method_intervals(
    *,
    labels: np.ndarray,
    rank: np.ndarray,
    probability: np.ndarray,
    groups: np.ndarray,
    method: str,
    config: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    functions: dict[str, tuple[Callable[[np.ndarray, np.ndarray], float], np.ndarray]] = {
        "auroc": (roc_auc_score, rank),
        "auprc": (average_precision_score, rank),
        "aurc": (_aurc, rank),
        "brier": (_brier, probability),
        "nll": (_nll, probability),
        "ece": (_ece, probability),
    }
    repetitions = int(config["evaluation"]["bootstrap_repetitions"])
    seed = int(config["project"]["seed"])
    confidence_level = float(config["evaluation"]["confidence_level"])
    intervals = {}
    for metric, (function, values_input) in functions.items():
        values = clustered_bootstrap(
            labels,
            values_input,
            groups,
            function,
            repetitions,
            seed + sum(ord(character) for character in method + metric),
        )
        lower, upper = percentile_interval(values, confidence_level)
        intervals[metric] = {"lower": lower, "upper": upper}
        records.extend(
            {
                "analysis": "method_interval",
                "method": method,
                "baseline": None,
                "metric": metric,
                "repetition": index,
                "value": float(value),
            }
            for index, value in enumerate(values)
            if np.isfinite(value)
        )
    return intervals


def evaluate_fusions(config: dict[str, Any]) -> dict[str, Any]:
    outputs = config["rq3"]["outputs"]
    test_path = project_path(config, outputs["test_features"])
    extracted, test_metadata = read_validated_features(config, test_path)
    image_summary_path = project_path(config, outputs["test_image_summary"])
    if (
        not image_summary_path.is_file()
        or test_metadata.get("image_summary_sha256")
        != sha256_file(image_summary_path)
    ):
        raise RuntimeError("RQ3 test image-summary integrity failed")
    image_summaries = pd.read_parquet(image_summary_path)
    if extracted.empty:
        raise ValueError("RQ3 test feature artifact is empty")
    for feature in test_metadata["feature_schema"]:
        if feature in {
            "file_name",
            "sequence_id",
            "timeofday",
            "weather",
            "scene",
            "category_name",
            "object_size",
        }:
            continue
        if pd.api.types.is_numeric_dtype(extracted[feature]):
            finite_feature_audit(
                extracted[feature].to_numpy(),
                allow_all_missing=feature.startswith("spatial_")
                or feature.startswith("semantic_")
                or feature.startswith("representation_")
                or feature == "mc_score_mean",
            )

    fusions = load_fusions(config)
    sensitivity_fusions = load_sensitivity_fusions(config)
    prediction_columns = [
        "image_id",
        "file_name",
        "sequence_id",
        "timeofday",
        "weather",
        "scene",
        "detection_index",
        "query_index",
        "category_id",
        "category_name",
        "object_size",
        "score",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "is_error",
        "is_well_localized",
        "is_well_localized_050",
        "is_well_localized_075",
        "localization_iou",
        "localization_class_agreement",
    ]
    all_predictions = extracted[prediction_columns].copy()
    prediction_started = time.perf_counter()
    for method, fusion in {**fusions, **sensitivity_fusions}.items():
        rank = fusion.rank_score(extracted)
        probability = fusion.error_probability(extracted)
        all_predictions[f"rank_{method}"] = rank
        all_predictions[f"prob_error_{method}"] = probability
        all_predictions[f"quality_{method}"] = np.clip(1.0 - rank, 0.0, 1.0)
    prediction_seconds = time.perf_counter() - prediction_started

    threshold = float(config["evaluation"]["primary_score_threshold"])
    operating_mask = extracted["score"].to_numpy(dtype=np.float64) >= threshold
    test = extracted.loc[operating_mask].reset_index(drop=True)
    predictions = all_predictions.loc[operating_mask].reset_index(drop=True)
    if test.empty or test["is_error"].nunique() != 2:
        raise ValueError("RQ3 operational test requires correct and error detections")
    labels = test["is_error"].to_numpy(dtype=np.int64)
    groups = test["sequence_id"].astype(str).to_numpy()
    if not len(np.unique(groups)):
        raise ValueError("RQ3 operational test has no source groups")
    partition = test_metadata["manifest_test_partition"]
    result: dict[str, Any] = {
        "schema_version": 1,
        "research_question": config["rq3"]["question"],
        "protocol_freeze_date": "2026-07-31",
        "test_partition": partition,
        "evidence_status": (
            "diagnostic_not_scientific_evidence"
            if partition == "diagnostic"
            else "confirmatory_requires_human_interpretation"
        ),
        "extracted_test_rows": len(extracted),
        "primary_score_threshold": threshold,
        "test_rows": len(test),
        "test_images": int(image_summaries["image_id"].nunique()),
        "test_sequences": int(image_summaries["sequence_id"].nunique()),
        "methods": {},
        "primary_inference": {},
        "mc_pass_sensitivity": {},
        "score_threshold_sensitivity": {},
        "localization_target_sensitivity": {},
        "ece_bin_sensitivity": {},
        "subgroups": {},
    }
    bootstrap_records: list[dict[str, Any]] = []
    for method, fusion in fusions.items():
        rank = predictions[f"rank_{method}"].to_numpy(dtype=np.float64)
        probability = predictions[f"prob_error_{method}"].to_numpy(dtype=np.float64)
        result["methods"][method] = {
            "family": fusion.family,
            "features": fusion.features,
            "metrics": _method_metrics(labels, rank, probability, config),
            "clustered_bootstrap_interval": _bootstrap_method_intervals(
                labels=labels,
                rank=rank,
                probability=probability,
                groups=groups,
                method=method,
                config=config,
                records=bootstrap_records,
            ),
        }

    primary = str(config["rq3"]["estimators"]["primary_method"])
    baselines = list(config["rq3"]["estimators"]["primary_baselines"])
    metric_functions: dict[
        str, tuple[Callable[[np.ndarray, np.ndarray], float], bool, str]
    ] = {
        "auroc": (roc_auc_score, False, "rank"),
        "aurc": (_aurc, True, "rank"),
        "brier": (_brier, True, "prob_error"),
    }
    raw_p_values: dict[str, float] = {}
    comparisons: dict[str, Any] = {}
    repetitions = int(config["evaluation"]["bootstrap_repetitions"])
    confidence_level = float(config["evaluation"]["confidence_level"])
    seed = int(config["project"]["seed"])
    for baseline in baselines:
        comparisons[baseline] = {}
        for metric in config["rq3"]["inference"]["primary_metrics"]:
            function, lower_is_better, prefix = metric_functions[metric]
            candidate = predictions[f"{prefix}_{primary}"].to_numpy(dtype=np.float64)
            baseline_values = predictions[f"{prefix}_{baseline}"].to_numpy(
                dtype=np.float64
            )
            point_candidate = function(labels, candidate)
            point_baseline = function(labels, baseline_values)
            improvement = (
                point_baseline - point_candidate
                if lower_is_better
                else point_candidate - point_baseline
            )
            name = f"{metric}_vs_{baseline}"
            values = clustered_bootstrap_difference(
                labels,
                candidate,
                baseline_values,
                groups,
                function,
                repetitions,
                seed + sum(ord(character) for character in "rq3_primary" + name),
                lower_is_better=lower_is_better,
            )
            raw_p_values[name] = _one_sided_bootstrap_p(values, improvement)
            lower, upper = percentile_interval(values, confidence_level)
            comparisons[baseline][metric] = {
                "improvement": float(improvement),
                "lower": lower,
                "upper": upper,
                "raw_one_sided_p": raw_p_values[name],
                "positive_is_better": True,
            }
            bootstrap_records.extend(
                {
                    "analysis": "primary_difference",
                    "method": primary,
                    "baseline": baseline,
                    "metric": metric,
                    "repetition": index,
                    "value": float(value),
                }
                for index, value in enumerate(values)
                if np.isfinite(value)
            )
    adjusted = _holm_adjust(raw_p_values)
    alpha = float(config["rq3"]["inference"]["familywise_alpha"])
    ranking_decisions: list[bool] = []
    calibration_decisions: list[bool] = []
    for baseline, metrics in comparisons.items():
        for metric, record in metrics.items():
            name = f"{metric}_vs_{baseline}"
            record["holm_adjusted_p"] = adjusted[name]
            record["reject_no_improvement"] = bool(adjusted[name] < alpha)
            decision = bool(
                record["improvement"] > 0 and record["reject_no_improvement"]
            )
            if metric == "brier":
                calibration_decisions.append(decision)
            else:
                ranking_decisions.append(decision)
    nominal_ranking = bool(all(ranking_decisions))
    nominal_calibration = bool(all(calibration_decisions))
    nominal_overall = nominal_ranking and nominal_calibration
    result["primary_inference"] = {
        "method": primary,
        "test": "null_centered_paired_cluster_bootstrap",
        "baselines": baselines,
        "metrics": list(config["rq3"]["inference"]["primary_metrics"]),
        "familywise_alpha": alpha,
        "correction": config["rq3"]["inference"]["correction"],
        "comparisons": comparisons,
        "nominal_ranking_criterion_met": nominal_ranking,
        "nominal_calibration_criterion_met": nominal_calibration,
        "nominal_overall_criterion_met": nominal_overall,
        "success_criterion_met": bool(
            nominal_overall and partition == "confirmatory"
        ),
        "interpretation_guard": (
            "Diagnostic inference cannot satisfy the scientific success rule."
            if partition == "diagnostic"
            else "Preserve mixed or negative outcomes without test retuning."
        ),
    }

    for method, fusion in sensitivity_fusions.items():
        rank = predictions[f"rank_{method}"].to_numpy(dtype=np.float64)
        probability = predictions[f"prob_error_{method}"].to_numpy(dtype=np.float64)
        result["mc_pass_sensitivity"][method] = {
            "mc_passes": int(method.rsplit("mc", 1)[1]),
            "features": fusion.features,
            "metrics": _method_metrics(labels, rank, probability, config),
        }

    for score_threshold in config["evaluation"]["score_threshold_sensitivity"]:
        value = float(score_threshold)
        mask = extracted["score"].to_numpy(dtype=np.float64) >= value
        threshold_labels = extracted.loc[mask, "is_error"].to_numpy(dtype=np.int64)
        record: dict[str, Any] = {
            "rows": int(mask.sum()),
            "error_prevalence": (
                float(threshold_labels.mean()) if len(threshold_labels) else None
            ),
            "status": "estimable",
            "methods": {},
        }
        if len(np.unique(threshold_labels)) != 2:
            record["status"] = "not_estimable_two_outcomes_required"
        else:
            for method in fusions:
                record["methods"][method] = _method_metrics(
                    threshold_labels,
                    all_predictions.loc[mask, f"rank_{method}"].to_numpy(),
                    all_predictions.loc[mask, f"prob_error_{method}"].to_numpy(),
                    config,
                )
        result["score_threshold_sensitivity"][f"{value:.2f}"] = record

    learned_spatial_rank = predictions[
        "rank_learned_spatial_quality"
    ].to_numpy(dtype=np.float64)
    for target_threshold in config["rq3"]["targets"][
        "localization_iou_sensitivity"
    ]:
        value = float(target_threshold)
        target_name = f"is_well_localized_{int(round(value * 100)):03d}"
        localization_error = 1 - predictions[target_name].to_numpy(dtype=np.int64)
        if len(np.unique(localization_error)) != 2:
            result["localization_target_sensitivity"][f"{value:.2f}"] = {
                "status": "not_estimable_two_outcomes_required",
                "rows": len(localization_error),
            }
        else:
            result["localization_target_sensitivity"][f"{value:.2f}"] = {
                "status": "estimable",
                "rows": len(localization_error),
                "metrics": binary_uncertainty_metrics(
                    localization_error,
                    learned_spatial_rank,
                    coverages=config["evaluation"]["risk_coverages"],
                    risk_targets=config["evaluation"]["risk_targets"],
                ),
            }

    for method in fusions:
        probability = predictions[f"prob_error_{method}"].to_numpy(dtype=np.float64)
        result["ece_bin_sensitivity"][method] = {
            str(int(bins)): expected_calibration_error(labels, probability, int(bins))
            for bins in config["evaluation"]["calibration_bins"]
        }

    minimum_rows = int(config["evaluation"]["minimum_subgroup_rows"])
    subgroup_attributes = list(config["evaluation"]["subgroup_attributes"]) + [
        "category_name",
        "object_size",
    ]
    for attribute in subgroup_attributes:
        result["subgroups"][attribute] = {}
        for value in sorted(predictions[attribute].astype(str).unique()):
            mask = predictions[attribute].astype(str).to_numpy() == value
            subgroup_labels = predictions.loc[mask, "is_error"].to_numpy(
                dtype=np.int64
            )
            record: dict[str, Any] = {
                "rows": int(mask.sum()),
                "error_prevalence": float(subgroup_labels.mean()),
                "status": "estimable",
                "methods": {},
            }
            if mask.sum() < minimum_rows:
                record["status"] = "not_estimable_minimum_rows"
            elif len(np.unique(subgroup_labels)) != 2:
                record["status"] = "not_estimable_two_outcomes_required"
            else:
                for method in fusions:
                    record["methods"][method] = _method_metrics(
                        subgroup_labels,
                        predictions.loc[mask, f"rank_{method}"].to_numpy(),
                        predictions.loc[
                            mask, f"prob_error_{method}"
                        ].to_numpy(),
                        config,
                    )
            result["subgroups"][attribute][value] = record

    result["error_taxonomy"] = {
        "well_localized_and_true_positive": int(
            ((test["is_well_localized"] == 1) & (test["is_error"] == 0)).sum()
        ),
        "well_localized_but_detection_error": int(
            ((test["is_well_localized"] == 1) & (test["is_error"] == 1)).sum()
        ),
        "poorly_localized": int((test["is_well_localized"] == 0).sum()),
        "mean_localization_iou": float(test["localization_iou"].mean()),
    }

    coco_input = all_predictions.copy()
    coco_input["quality_confidence"] = coco_input["score"]
    annotations = project_path(config, config["data"]["evaluation_annotations"])
    image_ids = image_summaries["image_id"].astype(int).tolist()
    result["detector_ranking"] = {}
    for method in fusions:
        score_column = "quality_confidence" if method == "confidence" else f"quality_{method}"
        result["detector_ranking"][method] = _coco_detection_metrics(
            annotations, coco_input, image_ids, score_column
        )

    result["computational_cost"] = {
        "images": len(image_summaries),
        "mc_passes": int(config["rq3"]["extraction"]["mc_passes"]),
        "preprocess_seconds": float(image_summaries["preprocess_seconds"].sum()),
        "deterministic_detector_seconds": float(
            image_summaries["deterministic_seconds"].sum()
        ),
        "stochastic_detector_seconds": float(
            image_summaries["stochastic_seconds"].sum()
        ),
        "rq3_feature_aggregation_seconds": float(
            image_summaries["aggregation_seconds"].sum()
        ),
        "all_method_prediction_seconds": prediction_seconds,
        "peak_gpu_memory_bytes": int(image_summaries["peak_gpu_memory_bytes"].max()),
        "shared_shards_computed_for_test_request": int(
            test_metadata["shared_shards_computed"]
        ),
        "shared_shards_reused_for_test_request": int(
            test_metadata["shared_shards_reused"]
        ),
        "stochastic_modules": test_metadata["stochastic_modules"],
        "environment": test_metadata["environment"],
    }

    predictions_path = project_path(config, outputs["predictions"])
    bootstrap_path = project_path(config, outputs["bootstrap"])
    metrics_path = project_path(config, outputs["metrics"])
    model_index_path = project_path(config, outputs["models"]) / "model_index.json"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_predictions = predictions_path.with_suffix(".parquet.tmp")
    predictions.to_parquet(temporary_predictions, index=False)
    temporary_predictions.replace(predictions_path)
    temporary_bootstrap = bootstrap_path.with_suffix(".parquet.tmp")
    pd.DataFrame(bootstrap_records).to_parquet(temporary_bootstrap, index=False)
    temporary_bootstrap.replace(bootstrap_path)
    required_finite = all(
        np.isfinite(value)
        for method in result["methods"].values()
        for value in method["metrics"].values()
        if isinstance(value, (int, float))
    )
    if not required_finite:
        raise ValueError("RQ3 produced a non-finite required point metric")
    result["artifact_integrity"] = {
        "test_features_sha256": test_metadata["features_sha256"],
        "test_image_summary_sha256": test_metadata["image_summary_sha256"],
        "shared_request_metadata_sha256": test_metadata[
            "shared_request_metadata_sha256"
        ],
        "model_index_sha256": sha256_file(model_index_path),
        "predictions_sha256": sha256_file(predictions_path),
        "bootstrap_sha256": sha256_file(bootstrap_path),
        "all_required_point_metrics_finite": required_finite,
    }
    write_json(metrics_path, result)
    return result
