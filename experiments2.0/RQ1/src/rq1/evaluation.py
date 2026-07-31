from __future__ import annotations

import contextlib
import io
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

from .fusion import load_fusions, load_mc_sensitivity_fusions
from .extraction import read_validated_features


def _aurc(labels: np.ndarray, uncertainty: np.ndarray) -> float:
    return risk_coverage_curve(labels, uncertainty).aurc


def _holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, name in enumerate(ordered):
        running = max(running, (total - index) * p_values[name])
        adjusted[name] = min(running, 1.0)
    return adjusted


def _one_sided_bootstrap_p(values: np.ndarray, observed: float) -> float:
    return centered_bootstrap_p_value(values, observed)


def _reliability_bins(
    labels: np.ndarray, probabilities: np.ndarray, bins: int
) -> list[dict[str, float | int]]:
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    bin_ids = np.clip(
        np.digitize(probabilities, edges[1:-1]), 0, int(bins) - 1
    )
    records: list[dict[str, float | int]] = []
    for bin_id in range(int(bins)):
        mask = bin_ids == bin_id
        if not mask.any():
            continue
        records.append(
            {
                "bin": bin_id,
                "count": int(mask.sum()),
                "mean_probability": float(probabilities[mask].mean()),
                "observed_error_rate": float(labels[mask].mean()),
            }
        )
    return records


def _coco_detection_metrics(
    annotations_path: Path, detections: pd.DataFrame
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
        raise ValueError("COCO detector evaluation requires predictions")
    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth = COCO(str(annotations_path))
        predictions = ground_truth.loadRes(records)
        evaluator = COCOeval(ground_truth, predictions, "bbox")
        evaluator.params.imgIds = sorted(
            {int(image_id) for image_id in detections["image_id"]}
        )
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


def evaluate_fusions(config: dict[str, Any]) -> dict[str, Any]:
    outputs = config["rq1"]["outputs"]
    test_features_path = project_path(config, outputs["test_features"])
    extracted, test_metadata = read_validated_features(
        config, test_features_path
    )
    image_summary_path = project_path(config, outputs["test_image_summary"])
    if (
        not image_summary_path.is_file()
        or test_metadata.get("image_summary_sha256")
        != sha256_file(image_summary_path)
    ):
        raise RuntimeError("Test image-summary integrity check failed")
    image_summaries = pd.read_parquet(image_summary_path)
    if extracted.empty or extracted["is_error"].nunique() < 2:
        raise ValueError("Confirmatory test features require both outcome classes")
    models = load_fusions(config)
    sensitivity_models = load_mc_sensitivity_fusions(config)
    required_attributes = list(
        dict.fromkeys(
            list(config["evaluation"]["subgroup_attributes"])
            + ["category_name", "object_size"]
        )
    )
    missing_attributes = sorted(set(required_attributes) - set(extracted.columns))
    if missing_attributes:
        raise KeyError(
            "Test features are missing frozen subgroup attributes: "
            f"{missing_attributes}"
        )
    all_predictions = extracted[
        [
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
    ].copy()
    prediction_started = time.perf_counter()
    for method, model in models.items():
        rank = model.rank_score(extracted)
        probability = np.clip(
            model.error_probability(extracted), 1e-7, 1 - 1e-7
        )
        all_predictions[f"rank_{method}"] = rank
        all_predictions[f"prob_error_{method}"] = probability
    for method, model in sensitivity_models.items():
        all_predictions[f"rank_{method}"] = model.rank_score(extracted)
        all_predictions[f"prob_error_{method}"] = np.clip(
            model.error_probability(extracted), 1e-7, 1 - 1e-7
        )
    prediction_seconds = time.perf_counter() - prediction_started

    operating_threshold = float(
        config["evaluation"]["primary_score_threshold"]
    )
    operating_mask = extracted["score"].to_numpy() >= operating_threshold
    test = extracted.loc[operating_mask].reset_index(drop=True)
    predictions = all_predictions.loc[operating_mask].reset_index(drop=True)
    if test.empty or test["is_error"].nunique() < 2:
        raise ValueError(
            "The primary operating threshold leaves fewer than two outcome "
            "classes on the confirmatory test set"
        )
    labels = test["is_error"].to_numpy(dtype=np.int64)
    groups = test["sequence_id"].astype(str).to_numpy()
    result: dict[str, Any] = {
        "schema_version": 3,
        "evidence_tier": (
            "confirmatory"
            if config["rq1"]["manifest"]["test_partition"]
            == "confirmatory"
            else "diagnostic"
        ),
        "extracted_test_rows": len(extracted),
        "primary_score_threshold": operating_threshold,
        "test_rows": len(test),
        "test_images": int(test["image_id"].nunique()),
        "test_sequences": int(test["sequence_id"].nunique()),
        "methods": {},
        "primary_inference": {},
        "score_threshold_sensitivity": {},
        "mc_pass_sensitivity": {},
        "subgroups": {},
        "image_level_safety": {},
        "detector_performance": _coco_detection_metrics(
            project_path(
                config, config["data"]["evaluation_annotations"]
            ),
            extracted,
        ),
        "computational_cost": {},
    }
    bootstrap_records: list[dict[str, Any]] = []
    repetitions = int(config["evaluation"]["bootstrap_repetitions"])
    confidence_level = float(config["evaluation"]["confidence_level"])
    seed = int(config["project"]["seed"])

    bootstrap_metrics: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
        "auroc": lambda y, u: float(roc_auc_score(y, u)),
        "auprc": lambda y, u: float(average_precision_score(y, u)),
        "aurc": _aurc,
    }
    probability_bootstrap_metrics: dict[
        str, Callable[[np.ndarray, np.ndarray], float]
    ] = {
        "brier": lambda y, p: float(brier_score_loss(y, p)),
        "nll": lambda y, p: float(log_loss(y, p, labels=[0, 1])),
        "ece": lambda y, p: expected_calibration_error(y, p, bins=15),
    }
    paired_bootstrap: dict[tuple[str, str], np.ndarray] = {}
    for method, model in models.items():
        rank = predictions[f"rank_{method}"].to_numpy()
        probability = predictions[f"prob_error_{method}"].to_numpy()
        rank_metrics = binary_uncertainty_metrics(
            labels,
            rank,
            calibrated_probability=False,
            coverages=config["evaluation"]["risk_coverages"],
            risk_targets=config["evaluation"]["risk_targets"],
        )
        probability_metrics = binary_uncertainty_metrics(
            labels,
            probability,
            calibrated_probability=True,
            coverages=config["evaluation"]["risk_coverages"],
            risk_targets=config["evaluation"]["risk_targets"],
        )
        metrics = dict(rank_metrics)
        metrics.update(
            {
                "brier": probability_metrics["brier"],
                "nll": probability_metrics["nll"],
                "ece": probability_metrics["ece"],
            }
        )
        intervals = {}
        for metric_name, metric_function in bootstrap_metrics.items():
            values = clustered_bootstrap(
                labels_error=labels,
                uncertainty=rank,
                groups=groups,
                metric=metric_function,
                repetitions=repetitions,
                seed=seed + sum(ord(character) for character in method + metric_name),
            )
            lower, upper = percentile_interval(values, confidence_level)
            intervals[metric_name] = {"lower": lower, "upper": upper}
            bootstrap_records.extend(
                {
                    "method": method,
                    "metric": metric_name,
                    "repetition": repetition,
                    "value": float(value),
                }
                for repetition, value in enumerate(values)
                if np.isfinite(value)
            )
        for metric_name, metric_function in probability_bootstrap_metrics.items():
            values = clustered_bootstrap(
                labels_error=labels,
                uncertainty=probability,
                groups=groups,
                metric=metric_function,
                repetitions=repetitions,
                seed=seed
                + sum(ord(character) for character in method + metric_name),
            )
            lower, upper = percentile_interval(values, confidence_level)
            intervals[metric_name] = {"lower": lower, "upper": upper}
            bootstrap_records.extend(
                {
                    "method": method,
                    "metric": metric_name,
                    "repetition": repetition,
                    "value": float(value),
                }
                for repetition, value in enumerate(values)
                if np.isfinite(value)
            )
        calibration_bins = [
            int(value) for value in config["evaluation"]["calibration_bins"]
        ]
        result["methods"][method] = {
            "features": model.features,
            "metrics": metrics,
            "clustered_bootstrap_interval": intervals,
            "ece_bin_sensitivity": {
                str(bins): expected_calibration_error(
                    labels, probability, bins=bins
                )
                for bins in calibration_bins
            },
            "reliability_bins": _reliability_bins(
                labels, probability, bins=15
            ),
        }

    baseline = predictions["rank_confidence"].to_numpy()
    for method in models:
        comparisons: dict[str, Any] = {}
        candidate = predictions[f"rank_{method}"].to_numpy()
        for metric_name, metric_function in bootstrap_metrics.items():
            if method == "confidence":
                point = 0.0
                lower = 0.0
                upper = 0.0
                values = np.zeros(repetitions, dtype=np.float64)
            else:
                lower_is_better = metric_name == "aurc"
                candidate_point = metric_function(labels, candidate)
                baseline_point = metric_function(labels, baseline)
                point = (
                    baseline_point - candidate_point
                    if lower_is_better
                    else candidate_point - baseline_point
                )
                values = clustered_bootstrap_difference(
                    labels_error=labels,
                    candidate_uncertainty=candidate,
                    baseline_uncertainty=baseline,
                    groups=groups,
                    metric=metric_function,
                    repetitions=repetitions,
                    seed=seed
                    + sum(
                        ord(character)
                        for character in "paired" + method + metric_name
                    ),
                    lower_is_better=lower_is_better,
                )
                lower, upper = percentile_interval(values, confidence_level)
                paired_bootstrap[(method, metric_name)] = values
            comparisons[metric_name] = {
                "improvement": float(point),
                "lower": float(lower),
                "upper": float(upper),
                "positive_is_better": True,
            }
            bootstrap_records.extend(
                {
                    "method": method,
                    "metric": f"improvement_{metric_name}_vs_confidence",
                    "repetition": repetition,
                    "value": float(value),
                }
                for repetition, value in enumerate(values)
                if np.isfinite(value)
            )
        result["methods"][method][
            "paired_improvement_vs_confidence"
        ] = comparisons

    inference_config = config["rq1"]["inference"]
    primary_method = str(config["rq1"]["fusion"]["primary_method"])
    primary_baseline = str(config["rq1"]["fusion"]["primary_baseline"])
    raw_p_values: dict[str, float] = {}
    inference_metrics: dict[str, Any] = {}
    for metric_name in inference_config["primary_metrics"]:
        values = paired_bootstrap[(primary_method, str(metric_name))]
        comparison = result["methods"][primary_method][
            "paired_improvement_vs_confidence"
        ][str(metric_name)]
        raw_p = _one_sided_bootstrap_p(values, comparison["improvement"])
        raw_p_values[str(metric_name)] = raw_p
        inference_metrics[str(metric_name)] = {
            **comparison,
            "one_sided_p": raw_p,
        }
    adjusted = _holm_adjust(raw_p_values)
    alpha = float(inference_config["familywise_alpha"])
    for metric_name, adjusted_p in adjusted.items():
        inference_metrics[metric_name]["holm_adjusted_p"] = adjusted_p
        inference_metrics[metric_name]["reject_no_improvement"] = bool(
            inference_metrics[metric_name]["improvement"] > 0.0
            and adjusted_p < alpha
        )

    candidate_probability = predictions[
        f"prob_error_{primary_method}"
    ].to_numpy()
    baseline_probability = predictions[
        f"prob_error_{primary_baseline}"
    ].to_numpy()
    brier_values = clustered_bootstrap_difference(
        labels_error=labels,
        candidate_uncertainty=candidate_probability,
        baseline_uncertainty=baseline_probability,
        groups=groups,
        metric=probability_bootstrap_metrics["brier"],
        repetitions=repetitions,
        seed=seed + 90731,
        lower_is_better=True,
    )
    brier_lower, brier_upper = percentile_interval(
        brier_values, confidence_level
    )
    brier_improvement = float(
        brier_score_loss(labels, baseline_probability)
        - brier_score_loss(labels, candidate_probability)
    )
    margin = float(inference_config["calibration_noninferiority_margin"])
    calibration_noninferior = bool(brier_lower >= -margin)
    confirmatory = result["evidence_tier"] == "confirmatory"
    statistical_conditions = all(
        record["reject_no_improvement"]
        for record in inference_metrics.values()
    ) and calibration_noninferior
    result["primary_inference"] = {
        "method": primary_method,
        "baseline": primary_baseline,
        "familywise_alpha": alpha,
        "correction": inference_config["correction"],
        "test": "null_centered_paired_cluster_bootstrap",
        "metrics": inference_metrics,
        "calibration_noninferiority": {
            "metric": "brier",
            "margin": margin,
            "improvement": brier_improvement,
            "lower": brier_lower,
            "upper": brier_upper,
            "passed": calibration_noninferior,
        },
        "success_rule": inference_config["success_rule"],
        "statistical_conditions_met": statistical_conditions,
        "rq1_answer_supported": bool(confirmatory and statistical_conditions),
        "status": (
            "supported"
            if confirmatory and statistical_conditions
            else "not_supported"
            if confirmatory
            else "diagnostic_only"
        ),
    }

    for method, model in sensitivity_models.items():
        rank = predictions[f"rank_{method}"].to_numpy()
        probability = predictions[f"prob_error_{method}"].to_numpy()
        rank_metrics = binary_uncertainty_metrics(
            labels,
            rank,
            coverages=config["evaluation"]["risk_coverages"],
            risk_targets=config["evaluation"]["risk_targets"],
        )
        probability_metrics = binary_uncertainty_metrics(
            labels,
            probability,
            calibrated_probability=True,
        )
        intervals = {}
        for metric_name, metric_function in bootstrap_metrics.items():
            values = clustered_bootstrap(
                labels_error=labels,
                uncertainty=rank,
                groups=groups,
                metric=metric_function,
                repetitions=repetitions,
                seed=seed
                + sum(
                    ord(character)
                    for character in "sensitivity" + method + metric_name
                ),
            )
            lower, upper = percentile_interval(values, confidence_level)
            intervals[metric_name] = {"lower": lower, "upper": upper}
            bootstrap_records.extend(
                {
                    "method": method,
                    "metric": metric_name,
                    "repetition": repetition,
                    "value": float(value),
                }
                for repetition, value in enumerate(values)
                if np.isfinite(value)
            )
        result["mc_pass_sensitivity"][method] = {
            "mc_passes": int(method.rsplit("_", 1)[1]),
            "features": model.features,
            "metrics": {
                **rank_metrics,
                "brier": probability_metrics["brier"],
                "nll": probability_metrics["nll"],
                "ece": probability_metrics["ece"],
            },
            "clustered_bootstrap_interval": intervals,
        }

    for threshold in config["evaluation"]["score_threshold_sensitivity"]:
        threshold = float(threshold)
        mask = extracted["score"].to_numpy() >= threshold
        labels_threshold = extracted.loc[mask, "is_error"].to_numpy(
            dtype=np.int64
        )
        key = f"{threshold:.2f}"
        result["score_threshold_sensitivity"][key] = {
            "rows": int(mask.sum()),
            "error_prevalence": (
                float(labels_threshold.mean()) if len(labels_threshold) else None
            ),
            "methods": {},
        }
        if len(np.unique(labels_threshold)) < 2:
            continue
        for method in models:
            uncertainty = all_predictions.loc[
                mask, f"rank_{method}"
            ].to_numpy()
            result["score_threshold_sensitivity"][key]["methods"][method] = (
                binary_uncertainty_metrics(labels_threshold, uncertainty)
            )

    minimum_subgroup_rows = int(
        config["evaluation"]["minimum_subgroup_rows"]
    )
    for attribute in required_attributes:
        result["subgroups"][attribute] = {}
        for value in sorted(predictions[attribute].astype(str).unique()):
            mask = predictions[attribute].astype(str).to_numpy() == value
            subgroup_labels = predictions.loc[mask, "is_error"].to_numpy(
                dtype=np.int64
            )
            record: dict[str, Any] = {
                "rows": int(mask.sum()),
                "error_prevalence": float(subgroup_labels.mean()),
                "methods": {},
            }
            if (
                int(mask.sum()) >= minimum_subgroup_rows
                and len(np.unique(subgroup_labels)) == 2
            ):
                for method in models:
                    uncertainty = predictions.loc[
                        mask, f"rank_{method}"
                    ].to_numpy()
                    record["methods"][method] = binary_uncertainty_metrics(
                        subgroup_labels, uncertainty
                    )
            result["subgroups"][attribute][value] = record

    operational_counts = (
        predictions.groupby("image_id", sort=False)
        .agg(
            operational_detections=("is_error", "size"),
            operational_true_positives=(
                "is_error",
                lambda values: int((values == 0).sum()),
            ),
            operational_false_positives=("is_error", "sum"),
        )
        .reset_index()
    )
    image_evaluation = image_summaries.merge(
        operational_counts, on="image_id", how="left", validate="one_to_one"
    )
    count_columns = [
        "operational_detections",
        "operational_true_positives",
        "operational_false_positives",
    ]
    image_evaluation[count_columns] = image_evaluation[count_columns].fillna(
        0
    )
    image_evaluation["operational_false_negatives"] = (
        image_evaluation["ground_truth_objects"]
        - image_evaluation["operational_true_positives"]
    ).clip(lower=0)
    image_evaluation["has_false_negative"] = (
        image_evaluation["operational_false_negatives"] > 0
    ).astype(int)
    image_evaluation["has_any_error"] = (
        (image_evaluation["operational_false_negatives"] > 0)
        | (image_evaluation["operational_false_positives"] > 0)
    ).astype(int)
    result["image_level_safety"]["aggregation"] = (
        "maximum detection uncertainty; images without operational "
        "detections receive uncertainty 1.0"
    )
    for method in models:
        aggregate = predictions.groupby("image_id")[
            f"rank_{method}"
        ].max()
        uncertainty = (
            image_evaluation["image_id"].map(aggregate).fillna(1.0).to_numpy()
        )
        image_evaluation[f"rank_{method}"] = uncertainty
        method_results: dict[str, Any] = {}
        for outcome in ("has_false_negative", "has_any_error"):
            outcome_labels = image_evaluation[outcome].to_numpy(
                dtype=np.int64
            )
            if len(np.unique(outcome_labels)) < 2:
                method_results[outcome] = {
                    "status": "not_estimable",
                    "n": len(outcome_labels),
                    "prevalence": float(outcome_labels.mean()),
                }
            else:
                method_results[outcome] = {
                    "status": "ok",
                    **binary_uncertainty_metrics(
                        outcome_labels, uncertainty
                    ),
                }
        result["image_level_safety"][method] = method_results

    image_count = len(image_summaries)
    timing = {
        name: float(image_summaries[name].sum())
        for name in (
            "preprocess_seconds",
            "deterministic_seconds",
            "stochastic_seconds",
            "aggregation_seconds",
            "total_seconds",
        )
    }
    result["computational_cost"] = {
        "images": image_count,
        "mc_passes": int(config["rq1"]["extraction"]["mc_passes"]),
        **timing,
        "all_fusion_prediction_seconds": prediction_seconds,
        "full_path_images_per_second": (
            image_count / timing["total_seconds"]
            if timing["total_seconds"] > 0
            else None
        ),
        "stochastic_over_deterministic_ratio": (
            timing["stochastic_seconds"] / timing["deterministic_seconds"]
            if timing["deterministic_seconds"] > 0
            else None
        ),
        "peak_gpu_memory_bytes": int(
            image_summaries["peak_gpu_memory_bytes"].max()
        ),
        "device": config["model"]["device"],
        "timing_scope": (
            "per-image preprocessing and synchronized warm-model inference; "
            "one-time model loading is excluded"
        ),
    }

    predictions_path = project_path(config, outputs["predictions"])
    metrics_path = project_path(config, outputs["metrics"])
    bootstrap_path = project_path(config, outputs["bootstrap"])
    image_predictions_path = project_path(
        config, outputs["image_predictions"]
    )
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(predictions_path, index=False)
    image_evaluation.to_parquet(image_predictions_path, index=False)
    pd.DataFrame(bootstrap_records).to_parquet(bootstrap_path, index=False)
    write_json(metrics_path, result)
    return result
