from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from adas_ovd.config import project_path
from adas_ovd.metrics import (
    binary_uncertainty_metrics,
    centered_bootstrap_p_value,
    expected_calibration_error,
    percentile_interval,
    risk_coverage_curve,
)
from adas_ovd.reproducibility import sha256_file, source_tree_sha256, stable_fingerprint, write_json

from .calibration import load_calibrations, load_sensitivity_calibrations
from .extraction import read_validated_features
from .features import finite_feature_audit


def _aurc(labels: np.ndarray, uncertainty: np.ndarray) -> float:
    return risk_coverage_curve(labels, uncertainty).aurc


def _brier(labels: np.ndarray, probability: np.ndarray) -> float:
    return float(brier_score_loss(labels, np.clip(probability, 1e-7, 1 - 1e-7)))


def _nll(labels: np.ndarray, probability: np.ndarray) -> float:
    return float(log_loss(labels, np.clip(probability, 1e-7, 1 - 1e-7), labels=[0, 1]))


def maximum_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    bin_ids = np.clip(np.digitize(probabilities, edges[1:-1]), 0, int(bins) - 1)
    gaps = [
        abs(float(labels[bin_ids == index].mean()) - float(probabilities[bin_ids == index].mean()))
        for index in range(int(bins))
        if np.any(bin_ids == index)
    ]
    return float(max(gaps)) if gaps else float("nan")


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
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


def _cluster_bootstrap_plan(
    groups: np.ndarray, repetitions: int, seed: int
) -> dict[str, np.ndarray]:
    groups = np.asarray(groups).astype(str)
    unique_groups, codes = np.unique(groups, return_inverse=True)
    if len(unique_groups) == 0:
        raise ValueError("RQ4 clustered bootstrap requires source groups")
    rng = np.random.default_rng(int(seed))
    counts = rng.multinomial(
        len(unique_groups),
        np.full(len(unique_groups), 1.0 / len(unique_groups)),
        size=int(repetitions),
    )
    if counts.max(initial=0) <= np.iinfo(np.int16).max:
        counts = counts.astype(np.int16)
    return {
        "counts": counts,
        "group_codes": codes.astype(np.int32),
        "group_sizes": np.bincount(codes, minlength=len(unique_groups)).astype(np.int64),
    }


def _weighted_cluster_bootstrap(
    labels: np.ndarray,
    values: np.ndarray,
    plan: dict[str, np.ndarray],
    metric: str,
    *,
    chunk_size: int = 32,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    counts = plan["counts"]
    codes = plan["group_codes"]
    group_sizes = plan["group_sizes"]
    if len(labels) != len(values) or len(labels) != len(codes):
        raise ValueError("RQ4 bootstrap inputs are not row-aligned")
    repetitions = len(counts)
    result = np.full(repetitions, np.nan, dtype=np.float64)
    denominators = counts.astype(np.float64) @ group_sizes.astype(np.float64)

    if metric in {"brier", "nll"}:
        clipped = np.clip(values, 1e-7, 1 - 1e-7)
        row_loss = (
            np.square(labels - clipped)
            if metric == "brier"
            else -(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped))
        )
        group_loss = np.bincount(codes, weights=row_loss, minlength=len(group_sizes))
        return np.divide(
            counts.astype(np.float64) @ group_loss,
            denominators,
            out=result,
            where=denominators > 0,
        )

    if metric == "ece":
        bins = 15
        bin_ids = np.clip(
            np.digitize(values, np.linspace(0.0, 1.0, bins + 1)[1:-1]), 0, bins - 1
        )
        flat_ids = codes * bins + bin_ids
        shape = (len(group_sizes), bins)
        group_count = np.bincount(flat_ids, minlength=np.prod(shape)).reshape(shape)
        group_label = np.bincount(flat_ids, weights=labels, minlength=np.prod(shape)).reshape(shape)
        group_probability = np.bincount(flat_ids, weights=values, minlength=np.prod(shape)).reshape(shape)
        sampled_label = counts.astype(np.float64) @ group_label
        sampled_probability = counts.astype(np.float64) @ group_probability
        result = np.abs(sampled_label - sampled_probability).sum(axis=1) / denominators
        return result

    if metric not in {"auroc", "auprc", "aurc"}:
        raise ValueError(f"Unsupported optimized RQ4 bootstrap metric: {metric}")

    descending = metric == "auprc"
    order = np.argsort(-values if descending else values, kind="stable")
    ordered_values = values[order]
    ordered_labels = labels[order]
    ordered_codes = codes[order]
    tie_starts = np.r_[0, np.flatnonzero(np.diff(ordered_values) != 0.0) + 1]
    max_total = int(denominators.max(initial=0))
    harmonic = None
    if metric == "aurc":
        harmonic = np.zeros(max_total + 1, dtype=np.float64)
        if max_total:
            harmonic[1:] = np.cumsum(1.0 / np.arange(1, max_total + 1, dtype=np.float64))

    for start in range(0, repetitions, int(chunk_size)):
        stop = min(start + int(chunk_size), repetitions)
        weights = counts[start:stop, ordered_codes].astype(np.int64, copy=False)
        total = denominators[start:stop]
        if metric in {"auroc", "auprc"}:
            positive = np.add.reduceat(weights * ordered_labels, tie_starts, axis=1)
            negative = np.add.reduceat(weights * (1.0 - ordered_labels), tie_starts, axis=1)
            total_positive = positive.sum(axis=1)
            total_negative = negative.sum(axis=1)
            if metric == "auroc":
                negative_before = np.cumsum(negative, axis=1) - negative
                numerator = np.sum(
                    positive * (negative_before + 0.5 * negative), axis=1
                )
                denominator = total_positive * total_negative
            else:
                cumulative_positive = np.cumsum(positive, axis=1)
                cumulative_total = cumulative_positive + np.cumsum(negative, axis=1)
                precision = np.divide(
                    cumulative_positive,
                    cumulative_total,
                    out=np.zeros_like(cumulative_positive),
                    where=cumulative_total > 0,
                )
                numerator = np.sum(precision * positive, axis=1)
                denominator = total_positive
            result[start:stop] = np.divide(
                numerator,
                denominator,
                out=np.full(stop - start, np.nan, dtype=np.float64),
                where=denominator > 0,
            )
        else:
            assert harmonic is not None
            cumulative_count = np.cumsum(weights, axis=1)
            cumulative_errors = np.cumsum(weights * ordered_labels, axis=1)
            count_before = cumulative_count - weights
            errors_before = cumulative_errors - weights * ordered_labels
            harmonic_delta = harmonic[cumulative_count] - harmonic[count_before]
            contribution = np.where(
                ordered_labels[None, :] > 0.5,
                weights + (errors_before - count_before) * harmonic_delta,
                errors_before * harmonic_delta,
            )
            result[start:stop] = contribution.sum(axis=1) / total
    return result


def _method_metrics(
    labels: np.ndarray, rank: np.ndarray, probability: np.ndarray, config: dict[str, Any]
) -> dict[str, float]:
    rank_metrics = binary_uncertainty_metrics(
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
    rank_metrics.update({name: calibrated[name] for name in ("brier", "nll", "ece")})
    rank_metrics["mce"] = maximum_calibration_error(labels, probability)
    return rank_metrics


def _estimable_record(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    methods: list[str],
    config: dict[str, Any],
    *,
    minimum_rows: int,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "rows": int(len(frame)),
        "sequences": int(frame["sequence_id"].astype(str).nunique()) if len(frame) else 0,
        "status": "estimable",
        "error_prevalence": float(frame["is_error"].mean()) if len(frame) else None,
        "methods": {},
    }
    if len(frame) < int(minimum_rows):
        record["status"] = "not_estimable_minimum_rows"
    elif frame["is_error"].nunique() != 2:
        record["status"] = "not_estimable_two_outcomes_required"
    elif frame["sequence_id"].astype(str).nunique() == 0:
        record["status"] = "not_estimable_source_groups_required"
    if record["status"] != "estimable":
        return record
    labels = frame["is_error"].to_numpy(dtype=np.int64)
    for method in methods:
        rank = predictions.loc[frame.index, f"rank_{method}"].to_numpy(dtype=np.float64)
        probability = predictions.loc[frame.index, f"prob_error_{method}"].to_numpy(dtype=np.float64)
        if not np.isfinite(rank).all() or not np.isfinite(probability).all():
            record["methods"][method] = {"status": "not_estimable_nonfinite_output"}
        else:
            record["methods"][method] = {
                "status": "estimable",
                "metrics": _method_metrics(labels, rank, probability, config),
            }
    return record


def _bootstrap_intervals(
    *,
    labels: np.ndarray,
    rank: np.ndarray,
    probability: np.ndarray,
    plan: dict[str, np.ndarray],
    method: str,
    config: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    inputs: dict[str, np.ndarray] = {
        "auroc": rank,
        "auprc": rank,
        "aurc": rank,
        "brier": probability,
        "nll": probability,
        "ece": probability,
    }
    confidence_level = float(config["evaluation"]["confidence_level"])
    intervals = {}
    for metric, values_input in inputs.items():
        values = _weighted_cluster_bootstrap(labels, values_input, plan, metric)
        lower, upper = percentile_interval(values, confidence_level)
        intervals[metric] = {"lower": lower, "upper": upper}
        records.extend(
            {
                "analysis": "method_interval",
                "domain": "shifted",
                "method": method,
                "baseline": None,
                "metric": metric,
                "repetition": index,
                "value": float(value),
            }
            for index, value in enumerate(values) if np.isfinite(value)
        )
    return intervals


def _evaluation_identity(
    config: dict[str, Any],
    *,
    test_features_sha256: str,
    test_image_summary_sha256: str,
    model_index_sha256: str,
) -> str:
    source_sha256 = source_tree_sha256(config["_meta"]["project_root"], ("RQ4/src/rq4/evaluation.py",))
    return stable_fingerprint(
        {
            "schema_version": 2,
            "source_tree_sha256": source_sha256,
            "test_features_sha256": test_features_sha256,
            "test_image_summary_sha256": test_image_summary_sha256,
            "model_index_sha256": model_index_sha256,
            "evaluation": config["evaluation"],
            "domain_shift": config["rq4"]["domain_shift"],
            "calibration": config["rq4"]["calibration"],
            "inference": config["rq4"]["inference"],
        }
    )


def _load_cached_evaluation(
    *,
    identity: str,
    metrics_path: Path,
    predictions_path: Path,
    bootstrap_path: Path,
    model_index_sha256: str,
    test_features_sha256: str,
    test_image_summary_sha256: str,
) -> dict[str, Any] | None:
    if not all(path.is_file() for path in (metrics_path, predictions_path, bootstrap_path)):
        return None
    try:
        with metrics_path.open("r", encoding="utf-8") as handle:
            cached = json.load(handle)
        integrity = cached["artifact_integrity"]
        valid = (
            cached.get("evaluation_identity") == identity
            and integrity.get("test_features_sha256") == test_features_sha256
            and integrity.get("test_image_summary_sha256") == test_image_summary_sha256
            and integrity.get("model_index_sha256") == model_index_sha256
            and integrity.get("predictions_sha256") == sha256_file(predictions_path)
            and integrity.get("bootstrap_sha256") == sha256_file(bootstrap_path)
        )
        return cached if valid else None
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def evaluate_calibrations(config: dict[str, Any]) -> dict[str, Any]:
    outputs = config["rq4"]["outputs"]
    extracted, test_metadata = read_validated_features(config, project_path(config, outputs["test_features"]))
    image_summary_path = project_path(config, outputs["test_image_summary"])
    if not image_summary_path.is_file() or test_metadata.get("image_summary_sha256") != sha256_file(image_summary_path):
        raise RuntimeError("RQ4 test image-summary integrity failed")
    image_summaries = pd.read_parquet(image_summary_path)
    if extracted.empty:
        raise ValueError("RQ4 test feature artifact is empty")

    methods = load_calibrations(config)
    sensitivities = load_sensitivity_calibrations(config)
    predictions_path = project_path(config, outputs["predictions"])
    bootstrap_path = project_path(config, outputs["bootstrap"])
    metrics_path = project_path(config, outputs["metrics"])
    model_index_path = project_path(config, outputs["models"]) / "model_index.json"
    model_index_sha256 = sha256_file(model_index_path)
    evaluation_identity = _evaluation_identity(
        config,
        test_features_sha256=test_metadata["features_sha256"],
        test_image_summary_sha256=test_metadata["image_summary_sha256"],
        model_index_sha256=model_index_sha256,
    )
    cached = _load_cached_evaluation(
        identity=evaluation_identity,
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        bootstrap_path=bootstrap_path,
        model_index_sha256=model_index_sha256,
        test_features_sha256=test_metadata["features_sha256"],
        test_image_summary_sha256=test_metadata["image_summary_sha256"],
    )
    if cached is not None:
        return cached

    for feature in test_metadata["feature_schema"]:
        if feature in {"file_name", "sequence_id", "timeofday", "weather", "scene", "category_name", "object_size", "domain_stratum"}:
            continue
        if pd.api.types.is_numeric_dtype(extracted[feature]):
            finite_feature_audit(extracted[feature].to_numpy(), allow_all_missing=feature not in {"score", "is_error"})

    prediction_columns = [
        "image_id", "file_name", "sequence_id", "timeofday", "weather", "scene",
        "detection_index", "query_index", "category_id", "category_name", "object_size",
        "score", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "is_error",
        "is_class_correct", "is_well_localized", "localization_iou",
        "shift_timeofday", "shift_weather", "shift_scene", "unknown_timeofday",
        "unknown_weather", "unknown_scene", "shift_axis_count", "unknown_axis_count",
        "is_domain_shift", "domain_stratum",
    ]
    all_predictions = extracted[prediction_columns].copy()
    prediction_started = time.perf_counter()
    for method, calibration in {**methods, **sensitivities}.items():
        all_predictions[f"rank_{method}"] = calibration.rank_score(extracted)
        all_predictions[f"prob_error_{method}"] = calibration.error_probability(extracted)
    prediction_seconds = time.perf_counter() - prediction_started

    threshold = float(config["evaluation"]["primary_score_threshold"])
    operating_mask = extracted["score"].to_numpy(dtype=np.float64) >= threshold
    test = extracted.loc[operating_mask].copy()
    predictions = all_predictions.loc[operating_mask].copy()
    if test.empty or test["is_error"].nunique() != 2:
        raise ValueError("RQ4 operational test requires correct and error detections")
    partition = test_metadata["manifest_test_partition"]
    result: dict[str, Any] = {
        "schema_version": 2,
        "evaluation_identity": evaluation_identity,
        "research_question": config["rq4"]["question"],
        "protocol_freeze_date": "2026-07-31",
        "test_partition": partition,
        "evidence_status": "diagnostic_not_scientific_evidence" if partition == "diagnostic" else "confirmatory_requires_human_interpretation",
        "extracted_test_rows": len(extracted),
        "primary_score_threshold": threshold,
        "test_rows": len(test),
        "test_images": int(image_summaries["image_id"].nunique()),
        "test_sequences": int(image_summaries["sequence_id"].nunique()),
        "domain_definition": config["rq4"]["domain_shift"],
        "methods": {},
        "primary_inference": {},
        "domain_analysis": {},
        "domain_gaps": {},
        "mc_pass_sensitivity": {},
        "score_threshold_sensitivity": {},
        "ece_bin_sensitivity": {},
        "subgroups": {},
    }
    bootstrap_records: list[dict[str, Any]] = []
    minimum_domain_rows = int(config["rq4"]["domain_shift"]["minimum_rows"])
    shifted = test.loc[test["is_domain_shift"].astype(bool)]
    shifted_record = _estimable_record(test.loc[shifted.index], predictions, list(methods), config, minimum_rows=minimum_domain_rows)
    result["domain_analysis"]["shifted"] = shifted_record
    reference = test.loc[~test["is_domain_shift"].astype(bool)]
    result["domain_analysis"]["reference"] = _estimable_record(reference, predictions, list(methods), config, minimum_rows=minimum_domain_rows)

    bootstrap_plan: dict[str, np.ndarray] | None = None
    if shifted_record["status"] == "estimable":
        labels = shifted["is_error"].to_numpy(dtype=np.int64)
        groups = shifted["sequence_id"].astype(str).to_numpy()
        bootstrap_plan = _cluster_bootstrap_plan(
            groups,
            int(config["evaluation"]["bootstrap_repetitions"]),
            int(config["project"]["seed"]) + sum(ord(character) for character in "rq4_common_bootstrap"),
        )
        result["bootstrap_design"] = {
            "implementation": "exact_cluster_multiplicity_weights",
            "common_draws_across_methods_and_metrics": True,
            "repetitions": int(config["evaluation"]["bootstrap_repetitions"]),
            "clusters": int(len(np.unique(groups))),
        }
        for method, calibration in methods.items():
            rank = predictions.loc[shifted.index, f"rank_{method}"].to_numpy(dtype=np.float64)
            probability = predictions.loc[shifted.index, f"prob_error_{method}"].to_numpy(dtype=np.float64)
            result["methods"][method] = {
                "family": calibration.family,
                "features": calibration.features,
                "coefficient_count": calibration.coefficient_count,
                "metrics": _method_metrics(labels, rank, probability, config),
                "clustered_bootstrap_interval": _bootstrap_intervals(
                    labels=labels, rank=rank, probability=probability, plan=bootstrap_plan,
                    method=method, config=config, records=bootstrap_records,
                ),
            }
    else:
        result["methods_status"] = shifted_record["status"]

    primary = str(config["rq4"]["calibration"]["primary_method"])
    baselines = list(config["rq4"]["calibration"]["primary_baselines"])
    if shifted_record["status"] != "estimable":
        result["primary_inference"] = {
            "status": shifted_record["status"],
            "success_criterion_met": False,
            "interpretation_guard": "Primary shifted-domain metrics are not estimable.",
        }
    else:
        assert bootstrap_plan is not None
        labels = shifted["is_error"].to_numpy(dtype=np.int64)
        functions: dict[str, tuple[Callable[[np.ndarray, np.ndarray], float], str]] = {
            "brier": (_brier, "prob_error"), "nll": (_nll, "prob_error"), "aurc": (_aurc, "rank")
        }
        comparisons: dict[str, Any] = {}
        raw_p_values: dict[str, float] = {}
        confidence_level = float(config["evaluation"]["confidence_level"])
        for baseline in baselines:
            comparisons[baseline] = {}
            for metric in config["rq4"]["inference"]["primary_metrics"]:
                function, prefix = functions[metric]
                candidate = predictions.loc[shifted.index, f"{prefix}_{primary}"].to_numpy(dtype=np.float64)
                baseline_values = predictions.loc[shifted.index, f"{prefix}_{baseline}"].to_numpy(dtype=np.float64)
                candidate_point = function(labels, candidate)
                baseline_point = function(labels, baseline_values)
                improvement = baseline_point - candidate_point
                name = f"{metric}_vs_{baseline}"
                values = (
                    _weighted_cluster_bootstrap(
                        labels, baseline_values, bootstrap_plan, metric
                    )
                    - _weighted_cluster_bootstrap(
                        labels, candidate, bootstrap_plan, metric
                    )
                )
                raw_p_values[name] = _one_sided_bootstrap_p(values, improvement)
                lower, upper = percentile_interval(values, confidence_level)
                comparisons[baseline][metric] = {
                    "candidate": float(candidate_point), "baseline": float(baseline_point),
                    "improvement": float(improvement),
                    "relative_improvement": float(improvement / abs(baseline_point)) if baseline_point != 0 else None,
                    "lower": lower, "upper": upper,
                    "raw_one_sided_p": raw_p_values[name], "positive_is_better": True,
                }
                bootstrap_records.extend(
                    {
                        "analysis": "primary_difference", "domain": "shifted", "method": primary,
                        "baseline": baseline, "metric": metric, "repetition": index, "value": float(value),
                    }
                    for index, value in enumerate(values) if np.isfinite(value)
                )
        adjusted = holm_adjust(raw_p_values)
        alpha = float(config["rq4"]["inference"]["familywise_alpha"])
        decisions = []
        for baseline, metrics in comparisons.items():
            for metric, record in metrics.items():
                name = f"{metric}_vs_{baseline}"
                record["holm_adjusted_p"] = adjusted[name]
                record["reject_no_improvement"] = bool(adjusted[name] < alpha)
                decisions.append(bool(record["improvement"] > 0 and record["reject_no_improvement"]))
        nominal = bool(all(decisions))
        result["primary_inference"] = {
            "status": "estimable", "domain": "shifted", "method": primary, "baselines": baselines,
            "test": "null_centered_paired_cluster_bootstrap",
            "metrics": list(config["rq4"]["inference"]["primary_metrics"]),
            "familywise_alpha": alpha, "correction": config["rq4"]["inference"]["correction"],
            "comparisons": comparisons, "nominal_criterion_met": nominal,
            "success_criterion_met": bool(nominal and partition == "confirmatory"),
            "interpretation_guard": "Diagnostic inference cannot satisfy the scientific success rule." if partition == "diagnostic" else "Preserve mixed or negative outcomes without test retuning.",
        }

    # Prespecified overlapping axis, severity and observed-value strata.
    for axis in config["rq4"]["domain_shift"]["attributes"]:
        mask = test[f"shift_{axis}"].astype(bool) & ~test[f"unknown_{axis}"].astype(bool)
        result["domain_analysis"][f"shift_{axis}"] = _estimable_record(test.loc[mask], predictions, list(methods), config, minimum_rows=minimum_domain_rows)
    for severity in range(len(config["rq4"]["domain_shift"]["attributes"]) + 1):
        mask = test["shift_axis_count"].astype(int) == severity
        result["domain_analysis"][f"severity_{severity}"] = _estimable_record(test.loc[mask], predictions, list(methods), config, minimum_rows=minimum_domain_rows)
    for attribute in config["rq4"]["domain_shift"]["attributes"]:
        result["subgroups"][attribute] = {}
        for value in sorted(test[attribute].astype(str).unique()):
            mask = test[attribute].astype(str) == value
            result["subgroups"][attribute][value] = _estimable_record(test.loc[mask], predictions, list(methods), config, minimum_rows=minimum_domain_rows)
    for attribute in ("category_name", "object_size"):
        result["subgroups"][attribute] = {}
        for value in sorted(test[attribute].astype(str).unique()):
            mask = test[attribute].astype(str) == value
            result["subgroups"][attribute][value] = _estimable_record(test.loc[mask], predictions, list(methods), config, minimum_rows=int(config["evaluation"]["minimum_subgroup_rows"]))

    if result["domain_analysis"]["reference"]["status"] == "estimable" and shifted_record["status"] == "estimable":
        for method in methods:
            reference_metrics = result["domain_analysis"]["reference"]["methods"][method]["metrics"]
            shifted_metrics = shifted_record["methods"][method]["metrics"]
            result["domain_gaps"][method] = {
                metric: float(shifted_metrics[metric] - reference_metrics[metric])
                for metric in ("brier", "nll", "ece", "aurc")
            }
    else:
        result["domain_gaps_status"] = "not_estimable_reference_and_shifted_required"

    if shifted_record["status"] == "estimable":
        labels = shifted["is_error"].to_numpy(dtype=np.int64)
        for method, calibration in sensitivities.items():
            result["mc_pass_sensitivity"][method] = {
                "mc_passes": int(method.rsplit("mc", 1)[1]),
                "features": calibration.features,
                "metrics": _method_metrics(
                    labels,
                    predictions.loc[shifted.index, f"rank_{method}"].to_numpy(dtype=np.float64),
                    predictions.loc[shifted.index, f"prob_error_{method}"].to_numpy(dtype=np.float64),
                    config,
                ),
            }

    for score_threshold in config["evaluation"]["score_threshold_sensitivity"]:
        value = float(score_threshold)
        mask = (extracted["score"] >= value) & extracted["is_domain_shift"].astype(bool)
        result["score_threshold_sensitivity"][f"{value:.2f}"] = _estimable_record(
            extracted.loc[mask], all_predictions, list(methods), config, minimum_rows=minimum_domain_rows
        )
    if shifted_record["status"] == "estimable":
        labels = shifted["is_error"].to_numpy(dtype=np.int64)
        for method in methods:
            probability = predictions.loc[shifted.index, f"prob_error_{method}"].to_numpy(dtype=np.float64)
            result["ece_bin_sensitivity"][method] = {
                str(int(bins)): expected_calibration_error(labels, probability, int(bins))
                for bins in config["evaluation"]["calibration_bins"]
            }

    result["computational_cost"] = {
        "images": len(image_summaries),
        "mc_passes": int(config["rq4"]["extraction"]["mc_passes"]),
        "preprocess_seconds": float(image_summaries["preprocess_seconds"].sum()),
        "deterministic_detector_seconds": float(image_summaries["deterministic_seconds"].sum()),
        "stochastic_detector_seconds": float(image_summaries["stochastic_seconds"].sum()),
        "rq4_feature_aggregation_seconds": float(image_summaries["aggregation_seconds"].sum()),
        "all_method_prediction_seconds": prediction_seconds,
        "peak_gpu_memory_bytes": int(image_summaries["peak_gpu_memory_bytes"].max()),
        "shared_shards_computed_for_test_request": int(test_metadata["shared_shards_computed"]),
        "shared_shards_reused_for_test_request": int(test_metadata["shared_shards_reused"]),
        "stochastic_modules": test_metadata["stochastic_modules"],
        "environment": test_metadata["environment"],
    }
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_predictions = predictions_path.with_suffix(".parquet.tmp")
    predictions.to_parquet(temporary_predictions, index=False)
    temporary_predictions.replace(predictions_path)
    temporary_bootstrap = bootstrap_path.with_suffix(".parquet.tmp")
    pd.DataFrame(bootstrap_records).to_parquet(temporary_bootstrap, index=False)
    temporary_bootstrap.replace(bootstrap_path)
    required_finite = all(
        np.isfinite(value)
        for method in result.get("methods", {}).values()
        for value in method.get("metrics", {}).values()
        if isinstance(value, (int, float))
    )
    if result.get("methods") and not required_finite:
        raise ValueError("RQ4 produced a non-finite required point metric")
    result["artifact_integrity"] = {
        "test_features_sha256": test_metadata["features_sha256"],
        "test_image_summary_sha256": test_metadata["image_summary_sha256"],
        "shared_request_metadata_sha256": test_metadata["shared_request_metadata_sha256"],
        "model_index_sha256": model_index_sha256,
        "predictions_sha256": sha256_file(predictions_path),
        "bootstrap_sha256": sha256_file(bootstrap_path),
        "all_required_point_metrics_finite": required_finite,
    }
    write_json(metrics_path, result)
    return result
