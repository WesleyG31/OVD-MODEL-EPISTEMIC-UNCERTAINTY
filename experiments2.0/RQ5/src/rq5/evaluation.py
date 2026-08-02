from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from adas_ovd.config import project_path
from adas_ovd.metrics import (
    binary_uncertainty_metrics,
    centered_bootstrap_p_value,
    clustered_bootstrap_difference,
    expected_calibration_error,
    percentile_interval,
)
from adas_ovd.reproducibility import (
    environment_metadata,
    sha256_file,
    stable_fingerprint,
    write_json,
)

from .extraction import read_validated_features
from .fusion import DecisionPolicy, load_policies, weighted_selective_risk
from .manifest import validate_manifest


def weighted_risk_coverage_curve(
    labels: np.ndarray, decision_risk: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    labels = np.asarray(labels, dtype=np.float64)
    risk = np.asarray(decision_risk, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    finite = np.isfinite(labels) & np.isfinite(risk) & np.isfinite(weights) & (weights > 0)
    labels, risk, weights = labels[finite], risk[finite], weights[finite]
    if len(labels) == 0:
        return np.array([]), np.array([]), float("nan")
    order = np.argsort(risk, kind="stable")
    ordered_labels = labels[order]
    ordered_weights = weights[order]
    cumulative = np.cumsum(ordered_weights * ordered_labels) / np.cumsum(
        ordered_weights
    )
    coverage = np.cumsum(ordered_weights) / ordered_weights.sum()
    weighted_aurc = float(np.average(cumulative, weights=ordered_weights))
    return coverage, cumulative, weighted_aurc


def weighted_risk_at_coverage(
    labels: np.ndarray,
    decision_risk: np.ndarray,
    weights: np.ndarray,
    coverage: float,
) -> float:
    curve_coverage, curve_risk, _ = weighted_risk_coverage_curve(
        labels, decision_risk, weights
    )
    if len(curve_coverage) == 0:
        return float("nan")
    index = min(
        int(np.searchsorted(curve_coverage, float(coverage), side="left")),
        len(curve_risk) - 1,
    )
    return float(curve_risk[index])


def coverage_at_weighted_risk(
    labels: np.ndarray,
    decision_risk: np.ndarray,
    weights: np.ndarray,
    maximum_risk: float,
) -> float:
    coverage, risk, _ = weighted_risk_coverage_curve(labels, decision_risk, weights)
    feasible = coverage[risk <= float(maximum_risk)]
    return float(feasible.max()) if len(feasible) else 0.0


def maximum_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 15
) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    bin_ids = np.clip(np.digitize(probabilities, edges[1:-1]), 0, int(bins) - 1)
    errors = [
        abs(float(labels[bin_ids == index].mean()) - float(probabilities[bin_ids == index].mean()))
        for index in range(int(bins))
        if (bin_ids == index).any()
    ]
    return float(max(errors)) if errors else float("nan")


def _method_metrics(
    frame: pd.DataFrame,
    policy: DecisionPolicy,
    config: dict[str, Any],
) -> dict[str, Any]:
    labels = frame["is_error"].to_numpy(dtype=np.int64)
    probability = policy.error_probability(frame)
    decision_risk = policy.decision_risk(frame)
    weights = frame["criticality_weight"].to_numpy(dtype=np.float64)
    if len(frame) == 0 or len(np.unique(labels)) < 2:
        return {"status": "not_estimable", "n": int(len(frame))}
    probability_metrics = binary_uncertainty_metrics(
        labels,
        probability,
        calibrated_probability=True,
        coverages=config["evaluation"]["risk_coverages"],
        risk_targets=config["evaluation"]["risk_targets"],
    )
    _, _, weighted_aurc = weighted_risk_coverage_curve(
        labels, decision_risk, weights
    )
    accepted = policy.accept(frame)
    metrics: dict[str, Any] = {
        "status": "estimable",
        **probability_metrics,
        "nll": float(log_loss(labels, probability, labels=[0, 1])),
        "weighted_aurc": weighted_aurc,
        "operating_threshold": float(policy.operating_threshold),
        "operating_coverage": float(accepted.mean()),
        "operating_criticality_mass_coverage": float(
            weights[accepted].sum() / weights.sum()
        ),
        "operating_defer_rate": float(1.0 - accepted.mean()),
        "operating_weighted_risk": weighted_selective_risk(
            labels, weights, accepted
        ),
        "accepted_errors": int(labels[accepted].sum()),
        "accepted_detections": int(accepted.sum()),
        "deferred_detections": int((~accepted).sum()),
        "accepted_criticality_mass": float(weights[accepted].sum()),
        "deferred_criticality_mass": float(weights[~accepted].sum()),
    }
    for coverage in config["evaluation"]["risk_coverages"]:
        metrics[f"weighted_risk_at_{float(coverage):.2f}"] = weighted_risk_at_coverage(
            labels, decision_risk, weights, float(coverage)
        )
    for target in config["evaluation"]["risk_targets"]:
        metrics[
            f"coverage_at_weighted_risk_{float(target):.2f}"
        ] = coverage_at_weighted_risk(
            labels, decision_risk, weights, float(target)
        )
    for bins in config["evaluation"]["calibration_bins"]:
        metrics[f"ece_{int(bins):02d}"] = expected_calibration_error(
            labels, probability, int(bins)
        )
        metrics[f"mce_{int(bins):02d}"] = maximum_calibration_error(
            labels, probability, int(bins)
        )
    return metrics


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    finite = [(name, float(value)) for name, value in p_values.items() if np.isfinite(value)]
    finite.sort(key=lambda item: item[1])
    adjusted: dict[str, float] = {name: float("nan") for name in p_values}
    running = 0.0
    count = len(finite)
    for index, (name, value) in enumerate(finite):
        running = max(running, min(1.0, (count - index) * value))
        adjusted[name] = running
    return adjusted


def _one_sided_p(
    improvements: np.ndarray,
    observed: float,
    *,
    null_value: float = 0.0,
) -> float:
    return centered_bootstrap_p_value(
        improvements, observed, null_value=null_value
    )


def _analysis_fingerprint(
    config: dict[str, Any],
    test_metadata: dict[str, Any],
    models_sha256: str,
    evaluation_source_sha256: str,
) -> str:
    return stable_fingerprint(
        {
            "manifest_sha256": test_metadata["manifest_sha256"],
            "test_features_sha256": test_metadata["features_sha256"],
            "models_sha256": models_sha256,
            "evaluation_source_sha256": evaluation_source_sha256,
            "inference": config["rq5"]["inference"],
            "realtime": config["rq5"]["realtime"],
        }
    )


def _bootstrap_difference(
    frame: pd.DataFrame,
    candidate: DecisionPolicy,
    baseline: DecisionPolicy,
    metric: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    *,
    lower_is_better: bool,
    repetitions: int,
    seed: int,
) -> np.ndarray:
    labels = frame["is_error"].to_numpy(dtype=np.int64)
    weights = frame["criticality_weight"].to_numpy(dtype=np.float64)
    candidate_risk = candidate.decision_risk(frame)
    baseline_risk = baseline.decision_risk(frame)
    groups = frame["sequence_id"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    members = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(int(seed))
    values = np.empty(int(repetitions), dtype=np.float64)
    for repetition in range(int(repetitions)):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([members[group] for group in sampled])
        candidate_value = metric(labels[indices], candidate_risk[indices], weights[indices])
        baseline_value = metric(labels[indices], baseline_risk[indices], weights[indices])
        values[repetition] = (
            baseline_value - candidate_value
            if lower_is_better
            else candidate_value - baseline_value
        )
    return values


def _weighted_aurc_metric(
    labels: np.ndarray, risk: np.ndarray, weights: np.ndarray
) -> float:
    return weighted_risk_coverage_curve(labels, risk, weights)[2]


def _coverage_target_metric(target: float) -> Callable[[np.ndarray, np.ndarray, np.ndarray], float]:
    return lambda labels, risk, weights: coverage_at_weighted_risk(
        labels, risk, weights, target
    )


def _latency_audit(
    config: dict[str, Any],
    frame: pd.DataFrame,
    image_summaries: pd.DataFrame,
    primary: DecisionPolicy,
) -> dict[str, Any]:
    specification = config["rq5"]["realtime"]
    image_ids = sorted(frame["image_id"].astype(int).unique())
    maximum = int(specification.get("benchmark_max_images", 128))
    benchmark_ids = image_ids[:maximum]
    timings: list[float] = []
    repetitions = int(specification["benchmark_repetitions"])
    groups = {
        image_id: frame.loc[frame["image_id"] == image_id]
        for image_id in benchmark_ids
    }
    warmup_repetitions = int(specification["benchmark_warmup_repetitions"])
    for _ in range(warmup_repetitions):
        for image_id in benchmark_ids:
            subset = groups[image_id]
            risk = primary.decision_risk(subset)
            np.less_equal(risk, primary.operating_threshold)
    for _ in range(repetitions):
        for image_id in benchmark_ids:
            subset = groups[image_id]
            started = time.perf_counter_ns()
            risk = primary.decision_risk(subset)
            np.less_equal(risk, primary.operating_threshold)
            timings.append((time.perf_counter_ns() - started) / 1_000_000.0)
    timing_values = np.asarray(timings, dtype=np.float64)
    decision = {
        "measurement": "direct_cpu_wall_clock",
        "operation": "single_decision_risk_plus_threshold",
        "benchmark_images": len(benchmark_ids),
        "warmup_repetitions": warmup_repetitions,
        "repetitions": repetitions,
        "calls": len(timings),
        "mean_ms_per_image": float(timing_values.mean()) if len(timing_values) else float("nan"),
        "p50_ms_per_image": float(np.quantile(timing_values, 0.50)) if len(timing_values) else float("nan"),
        "p95_ms_per_image": float(np.quantile(timing_values, 0.95)) if len(timing_values) else float("nan"),
    }
    full_passes = int(config["rq5"]["extraction"]["mc_passes"])
    prefixes = [0, *sorted(set(int(v) for v in config["rq5"]["extraction"]["mc_sensitivity_passes"]))]
    latency: dict[str, Any] = {}
    for count in prefixes:
        stochastic_fraction = float(count / full_passes)
        values_ms = 1000.0 * (
            image_summaries["preprocess_seconds"].to_numpy(dtype=np.float64)
            + image_summaries["deterministic_seconds"].to_numpy(dtype=np.float64)
            + stochastic_fraction
            * image_summaries["stochastic_seconds"].to_numpy(dtype=np.float64)
        ) + float(decision["p50_ms_per_image"])
        kind = (
            "measured_synchronized_gpu_blocks_plus_measured_decision"
            if count == full_passes
            else "linear_prefix_estimate_from_measured_stochastic_block_plus_measured_decision"
        )
        record = {
            "mc_passes": count,
            "measurement_kind": kind,
            "mean_ms": float(values_ms.mean()),
            "p50_ms": float(np.quantile(values_ms, 0.50)),
            "p95_ms": float(np.quantile(values_ms, 0.95)),
            "estimated_fps_from_mean": float(1000.0 / values_ms.mean()),
            "budgets_met": {
                f"{float(budget):g}_ms": bool(np.quantile(values_ms, 0.95) <= float(budget))
                for budget in specification["budgets_ms"]
            },
        }
        latency[f"mc{count:02d}"] = record
    operating = int(config["rq5"]["extraction"]["mc_operating_passes"])
    systems_gate = {
        "claim_mode": str(specification["claim_mode"]),
        "decision_p95_pass": bool(
            decision["p95_ms_per_image"]
            <= float(specification["maximum_decision_p95_ms"])
        ),
        "primary_latency_p95_pass": bool(
            latency[f"mc{operating:02d}"]["p95_ms"]
            <= float(specification["primary_budget_ms"])
        ),
        "total_latency_is_confirmatory_gate": bool(
            specification["enforce_total_latency_gate"]
        ),
    }
    systems_gate["pass"] = bool(
        systems_gate["decision_p95_pass"]
        and (
            systems_gate["primary_latency_p95_pass"]
            or not systems_gate["total_latency_is_confirmatory_gate"]
        )
    )
    return {
        "decision_overhead": decision,
        "prefix_latency": latency,
        "systems_gate": systems_gate,
        "peak_gpu_memory_bytes": int(image_summaries["peak_gpu_memory_bytes"].max()),
        "prefix_estimation_limitation": (
            "mc02/mc05 assume linear stochastic-pass cost; only mc10 GPU blocks were measured"
        ),
    }


def _subgroup_results(
    frame: pd.DataFrame,
    policy: DecisionPolicy,
    config: dict[str, Any],
) -> dict[str, Any]:
    minimum = int(config["evaluation"]["minimum_subgroup_rows"])
    result: dict[str, Any] = {}
    for attribute in (
        "category_name",
        "object_size",
        "criticality_tier",
        "timeofday",
        "weather",
        "scene",
    ):
        values: dict[str, Any] = {}
        for value, subset in frame.groupby(attribute, dropna=False):
            key = str(value)
            if len(subset) < minimum or subset["is_error"].nunique() < 2:
                values[key] = {"status": "not_estimable", "n": int(len(subset))}
            else:
                values[key] = _method_metrics(subset, policy, config)
        result[attribute] = values
    return result


def evaluate_policies(config: dict[str, Any]) -> dict[str, Any]:
    manifest_path, manifest = validate_manifest(config)
    outputs = config["rq5"]["outputs"]
    test, test_metadata = read_validated_features(
        config, project_path(config, outputs["test_features"])
    )
    threshold = float(config["rq5"]["fusion"]["training_score_threshold"])
    test = test.loc[test["score"] >= threshold].copy()
    if test.empty:
        raise ValueError("RQ5 operational test features are empty")
    policies = load_policies(config)
    primary_name = str(config["rq5"]["fusion"]["primary_method"])
    baselines = list(config["rq5"]["fusion"]["primary_baselines"])
    primary = policies[primary_name]

    predictions: list[pd.DataFrame] = []
    methods: dict[str, Any] = {}
    for name, policy in policies.items():
        probability = policy.error_probability(test)
        risk = policy.decision_risk(test)
        accepted = risk <= policy.operating_threshold
        prediction = test[
            [
                "image_id",
                "sequence_id",
                "detection_index",
                "category_name",
                "object_size",
                "score",
                "is_error",
                "criticality_weight",
                "criticality_tier",
            ]
        ].copy()
        prediction["method"] = name
        prediction["family"] = policy.family
        prediction["mc_passes"] = policy.mc_passes
        prediction["probability_error"] = probability
        prediction["decision_risk"] = risk
        prediction["action"] = np.where(accepted, "accept", "defer")
        predictions.append(prediction)
        methods[name] = _method_metrics(test, policy, config)

    prediction_frame = pd.concat(predictions, ignore_index=True)
    prediction_path = project_path(config, outputs["predictions"])
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_prediction = prediction_path.with_suffix(".parquet.tmp")
    prediction_frame.to_parquet(temporary_prediction, index=False)
    temporary_prediction.replace(prediction_path)

    bootstrap_records: list[dict[str, Any]] = []
    comparisons: dict[str, Any] = {}
    repetitions = int(config["evaluation"]["bootstrap_repetitions"])
    target = float(config["rq5"]["fusion"]["target_weighted_risk"])
    comparison_specs = {
        "weighted_aurc": (_weighted_aurc_metric, True),
        "coverage_at_weighted_risk_0.10": (_coverage_target_metric(target), False),
    }
    raw_p_values: dict[str, float] = {}
    seed = int(config["project"]["seed"])
    for baseline_index, baseline_name in enumerate(baselines):
        baseline = policies[baseline_name]
        for metric_index, (metric_name, (metric, lower_is_better)) in enumerate(
            comparison_specs.items()
        ):
            key = f"{metric_name}__{primary_name}__vs__{baseline_name}"
            values = _bootstrap_difference(
                test,
                primary,
                baseline,
                metric,
                lower_is_better=lower_is_better,
                repetitions=repetitions,
                seed=seed + baseline_index * 1009 + metric_index * 7919,
            )
            low, high = percentile_interval(
                values, float(config["evaluation"]["confidence_level"])
            )
            labels = test["is_error"].to_numpy(dtype=np.int64)
            weights = test["criticality_weight"].to_numpy(dtype=np.float64)
            candidate_point = metric(
                labels, primary.decision_risk(test), weights
            )
            baseline_point = metric(
                labels, baseline.decision_risk(test), weights
            )
            observed = (
                baseline_point - candidate_point
                if lower_is_better
                else candidate_point - baseline_point
            )
            raw_p_values[key] = _one_sided_p(values, observed)
            comparisons[key] = {
                "metric": metric_name,
                "candidate": primary_name,
                "baseline": baseline_name,
                "improvement": float(observed),
                "bootstrap_improvement_mean": float(np.nanmean(values)),
                "confidence_interval": [low, high],
                "p_value_one_sided": raw_p_values[key],
            }
            bootstrap_records.extend(
                {
                    "repetition": int(index),
                    "comparison": key,
                    "metric": metric_name,
                    "candidate": primary_name,
                    "baseline": baseline_name,
                    "improvement": float(value),
                }
                for index, value in enumerate(values)
            )
    margin = float(config["rq5"]["inference"]["brier_noninferiority_margin"])
    brier_inference: dict[str, Any] = {}
    labels = test["is_error"].to_numpy(dtype=np.int64)
    groups = test["sequence_id"].astype(str).to_numpy()
    for baseline_index, baseline_name in enumerate(baselines):
        key = f"brier_noninferiority__{primary_name}__vs__{baseline_name}"
        candidate_probability = primary.error_probability(test)
        baseline_probability = policies[baseline_name].error_probability(test)
        observed = float(
            np.mean((baseline_probability - labels) ** 2)
            - np.mean((candidate_probability - labels) ** 2)
        )
        values = clustered_bootstrap_difference(
            labels,
            candidate_probability,
            baseline_probability,
            groups,
            lambda y, p: float(np.mean((np.asarray(p) - np.asarray(y)) ** 2)),
            repetitions,
            seed + 50021 + baseline_index * 1009,
            lower_is_better=True,
        )
        low, high = percentile_interval(
            values, float(config["evaluation"]["confidence_level"])
        )
        raw_p_values[key] = _one_sided_p(
            values, observed, null_value=-margin
        )
        brier_inference[baseline_name] = {
            "candidate": primary_name,
            "baseline": baseline_name,
            "margin": margin,
            "improvement": observed,
            "confidence_interval": [low, high],
            "raw_one_sided_p": raw_p_values[key],
        }
        bootstrap_records.extend(
            {
                "repetition": int(index),
                "comparison": key,
                "metric": "brier_noninferiority",
                "candidate": primary_name,
                "baseline": baseline_name,
                "improvement": float(value),
            }
            for index, value in enumerate(values)
            if np.isfinite(value)
        )

    adjusted = holm_adjust(raw_p_values)
    alpha = float(config["rq5"]["inference"]["familywise_alpha"])
    for key in comparisons:
        value = adjusted[key]
        comparisons[key]["p_value_holm"] = value
        comparisons[key]["favorable_and_significant"] = bool(
            comparisons[key]["improvement"] > 0.0 and value <= alpha
        )
    for baseline_name, record in brier_inference.items():
        key = f"brier_noninferiority__{primary_name}__vs__{baseline_name}"
        record["holm_adjusted_p"] = adjusted[key]
        record["passed"] = bool(
            record["improvement"] > -margin
            and record["confidence_interval"][0] >= -margin
            and adjusted[key] <= alpha
        )
    brier_inference["pass"] = bool(
        all(brier_inference[name]["passed"] for name in baselines)
    )

    bootstrap_frame = pd.DataFrame(bootstrap_records)
    bootstrap_path = project_path(config, outputs["bootstrap"])
    bootstrap_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_bootstrap = bootstrap_path.with_suffix(".parquet.tmp")
    bootstrap_frame.to_parquet(temporary_bootstrap, index=False)
    temporary_bootstrap.replace(bootstrap_path)

    image_summary_path = project_path(config, outputs["test_image_summary"])
    image_summaries = pd.read_parquet(image_summary_path)
    latency = _latency_audit(config, test, image_summaries, primary)
    inference_pass = bool(
        all(record["favorable_and_significant"] for record in comparisons.values())
    )
    confirmatory = manifest["test_partition"] == "confirmatory"
    success = bool(
        confirmatory
        and inference_pass
        and brier_inference["pass"]
        and latency["systems_gate"]["pass"]
    )

    score_sensitivity: dict[str, Any] = {}
    full_test, _ = read_validated_features(
        config, project_path(config, outputs["test_features"])
    )
    for score_threshold in config["evaluation"]["score_threshold_sensitivity"]:
        subset = full_test.loc[full_test["score"] >= float(score_threshold)]
        score_sensitivity[f"{float(score_threshold):.2f}"] = _method_metrics(
            subset, primary, config
        )

    criticality_sensitivity: dict[str, Any] = {}
    severity = test["criticality_class_severity"].to_numpy(dtype=np.float64)
    bottomness = test["criticality_bottomness"].to_numpy(dtype=np.float64)
    centrality = test["criticality_centrality"].to_numpy(dtype=np.float64)
    for name, coefficients in config["rq5"]["criticality"][
        "sensitivity_coefficients"
    ].items():
        alternative = test.copy()
        alternative["criticality_weight"] = severity * (
            1.0
            + float(coefficients["bottomness"]) * bottomness
            + float(coefficients["centrality"]) * centrality
        )
        criticality_sensitivity[str(name)] = _method_metrics(
            alternative, primary, config
        )

    model_index_path = project_path(config, outputs["models"]) / "model_index.json"
    models_sha256 = sha256_file(model_index_path)
    evaluation_source_sha256 = sha256_file(Path(__file__).resolve())
    artifacts = {
        "test_features": {
            "path": str(project_path(config, outputs["test_features"])),
            "sha256": test_metadata["features_sha256"],
        },
        "test_image_summary": {
            "path": str(image_summary_path),
            "sha256": sha256_file(image_summary_path),
        },
        "model_index": {
            "path": str(model_index_path),
            "sha256": models_sha256,
        },
        "predictions": {"path": str(prediction_path), "sha256": sha256_file(prediction_path)},
        "bootstrap": {"path": str(bootstrap_path), "sha256": sha256_file(bootstrap_path)},
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "rq": "RQ5",
        "question": config["rq5"]["question"],
        "manifest": str(manifest_path),
        "manifest_test_partition": manifest["test_partition"],
        "evidence_status": (
            "confirmatory_evaluation" if confirmatory else "diagnostic_not_scientific_evidence"
        ),
        "test_rows": int(len(test)),
        "test_images": int(test["image_id"].nunique()),
        "methods": methods,
        "primary_inference": {
            "familywise_alpha": alpha,
            "test": "null_centered_paired_cluster_bootstrap",
            "holm_family_members": sorted(raw_p_values),
            "comparisons": comparisons,
            "holm_family_pass": inference_pass,
            "brier_noninferiority": brier_inference,
            "systems_gate": latency["systems_gate"],
            "success_status": (
                "pass" if success else "fail" if confirmatory else "diagnostic_not_eligible"
            ),
        },
        "mc_pass_sensitivity": {
            name: methods[name]
            for name in methods
            if name.startswith("risk_aware_fusion_mc") or name == primary_name
        },
        "score_threshold_sensitivity": score_sensitivity,
        "criticality_weight_sensitivity": criticality_sensitivity,
        "subgroups": _subgroup_results(test, primary, config),
        "computational_cost": latency,
        "artifact_integrity": artifacts,
        "shared_fingerprint": test_metadata["shared_fingerprint"],
        "analysis_source_sha256": evaluation_source_sha256,
        "analysis_fingerprint": _analysis_fingerprint(
            config,
            test_metadata,
            models_sha256,
            evaluation_source_sha256,
        ),
        "environment": environment_metadata(config["_meta"]["project_root"]),
        "limitations": [
            "Detection-conditioned evaluation does not measure missed objects or fallback outcomes.",
            "mc02/mc05 latency is a linear offline prefix estimate, not an online deadline measurement.",
            "Diagnostic mini outputs are technical checks and cannot answer RQ5 scientifically.",
        ],
    }
    metrics_path = project_path(config, outputs["metrics"])
    write_json(metrics_path, payload)
    write_json(
        prediction_path.with_suffix(".metadata.json"),
        {
            "schema_version": 1,
            "analysis_fingerprint": payload["analysis_fingerprint"],
            "rows": len(prediction_frame),
            "predictions_sha256": sha256_file(prediction_path),
            "test_features_sha256": test_metadata["features_sha256"],
        },
    )
    write_json(
        bootstrap_path.with_suffix(".metadata.json"),
        {
            "schema_version": 1,
            "analysis_fingerprint": payload["analysis_fingerprint"],
            "rows": len(bootstrap_frame),
            "bootstrap_sha256": sha256_file(bootstrap_path),
            "cluster": "sequence_id",
            "repetitions": repetitions,
        },
    )
    return payload
