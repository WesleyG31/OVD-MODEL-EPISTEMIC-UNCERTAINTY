from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 15
) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.clip(np.digitize(probabilities, edges[1:-1]), 0, bins - 1)
    error = 0.0
    for bin_id in range(bins):
        mask = bin_ids == bin_id
        if not mask.any():
            continue
        error += mask.mean() * abs(labels[mask].mean() - probabilities[mask].mean())
    return float(error)


@dataclass(frozen=True)
class RiskCoverage:
    coverage: np.ndarray
    risk: np.ndarray
    aurc: float


def risk_coverage_curve(labels_error: np.ndarray, uncertainty: np.ndarray) -> RiskCoverage:
    labels_error = np.asarray(labels_error, dtype=np.float64)
    uncertainty = np.asarray(uncertainty, dtype=np.float64)
    if len(labels_error) == 0:
        return RiskCoverage(np.array([]), np.array([]), float("nan"))
    order = np.argsort(uncertainty, kind="stable")
    ordered_errors = labels_error[order]
    cumulative_risk = np.cumsum(ordered_errors) / np.arange(1, len(order) + 1)
    coverage = np.arange(1, len(order) + 1) / len(order)
    aurc = float(cumulative_risk.mean())
    return RiskCoverage(coverage=coverage, risk=cumulative_risk, aurc=aurc)


def risk_at_coverage(
    labels_error: np.ndarray, uncertainty: np.ndarray, coverage: float
) -> float:
    curve = risk_coverage_curve(labels_error, uncertainty)
    if len(curve.coverage) == 0:
        return float("nan")
    index = min(max(int(np.ceil(coverage * len(curve.coverage))) - 1, 0), len(curve.risk) - 1)
    return float(curve.risk[index])


def coverage_at_risk(
    labels_error: np.ndarray, uncertainty: np.ndarray, maximum_risk: float
) -> float:
    curve = risk_coverage_curve(labels_error, uncertainty)
    feasible = curve.coverage[curve.risk <= maximum_risk]
    return float(feasible.max()) if len(feasible) else 0.0


def binary_uncertainty_metrics(
    labels_error: np.ndarray,
    uncertainty: np.ndarray,
    calibrated_probability: bool = False,
    coverages: Iterable[float] = (0.5, 0.8, 0.9, 1.0),
    risk_targets: Iterable[float] = (),
) -> dict[str, float]:
    labels_error = np.asarray(labels_error, dtype=np.int64)
    uncertainty = np.asarray(uncertainty, dtype=np.float64)
    finite = np.isfinite(uncertainty)
    labels_error = labels_error[finite]
    uncertainty = uncertainty[finite]
    if len(np.unique(labels_error)) < 2:
        raise ValueError("Both correct and erroneous detections are required")

    curve = risk_coverage_curve(labels_error, uncertainty)
    metrics = {
        "n": int(len(labels_error)),
        "error_prevalence": float(labels_error.mean()),
        "auroc": float(roc_auc_score(labels_error, uncertainty)),
        "auprc": float(average_precision_score(labels_error, uncertainty)),
        "aurc": curve.aurc,
    }
    for coverage in coverages:
        metrics[f"risk_at_{coverage:.2f}"] = risk_at_coverage(
            labels_error, uncertainty, coverage
        )
    for risk in risk_targets:
        metrics[f"coverage_at_risk_{risk:.2f}"] = coverage_at_risk(
            labels_error, uncertainty, risk
        )
    if calibrated_probability:
        probability = np.clip(uncertainty, 1e-7, 1 - 1e-7)
        metrics.update(
            {
                "brier": float(brier_score_loss(labels_error, probability)),
                "nll": float(log_loss(labels_error, probability, labels=[0, 1])),
                "ece": expected_calibration_error(labels_error, probability),
            }
        )
    return metrics


def clustered_bootstrap(
    labels_error: np.ndarray,
    uncertainty: np.ndarray,
    groups: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    repetitions: int,
    seed: int,
) -> np.ndarray:
    labels_error = np.asarray(labels_error)
    uncertainty = np.asarray(uncertainty)
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    members = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    values = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([members[group] for group in sampled])
        try:
            values[repetition] = metric(
                labels_error[indices], uncertainty[indices]
            )
        except ValueError:
            values[repetition] = np.nan
    return values


def clustered_bootstrap_difference(
    labels_error: np.ndarray,
    candidate_uncertainty: np.ndarray,
    baseline_uncertainty: np.ndarray,
    groups: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    repetitions: int,
    seed: int,
    *,
    lower_is_better: bool = False,
) -> np.ndarray:
    labels_error = np.asarray(labels_error)
    candidate_uncertainty = np.asarray(candidate_uncertainty)
    baseline_uncertainty = np.asarray(baseline_uncertainty)
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    members = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    values = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([members[group] for group in sampled])
        try:
            candidate = metric(
                labels_error[indices], candidate_uncertainty[indices]
            )
            baseline = metric(
                labels_error[indices], baseline_uncertainty[indices]
            )
            values[repetition] = (
                baseline - candidate
                if lower_is_better
                else candidate - baseline
            )
        except ValueError:
            values[repetition] = np.nan
    return values


def percentile_interval(
    values: np.ndarray, confidence_level: float = 0.95
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    alpha = (1.0 - confidence_level) / 2.0
    return (
        float(np.quantile(values, alpha)),
        float(np.quantile(values, 1.0 - alpha)),
    )


def centered_bootstrap_p_value(
    bootstrap_estimates: np.ndarray,
    observed_estimate: float,
    *,
    null_value: float = 0.0,
) -> float:
    """One-sided upper-tail test from a null-centered bootstrap distribution.

    The alternative is ``estimate > null_value``. Ordinary paired cluster
    bootstrap replicates are centered at the observed estimate to approximate
    the sampling distribution under the boundary null. This is deliberately
    distinct from counting the sign of the uncentered bootstrap replicates.
    """

    finite = np.asarray(bootstrap_estimates, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    observed = float(observed_estimate)
    null = float(null_value)
    if len(finite) == 0 or not np.isfinite(observed) or not np.isfinite(null):
        return float("nan")
    centered_under_null = finite - observed + null
    return float(
        (1 + np.count_nonzero(centered_under_null >= observed))
        / (len(centered_under_null) + 1)
    )
