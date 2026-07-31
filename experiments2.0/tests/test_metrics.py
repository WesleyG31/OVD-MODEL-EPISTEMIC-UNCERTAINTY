import numpy as np

from adas_ovd.metrics import (
    centered_bootstrap_p_value,
    clustered_bootstrap_difference,
    coverage_at_risk,
    risk_coverage_curve,
)


def test_centered_bootstrap_p_value_uses_null_centering() -> None:
    estimates = np.array([0.8, 1.0, 1.2])
    assert centered_bootstrap_p_value(estimates, 1.0) == 1 / 4
    assert centered_bootstrap_p_value(
        estimates, 1.0, null_value=-0.5
    ) == 1 / 4


def test_risk_coverage_retains_low_uncertainty_first() -> None:
    labels_error = np.array([0, 0, 1, 1])
    uncertainty = np.array([0.1, 0.2, 0.8, 0.9])
    curve = risk_coverage_curve(labels_error, uncertainty)
    assert curve.risk[0] == 0.0
    assert curve.risk[1] == 0.0
    assert curve.risk[-1] == 0.5
    assert curve.aurc == np.mean([0.0, 0.0, 1.0 / 3.0, 0.5])
    assert coverage_at_risk(labels_error, uncertainty, 0.1) == 0.5


def test_paired_clustered_bootstrap_preserves_improvement_direction() -> None:
    labels_error = np.array([0, 1, 0, 1, 0, 1])
    candidate = np.array([0.0, 1.0, 0.1, 0.9, 0.2, 0.8])
    baseline = 1.0 - candidate
    groups = np.array(["a", "a", "b", "b", "c", "c"])

    values = clustered_bootstrap_difference(
        labels_error,
        candidate,
        baseline,
        groups,
        metric=lambda labels, uncertainty: float(
            np.mean(uncertainty[labels == 1])
            - np.mean(uncertainty[labels == 0])
        ),
        repetitions=20,
        seed=7,
    )

    assert np.all(values > 0)
