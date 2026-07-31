from __future__ import annotations

import numpy as np
import pytest

from rq5.evaluation import (
    _one_sided_p,
    coverage_at_weighted_risk,
    holm_adjust,
    maximum_calibration_error,
    weighted_risk_at_coverage,
    weighted_risk_coverage_curve,
)


def test_weighted_risk_coverage_formula() -> None:
    coverage, risk, aurc = weighted_risk_coverage_curve(
        np.array([0, 1, 1]),
        np.array([0.1, 0.2, 0.3]),
        np.array([1.0, 2.0, 1.0]),
    )
    np.testing.assert_allclose(coverage, [1 / 4, 3 / 4, 1.0])
    np.testing.assert_allclose(risk, [0.0, 2 / 3, 3 / 4])
    assert aurc == pytest.approx(np.average([0.0, 2 / 3, 3 / 4], weights=[1, 2, 1]))


def test_weighted_risk_empty_and_operating_points() -> None:
    coverage, risk, aurc = weighted_risk_coverage_curve(
        np.array([]), np.array([]), np.array([])
    )
    assert len(coverage) == len(risk) == 0
    assert np.isnan(aurc)
    labels = np.array([0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3])
    weights = np.ones(3)
    assert weighted_risk_at_coverage(labels, scores, weights, 2 / 3) == pytest.approx(0.5)
    assert coverage_at_weighted_risk(labels, scores, weights, 0.5) == pytest.approx(2 / 3)


def test_holm_adjustment_is_monotone() -> None:
    result = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.20})
    assert result["a"] == pytest.approx(0.03)
    assert result["b"] == pytest.approx(0.06)
    assert result["c"] == pytest.approx(0.20)


def test_maximum_calibration_error() -> None:
    value = maximum_calibration_error(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.7, 0.8]), bins=2
    )
    assert value == pytest.approx(0.25)


def test_noninferiority_p_uses_margin_as_null_boundary() -> None:
    estimates = np.array([-0.002, 0.0, 0.002])
    assert _one_sided_p(estimates, 0.0, null_value=-0.01) == pytest.approx(
        1 / 4
    )
