from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rq5.fusion import (
    _fit_component,
    select_operating_threshold,
    split_validation_groups,
    validate_no_target_leakage,
    weighted_selective_risk,
)


def test_leakage_guard_rejects_targets_and_decision_descriptors() -> None:
    with pytest.raises(RuntimeError):
        validate_no_target_leakage({"bad": ["matched_iou"]})
    with pytest.raises(RuntimeError):
        validate_no_target_leakage({"bad": ["criticality_weight"]})
    with pytest.raises(RuntimeError):
        validate_no_target_leakage({"bad": ["x", "x"]})
    validate_no_target_leakage({"good": ["confidence_uncertainty", "feature_mc02"]})


def test_three_way_group_split_has_no_overlap() -> None:
    frame = pd.DataFrame(
        {
            "sequence_id": np.repeat([f"g{index}" for index in range(6)], 4),
            "is_error": np.tile([0, 1, 0, 1], 6),
        }
    )
    result = split_validation_groups(
        frame,
        group_column="sequence_id",
        fractions={
            "selection": 0.4,
            "component_calibration": 0.3,
            "policy_calibration": 0.3,
        },
        seed=7,
        minimum_samples={
            "selection": 1,
            "component_calibration": 1,
            "policy_calibration": 1,
        },
    )
    group_sets = [set(value["sequence_id"]) for value in result.values()]
    assert not (group_sets[0] & group_sets[1])
    assert not (group_sets[0] & group_sets[2])
    assert not (group_sets[1] & group_sets[2])
    assert sum(map(len, result.values())) == len(frame)


def test_three_way_split_rejects_too_few_groups() -> None:
    frame = pd.DataFrame({"sequence_id": ["a", "b"], "is_error": [0, 1]})
    with pytest.raises(ValueError):
        split_validation_groups(
            frame,
            group_column="sequence_id",
            fractions={"selection": 0.4, "component_calibration": 0.3, "policy_calibration": 0.3},
            seed=1,
            minimum_samples={"selection": 1, "component_calibration": 1, "policy_calibration": 1},
        )


def test_operating_threshold_respects_weighted_risk() -> None:
    threshold, audit = select_operating_threshold(
        np.array([0, 1, 0]),
        np.array([0.1, 0.2, 0.3]),
        np.ones(3),
        0.10,
    )
    assert threshold == pytest.approx(0.1)
    assert audit["coverage"] == pytest.approx(1 / 3)
    assert audit["criticality_mass_coverage"] == pytest.approx(1 / 3)
    assert audit["weighted_risk"] == pytest.approx(0.0)


def test_operating_threshold_can_defer_all() -> None:
    threshold, audit = select_operating_threshold(
        np.array([1, 1]), np.array([0.1, 0.2]), np.ones(2), 0.0
    )
    assert threshold == float("-inf")
    assert audit["coverage"] == 0.0
    assert audit["criticality_mass_coverage"] == 0.0


def test_weighted_selective_risk_and_empty_acceptance() -> None:
    assert weighted_selective_risk(
        np.array([0, 1]), np.array([1.0, 3.0]), np.array([True, True])
    ) == pytest.approx(0.75)
    assert np.isnan(
        weighted_selective_risk(
            np.array([0, 1]), np.ones(2), np.array([False, False])
        )
    )


def test_logistic_component_fits_deterministically() -> None:
    train = pd.DataFrame(
        {"x": [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0], "is_error": [0, 0, 0, 1, 1, 1]}
    )
    selection = pd.DataFrame({"x": [-1.5, -0.25, 0.25, 1.5], "is_error": [0, 0, 1, 1]})
    config = {
        "project": {"seed": 17},
        "rq5": {"fusion": {"regularization_grid": [0.1, 1.0]}},
    }
    first = _fit_component(config, train, selection, name="toy", features=["x"])
    second = _fit_component(config, train, selection, name="toy", features=["x"])
    np.testing.assert_array_equal(
        first.raw_probability(selection), second.raw_probability(selection)
    )
    assert first.selection_auroc == pytest.approx(1.0)
