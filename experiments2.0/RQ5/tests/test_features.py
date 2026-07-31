from __future__ import annotations

import numpy as np
import pytest

from rq5.features import (
    criticality_descriptors,
    detection_feature_bundle,
    finite_feature_audit,
    reference_iou_statistics,
)


SPECIFICATION = {
    "class_severity": {"person": 2.0, "car": 1.25},
    "bottomness_coefficient": 0.5,
    "centrality_coefficient": 0.5,
    "tier_boundaries": [1.75, 2.75],
}


def _bundle(present: np.ndarray) -> dict[str, float]:
    passes = len(present)
    return detection_feature_bundle(
        reference_box_cxcywh=np.array([0.5, 0.5, 0.2, 0.2]),
        category_scores=np.tile(np.array([[0.8, 0.2]]), (passes, 1)),
        scores=np.full(passes, 0.8),
        boxes_cxcywh=np.tile(np.array([[0.5, 0.5, 0.2, 0.2]]), (passes, 1)),
        embeddings=np.tile(np.array([[1.0, 0.0]]), (passes, 1)),
        present=present,
        base_category=0,
        deterministic_reference_variance=0.1,
        deterministic_reference_step=0.2,
        deterministic_hidden_step=0.3,
        bbox_area_fraction=0.04,
    )


def test_criticality_formula_and_tier() -> None:
    result = criticality_descriptors(
        np.array([40.0, 50.0, 60.0, 100.0]),
        width=100,
        height=100,
        category_name="person",
        specification=SPECIFICATION,
    )
    assert result["criticality_centrality"] == pytest.approx(1.0)
    assert result["criticality_bottomness"] == pytest.approx(1.0)
    assert result["criticality_geometry_factor"] == pytest.approx(2.0)
    assert result["criticality_weight"] == pytest.approx(4.0)
    assert result["criticality_tier"] == "high"


def test_criticality_rejects_unknown_class_and_bad_dimensions() -> None:
    with pytest.raises(KeyError):
        criticality_descriptors(
            np.zeros(4), width=10, height=10, category_name="bus", specification=SPECIFICATION
        )
    with pytest.raises(ValueError):
        criticality_descriptors(
            np.zeros(4), width=0, height=10, category_name="car", specification=SPECIFICATION
        )


def test_single_stochastic_observation_has_zero_dispersion() -> None:
    result = _bundle(np.array([True]))
    assert result["semantic_score_variance"] == pytest.approx(0.0)
    assert result["geometric_box_variance"] == pytest.approx(0.0)
    assert result["geometric_pairwise_iou_loss"] == pytest.approx(0.0)
    assert result["representation_embedding_variance"] == pytest.approx(0.0)
    assert result["mc_absence_rate"] == pytest.approx(0.0)


def test_complete_stochastic_absence_is_explicit() -> None:
    result = _bundle(np.array([False, False]))
    assert result["mc_absence_rate"] == pytest.approx(1.0)
    assert np.isnan(result["semantic_mutual_information"])
    assert np.isnan(result["geometric_box_variance"])
    assert np.isnan(result["representation_embedding_variance"])
    assert np.isnan(result["mc_score_mean"])


def test_missing_values_are_masked_by_presence() -> None:
    boxes = np.array([[0.5, 0.5, 0.2, 0.2], [np.nan, np.nan, np.nan, np.nan]])
    mean, standard_deviation = reference_iou_statistics(
        np.array([0.5, 0.5, 0.2, 0.2]), boxes, np.array([True, False])
    )
    assert mean == pytest.approx(1.0)
    assert standard_deviation == pytest.approx(0.0)


def test_reference_iou_rejects_malformed_arrays() -> None:
    with pytest.raises(ValueError):
        reference_iou_statistics(np.zeros(4), np.zeros((2, 3)), np.ones(2, dtype=bool))


def test_finite_feature_audit_allows_nan_but_rejects_infinity() -> None:
    result = finite_feature_audit(np.array([1.0, np.nan]))
    assert result["finite"] == 1
    with pytest.raises(ValueError):
        finite_feature_audit(np.array([np.inf]))
    with pytest.raises(ValueError):
        finite_feature_audit(np.array([np.nan]))
    assert finite_feature_audit(np.array([np.nan]), allow_all_missing=True)["finite"] == 0

