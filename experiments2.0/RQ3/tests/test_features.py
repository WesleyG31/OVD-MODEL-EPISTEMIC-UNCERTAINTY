from __future__ import annotations

import numpy as np

from rq3.features import (
    class_agnostic_localization_targets,
    detection_feature_bundle,
)


def _bundle(present: np.ndarray) -> dict[str, float]:
    passes = len(present)
    category_scores = np.tile(np.array([[0.8, 0.2]]), (passes, 1)).astype(float)
    scores = np.full(passes, 0.8, dtype=float)
    boxes = np.tile(np.array([[0.5, 0.5, 0.2, 0.2]]), (passes, 1)).astype(float)
    embeddings = np.tile(np.array([[1.0, 0.0]]), (passes, 1)).astype(float)
    category_scores[~present] = np.nan
    scores[~present] = np.nan
    boxes[~present] = np.nan
    embeddings[~present] = np.nan
    return detection_feature_bundle(
        reference_box_cxcywh=np.array([0.5, 0.5, 0.2, 0.2]),
        category_scores=category_scores,
        scores=scores,
        boxes_cxcywh=boxes,
        embeddings=embeddings,
        present=present,
        base_category=0,
        deterministic_reference_variance=0.1,
        deterministic_reference_step=0.2,
        deterministic_hidden_step=0.3,
        bbox_area_fraction=0.04,
    )


def test_reference_iou_feature_has_exact_spatial_interpretation() -> None:
    present = np.array([True, True])
    boxes = np.array(
        [
            [0.5, 0.5, 0.2, 0.2],
            [0.6, 0.5, 0.2, 0.2],
        ]
    )
    result = detection_feature_bundle(
        reference_box_cxcywh=np.array([0.5, 0.5, 0.2, 0.2]),
        category_scores=np.array([[0.9, 0.1], [0.9, 0.1]]),
        scores=np.array([0.9, 0.9]),
        boxes_cxcywh=boxes,
        embeddings=np.array([[1.0, 0.0], [1.0, 0.0]]),
        present=present,
        base_category=0,
        deterministic_reference_variance=0.0,
        deterministic_reference_step=0.0,
        deterministic_hidden_step=0.0,
        bbox_area_fraction=0.04,
    )
    # The shifted box overlaps by 0.1 * 0.2 and has union 0.06, so IoU=1/3.
    assert np.isclose(result["spatial_reference_iou_mean"], (1.0 + 1 / 3) / 2)
    assert np.isclose(
        result["spatial_reference_iou_std"], np.std([1.0, 1 / 3], ddof=1)
    )
    assert result["spatial_pairwise_iou_loss"] == 1 - 1 / 3


def test_single_present_pass_is_finite_and_has_zero_variation() -> None:
    result = _bundle(np.array([True]))
    assert result["spatial_reference_iou_mean"] == 1.0
    assert result["spatial_reference_iou_std"] == 0.0
    assert result["spatial_pairwise_iou_loss"] == 0.0
    assert result["spatial_box_variance"] == 0.0
    assert result["spatial_absence_rate"] == 0.0
    assert all(not np.isinf(value) for value in result.values())


def test_total_stochastic_absence_remains_missing_and_explicit() -> None:
    result = _bundle(np.array([False, False]))
    assert result["spatial_absence_rate"] == 1.0
    assert np.isnan(result["spatial_reference_iou_mean"])
    assert np.isnan(result["semantic_mutual_information"])
    assert np.isnan(result["mc_score_mean"])
    assert result["deterministic_reference_step"] == 0.2
    assert all(not np.isinf(value) for value in result.values())


def test_localization_target_is_class_agnostic_but_records_class_agreement() -> None:
    result = class_agnostic_localization_targets(
        prediction_boxes=np.array([[0.0, 0.0, 10.0, 10.0]]),
        prediction_categories=np.array([2]),
        ground_truth_boxes=np.array([[0.0, 0.0, 10.0, 10.0]]),
        ground_truth_categories=np.array([7]),
        thresholds=(0.5, 0.75),
    )
    assert result["localization_iou"].tolist() == [1.0]
    assert result["localization_class_agreement"].tolist() == [False]
    assert result["is_well_localized_050"].tolist() == [1]
    assert result["is_well_localized_075"].tolist() == [1]


def test_localization_targets_support_no_detections_and_no_ground_truth() -> None:
    empty = class_agnostic_localization_targets(
        np.empty((0, 4)), np.array([], dtype=int), np.empty((0, 4)), np.array([], dtype=int)
    )
    assert all(len(value) == 0 for value in empty.values())
    no_ground_truth = class_agnostic_localization_targets(
        np.array([[0.0, 0.0, 1.0, 1.0]]),
        np.array([1]),
        np.empty((0, 4)),
        np.array([], dtype=int),
    )
    assert no_ground_truth["localization_iou"].tolist() == [0.0]
    assert no_ground_truth["localization_ground_truth_index"].tolist() == [-1]

