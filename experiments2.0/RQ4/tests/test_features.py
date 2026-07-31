import numpy as np

from rq4.features import (
    calibration_targets,
    detection_feature_bundle,
    domain_descriptors,
    reference_iou_statistics,
)


def test_calibration_targets_empty_predictions():
    result = calibration_targets(
        np.empty((0, 4)), np.empty(0), np.empty((0, 4)), np.empty(0),
        localization_iou_threshold=0.5, class_assignment_iou_threshold=0.1,
    )
    assert all(len(value) == 0 for value in result.values())


def test_calibration_targets_no_ground_truth_are_negative():
    result = calibration_targets(
        np.array([[0, 0, 1, 1]]), np.array([2]), np.empty((0, 4)), np.empty(0),
        localization_iou_threshold=0.5, class_assignment_iou_threshold=0.1,
    )
    assert result["localization_iou"].tolist() == [0.0]
    assert result["is_class_correct"].tolist() == [0]
    assert result["is_well_localized"].tolist() == [0]


def test_class_and_localization_targets_are_distinct():
    result = calibration_targets(
        np.array([[0, 0, 1, 1], [0, 0, 0.2, 0.2]]),
        np.array([1, 1]), np.array([[0, 0, 1, 1]]), np.array([1]),
        localization_iou_threshold=0.5, class_assignment_iou_threshold=0.1,
    )
    assert result["is_class_correct"].tolist() == [1, 0]
    assert result["is_well_localized"].tolist() == [1, 0]


def test_domain_descriptors_keep_unknown_as_shifted():
    result = domain_descriptors(
        {"timeofday": "undefined", "weather": "clear", "scene": "highway"},
        {"timeofday": "daytime", "weather": "clear", "scene": "city street"},
        ["undefined", "unknown", ""],
    )
    assert result["unknown_timeofday"] is True
    assert result["is_domain_shift"] is True
    assert result["shift_axis_count"] == 2
    assert result["domain_stratum"] == "timeofday+scene"


def test_reference_iou_handles_one_and_zero_present_passes():
    reference = np.array([0.5, 0.5, 0.2, 0.2])
    boxes = np.array([[0.5, 0.5, 0.2, 0.2], [np.nan] * 4])
    assert reference_iou_statistics(reference, boxes, np.array([True, False])) == (1.0, 0.0)
    mean, std = reference_iou_statistics(reference, boxes, np.array([False, False]))
    assert np.isnan(mean) and np.isnan(std)


def test_feature_bundle_preserves_complete_stochastic_absence():
    bundle = detection_feature_bundle(
        reference_box_cxcywh=np.array([0.5, 0.5, 0.2, 0.2]),
        category_scores=np.full((2, 3), np.nan), scores=np.full(2, np.nan),
        boxes_cxcywh=np.full((2, 4), np.nan), embeddings=np.full((2, 4), np.nan),
        present=np.array([False, False]), base_category=0,
        deterministic_reference_variance=0.1, deterministic_reference_step=0.2,
        deterministic_hidden_step=0.3, bbox_area_fraction=0.04,
    )
    assert bundle["mc_absence_rate"] == 1.0
    assert bundle["spatial_absence_rate"] == 1.0
    assert np.isnan(bundle["semantic_mutual_information"])
    assert bundle["bbox_area_fraction"] == 0.04

