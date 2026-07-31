import numpy as np

from adas_ovd.matching import (
    associate_detections,
    match_predictions_to_ground_truth,
)


def test_ground_truth_cannot_be_reused() -> None:
    predictions = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=float)
    result = match_predictions_to_ground_truth(
        prediction_boxes=predictions,
        prediction_scores=np.array([0.9, 0.8]),
        prediction_categories=np.array([1, 1]),
        ground_truth_boxes=np.array([[0, 0, 10, 10]], dtype=float),
        ground_truth_categories=np.array([1]),
        iou_threshold=0.5,
    )
    assert result.is_true_positive.tolist() == [True, False]
    assert result.false_negatives == 0


def test_association_is_one_to_one_and_respects_class() -> None:
    reference = np.array([[0, 0, 10, 10], [0, 0, 9, 9]], dtype=float)
    candidates = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=float)
    association = associate_detections(
        reference_boxes=reference,
        reference_categories=np.array([1, 1]),
        candidate_boxes=candidates,
        candidate_categories=np.array([1, 2]),
        minimum_iou=0.3,
        class_penalty=1.0,
        unmatched_cost=2.0,
    )
    assert association.tolist().count(0) == 1
    assert association.tolist().count(-1) == 1

