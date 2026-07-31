from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


def box_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    boxes_a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    boxes_b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)

    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    intersection_wh = np.clip(bottom_right - top_left, a_min=0.0, a_max=None)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]

    area_a = np.clip(boxes_a[:, 2] - boxes_a[:, 0], 0, None) * np.clip(
        boxes_a[:, 3] - boxes_a[:, 1], 0, None
    )
    area_b = np.clip(boxes_b[:, 2] - boxes_b[:, 0], 0, None) * np.clip(
        boxes_b[:, 3] - boxes_b[:, 1], 0, None
    )
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0,
    )


@dataclass(frozen=True)
class DetectionMatch:
    is_true_positive: np.ndarray
    matched_ground_truth: np.ndarray
    matched_iou: np.ndarray
    false_negatives: int


def match_predictions_to_ground_truth(
    prediction_boxes: np.ndarray,
    prediction_scores: np.ndarray,
    prediction_categories: np.ndarray,
    ground_truth_boxes: np.ndarray,
    ground_truth_categories: np.ndarray,
    iou_threshold: float = 0.5,
) -> DetectionMatch:
    prediction_boxes = np.asarray(prediction_boxes, dtype=np.float64).reshape(-1, 4)
    prediction_scores = np.asarray(prediction_scores, dtype=np.float64)
    prediction_categories = np.asarray(prediction_categories)
    ground_truth_boxes = np.asarray(ground_truth_boxes, dtype=np.float64).reshape(-1, 4)
    ground_truth_categories = np.asarray(ground_truth_categories)

    n_predictions = len(prediction_boxes)
    is_tp = np.zeros(n_predictions, dtype=bool)
    matched_gt = np.full(n_predictions, -1, dtype=np.int64)
    matched_iou = np.zeros(n_predictions, dtype=np.float64)
    available = np.ones(len(ground_truth_boxes), dtype=bool)
    ious = box_iou_matrix(prediction_boxes, ground_truth_boxes)

    for prediction_index in np.argsort(-prediction_scores, kind="stable"):
        compatible = np.flatnonzero(
            available
            & (ground_truth_categories == prediction_categories[prediction_index])
        )
        if len(compatible) == 0:
            continue
        local = compatible[np.argmax(ious[prediction_index, compatible])]
        best_iou = ious[prediction_index, local]
        if best_iou >= iou_threshold:
            is_tp[prediction_index] = True
            matched_gt[prediction_index] = int(local)
            matched_iou[prediction_index] = float(best_iou)
            available[local] = False

    return DetectionMatch(
        is_true_positive=is_tp,
        matched_ground_truth=matched_gt,
        matched_iou=matched_iou,
        false_negatives=int(available.sum()),
    )


def associate_detections(
    reference_boxes: np.ndarray,
    reference_categories: np.ndarray,
    candidate_boxes: np.ndarray,
    candidate_categories: np.ndarray,
    minimum_iou: float,
    class_penalty: float,
    unmatched_cost: float,
) -> np.ndarray:
    reference_boxes = np.asarray(reference_boxes, dtype=np.float64).reshape(-1, 4)
    candidate_boxes = np.asarray(candidate_boxes, dtype=np.float64).reshape(-1, 4)
    reference_categories = np.asarray(reference_categories)
    candidate_categories = np.asarray(candidate_categories)
    n_reference = len(reference_boxes)
    if n_reference == 0:
        return np.empty(0, dtype=np.int64)
    if len(candidate_boxes) == 0:
        return np.full(n_reference, -1, dtype=np.int64)

    ious = box_iou_matrix(reference_boxes, candidate_boxes)
    class_mismatch = (
        reference_categories[:, None] != candidate_categories[None, :]
    ).astype(np.float64)
    candidate_cost = 1.0 - ious + class_penalty * class_mismatch
    # Class disagreement is penalized but not forbidden. Keeping it observable is
    # necessary for measuring MC class instability without allowing geometrically
    # unrelated candidates to match.
    invalid = ious < minimum_iou
    candidate_cost[invalid] = unmatched_cost + 1.0

    dummy_cost = np.full((n_reference, n_reference), unmatched_cost)
    cost = np.concatenate([candidate_cost, dummy_cost], axis=1)
    rows, columns = linear_sum_assignment(cost)
    association = np.full(n_reference, -1, dtype=np.int64)
    for row, column in zip(rows, columns, strict=True):
        if column < len(candidate_boxes) and cost[row, column] < unmatched_cost:
            association[row] = int(column)
    return association
