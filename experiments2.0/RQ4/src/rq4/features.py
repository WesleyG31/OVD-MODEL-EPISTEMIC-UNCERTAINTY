from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from adas_ovd.matching import box_iou_matrix
from adas_ovd.mc_features import mc_uncertainty_features


def calibration_targets(
    prediction_boxes: np.ndarray,
    prediction_categories: np.ndarray,
    ground_truth_boxes: np.ndarray,
    ground_truth_categories: np.ndarray,
    *,
    localization_iou_threshold: float,
    class_assignment_iou_threshold: float,
) -> dict[str, np.ndarray]:
    """Create label-only class/localization targets for fixed detections."""
    predictions = np.asarray(prediction_boxes, dtype=np.float64).reshape(-1, 4)
    prediction_categories = np.asarray(prediction_categories, dtype=np.int64)
    ground_truth = np.asarray(ground_truth_boxes, dtype=np.float64).reshape(-1, 4)
    ground_truth_categories = np.asarray(ground_truth_categories, dtype=np.int64)
    if len(predictions) != len(prediction_categories):
        raise ValueError("Prediction boxes/categories have different lengths")
    if len(ground_truth) != len(ground_truth_categories):
        raise ValueError("Ground-truth boxes/categories have different lengths")
    for value in (localization_iou_threshold, class_assignment_iou_threshold):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError("Target IoU thresholds must lie in [0, 1]")

    if len(predictions) == 0:
        best_iou = np.empty(0, dtype=np.float64)
        best_index = np.empty(0, dtype=np.int64)
        class_agreement = np.empty(0, dtype=bool)
    elif len(ground_truth) == 0:
        best_iou = np.zeros(len(predictions), dtype=np.float64)
        best_index = np.full(len(predictions), -1, dtype=np.int64)
        class_agreement = np.zeros(len(predictions), dtype=bool)
    else:
        ious = box_iou_matrix(predictions, ground_truth)
        best_index = ious.argmax(axis=1).astype(np.int64)
        best_iou = ious[np.arange(len(predictions)), best_index]
        class_agreement = (
            prediction_categories == ground_truth_categories[best_index]
        )
    return {
        "localization_iou": best_iou,
        "localization_ground_truth_index": best_index,
        "localization_class_agreement": class_agreement,
        "is_class_correct": (
            class_agreement
            & (best_iou >= float(class_assignment_iou_threshold))
        ).astype(np.int64),
        "is_well_localized": (
            best_iou >= float(localization_iou_threshold)
        ).astype(np.int64),
    }


def domain_descriptors(
    attributes: dict[str, Any],
    reference: dict[str, str],
    undefined_values: Iterable[str],
) -> dict[str, Any]:
    """Return frozen evaluation-only covariate-shift descriptors."""
    undefined = {str(value).strip().lower() for value in undefined_values}
    result: dict[str, Any] = {}
    shift_count = 0
    unknown_count = 0
    shifted_axes: list[str] = []
    for axis, reference_value in reference.items():
        value = str(attributes.get(axis, "unknown"))
        is_unknown = value.strip().lower() in undefined
        shifted = value != str(reference_value)
        result[f"shift_{axis}"] = bool(shifted)
        result[f"unknown_{axis}"] = bool(is_unknown)
        shift_count += int(shifted)
        unknown_count += int(is_unknown)
        if shifted:
            shifted_axes.append(axis)
    result.update(
        {
            "shift_axis_count": int(shift_count),
            "unknown_axis_count": int(unknown_count),
            "is_domain_shift": bool(shift_count > 0),
            "domain_stratum": (
                "reference" if shift_count == 0 else "+".join(shifted_axes)
            ),
        }
    )
    return result


def _cxcywh_to_xyxy_unit(boxes: np.ndarray) -> np.ndarray:
    values = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return np.column_stack(
        (
            values[:, 0] - values[:, 2] / 2.0,
            values[:, 1] - values[:, 3] / 2.0,
            values[:, 0] + values[:, 2] / 2.0,
            values[:, 1] + values[:, 3] / 2.0,
        )
    )


def reference_iou_statistics(
    reference_box_cxcywh: np.ndarray,
    mc_boxes_cxcywh: np.ndarray,
    present: np.ndarray,
) -> tuple[float, float]:
    mask = np.asarray(present, dtype=bool)
    boxes = np.asarray(mc_boxes_cxcywh, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or len(boxes) != len(mask):
        raise ValueError("MC boxes must have shape [passes, 4]")
    finite = mask & np.isfinite(boxes).all(axis=1)
    if not finite.any():
        return float("nan"), float("nan")
    reference = _cxcywh_to_xyxy_unit(
        np.asarray(reference_box_cxcywh, dtype=np.float64).reshape(1, 4)
    )
    candidates = _cxcywh_to_xyxy_unit(boxes[finite])
    values = box_iou_matrix(reference, candidates)[0]
    return (
        float(values.mean()),
        float(values.std(ddof=1)) if len(values) >= 2 else 0.0,
    )


def detection_feature_bundle(
    *,
    reference_box_cxcywh: np.ndarray,
    category_scores: np.ndarray,
    scores: np.ndarray,
    boxes_cxcywh: np.ndarray,
    embeddings: np.ndarray,
    present: np.ndarray,
    base_category: int,
    deterministic_reference_variance: float,
    deterministic_reference_step: float,
    deterministic_hidden_step: float,
    bbox_area_fraction: float,
) -> dict[str, float]:
    """Derive the frozen RQ4 features from shared schema-v1 arrays."""
    canonical = mc_uncertainty_features(
        category_scores=category_scores,
        scores=scores,
        boxes_cxcywh=boxes_cxcywh,
        embeddings=embeddings,
        present=present,
        base_category=base_category,
    )
    reference_mean, reference_std = reference_iou_statistics(
        reference_box_cxcywh, boxes_cxcywh, present
    )
    mask = np.asarray(present, dtype=bool)
    score_values = np.asarray(scores, dtype=np.float64)
    finite_scores = mask & np.isfinite(score_values)
    mc_score_mean = (
        float(score_values[finite_scores].mean())
        if finite_scores.any()
        else float("nan")
    )
    return {
        "spatial_reference_iou_mean": reference_mean,
        "spatial_reference_iou_std": reference_std,
        "spatial_pairwise_iou_loss": canonical["pairwise_iou_loss"],
        "spatial_box_variance": canonical["box_variance"],
        "spatial_absence_rate": canonical["absence_rate"],
        "deterministic_reference_variance": float(
            deterministic_reference_variance
        ),
        "deterministic_reference_step": float(deterministic_reference_step),
        "bbox_area_fraction": float(bbox_area_fraction),
        "semantic_mutual_information": canonical["mutual_information"],
        "semantic_predictive_entropy": canonical["predictive_entropy"],
        "semantic_class_disagreement": canonical["class_disagreement"],
        "semantic_score_variance": canonical["score_variance"],
        "geometric_pairwise_iou_loss": canonical["pairwise_iou_loss"],
        "geometric_box_variance": canonical["box_variance"],
        "mc_absence_rate": canonical["absence_rate"],
        "representation_embedding_variance": canonical["embedding_variance"],
        "representation_embedding_cosine_instability": canonical[
            "embedding_cosine_instability"
        ],
        "deterministic_hidden_step": float(deterministic_hidden_step),
        "mc_score_mean": mc_score_mean,
    }


def finite_feature_audit(
    values: np.ndarray, *, allow_all_missing: bool = False
) -> dict[str, int]:
    array = np.asarray(values, dtype=np.float64)
    result = {
        "values": int(array.size),
        "finite": int(np.isfinite(array).sum()),
        "nan": int(np.isnan(array).sum()),
        "positive_infinity": int(np.isposinf(array).sum()),
        "negative_infinity": int(np.isneginf(array).sum()),
    }
    if result["positive_infinity"] or result["negative_infinity"]:
        raise ValueError("RQ4 feature contains an infinite value")
    if not allow_all_missing and result["finite"] == 0:
        raise ValueError("RQ4 feature contains no finite values")
    return result

