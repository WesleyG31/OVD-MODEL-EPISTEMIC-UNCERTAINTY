from __future__ import annotations

from typing import Any

import numpy as np

from adas_ovd.matching import box_iou_matrix
from adas_ovd.mc_features import mc_uncertainty_features


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
    return float(values.mean()), (
        float(values.std(ddof=1)) if len(values) >= 2 else 0.0
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
    """Derive RQ5 features, delegating canonical MC reductions to schema v1."""
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
    return {
        "semantic_mutual_information": canonical["mutual_information"],
        "semantic_predictive_entropy": canonical["predictive_entropy"],
        "semantic_class_disagreement": canonical["class_disagreement"],
        "semantic_score_variance": canonical["score_variance"],
        "spatial_reference_iou_mean": reference_mean,
        "spatial_reference_iou_std": reference_std,
        "geometric_pairwise_iou_loss": canonical["pairwise_iou_loss"],
        "geometric_box_variance": canonical["box_variance"],
        "mc_absence_rate": canonical["absence_rate"],
        "representation_embedding_variance": canonical["embedding_variance"],
        "representation_embedding_cosine_instability": canonical[
            "embedding_cosine_instability"
        ],
        "deterministic_reference_variance": float(
            deterministic_reference_variance
        ),
        "deterministic_reference_step": float(deterministic_reference_step),
        "deterministic_hidden_step": float(deterministic_hidden_step),
        "bbox_area_fraction": float(bbox_area_fraction),
        "mc_score_mean": (
            float(score_values[finite_scores].mean())
            if finite_scores.any()
            else float("nan")
        ),
    }


def criticality_descriptors(
    box_xyxy: np.ndarray,
    *,
    width: int,
    height: int,
    category_name: str,
    specification: dict[str, Any],
) -> dict[str, float | str]:
    """Frozen label-free ADAS criticality from predicted class and geometry."""
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    box = np.asarray(box_xyxy, dtype=np.float64).reshape(4)
    center_x = float((box[0] + box[2]) / 2.0 / width)
    bottom = float(box[3] / height)
    centrality = float(np.clip(1.0 - abs(center_x - 0.5) / 0.5, 0.0, 1.0))
    bottomness = float(np.clip((bottom - 0.5) / 0.5, 0.0, 1.0))
    severity_map = specification["class_severity"]
    if category_name not in severity_map:
        raise KeyError(f"RQ5 criticality severity is missing: {category_name}")
    severity = float(severity_map[category_name])
    geometry = (
        1.0
        + float(specification["bottomness_coefficient"]) * bottomness
        + float(specification["centrality_coefficient"]) * centrality
    )
    weight = severity * geometry
    lower, upper = map(float, specification["tier_boundaries"])
    tier = "low" if weight < lower else "medium" if weight < upper else "high"
    return {
        "criticality_class_severity": severity,
        "criticality_bottomness": bottomness,
        "criticality_centrality": centrality,
        "criticality_geometry_factor": geometry,
        "criticality_weight": float(weight),
        "criticality_tier": tier,
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
        raise ValueError("RQ5 feature contains an infinite value")
    if not allow_all_missing and result["finite"] == 0:
        raise ValueError("RQ5 feature contains no finite values")
    return result

