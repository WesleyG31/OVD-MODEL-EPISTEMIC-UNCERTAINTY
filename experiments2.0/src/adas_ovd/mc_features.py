from __future__ import annotations

import numpy as np

from .matching import box_iou_matrix


EPSILON = 1e-12


def _simplex(category_scores: np.ndarray) -> np.ndarray:
    values = np.asarray(category_scores, dtype=np.float64)
    denominator = values.sum(axis=-1, keepdims=True)
    return np.divide(
        values,
        denominator,
        out=np.full_like(values, 1.0 / values.shape[-1]),
        where=denominator > EPSILON,
    )


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, EPSILON, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=-1)


def semantic_mc_features(
    category_scores: np.ndarray,
    scores: np.ndarray,
    present: np.ndarray,
    base_category: int,
) -> dict[str, float]:
    mask = np.asarray(present, dtype=bool)
    if not mask.any():
        return {
            "mutual_information": float("nan"),
            "predictive_entropy": float("nan"),
            "class_disagreement": float("nan"),
            "score_variance": float("nan"),
        }
    valid_category_scores = np.asarray(category_scores)[mask]
    probabilities = _simplex(valid_category_scores)
    mean_probability = probabilities.mean(axis=0)
    entropy_scale = max(float(np.log(probabilities.shape[1])), EPSILON)
    predictive_entropy = float(_entropy(mean_probability) / entropy_scale)
    expected_entropy = float(_entropy(probabilities).mean() / entropy_scale)
    valid_scores = np.asarray(scores, dtype=np.float64)[mask]
    return {
        "mutual_information": max(
            predictive_entropy - expected_entropy, 0.0
        ),
        "predictive_entropy": predictive_entropy,
        "class_disagreement": float(
            (valid_category_scores.argmax(axis=1) != int(base_category)).mean()
        ),
        "score_variance": (
            float(np.var(valid_scores, ddof=1)) if len(valid_scores) >= 2 else 0.0
        ),
    }


def geometric_mc_features(
    boxes_cxcywh: np.ndarray, present: np.ndarray
) -> dict[str, float]:
    mask = np.asarray(present, dtype=bool)
    boxes = np.asarray(boxes_cxcywh, dtype=np.float64)[mask]
    if len(boxes) == 0:
        return {
            "box_variance": float("nan"),
            "pairwise_iou_loss": float("nan"),
        }
    if len(boxes) == 1:
        return {"box_variance": 0.0, "pairwise_iou_loss": 0.0}
    box_variance = float(np.var(boxes, axis=0, ddof=1).mean())
    xyxy = np.column_stack(
        (
            boxes[:, 0] - boxes[:, 2] / 2.0,
            boxes[:, 1] - boxes[:, 3] / 2.0,
            boxes[:, 0] + boxes[:, 2] / 2.0,
            boxes[:, 1] + boxes[:, 3] / 2.0,
        )
    )
    pairwise = box_iou_matrix(xyxy, xyxy)
    upper = pairwise[np.triu_indices(len(pairwise), k=1)]
    return {
        "box_variance": box_variance,
        "pairwise_iou_loss": float(1.0 - upper.mean()),
    }


def representation_mc_features(
    embeddings: np.ndarray, present: np.ndarray
) -> dict[str, float]:
    mask = np.asarray(present, dtype=bool)
    valid = np.asarray(embeddings, dtype=np.float64)[mask]
    if len(valid) == 0:
        return {
            "embedding_variance": float("nan"),
            "embedding_cosine_instability": float("nan"),
        }
    if len(valid) == 1:
        return {
            "embedding_variance": 0.0,
            "embedding_cosine_instability": 0.0,
        }
    embedding_variance = float(np.var(valid, axis=0, ddof=1).mean())
    normalized = valid / np.clip(
        np.linalg.norm(valid, axis=1, keepdims=True), EPSILON, None
    )
    similarities = normalized @ normalized.T
    upper = similarities[np.triu_indices(len(similarities), k=1)]
    return {
        "embedding_variance": embedding_variance,
        "embedding_cosine_instability": float(1.0 - upper.mean()),
    }


def mc_uncertainty_features(
    *,
    category_scores: np.ndarray,
    scores: np.ndarray,
    boxes_cxcywh: np.ndarray,
    embeddings: np.ndarray,
    present: np.ndarray,
    base_category: int,
) -> dict[str, float]:
    """Canonical float64 MC reductions shared by all compatible RQs."""
    mask = np.asarray(present, dtype=bool)
    result = {"absence_rate": float(1.0 - mask.mean())}
    result.update(
        semantic_mc_features(category_scores, scores, mask, base_category)
    )
    result.update(geometric_mc_features(boxes_cxcywh, mask))
    result.update(representation_mc_features(embeddings, mask))
    return result
