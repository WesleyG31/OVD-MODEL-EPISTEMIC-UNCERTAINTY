from __future__ import annotations

import numpy as np

from adas_ovd.mc_features import (
    geometric_mc_features,
    representation_mc_features,
    semantic_mc_features,
)


EPSILON = 1e-12


def _nanmean(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    if not np.isfinite(finite).any():
        return float("nan")
    return float(np.nanmean(finite))


def semantic_features(
    category_scores: np.ndarray,
    scores: np.ndarray,
    present: np.ndarray,
    base_category: int,
) -> dict[str, float]:
    values = semantic_mc_features(
        category_scores, scores, present, base_category
    )
    return {
        "semantic_mutual_information": values["mutual_information"],
        "semantic_predictive_entropy": values["predictive_entropy"],
        "class_disagreement": values["class_disagreement"],
        "score_variance": values["score_variance"],
    }


def geometric_features(
    boxes_cxcywh: np.ndarray,
    present: np.ndarray,
    reference_variance: np.ndarray,
    reference_step: np.ndarray,
) -> dict[str, float]:
    values = geometric_mc_features(boxes_cxcywh, present)
    return {
        "box_variance": values["box_variance"],
        "box_mean_pairwise_iou_loss": values["pairwise_iou_loss"],
        "decoder_reference_variance": _nanmean(reference_variance[present]),
        "decoder_reference_step": _nanmean(reference_step[present]),
    }


def representation_features(
    embeddings: np.ndarray,
    present: np.ndarray,
    hidden_step: np.ndarray,
) -> dict[str, float]:
    values = representation_mc_features(embeddings, present)
    return {
        "embedding_variance": values["embedding_variance"],
        "embedding_cosine_instability": values[
            "embedding_cosine_instability"
        ],
        "decoder_hidden_step": _nanmean(hidden_step[present]),
    }


def decoder_reference_features(reference_points: np.ndarray) -> tuple[float, float]:
    reference_points = np.asarray(reference_points, dtype=np.float64)
    if len(reference_points) == 0:
        return float("nan"), float("nan")
    variance = float(np.var(reference_points, axis=0).mean())
    if len(reference_points) == 1:
        return variance, 0.0
    step = float(np.linalg.norm(np.diff(reference_points, axis=0), axis=1).mean())
    return variance, step


def decoder_hidden_step(hidden_states: np.ndarray) -> float:
    hidden_states = np.asarray(hidden_states, dtype=np.float64)
    if len(hidden_states) <= 1:
        return 0.0
    normalized = hidden_states / np.clip(
        np.linalg.norm(hidden_states, axis=1, keepdims=True), EPSILON, None
    )
    similarities = (normalized[:-1] * normalized[1:]).sum(axis=1)
    return float((1.0 - similarities).mean())
