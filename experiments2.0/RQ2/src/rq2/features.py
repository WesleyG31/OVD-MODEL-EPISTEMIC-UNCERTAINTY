from __future__ import annotations

import numpy as np

from adas_ovd.mc_features import mc_uncertainty_features


EPSILON = 1e-12


def decoder_trajectory_features(
    hidden_states: np.ndarray, reference_points: np.ndarray
) -> dict[str, float]:
    """Single-pass decoder dynamics, positively oriented as instability."""
    hidden = np.asarray(hidden_states, dtype=np.float64)
    references = np.asarray(reference_points, dtype=np.float64)
    if hidden.ndim != 2 or references.ndim != 2:
        raise ValueError("Decoder trajectories must be [layers, features]")
    reference_variance = float(np.var(references, axis=0).mean())
    reference_step = (
        float(np.linalg.norm(np.diff(references, axis=0), axis=1).mean())
        if len(references) > 1
        else 0.0
    )
    if len(hidden) > 1:
        normalized = hidden / np.clip(
            np.linalg.norm(hidden, axis=1, keepdims=True), EPSILON, None
        )
        hidden_step = float(
            (1.0 - (normalized[:-1] * normalized[1:]).sum(axis=1)).mean()
        )
    else:
        hidden_step = 0.0
    return {
        "deterministic_reference_variance": reference_variance,
        "deterministic_reference_step": reference_step,
        "deterministic_hidden_step": hidden_step,
    }


def stochastic_features(
    *,
    category_scores: np.ndarray,
    scores: np.ndarray,
    boxes_cxcywh: np.ndarray,
    embeddings: np.ndarray,
    present: np.ndarray,
    base_category: int,
) -> dict[str, float]:
    """Aggregate matched MC passes without imputing absent detections."""
    values = mc_uncertainty_features(
        category_scores=category_scores,
        scores=scores,
        boxes_cxcywh=boxes_cxcywh,
        embeddings=embeddings,
        present=present,
        base_category=base_category,
    )
    return {f"stochastic_{name}": value for name, value in values.items()}


def finite_feature_audit(
    values: np.ndarray, *, allow_all_missing: bool = False
) -> dict[str, int]:
    array = np.asarray(values, dtype=np.float64)
    counts = {
        "values": int(array.size),
        "finite": int(np.isfinite(array).sum()),
        "nan": int(np.isnan(array).sum()),
        "positive_infinity": int(np.isposinf(array).sum()),
        "negative_infinity": int(np.isneginf(array).sum()),
    }
    if counts["positive_infinity"] or counts["negative_infinity"]:
        raise ValueError("Uncertainty feature contains an infinite value")
    if not allow_all_missing and counts["finite"] == 0:
        raise ValueError("Uncertainty feature contains no finite values")
    return counts
