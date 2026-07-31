from __future__ import annotations

import numpy as np

from adas_ovd.config import load_config
from rq2.extraction import _empty_frame
from rq2.features import (
    decoder_trajectory_features,
    finite_feature_audit,
    stochastic_features,
)


def test_decoder_trajectory_features_are_finite_and_oriented() -> None:
    hidden = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    references = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
    result = decoder_trajectory_features(hidden, references)
    assert set(result) == {
        "deterministic_reference_variance",
        "deterministic_reference_step",
        "deterministic_hidden_step",
    }
    assert all(np.isfinite(value) and value >= 0 for value in result.values())
    assert result["deterministic_reference_step"] == 0.5


def test_stochastic_features_keep_absence_explicit() -> None:
    result = stochastic_features(
        category_scores=np.array([[0.8, 0.2], [np.nan, np.nan], [0.6, 0.4]]),
        scores=np.array([0.8, np.nan, 0.6]),
        boxes_cxcywh=np.array(
            [[0.5, 0.5, 0.2, 0.2], [np.nan] * 4, [0.52, 0.5, 0.2, 0.2]]
        ),
        embeddings=np.array([[1.0, 0.0], [np.nan, np.nan], [0.9, 0.1]]),
        present=np.array([True, False, True]),
        base_category=0,
    )
    assert np.isclose(result["stochastic_absence_rate"], 1 / 3)
    assert result["stochastic_mutual_information"] >= 0
    assert result["stochastic_box_variance"] > 0
    assert all(not np.isinf(value) for value in result.values())


def test_finite_feature_audit_rejects_infinity() -> None:
    try:
        finite_feature_audit(np.array([0.0, np.inf]))
    except ValueError as error:
        assert "infinite" in str(error)
    else:
        raise AssertionError("Infinity was not rejected")


def test_frozen_extraction_schema_has_unique_columns() -> None:
    from pathlib import Path

    project = Path(__file__).resolve().parents[2]
    config = load_config(project / "RQ2" / "configs" / "rq2_mini.yaml")
    columns = list(_empty_frame(config).columns)
    assert len(columns) == len(set(columns))
    assert columns.count("confidence_uncertainty") == 1
