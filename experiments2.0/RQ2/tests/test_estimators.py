from __future__ import annotations

import numpy as np
import pandas as pd

from rq2.estimators import (
    AverageScorer,
    EmpiricalCDFScorer,
    _split_validation_groups,
)
from rq2.evaluation import _holm_adjust


def test_empirical_cdf_uses_train_distribution_and_imputes_nan() -> None:
    train = pd.DataFrame({"a": [0.0, 1.0, 2.0], "b": [0.0, 2.0, 4.0]})
    scorer = EmpiricalCDFScorer.fit(train, ["a", "b"])
    test = pd.DataFrame({"a": [0.0, np.nan, 3.0], "b": [0.0, 2.0, 5.0]})
    scores = scorer.rank_score(test)
    assert np.isfinite(scores).all()
    assert scores[0] < scores[1] < scores[2]


def test_equal_fusion_is_arithmetic_mean() -> None:
    train = pd.DataFrame({"a": [0.0, 1.0], "b": [0.0, 2.0]})
    first = EmpiricalCDFScorer.fit(train, ["a"])
    second = EmpiricalCDFScorer.fit(train, ["b"])
    frame = pd.DataFrame({"a": [0.0, 1.0], "b": [2.0, 0.0]})
    fused = AverageScorer([first, second]).rank_score(frame)
    expected = (first.rank_score(frame) + second.rank_score(frame)) / 2.0
    np.testing.assert_allclose(fused, expected)


def test_holm_adjustment_is_monotone_in_sorted_order() -> None:
    adjusted = _holm_adjust({"a": 0.01, "b": 0.03, "c": 0.20})
    assert adjusted["a"] == 0.03
    assert adjusted["b"] == 0.06
    assert adjusted["c"] == 0.20


def test_selection_and_calibration_are_group_disjoint() -> None:
    frame = pd.DataFrame(
        {
            "sequence_id": np.repeat(["a", "b", "c", "d"], 4),
            "is_error": np.tile([0, 1, 0, 1], 4),
        }
    )
    config = {
        "project": {"seed": 41},
        "rq2": {
            "estimators": {
                "validation_calibration_fraction": 0.5,
                "minimum_partition_samples": 4,
            }
        },
    }
    selection, calibration = _split_validation_groups(frame, config)
    assert set(selection["sequence_id"]).isdisjoint(calibration["sequence_id"])
    assert selection["is_error"].nunique() == 2
    assert calibration["is_error"].nunique() == 2
