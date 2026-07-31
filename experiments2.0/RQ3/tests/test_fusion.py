from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from adas_ovd.config import load_config
from rq3.fusion import (
    EqualFusionScorer,
    ProductFusionScorer,
    QualityModel,
    _fit_selected_logistic,
    method_feature_sets,
    split_validation_groups,
    validate_no_target_leakage,
)


PROJECT = Path(__file__).resolve().parents[2]


class _FixedQuality:
    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        quality = frame.iloc[:, 0].to_numpy(dtype=float)
        return np.column_stack([1.0 - quality, quality])


def test_product_and_equal_fusion_formulas() -> None:
    frame = pd.DataFrame({"score": [0.8, 0.4], "quality": [0.5, 0.25]})
    quality = QualityModel(["quality"], _FixedQuality())  # type: ignore[arg-type]
    np.testing.assert_allclose(
        ProductFusionScorer(quality).rank_score(frame), [0.6, 0.9]
    )
    np.testing.assert_allclose(
        EqualFusionScorer(quality).rank_score(frame), [0.35, 0.675]
    )


def test_validation_split_is_reproducible_and_group_disjoint() -> None:
    frame = pd.DataFrame(
        {
            "sequence_id": [group for group in "abcd" for _ in range(10)],
            "is_error": [0, 1] * 20,
            "is_well_localized": [1, 0] * 20,
        }
    )
    first = split_validation_groups(
        frame,
        group_column="sequence_id",
        selection_fraction=0.5,
        seed=23,
        minimum_selection_samples=10,
        minimum_calibration_samples=10,
    )
    second = split_validation_groups(
        frame,
        group_column="sequence_id",
        selection_fraction=0.5,
        seed=23,
        minimum_selection_samples=10,
        minimum_calibration_samples=10,
    )
    assert set(first[0]["sequence_id"]).isdisjoint(set(first[1]["sequence_id"]))
    assert list(first[0].index) == list(second[0].index)
    assert list(first[1].index) == list(second[1].index)


def test_target_leakage_is_rejected_and_frozen_methods_are_clean() -> None:
    config = load_config(PROJECT / "RQ3" / "configs" / "rq3_mini.yaml")
    definitions = method_feature_sets(config)
    assert len(definitions["product_fusion"]) == len(
        definitions["capacity_control_product"]
    )
    validate_no_target_leakage(definitions)
    try:
        validate_no_target_leakage(
            {"bad": ["score", "localization_iou", "is_well_localized_090"]}
        )
    except RuntimeError as error:
        assert "localization_iou" in str(error)
        assert "is_well_localized_090" in str(error)
    else:
        raise AssertionError("Oracle localization IoU was accepted as a feature")


def test_logistic_quality_model_selects_and_produces_finite_probabilities() -> None:
    config = load_config(PROJECT / "RQ3" / "configs" / "rq3_mini.yaml")
    train = pd.DataFrame(
        {"x": np.linspace(-2, 2, 80), "is_well_localized": [0] * 40 + [1] * 40}
    )
    selection = pd.DataFrame(
        {"x": np.linspace(-1.5, 1.5, 40), "is_well_localized": [0] * 20 + [1] * 20}
    )
    estimator, selected_c, auroc = _fit_selected_logistic(
        config, train, selection, ["x"], "is_well_localized"
    )
    probability = estimator.predict_proba(selection[["x"]])[:, 1]
    assert selected_c in config["rq3"]["estimators"]["regularization_grid"]
    assert auroc > 0.9
    assert np.isfinite(probability).all()
