import numpy as np
import pandas as pd
import pytest

from adas_ovd.config import load_config
from rq4.calibration import (
    ComponentModel,
    MethodScorer,
    _fit_component,
    component_feature_sets,
    method_feature_sets,
    sensitivity_feature_sets,
    source_domain_mask,
    split_validation_groups,
    validate_no_target_leakage,
)


def _config():
    return load_config("RQ4/configs/rq4_mini.yaml")


def test_feature_contract_has_capacity_matched_union():
    config = _config()
    components = component_feature_sets(config)
    methods = method_feature_sets(config)
    assert methods["multilevel"] == methods["flat_joint"]
    assert len(components["flat"]) == len(set(components["flat"]))
    assert components["class"] == ["score", "category_index"]


def test_full_mc_sensitivity_reuses_primary_features():
    config = _config()
    sensitivity = sensitivity_feature_sets(config)
    assert sensitivity["multilevel_mc10"] == component_feature_sets(config)["flat"]
    assert any(name.endswith("_mc02") for name in sensitivity["multilevel_mc02"])
    assert not any(name.endswith("_mc10") for name in sensitivity["multilevel_mc10"])


def test_domain_or_target_feature_is_rejected():
    with pytest.raises(RuntimeError):
        validate_no_target_leakage({"bad": ["score", "shift_weather"]})
    with pytest.raises(RuntimeError):
        validate_no_target_leakage({"bad": ["matched_iou"]})


def test_group_split_is_disjoint():
    frame = pd.DataFrame(
        {
            "sequence_id": np.repeat(["a", "b", "c", "d"], 10),
            "is_class_correct": np.tile([0, 1], 20),
            "is_well_localized": np.tile([0, 1], 20),
            "is_detection_correct": np.tile([1, 0], 20),
            "is_error": np.tile([0, 1], 20),
        }
    )
    selection, calibration = split_validation_groups(
        frame, group_column="sequence_id", selection_fraction=0.5, seed=7,
        minimum_selection_samples=10, minimum_calibration_samples=10,
    )
    assert set(selection.sequence_id).isdisjoint(set(calibration.sequence_id))


def test_selected_component_and_product_scorer_are_finite():
    config = _config()
    train = pd.DataFrame({"score": np.linspace(0.05, 0.95, 40)})
    train["target"] = (train.score > 0.5).astype(int)
    selection = train.sample(frac=1.0, random_state=1).reset_index(drop=True)
    component = _fit_component(
        config, train, selection, name="class", features=["score"], target="target"
    )
    scorer = MethodScorer("class_only", {"class": component}, ("class",))
    values = scorer.rank_score(selection)
    assert np.isfinite(values).all()
    assert component.coefficient_count > 0


def test_source_domain_mask_is_exact_and_rejects_unknowns():
    config = _config()
    frame = pd.DataFrame(
        {
            "timeofday": ["daytime", "night", "daytime"],
            "weather": ["clear", "clear", "clear"],
            "scene": ["city street", "city street", "city street"],
            "unknown_axis_count": [0, 0, 1],
            "is_domain_shift": [False, True, True],
        }
    )
    assert source_domain_mask(frame, config).tolist() == [True, False, False]


def test_category_conditioning_has_global_fallback_for_rare_and_unseen_classes():
    config = _config()
    score = np.linspace(0.05, 0.95, 40)
    train = pd.DataFrame(
        {
            "score": score,
            "category_index": [0] * 36 + [1] * 4,
            "target": (np.arange(40) % 2).astype(int),
        }
    )
    selection = train.sample(frac=1.0, random_state=3).reset_index(drop=True)
    component = _fit_component(
        config,
        train,
        selection,
        name="class",
        features=["score", "category_index"],
        target="target",
    )
    assert component.known_categories == (0,)
    fallback = pd.DataFrame(
        {"score": [0.5, 0.5], "category_index": [1, 999]}
    )
    probabilities = component.probability(fallback)
    np.testing.assert_allclose(probabilities[0], probabilities[1], rtol=0.0, atol=0.0)
