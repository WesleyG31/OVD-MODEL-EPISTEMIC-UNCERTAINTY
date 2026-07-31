import pandas as pd

from rq1.fusion import mc_sensitivity_features, split_validation_groups


def test_mc_sensitivity_uses_suffixed_internal_features() -> None:
    config = {
        "rq1": {
            "feature_groups": {
                "confidence": ["confidence_uncertainty"],
                "semantic": ["semantic_mutual_information"],
                "geometric": ["box_variance"],
                "representation": ["embedding_variance"],
                "presence": ["absence_rate"],
            },
            "fusion": {
                "methods": [
                    "confidence",
                    "semantic",
                    "geometric",
                    "representation",
                    "presence",
                    "semantic_geometric",
                    "all_internal",
                    "all_plus_confidence",
                ]
            },
            "extraction": {"mc_sensitivity_passes": [2, 5, 10]},
        }
    }

    methods = mc_sensitivity_features(config)

    assert list(methods) == [
        "mc_passes_02",
        "mc_passes_05",
        "mc_passes_10",
    ]
    assert methods["mc_passes_02"] == [
        "semantic_mutual_information_mc02",
        "box_variance_mc02",
        "embedding_variance_mc02",
        "absence_rate_mc02",
    ]


def test_validation_selection_and_calibration_are_group_disjoint() -> None:
    frame = pd.DataFrame(
        {
            "sequence_id": [
                group for group in ("a", "b", "c", "d") for _ in range(10)
            ],
            "is_error": [0, 1] * 20,
        }
    )
    selection, calibration = split_validation_groups(
        frame,
        group_column="sequence_id",
        selection_fraction=0.5,
        seed=17,
        minimum_calibration_samples=10,
    )
    assert set(selection["sequence_id"]).isdisjoint(
        set(calibration["sequence_id"])
    )
    assert selection["is_error"].nunique() == 2
    assert calibration["is_error"].nunique() == 2

    repeat_selection, repeat_calibration = split_validation_groups(
        frame,
        group_column="sequence_id",
        selection_fraction=0.5,
        seed=17,
        minimum_calibration_samples=10,
    )
    assert list(selection.index) == list(repeat_selection.index)
    assert list(calibration.index) == list(repeat_calibration.index)
