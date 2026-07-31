import numpy as np

from rq1.features import semantic_features


def test_semantic_mutual_information_detects_between_pass_disagreement() -> None:
    scores = np.array(
        [
            [0.99, 0.01],
            [0.01, 0.99],
            [0.99, 0.01],
            [0.01, 0.99],
        ]
    )
    features = semantic_features(
        category_scores=scores,
        scores=scores.max(axis=1),
        present=np.ones(4, dtype=bool),
        base_category=0,
    )
    assert features["semantic_mutual_information"] > 0.8
    assert features["class_disagreement"] == 0.5

