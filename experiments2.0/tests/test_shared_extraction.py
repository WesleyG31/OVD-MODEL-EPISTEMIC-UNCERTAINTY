from pathlib import Path

import numpy as np

from adas_ovd.shared_extraction import (
    ARRAY_SCHEMA,
    _shard_is_valid,
    _trajectory_statistics,
    _write_shard,
    load_shared_shard,
    validate_consumer_compatibility,
)
from adas_ovd.groundingdino_adapter import select_candidate_indices
from rq1.features import (
    decoder_hidden_step,
    decoder_reference_features,
    geometric_features,
    representation_features,
    semantic_features,
)
from rq2.features import decoder_trajectory_features, stochastic_features


def _arrays() -> dict[str, np.ndarray]:
    passes, references, classes, hidden = 2, 2, 3, 4
    return {
        "reference_boxes_cxcywh": np.zeros((references, 4), np.float32),
        "reference_boxes_xyxy": np.zeros((references, 4), np.float64),
        "reference_category_scores": np.zeros((references, classes), np.float32),
        "reference_scores": np.zeros(references, np.float32),
        "reference_category_indices": np.zeros(references, np.int64),
        "reference_query_indices": np.arange(references, dtype=np.int64),
        "deterministic_reference_variance": np.zeros(references, np.float64),
        "deterministic_reference_step": np.zeros(references, np.float64),
        "deterministic_hidden_step": np.zeros(references, np.float64),
        "present": np.ones((passes, references), bool),
        "mc_category_scores": np.zeros(
            (passes, references, classes), np.float32
        ),
        "mc_scores": np.zeros((passes, references), np.float32),
        "mc_boxes_cxcywh": np.zeros((passes, references, 4), np.float32),
        "mc_embeddings": np.zeros(
            (passes, references, hidden), np.float32
        ),
        "mc_reference_variance": np.zeros(
            (passes, references), np.float64
        ),
        "mc_reference_step": np.zeros((passes, references), np.float64),
        "mc_hidden_step": np.zeros((passes, references), np.float64),
    }


def test_candidate_cap_preserves_eligible_reference_queries() -> None:
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    selected, before_cap, protected = select_candidate_indices(
        scores,
        candidate_threshold=0.05,
        max_detections=2,
        required_query_indices=np.array([4]),
    )
    assert before_cap == 5
    assert protected == 1
    np.testing.assert_array_equal(selected, [0, 1, 4])


def test_trajectory_statistics_match_both_rq_definitions() -> None:
    rng = np.random.default_rng(41)
    hidden = rng.normal(size=(6, 3, 8)).astype(np.float32)
    references = rng.normal(size=(6, 3, 4)).astype(np.float32)
    variance, reference_step, hidden_step = _trajectory_statistics(
        hidden, references
    )
    for index in range(3):
        expected_variance, expected_reference_step = decoder_reference_features(
            references[:, index]
        )
        expected_hidden_step = decoder_hidden_step(hidden[:, index])
        rq2 = decoder_trajectory_features(
            hidden[:, index], references[:, index]
        )
        assert variance[index] == expected_variance
        assert reference_step[index] == expected_reference_step
        assert hidden_step[index] == expected_hidden_step
        assert variance[index] == rq2["deterministic_reference_variance"]
        assert reference_step[index] == rq2["deterministic_reference_step"]
        assert hidden_step[index] == rq2["deterministic_hidden_step"]


def test_future_rq_uses_the_same_compatibility_contract() -> None:
    shared = {
        "schema_version": 1,
        "artifact_root": "data/derived/groundingdino_mc_v1",
        "cache_namespace": "canonical",
        "format": "npz_compressed",
        "mc_passes": 10,
        "mc_seed_stride": 1009,
        "stochastic_module_types": ["DropPath"],
        "candidate_threshold": 0.01,
        "association_iou": 0.3,
        "association_class_penalty": 1.0,
        "unmatched_cost": 2.0,
    }
    config = {
        "shared_extraction": shared,
        "rq3": {
            "extraction": {
                key: shared[key]
                for key in (
                    "mc_passes",
                    "mc_seed_stride",
                    "stochastic_module_types",
                    "candidate_threshold",
                    "association_iou",
                    "association_class_penalty",
                    "unmatched_cost",
                )
            }
        },
    }
    validate_consumer_compatibility(config, "rq3")
    config["rq3"]["extraction"]["mc_passes"] = 5
    try:
        validate_consumer_compatibility(config, "rq3")
    except RuntimeError as error:
        assert "mc_passes" in str(error)
    else:
        raise AssertionError("An incompatible future RQ was accepted")


def test_rq1_rq2_common_mc_reductions_are_identical() -> None:
    rng = np.random.default_rng(73)
    passes, classes, hidden = 10, 4, 12
    present = np.asarray(
        [True, True, False, True, True, True, False, True, True, True]
    )
    category_scores = rng.uniform(size=(passes, classes)).astype(np.float32)
    scores = rng.uniform(size=passes).astype(np.float32)
    boxes = rng.uniform(size=(passes, 4)).astype(np.float32)
    embeddings = rng.normal(size=(passes, hidden)).astype(np.float32)
    reference_variance = rng.uniform(size=passes)
    reference_step = rng.uniform(size=passes)
    hidden_step = rng.uniform(size=passes)
    rq1 = {"absence_rate": float(1.0 - present.mean())}
    rq1.update(semantic_features(category_scores, scores, present, 2))
    rq1.update(
        geometric_features(
            boxes, present, reference_variance, reference_step
        )
    )
    rq1.update(representation_features(embeddings, present, hidden_step))
    rq2 = stochastic_features(
        category_scores=category_scores,
        scores=scores,
        boxes_cxcywh=boxes,
        embeddings=embeddings,
        present=present,
        base_category=2,
    )
    mapping = {
        "absence_rate": "stochastic_absence_rate",
        "semantic_mutual_information": "stochastic_mutual_information",
        "semantic_predictive_entropy": "stochastic_predictive_entropy",
        "class_disagreement": "stochastic_class_disagreement",
        "score_variance": "stochastic_score_variance",
        "box_variance": "stochastic_box_variance",
        "box_mean_pairwise_iou_loss": "stochastic_pairwise_iou_loss",
        "embedding_variance": "stochastic_embedding_variance",
        "embedding_cosine_instability": (
            "stochastic_embedding_cosine_instability"
        ),
    }
    for rq1_name, rq2_name in mapping.items():
        assert rq1[rq1_name] == rq2[rq2_name]


def test_shared_npz_shard_has_schema_and_hash_integrity(tmp_path: Path) -> None:
    path = tmp_path / "7.npz"
    metadata_path = tmp_path / "7.json"
    arrays = _arrays()
    config = {
        "shared_extraction": {"schema_version": 1, "mc_passes": 2},
        "data": {"classes": ["a", "b", "c"]},
    }
    _write_shard(
        path,
        metadata_path,
        arrays,
        {"image_id": 7, "file_name": "image.jpg"},
        {"sha256": "image-hash", "bytes": 123},
        "shared-fingerprint",
        [],
        config,
    )
    assert _shard_is_valid(
        path,
        metadata_path,
        image_id=7,
        file_name="image.jpg",
        image_record={"sha256": "image-hash", "bytes": 123},
        fingerprint="shared-fingerprint",
    )
    import json

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    loaded = load_shared_shard(
        path, metadata, config, "shared-fingerprint"
    )
    assert set(loaded) == set(ARRAY_SCHEMA)
    for name in arrays:
        np.testing.assert_array_equal(loaded[name], arrays[name])
