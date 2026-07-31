from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pandas as pd

from adas_ovd.config import load_config
from adas_ovd.shared_extraction import (
    ARRAY_SCHEMA,
    shared_identity,
    validate_consumer_compatibility,
)
from rq3.extraction import (
    _empty_frame,
    _materialization_identity,
    _valid_feature_shard,
    _write_feature_shard,
    feature_schema,
)


PROJECT = Path(__file__).resolve().parents[2]


def _config() -> dict:
    return load_config(PROJECT / "RQ3" / "configs" / "rq3_mini.yaml")


def test_rq3_config_is_v1_compatible_and_mismatch_fails() -> None:
    config = _config()
    validate_consumer_compatibility(config, "rq3")
    incompatible = deepcopy(config)
    incompatible["rq3"]["extraction"]["association_iou"] = 0.5
    try:
        validate_consumer_compatibility(incompatible, "rq3")
    except RuntimeError as error:
        assert "association_iou" in str(error)
    else:
        raise AssertionError("An incompatible RQ3 configuration was accepted")


def test_rq3_feature_schema_is_unique_and_does_not_extend_shared_v1() -> None:
    config = _config()
    columns = feature_schema(config)
    assert len(columns) == len(set(columns))
    assert "localization_iou" in columns
    assert "product_fusion" not in columns
    assert "localization_iou" not in ARRAY_SCHEMA
    assert "is_error" not in ARRAY_SCHEMA
    assert set(ARRAY_SCHEMA) == {
        "reference_boxes_cxcywh",
        "reference_boxes_xyxy",
        "reference_category_scores",
        "reference_scores",
        "reference_category_indices",
        "reference_query_indices",
        "deterministic_reference_variance",
        "deterministic_reference_step",
        "deterministic_hidden_step",
        "present",
        "mc_category_scores",
        "mc_scores",
        "mc_boxes_cxcywh",
        "mc_embeddings",
        "mc_reference_variance",
        "mc_reference_step",
        "mc_hidden_step",
    }


def test_empty_detection_frame_preserves_the_complete_frozen_schema() -> None:
    config = _config()
    frame = _empty_frame(config)
    assert frame.empty
    assert list(frame.columns) == feature_schema(config)


def test_downstream_feature_change_does_not_change_gpu_fingerprint() -> None:
    config = _config()
    baseline_shared = shared_identity(config)["configuration_fingerprint"]
    baseline_materialization = _materialization_identity(config, "train")[
        "materialization_fingerprint"
    ]
    changed = deepcopy(config)
    changed["rq3"]["feature_groups"]["spatial_static"] = list(
        changed["rq3"]["feature_groups"]["spatial_static"]
    ) + ["future_cpu_feature"]
    assert shared_identity(changed)["configuration_fingerprint"] == baseline_shared
    assert (
        _materialization_identity(changed, "train")["materialization_fingerprint"]
        != baseline_materialization
    )


def test_feature_shard_hash_schema_and_source_detect_corruption(tmp_path: Path) -> None:
    detection = tmp_path / "3.parquet"
    image = tmp_path / "3_image.parquet"
    metadata = tmp_path / "3.json"
    schema = ["image_id", "score"]
    arguments = {
        "image_id": 3,
        "materialization_fingerprint": "materialization",
        "source_tree_sha256_value": "source",
        "shared_shard_sha256": "shared",
        "expected_schema": schema,
    }
    _write_feature_shard(
        detection,
        image,
        metadata,
        pd.DataFrame({"image_id": [3], "score": [0.5]}),
        {"image_id": 3, "reference_detections": 1},
        materialization_fingerprint="materialization",
        source_tree_sha256_value="source",
        shared_shard_sha256="shared",
        expected_schema=schema,
    )
    assert _valid_feature_shard(detection, image, metadata, **arguments)
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    assert payload["feature_schema"] == schema
    assert payload["source_tree_sha256"] == "source"
    with detection.open("ab") as handle:
        handle.write(b"corrupt")
    assert not _valid_feature_shard(detection, image, metadata, **arguments)
