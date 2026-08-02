import json

import pandas as pd
import pytest

from adas_ovd.config import load_config
from adas_ovd.shared_extraction import ensure_shared_split, shared_identity, validate_consumer_compatibility
from rq4.extraction import (
    _materialization_identity,
    _valid_feature_shard,
    _write_feature_shard,
    configured_mini_image_ids,
    feature_schema,
    materialization_image_ids,
    source_domain_image_ids,
)
from rq4.manifest import validate_manifest


def test_shared_v1_contract_and_computed_identity():
    config = load_config("RQ4/configs/rq4_mini.yaml")
    validate_consumer_compatibility(config, "rq4")
    identity = shared_identity(config)
    assert identity["schema_version"] == 1
    assert len(identity["configuration_fingerprint"]) == 64
    assert config["rq4"]["extraction"]["mc_passes"] == 10


def test_feature_schema_is_unique_and_contains_future_contract_fields():
    config = load_config("RQ4/configs/rq4_mini.yaml")
    schema = feature_schema(config)
    assert len(schema) == len(set(schema))
    assert {"is_domain_shift", "is_class_correct", "mc_absence_rate"}.issubset(schema)
    assert "semantic_mutual_information_mc02" in schema
    assert "semantic_mutual_information_mc10" not in schema


def test_mini_development_ids_are_frozen_source_only_selections():
    config = load_config("RQ4/configs/rq4_mini.yaml")
    assert configured_mini_image_ids(config, "train") == [39, 89, 111, 138, 161, 171]
    assert configured_mini_image_ids(config, "validation") == [51, 79, 174, 230]
    assert configured_mini_image_ids(config, "test") == [6, 9, 18, 25, 27, 35, 42, 43]


def test_full_development_extraction_is_pruned_to_source_metadata():
    config = load_config("RQ4/configs/rq4.yaml")
    _, manifest = validate_manifest(config)
    train = source_domain_image_ids(config, "train", manifest["splits"]["train"])
    validation = source_domain_image_ids(config, "validation", manifest["splits"]["validation"])
    assert train is not None and validation is not None
    assert len(train) == 519
    assert len(validation) == 243
    assert train[:6] == [39, 89, 111, 138, 161, 171]
    assert validation[:4] == [51, 79, 174, 230]


def test_full_reader_identity_uses_the_same_source_only_image_universe():
    config = load_config("RQ4/configs/rq4.yaml")
    _, manifest = validate_manifest(config)
    for split, expected_count in (("train", 519), ("validation", 243), ("test", 1992)):
        image_ids = materialization_image_ids(config, split, manifest["splits"][split])
        implicit = _materialization_identity(config, split)
        explicit = _materialization_identity(config, split, requested_image_ids=image_ids)
        assert len(image_ids) == expected_count
        assert implicit["configuration_fingerprint"] == explicit["configuration_fingerprint"]


def test_shared_image_override_rejects_ids_outside_the_manifest():
    config = load_config("RQ4/configs/rq4_mini.yaml")
    with pytest.raises(ValueError, match="outside train"):
        ensure_shared_split(
            config,
            manifest_path="artifacts/bdd100k_diagnostic_manifest.json",
            split="train",
            image_ids_override=[999999],
        )


def test_corrupt_feature_shard_is_not_reused(tmp_path):
    detection = tmp_path / "1.parquet"
    summary = tmp_path / "summary.parquet"
    metadata = tmp_path / "1.json"
    frame = pd.DataFrame({"score": [0.5]})
    _write_feature_shard(
        detection, summary, metadata, frame, {"image_id": 1},
        materialization_fingerprint="f", source_tree_sha256_value="s",
        shared_shard_sha256="shared", expected_schema=["score"],
    )
    arguments = dict(
        image_id=1, materialization_fingerprint="f", source_tree_sha256_value="s",
        shared_shard_sha256="shared", expected_schema=["score"],
    )
    assert _valid_feature_shard(detection, summary, metadata, **arguments)
    detection.write_bytes(b"corrupt")
    assert not _valid_feature_shard(detection, summary, metadata, **arguments)
