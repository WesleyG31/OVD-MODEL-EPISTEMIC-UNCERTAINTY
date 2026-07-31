from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from adas_ovd.config import load_config
from adas_ovd.reproducibility import sha256_file, write_json
from adas_ovd.shared_extraction import validate_consumer_compatibility
from rq5 import extraction
from rq5.extraction import _empty_frame, _valid_feature_shard, feature_schema, read_validated_features
from rq5.manifest import validate_manifest


ROOT = Path(__file__).resolve().parents[2]


def test_rq5_declares_exact_shared_v1_compatibility() -> None:
    config = load_config(ROOT / "RQ5/configs/rq5.yaml")
    validate_consumer_compatibility(config, "rq5")
    changed = load_config(ROOT / "RQ5/configs/rq5.yaml")
    changed["rq5"]["extraction"]["association_iou"] = 0.31
    with pytest.raises(RuntimeError):
        validate_consumer_compatibility(changed, "rq5")


def test_manifest_and_diagnostic_namespace_are_frozen() -> None:
    full = load_config(ROOT / "RQ5/configs/rq5.yaml")
    mini = load_config(ROOT / "RQ5/configs/rq5_mini.yaml")
    _, full_manifest = validate_manifest(full)
    _, mini_manifest = validate_manifest(mini)
    assert full_manifest["test_partition"] == "confirmatory"
    assert mini_manifest["test_partition"] == "diagnostic"


def test_schema_is_unique_and_empty_case_preserves_contract() -> None:
    config = load_config(ROOT / "RQ5/configs/rq5.yaml")
    schema = feature_schema(config)
    assert len(schema) == len(set(schema))
    assert list(_empty_frame(config).columns) == schema
    assert "semantic_mutual_information_mc02" in schema
    assert "semantic_mutual_information_mc10" in schema


def test_feature_shard_hash_validation_detects_corruption(tmp_path: Path) -> None:
    detection = tmp_path / "1.parquet"
    summary = tmp_path / "summary.parquet"
    metadata = tmp_path / "1.json"
    pd.DataFrame({"x": [1]}).to_parquet(detection, index=False)
    pd.DataFrame({"image_id": [1]}).to_parquet(summary, index=False)
    payload = {
        "schema_version": 1,
        "image_id": 1,
        "materialization_fingerprint": "material",
        "source_tree_sha256": "source",
        "shared_shard_sha256": "shared",
        "feature_schema": ["x"],
        "feature_shard_sha256": sha256_file(detection),
        "image_summary_sha256": sha256_file(summary),
    }
    write_json(metadata, payload)
    arguments = dict(
        image_id=1,
        materialization_fingerprint="material",
        source_tree_sha256_value="source",
        shared_shard_sha256="shared",
        expected_schema=["x"],
    )
    assert _valid_feature_shard(detection, summary, metadata, **arguments)
    pd.DataFrame({"x": [2]}).to_parquet(detection, index=False)
    assert not _valid_feature_shard(detection, summary, metadata, **arguments)


def test_combined_artifact_rejects_bad_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    feature_path = tmp_path / "features.parquet"
    receipt = tmp_path / "receipt.json"
    image_summary = tmp_path / "images.parquet"
    pd.DataFrame({"x": [1.0]}).to_parquet(feature_path, index=False)
    pd.DataFrame({"image_id": [1]}).to_parquet(image_summary, index=False)
    receipt.write_text("{}", encoding="utf-8")
    expected = {
        "split": "train",
        "feature_schema": ["x"],
        "materialization_fingerprint": "m",
    }
    metadata = {
        **expected,
        "features_sha256": "bad",
        "shared_request_metadata_path": str(receipt),
        "shared_request_metadata_sha256": sha256_file(receipt),
        "image_summary_path": str(image_summary),
        "image_summary_sha256": sha256_file(image_summary),
    }
    feature_path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    monkeypatch.setattr(extraction, "_materialization_identity", lambda config, split: expected)
    with pytest.raises(RuntimeError, match="SHA-256"):
        read_validated_features({}, feature_path)

