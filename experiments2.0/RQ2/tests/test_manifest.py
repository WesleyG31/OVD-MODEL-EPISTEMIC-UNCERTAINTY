from __future__ import annotations

from pathlib import Path

from adas_ovd.config import load_config
from rq2.manifest import validate_manifest


PROJECT = Path(__file__).resolve().parents[2]


def test_frozen_diagnostic_manifest_is_valid_and_disjoint() -> None:
    config = load_config(PROJECT / "RQ2" / "configs" / "rq2_mini.yaml")
    _, manifest = validate_manifest(config)
    assert manifest["test_partition"] == "diagnostic"
    assert len(manifest["splits"]["test"]) == 8
    assert not (
        set(manifest["splits"]["diagnostic_test"])
        & set(manifest["splits"]["confirmatory_test"])
    )

