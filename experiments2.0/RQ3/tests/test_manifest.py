from __future__ import annotations

from pathlib import Path

from adas_ovd.config import load_config
from rq3.manifest import validate_manifest


PROJECT = Path(__file__).resolve().parents[2]


def test_diagnostic_manifest_is_frozen_and_group_disjoint() -> None:
    config = load_config(PROJECT / "RQ3" / "configs" / "rq3_mini.yaml")
    _, manifest = validate_manifest(config)
    assert manifest["test_partition"] == "diagnostic"
    assert len(manifest["splits"]["train"]) == 5600
    assert len(manifest["splits"]["validation"]) == 2400
    assert len(manifest["splits"]["test"]) == 8
    assert not (
        set(manifest["splits"]["diagnostic_test"])
        & set(manifest["splits"]["confirmatory_test"])
    )
