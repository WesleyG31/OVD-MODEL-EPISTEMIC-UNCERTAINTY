from copy import deepcopy

import pytest

from adas_ovd.config import load_config
from rq4.manifest import validate_manifest


def test_diagnostic_manifest_is_frozen_and_valid():
    config = load_config("RQ4/configs/rq4_mini.yaml")
    _, manifest = validate_manifest(config)
    assert manifest["test_partition"] == "diagnostic"
    assert len(manifest["splits"]["test"]) == 8


def test_manifest_partition_mismatch_fails():
    config = load_config("RQ4/configs/rq4_mini.yaml")
    bad = deepcopy(config)
    bad["rq4"]["manifest"]["expected_test_partition"] = "confirmatory"
    with pytest.raises(RuntimeError):
        validate_manifest(bad)

