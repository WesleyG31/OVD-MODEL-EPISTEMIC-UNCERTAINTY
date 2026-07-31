from types import SimpleNamespace

import pytest

from rq1.manifest import _partition_evaluation


def _evaluation() -> SimpleNamespace:
    return SimpleNamespace(
        images={
            100: SimpleNamespace(sequence_id="sequence-a"),
            101: SimpleNamespace(sequence_id="sequence-a"),
            102: SimpleNamespace(sequence_id="sequence-b"),
        }
    )


def _manifest() -> dict:
    return {
        "schema_version": 3,
        "splits": {"train": [1], "validation": [2], "test": [100, 101, 102]},
        "sequence_counts": {"train": 1, "validation": 1, "test": 2},
        "group_overlap_counts": {},
    }


def test_diagnostic_partition_is_closed_over_source_groups() -> None:
    result = _partition_evaluation(
        _manifest(), _evaluation(), [100], "diagnostic"
    )

    assert result["splits"]["diagnostic_test"] == [100, 101]
    assert result["splits"]["confirmatory_test"] == [102]
    assert result["splits"]["test"] == [100, 101]
    assert result["group_overlap_counts"][
        "diagnostic_confirmatory_test"
    ] == 0


def test_confirmatory_alias_excludes_the_diagnostic_group() -> None:
    result = _partition_evaluation(
        _manifest(), _evaluation(), [100], "confirmatory"
    )

    assert result["splits"]["test"] == [102]
    assert not set(result["splits"]["test"]) & set(
        result["splits"]["diagnostic_test"]
    )


def test_diagnostic_partition_rejects_unknown_image() -> None:
    with pytest.raises(ValueError, match="absent"):
        _partition_evaluation(
            _manifest(), _evaluation(), [999], "confirmatory"
        )
