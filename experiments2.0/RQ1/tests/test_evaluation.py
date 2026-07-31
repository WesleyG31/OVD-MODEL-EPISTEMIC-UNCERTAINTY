import numpy as np

from rq1.evaluation import _holm_adjust, _one_sided_bootstrap_p, _reliability_bins


def test_holm_adjustment_is_monotone_in_ordered_p_values() -> None:
    adjusted = _holm_adjust({"auroc": 0.01, "aurc": 0.03})
    assert adjusted == {"auroc": 0.02, "aurc": 0.03}


def test_one_sided_bootstrap_p_centers_distribution_under_null() -> None:
    assert _one_sided_bootstrap_p(
        np.array([0.8, 1.0, 1.2]), observed=1.0
    ) == 1 / 4


def test_reliability_bins_preserve_all_observations() -> None:
    records = _reliability_bins(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.2, 0.8, 0.9]),
        bins=2,
    )
    assert sum(record["count"] for record in records) == 4
    assert records[0]["observed_error_rate"] == 0.0
    assert records[1]["observed_error_rate"] == 1.0
