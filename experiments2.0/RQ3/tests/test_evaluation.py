from __future__ import annotations

import numpy as np

from rq3.evaluation import _holm_adjust, _one_sided_bootstrap_p


def test_holm_adjustment_is_monotone_and_bounded() -> None:
    adjusted = _holm_adjust({"a": 0.01, "b": 0.03, "c": 0.20})
    assert adjusted == {"a": 0.03, "b": 0.06, "c": 0.20}


def test_one_sided_bootstrap_p_centers_distribution_under_null() -> None:
    assert _one_sided_bootstrap_p(
        np.array([0.8, 1.0, 1.2]), observed=1.0
    ) == 1 / 4
