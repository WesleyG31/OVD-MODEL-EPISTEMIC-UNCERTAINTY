import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from adas_ovd.metrics import expected_calibration_error
from rq4.evaluation import (
    _aurc,
    _brier,
    _nll,
    _weighted_cluster_bootstrap,
    holm_adjust,
    maximum_calibration_error,
)


def test_maximum_calibration_error_known_bins():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.1, 0.8, 0.8])
    assert np.isclose(maximum_calibration_error(labels, probabilities, bins=2), 0.2)


def test_holm_adjust_is_monotone_in_sorted_p_values():
    adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.2})
    assert adjusted["a"] <= adjusted["b"] <= adjusted["c"]
    assert all(0.0 <= value <= 1.0 for value in adjusted.values())


def test_weighted_bootstrap_matches_unweighted_point_metrics():
    labels = np.array([0, 1, 0, 1, 1, 0])
    rank = np.array([0.05, 0.90, 0.20, 0.75, 0.60, 0.35])
    probability = np.array([0.10, 0.80, 0.25, 0.70, 0.65, 0.30])
    plan = {
        "counts": np.ones((1, len(labels)), dtype=np.int16),
        "group_codes": np.arange(len(labels), dtype=np.int32),
        "group_sizes": np.ones(len(labels), dtype=np.int64),
    }
    expected = {
        "auroc": roc_auc_score(labels, rank),
        "auprc": average_precision_score(labels, rank),
        "aurc": _aurc(labels, rank),
        "brier": _brier(labels, probability),
        "nll": _nll(labels, probability),
        "ece": expected_calibration_error(labels, probability),
    }
    for metric, point in expected.items():
        values = rank if metric in {"auroc", "auprc", "aurc"} else probability
        assert np.isclose(_weighted_cluster_bootstrap(labels, values, plan, metric)[0], point)


def test_weighted_bootstrap_matches_explicit_cluster_duplication():
    labels = np.array([0, 1, 0, 1, 1, 0])
    rank = np.array([0.05, 0.90, 0.20, 0.75, 0.60, 0.35])
    probability = np.array([0.10, 0.80, 0.25, 0.70, 0.65, 0.30])
    plan = {
        "counts": np.array([[2, 0, 1]], dtype=np.int16),
        "group_codes": np.repeat(np.arange(3), 2).astype(np.int32),
        "group_sizes": np.repeat(2, 3).astype(np.int64),
    }
    indices = np.array([0, 1, 0, 1, 4, 5])
    expected = {
        "auroc": roc_auc_score(labels[indices], rank[indices]),
        "auprc": average_precision_score(labels[indices], rank[indices]),
        "aurc": _aurc(labels[indices], rank[indices]),
        "brier": _brier(labels[indices], probability[indices]),
        "nll": _nll(labels[indices], probability[indices]),
        "ece": expected_calibration_error(labels[indices], probability[indices]),
    }
    for metric, point in expected.items():
        values = rank if metric in {"auroc", "auprc", "aurc"} else probability
        assert np.isclose(_weighted_cluster_bootstrap(labels, values, plan, metric)[0], point)
