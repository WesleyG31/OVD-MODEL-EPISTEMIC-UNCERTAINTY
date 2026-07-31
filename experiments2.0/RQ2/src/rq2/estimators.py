from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import IsotonicRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from adas_ovd.config import project_path
from adas_ovd.metrics import binary_uncertainty_metrics
from adas_ovd.reproducibility import (
    sha256_file,
    source_tree_sha256,
    stable_fingerprint,
    write_json,
)

from .extraction import read_validated_features
from .features import finite_feature_audit


class RankScorer(Protocol):
    def rank_score(self, frame: pd.DataFrame) -> np.ndarray: ...


@dataclass
class ColumnScorer:
    feature: str

    def rank_score(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame[self.feature].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Direct score {self.feature} contains non-finite values")
        return values


@dataclass
class EmpiricalCDFScorer:
    features: list[str]
    sorted_values: dict[str, np.ndarray]
    medians: dict[str, float]

    @classmethod
    def fit(cls, frame: pd.DataFrame, features: list[str]) -> "EmpiricalCDFScorer":
        sorted_values: dict[str, np.ndarray] = {}
        medians: dict[str, float] = {}
        for feature in features:
            values = frame[feature].to_numpy(dtype=np.float64)
            finite_feature_audit(values)
            finite = values[np.isfinite(values)]
            sorted_values[feature] = np.sort(finite)
            medians[feature] = float(np.median(finite))
        return cls(features=list(features), sorted_values=sorted_values, medians=medians)

    def rank_score(self, frame: pd.DataFrame) -> np.ndarray:
        ranks = []
        for feature in self.features:
            values = frame[feature].to_numpy(dtype=np.float64)
            values = np.where(np.isfinite(values), values, self.medians[feature])
            reference = self.sorted_values[feature]
            ranks.append(
                np.searchsorted(reference, values, side="right").astype(np.float64)
                / len(reference)
            )
        return np.mean(np.column_stack(ranks), axis=1)


@dataclass
class AverageScorer:
    scorers: list[RankScorer]

    def rank_score(self, frame: pd.DataFrame) -> np.ndarray:
        return np.mean(
            np.column_stack([scorer.rank_score(frame) for scorer in self.scorers]),
            axis=1,
        )


@dataclass
class SklearnScorer:
    estimator: Pipeline

    def rank_score(self, frame: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(frame)[:, 1]


@dataclass
class FittedEstimator:
    method: str
    features: list[str]
    scorer: RankScorer
    calibrator: IsotonicRegression | None
    selected_regularization: float | None
    family: str

    def rank_score(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.scorer.rank_score(frame[self.features]), dtype=np.float64)

    def error_probability(self, frame: pd.DataFrame) -> np.ndarray:
        rank = self.rank_score(frame)
        return rank if self.calibrator is None else self.calibrator.predict(rank)


def _groups(config: dict[str, Any]) -> dict[str, list[str]]:
    return {
        name: list(values)
        for name, values in config["rq2"]["feature_groups"].items()
    }


def method_features(config: dict[str, Any]) -> dict[str, list[str]]:
    groups = _groups(config)
    deterministic = groups["deterministic"]
    stochastic = groups["stochastic"]
    all_uncertainty = deterministic + stochastic
    available = {
        "confidence": groups["confidence"],
        "deterministic_fixed": deterministic,
        "stochastic_fixed": stochastic,
        "equal_fixed_fusion": all_uncertainty,
        "learned_deterministic": deterministic,
        "learned_stochastic": stochastic,
        "learned_fusion": all_uncertainty,
        "learned_fusion_plus_confidence": all_uncertainty + groups["confidence"],
        "nonlinear_fusion": all_uncertainty,
    }
    requested = list(config["rq2"]["estimators"]["methods"])
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"Unknown RQ2 estimator methods: {unknown}")
    return {name: available[name] for name in requested}


def sensitivity_method_features(config: dict[str, Any]) -> dict[str, list[str]]:
    groups = _groups(config)
    return {
        f"learned_fusion_mc{count:02d}": groups["deterministic"]
        + [f"{feature}_mc{count:02d}" for feature in groups["stochastic"]]
        for count in sorted(
            int(value)
            for value in config["rq2"]["extraction"]["mc_sensitivity_passes"]
        )
    }


def _logistic_pipeline(regularization: float) -> Pipeline:
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median", add_indicator=True, keep_empty_features=True
                ),
            ),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=float(regularization),
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=0,
                ),
            ),
        ]
    )


def _random_forest_pipeline(config: dict[str, Any]) -> Pipeline:
    settings = dict(config["rq2"]["estimators"]["random_forest"])
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median", add_indicator=True, keep_empty_features=True
                ),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=int(settings["n_estimators"]),
                    max_depth=(
                        None
                        if settings.get("max_depth") is None
                        else int(settings["max_depth"])
                    ),
                    min_samples_leaf=int(settings["min_samples_leaf"]),
                    max_features=settings["max_features"],
                    class_weight=settings["class_weight"],
                    n_jobs=int(settings["n_jobs"]),
                    random_state=int(config["project"]["seed"]),
                ),
            ),
        ]
    )


def _calibrator(
    config: dict[str, Any], rank: np.ndarray, labels: pd.Series
) -> tuple[IsotonicRegression | None, np.ndarray]:
    kind = config["rq2"]["estimators"]["calibrator"]
    if kind == "isotonic":
        calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        calibrator.fit(rank, labels.astype(int))
        return calibrator, calibrator.predict(rank)
    if kind in {None, "none"}:
        return None, rank
    raise ValueError(f"Unsupported RQ2 calibrator: {kind}")


def _fit_logistic(
    config: dict[str, Any],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
) -> tuple[SklearnScorer, float, float, np.ndarray]:
    candidates = []
    for regularization in config["rq2"]["estimators"]["regularization_grid"]:
        pipeline = _logistic_pipeline(float(regularization))
        pipeline.fit(train[features], train["is_error"].astype(int))
        score = pipeline.predict_proba(validation[features])[:, 1]
        candidates.append(
            (
                float(roc_auc_score(validation["is_error"], score)),
                float(regularization),
                pipeline,
                score,
            )
        )
    candidates.sort(key=lambda item: (-item[0], item[1]))
    auroc, selected_c, pipeline, score = candidates[0]
    return SklearnScorer(pipeline), selected_c, auroc, score


def _fit_method(
    method: str,
    features: list[str],
    config: dict[str, Any],
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[RankScorer, float | None, float | None, np.ndarray, str]:
    groups = _groups(config)
    if method == "confidence":
        scorer: RankScorer = ColumnScorer(groups["confidence"][0])
        score = scorer.rank_score(validation[features])
        return scorer, None, None, score, "direct_control"
    if method == "deterministic_fixed":
        scorer = EmpiricalCDFScorer.fit(train, groups["deterministic"])
        score = scorer.rank_score(validation[features])
        return scorer, None, None, score, "fixed_empirical_cdf"
    if method == "stochastic_fixed":
        scorer = EmpiricalCDFScorer.fit(train, groups["stochastic"])
        score = scorer.rank_score(validation[features])
        return scorer, None, None, score, "fixed_empirical_cdf"
    if method == "equal_fixed_fusion":
        scorer = AverageScorer(
            [
                EmpiricalCDFScorer.fit(train, groups["deterministic"]),
                EmpiricalCDFScorer.fit(train, groups["stochastic"]),
            ]
        )
        score = scorer.rank_score(validation[features])
        return scorer, None, None, score, "fixed_equal_late_fusion"
    if method == "nonlinear_fusion":
        pipeline = _random_forest_pipeline(config)
        pipeline.fit(train[features], train["is_error"].astype(int))
        scorer = SklearnScorer(pipeline)
        score = scorer.rank_score(validation[features])
        return (
            scorer,
            None,
            float(roc_auc_score(validation["is_error"], score)),
            score,
            "random_forest_sensitivity",
        )
    scorer, selected_c, auroc, score = _fit_logistic(
        config, train, validation, features
    )
    return scorer, selected_c, auroc, score, "regularized_logistic"


def _estimator_signature(config: dict[str, Any]) -> tuple[str, str]:
    source_sha256 = source_tree_sha256(
        config["_meta"]["project_root"], ("RQ2/src/rq2/estimators.py",)
    )
    fingerprint = stable_fingerprint(
        {
            "schema_version": 1,
            "source_sha256": source_sha256,
            "feature_groups": config["rq2"]["feature_groups"],
            "estimators": config["rq2"]["estimators"],
            "operating_score_threshold": config["evaluation"][
                "primary_score_threshold"
            ],
            "mc_sensitivity_passes": config["rq2"]["extraction"][
                "mc_sensitivity_passes"
            ],
        }
    )
    return source_sha256, fingerprint


def _split_validation_groups(
    validation: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create label-valid, group-disjoint selection and calibration folds."""

    groups = validation["sequence_id"].astype(str)
    unique_groups = groups.nunique()
    if unique_groups < 2:
        raise ValueError(
            "RQ2 validation needs at least two sequence groups to separate "
            "model selection from calibration"
        )
    fraction = float(
        config["rq2"]["estimators"]["validation_calibration_fraction"]
    )
    minimum = int(
        config["rq2"]["estimators"]["minimum_partition_samples"]
    )
    base_seed = int(config["project"]["seed"])
    for offset in range(256):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=fraction,
            random_state=base_seed + offset,
        )
        selection_index, calibration_index = next(
            splitter.split(validation, validation["is_error"], groups)
        )
        selection = validation.iloc[selection_index].copy()
        calibration = validation.iloc[calibration_index].copy()
        if (
            len(selection) >= minimum
            and len(calibration) >= minimum
            and selection["is_error"].nunique() == 2
            and calibration["is_error"].nunique() == 2
        ):
            if set(selection["sequence_id"]) & set(calibration["sequence_id"]):
                raise RuntimeError("RQ2 validation group split leaked sequence IDs")
            return selection, calibration
    raise ValueError(
        "RQ2 could not create group-disjoint, label-valid selection and "
        "calibration folds with the frozen minimum sizes"
    )


def fit_estimators(config: dict[str, Any]) -> dict[str, FittedEstimator]:
    outputs = config["rq2"]["outputs"]
    train_path = project_path(config, outputs["train_features"])
    validation_path = project_path(config, outputs["validation_features"])
    train, train_metadata = read_validated_features(config, train_path)
    validation, validation_metadata = read_validated_features(config, validation_path)
    raw_train_rows = len(train)
    raw_validation_rows = len(validation)
    operating_threshold = float(config["evaluation"]["primary_score_threshold"])
    train = train.loc[train["score"] >= operating_threshold].copy()
    validation = validation.loc[
        validation["score"] >= operating_threshold
    ].copy()
    if train.empty or validation.empty:
        raise ValueError("RQ2 train and validation features must be non-empty")
    if len(validation) < int(config["rq2"]["estimators"]["minimum_validation_samples"]):
        raise ValueError("RQ2 validation detections are below the frozen minimum")
    for name, frame in (("train", train), ("validation", validation)):
        if frame["is_error"].nunique() != 2:
            raise ValueError(f"RQ2 {name} requires both correct and error detections")
    selection, calibration = _split_validation_groups(validation, config)

    primary = method_features(config)
    sensitivity = sensitivity_method_features(config)
    definitions = {**primary, **sensitivity}
    model_root = project_path(config, outputs["models"])
    model_root.mkdir(parents=True, exist_ok=True)
    source_sha256, fingerprint = _estimator_signature(config)
    index: dict[str, Any] = {
        "schema_version": 1,
        "source_tree_sha256": source_sha256,
        "estimator_fingerprint": fingerprint,
        "operating_score_threshold": operating_threshold,
        "raw_train_rows": raw_train_rows,
        "raw_validation_rows": raw_validation_rows,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "selection_rows": len(selection),
        "calibration_rows": len(calibration),
        "selection_groups": int(selection["sequence_id"].nunique()),
        "calibration_groups": int(calibration["sequence_id"].nunique()),
        "selection_calibration_group_overlap": 0,
        "train_features_sha256": train_metadata["features_sha256"],
        "validation_features_sha256": validation_metadata["features_sha256"],
        "methods": {},
    }
    validation_predictions = validation[
        ["image_id", "sequence_id", "detection_index", "is_error"]
    ].copy()
    fitted: dict[str, FittedEstimator] = {}
    for method, features in definitions.items():
        missing = sorted(
            (set(features) - set(train.columns))
            | (set(features) - set(validation.columns))
        )
        if missing:
            raise KeyError(f"RQ2 method {method} lacks features: {missing}")
        if method.startswith("learned_fusion_mc"):
            scorer, selected_c, selection_auroc, _ = _fit_logistic(
                config, train, selection, features
            )
            family = "mc_pass_sensitivity"
        else:
            scorer, selected_c, selection_auroc, _, family = _fit_method(
                method, features, config, train, selection
            )
        calibration_rank = scorer.rank_score(calibration[features])
        calibrator, probability = _calibrator(
            config, calibration_rank, calibration["is_error"]
        )
        estimator = FittedEstimator(
            method=method,
            features=features,
            scorer=scorer,
            calibrator=calibrator,
            selected_regularization=selected_c,
            family=family,
        )
        destination = model_root / f"{method}.joblib"
        temporary = destination.with_suffix(".joblib.tmp")
        joblib.dump(estimator, temporary)
        temporary.replace(destination)
        fitted[method] = estimator
        validation_rank = scorer.rank_score(validation[features])
        validation_probability = (
            validation_rank
            if calibrator is None
            else calibrator.predict(validation_rank)
        )
        validation_predictions[f"rank_{method}"] = validation_rank
        validation_predictions[f"prob_error_{method}"] = validation_probability
        selection_rank = scorer.rank_score(selection[features])
        rank_metrics = binary_uncertainty_metrics(
            selection["is_error"], selection_rank
        )
        probability_metrics = binary_uncertainty_metrics(
            calibration["is_error"],
            probability,
            calibrated_probability=True,
        )
        index["methods"][method] = {
            "family": family,
            "features": features,
            "selected_regularization": selected_c,
            "selection_auroc": selection_auroc,
            "selection_rank_metrics": rank_metrics,
            "heldout_calibration_probability_metrics": {
                name: probability_metrics[name] for name in ("brier", "nll", "ece")
            },
            "model_sha256": sha256_file(destination),
        }
    predictions_path = model_root / "validation_predictions.parquet"
    validation_predictions.to_parquet(predictions_path, index=False)
    index["validation_predictions_sha256"] = sha256_file(predictions_path)
    write_json(model_root / "model_index.json", index)
    return fitted


def _load(config: dict[str, Any], names: list[str]) -> dict[str, FittedEstimator]:
    outputs = config["rq2"]["outputs"]
    model_root = project_path(config, outputs["models"])
    index_path = model_root / "model_index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"RQ2 model index is missing: {index_path}")
    import json

    with index_path.open("r", encoding="utf-8") as handle:
        index = json.load(handle)
    _, expected_fingerprint = _estimator_signature(config)
    if index.get("estimator_fingerprint") != expected_fingerprint:
        raise RuntimeError("RQ2 estimator configuration/source changed after fitting")
    for metadata_key, output_key in (
        ("train_features_sha256", "train_features"),
        ("validation_features_sha256", "validation_features"),
    ):
        path = project_path(config, outputs[output_key])
        if not path.is_file() or index.get(metadata_key) != sha256_file(path):
            raise RuntimeError(f"RQ2 model input integrity failed: {path}")
    loaded = {}
    for name in names:
        path = model_root / f"{name}.joblib"
        expected_hash = index.get("methods", {}).get(name, {}).get("model_sha256")
        if not path.is_file() or expected_hash != sha256_file(path):
            raise RuntimeError(f"RQ2 model integrity failed: {path}")
        loaded[name] = joblib.load(path)
    return loaded


def load_estimators(config: dict[str, Any]) -> dict[str, FittedEstimator]:
    return _load(config, list(method_features(config)))


def load_sensitivity_estimators(
    config: dict[str, Any],
) -> dict[str, FittedEstimator]:
    return _load(config, list(sensitivity_method_features(config)))
