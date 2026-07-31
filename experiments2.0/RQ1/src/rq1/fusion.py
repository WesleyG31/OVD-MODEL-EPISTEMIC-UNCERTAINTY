from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import IsotonicRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit

from adas_ovd.config import project_path
from adas_ovd.metrics import binary_uncertainty_metrics
from adas_ovd.reproducibility import (
    sha256_file,
    source_tree_sha256,
    stable_fingerprint,
    write_json,
)

from .extraction import read_validated_features


@dataclass
class FittedFusion:
    method: str
    features: list[str]
    estimator: Pipeline
    calibrator: IsotonicRegression | None
    family: str
    selected_regularization: float | None

    def rank_score(self, frame: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(frame[self.features])[:, 1]

    def error_probability(self, frame: pd.DataFrame) -> np.ndarray:
        raw = self.rank_score(frame)
        if self.calibrator is None:
            return raw
        return self.calibrator.predict(raw)


def _fusion_signature(config: dict[str, Any]) -> tuple[str, str]:
    source_sha256 = source_tree_sha256(
        config["_meta"]["project_root"], ("RQ1/src/rq1/fusion.py",)
    )
    fingerprint = stable_fingerprint(
        {
            "schema_version": 1,
            "source_sha256": source_sha256,
            "feature_groups": config["rq1"]["feature_groups"],
            "fusion": config["rq1"]["fusion"],
            "mc_sensitivity_passes": config["rq1"]["extraction"][
                "mc_sensitivity_passes"
            ],
        }
    )
    return source_sha256, fingerprint


def method_features(config: dict[str, Any]) -> dict[str, list[str]]:
    groups = config["rq1"]["feature_groups"]

    def combine(*names: str) -> list[str]:
        values: list[str] = []
        for name in names:
            for feature in groups[name]:
                if feature not in values:
                    values.append(feature)
        return values

    available = {
        "confidence": combine("confidence"),
        "semantic_mi": ["semantic_mutual_information"],
        "predictive_entropy": ["semantic_predictive_entropy"],
        "box_variance": ["box_variance"],
        "embedding_variance": ["embedding_variance"],
        "semantic": combine("semantic"),
        "geometric": combine("geometric"),
        "representation": combine("representation"),
        "presence": combine("presence"),
        "semantic_geometric": combine("semantic", "geometric"),
        "all_internal": combine(
            "semantic", "geometric", "representation", "presence"
        ),
        "all_internal_rf": combine(
            "semantic", "geometric", "representation", "presence"
        ),
        "all_plus_confidence": combine(
            "confidence", "semantic", "geometric", "representation", "presence"
        ),
    }
    requested = config["rq1"]["fusion"]["methods"]
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"Unknown fusion methods: {unknown}")
    return {method: available[method] for method in requested}


def mc_sensitivity_features(
    config: dict[str, Any]
) -> dict[str, list[str]]:
    primary = method_features(config)["all_internal"]
    counts = sorted(
        {
            int(value)
            for value in config["rq1"]["extraction"][
                "mc_sensitivity_passes"
            ]
        }
    )
    return {
        f"mc_passes_{count:02d}": [
            f"{feature}_mc{count:02d}" for feature in primary
        ]
        for count in counts
    }


def _logistic_estimator(regularization: float, seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            ),
            ("scaler", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=float(regularization),
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=int(seed),
                ),
            ),
        ]
    )


def _random_forest_estimator(
    settings: dict[str, Any], seed: int
) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            ),
            (
                "random_forest",
                RandomForestClassifier(
                    n_estimators=int(settings["n_estimators"]),
                    max_depth=(
                        None
                        if settings.get("max_depth") is None
                        else int(settings["max_depth"])
                    ),
                    min_samples_leaf=int(settings["min_samples_leaf"]),
                    max_features=settings.get("max_features", "sqrt"),
                    class_weight="balanced",
                    n_jobs=1,
                    random_state=int(seed),
                ),
            ),
        ]
    )


def split_validation_groups(
    validation: pd.DataFrame,
    *,
    group_column: str,
    selection_fraction: float,
    seed: int,
    minimum_calibration_samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create deterministic, group-disjoint selection/calibration folds."""
    if group_column not in validation:
        raise KeyError(f"Validation group column is missing: {group_column}")
    if not 0.0 < selection_fraction < 1.0:
        raise ValueError("validation_selection_fraction must be in (0, 1)")
    if validation[group_column].nunique() < 2:
        raise ValueError("At least two validation groups are required")
    groups = validation[group_column].astype(str).to_numpy()
    for attempt in range(256):
        splitter = GroupShuffleSplit(
            n_splits=1,
            train_size=float(selection_fraction),
            random_state=int(seed) + attempt,
        )
        selection_indices, calibration_indices = next(
            splitter.split(validation, validation["is_error"], groups)
        )
        selection = validation.iloc[selection_indices].copy()
        calibration = validation.iloc[calibration_indices].copy()
        if (
            selection["is_error"].nunique() == 2
            and calibration["is_error"].nunique() == 2
            and len(calibration) >= int(minimum_calibration_samples)
        ):
            overlap = set(selection[group_column].astype(str)) & set(
                calibration[group_column].astype(str)
            )
            if overlap:
                raise RuntimeError("Validation group split unexpectedly overlaps")
            return selection, calibration
    raise ValueError(
        "Could not create group-disjoint validation selection/calibration "
        "folds with both outcome classes and the requested sample minimum"
    )


def fit_fusions(config: dict[str, Any]) -> dict[str, FittedFusion]:
    outputs = config["rq1"]["outputs"]
    train_path = project_path(config, outputs["train_features"])
    validation_path = project_path(config, outputs["validation_features"])
    model_root = project_path(config, outputs["models"])
    model_root.mkdir(parents=True, exist_ok=True)
    train, train_metadata = read_validated_features(config, train_path)
    validation, validation_metadata = read_validated_features(
        config, validation_path
    )
    if train.empty or validation.empty:
        raise ValueError("Training and validation feature files must be non-empty")
    score_threshold = float(
        config["rq1"]["fusion"]["training_score_threshold"]
    )
    train = train.loc[train["score"] >= score_threshold].reset_index(drop=True)
    validation = validation.loc[
        validation["score"] >= score_threshold
    ].reset_index(drop=True)
    if len(validation) < int(config["rq1"]["fusion"]["minimum_validation_samples"]):
        raise ValueError("Validation set is smaller than the frozen protocol permits")
    if train["is_error"].nunique() < 2 or validation["is_error"].nunique() < 2:
        raise ValueError("Both train and validation require correct and error detections")

    fusion_config = config["rq1"]["fusion"]
    group_column = str(fusion_config["validation_group_column"])
    selection, calibration = split_validation_groups(
        validation,
        group_column=group_column,
        selection_fraction=float(
            fusion_config["validation_selection_fraction"]
        ),
        seed=int(config["project"]["seed"]),
        minimum_calibration_samples=int(
            fusion_config["minimum_calibration_samples"]
        ),
    )

    primary_features = method_features(config)
    sensitivity_features = mc_sensitivity_features(config)
    features_by_method = {**primary_features, **sensitivity_features}
    fusion_source_sha256, fusion_fingerprint = _fusion_signature(config)
    regularization_grid = fusion_config["regularization_grid"]
    seed = int(config["project"]["seed"])
    fitted: dict[str, FittedFusion] = {}
    index: dict[str, Any] = {
        "schema_version": 3,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "selection_rows": len(selection),
        "calibration_rows": len(calibration),
        "training_score_threshold": score_threshold,
        "validation_group_column": group_column,
        "selection_groups": int(selection[group_column].nunique()),
        "calibration_groups": int(calibration[group_column].nunique()),
        "selection_group_fingerprint": stable_fingerprint(
            sorted(selection[group_column].astype(str).unique())
        ),
        "calibration_group_fingerprint": stable_fingerprint(
            sorted(calibration[group_column].astype(str).unique())
        ),
        "train_features_sha256": train_metadata["features_sha256"],
        "validation_features_sha256": validation_metadata[
            "features_sha256"
        ],
        "source_tree_sha256": train_metadata["source_tree_sha256"],
        "fusion_source_sha256": fusion_source_sha256,
        "fusion_fingerprint": fusion_fingerprint,
        "methods": {},
    }
    validation_predictions = validation[
        [
            "image_id",
            "sequence_id",
            "detection_index",
            "is_error",
        ]
    ].copy()
    selection_groups = set(selection[group_column].astype(str))
    validation_predictions["validation_role"] = np.where(
        validation_predictions[group_column].astype(str).isin(selection_groups),
        "selection",
        "calibration",
    )

    for method, features in features_by_method.items():
        missing = sorted((set(features) - set(train.columns)) | (set(features) - set(validation.columns)))
        if missing:
            raise KeyError(f"{method} is missing features: {missing}")
        if method == "all_internal_rf":
            family = "random_forest"
            selected_c = None
            selection_estimator = _random_forest_estimator(
                fusion_config["random_forest"], seed
            )
            selection_estimator.fit(
                train[features], train["is_error"].astype(int)
            )
            selection_rank = selection_estimator.predict_proba(
                selection[features]
            )[:, 1]
            validation_auroc = float(
                roc_auc_score(selection["is_error"], selection_rank)
            )
            estimator = _random_forest_estimator(
                fusion_config["random_forest"], seed
            )
        else:
            family = "logistic"
            candidates = []
            for regularization in regularization_grid:
                candidate_estimator = _logistic_estimator(
                    float(regularization), seed
                )
                candidate_estimator.fit(
                    train[features], train["is_error"].astype(int)
                )
                score = candidate_estimator.predict_proba(
                    selection[features]
                )[:, 1]
                auroc = float(
                    roc_auc_score(selection["is_error"], score)
                )
                candidates.append((auroc, float(regularization)))
            candidates.sort(key=lambda item: (-item[0], item[1]))
            validation_auroc, selected_c = candidates[0]
            selection_estimator = _logistic_estimator(selected_c, seed)
            selection_estimator.fit(
                train[features], train["is_error"].astype(int)
            )
            selection_rank = selection_estimator.predict_proba(
                selection[features]
            )[:, 1]
            estimator = _logistic_estimator(selected_c, seed)

        fit_frame = pd.concat([train, selection], ignore_index=True)
        estimator.fit(fit_frame[features], fit_frame["is_error"].astype(int))
        validation_rank = estimator.predict_proba(validation[features])[:, 1]
        calibration_rank = estimator.predict_proba(calibration[features])[:, 1]

        calibrator: IsotonicRegression | None
        if fusion_config["calibrator"] == "isotonic":
            calibrator = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds="clip"
            )
            calibrator.fit(
                calibration_rank, calibration["is_error"].astype(int)
            )
            validation_probability = calibrator.predict(validation_rank)
            calibration_probability = calibrator.predict(calibration_rank)
        elif fusion_config["calibrator"] in {None, "none"}:
            calibrator = None
            validation_probability = validation_rank
            calibration_probability = calibration_rank
        else:
            raise ValueError(
                f"Unsupported calibrator: {fusion_config['calibrator']}"
            )

        model = FittedFusion(
            method=method,
            features=features,
            estimator=estimator,
            calibrator=calibrator,
            family=family,
            selected_regularization=selected_c,
        )
        joblib.dump(model, model_root / f"{method}.joblib")
        model_sha256 = sha256_file(model_root / f"{method}.joblib")
        fitted[method] = model
        validation_predictions[f"rank_{method}"] = validation_rank
        validation_predictions[f"prob_error_{method}"] = validation_probability
        rank_metrics = binary_uncertainty_metrics(
            selection["is_error"], selection_rank
        )
        probability_metrics = binary_uncertainty_metrics(
            calibration["is_error"],
            calibration_probability,
            calibrated_probability=True,
        )
        index["methods"][method] = {
            "experiment": (
                "mc_pass_sensitivity"
                if method in sensitivity_features
                else "primary_ablation"
            ),
            "model_sha256": model_sha256,
            "features": features,
            "family": family,
            "selected_regularization": selected_c,
            "selection_auroc": validation_auroc,
            "selection_rank_metrics": rank_metrics,
            "calibration_probability_metrics": {
                key: probability_metrics[key] for key in ("brier", "nll", "ece")
            },
        }

    validation_predictions.to_parquet(
        model_root / "validation_predictions.parquet", index=False
    )
    write_json(model_root / "model_index.json", index)
    return fitted


def _load_named_fusions(
    config: dict[str, Any], names: list[str]
) -> dict[str, FittedFusion]:
    model_root = project_path(config, config["rq1"]["outputs"]["models"])
    index_path = model_root / "model_index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Fusion model index is missing: {index_path}")
    import json

    with index_path.open("r", encoding="utf-8") as handle:
        index = json.load(handle)
    _, expected_fingerprint = _fusion_signature(config)
    if index.get("fusion_fingerprint") != expected_fingerprint:
        raise RuntimeError(
            "Fusion configuration/source changed after model fitting. "
            "Re-run the fit command."
        )
    for feature_key, output_key in (
        ("train_features_sha256", "train_features"),
        ("validation_features_sha256", "validation_features"),
    ):
        path = project_path(config, config["rq1"]["outputs"][output_key])
        if (
            not path.is_file()
            or index.get(feature_key) != sha256_file(path)
        ):
            raise RuntimeError(
                f"Fusion input integrity check failed for {path}"
            )
    for name in names:
        path = model_root / f"{name}.joblib"
        expected_hash = index.get("methods", {}).get(name, {}).get(
            "model_sha256"
        )
        if not path.is_file() or expected_hash != sha256_file(path):
            raise RuntimeError(f"Fusion model integrity check failed: {path}")
    return {
        method: joblib.load(model_root / f"{method}.joblib")
        for method in names
    }


def load_fusions(config: dict[str, Any]) -> dict[str, FittedFusion]:
    return _load_named_fusions(config, list(method_features(config)))


def load_mc_sensitivity_fusions(
    config: dict[str, Any]
) -> dict[str, FittedFusion]:
    return _load_named_fusions(
        config, list(mc_sensitivity_features(config))
    )
