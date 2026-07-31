from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import IsotonicRegression
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

from .extraction import (
    nonspatial_feature_names,
    read_validated_features,
    spatial_feature_names,
)
from .features import finite_feature_audit


TARGET_OR_ORACLE_COLUMNS = {
    "is_true_positive",
    "is_error",
    "matched_iou",
    "matched_ground_truth_index",
    "localization_iou",
    "localization_ground_truth_index",
    "localization_class_agreement",
    "is_well_localized",
    "is_well_localized_050",
    "is_well_localized_075",
    "false_negatives_image",
    "sequence_id",
    "timeofday",
    "weather",
    "scene",
    "category_name",
    "object_size",
}


class RankScorer(Protocol):
    def rank_score(self, frame: pd.DataFrame) -> np.ndarray: ...


@dataclass
class ConfidenceScorer:
    def rank_score(self, frame: pd.DataFrame) -> np.ndarray:
        return 1.0 - frame["score"].to_numpy(dtype=np.float64)


@dataclass
class SpatialAgreementScorer:
    feature: str = "spatial_reference_iou_mean"

    def rank_score(self, frame: pd.DataFrame) -> np.ndarray:
        quality = frame[self.feature].to_numpy(dtype=np.float64)
        quality = np.where(np.isfinite(quality), quality, 0.0)
        return 1.0 - np.clip(quality, 0.0, 1.0)


@dataclass
class QualityModel:
    features: list[str]
    estimator: Pipeline
    target: str = "is_well_localized"

    def quality(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(
            self.estimator.predict_proba(frame[self.features])[:, 1],
            dtype=np.float64,
        )


@dataclass
class LearnedSpatialQualityScorer:
    quality_model: QualityModel

    def rank_score(self, frame: pd.DataFrame) -> np.ndarray:
        return 1.0 - self.quality_model.quality(frame)


@dataclass
class ProductFusionScorer:
    quality_model: QualityModel

    def rank_score(self, frame: pd.DataFrame) -> np.ndarray:
        confidence = frame["score"].to_numpy(dtype=np.float64)
        quality = self.quality_model.quality(frame)
        return 1.0 - np.clip(confidence * quality, 0.0, 1.0)


@dataclass
class EqualFusionScorer:
    quality_model: QualityModel

    def rank_score(self, frame: pd.DataFrame) -> np.ndarray:
        confidence_uncertainty = 1.0 - frame["score"].to_numpy(dtype=np.float64)
        spatial_uncertainty = 1.0 - self.quality_model.quality(frame)
        return (confidence_uncertainty + spatial_uncertainty) / 2.0


@dataclass
class DirectErrorScorer:
    features: list[str]
    estimator: Pipeline

    def rank_score(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(
            self.estimator.predict_proba(frame[self.features])[:, 1],
            dtype=np.float64,
        )


@dataclass
class FittedFusion:
    method: str
    features: list[str]
    scorer: RankScorer
    calibrator: IsotonicRegression | None
    selected_regularization: float | None
    selection_auroc: float | None
    family: str

    def rank_score(self, frame: pd.DataFrame) -> np.ndarray:
        values = np.asarray(self.scorer.rank_score(frame), dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"RQ3 method {self.method} produced non-finite ranks")
        return values

    def error_probability(self, frame: pd.DataFrame) -> np.ndarray:
        rank = self.rank_score(frame)
        probability = rank if self.calibrator is None else self.calibrator.predict(rank)
        return np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1 - 1e-7)


def method_feature_sets(config: dict[str, Any]) -> dict[str, list[str]]:
    spatial = spatial_feature_names(config)
    control = nonspatial_feature_names(config)
    available = {
        "confidence": ["score"],
        "spatial_agreement": ["spatial_reference_iou_mean"],
        "learned_spatial_quality": spatial,
        "equal_fusion": ["score", *spatial],
        "product_fusion": ["score", *spatial],
        "capacity_control_product": ["score", *control],
        "direct_spatial_fusion": ["score", *spatial],
    }
    requested = list(config["rq3"]["estimators"]["methods"])
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"Unknown RQ3 fusion methods: {unknown}")
    definitions = {name: available[name] for name in requested}
    validate_no_target_leakage(definitions)
    return definitions


def sensitivity_feature_sets(config: dict[str, Any]) -> dict[str, list[str]]:
    definitions = {
        f"product_fusion_mc{int(count):02d}": [
            "score",
            *spatial_feature_names(config, int(count)),
        ]
        for count in sorted(
            int(value)
            for value in config["rq3"]["extraction"]["mc_sensitivity_passes"]
        )
    }
    validate_no_target_leakage(definitions)
    return definitions


def validate_no_target_leakage(definitions: dict[str, list[str]]) -> None:
    for method, features in definitions.items():
        leaked = sorted(
            feature
            for feature in features
            if feature in TARGET_OR_ORACLE_COLUMNS
            or feature.startswith("localization_")
            or feature.startswith("is_well_localized")
            or feature.startswith("matched_")
        )
        if leaked:
            raise RuntimeError(f"RQ3 target leakage in {method}: {leaked}")
        if len(features) != len(set(features)):
            raise RuntimeError(f"RQ3 method {method} repeats a feature")


def _logistic_pipeline(regularization: float, seed: int) -> Pipeline:
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
                    random_state=int(seed),
                ),
            ),
        ]
    )


def split_validation_groups(
    frame: pd.DataFrame,
    *,
    group_column: str,
    selection_fraction: float,
    seed: int,
    minimum_selection_samples: int,
    minimum_calibration_samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < float(selection_fraction) < 1.0:
        raise ValueError("RQ3 validation selection fraction must lie in (0, 1)")
    if group_column not in frame:
        raise KeyError(f"RQ3 validation group column is missing: {group_column}")
    groups = frame[group_column].astype(str).to_numpy()
    if len(np.unique(groups)) < 2:
        raise ValueError("RQ3 validation requires at least two source groups")
    splitter = GroupShuffleSplit(
        n_splits=1,
        train_size=float(selection_fraction),
        random_state=int(seed),
    )
    selection_indices, calibration_indices = next(
        splitter.split(frame, groups=groups)
    )
    selection = frame.iloc[selection_indices].copy()
    calibration = frame.iloc[calibration_indices].copy()
    if len(selection) < int(minimum_selection_samples):
        raise ValueError("RQ3 selection detections are below the frozen minimum")
    if len(calibration) < int(minimum_calibration_samples):
        raise ValueError("RQ3 calibration detections are below the frozen minimum")
    selection_groups = set(selection[group_column].astype(str))
    calibration_groups = set(calibration[group_column].astype(str))
    if selection_groups & calibration_groups:
        raise RuntimeError("RQ3 validation selection/calibration group leakage")
    for name, subset in (("selection", selection), ("calibration", calibration)):
        for target in ("is_well_localized", "is_error"):
            if subset[target].nunique() != 2:
                raise ValueError(f"RQ3 {name} fold requires both {target} classes")
    return selection, calibration


def _fit_selected_logistic(
    config: dict[str, Any],
    train: pd.DataFrame,
    selection: pd.DataFrame,
    features: list[str],
    target: str,
) -> tuple[Pipeline, float, float]:
    candidates = []
    seed = int(config["project"]["seed"])
    for regularization in config["rq3"]["estimators"]["regularization_grid"]:
        estimator = _logistic_pipeline(float(regularization), seed)
        estimator.fit(train[features], train[target].astype(int))
        score = estimator.predict_proba(selection[features])[:, 1]
        candidates.append(
            (
                float(roc_auc_score(selection[target].astype(int), score)),
                float(regularization),
            )
        )
    candidates.sort(key=lambda item: (-item[0], item[1]))
    selection_auroc, selected_regularization = candidates[0]
    final_estimator = _logistic_pipeline(selected_regularization, seed)
    refit = pd.concat([train, selection], ignore_index=True)
    final_estimator.fit(refit[features], refit[target].astype(int))
    return final_estimator, selected_regularization, selection_auroc


def _calibrator(
    config: dict[str, Any], rank: np.ndarray, labels: pd.Series
) -> IsotonicRegression | None:
    kind = config["rq3"]["estimators"]["calibrator"]
    if kind == "isotonic":
        calibrator = IsotonicRegression(
            y_min=0.0, y_max=1.0, out_of_bounds="clip"
        )
        calibrator.fit(np.asarray(rank, dtype=np.float64), labels.astype(int))
        return calibrator
    if kind in {None, "none"}:
        return None
    raise ValueError(f"Unsupported RQ3 calibrator: {kind}")


def _fusion_signature(config: dict[str, Any]) -> tuple[str, str]:
    source_sha256 = source_tree_sha256(
        config["_meta"]["project_root"], ("RQ3/src/rq3/fusion.py",)
    )
    fingerprint = stable_fingerprint(
        {
            "schema_version": 1,
            "source_tree_sha256": source_sha256,
            "targets": config["rq3"]["targets"],
            "feature_groups": config["rq3"]["feature_groups"],
            "estimators": config["rq3"]["estimators"],
            "mc_sensitivity_passes": config["rq3"]["extraction"][
                "mc_sensitivity_passes"
            ],
        }
    )
    return source_sha256, fingerprint


def _make_method_scorers(
    config: dict[str, Any],
    train: pd.DataFrame,
    selection: pd.DataFrame,
) -> tuple[
    dict[str, tuple[RankScorer, float | None, float | None, str]],
    dict[str, tuple[RankScorer, float | None, float | None, str]],
]:
    spatial_features = spatial_feature_names(config)
    control_features = nonspatial_feature_names(config)
    spatial_estimator, spatial_c, spatial_auroc = _fit_selected_logistic(
        config,
        train,
        selection,
        spatial_features,
        "is_well_localized",
    )
    control_estimator, control_c, control_auroc = _fit_selected_logistic(
        config,
        train,
        selection,
        control_features,
        "is_well_localized",
    )
    direct_features = ["score", *spatial_features]
    direct_estimator, direct_c, direct_auroc = _fit_selected_logistic(
        config,
        train,
        selection,
        direct_features,
        "is_error",
    )
    spatial_model = QualityModel(spatial_features, spatial_estimator)
    control_model = QualityModel(control_features, control_estimator)
    available: dict[str, tuple[RankScorer, float | None, float | None, str]] = {
        "confidence": (ConfidenceScorer(), None, None, "confidence_control"),
        "spatial_agreement": (
            SpatialAgreementScorer(),
            None,
            None,
            "fixed_spatial_control",
        ),
        "learned_spatial_quality": (
            LearnedSpatialQualityScorer(spatial_model),
            spatial_c,
            spatial_auroc,
            "spatial_quality_only",
        ),
        "equal_fusion": (
            EqualFusionScorer(spatial_model),
            spatial_c,
            spatial_auroc,
            "fixed_equal_fusion",
        ),
        "product_fusion": (
            ProductFusionScorer(spatial_model),
            spatial_c,
            spatial_auroc,
            "localization_aware_product",
        ),
        "capacity_control_product": (
            ProductFusionScorer(control_model),
            control_c,
            control_auroc,
            "capacity_matched_nonspatial_control",
        ),
        "direct_spatial_fusion": (
            DirectErrorScorer(direct_features, direct_estimator),
            direct_c,
            direct_auroc,
            "direct_learned_sensitivity",
        ),
    }
    requested = method_feature_sets(config)
    main = {name: available[name] for name in requested}

    sensitivity: dict[
        str, tuple[RankScorer, float | None, float | None, str]
    ] = {}
    for method, method_features in sensitivity_feature_sets(config).items():
        quality_features = method_features[1:]
        estimator, selected_c, selection_auroc = _fit_selected_logistic(
            config,
            train,
            selection,
            quality_features,
            "is_well_localized",
        )
        quality_model = QualityModel(quality_features, estimator)
        sensitivity[method] = (
            ProductFusionScorer(quality_model),
            selected_c,
            selection_auroc,
            "mc_pass_sensitivity",
        )
    return main, sensitivity


def fit_fusions(config: dict[str, Any]) -> dict[str, FittedFusion]:
    outputs = config["rq3"]["outputs"]
    train_path = project_path(config, outputs["train_features"])
    validation_path = project_path(config, outputs["validation_features"])
    train_all, train_metadata = read_validated_features(config, train_path)
    validation_all, validation_metadata = read_validated_features(
        config, validation_path
    )
    threshold = float(config["rq3"]["estimators"]["training_score_threshold"])
    train = train_all.loc[train_all["score"] >= threshold].copy()
    validation = validation_all.loc[validation_all["score"] >= threshold].copy()
    if train.empty or validation.empty:
        raise ValueError("RQ3 operational train/validation features must be non-empty")
    if set(train["sequence_id"].astype(str)) & set(
        validation["sequence_id"].astype(str)
    ):
        raise RuntimeError("RQ3 train/validation source-group leakage")
    for name, frame in (("train", train), ("validation", validation)):
        for target in ("is_error", "is_well_localized"):
            if frame[target].nunique() != 2:
                raise ValueError(f"RQ3 {name} requires both {target} classes")
    definitions = {**method_feature_sets(config), **sensitivity_feature_sets(config)}
    for features in definitions.values():
        for feature in features:
            if feature not in train or feature not in validation:
                raise KeyError(f"RQ3 fitting feature is missing: {feature}")
            finite_feature_audit(
                pd.concat([train[feature], validation[feature]]).to_numpy(),
                allow_all_missing=False,
            )

    settings = config["rq3"]["estimators"]
    selection, calibration = split_validation_groups(
        validation,
        group_column=str(settings["validation_group_column"]),
        selection_fraction=float(settings["validation_selection_fraction"]),
        seed=int(config["project"]["seed"]),
        minimum_selection_samples=int(settings["minimum_selection_samples"]),
        minimum_calibration_samples=int(settings["minimum_calibration_samples"]),
    )
    main_scorers, sensitivity_scorers = _make_method_scorers(
        config, train, selection
    )
    all_scorers = {**main_scorers, **sensitivity_scorers}
    model_root = project_path(config, outputs["models"])
    model_root.mkdir(parents=True, exist_ok=True)
    source_sha256, fingerprint = _fusion_signature(config)
    index: dict[str, Any] = {
        "schema_version": 1,
        "source_tree_sha256": source_sha256,
        "fusion_fingerprint": fingerprint,
        "training_score_threshold": threshold,
        "train_rows_extracted": len(train_all),
        "train_rows_operational": len(train),
        "validation_rows_extracted": len(validation_all),
        "validation_rows_operational": len(validation),
        "selection_rows": len(selection),
        "calibration_rows": len(calibration),
        "train_features_sha256": train_metadata["features_sha256"],
        "validation_features_sha256": validation_metadata["features_sha256"],
        "group_audit": {
            "column": settings["validation_group_column"],
            "selection_groups_sha256": stable_fingerprint(
                sorted(selection[settings["validation_group_column"]].astype(str).unique())
            ),
            "calibration_groups_sha256": stable_fingerprint(
                sorted(calibration[settings["validation_group_column"]].astype(str).unique())
            ),
            "overlap_count": 0,
        },
        "methods": {},
    }
    validation_predictions = validation[
        [
            "image_id",
            "sequence_id",
            "detection_index",
            "score",
            "is_error",
            "is_well_localized",
        ]
    ].copy()
    selection_groups = set(
        selection[settings["validation_group_column"]].astype(str)
    )
    validation_predictions["validation_role"] = np.where(
        validation[settings["validation_group_column"]].astype(str).isin(
            selection_groups
        ),
        "selection",
        "calibration",
    )
    fitted: dict[str, FittedFusion] = {}
    for method, (scorer, selected_c, selection_auroc, family) in all_scorers.items():
        rank_calibration = np.asarray(
            scorer.rank_score(calibration), dtype=np.float64
        )
        if not np.isfinite(rank_calibration).all():
            raise ValueError(f"RQ3 scorer {method} is non-finite on calibration")
        calibrator = _calibrator(
            config, rank_calibration, calibration["is_error"]
        )
        method_features = definitions[method]
        fusion = FittedFusion(
            method=method,
            features=method_features,
            scorer=scorer,
            calibrator=calibrator,
            selected_regularization=selected_c,
            selection_auroc=selection_auroc,
            family=family,
        )
        destination = model_root / f"{method}.joblib"
        temporary = destination.with_suffix(".joblib.tmp")
        joblib.dump(fusion, temporary)
        temporary.replace(destination)
        fitted[method] = fusion
        rank_all = fusion.rank_score(validation)
        probability_all = fusion.error_probability(validation)
        validation_predictions[f"rank_{method}"] = rank_all
        validation_predictions[f"prob_error_{method}"] = probability_all
        calibration_mask = (
            validation_predictions["validation_role"].to_numpy() == "calibration"
        )
        calibration_metrics = binary_uncertainty_metrics(
            validation.loc[calibration_mask, "is_error"].to_numpy(),
            probability_all[calibration_mask],
            calibrated_probability=True,
        )
        index["methods"][method] = {
            "family": family,
            "features": method_features,
            "selected_regularization": selected_c,
            "selection_auroc": selection_auroc,
            "calibration_metrics": {
                name: calibration_metrics[name] for name in ("brier", "nll", "ece")
            },
            "model_sha256": sha256_file(destination),
        }
    validation_path = model_root / "validation_predictions.parquet"
    temporary_validation = validation_path.with_suffix(".parquet.tmp")
    validation_predictions.to_parquet(temporary_validation, index=False)
    temporary_validation.replace(validation_path)
    index["validation_predictions_sha256"] = sha256_file(validation_path)
    write_json(model_root / "model_index.json", index)
    return fitted


def _load_named(
    config: dict[str, Any], names: list[str]
) -> dict[str, FittedFusion]:
    outputs = config["rq3"]["outputs"]
    model_root = project_path(config, outputs["models"])
    index_path = model_root / "model_index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"RQ3 model index is missing: {index_path}")
    with index_path.open("r", encoding="utf-8") as handle:
        index = json.load(handle)
    _, expected_fingerprint = _fusion_signature(config)
    if index.get("fusion_fingerprint") != expected_fingerprint:
        raise RuntimeError("RQ3 fusion configuration/source changed after fitting")
    for metadata_key, output_key in (
        ("train_features_sha256", "train_features"),
        ("validation_features_sha256", "validation_features"),
    ):
        path = project_path(config, outputs[output_key])
        if not path.is_file() or index.get(metadata_key) != sha256_file(path):
            raise RuntimeError(f"RQ3 model input integrity failed: {path}")
    loaded: dict[str, FittedFusion] = {}
    for name in names:
        path = model_root / f"{name}.joblib"
        expected_hash = index.get("methods", {}).get(name, {}).get("model_sha256")
        if not path.is_file() or expected_hash != sha256_file(path):
            raise RuntimeError(f"RQ3 model integrity failed: {path}")
        loaded[name] = joblib.load(path)
    return loaded


def load_fusions(config: dict[str, Any]) -> dict[str, FittedFusion]:
    return _load_named(config, list(method_feature_sets(config)))


def load_sensitivity_fusions(config: dict[str, Any]) -> dict[str, FittedFusion]:
    return _load_named(config, list(sensitivity_feature_sets(config)))
