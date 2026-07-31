from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import IsotonicRegression
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from adas_ovd.config import project_path
from adas_ovd.metrics import binary_uncertainty_metrics
from adas_ovd.reproducibility import sha256_file, source_tree_sha256, stable_fingerprint, write_json

from .extraction import model_feature_names, read_validated_features
from .features import finite_feature_audit


TARGET_OR_DESCRIPTOR_COLUMNS = {
    "is_true_positive", "is_error", "is_class_correct", "is_well_localized",
    "matched_iou", "matched_ground_truth_index", "localization_iou",
    "localization_ground_truth_index", "localization_class_agreement",
    "false_negatives_image", "image_id", "file_name", "sequence_id",
    "timeofday", "weather", "scene", "category_name", "object_size",
    "shift_timeofday", "shift_weather", "shift_scene", "unknown_timeofday",
    "unknown_weather", "unknown_scene", "shift_axis_count", "unknown_axis_count",
    "is_domain_shift", "domain_stratum",
}


@dataclass
class ComponentModel:
    name: str
    features: list[str]
    target: str
    estimator: Pipeline
    selected_regularization: float
    selection_auroc: float
    known_categories: tuple[int, ...] = ()

    def prepared_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        prepared = frame[self.features].copy()
        if "category_index" in self.features:
            prepared["category_index"] = pd.to_numeric(
                prepared["category_index"], errors="coerce"
            ).fillna(-1).astype(int)
        return prepared

    def probability(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(
            self.estimator.predict_proba(self.prepared_features(frame))[:, 1],
            dtype=np.float64,
        )

    @property
    def coefficient_count(self) -> int:
        classifier = self.estimator.named_steps["classifier"]
        return int(classifier.coef_.size + classifier.intercept_.size)


@dataclass
class MethodScorer:
    kind: str
    components: dict[str, ComponentModel]
    product_levels: tuple[str, ...] = ()

    def rank_score(self, frame: pd.DataFrame) -> np.ndarray:
        if self.kind in {"raw_confidence", "confidence_calibrated"}:
            return 1.0 - frame["score"].to_numpy(dtype=np.float64)
        if self.kind == "flat_joint":
            return 1.0 - self.components["flat"].probability(frame)
        if not self.product_levels:
            raise RuntimeError(f"RQ4 scorer has no levels: {self.kind}")
        quality = np.ones(len(frame), dtype=np.float64)
        for level in self.product_levels:
            quality *= self.components[level].probability(frame)
        return 1.0 - np.clip(quality, 0.0, 1.0)

    @property
    def coefficient_count(self) -> int:
        used = {"flat"} if self.kind == "flat_joint" else set(self.product_levels)
        return int(sum(self.components[name].coefficient_count for name in used))


@dataclass
class FittedCalibration:
    method: str
    features: list[str]
    scorer: MethodScorer
    calibrator: IsotonicRegression | None
    family: str

    def rank_score(self, frame: pd.DataFrame) -> np.ndarray:
        values = np.asarray(self.scorer.rank_score(frame), dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"RQ4 method {self.method} produced non-finite ranks")
        return values

    def error_probability(self, frame: pd.DataFrame) -> np.ndarray:
        rank = self.rank_score(frame)
        values = rank if self.calibrator is None else self.calibrator.predict(rank)
        return np.clip(np.asarray(values, dtype=np.float64), 1e-7, 1 - 1e-7)

    @property
    def coefficient_count(self) -> int:
        return self.scorer.coefficient_count


def component_feature_sets(
    config: dict[str, Any], mc_passes: int | None = None
) -> dict[str, list[str]]:
    definitions = {
        "class": model_feature_names(config, "class"),
        "localization": model_feature_names(config, "localization", mc_passes),
        "uncertainty": model_feature_names(config, "uncertainty", mc_passes),
    }
    definitions["flat"] = list(dict.fromkeys(sum(definitions.values(), [])))
    validate_no_target_leakage(definitions)
    return definitions


def method_feature_sets(config: dict[str, Any]) -> dict[str, list[str]]:
    components = component_feature_sets(config)
    available = {
        "raw_confidence": ["score"],
        "confidence_calibrated": ["score"],
        "class_only": components["class"],
        "localization_only": components["localization"],
        "uncertainty_only": components["uncertainty"],
        "class_localization": list(dict.fromkeys(components["class"] + components["localization"])),
        "class_uncertainty": list(dict.fromkeys(components["class"] + components["uncertainty"])),
        "localization_uncertainty": list(dict.fromkeys(components["localization"] + components["uncertainty"])),
        "multilevel": components["flat"],
        "flat_joint": components["flat"],
    }
    requested = list(config["rq4"]["calibration"]["methods"])
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"Unknown RQ4 methods: {unknown}")
    result = {name: available[name] for name in requested}
    validate_no_target_leakage(result)
    return result


def sensitivity_feature_sets(config: dict[str, Any]) -> dict[str, list[str]]:
    total_passes = int(config["rq4"]["extraction"]["mc_passes"])
    result = {}
    for count in sorted(int(value) for value in config["rq4"]["extraction"]["mc_sensitivity_passes"]):
        features = component_feature_sets(config)["flat"] if count == total_passes else component_feature_sets(config, count)["flat"]
        result[f"multilevel_mc{count:02d}"] = features
    validate_no_target_leakage(result)
    return result


def validate_no_target_leakage(definitions: dict[str, list[str]]) -> None:
    for name, features in definitions.items():
        leaked = sorted(
            feature for feature in features
            if feature in TARGET_OR_DESCRIPTOR_COLUMNS
            or feature.startswith("matched_")
            or feature.startswith("localization_iou")
            or feature.startswith("shift_")
            or feature.startswith("unknown_")
        )
        if leaked:
            raise RuntimeError(f"RQ4 target/domain leakage in {name}: {leaked}")
        if len(features) != len(set(features)):
            raise RuntimeError(f"RQ4 definition {name} repeats a feature")


def source_domain_mask(frame: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    policy = str(config["rq4"]["domain_shift"].get("development_policy", ""))
    if policy != "source_only":
        raise ValueError(f"RQ4 requires the frozen source_only development policy, got: {policy!r}")
    reference = config["rq4"]["domain_shift"]["reference"]
    required = ("timeofday", "weather", "scene", "unknown_axis_count", "is_domain_shift")
    missing = [column for column in required if column not in frame]
    if missing:
        raise KeyError(f"RQ4 source-domain columns are missing: {missing}")
    return (
        frame["timeofday"].astype(str).eq(str(reference["timeofday"]))
        & frame["weather"].astype(str).eq(str(reference["weather"]))
        & frame["scene"].astype(str).eq(str(reference["scene"]))
        & frame["unknown_axis_count"].fillna(1).astype(int).eq(0)
        & ~frame["is_domain_shift"].astype(bool)
    )


def _logistic_pipeline(
    features: list[str],
    regularization: float,
    seed: int,
    known_categories: tuple[int, ...] = (),
) -> Pipeline:
    numeric_features = [feature for feature in features if feature != "category_index"]
    if "category_index" in features:
        numeric = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                ("scaler", StandardScaler()),
            ]
        )
        transformers: list[tuple[str, Any, list[str]]] = [("numeric", numeric, numeric_features)]
        if known_categories:
            transformers.append(
                (
                    "category",
                    OneHotEncoder(
                        categories=[list(known_categories)],
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                    ["category_index"],
                )
            )
        preprocessor: Any = ColumnTransformer(transformers, remainder="drop")
    else:
        preprocessor = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                ("scaler", StandardScaler()),
            ]
        )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(C=float(regularization), max_iter=2000, solver="lbfgs", random_state=int(seed))),
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
        raise ValueError("RQ4 validation selection fraction must lie in (0, 1)")
    if group_column not in frame:
        raise KeyError(f"RQ4 validation group column is missing: {group_column}")
    groups = frame[group_column].astype(str).to_numpy()
    if len(np.unique(groups)) < 2:
        raise ValueError("RQ4 validation requires at least two source groups")
    splitter = GroupShuffleSplit(n_splits=1, train_size=float(selection_fraction), random_state=int(seed))
    selection_indices, calibration_indices = next(splitter.split(frame, groups=groups))
    selection = frame.iloc[selection_indices].copy()
    calibration = frame.iloc[calibration_indices].copy()
    if len(selection) < int(minimum_selection_samples):
        raise ValueError("RQ4 selection detections are below the frozen minimum")
    if len(calibration) < int(minimum_calibration_samples):
        raise ValueError("RQ4 calibration detections are below the frozen minimum")
    if set(selection[group_column].astype(str)) & set(calibration[group_column].astype(str)):
        raise RuntimeError("RQ4 validation selection/calibration group leakage")
    for name, subset in (("selection", selection), ("calibration", calibration)):
        for target in ("is_class_correct", "is_well_localized", "is_detection_correct", "is_error"):
            if subset[target].nunique() != 2:
                raise ValueError(f"RQ4 {name} fold requires both {target} classes")
    return selection, calibration


def _fit_component(
    config: dict[str, Any],
    train: pd.DataFrame,
    selection: pd.DataFrame,
    *,
    name: str,
    features: list[str],
    target: str,
) -> ComponentModel:
    seed = int(config["project"]["seed"])
    known_categories: tuple[int, ...] = ()
    if "category_index" in features:
        minimum = int(config["rq4"]["calibration"]["minimum_class_rows"])
        counts = pd.to_numeric(train["category_index"], errors="coerce").dropna().astype(int).value_counts()
        known_categories = tuple(sorted(int(value) for value in counts[counts >= minimum].index))

    def prepared(frame: pd.DataFrame) -> pd.DataFrame:
        values = frame[features].copy()
        if "category_index" in features:
            values["category_index"] = pd.to_numeric(
                values["category_index"], errors="coerce"
            ).fillna(-1).astype(int)
        return values

    candidates: list[tuple[float, float]] = []
    for regularization in config["rq4"]["calibration"]["regularization_grid"]:
        estimator = _logistic_pipeline(features, float(regularization), seed, known_categories)
        estimator.fit(prepared(train), train[target].astype(int))
        probability = estimator.predict_proba(prepared(selection))[:, 1]
        candidates.append((float(roc_auc_score(selection[target].astype(int), probability)), float(regularization)))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    selection_auroc, selected_regularization = candidates[0]
    final = _logistic_pipeline(features, selected_regularization, seed, known_categories)
    refit = pd.concat([train, selection], ignore_index=True)
    final.fit(prepared(refit), refit[target].astype(int))
    return ComponentModel(
        name=name,
        features=features,
        target=target,
        estimator=final,
        selected_regularization=selected_regularization,
        selection_auroc=selection_auroc,
        known_categories=known_categories,
    )


def _fit_component_set(
    config: dict[str, Any],
    train: pd.DataFrame,
    selection: pd.DataFrame,
    mc_passes: int | None = None,
    names: tuple[str, ...] = ("class", "localization", "uncertainty", "flat"),
) -> dict[str, ComponentModel]:
    features = component_feature_sets(config, mc_passes)
    targets = {
        "class": "is_class_correct",
        "localization": "is_well_localized",
        "uncertainty": "is_detection_correct",
        "flat": "is_detection_correct",
    }
    return {
        name: _fit_component(config, train, selection, name=name, features=feature_names, target=targets[name])
        for name, feature_names in features.items()
        if name in names
    }


def _scorers(config: dict[str, Any], train: pd.DataFrame, selection: pd.DataFrame) -> dict[str, MethodScorer]:
    components = _fit_component_set(config, train, selection)
    available = {
        "raw_confidence": MethodScorer("raw_confidence", components),
        "confidence_calibrated": MethodScorer("confidence_calibrated", components),
        "class_only": MethodScorer("class_only", components, ("class",)),
        "localization_only": MethodScorer("localization_only", components, ("localization",)),
        "uncertainty_only": MethodScorer("uncertainty_only", components, ("uncertainty",)),
        "class_localization": MethodScorer("class_localization", components, ("class", "localization")),
        "class_uncertainty": MethodScorer("class_uncertainty", components, ("class", "uncertainty")),
        "localization_uncertainty": MethodScorer("localization_uncertainty", components, ("localization", "uncertainty")),
        "multilevel": MethodScorer("multilevel", components, ("class", "localization", "uncertainty")),
        "flat_joint": MethodScorer("flat_joint", components),
    }
    scorers = {name: available[name] for name in method_feature_sets(config)}
    total_passes = int(config["rq4"]["extraction"]["mc_passes"])
    for count in sorted(int(value) for value in config["rq4"]["extraction"]["mc_sensitivity_passes"]):
        if count == total_passes:
            prefix_components = {
                name: components[name] for name in ("class", "localization", "uncertainty")
            }
        else:
            prefix_components = {"class": components["class"]}
            prefix_components.update(
                _fit_component_set(
                    config,
                    train,
                    selection,
                    count,
                    names=("localization", "uncertainty"),
                )
            )
        name = f"multilevel_mc{count:02d}"
        scorers[name] = MethodScorer(name, prefix_components, ("class", "localization", "uncertainty"))
    return scorers


def _calibrator(config: dict[str, Any], rank: np.ndarray, labels: pd.Series) -> IsotonicRegression | None:
    kind = config["rq4"]["calibration"]["final_calibrator"]
    if kind == "isotonic":
        calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        calibrator.fit(np.asarray(rank, dtype=np.float64), labels.astype(int))
        return calibrator
    if kind in {None, "none"}:
        return None
    raise ValueError(f"Unsupported RQ4 final calibrator: {kind}")


def _calibration_signature(config: dict[str, Any]) -> tuple[str, str]:
    source_sha256 = source_tree_sha256(config["_meta"]["project_root"], ("RQ4/src/rq4/calibration.py",))
    fingerprint = stable_fingerprint(
        {
            "schema_version": 2,
            "source_tree_sha256": source_sha256,
            "targets": config["rq4"]["targets"],
            "feature_groups": config["rq4"]["feature_groups"],
            "calibration": config["rq4"]["calibration"],
            "domain_shift": config["rq4"]["domain_shift"],
            "mc_sensitivity_passes": config["rq4"]["extraction"]["mc_sensitivity_passes"],
        }
    )
    return source_sha256, fingerprint


def fit_calibrations(config: dict[str, Any]) -> dict[str, FittedCalibration]:
    requested_names = list(method_feature_sets(config)) + list(sensitivity_feature_sets(config))
    try:
        return _load_named(config, requested_names)
    except (FileNotFoundError, RuntimeError, OSError, EOFError, ValueError, KeyError, AttributeError):
        pass

    outputs = config["rq4"]["outputs"]
    train_all, train_metadata = read_validated_features(config, project_path(config, outputs["train_features"]))
    validation_all, validation_metadata = read_validated_features(config, project_path(config, outputs["validation_features"]))
    threshold = float(config["rq4"]["calibration"]["training_score_threshold"])
    train_operational = train_all["score"] >= threshold
    validation_operational = validation_all["score"] >= threshold
    train_source = source_domain_mask(train_all, config)
    validation_source = source_domain_mask(validation_all, config)
    train = train_all.loc[train_operational & train_source].copy()
    validation = validation_all.loc[validation_operational & validation_source].copy()
    for frame in (train, validation):
        frame["is_detection_correct"] = 1 - frame["is_error"].astype(int)
    if train.empty or validation.empty:
        raise ValueError("RQ4 source-domain operational train/validation features must be non-empty")
    if not source_domain_mask(train, config).all() or not source_domain_mask(validation, config).all():
        raise RuntimeError("RQ4 source-only development invariant failed")
    if set(train["sequence_id"].astype(str)) & set(validation["sequence_id"].astype(str)):
        raise RuntimeError("RQ4 train/validation source-group leakage")
    definitions = {**method_feature_sets(config), **sensitivity_feature_sets(config)}
    for features in definitions.values():
        for feature in features:
            if feature not in train or feature not in validation:
                raise KeyError(f"RQ4 fitting feature is missing: {feature}")
            finite_feature_audit(pd.concat([train[feature], validation[feature]]).to_numpy())
    settings = config["rq4"]["calibration"]
    selection, calibration = split_validation_groups(
        validation,
        group_column=str(settings["validation_group_column"]),
        selection_fraction=float(settings["validation_selection_fraction"]),
        seed=int(config["project"]["seed"]),
        minimum_selection_samples=int(settings["minimum_selection_samples"]),
        minimum_calibration_samples=int(settings["minimum_calibration_samples"]),
    )
    scorers = _scorers(config, train, selection)
    model_root = project_path(config, outputs["models"])
    model_root.mkdir(parents=True, exist_ok=True)
    source_sha256, fingerprint = _calibration_signature(config)
    index: dict[str, Any] = {
        "schema_version": 2,
        "source_tree_sha256": source_sha256,
        "calibration_fingerprint": fingerprint,
        "development_policy": "source_only",
        "source_domain": config["rq4"]["domain_shift"]["reference"],
        "training_score_threshold": threshold,
        "train_rows_extracted": len(train_all),
        "train_rows_operational_all_domains": int(train_operational.sum()),
        "train_rows_source_operational": len(train),
        "validation_rows_extracted": len(validation_all),
        "validation_rows_operational_all_domains": int(validation_operational.sum()),
        "validation_rows_source_operational": len(validation),
        "selection_rows": len(selection),
        "calibration_rows": len(calibration),
        "train_features_sha256": train_metadata["features_sha256"],
        "validation_features_sha256": validation_metadata["features_sha256"],
        "group_audit": {
            "column": settings["validation_group_column"],
            "selection_groups_sha256": stable_fingerprint(sorted(selection[settings["validation_group_column"]].astype(str).unique())),
            "calibration_groups_sha256": stable_fingerprint(sorted(calibration[settings["validation_group_column"]].astype(str).unique())),
            "overlap_count": 0,
        },
        "domain_audit": {
            "source_only_development": True,
            "shifted_rows_used": 0,
            "train_source_image_ids_sha256": stable_fingerprint(sorted(int(value) for value in train["image_id"].unique())),
            "validation_source_image_ids_sha256": stable_fingerprint(sorted(int(value) for value in validation["image_id"].unique())),
        },
        "methods": {},
    }
    validation_predictions = validation[["image_id", "sequence_id", "detection_index", "score", "is_error", "is_class_correct", "is_well_localized"]].copy()
    selection_groups = set(selection[settings["validation_group_column"]].astype(str))
    validation_predictions["validation_role"] = np.where(
        validation[settings["validation_group_column"]].astype(str).isin(selection_groups), "selection", "calibration"
    )
    main_features = method_feature_sets(config)
    sensitivity_features = sensitivity_feature_sets(config)
    fitted: dict[str, FittedCalibration] = {}
    for method, scorer in scorers.items():
        rank_calibration = scorer.rank_score(calibration)
        calibrator = None if method == "raw_confidence" else _calibrator(config, rank_calibration, calibration["is_error"])
        features = (main_features | sensitivity_features)[method]
        family = (
            "primary_multilevel" if method == "multilevel" else
            "flat_capacity_control" if method == "flat_joint" else
            "confidence_control" if method in {"raw_confidence", "confidence_calibrated"} else
            "mc_pass_sensitivity" if method.startswith("multilevel_mc") else
            "level_ablation"
        )
        fitted_method = FittedCalibration(method, features, scorer, calibrator, family)
        destination = model_root / f"{method}.joblib"
        temporary = destination.with_suffix(".joblib.tmp")
        joblib.dump(fitted_method, temporary)
        temporary.replace(destination)
        fitted[method] = fitted_method
        rank_all = fitted_method.rank_score(validation)
        probability_all = fitted_method.error_probability(validation)
        validation_predictions[f"rank_{method}"] = rank_all
        validation_predictions[f"prob_error_{method}"] = probability_all
        calibration_mask = validation_predictions["validation_role"].to_numpy() == "calibration"
        metrics = binary_uncertainty_metrics(
            validation.loc[calibration_mask, "is_error"].to_numpy(),
            probability_all[calibration_mask],
            calibrated_probability=True,
        )
        component_metadata = {
            name: {
                "target": component.target,
                "features": component.features,
                "selected_regularization": component.selected_regularization,
                "selection_auroc": component.selection_auroc,
                "coefficient_count": component.coefficient_count,
                "known_categories": list(component.known_categories),
                "rare_or_unseen_category_fallback": "all_zero_category_encoding" if "category_index" in component.features else None,
            }
            for name, component in scorer.components.items()
        }
        index["methods"][method] = {
            "family": family,
            "features": features,
            "coefficient_count": fitted_method.coefficient_count,
            "components": component_metadata,
            "calibration_metrics": {name: metrics[name] for name in ("brier", "nll", "ece")},
            "model_sha256": sha256_file(destination),
        }
    validation_path = model_root / "validation_predictions.parquet"
    temporary_validation = validation_path.with_suffix(".parquet.tmp")
    validation_predictions.to_parquet(temporary_validation, index=False)
    temporary_validation.replace(validation_path)
    index["validation_predictions_sha256"] = sha256_file(validation_path)
    write_json(model_root / "model_index.json", index)
    return fitted


def _load_named(config: dict[str, Any], names: list[str]) -> dict[str, FittedCalibration]:
    outputs = config["rq4"]["outputs"]
    model_root = project_path(config, outputs["models"])
    index_path = model_root / "model_index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"RQ4 model index is missing: {index_path}")
    with index_path.open("r", encoding="utf-8") as handle:
        index = json.load(handle)
    _, expected_fingerprint = _calibration_signature(config)
    if index.get("calibration_fingerprint") != expected_fingerprint:
        raise RuntimeError("RQ4 calibration configuration/source changed after fitting")
    for metadata_key, output_key in (("train_features_sha256", "train_features"), ("validation_features_sha256", "validation_features")):
        path = project_path(config, outputs[output_key])
        if not path.is_file() or index.get(metadata_key) != sha256_file(path):
            raise RuntimeError(f"RQ4 model input integrity failed: {path}")
    validation_path = model_root / "validation_predictions.parquet"
    if (
        not validation_path.is_file()
        or index.get("validation_predictions_sha256") != sha256_file(validation_path)
    ):
        raise RuntimeError(f"RQ4 validation-prediction integrity failed: {validation_path}")
    loaded = {}
    for name in names:
        path = model_root / f"{name}.joblib"
        expected_hash = index.get("methods", {}).get(name, {}).get("model_sha256")
        if not path.is_file() or expected_hash != sha256_file(path):
            raise RuntimeError(f"RQ4 model integrity failed: {path}")
        loaded[name] = joblib.load(path)
    return loaded


def load_calibrations(config: dict[str, Any]) -> dict[str, FittedCalibration]:
    return _load_named(config, list(method_feature_sets(config)))


def load_sensitivity_calibrations(config: dict[str, Any]) -> dict[str, FittedCalibration]:
    return _load_named(config, list(sensitivity_feature_sets(config)))
