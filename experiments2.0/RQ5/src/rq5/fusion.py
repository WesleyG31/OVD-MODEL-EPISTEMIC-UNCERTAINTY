from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import IsotonicRegression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from adas_ovd.config import project_path
from adas_ovd.reproducibility import (
    sha256_file,
    source_tree_sha256,
    stable_fingerprint,
    write_json,
)

from .extraction import read_validated_features, uncertainty_feature_names
from .features import finite_feature_audit


TARGET_OR_DESCRIPTOR_COLUMNS = {
    "is_true_positive",
    "is_error",
    "matched_iou",
    "matched_ground_truth_index",
    "false_negatives_image",
    "image_id",
    "file_name",
    "sequence_id",
    "timeofday",
    "weather",
    "scene",
    "category_name",
    "object_size",
    "criticality_class_severity",
    "criticality_bottomness",
    "criticality_centrality",
    "criticality_geometry_factor",
    "criticality_weight",
    "criticality_tier",
}


@dataclass
class BinaryComponent:
    name: str
    features: list[str]
    estimator: Pipeline
    selected_regularization: float
    selection_auroc: float

    def raw_probability(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(
            self.estimator.predict_proba(frame[self.features])[:, 1],
            dtype=np.float64,
        )

    @property
    def coefficient_count(self) -> int:
        classifier = self.estimator.named_steps["classifier"]
        return int(classifier.coef_.size + classifier.intercept_.size)


@dataclass
class DecisionPolicy:
    method: str
    family: str
    mc_passes: int
    components: dict[str, BinaryComponent]
    calibrators: dict[str, IsotonicRegression | None]
    confidence_weight: float
    uncertainty_weight: float
    operating_threshold: float
    training_error_prevalence: float

    @staticmethod
    def _apply_calibrator(
        calibrator: IsotonicRegression | None, values: np.ndarray
    ) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        calibrated = values if calibrator is None else calibrator.predict(values)
        return np.clip(calibrated, 1e-7, 1.0 - 1e-7)

    def _confidence_probability(self, frame: pd.DataFrame) -> np.ndarray:
        raw = 1.0 - frame["score"].to_numpy(dtype=np.float64)
        if self.method == "raw_confidence":
            return np.clip(raw, 1e-7, 1.0 - 1e-7)
        return self._apply_calibrator(self.calibrators.get("confidence"), raw)

    def error_probability(self, frame: pd.DataFrame) -> np.ndarray:
        if self.method == "raw_confidence":
            return self._confidence_probability(frame)
        if self.method == "calibrated_confidence":
            return self._confidence_probability(frame)
        if self.method == "criticality_only":
            return np.full(len(frame), self.training_error_prevalence, dtype=np.float64)
        if self.method == "flat_joint":
            raw = self.components["flat"].raw_probability(frame)
            return self._apply_calibrator(self.calibrators.get("flat"), raw)
        uncertainty_raw = self.components["uncertainty"].raw_probability(frame)
        uncertainty = self._apply_calibrator(
            self.calibrators.get("uncertainty"), uncertainty_raw
        )
        if self.method == "uncertainty_only":
            return uncertainty
        confidence = self._confidence_probability(frame)
        late = self.confidence_weight * confidence + self.uncertainty_weight * uncertainty
        return self._apply_calibrator(self.calibrators.get("late"), late)

    def decision_risk(self, frame: pd.DataFrame) -> np.ndarray:
        if self.method == "criticality_only":
            return frame["criticality_weight"].to_numpy(dtype=np.float64)
        probability = self.error_probability(frame)
        if self.method == "late_fusion_unweighted":
            return probability
        return probability * frame["criticality_weight"].to_numpy(dtype=np.float64)

    def accept(self, frame: pd.DataFrame) -> np.ndarray:
        return self.decision_risk(frame) <= float(self.operating_threshold)

    @property
    def coefficient_count(self) -> int:
        return int(
            sum(component.coefficient_count for component in self.components.values())
        )


def model_feature_sets(
    config: dict[str, Any], mc_passes: int | None = None
) -> dict[str, list[str]]:
    uncertainty = uncertainty_feature_names(config, mc_passes)
    definitions = {
        "uncertainty": uncertainty,
        "flat": ["confidence_uncertainty", *uncertainty],
    }
    validate_no_target_leakage(definitions)
    return definitions


def validate_no_target_leakage(definitions: dict[str, list[str]]) -> None:
    for name, features in definitions.items():
        leaked = sorted(
            feature
            for feature in features
            if feature in TARGET_OR_DESCRIPTOR_COLUMNS
            or feature.startswith("matched_")
            or feature.startswith("criticality_")
        )
        if leaked:
            raise RuntimeError(f"RQ5 target/decision leakage in {name}: {leaked}")
        if len(features) != len(set(features)):
            raise RuntimeError(f"RQ5 definition {name} repeats a feature")


def _logistic_pipeline(regularization: float, seed: int) -> Pipeline:
    return Pipeline(
        [
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
    fractions: dict[str, float],
    seed: int,
    minimum_samples: dict[str, int],
) -> dict[str, pd.DataFrame]:
    roles = ("selection", "component_calibration", "policy_calibration")
    if set(fractions) != set(roles):
        raise ValueError("RQ5 validation fractions must define all three roles")
    values = np.asarray([float(fractions[name]) for name in roles])
    if np.any(values <= 0.0) or not np.isclose(values.sum(), 1.0):
        raise ValueError("RQ5 validation fractions must be positive and sum to one")
    if group_column not in frame:
        raise KeyError(f"RQ5 validation group column is missing: {group_column}")
    groups = np.asarray(sorted(frame[group_column].astype(str).unique()))
    if len(groups) < 3:
        raise ValueError("RQ5 validation requires at least three source groups")
    shuffled = np.random.default_rng(int(seed)).permutation(groups)
    selection_count = max(1, int(round(len(groups) * values[0])))
    component_count = max(1, int(round(len(groups) * values[1])))
    if selection_count + component_count >= len(groups):
        component_count = max(1, len(groups) - selection_count - 1)
    policy_count = len(groups) - selection_count - component_count
    if policy_count < 1:
        raise ValueError("RQ5 validation group allocation left no policy fold")
    assigned = {
        "selection": set(shuffled[:selection_count]),
        "component_calibration": set(
            shuffled[selection_count : selection_count + component_count]
        ),
        "policy_calibration": set(shuffled[selection_count + component_count :]),
    }
    result = {
        role: frame.loc[frame[group_column].astype(str).isin(assigned[role])].copy()
        for role in roles
    }
    for role in roles:
        if len(result[role]) < int(minimum_samples[role]):
            raise ValueError(f"RQ5 {role} detections are below the frozen minimum")
    for left_index, left in enumerate(roles):
        for right in roles[left_index + 1 :]:
            if assigned[left] & assigned[right]:
                raise RuntimeError("RQ5 validation source-group leakage")
    return result


def _selection_auroc(labels: pd.Series, probability: np.ndarray) -> float:
    return (
        float(roc_auc_score(labels.astype(int), probability))
        if labels.nunique() == 2
        else 0.5
    )


def _fit_component(
    config: dict[str, Any],
    train: pd.DataFrame,
    selection: pd.DataFrame,
    *,
    name: str,
    features: list[str],
) -> BinaryComponent:
    if train["is_error"].nunique() != 2:
        raise ValueError("RQ5 training data requires correct and erroneous detections")
    seed = int(config["project"]["seed"])
    candidates: list[tuple[float, float]] = []
    for regularization in config["rq5"]["fusion"]["regularization_grid"]:
        estimator = _logistic_pipeline(float(regularization), seed)
        estimator.fit(train[features], train["is_error"].astype(int))
        probability = estimator.predict_proba(selection[features])[:, 1]
        candidates.append(
            (
                _selection_auroc(selection["is_error"], probability),
                float(regularization),
            )
        )
    candidates.sort(key=lambda item: (-item[0], item[1]))
    selection_auroc, selected_regularization = candidates[0]
    final = _logistic_pipeline(selected_regularization, seed)
    final.fit(train[features], train["is_error"].astype(int))
    return BinaryComponent(
        name,
        features,
        final,
        selected_regularization,
        selection_auroc,
    )


def _fit_isotonic(
    values: np.ndarray, labels: pd.Series
) -> IsotonicRegression | None:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if finite.sum() < 2 or labels.iloc[np.flatnonzero(finite)].nunique() < 2:
        return None
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(values[finite], labels.iloc[np.flatnonzero(finite)].astype(int))
    return calibrator


def _apply_isotonic(
    calibrator: IsotonicRegression | None, values: np.ndarray
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = values if calibrator is None else calibrator.predict(values)
    return np.clip(result, 1e-7, 1.0 - 1e-7)


def weighted_selective_risk(
    labels: np.ndarray, weights: np.ndarray, accepted: np.ndarray
) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    accepted = np.asarray(accepted, dtype=bool)
    denominator = float(weights[accepted].sum())
    if denominator <= 0.0:
        return float("nan")
    return float(np.sum(weights[accepted] * labels[accepted]) / denominator)


def select_operating_threshold(
    labels: np.ndarray,
    decision_risk: np.ndarray,
    weights: np.ndarray,
    target_risk: float,
) -> tuple[float, dict[str, float]]:
    labels = np.asarray(labels, dtype=np.float64)
    risk = np.asarray(decision_risk, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    finite = np.isfinite(risk) & np.isfinite(weights) & (weights > 0.0)
    labels, risk, weights = labels[finite], risk[finite], weights[finite]
    if len(labels) == 0:
        return float("-inf"), {
            "coverage": 0.0,
            "criticality_mass_coverage": 0.0,
            "weighted_risk": float("nan"),
        }
    order = np.argsort(risk, kind="stable")
    ordered_risk = risk[order]
    ordered_labels = labels[order]
    ordered_weights = weights[order]
    cumulative_weight = np.cumsum(ordered_weights)
    cumulative_error = np.cumsum(ordered_weights * ordered_labels)
    cumulative_risk = cumulative_error / cumulative_weight
    tie_end = np.r_[ordered_risk[1:] != ordered_risk[:-1], True]
    feasible = np.flatnonzero(tie_end & (cumulative_risk <= float(target_risk)))
    if len(feasible) == 0:
        return float("-inf"), {
            "coverage": 0.0,
            "criticality_mass_coverage": 0.0,
            "weighted_risk": float("nan"),
        }
    index = int(feasible[-1])
    return float(ordered_risk[index]), {
        "coverage": float((index + 1) / len(labels)),
        "criticality_mass_coverage": float(
            cumulative_weight[index] / cumulative_weight[-1]
        ),
        "weighted_risk": float(cumulative_risk[index]),
    }


def _fusion_signature(config: dict[str, Any]) -> tuple[str, str]:
    paths = ("RQ5/src/rq5/fusion.py", "RQ5/src/rq5/features.py")
    source_sha256 = source_tree_sha256(config["_meta"]["project_root"], paths)
    fingerprint = stable_fingerprint(
        {
            "schema_version": 1,
            "source_tree_sha256": source_sha256,
            "feature_groups": config["rq5"]["feature_groups"],
            "fusion": config["rq5"]["fusion"],
            "mc_prefixes": config["rq5"]["extraction"]["mc_sensitivity_passes"],
        }
    )
    return source_sha256, fingerprint


def fit_policies(config: dict[str, Any]) -> dict[str, DecisionPolicy]:
    outputs = config["rq5"]["outputs"]
    train_all, train_metadata = read_validated_features(
        config, project_path(config, outputs["train_features"])
    )
    validation_all, validation_metadata = read_validated_features(
        config, project_path(config, outputs["validation_features"])
    )
    settings = config["rq5"]["fusion"]
    threshold = float(settings["training_score_threshold"])
    train = train_all.loc[train_all["score"] >= threshold].copy()
    validation = validation_all.loc[validation_all["score"] >= threshold].copy()
    if train.empty or validation.empty:
        raise ValueError("RQ5 operational train/validation features must be non-empty")
    if set(train["sequence_id"].astype(str)) & set(
        validation["sequence_id"].astype(str)
    ):
        raise RuntimeError("RQ5 train/validation source-group leakage")

    counts = sorted(
        int(value) for value in config["rq5"]["extraction"]["mc_sensitivity_passes"]
    )
    definitions = {count: model_feature_sets(config, count) for count in counts}
    for feature_sets in definitions.values():
        for features in feature_sets.values():
            for feature in features:
                if feature not in train or feature not in validation:
                    raise KeyError(f"RQ5 fitting feature is missing: {feature}")
                finite_feature_audit(
                    pd.concat([train[feature], validation[feature]]).to_numpy()
                )

    folds = split_validation_groups(
        validation,
        group_column=str(settings["validation_group_column"]),
        fractions=settings["validation_fractions"],
        seed=int(config["project"]["seed"]),
        minimum_samples={
            "selection": int(settings["minimum_selection_samples"]),
            "component_calibration": int(
                settings["minimum_component_calibration_samples"]
            ),
            "policy_calibration": int(settings["minimum_policy_calibration_samples"]),
        },
    )
    selection = folds["selection"]
    component_calibration = folds["component_calibration"]
    policy_calibration = folds["policy_calibration"]

    components_by_count: dict[int, dict[str, BinaryComponent]] = {}
    for count in counts:
        components_by_count[count] = {
            name: _fit_component(
                config,
                train,
                selection,
                name=f"{name}_mc{count:02d}",
                features=features,
            )
            for name, features in definitions[count].items()
        }

    confidence_raw_component = 1.0 - component_calibration["score"].to_numpy(
        dtype=np.float64
    )
    confidence_calibrator = _fit_isotonic(
        confidence_raw_component, component_calibration["is_error"]
    )
    confidence_policy = _apply_isotonic(
        confidence_calibrator,
        1.0 - policy_calibration["score"].to_numpy(dtype=np.float64),
    )
    confidence_weight = float(settings["confidence_weight"])
    uncertainty_weight = float(settings["uncertainty_weight"])
    if not np.isclose(confidence_weight + uncertainty_weight, 1.0):
        raise ValueError("RQ5 late-fusion weights must sum to one")

    calibrators_by_count: dict[int, dict[str, IsotonicRegression | None]] = {}
    for count, components in components_by_count.items():
        uncertainty_calibrator = _fit_isotonic(
            components["uncertainty"].raw_probability(component_calibration),
            component_calibration["is_error"],
        )
        flat_calibrator = _fit_isotonic(
            components["flat"].raw_probability(component_calibration),
            component_calibration["is_error"],
        )
        uncertainty_policy = _apply_isotonic(
            uncertainty_calibrator,
            components["uncertainty"].raw_probability(policy_calibration),
        )
        late_policy_raw = (
            confidence_weight * confidence_policy
            + uncertainty_weight * uncertainty_policy
        )
        late_calibrator = _fit_isotonic(
            late_policy_raw, policy_calibration["is_error"]
        )
        calibrators_by_count[count] = {
            "confidence": confidence_calibrator,
            "uncertainty": uncertainty_calibrator,
            "flat": flat_calibrator,
            "late": late_calibrator,
        }

    operating_count = int(config["rq5"]["extraction"]["mc_operating_passes"])
    if operating_count not in components_by_count:
        raise RuntimeError("RQ5 operating MC prefix was not fitted")
    prevalence = float(train["is_error"].mean())
    policies: dict[str, DecisionPolicy] = {}

    def add_policy(
        method: str,
        family: str,
        count: int,
        components: dict[str, BinaryComponent],
        calibrators: dict[str, IsotonicRegression | None],
    ) -> None:
        provisional = DecisionPolicy(
            method,
            family,
            count,
            components,
            calibrators,
            confidence_weight,
            uncertainty_weight,
            float("inf"),
            prevalence,
        )
        selected_threshold, _ = select_operating_threshold(
            policy_calibration["is_error"].to_numpy(),
            provisional.decision_risk(policy_calibration),
            policy_calibration["criticality_weight"].to_numpy(),
            float(settings["target_weighted_risk"]),
        )
        provisional.operating_threshold = selected_threshold
        policies[method] = provisional

    primary_components = components_by_count[operating_count]
    primary_calibrators = calibrators_by_count[operating_count]
    add_policy(
        "raw_confidence",
        "confidence_control",
        0,
        {},
        {"confidence": None},
    )
    add_policy(
        "calibrated_confidence",
        "confidence_control",
        0,
        {},
        {"confidence": confidence_calibrator},
    )
    add_policy(
        "uncertainty_only",
        "uncertainty_ablation",
        operating_count,
        {"uncertainty": primary_components["uncertainty"]},
        primary_calibrators,
    )
    add_policy("criticality_only", "criticality_control", 0, {}, {})
    add_policy(
        "late_fusion_unweighted",
        "criticality_ablation",
        operating_count,
        {"uncertainty": primary_components["uncertainty"]},
        primary_calibrators,
    )
    add_policy(
        "risk_aware_fusion",
        "primary_risk_aware",
        operating_count,
        {"uncertainty": primary_components["uncertainty"]},
        primary_calibrators,
    )
    add_policy(
        "flat_joint",
        "flat_capacity_control",
        operating_count,
        {"flat": primary_components["flat"]},
        primary_calibrators,
    )
    for count in counts:
        if count == operating_count:
            continue
        add_policy(
            f"risk_aware_fusion_mc{count:02d}",
            "mc_pass_sensitivity",
            count,
            {"uncertainty": components_by_count[count]["uncertainty"]},
            calibrators_by_count[count],
        )

    requested = set(settings["methods"])
    if requested != {name for name in policies if not name.startswith("risk_aware_fusion_mc")}:
        raise RuntimeError("RQ5 configured methods differ from the frozen implementation")

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
        "fold_rows": {role: len(frame) for role, frame in folds.items()},
        "train_features_sha256": train_metadata["features_sha256"],
        "validation_features_sha256": validation_metadata["features_sha256"],
        "group_audit": {
            "column": settings["validation_group_column"],
            "roles": {
                role: stable_fingerprint(
                    sorted(frame[settings["validation_group_column"]].astype(str).unique())
                )
                for role, frame in folds.items()
            },
            "pairwise_overlap_count": 0,
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
            "criticality_weight",
        ]
    ].copy()
    role_by_group = {
        str(group): role
        for role, frame in folds.items()
        for group in frame[settings["validation_group_column"]].astype(str).unique()
    }
    validation_predictions["validation_role"] = validation[
        settings["validation_group_column"]
    ].astype(str).map(role_by_group)
    for method, policy in policies.items():
        destination = model_root / f"{method}.joblib"
        temporary = destination.with_suffix(".joblib.tmp")
        joblib.dump(policy, temporary)
        temporary.replace(destination)
        probability = policy.error_probability(validation)
        risk = policy.decision_risk(validation)
        validation_predictions[f"prob_error_{method}"] = probability
        validation_predictions[f"decision_risk_{method}"] = risk
        validation_predictions[f"accept_{method}"] = risk <= policy.operating_threshold
        component_metadata = {
            name: {
                "features": component.features,
                "selected_regularization": component.selected_regularization,
                "selection_auroc": component.selection_auroc,
                "coefficient_count": component.coefficient_count,
            }
            for name, component in policy.components.items()
        }
        index["methods"][method] = {
            "family": policy.family,
            "mc_passes": policy.mc_passes,
            "operating_threshold": policy.operating_threshold,
            "coefficient_count": policy.coefficient_count,
            "components": component_metadata,
            "model_sha256": sha256_file(destination),
        }
    validation_path = model_root / "validation_predictions.parquet"
    temporary_validation = validation_path.with_suffix(".parquet.tmp")
    validation_predictions.to_parquet(temporary_validation, index=False)
    temporary_validation.replace(validation_path)
    index["validation_predictions_sha256"] = sha256_file(validation_path)
    write_json(model_root / "model_index.json", index)
    return policies


def load_policies(config: dict[str, Any]) -> dict[str, DecisionPolicy]:
    outputs = config["rq5"]["outputs"]
    model_root = project_path(config, outputs["models"])
    index_path = model_root / "model_index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"RQ5 model index is missing: {index_path}")
    with index_path.open("r", encoding="utf-8") as handle:
        index = json.load(handle)
    _, expected_fingerprint = _fusion_signature(config)
    if index.get("fusion_fingerprint") != expected_fingerprint:
        raise RuntimeError("RQ5 fusion configuration/source changed after fitting")
    for metadata_key, output_key in (
        ("train_features_sha256", "train_features"),
        ("validation_features_sha256", "validation_features"),
    ):
        path = project_path(config, outputs[output_key])
        if not path.is_file() or index.get(metadata_key) != sha256_file(path):
            raise RuntimeError(f"RQ5 model input integrity failed: {path}")
    loaded: dict[str, DecisionPolicy] = {}
    for name, metadata in index.get("methods", {}).items():
        path = model_root / f"{name}.joblib"
        if not path.is_file() or metadata.get("model_sha256") != sha256_file(path):
            raise RuntimeError(f"RQ5 model integrity failed: {path}")
        loaded[name] = joblib.load(path)
    return loaded
