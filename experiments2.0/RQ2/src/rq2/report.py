from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from adas_ovd.config import project_path
from adas_ovd.metrics import risk_coverage_curve
from adas_ovd.reproducibility import sha256_file, write_json

from .estimators import SklearnScorer, load_estimators


def _load_metrics(config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = project_path(config, config["rq2"]["outputs"]["metrics"])
    if not path.is_file():
        raise FileNotFoundError(f"RQ2 metrics are missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return path, json.load(handle)


def _save_figure(figure, root: Path, stem: str) -> list[Path]:
    paths = [root / f"{stem}.png", root / f"{stem}.pdf"]
    figure.savefig(paths[0], dpi=180, bbox_inches="tight")
    figure.savefig(paths[1], bbox_inches="tight")
    return paths


def generate_report(config: dict[str, Any]) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outputs = config["rq2"]["outputs"]
    root = project_path(config, outputs["root"])
    root.mkdir(parents=True, exist_ok=True)
    metrics_path, payload = _load_metrics(config)
    predictions_path = project_path(config, outputs["predictions"])
    if (
        not predictions_path.is_file()
        or payload["artifact_integrity"]["predictions_sha256"]
        != sha256_file(predictions_path)
    ):
        raise RuntimeError("RQ2 prediction integrity check failed before reporting")
    predictions = pd.read_parquet(predictions_path)

    main_rows = []
    for method, record in payload["methods"].items():
        row = {"method": method, "family": record["family"]}
        row.update(record["metrics"])
        for metric, interval in record["clustered_bootstrap_interval"].items():
            row[f"{metric}_ci_lower"] = interval["lower"]
            row[f"{metric}_ci_upper"] = interval["upper"]
        main_rows.append(row)
    main_table = root / "Table_RQ2_main.csv"
    pd.DataFrame(main_rows).to_csv(main_table, index=False)

    primary_rows = []
    inference = payload["primary_inference"]
    for baseline, comparisons in inference["comparisons"].items():
        for metric, record in comparisons.items():
            primary_rows.append(
                {
                    "fusion": inference["method"],
                    "baseline": baseline,
                    "metric": metric,
                    **record,
                }
            )
    primary_table = root / "Table_RQ2_primary_inference.csv"
    pd.DataFrame(primary_rows).to_csv(primary_table, index=False)

    mc_rows = []
    for method, record in payload["mc_pass_sensitivity"].items():
        mc_rows.append({"method": method, "mc_passes": record["mc_passes"], **record["metrics"]})
    mc_table = root / "Table_RQ2_mc_pass_sensitivity.csv"
    pd.DataFrame(mc_rows).to_csv(mc_table, index=False)

    threshold_rows = []
    for threshold, record in payload["score_threshold_sensitivity"].items():
        for method, method_metrics in record["methods"].items():
            threshold_rows.append(
                {
                    "score_threshold": threshold,
                    "rows": record["rows"],
                    "method": method,
                    **method_metrics,
                }
            )
    threshold_table = root / "Table_RQ2_threshold_sensitivity.csv"
    pd.DataFrame(threshold_rows).to_csv(threshold_table, index=False)

    subgroup_rows = []
    for attribute, values in payload["subgroups"].items():
        for value, record in values.items():
            if not record["methods"]:
                subgroup_rows.append(
                    {
                        "attribute": attribute,
                        "value": value,
                        "rows": record["rows"],
                        "error_prevalence": record["error_prevalence"],
                        "method": "not_estimable",
                    }
                )
            for method, method_metrics in record["methods"].items():
                subgroup_rows.append(
                    {
                        "attribute": attribute,
                        "value": value,
                        "rows": record["rows"],
                        "error_prevalence": record["error_prevalence"],
                        "method": method,
                        **method_metrics,
                    }
                )
    subgroup_table = root / "Table_RQ2_subgroups.csv"
    pd.DataFrame(subgroup_rows).to_csv(subgroup_table, index=False)

    cost_table = root / "Table_RQ2_computational_cost.csv"
    cost_values = {
        key: value
        for key, value in payload["computational_cost"].items()
        if key not in {"stochastic_modules", "environment"}
    }
    pd.DataFrame([cost_values]).to_csv(cost_table, index=False)
    detector_table = root / "Table_RQ2_detector_performance.csv"
    pd.DataFrame([payload["detector_performance"]]).to_csv(detector_table, index=False)

    parameter_rows = []
    for method, estimator in load_estimators(config).items():
        if not isinstance(estimator.scorer, SklearnScorer):
            continue
        pipeline = estimator.scorer.estimator
        classifier = pipeline.named_steps["classifier"]
        imputer = pipeline.named_steps.get("imputer")
        names = (
            list(imputer.get_feature_names_out(estimator.features))
            if imputer is not None
            else list(estimator.features)
        )
        if hasattr(classifier, "coef_"):
            values = classifier.coef_[0]
            kind = "standardized_logistic_coefficient"
        elif hasattr(classifier, "feature_importances_"):
            values = classifier.feature_importances_
            kind = "random_forest_importance"
        else:
            continue
        parameter_rows.extend(
            {
                "method": method,
                "parameter": name,
                "value": float(value),
                "parameter_type": kind,
                "selected_regularization": estimator.selected_regularization,
            }
            for name, value in zip(names, values, strict=True)
        )
    parameter_table = root / "Table_RQ2_model_parameters.csv"
    pd.DataFrame(parameter_rows).to_csv(parameter_table, index=False)

    selected = [
        "confidence",
        "learned_deterministic",
        "learned_stochastic",
        "equal_fixed_fusion",
        "learned_fusion",
    ]
    labels = predictions["is_error"].to_numpy(dtype=np.int64)
    figure, axis = plt.subplots(figsize=(7.2, 5.0))
    for method in selected:
        curve = risk_coverage_curve(
            labels, predictions[f"rank_{method}"].to_numpy(dtype=np.float64)
        )
        axis.plot(
            curve.coverage,
            curve.risk,
            label=f"{method} (AURC={curve.aurc:.3f})",
        )
    axis.set_xlabel("Coverage")
    axis.set_ylabel("Detection error risk")
    axis.set_title("RQ2 risk–coverage comparison")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    risk_paths = _save_figure(figure, root, "Fig_RQ2_risk_coverage")
    plt.close(figure)

    model_root = project_path(config, outputs["models"])
    validation_path = model_root / "validation_predictions.parquet"
    validation = pd.read_parquet(validation_path)
    figure, axis = plt.subplots(figsize=(6.0, 5.2))
    colors = np.where(validation["is_error"].to_numpy(dtype=int) == 1, "#d95f02", "#1b9e77")
    axis.scatter(
        validation["rank_deterministic_fixed"],
        validation["rank_stochastic_fixed"],
        c=colors,
        s=12,
        alpha=0.45,
        linewidths=0,
    )
    axis.set_xlabel("Fixed deterministic score")
    axis.set_ylabel("Fixed stochastic score")
    axis.set_title("Validation complementarity (green=correct, orange=error)")
    axis.grid(alpha=0.2)
    complementarity_paths = _save_figure(
        figure, root, "Fig_RQ2_validation_complementarity"
    )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.4, 5.0))
    bins = np.linspace(0.0, 1.0, 11)
    for method in ("learned_deterministic", "learned_stochastic", "learned_fusion"):
        probability = predictions[f"prob_error_{method}"].to_numpy(dtype=np.float64)
        bin_ids = np.clip(np.digitize(probability, bins[1:-1]), 0, len(bins) - 2)
        x_values, y_values = [], []
        for bin_id in range(len(bins) - 1):
            mask = bin_ids == bin_id
            if mask.any():
                x_values.append(float(probability[mask].mean()))
                y_values.append(float(labels[mask].mean()))
        axis.plot(x_values, y_values, marker="o", label=method)
    axis.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    axis.set_xlabel("Predicted error probability")
    axis.set_ylabel("Observed error frequency")
    axis.set_title("RQ2 reliability diagram")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    calibration_paths = _save_figure(figure, root, "Fig_RQ2_reliability")
    plt.close(figure)

    captions = root / "figure_captions.md"
    captions.write_text(
        "# RQ2 figure captions\n\n"
        "These figures are diagnostic-only when generated with `rq2_mini.yaml`.\n\n"
        "- **Fig_RQ2_risk_coverage.** Detection error risk as progressively "
        "more uncertain detections are retained. Lower curves are better.\n"
        "- **Fig_RQ2_validation_complementarity.** Train-normalized fixed "
        "deterministic and stochastic scores on validation detections; this "
        "plot is descriptive and does not select a test-time feature direction.\n"
        "- **Fig_RQ2_reliability.** Calibrated predicted error probability "
        "against observed error frequency.\n",
        encoding="utf-8",
    )

    paths = {
        "main_table": main_table,
        "primary_inference_table": primary_table,
        "mc_pass_table": mc_table,
        "threshold_table": threshold_table,
        "subgroup_table": subgroup_table,
        "cost_table": cost_table,
        "detector_table": detector_table,
        "model_parameter_table": parameter_table,
        "risk_coverage_png": risk_paths[0],
        "risk_coverage_pdf": risk_paths[1],
        "complementarity_png": complementarity_paths[0],
        "complementarity_pdf": complementarity_paths[1],
        "reliability_png": calibration_paths[0],
        "reliability_pdf": calibration_paths[1],
        "captions": captions,
    }
    report_manifest = root / "report_manifest.json"
    write_json(
        report_manifest,
        {
            "schema_version": 1,
            "evidence_status": payload["evidence_status"],
            "metrics_sha256": sha256_file(metrics_path),
            "predictions_sha256": sha256_file(predictions_path),
            "artifacts": {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in paths.items()
            },
        },
    )
    return {**paths, "report_manifest": report_manifest}
