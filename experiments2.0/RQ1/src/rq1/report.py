from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from adas_ovd.config import project_path
from adas_ovd.metrics import risk_coverage_curve
from adas_ovd.reproducibility import sha256_file, write_json

from .fusion import load_fusions
from .extraction import read_validated_features


def generate_report(config: dict[str, Any]) -> dict[str, Path]:
    outputs = config["rq1"]["outputs"]
    output_root = project_path(config, outputs["root"])
    output_root.mkdir(parents=True, exist_ok=True)
    with project_path(config, outputs["metrics"]).open(
        "r", encoding="utf-8"
    ) as handle:
        metrics = json.load(handle)
    predictions = pd.read_parquet(
        project_path(config, outputs["predictions"])
    )

    records = []
    for method, result in metrics["methods"].items():
        values = result["metrics"]
        intervals = result["clustered_bootstrap_interval"]
        records.append(
            {
                "method": method,
                "AUROC": values["auroc"],
                "AUROC_CI_low": intervals["auroc"]["lower"],
                "AUROC_CI_high": intervals["auroc"]["upper"],
                "AUPRC": values["auprc"],
                "AURC": values["aurc"],
                "AURC_CI_low": intervals["aurc"]["lower"],
                "AURC_CI_high": intervals["aurc"]["upper"],
                "ECE": values["ece"],
                "ECE_CI_low": intervals["ece"]["lower"],
                "ECE_CI_high": intervals["ece"]["upper"],
                "Brier": values["brier"],
                "Brier_CI_low": intervals["brier"]["lower"],
                "Brier_CI_high": intervals["brier"]["upper"],
                "NLL": values["nll"],
                "NLL_CI_low": intervals["nll"]["lower"],
                "NLL_CI_high": intervals["nll"]["upper"],
            }
        )
    table = pd.DataFrame(records).sort_values("AUROC", ascending=False)
    table_path = output_root / "Table_RQ1_main.csv"
    table.to_csv(table_path, index=False)

    figure_auroc = output_root / "Fig_RQ1_auroc"
    figure_risk = output_root / "Fig_RQ1_risk_coverage"
    figure_reliability = output_root / "Fig_RQ1_reliability"
    _plot_auroc(table, figure_auroc)
    _plot_risk_coverage(predictions, table["method"].tolist(), figure_risk)
    _plot_reliability(metrics, figure_reliability)
    coefficient_path = _write_coefficients(config, output_root)
    threshold_path = _write_threshold_sensitivity(metrics, output_root)
    subgroup_path = _write_subgroups(metrics, output_root)
    mc_sensitivity_path = _write_mc_sensitivity(metrics, output_root)
    image_safety_path = _write_image_safety(metrics, output_root)
    detector_path = _write_detector_performance(metrics, output_root)
    primary_path = _write_primary_inference(metrics, output_root)
    calibration_path = _write_calibration(metrics, output_root)
    cost_path = _write_computational_cost(metrics, output_root)
    robustness_path = _write_robustness(config, output_root)
    complementarity_path = _write_feature_complementarity(config, output_root)
    captions = output_root / "figure_captions.md"
    captions.write_text(
        (
            "# RQ1 figure captions\n\n"
            "**Figure RQ1.1.** Confirmatory error-detection AUROC on the untouched "
            "BDD100K evaluation split. Error bars are sequence-clustered 95% "
            "bootstrap confidence intervals.\n\n"
            "**Figure RQ1.2.** Selective risk as detection coverage increases. "
            "Detections are retained from lowest to highest predicted uncertainty; "
            "lower curves indicate safer ranking.\n"
            "\n**Figure RQ1.3.** Reliability diagram for the frozen primary "
            "fusion and confidence baseline. Perfect calibration lies on the "
            "diagonal.\n"
        ),
        encoding="utf-8",
    )
    paths = {
        "table": table_path,
        "auroc_png": figure_auroc.with_suffix(".png"),
        "auroc_pdf": figure_auroc.with_suffix(".pdf"),
        "risk_png": figure_risk.with_suffix(".png"),
        "risk_pdf": figure_risk.with_suffix(".pdf"),
        "reliability_png": figure_reliability.with_suffix(".png"),
        "reliability_pdf": figure_reliability.with_suffix(".pdf"),
        "coefficients": coefficient_path,
        "threshold_sensitivity": threshold_path,
        "subgroups": subgroup_path,
        "mc_pass_sensitivity": mc_sensitivity_path,
        "image_safety": image_safety_path,
        "detector_performance": detector_path,
        "primary_inference": primary_path,
        "calibration": calibration_path,
        "computational_cost": cost_path,
        "robustness": robustness_path,
        "feature_complementarity": complementarity_path,
        "captions": captions,
    }
    manifest_path = output_root / "report_manifest.json"
    write_json(
        manifest_path,
        {
            "schema_version": 1,
            "evidence_tier": metrics["evidence_tier"],
            "rq1_answer_supported": metrics["primary_inference"][
                "rq1_answer_supported"
            ],
            "artifacts": {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in paths.items()
            },
        },
    )
    paths["manifest"] = manifest_path
    return paths


def _plot_auroc(table: pd.DataFrame, destination: Path) -> None:
    ordered = table.sort_values("AUROC", ascending=True)
    lower = ordered["AUROC"] - ordered["AUROC_CI_low"]
    upper = ordered["AUROC_CI_high"] - ordered["AUROC"]
    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    axis.errorbar(
        ordered["AUROC"],
        ordered["method"],
        xerr=[lower, upper],
        fmt="o",
        capsize=3,
        color="#1f5a94",
    )
    axis.axvline(0.5, color="0.5", linestyle="--", linewidth=1)
    axis.set_xlabel("Error-detection AUROC")
    axis.set_ylabel("")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination.with_suffix(".png"), dpi=300)
    figure.savefig(destination.with_suffix(".pdf"))
    plt.close(figure)


def _plot_risk_coverage(
    predictions: pd.DataFrame, methods: list[str], destination: Path
) -> None:
    labels = predictions["is_error"].to_numpy()
    figure, axis = plt.subplots(figsize=(7.5, 5.0))
    for method in methods:
        curve = risk_coverage_curve(
            labels, predictions[f"rank_{method}"].to_numpy()
        )
        axis.plot(curve.coverage, curve.risk, label=method, linewidth=1.5)
    axis.set_xlabel("Detection coverage")
    axis.set_ylabel("Selective risk")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(bottom=0.0)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(destination.with_suffix(".png"), dpi=300)
    figure.savefig(destination.with_suffix(".pdf"))
    plt.close(figure)


def _plot_reliability(metrics: dict[str, Any], destination: Path) -> None:
    inference = metrics["primary_inference"]
    methods = [inference["baseline"], inference["method"]]
    figure, axis = plt.subplots(figsize=(6.0, 5.0))
    for method in methods:
        records = metrics["methods"][method]["reliability_bins"]
        axis.plot(
            [record["mean_probability"] for record in records],
            [record["observed_error_rate"] for record in records],
            marker="o",
            label=method,
        )
    axis.plot([0, 1], [0, 1], "--", color="black", linewidth=1)
    axis.set_xlabel("Predicted error probability")
    axis.set_ylabel("Observed error frequency")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination.with_suffix(".png"), dpi=300)
    figure.savefig(destination.with_suffix(".pdf"))
    plt.close(figure)


def _write_coefficients(config: dict[str, Any], output_root: Path) -> Path:
    records = []
    for method, fitted in load_fusions(config).items():
        imputer = fitted.estimator.named_steps["imputer"]
        transformed_names = imputer.get_feature_names_out(fitted.features)
        if fitted.family == "logistic":
            parameter_name = "coefficient_standardized"
            parameters = fitted.estimator.named_steps["logistic"].coef_[0]
        elif fitted.family == "random_forest":
            parameter_name = "feature_importance"
            parameters = fitted.estimator.named_steps[
                "random_forest"
            ].feature_importances_
        else:
            raise ValueError(f"Unknown fitted fusion family: {fitted.family}")
        records.extend(
            {
                "method": method,
                "family": fitted.family,
                "transformed_feature": str(feature),
                "parameter": parameter_name,
                "value": float(parameter),
            }
            for feature, parameter in zip(
                transformed_names, parameters, strict=True
            )
        )
    path = output_root / "Table_RQ1_fusion_coefficients.csv"
    pd.DataFrame(records).to_csv(path, index=False)
    return path


def _write_primary_inference(
    metrics: dict[str, Any], output_root: Path
) -> Path:
    inference = metrics["primary_inference"]
    records = [
        {
            "method": inference["method"],
            "baseline": inference["baseline"],
            "metric": metric,
            **values,
            "evidence_tier": metrics["evidence_tier"],
            "rq1_answer_supported": inference["rq1_answer_supported"],
        }
        for metric, values in inference["metrics"].items()
    ]
    calibration = inference["calibration_noninferiority"]
    records.append(
        {
            "method": inference["method"],
            "baseline": inference["baseline"],
            "metric": "brier_noninferiority",
            "improvement": calibration["improvement"],
            "lower": calibration["lower"],
            "upper": calibration["upper"],
            "reject_no_improvement": calibration["passed"],
            "evidence_tier": metrics["evidence_tier"],
            "rq1_answer_supported": inference["rq1_answer_supported"],
        }
    )
    path = output_root / "Table_RQ1_primary_inference.csv"
    pd.DataFrame(records).to_csv(path, index=False)
    return path


def _write_calibration(metrics: dict[str, Any], output_root: Path) -> Path:
    records = []
    for method, result in metrics["methods"].items():
        intervals = result["clustered_bootstrap_interval"]
        for bins, ece in result["ece_bin_sensitivity"].items():
            records.append(
                {
                    "method": method,
                    "ece_bins": int(bins),
                    "ECE": ece,
                    "Brier": result["metrics"]["brier"],
                    "Brier_CI_low": intervals["brier"]["lower"],
                    "Brier_CI_high": intervals["brier"]["upper"],
                    "NLL": result["metrics"]["nll"],
                    "NLL_CI_low": intervals["nll"]["lower"],
                    "NLL_CI_high": intervals["nll"]["upper"],
                }
            )
    path = output_root / "Table_RQ1_calibration.csv"
    pd.DataFrame(records).to_csv(path, index=False)
    return path


def _write_computational_cost(
    metrics: dict[str, Any], output_root: Path
) -> Path:
    path = output_root / "Table_RQ1_computational_cost.csv"
    pd.DataFrame([metrics["computational_cost"]]).to_csv(path, index=False)
    return path


def _write_robustness(config: dict[str, Any], output_root: Path) -> Path:
    source = project_path(
        config, config["rq1"]["outputs"]["robustness"]
    ) / "robustness.json"
    if not source.is_file():
        raise FileNotFoundError(
            f"RQ1 robustness artifact is missing: {source}. Run robustness first."
        )
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    current_model = project_path(
        config, config["rq1"]["outputs"]["models"]
    ) / f"{config['rq1']['fusion']['primary_method']}.joblib"
    if payload.get("primary_model_sha256") != sha256_file(current_model):
        raise RuntimeError(
            "RQ1 robustness results were generated with a different primary model"
        )
    records = []
    for key, values in payload["conditions"].items():
        family, condition = key.split(":", 1)
        change = values.get("paired_image_uncertainty_change", {})
        records.append(
            {
                "family": family,
                "condition": condition,
                **{
                    name: value
                    for name, value in values.items()
                    if name
                    not in ("paired_image_uncertainty_change", "detector_coco")
                },
                **{
                    f"COCO_{name}": value
                    for name, value in values.get("detector_coco", {}).items()
                },
                **{
                    f"uncertainty_change_{name}": value
                    for name, value in change.items()
                },
            }
        )
    path = output_root / "Table_RQ1_robustness.csv"
    pd.DataFrame(records).to_csv(path, index=False)
    return path


def _write_feature_complementarity(
    config: dict[str, Any], output_root: Path
) -> Path:
    validation, _ = read_validated_features(
        config,
        project_path(
            config, config["rq1"]["outputs"]["validation_features"]
        ),
    )
    validation = validation.loc[
        validation["score"]
        >= float(config["rq1"]["fusion"]["training_score_threshold"])
    ]
    groups = config["rq1"]["feature_groups"]
    features = list(
        dict.fromkeys(
            feature
            for group in (
                "confidence",
                "semantic",
                "geometric",
                "representation",
                "presence",
            )
            for feature in groups[group]
        )
    )
    correlation = validation[features].corr(method="spearman")
    correlation.index.name = "feature"
    path = output_root / "Table_RQ1_feature_spearman.csv"
    correlation.to_csv(path)
    return path


def _write_threshold_sensitivity(
    metrics: dict[str, Any], output_root: Path
) -> Path:
    records = []
    for threshold, threshold_result in metrics[
        "score_threshold_sensitivity"
    ].items():
        for method, values in threshold_result["methods"].items():
            records.append(
                {
                    "score_threshold": float(threshold),
                    "rows": threshold_result["rows"],
                    "error_prevalence": threshold_result[
                        "error_prevalence"
                    ],
                    "method": method,
                    "AUROC": values["auroc"],
                    "AUPRC": values["auprc"],
                    "AURC": values["aurc"],
                }
            )
    path = output_root / "Table_RQ1_threshold_sensitivity.csv"
    pd.DataFrame(records).to_csv(path, index=False)
    return path


def _write_subgroups(metrics: dict[str, Any], output_root: Path) -> Path:
    records = []
    for attribute, values in metrics["subgroups"].items():
        for subgroup, subgroup_result in values.items():
            for method, method_values in subgroup_result["methods"].items():
                records.append(
                    {
                        "attribute": attribute,
                        "subgroup": subgroup,
                        "rows": subgroup_result["rows"],
                        "error_prevalence": subgroup_result[
                            "error_prevalence"
                        ],
                        "method": method,
                        "AUROC": method_values["auroc"],
                        "AUPRC": method_values["auprc"],
                        "AURC": method_values["aurc"],
                    }
                )
    path = output_root / "Table_RQ1_subgroups.csv"
    pd.DataFrame(records).to_csv(path, index=False)
    return path


def _write_mc_sensitivity(
    metrics: dict[str, Any], output_root: Path
) -> Path:
    records = []
    for method, result in metrics["mc_pass_sensitivity"].items():
        values = result["metrics"]
        intervals = result["clustered_bootstrap_interval"]
        records.append(
            {
                "method": method,
                "mc_passes": result["mc_passes"],
                "AUROC": values["auroc"],
                "AUROC_CI_low": intervals["auroc"]["lower"],
                "AUROC_CI_high": intervals["auroc"]["upper"],
                "AUPRC": values["auprc"],
                "AURC": values["aurc"],
                "ECE": values["ece"],
            }
        )
    path = output_root / "Table_RQ1_mc_pass_sensitivity.csv"
    pd.DataFrame(records).sort_values("mc_passes").to_csv(
        path, index=False
    )
    return path


def _write_image_safety(
    metrics: dict[str, Any], output_root: Path
) -> Path:
    records = []
    for method, outcomes in metrics["image_level_safety"].items():
        if method == "aggregation":
            continue
        for outcome, values in outcomes.items():
            record = {
                "method": method,
                "outcome": outcome,
                "status": values["status"],
                "n": values["n"],
            }
            if values["status"] == "ok":
                record.update(
                    {
                        "prevalence": values["error_prevalence"],
                        "AUROC": values["auroc"],
                        "AUPRC": values["auprc"],
                        "AURC": values["aurc"],
                    }
                )
            else:
                record["prevalence"] = values["prevalence"]
            records.append(record)
    path = output_root / "Table_RQ1_image_safety.csv"
    pd.DataFrame(records).to_csv(path, index=False)
    return path


def _write_detector_performance(
    metrics: dict[str, Any], output_root: Path
) -> Path:
    path = output_root / "Table_RQ1_detector_performance.csv"
    pd.DataFrame([metrics["detector_performance"]]).to_csv(
        path, index=False
    )
    return path
