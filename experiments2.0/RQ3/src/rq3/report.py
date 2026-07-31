from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from adas_ovd.config import project_path
from adas_ovd.metrics import risk_coverage_curve
from adas_ovd.reproducibility import sha256_file, write_json

from .fusion import (
    DirectErrorScorer,
    EqualFusionScorer,
    LearnedSpatialQualityScorer,
    ProductFusionScorer,
    load_fusions,
)


def _load_metrics(config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = project_path(config, config["rq3"]["outputs"]["metrics"])
    if not path.is_file():
        raise FileNotFoundError(f"RQ3 metrics are missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return path, json.load(handle)


def _save_figure(figure, root: Path, stem: str) -> list[Path]:
    paths = [root / f"{stem}.png", root / f"{stem}.pdf"]
    figure.savefig(paths[0], dpi=180, bbox_inches="tight")
    figure.savefig(paths[1], bbox_inches="tight")
    return paths


def _binned_observations(
    probability: np.ndarray, labels: np.ndarray, bins: int = 10
) -> tuple[list[float], list[float]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.clip(np.digitize(probability, edges[1:-1]), 0, bins - 1)
    predicted, observed = [], []
    for bin_id in range(bins):
        mask = bin_ids == bin_id
        if mask.any():
            predicted.append(float(probability[mask].mean()))
            observed.append(float(labels[mask].mean()))
    return predicted, observed


def _model_parameters(config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for method, fusion in load_fusions(config).items():
        scorer = fusion.scorer
        if isinstance(scorer, DirectErrorScorer):
            features = scorer.features
            pipeline = scorer.estimator
            target = "is_error"
        elif isinstance(
            scorer,
            (ProductFusionScorer, EqualFusionScorer, LearnedSpatialQualityScorer),
        ):
            features = scorer.quality_model.features
            pipeline = scorer.quality_model.estimator
            target = scorer.quality_model.target
        else:
            continue
        imputer = pipeline.named_steps["imputer"]
        classifier = pipeline.named_steps["classifier"]
        names = list(imputer.get_feature_names_out(features))
        rows.extend(
            {
                "method": method,
                "target": target,
                "parameter": name,
                "standardized_logistic_coefficient": float(value),
                "selected_regularization": fusion.selected_regularization,
                "selection_auroc": fusion.selection_auroc,
            }
            for name, value in zip(names, classifier.coef_[0], strict=True)
        )
    return pd.DataFrame(rows)


def generate_report(config: dict[str, Any]) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outputs = config["rq3"]["outputs"]
    root = project_path(config, outputs["root"])
    root.mkdir(parents=True, exist_ok=True)
    metrics_path, payload = _load_metrics(config)
    predictions_path = project_path(config, outputs["predictions"])
    if (
        not predictions_path.is_file()
        or payload["artifact_integrity"]["predictions_sha256"]
        != sha256_file(predictions_path)
    ):
        raise RuntimeError("RQ3 prediction integrity failed before reporting")
    predictions = pd.read_parquet(predictions_path)

    main_rows = []
    for method, record in payload["methods"].items():
        row = {"method": method, "family": record["family"]}
        row.update(record["metrics"])
        for metric, interval in record["clustered_bootstrap_interval"].items():
            row[f"{metric}_ci_lower"] = interval["lower"]
            row[f"{metric}_ci_upper"] = interval["upper"]
        main_rows.append(row)
    main_table = root / "Table_RQ3_main.csv"
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
    primary_table = root / "Table_RQ3_primary_inference.csv"
    pd.DataFrame(primary_rows).to_csv(primary_table, index=False)

    ranking_table = root / "Table_RQ3_detector_ranking.csv"
    pd.DataFrame(
        [
            {"method": method, **metrics}
            for method, metrics in payload["detector_ranking"].items()
        ]
    ).to_csv(ranking_table, index=False)

    mc_table = root / "Table_RQ3_mc_pass_sensitivity.csv"
    pd.DataFrame(
        [
            {
                "method": method,
                "mc_passes": record["mc_passes"],
                **record["metrics"],
            }
            for method, record in payload["mc_pass_sensitivity"].items()
        ]
    ).to_csv(mc_table, index=False)

    threshold_rows = []
    for threshold, record in payload["score_threshold_sensitivity"].items():
        if not record["methods"]:
            threshold_rows.append(
                {
                    "score_threshold": threshold,
                    "rows": record["rows"],
                    "status": record["status"],
                    "method": "not_estimable",
                }
            )
        for method, metrics in record["methods"].items():
            threshold_rows.append(
                {
                    "score_threshold": threshold,
                    "rows": record["rows"],
                    "status": record["status"],
                    "method": method,
                    **metrics,
                }
            )
    threshold_table = root / "Table_RQ3_threshold_sensitivity.csv"
    pd.DataFrame(threshold_rows).to_csv(threshold_table, index=False)

    localization_rows = []
    for threshold, record in payload["localization_target_sensitivity"].items():
        localization_rows.append(
            {
                "localization_iou_threshold": threshold,
                "status": record["status"],
                "rows": record["rows"],
                **record.get("metrics", {}),
            }
        )
    localization_table = root / "Table_RQ3_localization_sensitivity.csv"
    pd.DataFrame(localization_rows).to_csv(localization_table, index=False)

    ece_table = root / "Table_RQ3_ece_bins.csv"
    pd.DataFrame(
        [
            {"method": method, "bins": bins, "ece": value}
            for method, values in payload["ece_bin_sensitivity"].items()
            for bins, value in values.items()
        ]
    ).to_csv(ece_table, index=False)

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
                        "status": record["status"],
                        "method": "not_estimable",
                    }
                )
            for method, metrics in record["methods"].items():
                subgroup_rows.append(
                    {
                        "attribute": attribute,
                        "value": value,
                        "rows": record["rows"],
                        "error_prevalence": record["error_prevalence"],
                        "status": record["status"],
                        "method": method,
                        **metrics,
                    }
                )
    subgroup_table = root / "Table_RQ3_subgroups.csv"
    pd.DataFrame(subgroup_rows).to_csv(subgroup_table, index=False)

    cost_table = root / "Table_RQ3_computational_cost.csv"
    pd.DataFrame(
        [
            {
                key: value
                for key, value in payload["computational_cost"].items()
                if key not in {"stochastic_modules", "environment"}
            }
        ]
    ).to_csv(cost_table, index=False)
    taxonomy_table = root / "Table_RQ3_error_taxonomy.csv"
    pd.DataFrame([payload["error_taxonomy"]]).to_csv(taxonomy_table, index=False)
    parameter_table = root / "Table_RQ3_model_parameters.csv"
    _model_parameters(config).to_csv(parameter_table, index=False)

    labels = predictions["is_error"].to_numpy(dtype=np.int64)
    selected = [
        "confidence",
        "spatial_agreement",
        "learned_spatial_quality",
        "equal_fusion",
        "product_fusion",
        "capacity_control_product",
    ]
    figure, axis = plt.subplots(figsize=(7.4, 5.2))
    for method in selected:
        curve = risk_coverage_curve(
            labels, predictions[f"rank_{method}"].to_numpy(dtype=np.float64)
        )
        axis.plot(curve.coverage, curve.risk, label=f"{method} ({curve.aurc:.3f})")
    axis.set_xlabel("Coverage")
    axis.set_ylabel("Detection error risk")
    axis.set_title("RQ3 localization-aware risk–coverage")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7)
    risk_paths = _save_figure(figure, root, "Fig_RQ3_risk_coverage")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.5, 5.1))
    for method in ("confidence", "product_fusion", "capacity_control_product"):
        probability = predictions[f"prob_error_{method}"].to_numpy(dtype=np.float64)
        predicted, observed = _binned_observations(probability, labels)
        axis.plot(predicted, observed, marker="o", label=method)
    axis.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    axis.set_xlabel("Predicted detection-error probability")
    axis.set_ylabel("Observed detection-error frequency")
    axis.set_title("RQ3 reliability diagram")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    reliability_paths = _save_figure(figure, root, "Fig_RQ3_reliability")
    plt.close(figure)

    spatial_quality = 1.0 - predictions[
        "rank_learned_spatial_quality"
    ].to_numpy(dtype=np.float64)
    localized = predictions["is_well_localized"].to_numpy(dtype=np.int64)
    predicted, observed = _binned_observations(spatial_quality, localized)
    figure, axis = plt.subplots(figsize=(6.4, 5.0))
    axis.plot(predicted, observed, marker="o", color="#4c78a8")
    axis.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    axis.set_xlabel("Predicted spatial localization quality")
    axis.set_ylabel("Observed fraction with IoU ≥ 0.50")
    axis.set_title("RQ3 spatial-quality validation on test detections")
    axis.grid(alpha=0.2)
    spatial_paths = _save_figure(figure, root, "Fig_RQ3_spatial_quality")
    plt.close(figure)

    captions = root / "figure_captions.md"
    captions.write_text(
        "# RQ3 figure captions / Leyendas de figuras RQ3\n\n"
        "Diagnostic outputs generated with `rq3_mini.yaml` are not scientific "
        "evidence. / Las salidas de `rq3_mini.yaml` no son evidencia científica.\n\n"
        "- **Fig_RQ3_risk_coverage.** Error risk while retaining detections from "
        "least to most uncertain; lower is better. / Riesgo al retener detecciones "
        "de menor a mayor incertidumbre; menor es mejor.\n"
        "- **Fig_RQ3_reliability.** Calibrated detection-error probabilities. / "
        "Probabilidades calibradas de error de detección.\n"
        "- **Fig_RQ3_spatial_quality.** Predicted spatial quality against the "
        "class-agnostic IoU target. / Calidad espacial predicha frente al target "
        "IoU independiente de clase.\n",
        encoding="utf-8",
    )

    paths = {
        "main_table": main_table,
        "primary_inference_table": primary_table,
        "detector_ranking_table": ranking_table,
        "mc_pass_table": mc_table,
        "threshold_table": threshold_table,
        "localization_sensitivity_table": localization_table,
        "ece_bins_table": ece_table,
        "subgroup_table": subgroup_table,
        "cost_table": cost_table,
        "error_taxonomy_table": taxonomy_table,
        "model_parameter_table": parameter_table,
        "risk_coverage_png": risk_paths[0],
        "risk_coverage_pdf": risk_paths[1],
        "reliability_png": reliability_paths[0],
        "reliability_pdf": reliability_paths[1],
        "spatial_quality_png": spatial_paths[0],
        "spatial_quality_pdf": spatial_paths[1],
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
