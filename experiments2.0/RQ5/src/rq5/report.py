from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from adas_ovd.config import project_path
from adas_ovd.reproducibility import sha256_file, write_json

from .evaluation import weighted_risk_coverage_curve


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_figure(figure: plt.Figure, root: Path, stem: str) -> list[Path]:
    paths = [root / f"{stem}.png", root / f"{stem}.pdf"]
    for path in paths:
        figure.savefig(path, bbox_inches="tight", dpi=180)
    plt.close(figure)
    return paths


def _calibration_curve(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ids = np.clip(np.digitize(probabilities, edges[1:-1]), 0, bins - 1)
    predicted: list[float] = []
    observed: list[float] = []
    for index in range(bins):
        mask = ids == index
        if mask.any():
            predicted.append(float(probabilities[mask].mean()))
            observed.append(float(labels[mask].mean()))
    return np.asarray(predicted), np.asarray(observed)


def generate_report(config: dict[str, Any]) -> dict[str, Path]:
    outputs = config["rq5"]["outputs"]
    root = project_path(config, outputs["root"])
    root.mkdir(parents=True, exist_ok=True)
    metrics_path = project_path(config, outputs["metrics"])
    metrics = _load_json(metrics_path)
    predictions_path = project_path(config, outputs["predictions"])
    bootstrap_path = project_path(config, outputs["bootstrap"])
    for key, path in (("predictions", predictions_path), ("bootstrap", bootstrap_path)):
        expected = metrics["artifact_integrity"][key]["sha256"]
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"RQ5 report input integrity failed: {path}")
    predictions = pd.read_parquet(predictions_path)

    created: dict[str, Path] = {}
    main_rows: list[dict[str, Any]] = []
    for method, values in metrics["methods"].items():
        main_rows.append(
            {
                "method": method,
                "status": values.get("status"),
                "n": values.get("n"),
                "weighted_aurc": values.get("weighted_aurc"),
                "coverage_at_weighted_risk_0.10": values.get(
                    "coverage_at_weighted_risk_0.10"
                ),
                "operating_coverage": values.get("operating_coverage"),
                "operating_criticality_mass_coverage": values.get(
                    "operating_criticality_mass_coverage"
                ),
                "operating_weighted_risk": values.get("operating_weighted_risk"),
                "brier": values.get("brier"),
                "nll": values.get("nll"),
                "ece": values.get("ece"),
                "auroc": values.get("auroc"),
                "aurc": values.get("aurc"),
            }
        )
    main_path = root / "Table_RQ5_main.csv"
    pd.DataFrame(main_rows).to_csv(main_path, index=False)
    created["main_table"] = main_path

    inference_path = root / "Table_RQ5_primary_inference.csv"
    inference_rows = [
        {"comparison": key, **value}
        for key, value in metrics["primary_inference"]["comparisons"].items()
    ]
    inference_rows.extend(
        {
            "comparison": f"brier_noninferiority__{baseline}",
            "metric": "brier_noninferiority",
            **record,
        }
        for baseline, record in metrics["primary_inference"][
            "brier_noninferiority"
        ].items()
        if baseline != "pass"
    )
    pd.DataFrame(inference_rows).to_csv(inference_path, index=False)
    created["primary_inference_table"] = inference_path

    latency_rows = list(metrics["computational_cost"]["prefix_latency"].values())
    latency_path = root / "Table_RQ5_latency.csv"
    pd.DataFrame(latency_rows).to_csv(latency_path, index=False)
    created["latency_table"] = latency_path

    action_path = root / "Table_RQ5_actions.csv"
    pd.DataFrame(
        [
            {
                "method": method,
                "accepted": values.get("accepted_detections"),
                "deferred": values.get("deferred_detections"),
                "coverage": values.get("operating_coverage"),
                "criticality_mass_coverage": values.get(
                    "operating_criticality_mass_coverage"
                ),
                "weighted_risk": values.get("operating_weighted_risk"),
                "accepted_criticality_mass": values.get(
                    "accepted_criticality_mass"
                ),
                "deferred_criticality_mass": values.get(
                    "deferred_criticality_mass"
                ),
            }
            for method, values in metrics["methods"].items()
        ]
    ).to_csv(action_path, index=False)
    created["actions_table"] = action_path

    criticality_path = root / "Table_RQ5_criticality_sensitivity.csv"
    pd.DataFrame(
        [
            {"criticality_specification": name, **values}
            for name, values in metrics["criticality_weight_sensitivity"].items()
        ]
    ).to_csv(criticality_path, index=False)
    created["criticality_sensitivity_table"] = criticality_path

    model_index_path = project_path(config, outputs["models"]) / "model_index.json"
    model_index = _load_json(model_index_path)
    parameter_rows: list[dict[str, Any]] = []
    for method, metadata in model_index["methods"].items():
        if not metadata["components"]:
            parameter_rows.append(
                {
                    "method": method,
                    "component": "none",
                    "mc_passes": metadata["mc_passes"],
                    "coefficient_count": 0,
                    "selected_regularization": None,
                    "selection_auroc": None,
                }
            )
        for component, values in metadata["components"].items():
            parameter_rows.append(
                {
                    "method": method,
                    "component": component,
                    "mc_passes": metadata["mc_passes"],
                    "coefficient_count": values["coefficient_count"],
                    "selected_regularization": values["selected_regularization"],
                    "selection_auroc": values["selection_auroc"],
                }
            )
    parameters_path = root / "Table_RQ5_model_parameters.csv"
    pd.DataFrame(parameter_rows).to_csv(parameters_path, index=False)
    created["model_parameters_table"] = parameters_path

    subgroup_rows: list[dict[str, Any]] = []
    for attribute, values in metrics["subgroups"].items():
        for value, record in values.items():
            subgroup_rows.append(
                {
                    "attribute": attribute,
                    "value": value,
                    "status": record.get("status"),
                    "n": record.get("n"),
                    "weighted_aurc": record.get("weighted_aurc"),
                    "coverage_at_weighted_risk_0.10": record.get(
                        "coverage_at_weighted_risk_0.10"
                    ),
                    "brier": record.get("brier"),
                }
            )
    subgroups_path = root / "Table_RQ5_subgroups.csv"
    pd.DataFrame(subgroup_rows).to_csv(subgroups_path, index=False)
    created["subgroups_table"] = subgroups_path

    selected_methods = [
        name
        for name in (
            "calibrated_confidence",
            "uncertainty_only",
            "risk_aware_fusion",
            "flat_joint",
        )
        if name in set(predictions["method"])
    ]
    figure, axis = plt.subplots(figsize=(6.2, 4.2))
    for method in selected_methods:
        subset = predictions.loc[predictions["method"] == method]
        coverage, risk, _ = weighted_risk_coverage_curve(
            subset["is_error"].to_numpy(),
            subset["decision_risk"].to_numpy(),
            subset["criticality_weight"].to_numpy(),
        )
        axis.plot(coverage, risk, label=method)
    axis.set_xlabel("Detection coverage")
    axis.set_ylabel("Criticality-weighted selective risk")
    axis.set_title("RQ5 weighted risk-coverage")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25)
    for path in _save_figure(figure, root, "Fig_RQ5_weighted_risk_coverage"):
        created[f"weighted_risk_coverage_{path.suffix[1:]}"] = path

    figure, axis = plt.subplots(figsize=(5.2, 4.4))
    for method in selected_methods:
        subset = predictions.loc[predictions["method"] == method]
        predicted, observed = _calibration_curve(
            subset["is_error"].to_numpy(dtype=np.float64),
            subset["probability_error"].to_numpy(dtype=np.float64),
        )
        axis.plot(predicted, observed, marker="o", label=method)
    axis.plot([0, 1], [0, 1], "--", color="black", linewidth=1)
    axis.set_xlabel("Predicted error probability")
    axis.set_ylabel("Observed error frequency")
    axis.set_title("RQ5 reliability")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25)
    for path in _save_figure(figure, root, "Fig_RQ5_reliability"):
        created[f"reliability_{path.suffix[1:]}"] = path

    latency_frame = pd.DataFrame(latency_rows).sort_values("mc_passes")
    figure, axis = plt.subplots(figsize=(5.6, 4.2))
    axis.plot(
        latency_frame["p95_ms"],
        latency_frame["mc_passes"],
        marker="o",
        label="MC prefix",
    )
    for budget in config["rq5"]["realtime"]["budgets_ms"]:
        axis.axvline(float(budget), linestyle="--", linewidth=1, label=f"{budget:g} ms")
    axis.set_xlabel("Estimated/measured p95 latency (ms)")
    axis.set_ylabel("MC passes")
    axis.set_title("RQ5 latency frontier")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25)
    for path in _save_figure(figure, root, "Fig_RQ5_latency_frontier"):
        created[f"latency_frontier_{path.suffix[1:]}"] = path

    captions_path = root / "figure_captions.md"
    captions_path.write_text(
        "# RQ5 diagnostic figure captions\n\n"
        "- **Weighted risk-coverage:** lower is better; defer is an unevaluated fallback request, not a safe outcome.\n"
        "- **Reliability:** calibrated detection-conditioned error probabilities.\n"
        "- **Latency frontier:** mc10 GPU blocks are measured; mc02/mc05 are explicitly linear prefix estimates.\n"
        "\nMini figures are diagnostic and are not paper evidence.\n",
        encoding="utf-8",
    )
    created["figure_captions"] = captions_path

    manifest_path = root / "report_manifest.json"
    manifest = {
        "schema_version": 1,
        "rq": "RQ5",
        "evidence_status": metrics["evidence_status"],
        "metrics_path": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "predictions_sha256": sha256_file(predictions_path),
        "bootstrap_sha256": sha256_file(bootstrap_path),
        "artifacts": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in created.items()
        },
    }
    write_json(manifest_path, manifest)
    created["report_manifest"] = manifest_path
    return created
