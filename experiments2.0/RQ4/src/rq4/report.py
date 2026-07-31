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
from adas_ovd.metrics import risk_coverage_curve
from adas_ovd.reproducibility import sha256_file, source_tree_sha256, stable_fingerprint, write_json


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_figure(figure: plt.Figure, root: Path, stem: str) -> list[Path]:
    paths = [root / f"{stem}.png", root / f"{stem}.pdf"]
    figure.tight_layout()
    figure.savefig(paths[0], dpi=180, bbox_inches="tight")
    figure.savefig(paths[1], bbox_inches="tight")
    plt.close(figure)
    return paths


def _calibration_curve(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ids = np.clip(np.digitize(probabilities, edges[1:-1]), 0, bins - 1)
    predicted = []
    observed = []
    for index in range(bins):
        mask = ids == index
        if mask.any():
            predicted.append(float(probabilities[mask].mean()))
            observed.append(float(labels[mask].mean()))
    return np.asarray(predicted), np.asarray(observed)


def generate_report(config: dict[str, Any]) -> dict[str, Path]:
    outputs = config["rq4"]["outputs"]
    root = project_path(config, outputs["root"])
    root.mkdir(parents=True, exist_ok=True)
    metrics_path = project_path(config, outputs["metrics"])
    predictions_path = project_path(config, outputs["predictions"])
    bootstrap_path = project_path(config, outputs["bootstrap"])
    model_index_path = project_path(config, outputs["models"]) / "model_index.json"
    for path in (metrics_path, predictions_path, bootstrap_path, model_index_path):
        if not path.is_file():
            raise FileNotFoundError(f"RQ4 report input is missing: {path}")
    input_hashes = {
        str(path): sha256_file(path)
        for path in (metrics_path, predictions_path, bootstrap_path, model_index_path)
    }
    report_identity = stable_fingerprint(
        {
            "schema_version": 2,
            "source_tree_sha256": source_tree_sha256(
                config["_meta"]["project_root"], ("RQ4/src/rq4/report.py",)
            ),
            "inputs": input_hashes,
        }
    )
    manifest_path = root / "report_manifest.json"
    if manifest_path.is_file():
        try:
            cached_manifest = _load_json(manifest_path)
            cached_artifacts = cached_manifest.get("artifacts", {})
            cache_valid = (
                cached_manifest.get("report_identity") == report_identity
                and cached_manifest.get("inputs") == input_hashes
                and all(
                    Path(record["path"]).is_file()
                    and sha256_file(Path(record["path"])) == record["sha256"]
                    for record in cached_artifacts.values()
                )
            )
            if cache_valid:
                generated = {
                    name: Path(record["path"])
                    for name, record in cached_artifacts.items()
                }
                generated["report_manifest"] = manifest_path
                return generated
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            pass
    metrics = _load_json(metrics_path)
    model_index = _load_json(model_index_path)
    integrity = metrics["artifact_integrity"]
    expected = {
        predictions_path: integrity["predictions_sha256"],
        bootstrap_path: integrity["bootstrap_sha256"],
        model_index_path: integrity["model_index_sha256"],
    }
    for path, expected_hash in expected.items():
        if sha256_file(path) != expected_hash:
            raise RuntimeError(f"RQ4 report input hash mismatch: {path}")
    predictions = pd.read_parquet(predictions_path)

    generated: dict[str, Path] = {}
    main_rows = []
    for method, record in metrics.get("methods", {}).items():
        main_rows.append(
            {
                "method": method,
                "family": record["family"],
                "coefficient_count": record["coefficient_count"],
                **record["metrics"],
            }
        )
    main_path = root / "Table_RQ4_main.csv"
    pd.DataFrame(main_rows).to_csv(main_path, index=False)
    generated["main_table"] = main_path

    primary_rows = []
    primary = metrics.get("primary_inference", {})
    for baseline, comparisons in primary.get("comparisons", {}).items():
        for metric, record in comparisons.items():
            primary_rows.append({"baseline": baseline, "metric": metric, **record})
    primary_path = root / "Table_RQ4_primary_inference.csv"
    pd.DataFrame(primary_rows).to_csv(primary_path, index=False)
    generated["primary_table"] = primary_path

    domain_rows = []
    for domain, record in metrics.get("domain_analysis", {}).items():
        if record.get("status") != "estimable":
            domain_rows.append({"domain": domain, "status": record.get("status"), "rows": record.get("rows")})
            continue
        for method, method_record in record.get("methods", {}).items():
            if method_record.get("status") == "estimable":
                domain_rows.append(
                    {
                        "domain": domain, "status": "estimable", "rows": record["rows"],
                        "method": method, **method_record["metrics"],
                    }
                )
    domain_path = root / "Table_RQ4_domains.csv"
    pd.DataFrame(domain_rows).to_csv(domain_path, index=False)
    generated["domains_table"] = domain_path

    parameter_rows = []
    for method, record in model_index["methods"].items():
        parameter_rows.append(
            {
                "method": method,
                "family": record["family"],
                "coefficient_count": record["coefficient_count"],
                "feature_count": len(record["features"]),
                "model_sha256": record["model_sha256"],
            }
        )
    parameter_path = root / "Table_RQ4_model_parameters.csv"
    pd.DataFrame(parameter_rows).to_csv(parameter_path, index=False)
    generated["parameter_table"] = parameter_path

    sensitivity_rows = []
    for method, record in metrics.get("mc_pass_sensitivity", {}).items():
        sensitivity_rows.append({"method": method, "mc_passes": record["mc_passes"], **record["metrics"]})
    sensitivity_path = root / "Table_RQ4_mc_pass_sensitivity.csv"
    pd.DataFrame(sensitivity_rows).to_csv(sensitivity_path, index=False)
    generated["mc_sensitivity_table"] = sensitivity_path

    cost_path = root / "Table_RQ4_computational_cost.csv"
    cost = {
        key: value for key, value in metrics["computational_cost"].items()
        if key not in {"environment", "stochastic_modules"}
    }
    pd.DataFrame([cost]).to_csv(cost_path, index=False)
    generated["cost_table"] = cost_path

    shifted = predictions["is_domain_shift"].astype(bool).to_numpy()
    labels = predictions.loc[shifted, "is_error"].to_numpy(dtype=np.int64)
    plot_methods = [name for name in ("confidence_calibrated", "flat_joint", "multilevel") if f"prob_error_{name}" in predictions]
    figure, axis = plt.subplots(figsize=(6.2, 5.2))
    axis.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1, label="ideal")
    for method in plot_methods:
        probability = predictions.loc[shifted, f"prob_error_{method}"].to_numpy(dtype=np.float64)
        x, y = _calibration_curve(labels, probability, 10)
        axis.plot(x, y, marker="o", label=method)
    axis.set(xlabel="Predicted error probability", ylabel="Observed error rate", title="RQ4 shifted-domain reliability")
    axis.legend(fontsize=8)
    for path in _save_figure(figure, root, "Fig_RQ4_reliability"):
        generated[f"reliability_{path.suffix[1:]}"] = path

    figure, axis = plt.subplots(figsize=(6.2, 5.2))
    for method in plot_methods:
        rank = predictions.loc[shifted, f"rank_{method}"].to_numpy(dtype=np.float64)
        curve = risk_coverage_curve(labels, rank)
        axis.plot(curve.coverage, curve.risk, label=method)
    axis.set(xlabel="Coverage", ylabel="Selective risk", title="RQ4 shifted-domain risk--coverage")
    axis.legend(fontsize=8)
    for path in _save_figure(figure, root, "Fig_RQ4_risk_coverage"):
        generated[f"risk_coverage_{path.suffix[1:]}"] = path

    comparison_labels = []
    comparison_values = []
    for baseline, comparisons in primary.get("comparisons", {}).items():
        for metric, record in comparisons.items():
            comparison_labels.append(f"{metric}\nvs {baseline}")
            comparison_values.append(float(record["improvement"]))
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    if comparison_values:
        colors = ["#2a9d8f" if value > 0 else "#e76f51" for value in comparison_values]
        axis.bar(np.arange(len(comparison_values)), comparison_values, color=colors)
        axis.set_xticks(np.arange(len(comparison_values)), comparison_labels, rotation=25, ha="right", fontsize=8)
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set(ylabel="Improvement (positive is better)", title="RQ4 primary shifted-domain contrasts")
    for path in _save_figure(figure, root, "Fig_RQ4_primary_improvements"):
        generated[f"primary_improvements_{path.suffix[1:]}"] = path

    captions_path = root / "figure_captions.md"
    captions_path.write_text(
        "# RQ4 figure captions / Leyendas de figuras\n\n"
        "- `Fig_RQ4_reliability`: shifted-domain reliability diagram; diagnostic outputs are not paper evidence. / Diagrama de fiabilidad en dominio desplazado; los outputs diagnósticos no son evidencia.\n"
        "- `Fig_RQ4_risk_coverage`: selective error risk as detections are retained from lowest to highest uncertainty. / Riesgo selectivo al retener detecciones de menor a mayor incertidumbre.\n"
        "- `Fig_RQ4_primary_improvements`: prespecified Brier, NLL and AURC contrasts; positive favors multilevel. / Contrastes preespecificados; positivo favorece multinivel.\n",
        encoding="utf-8",
    )
    generated["figure_captions"] = captions_path

    manifest = {
        "schema_version": 2,
        "report_identity": report_identity,
        "rq": "rq4",
        "test_partition": metrics["test_partition"],
        "evidence_status": metrics["evidence_status"],
        "inputs": input_hashes,
        "artifacts": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in generated.items()
        },
    }
    write_json(manifest_path, manifest)
    generated["report_manifest"] = manifest_path
    return generated
