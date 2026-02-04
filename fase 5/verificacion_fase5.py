"""
Verificación Completa de Fase 5 - Outputs y Resultados
Verifica que todos los archivos se generaron correctamente y analiza los resultados
"""

import json
import pandas as pd
from pathlib import Path
import numpy as np

print("=" * 80)
print("  VERIFICACIÓN COMPLETA - FASE 5")
print("  Comparación de Métodos de Incertidumbre y Calibración")
print("=" * 80)

# Directorio de outputs
FASE5_OUT = Path("outputs/comparison")

# 1. Verificar estructura de directorios
print("\n1. ESTRUCTURA DE DIRECTORIOS")
print("-" * 80)

if FASE5_OUT.exists():
    print(f"✓ Directorio outputs existe: {FASE5_OUT}")
    all_files = list(FASE5_OUT.rglob("*.*"))
    print(f"✓ Total de archivos generados: {len(all_files)}")
else:
    print(f"✗ ERROR: No existe el directorio {FASE5_OUT}")
    exit(1)

# 2. Verificar archivos JSON críticos
print("\n2. ARCHIVOS JSON DE RESULTADOS")
print("-" * 80)

critical_json_files = {
    "detection_metrics.json": "Métricas de detección (mAP)",
    "calibration_metrics.json": "Métricas de calibración (ECE, NLL, Brier)",
    "temperatures.json": "Temperaturas de calibración",
    "risk_coverage_auc.json": "AUC de curvas risk-coverage",
    "uncertainty_auroc.json": "AUROC de incertidumbre",
    "final_report.json": "Reporte final comparativo",
}

json_files_status = {}
for filename, description in critical_json_files.items():
    filepath = FASE5_OUT / filename
    exists = filepath.exists()
    json_files_status[filename] = exists
    status = "✓" if exists else "✗"
    size = f"({filepath.stat().st_size:,} bytes)" if exists else ""
    print(f"{status} {filename:<30} - {description} {size}")

# 3. Verificar archivos de visualización
print("\n3. VISUALIZACIONES GENERADAS")
print("-" * 80)

critical_plots = {
    "final_comparison_summary.png": "Resumen comparativo de todos los métodos",
    "reliability_diagrams.png": "Diagramas de confiabilidad",
    "risk_coverage_curves.png": "Curvas de risk-coverage",
    "uncertainty_analysis.png": "Análisis de incertidumbre",
}

plot_files_status = {}
for filename, description in critical_plots.items():
    filepath = FASE5_OUT / filename
    exists = filepath.exists()
    plot_files_status[filename] = exists
    status = "✓" if exists else "✗"
    size = f"({filepath.stat().st_size:,} bytes)" if exists else ""
    print(f"{status} {filename:<35} - {description} {size}")

# 4. Verificar archivos de predicciones por método
print("\n4. PREDICCIONES POR MÉTODO")
print("-" * 80)

methods = [
    "baseline",
    "baseline_ts",
    "mc_dropout",
    "mc_dropout_ts",
    "decoder_variance",
    "decoder_variance_ts",
]

prediction_files = {}
for method in methods:
    filepath = FASE5_OUT / f"eval_{method}.json"
    exists = filepath.exists()
    prediction_files[method] = exists
    status = "✓" if exists else "✗"

    if exists:
        try:
            with open(filepath) as f:
                preds = json.load(f)
            size = len(preds)
            print(f"{status} {method:<25} - {size:,} predicciones")
        except Exception as e:
            print(f"{status} {method:<25} - Error al cargar: {e}")
    else:
        print(f"{status} {method:<25} - Archivo no encontrado")

# 5. Analizar métricas de detección
print("\n5. MÉTRICAS DE DETECCIÓN (mAP)")
print("-" * 80)

if (FASE5_OUT / "detection_metrics.json").exists():
    with open(FASE5_OUT / "detection_metrics.json") as f:
        det_metrics = json.load(f)

    print(f"{'Método':<25} {'mAP@0.5':<10} {'AP50':<10} {'AP75':<10}")
    print("-" * 60)

    for method, metrics in det_metrics.items():
        map_val = metrics.get("mAP@0.5", metrics.get("mAP", 0))
        ap50 = metrics.get("AP50", 0)
        ap75 = metrics.get("AP75", 0)
        print(f"{method:<25} {map_val:<10.4f} {ap50:<10.4f} {ap75:<10.4f}")
else:
    print("✗ Archivo detection_metrics.json no encontrado")

# 6. Analizar métricas de calibración
print("\n6. MÉTRICAS DE CALIBRACIÓN")
print("-" * 80)

if (FASE5_OUT / "calibration_metrics.json").exists():
    with open(FASE5_OUT / "calibration_metrics.json") as f:
        calib_metrics = json.load(f)

    print(f"{'Método':<25} {'ECE':<12} {'NLL':<12} {'Brier':<12}")
    print("-" * 65)

    for method, metrics in calib_metrics.items():
        ece = metrics.get("ECE", metrics.get("ece", 0))
        nll = metrics.get("NLL", metrics.get("nll", 0))
        brier = metrics.get("Brier", metrics.get("brier", 0))
        print(f"{method:<25} {ece:<12.6f} {nll:<12.6f} {brier:<12.6f}")
else:
    print("✗ Archivo calibration_metrics.json no encontrado")

# 7. Analizar temperaturas
print("\n7. TEMPERATURAS DE CALIBRACIÓN")
print("-" * 80)

if (FASE5_OUT / "temperatures.json").exists():
    with open(FASE5_OUT / "temperatures.json") as f:
        temps = json.load(f)

    print(f"{'Método':<25} {'Temperatura':<15} {'Interpretación'}")
    print("-" * 70)

    for method, temp_data in temps.items():
        # Manejar tanto valores simples como diccionarios
        if isinstance(temp_data, dict):
            temp = temp_data.get("T", temp_data.get("T_global", 1.0))
        else:
            temp = temp_data

        if temp > 1.0:
            interpretation = "Sobreconfiado (T > 1.0)"
        elif temp < 1.0:
            interpretation = "Subconfiado (T < 1.0)"
        else:
            interpretation = "Bien calibrado (T = 1.0)"
        print(f"{method:<25} {temp:<15.4f} {interpretation}")
else:
    print("✗ Archivo temperatures.json no encontrado")

# 8. Analizar risk-coverage
print("\n8. RISK-COVERAGE ANALYSIS")
print("-" * 80)

if (FASE5_OUT / "risk_coverage_auc.json").exists():
    with open(FASE5_OUT / "risk_coverage_auc.json") as f:
        rc_auc = json.load(f)

    print(f"{'Método':<25} {'AUC-RC':<12} {'Calidad'}")
    print("-" * 60)

    for method, auc in rc_auc.items():
        if auc > 0.9:
            quality = "Excelente"
        elif auc > 0.8:
            quality = "Buena"
        elif auc > 0.7:
            quality = "Aceptable"
        else:
            quality = "Mejorable"
        print(f"{method:<25} {auc:<12.4f} {quality}")
else:
    print("✗ Archivo risk_coverage_auc.json no encontrado")

# 9. Analizar AUROC de incertidumbre
print("\n9. AUROC DE INCERTIDUMBRE (TP vs FP)")
print("-" * 80)

if (FASE5_OUT / "uncertainty_auroc.json").exists():
    with open(FASE5_OUT / "uncertainty_auroc.json") as f:
        auroc = json.load(f)

    print(f"{'Método':<25} {'AUROC':<12} {'Capacidad Discriminativa'}")
    print("-" * 70)

    for method, auc_data in auroc.items():
        # Manejar tanto valores simples como diccionarios
        if isinstance(auc_data, dict):
            auc = auc_data.get("auroc", auc_data.get("AUROC", 0.5))
        else:
            auc = auc_data

        if auc > 0.7:
            capacity = "Excelente (separa bien TP/FP)"
        elif auc > 0.6:
            capacity = "Buena"
        elif auc > 0.5:
            capacity = "Moderada"
        else:
            capacity = "Pobre (no separa TP/FP)"
        print(f"{method:<25} {auc:<12.4f} {capacity}")
else:
    print("✗ Archivo uncertainty_auroc.json no encontrado")

# 10. Resumen final del reporte
print("\n10. RESUMEN FINAL")
print("-" * 80)

if (FASE5_OUT / "final_report.json").exists():
    with open(FASE5_OUT / "final_report.json") as f:
        final_report = json.load(f)

    if "best_method" in final_report:
        print(f"Mejor método global: {final_report['best_method']}")

    if "rankings" in final_report:
        print("\nRanking por dimensión:")
        for dimension, ranking in final_report["rankings"].items():
            print(f"  {dimension}: {ranking[0] if ranking else 'N/A'}")

    if "recommendations" in final_report:
        print("\nRecomendaciones:")
        for rec in final_report["recommendations"]:
            print(f"  • {rec}")
else:
    print("✗ Archivo final_report.json no encontrado")

# 11. Resumen de verificación
print("\n" + "=" * 80)
print("  RESUMEN DE VERIFICACIÓN")
print("=" * 80)

all_checks = []

# Check JSON files
json_ok = all(json_files_status.values())
all_checks.append(
    (
        "Archivos JSON críticos",
        json_ok,
        f"{sum(json_files_status.values())}/{len(json_files_status)}",
    )
)

# Check plot files
plots_ok = all(plot_files_status.values())
all_checks.append(
    (
        "Visualizaciones",
        plots_ok,
        f"{sum(plot_files_status.values())}/{len(plot_files_status)}",
    )
)

# Check prediction files
preds_ok = all(prediction_files.values())
all_checks.append(
    (
        "Archivos predicciones",
        preds_ok,
        f"{sum(prediction_files.values())}/{len(prediction_files)}",
    )
)

# Check critical outputs
has_detection = (FASE5_OUT / "detection_metrics.json").exists()
has_calibration = (FASE5_OUT / "calibration_metrics.json").exists()
has_temperatures = (FASE5_OUT / "temperatures.json").exists()
has_report = (FASE5_OUT / "final_report.json").exists()

all_checks.append(("Métricas de detección", has_detection, ""))
all_checks.append(("Métricas de calibración", has_calibration, ""))
all_checks.append(("Temperaturas", has_temperatures, ""))
all_checks.append(("Reporte final", has_report, ""))

print()
for check_name, passed, detail in all_checks:
    status = "✓" if passed else "✗"
    detail_str = f" ({detail})" if detail else ""
    print(f"{status} {check_name}{detail_str}")

all_passed = all(check[1] for check in all_checks)

print("\n" + "=" * 80)
if all_passed:
    print("✓✓✓ FASE 5 EJECUTADA EXITOSAMENTE - TODOS LOS OUTPUTS GENERADOS ✓✓✓")
else:
    failed = [check[0] for check in all_checks if not check[1]]
    print(f"✗✗✗ {len(failed)} VERIFICACIONES FALLARON ✗✗✗")
    print(f"Checks fallidos: {', '.join(failed)}")
print("=" * 80 + "\n")
