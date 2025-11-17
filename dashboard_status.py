"""
DASHBOARD DE ESTADO - VERIFICACIÓN RÁPIDA
==========================================
Muestra un resumen visual rápido del estado del proyecto
"""

import json
import pandas as pd
from pathlib import Path


def status_icon(condition):
    return "✅" if condition else "❌"


print("\n" + "=" * 70)
print(" " * 20 + "DASHBOARD DE ESTADO")
print("=" * 70 + "\n")

# Estado de archivos críticos
print("📦 ARCHIVOS CRÍTICOS:")
files = {
    "Fase 2 - Baseline": Path("fase 2/outputs/baseline/preds_raw.json"),
    "Fase 3 - MC Stats": Path("fase 3/outputs/mc_dropout/mc_stats_labeled.parquet"),
    "Fase 3 - Timing": Path("fase 3/outputs/mc_dropout/timing_data.parquet"),
    "Fase 4 - Temps": Path("fase 4/outputs/temperature_scaling/temperature.json"),
    "Fase 4 - Calib": Path("fase 4/outputs/temperature_scaling/calib_detections.csv"),
    "Fase 4 - Eval": Path("fase 4/outputs/temperature_scaling/eval_detections.csv"),
}

for name, path in files.items():
    exists = path.exists()
    size = ""
    if exists:
        size_bytes = path.stat().st_size
        if size_bytes > 1024 * 1024:
            size = f"({size_bytes/(1024*1024):.1f}MB)"
        elif size_bytes > 1024:
            size = f"({size_bytes/1024:.1f}KB)"
    print(f"  {status_icon(exists)} {name:20s} {size}")

print("\n" + "-" * 70 + "\n")

# Análisis de cobertura
print("📊 COBERTURA DE DATOS:")

fase2_path = Path("fase 2/outputs/baseline/preds_raw.json")
fase3_path = Path("fase 3/outputs/mc_dropout/mc_stats_labeled.parquet")

if fase2_path.exists():
    with open(fase2_path, "r") as f:
        baseline = json.load(f)
    fase2_images = len(set(p["image_id"] for p in baseline))
else:
    fase2_images = 0

if fase3_path.exists():
    df = pd.read_parquet(fase3_path)
    fase3_images = df["image_id"].nunique()
    has_uncertainty = "uncertainty" in df.columns and (df["uncertainty"] > 0).any()
else:
    fase3_images = 0
    has_uncertainty = False

coverage = (fase3_images / fase2_images * 100) if fase2_images > 0 else 0

print(
    f"  Fase 2 (Baseline):   {fase2_images:,} imágenes {status_icon(fase2_images > 0)}"
)
print(
    f"  Fase 3 (MC-Dropout): {fase3_images:,} imágenes {status_icon(fase3_images > 1000)}"
)
print(f"  Cobertura MC-Dropout: {coverage:.1f}% {status_icon(coverage > 90)}")

print("\n" + "-" * 70 + "\n")

# Estado de variables críticas
print("🔍 VARIABLES CRÍTICAS:")

if fase3_path.exists():
    critical_vars = [
        "image_id",
        "category_id",
        "bbox",
        "score_mean",
        "score_std",
        "score_var",
        "uncertainty",
        "num_passes",
        "is_tp",
        "max_iou",
    ]

    missing = [v for v in critical_vars if v not in df.columns]

    print(
        f"  {status_icon(len(missing) == 0)} Campos requeridos: {len(critical_vars) - len(missing)}/{len(critical_vars)}"
    )
    print(f"  {status_icon(has_uncertainty)} Campo 'uncertainty' con valores > 0")

    if "uncertainty" in df.columns:
        unc_stats = df["uncertainty"]
        print(f"     → Min: {unc_stats.min():.6f}, Max: {unc_stats.max():.6f}")

print("\n" + "-" * 70 + "\n")

# Diagnóstico
print("💡 DIAGNÓSTICO:")

all_files_exist = all(path.exists() for path in files.values())
full_coverage = coverage > 90
all_vars_present = fase3_path.exists() and len(missing) == 0 and has_uncertainty

if all_files_exist and full_coverage and all_vars_present:
    status = "🎉 EXCELENTE"
    message = "Todo está listo para ejecutar Fase 5"
    color = "verde"
elif all_files_exist and all_vars_present:
    status = "⚠️  PARCIAL"
    message = "Archivos presentes pero cobertura incompleta"
    color = "amarillo"
else:
    status = "❌ INCOMPLETO"
    message = "Faltan archivos o variables críticas"
    color = "rojo"

print(f"  Estado: {status}")
print(f"  Mensaje: {message}")

print("\n" + "-" * 70 + "\n")

# Acciones recomendadas
print("📋 ACCIONES RECOMENDADAS:")

if not all_files_exist:
    print("  1. ⚠️  Ejecutar fases faltantes para generar archivos")
elif not full_coverage:
    print("  1. 🔴 CRÍTICO: Ejecutar Fase 3 completa (sin limitación [:100])")
    print("  2. 🟡 Opcional: Re-ejecutar Fase 4 con datos completos")
    print("  3. 🟢 Final: Ejecutar Fase 5 para comparación completa")
else:
    print("  1. ✅ Ejecutar Fase 5 (todo listo)")

print("\n" + "=" * 70 + "\n")

# Tiempo estimado
print("⏱️  TIEMPO ESTIMADO:")
if not full_coverage:
    print("  • Fase 3 completa: ~2-3 horas")
    print("  • Fase 4 re-run: ~30 minutos")
    print("  • Fase 5 final: ~15 minutos")
    print("  • TOTAL: ~3-4 horas")
else:
    print("  • Fase 5: ~15 minutos")

print("\n" + "=" * 70 + "\n")

# Código de salida
exit_code = 0 if (all_files_exist and full_coverage and all_vars_present) else 1
print(f"Estado: {'READY' if exit_code == 0 else 'NOT READY'}")
print(f"Exit code: {exit_code}\n")
