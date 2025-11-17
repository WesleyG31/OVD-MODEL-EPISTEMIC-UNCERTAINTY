"""
VERIFICACIÓN COMPLETA DEL FLUJO DE VARIABLES
=============================================
Este script verifica que todas las variables necesarias estén presentes
y correctamente configuradas en el flujo completo Fase 2 → Fase 3 → Fase 4 → Fase 5
"""

import json
import pandas as pd
from pathlib import Path
import numpy as np


def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def check_file_exists(path, description):
    """Verifica si un archivo existe y muestra su tamaño"""
    path = Path(path)
    exists = path.exists()
    status = "✅" if exists else "❌"

    size_info = ""
    if exists:
        size = path.stat().st_size
        if size > 1024 * 1024:
            size_info = f"({size/(1024*1024):.2f} MB)"
        elif size > 1024:
            size_info = f"({size/1024:.2f} KB)"
        else:
            size_info = f"({size} bytes)"

    print(f"{status} {description}")
    print(f"   Ruta: {path}")
    print(f"   {size_info if exists else 'NO EXISTE'}")

    return exists


# ============================================================================
# 1. VERIFICAR ARCHIVOS DE FASE 2 (BASELINE)
# ============================================================================
print_section("FASE 2: BASELINE")

fase2_preds = Path("fase 2/outputs/baseline/preds_raw.json")
if check_file_exists(fase2_preds, "Predicciones baseline"):
    with open(fase2_preds, "r") as f:
        baseline_data = json.load(f)

    print(f"\n📊 Análisis de predicciones baseline:")
    print(f"   - Total predicciones: {len(baseline_data)}")

    # Verificar estructura
    if len(baseline_data) > 0:
        sample = baseline_data[0]
        print(f"   - Campos en cada predicción: {list(sample.keys())}")

        required_fields = ["image_id", "category_id", "bbox", "score"]
        missing_fields = [f for f in required_fields if f not in sample]
        if missing_fields:
            print(f"   ⚠️  CAMPOS FALTANTES: {missing_fields}")
        else:
            print(f"   ✅ Todos los campos requeridos presentes")

        # Verificar imágenes únicas
        unique_images = len(set(p["image_id"] for p in baseline_data))
        print(f"   - Imágenes únicas: {unique_images}")
        print(
            f"   - Promedio predicciones por imagen: {len(baseline_data)/unique_images:.2f}"
        )

# ============================================================================
# 2. VERIFICAR ARCHIVOS DE FASE 3 (MC-DROPOUT)
# ============================================================================
print_section("FASE 3: MC-DROPOUT")

# Archivo principal con incertidumbre
fase3_parquet = Path("fase 3/outputs/mc_dropout/mc_stats_labeled.parquet")
if check_file_exists(fase3_parquet, "MC-Dropout stats (PARQUET con incertidumbre)"):
    df = pd.read_parquet(fase3_parquet)

    print(f"\n📊 Análisis de MC-Dropout stats:")
    print(f"   - Total detecciones: {len(df)}")
    print(f"   - Campos disponibles: {list(df.columns)}")

    # Verificar campos críticos
    critical_fields = [
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

    print(f"\n   🔍 Verificación de campos críticos:")
    for field in critical_fields:
        if field in df.columns:
            print(f"   ✅ {field}: presente")
            # Mostrar estadísticas básicas
            if field in ["score_mean", "score_std", "score_var", "uncertainty"]:
                print(
                    f"      → Min: {df[field].min():.6f}, Max: {df[field].max():.6f}, Mean: {df[field].mean():.6f}"
                )
        else:
            print(f"   ❌ {field}: FALTANTE")

    # Verificar imágenes únicas
    unique_images_mc = df["image_id"].nunique()
    print(f"\n   - Imágenes únicas: {unique_images_mc}")
    print(f"   - Promedio detecciones por imagen: {len(df)/unique_images_mc:.2f}")

    # Verificar distribución de uncertainty
    print(f"\n   📈 Distribución de incertidumbre (uncertainty):")
    print(f"      - Min: {df['uncertainty'].min():.6f}")
    print(f"      - 25%: {df['uncertainty'].quantile(0.25):.6f}")
    print(f"      - 50%: {df['uncertainty'].quantile(0.50):.6f}")
    print(f"      - 75%: {df['uncertainty'].quantile(0.75):.6f}")
    print(f"      - Max: {df['uncertainty'].max():.6f}")

    # Verificar si todas son cero (indicaría problema)
    if (df["uncertainty"] == 0).all():
        print(f"   ⚠️  ADVERTENCIA: Todas las incertidumbres son 0")
    elif (df["uncertainty"] > 0).any():
        print(f"   ✅ Incertidumbres tienen valores > 0")

    # Verificar TP/FP
    if "is_tp" in df.columns:
        tp_count = df["is_tp"].sum()
        fp_count = (~df["is_tp"]).sum()
        print(f"\n   📊 Balance TP/FP:")
        print(f"      - True Positives: {tp_count} ({100*tp_count/len(df):.2f}%)")
        print(f"      - False Positives: {fp_count} ({100*fp_count/len(df):.2f}%)")

# Archivo JSON agregado
fase3_json = Path("fase 3/outputs/mc_dropout/preds_mc_aggregated.json")
if check_file_exists(fase3_json, "MC-Dropout predicciones agregadas (JSON)"):
    with open(fase3_json, "r") as f:
        mc_preds = json.load(f)

    print(f"\n📊 Análisis de predicciones MC-Dropout:")
    print(f"   - Total predicciones: {len(mc_preds)}")

    if len(mc_preds) > 0:
        sample_mc = mc_preds[0]
        print(f"   - Campos: {list(sample_mc.keys())}")

        # Verificar si tiene uncertainty
        if "uncertainty" in sample_mc:
            print(f"   ✅ Campo 'uncertainty' presente en JSON")
            uncertainties = [p.get("uncertainty", 0) for p in mc_preds]
            print(f"   - Min uncertainty: {min(uncertainties):.6f}")
            print(f"   - Max uncertainty: {max(uncertainties):.6f}")
        else:
            print(f"   ⚠️  Campo 'uncertainty' NO presente en JSON")
            print(f"   → RECOMENDACIÓN: Usar mc_stats_labeled.parquet en su lugar")

# Timing data
fase3_timing = Path("fase 3/outputs/mc_dropout/timing_data.parquet")
check_file_exists(fase3_timing, "Datos de timing")

# ============================================================================
# 3. VERIFICAR ARCHIVOS DE FASE 4 (TEMPERATURE SCALING)
# ============================================================================
print_section("FASE 4: TEMPERATURE SCALING")

fase4_temp = Path("fase 4/outputs/temperature_scaling/temperature.json")
if check_file_exists(fase4_temp, "Temperaturas optimizadas"):
    with open(fase4_temp, "r") as f:
        temp_data = json.load(f)

    print(f"\n📊 Análisis de temperaturas:")
    print(f"   - Campos: {list(temp_data.keys())}")

    if "optimal_temperature" in temp_data:
        print(f"   - Temperatura global: {temp_data['optimal_temperature']:.4f}")

    if "per_class_temperature" in temp_data:
        print(f"   - Temperaturas por clase:")
        for cat_id, temp in temp_data["per_class_temperature"].items():
            print(f"      • Clase {cat_id}: {temp:.4f}")

fase4_calib = Path("fase 4/outputs/temperature_scaling/calib_detections.csv")
if check_file_exists(fase4_calib, "Detecciones de calibración"):
    calib_df = pd.read_csv(fase4_calib)
    print(f"\n📊 Análisis de datos de calibración:")
    print(f"   - Total detecciones: {len(calib_df)}")
    print(f"   - Campos: {list(calib_df.columns)}")

    # Verificar campos esperados
    expected_calib_fields = ["logit", "score", "category", "is_tp", "iou"]
    for field in expected_calib_fields:
        if field in calib_df.columns:
            print(f"   ✅ {field}: presente")
        else:
            print(f"   ⚠️  {field}: FALTANTE")

fase4_eval = Path("fase 4/outputs/temperature_scaling/eval_detections.csv")
if check_file_exists(fase4_eval, "Detecciones de evaluación"):
    eval_df = pd.read_csv(fase4_eval)
    print(f"\n📊 Análisis de datos de evaluación:")
    print(f"   - Total detecciones: {len(eval_df)}")
    print(f"   - Campos: {list(eval_df.columns)}")

    # Verificar campos esperados
    expected_eval_fields = ["logit", "score", "category", "is_tp", "iou"]
    for field in expected_eval_fields:
        if field in eval_df.columns:
            print(f"   ✅ {field}: presente")
        else:
            print(f"   ⚠️  {field}: FALTANTE")

# ============================================================================
# 4. VERIFICAR COMPATIBILIDAD ENTRE FASES
# ============================================================================
print_section("COMPATIBILIDAD ENTRE FASES")

# Verificar que las imágenes coincidan
if fase2_preds.exists() and fase3_parquet.exists():
    baseline_images = set(p["image_id"] for p in baseline_data)
    mc_images = set(df["image_id"].unique())

    print(f"📊 Comparación de conjuntos de imágenes:")
    print(f"   - Fase 2 (Baseline): {len(baseline_images)} imágenes")
    print(f"   - Fase 3 (MC-Dropout): {len(mc_images)} imágenes")

    common_images = baseline_images & mc_images
    only_baseline = baseline_images - mc_images
    only_mc = mc_images - baseline_images

    print(f"   - Imágenes en común: {len(common_images)}")

    if only_baseline:
        print(f"   ⚠️  {len(only_baseline)} imágenes solo en Baseline")
        if len(only_baseline) <= 10:
            print(f"      IDs: {sorted(only_baseline)}")

    if only_mc:
        print(f"   ⚠️  {len(only_mc)} imágenes solo en MC-Dropout")
        if len(only_mc) <= 10:
            print(f"      IDs: {sorted(only_mc)}")

    if len(common_images) == len(baseline_images) == len(mc_images):
        print(f"   ✅ Todas las imágenes coinciden entre fases")

# ============================================================================
# 5. VERIFICAR FORMATO DE BBOX
# ============================================================================
print_section("FORMATO DE BOUNDING BOXES")

if fase3_parquet.exists():
    print(f"🔍 Verificando formato de bbox en mc_stats_labeled.parquet:")

    # Tomar muestra de bboxes
    sample_bboxes = df["bbox"].head(5).tolist()

    for i, bbox in enumerate(sample_bboxes):
        print(f"\n   Ejemplo {i+1}: {bbox}")

        # Verificar formato
        if isinstance(bbox, (list, np.ndarray)) and len(bbox) == 4:
            # Detectar formato
            x1, y1, x2_or_w, y2_or_h = bbox

            # Si x2 > x1 y y2 > y1 pero no mucho, probablemente sea xywh
            if x2_or_w > x1 and y2_or_h > y1:
                # Podría ser xyxy
                if x2_or_w < x1 + 2000 and y2_or_h < y1 + 2000:  # Rangos razonables
                    print(f"      Formato probable: XYXY")
                else:
                    print(f"      Formato probable: XYWH")
            else:
                print(f"      ⚠️  Formato ambiguo o inválido")

# ============================================================================
# 6. RESUMEN FINAL
# ============================================================================
print_section("RESUMEN FINAL Y RECOMENDACIONES")

# Contar archivos existentes
critical_files = {
    "Fase 2 - Baseline": fase2_preds.exists(),
    "Fase 3 - MC Stats (Parquet)": fase3_parquet.exists(),
    "Fase 3 - MC Preds (JSON)": fase3_json.exists(),
    "Fase 4 - Temperaturas": fase4_temp.exists(),
    "Fase 4 - Calib Data": fase4_calib.exists(),
    "Fase 4 - Eval Data": fase4_eval.exists(),
}

print(f"Estado de archivos críticos:")
for name, exists in critical_files.items():
    status = "✅" if exists else "❌"
    print(f"   {status} {name}")

print(f"\n{'='*70}")

# Recomendaciones basadas en hallazgos
if not fase3_parquet.exists():
    print(f"\n⚠️  ACCIÓN REQUERIDA:")
    print(f"   1. Ejecutar Fase 3 completa para generar mc_stats_labeled.parquet")
    print(f"   2. Asegurarse de procesar TODAS las imágenes (no solo [:100])")
elif fase3_parquet.exists():
    df_check = pd.read_parquet(fase3_parquet)
    if len(df_check["image_id"].unique()) < 1000:
        print(f"\n⚠️  ACCIÓN REQUERIDA:")
        print(f"   - Fase 3 solo procesó {df_check['image_id'].nunique()} imágenes")
        print(f"   - Se esperan ~2000 imágenes en el dataset completo")
        print(f"   - Re-ejecutar Fase 3 sin limitaciones [:100]")
    else:
        print(
            f"\n✅ Fase 3 procesó {df_check['image_id'].nunique()} imágenes (dataset completo)"
        )

if not fase4_temp.exists():
    print(f"\n⚠️  ACCIÓN REQUERIDA:")
    print(f"   - Ejecutar Fase 4 para calcular temperaturas optimizadas")

if all(critical_files.values()):
    print(f"\n🎉 EXCELENTE: Todos los archivos críticos están presentes")
    print(f"   → Fase 5 puede ejecutarse con caché completo")
    print(f"   → Tiempo estimado: ~15 minutos vs ~2 horas sin caché")

print(f"\n{'='*70}")
print(f"VERIFICACIÓN COMPLETADA")
print(f"{'='*70}\n")
