"""
VERIFICACIÓN DE VARIABLES PARA FASE 5
======================================
Este script verifica que todas las variables necesarias para Fase 5
estén disponibles y en el formato correcto.
"""

import json
import pandas as pd
from pathlib import Path
import sys


def check_variable(name, condition, details=""):
    """Verifica una condición y reporta el resultado"""
    status = "✅" if condition else "❌"
    result = "PASS" if condition else "FAIL"
    print(f"{status} {name}: {result}")
    if details:
        print(f"   {details}")
    return condition


def main():
    print("=" * 70)
    print("  VERIFICACIÓN DE VARIABLES PARA FASE 5")
    print("=" * 70)
    print()

    all_checks_passed = True

    # ========================================================================
    # 1. VERIFICAR FASE 2: BASELINE
    # ========================================================================
    print("📦 FASE 2: Variables de Baseline")
    print("-" * 70)

    fase2_path = Path("fase 2/outputs/baseline/preds_raw.json")
    fase2_exists = check_variable("Archivo preds_raw.json existe", fase2_path.exists())
    all_checks_passed &= fase2_exists

    if fase2_exists:
        with open(fase2_path, "r") as f:
            baseline = json.load(f)

        check_variable(
            "Baseline tiene predicciones",
            len(baseline) > 0,
            f"Total: {len(baseline)} predicciones",
        )

        if len(baseline) > 0:
            sample = baseline[0]
            has_image_id = check_variable(
                "Campo 'image_id' presente", "image_id" in sample
            )
            has_category_id = check_variable(
                "Campo 'category_id' presente", "category_id" in sample
            )
            has_bbox = check_variable("Campo 'bbox' presente", "bbox" in sample)
            has_score = check_variable("Campo 'score' presente", "score" in sample)

            all_checks_passed &= (
                has_image_id and has_category_id and has_bbox and has_score
            )

            # Verificar número de imágenes
            unique_images = len(set(p["image_id"] for p in baseline))
            check_variable(
                "Dataset completo (>1000 imágenes)",
                unique_images > 1000,
                f"Imágenes únicas: {unique_images}",
            )

    print()

    # ========================================================================
    # 2. VERIFICAR FASE 3: MC-DROPOUT
    # ========================================================================
    print("📦 FASE 3: Variables de MC-Dropout")
    print("-" * 70)

    fase3_parquet = Path("fase 3/outputs/mc_dropout/mc_stats_labeled.parquet")
    fase3_exists = check_variable(
        "Archivo mc_stats_labeled.parquet existe", fase3_parquet.exists()
    )
    all_checks_passed &= fase3_exists

    if fase3_exists:
        df = pd.read_parquet(fase3_parquet)

        check_variable(
            "MC-Dropout tiene detecciones", len(df) > 0, f"Total: {len(df)} detecciones"
        )

        # Campos críticos para Fase 5
        critical_fields = {
            "image_id": "Identificador de imagen",
            "category_id": "Categoría de objeto",
            "bbox": "Bounding box",
            "score_mean": "Score promedio (K pases)",
            "score_std": "Desviación estándar de scores",
            "score_var": "Varianza de scores",
            "uncertainty": "Métrica de incertidumbre principal",
            "num_passes": "Número de pases MC-Dropout",
            "is_tp": "True/False Positive",
            "max_iou": "IoU máximo con ground truth",
        }

        print()
        print("   🔍 Campos críticos:")
        for field, description in critical_fields.items():
            field_exists = field in df.columns
            check_variable(f"   {field}", field_exists, description)
            all_checks_passed &= field_exists

        print()

        # Verificar calidad de datos
        if "uncertainty" in df.columns:
            has_nonzero = (df["uncertainty"] > 0).any()
            check_variable(
                "Uncertainty tiene valores > 0",
                has_nonzero,
                f"Min: {df['uncertainty'].min():.6f}, Max: {df['uncertainty'].max():.6f}",
            )
            all_checks_passed &= has_nonzero

        # Verificar cobertura de imágenes
        unique_mc_images = df["image_id"].nunique()
        is_complete = check_variable(
            "Dataset MC-Dropout completo (>1000 imágenes)",
            unique_mc_images > 1000,
            f"Imágenes únicas: {unique_mc_images}",
        )

        if not is_complete:
            print()
            print("   ⚠️  ADVERTENCIA: MC-Dropout no procesó todas las imágenes")
            print("   → Esto causará que Fase 5 use fallback a baseline")
            print("   → Resultado: temperaturas idénticas en calib/eval")
            all_checks_passed = False

    print()

    # ========================================================================
    # 3. VERIFICAR FASE 4: TEMPERATURE SCALING
    # ========================================================================
    print("📦 FASE 4: Variables de Temperature Scaling")
    print("-" * 70)

    fase4_temp = Path("fase 4/outputs/temperature_scaling/temperature.json")
    temp_exists = check_variable("Archivo temperature.json existe", fase4_temp.exists())
    all_checks_passed &= temp_exists

    if temp_exists:
        with open(fase4_temp, "r") as f:
            temps = json.load(f)

        has_tglobal = check_variable(
            "Campo 'T_global' presente",
            "T_global" in temps,
            f"Valor: {temps.get('T_global', 'N/A')}" if "T_global" in temps else "",
        )

        # Verificar si tiene temperaturas optimizadas o solo global
        if "optimal_temperature" in temps:
            print("   ℹ️  Formato con 'optimal_temperature'")
        elif "T_global" in temps:
            print("   ℹ️  Formato con 'T_global' (equivalente)")

        if "per_class_temperature" in temps:
            print(
                f"   ✅ Temperaturas por clase: {len(temps['per_class_temperature'])} clases"
            )

    # Verificar CSVs de calibración/evaluación
    fase4_calib = Path("fase 4/outputs/temperature_scaling/calib_detections.csv")
    check_variable("Archivo calib_detections.csv existe", fase4_calib.exists())

    fase4_eval = Path("fase 4/outputs/temperature_scaling/eval_detections.csv")
    check_variable("Archivo eval_detections.csv existe", fase4_eval.exists())

    print()

    # ========================================================================
    # 4. VERIFICAR FORMATO Y COMPATIBILIDAD
    # ========================================================================
    print("📦 COMPATIBILIDAD Y FORMATO")
    print("-" * 70)

    if fase2_exists and fase3_exists:
        baseline_images = set(p["image_id"] for p in baseline)
        mc_images = set(df["image_id"].unique())

        common = baseline_images & mc_images
        coverage = len(common) / len(baseline_images) * 100

        check_variable(
            "Cobertura de imágenes MC-Dropout",
            coverage > 90,
            f"{coverage:.1f}% ({len(common)}/{len(baseline_images)} imágenes)",
        )

        if coverage < 90:
            print()
            print("   ⚠️  PROBLEMA: Cobertura insuficiente")
            print("   → Fase 3 necesita re-ejecutarse con todas las imágenes")
            all_checks_passed = False

    # Verificar formato de bbox
    if fase3_exists and "bbox" in df.columns:
        sample_bbox = df["bbox"].iloc[0]
        print()
        print(f"   ℹ️  Formato bbox: {sample_bbox}")
        if len(sample_bbox) == 4:
            print("   ✅ Bbox tiene 4 coordenadas (formato válido)")

    print()

    # ========================================================================
    # RESULTADO FINAL
    # ========================================================================
    print("=" * 70)
    print("  RESULTADO FINAL")
    print("=" * 70)
    print()

    if all_checks_passed:
        print("🎉 ÉXITO: Todas las verificaciones pasaron")
        print()
        print("✅ Fase 5 puede ejecutarse correctamente")
        print("✅ Todas las variables críticas están disponibles")
        print("✅ Formato de datos es correcto")
        print()
        print("➡️  SIGUIENTE PASO: Ejecutar fase 5/main.ipynb")
        return 0
    else:
        print("❌ FALLO: Algunas verificaciones no pasaron")
        print()
        print("⚠️  Problemas detectados:")
        print()

        if not fase3_exists or (fase3_exists and df["image_id"].nunique() < 1000):
            print("   1. Fase 3 incompleta (solo 100 imágenes procesadas)")
            print("      → Acción: Ejecutar fase 3/main.ipynb completo")
            print("      → Tiempo: ~2-3 horas")
            print()

        if not temp_exists:
            print("   2. Temperaturas no calculadas")
            print("      → Acción: Ejecutar fase 4/main.ipynb")
            print("      → Tiempo: ~30 minutos")
            print()

        print("⚠️  Fase 5 puede ejecutarse pero:")
        print("   - Usará fallback a baseline para imágenes sin caché MC-Dropout")
        print("   - Las temperaturas serán idénticas en calib/eval")
        print("   - Los resultados no serán representativos del workflow completo")
        print()
        print("➡️  RECOMENDACIÓN: Completar Fase 3 antes de ejecutar Fase 5")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
