# Script para verificar que todas las variables necesarias se guardan correctamente

"""
Verifica que los archivos de salida de Fase 3 y Fase 5 contienen
todas las variables necesarias para el flujo completo.

Uso:
    python verify_saved_variables.py
"""

import json
import pandas as pd
from pathlib import Path
import sys


def print_section(title):
    """Imprime encabezado de sección"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def check_fase3_outputs():
    """Verifica outputs de Fase 3"""
    print_section("VERIFICACIÓN: FASE 3 - MC-DROPOUT")

    fase3_dir = Path("fase 3/outputs/mc_dropout")

    required_files = {
        "mc_stats.parquet": "Estadísticas base (sin TP/FP)",
        "mc_stats_labeled.parquet": "Estadísticas con etiquetas TP/FP",
        "preds_mc_aggregated.json": "Predicciones agregadas formato COCO",
        "timing_data.parquet": "Tiempos de inferencia",
        "metrics.json": "Métricas de detección (mAP)",
        "tp_fp_analysis.json": "Análisis de incertidumbre TP/FP",
    }

    all_ok = True

    for filename, description in required_files.items():
        filepath = fase3_dir / filename
        exists = filepath.exists()
        status = "✅" if exists else "❌"
        print(f"{status} {filename}")
        print(f"   {description}")

        if not exists:
            all_ok = False
            print(f"   ⚠️  Archivo no encontrado: {filepath}")
            continue

        # Verificar contenido según tipo de archivo
        try:
            if filename.endswith(".parquet"):
                df = pd.read_parquet(filepath)
                print(f"   📊 {len(df)} filas")

                # Verificar columnas específicas
                if filename == "mc_stats_labeled.parquet":
                    required_cols = [
                        "image_id",
                        "category_id",
                        "bbox",
                        "score_mean",
                        "score_std",
                        "score_var",
                        "uncertainty",
                        "is_tp",
                        "max_iou",
                    ]
                    missing_cols = set(required_cols) - set(df.columns)

                    if missing_cols:
                        print(f"   ❌ Columnas faltantes: {missing_cols}")
                        all_ok = False
                    else:
                        print(f"   ✅ Todas las columnas necesarias presentes")
                        print(f"   📈 Columnas: {list(df.columns)}")
                        print(f"   🖼️  Imágenes únicas: {df['image_id'].nunique()}")
                        print(f"   📦 Detecciones totales: {len(df)}")

                        # Verificar que uncertainty está presente y no es todo 0
                        if "uncertainty" in df.columns:
                            unc_mean = df["uncertainty"].mean()
                            unc_std = df["uncertainty"].std()
                            unc_min = df["uncertainty"].min()
                            unc_max = df["uncertainty"].max()
                            print(f"   📊 Uncertainty stats:")
                            print(f"      Mean: {unc_mean:.6f}, Std: {unc_std:.6f}")
                            print(f"      Min: {unc_min:.6f}, Max: {unc_max:.6f}")

                            if unc_max == 0:
                                print(
                                    f"   ⚠️  ADVERTENCIA: Uncertainty es 0 para todas las detecciones"
                                )

                        # Verificar TP/FP
                        if "is_tp" in df.columns:
                            tp_count = df["is_tp"].sum()
                            fp_count = (~df["is_tp"]).sum()
                            print(f"   ✅ TP: {tp_count}, FP: {fp_count}")

                elif filename == "mc_stats.parquet":
                    required_cols = [
                        "image_id",
                        "category_id",
                        "bbox",
                        "score_mean",
                        "score_std",
                        "score_var",
                        "uncertainty",
                    ]
                    missing_cols = set(required_cols) - set(df.columns)

                    if missing_cols:
                        print(f"   ❌ Columnas faltantes: {missing_cols}")
                        all_ok = False
                    else:
                        print(f"   ✅ Todas las columnas necesarias presentes")
                        print(f"   🖼️  Imágenes únicas: {df['image_id'].nunique()}")

                elif filename == "timing_data.parquet":
                    required_cols = ["image_id", "time_seconds", "num_detections"]
                    missing_cols = set(required_cols) - set(df.columns)

                    if missing_cols:
                        print(f"   ❌ Columnas faltantes: {missing_cols}")
                        all_ok = False
                    else:
                        print(f"   ✅ Columnas correctas")
                        print(
                            f"   ⏱️  Tiempo promedio: {df['time_seconds'].mean():.2f}s"
                        )

            elif filename.endswith(".json"):
                with open(filepath) as f:
                    data = json.load(f)

                if filename == "preds_mc_aggregated.json":
                    print(f"   📦 {len(data)} predicciones")
                    if len(data) > 0:
                        sample = data[0]
                        required_keys = ["image_id", "category_id", "bbox", "score"]
                        missing_keys = set(required_keys) - set(sample.keys())
                        if missing_keys:
                            print(f"   ❌ Keys faltantes: {missing_keys}")
                            all_ok = False
                        else:
                            print(f"   ✅ Keys correctas: {list(sample.keys())}")

                elif filename == "metrics.json":
                    print(f"   📊 Métricas disponibles:")
                    for key, value in data.items():
                        print(f"      {key}: {value:.4f}")

                elif filename == "tp_fp_analysis.json":
                    required_keys = [
                        "auroc_uncertainty",
                        "num_tp",
                        "num_fp",
                        "uncertainty_tp_mean",
                        "uncertainty_fp_mean",
                    ]
                    missing_keys = set(required_keys) - set(data.keys())
                    if missing_keys:
                        print(f"   ❌ Keys faltantes: {missing_keys}")
                        all_ok = False
                    else:
                        print(f"   ✅ Keys correctas")
                        print(f"   📊 AUROC: {data['auroc_uncertainty']:.4f}")
                        print(f"   📦 TP: {data['num_tp']}, FP: {data['num_fp']}")

        except Exception as e:
            print(f"   ❌ Error al leer archivo: {e}")
            all_ok = False

    return all_ok


def check_fase5_inputs():
    """Verifica que Fase 5 puede cargar todos los inputs necesarios"""
    print_section("VERIFICACIÓN: FASE 5 - INPUTS NECESARIOS")

    inputs_needed = {
        "fase 2/outputs/baseline/preds_raw.json": "Predicciones baseline",
        "fase 3/outputs/mc_dropout/mc_stats_labeled.parquet": "MC-Dropout con incertidumbre",
        "fase 4/outputs/temperature_scaling/temperature.json": "Temperatura optimizada",
        "data/bdd100k_coco/val_eval.json": "Anotaciones val_eval",
    }

    all_ok = True

    for filepath_str, description in inputs_needed.items():
        filepath = Path(filepath_str)
        exists = filepath.exists()
        status = "✅" if exists else "❌"
        print(f"{status} {filepath}")
        print(f"   {description}")

        if not exists:
            all_ok = False
            print(f"   ⚠️  FALTA: Fase 5 no podrá usar cache para este componente")

    return all_ok


def check_fase5_outputs():
    """Verifica outputs de Fase 5"""
    print_section("VERIFICACIÓN: FASE 5 - OUTPUTS ESPERADOS")

    fase5_dir = Path("outputs/comparison")

    if not fase5_dir.exists():
        print("⚠️  Directorio outputs/comparison no existe")
        print("   (Es normal si Fase 5 aún no se ha ejecutado)")
        return None

    expected_files = {
        "temperatures.json": "Temperaturas por método",
        "calib_baseline.csv": "Datos calibración baseline",
        "calib_mc_dropout.csv": "Datos calibración MC-Dropout",
        "calib_decoder_variance.csv": "Datos calibración decoder variance",
        "eval_baseline.csv": "Evaluación baseline",
        "eval_mc_dropout.csv": "Evaluación MC-Dropout",
        "eval_decoder_variance.csv": "Evaluación decoder variance",
        "final_report.json": "Reporte final",
    }

    all_ok = True

    for filename, description in expected_files.items():
        filepath = fase5_dir / filename
        exists = filepath.exists()
        status = "✅" if exists else "⚠️"
        print(f"{status} {filename}")
        print(f"   {description}")

        if not exists:
            print(f"   (Será generado cuando se ejecute Fase 5)")
            continue

        try:
            if filename == "temperatures.json":
                with open(filepath) as f:
                    temps = json.load(f)

                print(f"   📊 Temperaturas:")
                for method, data in temps.items():
                    if isinstance(data, dict) and "T" in data:
                        print(f"      {method}: T={data['T']:.4f}")

                # Verificar que son diferentes
                t_values = [
                    data["T"]
                    for data in temps.values()
                    if isinstance(data, dict) and "T" in data
                ]
                if len(set(t_values)) == 1:
                    print(f"   ⚠️  ADVERTENCIA: Todas las temperaturas son iguales!")
                    all_ok = False
                else:
                    print(f"   ✅ Temperaturas son diferentes (correcto)")

            elif filename.endswith(".csv"):
                df = pd.read_csv(filepath)
                print(f"   📊 {len(df)} filas")

                # Verificar columnas importantes
                required_cols = ["logit", "score", "uncertainty", "is_tp"]
                missing_cols = set(required_cols) - set(df.columns)

                if missing_cols:
                    print(f"   ❌ Columnas faltantes: {missing_cols}")
                    all_ok = False
                else:
                    print(f"   ✅ Columnas correctas")

                    # Verificar uncertainty
                    if "mc_dropout" in filename or "decoder" in filename:
                        unc_mean = df["uncertainty"].mean()
                        unc_max = df["uncertainty"].max()
                        print(
                            f"   📈 Uncertainty: mean={unc_mean:.6f}, max={unc_max:.6f}"
                        )

                        if unc_max == 0:
                            print(
                                f"   ⚠️  ADVERTENCIA: Uncertainty es 0 para todas las detecciones"
                            )

        except Exception as e:
            print(f"   ❌ Error al leer archivo: {e}")
            all_ok = False

    return all_ok


def main():
    """Función principal"""
    print("=" * 70)
    print("  VERIFICACIÓN DE VARIABLES GUARDADAS")
    print("=" * 70)
    print("\nEste script verifica que todos los archivos necesarios existen")
    print("y contienen las variables correctas para el flujo completo.\n")

    # Verificar Fase 3
    fase3_ok = check_fase3_outputs()

    # Verificar inputs de Fase 5
    fase5_inputs_ok = check_fase5_inputs()

    # Verificar outputs de Fase 5 (si existen)
    fase5_outputs_ok = check_fase5_outputs()

    # Resumen final
    print_section("RESUMEN FINAL")

    print("\n📋 Estado de las fases:")
    print(f"  Fase 3 outputs: {'✅ OK' if fase3_ok else '❌ PROBLEMAS'}")
    print(f"  Fase 5 inputs:  {'✅ OK' if fase5_inputs_ok else '❌ PROBLEMAS'}")

    if fase5_outputs_ok is None:
        print(f"  Fase 5 outputs: ⏳ Pendiente (Fase 5 no ejecutada)")
    else:
        print(f"  Fase 5 outputs: {'✅ OK' if fase5_outputs_ok else '❌ PROBLEMAS'}")

    print("\n" + "=" * 70)

    if not fase3_ok:
        print("\n⚠️  ACCIÓN REQUERIDA:")
        print("   Fase 3 tiene problemas. Necesitas correr/re-correr:")
        print("   → Abrir fase 3/main.ipynb")
        print("   → Run All Cells")
        print("   → Esperar ~6-7 horas")
        return 1

    if not fase5_inputs_ok:
        print("\n⚠️  ACCIÓN REQUERIDA:")
        print("   Faltan inputs para Fase 5.")
        print("   Fase 5 funcionará pero sin aprovechar el cache completo.")
        return 1

    if fase5_outputs_ok is None:
        print("\n✅ LISTO PARA EJECUTAR:")
        print("   Fase 3 está completa y correcta")
        print("   Inputs para Fase 5 están disponibles")
        print("   Puedes proceder a ejecutar Fase 5:")
        print("   → Abrir fase 5/main.ipynb")
        print("   → Run All Cells")
        print("   → Esperar ~30-45 minutos")
        return 0

    if not fase5_outputs_ok:
        print("\n⚠️  PROBLEMAS EN FASE 5:")
        print("   Fase 5 se ejecutó pero hay problemas en los outputs")
        print("   Revisa los detalles arriba")
        return 1

    print("\n🎉 TODO CORRECTO:")
    print("   ✅ Fase 3 completa y correcta")
    print("   ✅ Fase 5 completa y correcta")
    print("   ✅ Todas las variables guardadas correctamente")
    print("   ✅ Temperaturas son diferentes (problema resuelto)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
