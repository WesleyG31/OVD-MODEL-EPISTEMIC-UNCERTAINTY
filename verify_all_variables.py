# Verificación Exhaustiva de Variables del Proyecto

"""
Script de verificación completa de todas las variables críticas
necesarias para el flujo completo del proyecto.

Verifica:
1. Fase 3: Generación y guardado de variables
2. Fase 5: Carga y uso de variables
3. Flujo completo de datos
4. Integridad de variables críticas

Uso:
    python verify_all_variables.py
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys


def print_header(title):
    """Imprime encabezado de sección"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_check(name, status, details="", warning=""):
    """Imprime resultado de verificación"""
    icon = "✅" if status else "❌"
    print(f"{icon} {name}")
    if details:
        print(f"   {details}")
    if warning:
        print(f"   ⚠️  {warning}")
    return status


# ============================================================================
# VERIFICACIÓN 1: FASE 3 - GENERACIÓN DE VARIABLES
# ============================================================================


def verify_fase3_variable_generation():
    """Verifica que Fase 3 genera todas las variables necesarias"""
    print_header("FASE 3: GENERACIÓN DE VARIABLES")

    fase3_dir = Path("fase 3/outputs/mc_dropout")
    all_ok = True

    # 1. Verificar mc_stats.parquet (sin TP/FP)
    print("\n📊 1. mc_stats.parquet (base)")
    stats_file = fase3_dir / "mc_stats.parquet"

    if not stats_file.exists():
        print_check("Archivo existe", False, str(stats_file))
        return False

    try:
        df = pd.read_parquet(stats_file)
        print_check("Archivo cargado", True, f"{len(df)} filas")

        # Verificar columnas críticas
        required_cols = {
            "image_id": "ID de imagen",
            "category_id": "Categoría de detección",
            "bbox": "Bounding box",
            "score_mean": "Score promedio (K pases)",
            "score_std": "Desviación estándar score",
            "score_var": "Varianza score",
            "uncertainty": "⭐ INCERTIDUMBRE EPISTÉMICA",
            "num_passes": "Número de pases MC-Dropout",
        }

        for col, description in required_cols.items():
            if col in df.columns:
                print_check(f"Columna '{col}'", True, description)
            else:
                print_check(f"Columna '{col}'", False, f"FALTANTE: {description}")
                all_ok = False

        # Verificar que uncertainty tiene valores
        if "uncertainty" in df.columns:
            unc_stats = {
                "mean": df["uncertainty"].mean(),
                "std": df["uncertainty"].std(),
                "min": df["uncertainty"].min(),
                "max": df["uncertainty"].max(),
                "zeros": (df["uncertainty"] == 0).sum(),
            }

            print(f"\n   📈 Estadísticas de uncertainty:")
            print(f"      Mean: {unc_stats['mean']:.6f}")
            print(f"      Std:  {unc_stats['std']:.6f}")
            print(f"      Min:  {unc_stats['min']:.6f}")
            print(f"      Max:  {unc_stats['max']:.6f}")
            print(f"      Zeros: {unc_stats['zeros']} / {len(df)}")

            if unc_stats["max"] == 0:
                print_check(
                    "Uncertainty tiene valores > 0",
                    False,
                    "Todas las detecciones tienen uncertainty = 0",
                )
                all_ok = False
            else:
                print_check(
                    "Uncertainty tiene valores > 0",
                    True,
                    f"Max: {unc_stats['max']:.6f}",
                )

        print(f"\n   🖼️  Imágenes únicas: {df['image_id'].nunique()}")

    except Exception as e:
        print_check("Error al cargar", False, str(e))
        return False

    # 2. Verificar mc_stats_labeled.parquet (con TP/FP)
    print("\n📊 2. mc_stats_labeled.parquet (con TP/FP)")
    labeled_file = fase3_dir / "mc_stats_labeled.parquet"

    if not labeled_file.exists():
        print_check("Archivo existe", False, str(labeled_file))
        return False

    try:
        df_labeled = pd.read_parquet(labeled_file)
        print_check("Archivo cargado", True, f"{len(df_labeled)} filas")

        # Verificar columnas adicionales (TP/FP)
        additional_cols = {
            "is_tp": "Etiqueta TP/FP (True/False)",
            "max_iou": "IoU máximo con GT",
        }

        for col, description in additional_cols.items():
            if col in df_labeled.columns:
                print_check(f"Columna '{col}'", True, description)
            else:
                print_check(f"Columna '{col}'", False, f"FALTANTE: {description}")
                all_ok = False

        # Verificar todas las columnas de mc_stats están presentes
        for col in required_cols.keys():
            if col not in df_labeled.columns:
                print_check(f"Columna heredada '{col}'", False, "FALTANTE")
                all_ok = False

        # Verificar que uncertainty sigue presente
        if "uncertainty" in df_labeled.columns:
            unc_mean = df_labeled["uncertainty"].mean()
            unc_max = df_labeled["uncertainty"].max()
            print_check(
                "Uncertainty preservada",
                True,
                f"Mean: {unc_mean:.6f}, Max: {unc_max:.6f}",
            )

            if unc_max == 0:
                print_check("Uncertainty > 0", False, "Todos los valores son 0")
                all_ok = False
        else:
            print_check("Uncertainty preservada", False, "COLUMNA FALTANTE")
            all_ok = False

        # Verificar distribución TP/FP
        if "is_tp" in df_labeled.columns:
            tp_count = df_labeled["is_tp"].sum()
            fp_count = (~df_labeled["is_tp"]).sum()
            total = len(df_labeled)

            print(f"\n   📊 Distribución TP/FP:")
            print(f"      TP: {tp_count} ({100*tp_count/total:.1f}%)")
            print(f"      FP: {fp_count} ({100*fp_count/total:.1f}%)")

            if tp_count == 0 or fp_count == 0:
                print_check(
                    "Balance TP/FP", False, "Solo hay TPs o solo FPs (desbalanceado)"
                )
                all_ok = False
            else:
                print_check("Balance TP/FP", True, f"TP: {tp_count}, FP: {fp_count}")

        print(f"\n   🖼️  Imágenes únicas: {df_labeled['image_id'].nunique()}")

    except Exception as e:
        print_check("Error al cargar", False, str(e))
        return False

    return all_ok


# ============================================================================
# VERIFICACIÓN 2: FASE 5 - CARGA DE VARIABLES
# ============================================================================


def verify_fase5_variable_loading():
    """Verifica que Fase 5 puede cargar todas las variables necesarias"""
    print_header("FASE 5: CARGA DE VARIABLES")

    all_ok = True

    # 1. Baseline (Fase 2)
    print("\n📁 1. Baseline (Fase 2)")
    baseline_file = Path("fase 2/outputs/baseline/preds_raw.json")

    if baseline_file.exists():
        try:
            with open(baseline_file) as f:
                baseline_data = json.load(f)

            print_check("Archivo existe", True, f"{len(baseline_data)} predicciones")

            # Verificar estructura
            if len(baseline_data) > 0:
                sample = baseline_data[0]
                required_keys = ["image_id", "category_id", "bbox", "score"]
                missing = set(required_keys) - set(sample.keys())

                if missing:
                    print_check("Estructura correcta", False, f"Faltan keys: {missing}")
                    all_ok = False
                else:
                    print_check("Estructura correcta", True, "Todas las keys presentes")

                    # Baseline NO debe tener uncertainty
                    if "uncertainty" in sample:
                        print_check(
                            "Sin uncertainty",
                            False,
                            "Baseline no debería tener uncertainty",
                        )
                    else:
                        print_check(
                            "Sin uncertainty",
                            True,
                            "Correcto (baseline no necesita uncertainty)",
                        )

        except Exception as e:
            print_check("Error al cargar", False, str(e))
            all_ok = False
    else:
        print_check("Archivo existe", False, str(baseline_file))
        all_ok = False

    # 2. MC-Dropout (Fase 3)
    print("\n📁 2. MC-Dropout (Fase 3)")
    mc_parquet = Path("fase 3/outputs/mc_dropout/mc_stats_labeled.parquet")
    mc_json = Path("fase 3/outputs/mc_dropout/preds_mc_aggregated.json")

    # Preferencia: Parquet (con uncertainty)
    if mc_parquet.exists():
        try:
            df = pd.read_parquet(mc_parquet)
            print_check(
                "Parquet existe",
                True,
                f"{len(df)} predicciones, {df['image_id'].nunique()} imágenes",
            )

            # Verificar columnas necesarias para Fase 5
            required_for_fase5 = [
                "image_id",
                "category_id",
                "bbox",
                "score_mean",
                "uncertainty",  # ⭐ CRÍTICO
            ]

            missing = set(required_for_fase5) - set(df.columns)
            if missing:
                print_check("Columnas necesarias", False, f"Faltan: {missing}")
                all_ok = False
            else:
                print_check("Columnas necesarias", True, "Todas presentes")

                # Verificar que uncertainty tiene valores
                unc_mean = df["uncertainty"].mean()
                unc_max = df["uncertainty"].max()

                print(f"   📈 Uncertainty: mean={unc_mean:.6f}, max={unc_max:.6f}")

                if unc_max == 0:
                    print_check(
                        "Uncertainty con valores",
                        False,
                        "Todas las detecciones tienen uncertainty = 0",
                    )
                    all_ok = False
                else:
                    print_check(
                        "Uncertainty con valores",
                        True,
                        f"Valores presentes (max: {unc_max:.6f})",
                    )

        except Exception as e:
            print_check("Error al cargar parquet", False, str(e))
            all_ok = False
    elif mc_json.exists():
        print_check(
            "Parquet preferido",
            False,
            "Usando JSON (SIN uncertainty) como fallback",
            "JSON no tiene uncertainty, se calculará como 0.0",
        )
        all_ok = False
    else:
        print_check("Archivos MC-Dropout", False, "No se encontró ni parquet ni JSON")
        all_ok = False

    # 3. Temperature (Fase 4)
    print("\n📁 3. Temperature Scaling (Fase 4)")
    temp_file = Path("fase 4/outputs/temperature_scaling/temperature.json")

    if temp_file.exists():
        try:
            with open(temp_file) as f:
                temp_data = json.load(f)

            print_check("Archivo existe", True, "")

            # Verificar estructura
            if "optimal_temperature" in temp_data:
                T = temp_data["optimal_temperature"]
                print_check("Temperatura disponible", True, f"T = {T:.4f}")
            else:
                print_check("Estructura correcta", False, "Falta 'optimal_temperature'")
                all_ok = False

        except Exception as e:
            print_check("Error al cargar", False, str(e))
            all_ok = False
    else:
        print_check(
            "Archivo existe",
            False,
            str(temp_file),
            "Se calculará temperatura en Fase 5",
        )
        # No es crítico, se puede calcular

    return all_ok


# ============================================================================
# VERIFICACIÓN 3: FLUJO COMPLETO DE DATOS
# ============================================================================


def verify_data_flow():
    """Verifica el flujo completo de datos entre fases"""
    print_header("FLUJO COMPLETO DE DATOS")

    all_ok = True

    print("\n🔄 Verificando flujo: Fase 3 → Fase 5")

    # 1. Cargar datos de Fase 3
    mc_parquet = Path("fase 3/outputs/mc_dropout/mc_stats_labeled.parquet")

    if not mc_parquet.exists():
        print_check("Flujo Fase 3 → Fase 5", False, "Fase 3 no ha generado outputs")
        return False

    try:
        # Simular lo que hace Fase 5
        df = pd.read_parquet(mc_parquet)

        print(f"   📥 Fase 3 genera: {len(df)} detecciones")
        print(f"   🖼️  Imágenes: {df['image_id'].nunique()}")

        # Verificar variables críticas
        critical_vars = ["image_id", "bbox", "score_mean", "uncertainty", "category_id"]

        for var in critical_vars:
            if var not in df.columns:
                print_check(f"Variable '{var}' disponible", False, "FALTA")
                all_ok = False
            else:
                # Verificar que no hay NaNs
                nan_count = df[var].isna().sum()
                if nan_count > 0:
                    print_check(
                        f"Variable '{var}' completa", False, f"{nan_count} valores NaN"
                    )
                    all_ok = False
                else:
                    print_check(
                        f"Variable '{var}' completa", True, f"{len(df)} valores"
                    )

        # Simular conversión de Fase 5
        print("\n   🔄 Simulando conversión a formato Fase 5...")

        conversion_ok = True
        for idx, row in df.head(10).iterrows():  # Verificar primeras 10
            try:
                # Simular conversión
                pred = {
                    "image_id": int(row["image_id"]),
                    "category_id": int(row["category_id"]) + 1,
                    "bbox": (
                        list(row["bbox"])
                        if isinstance(row["bbox"], np.ndarray)
                        else row["bbox"]
                    ),
                    "score": float(row["score_mean"]),
                    "uncertainty": float(row["uncertainty"]),  # ⭐ CRÍTICO
                }

                # Verificar que uncertainty se preserva
                if pred["uncertainty"] != row["uncertainty"]:
                    conversion_ok = False
                    break

            except Exception as e:
                print_check("Conversión", False, f"Error en fila {idx}: {e}")
                conversion_ok = False
                all_ok = False
                break

        if conversion_ok:
            print_check(
                "Conversión a formato Fase 5",
                True,
                "Uncertainty se preserva correctamente",
            )
        else:
            print_check(
                "Conversión a formato Fase 5", False, "Problemas en la conversión"
            )
            all_ok = False

    except Exception as e:
        print_check("Flujo de datos", False, str(e))
        return False

    return all_ok


# ============================================================================
# VERIFICACIÓN 4: INTEGRIDAD DE VARIABLES CRÍTICAS
# ============================================================================


def verify_critical_variables():
    """Verificación específica de variables más críticas"""
    print_header("VARIABLES CRÍTICAS: VERIFICACIÓN DETALLADA")

    all_ok = True

    mc_file = Path("fase 3/outputs/mc_dropout/mc_stats_labeled.parquet")

    if not mc_file.exists():
        print("⚠️  Fase 3 no ha generado outputs. Ejecuta Fase 3 primero.")
        return False

    try:
        df = pd.read_parquet(mc_file)

        # 1. UNCERTAINTY (LA MÁS CRÍTICA)
        print("\n⭐ 1. UNCERTAINTY (Variable más crítica)")

        if "uncertainty" not in df.columns:
            print_check("Columna 'uncertainty' existe", False, "CRÍTICO: FALTA")
            return False

        print_check("Columna 'uncertainty' existe", True, "")

        # Estadísticas detalladas
        unc = df["uncertainty"]
        stats = {
            "count": len(unc),
            "mean": unc.mean(),
            "std": unc.std(),
            "min": unc.min(),
            "max": unc.max(),
            "q25": unc.quantile(0.25),
            "q50": unc.quantile(0.50),
            "q75": unc.quantile(0.75),
            "zeros": (unc == 0).sum(),
            "nans": unc.isna().sum(),
        }

        print("\n   📊 Estadísticas completas:")
        print(f"      Count: {stats['count']}")
        print(f"      Mean:  {stats['mean']:.6f}")
        print(f"      Std:   {stats['std']:.6f}")
        print(f"      Min:   {stats['min']:.6f}")
        print(f"      Q25:   {stats['q25']:.6f}")
        print(f"      Q50:   {stats['q50']:.6f}")
        print(f"      Q75:   {stats['q75']:.6f}")
        print(f"      Max:   {stats['max']:.6f}")
        print(
            f"      Zeros: {stats['zeros']} ({100*stats['zeros']/stats['count']:.1f}%)"
        )
        print(f"      NaNs:  {stats['nans']}")

        # Verificaciones
        if stats["nans"] > 0:
            print_check("Sin valores NaN", False, f"{stats['nans']} valores NaN")
            all_ok = False
        else:
            print_check("Sin valores NaN", True, "")

        if stats["max"] == 0:
            print_check(
                "Tiene valores > 0",
                False,
                "Todas las detecciones tienen uncertainty = 0",
            )
            all_ok = False
        else:
            print_check("Tiene valores > 0", True, f"Max: {stats['max']:.6f}")

        if stats["std"] == 0:
            print_check(
                "Tiene varianza", False, "Todas las detecciones tienen el mismo valor"
            )
            all_ok = False
        else:
            print_check("Tiene varianza", True, f"Std: {stats['std']:.6f}")

        # Distribución por percentiles
        print("\n   📈 Distribución por percentiles:")
        percentiles = [0, 10, 25, 50, 75, 90, 95, 99, 100]
        for p in percentiles:
            val = unc.quantile(p / 100) if p < 100 else unc.max()
            print(f"      P{p:3d}: {val:.6f}")

        # 2. IMAGE_ID
        print("\n📸 2. IMAGE_ID")

        if "image_id" not in df.columns:
            print_check("Columna 'image_id' existe", False, "CRÍTICO: FALTA")
            all_ok = False
        else:
            print_check("Columna 'image_id' existe", True, "")

            unique_images = df["image_id"].nunique()
            total_detections = len(df)
            avg_det_per_img = (
                total_detections / unique_images if unique_images > 0 else 0
            )

            print(f"   🖼️  Imágenes únicas: {unique_images}")
            print(f"   📦 Detecciones totales: {total_detections}")
            print(f"   📊 Promedio det/imagen: {avg_det_per_img:.1f}")

            if unique_images < 100:
                print_check(
                    "Suficientes imágenes",
                    False,
                    f"Solo {unique_images} imágenes (esperadas: 2000)",
                    "Necesitas volver a correr Fase 3 con todas las imágenes",
                )
                all_ok = False
            elif unique_images < 2000:
                print_check(
                    "Suficientes imágenes",
                    False,
                    f"Solo {unique_images}/2000 imágenes",
                    "Cache incompleto, mejor volver a correr Fase 3",
                )
                all_ok = False
            else:
                print_check("Suficientes imágenes", True, f"{unique_images} imágenes")

        # 3. IS_TP (para calibración)
        print("\n🎯 3. IS_TP (Etiquetas TP/FP)")

        if "is_tp" not in df.columns:
            print_check("Columna 'is_tp' existe", False, "CRÍTICO: FALTA")
            all_ok = False
        else:
            print_check("Columna 'is_tp' existe", True, "")

            tp_count = df["is_tp"].sum()
            fp_count = (~df["is_tp"]).sum()
            total = len(df)

            print(f"   ✅ TP: {tp_count} ({100*tp_count/total:.1f}%)")
            print(f"   ❌ FP: {fp_count} ({100*fp_count/total:.1f}%)")

            if tp_count == 0:
                print_check("Balance TP/FP", False, "No hay TPs")
                all_ok = False
            elif fp_count == 0:
                print_check("Balance TP/FP", False, "No hay FPs")
                all_ok = False
            elif tp_count < 100 or fp_count < 100:
                print_check(
                    "Balance TP/FP",
                    False,
                    "Pocos ejemplos de alguna clase",
                    "Puede afectar calibración de temperatura",
                )
            else:
                print_check("Balance TP/FP", True, f"TP: {tp_count}, FP: {fp_count}")

    except Exception as e:
        print(f"❌ Error en verificación: {e}")
        return False

    return all_ok


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================


def main():
    """Función principal de verificación"""
    print("=" * 80)
    print("  VERIFICACIÓN EXHAUSTIVA DE VARIABLES DEL PROYECTO")
    print("=" * 80)
    print("\nVerificando todas las variables críticas para el flujo completo...")
    print("Esto incluye: generación, guardado, carga y uso de variables.")
    print()

    # Ejecutar todas las verificaciones
    results = {
        "fase3_generation": verify_fase3_variable_generation(),
        "fase5_loading": verify_fase5_variable_loading(),
        "data_flow": verify_data_flow(),
        "critical_variables": verify_critical_variables(),
    }

    # Resumen final
    print_header("RESUMEN FINAL")

    print("\n📋 Resultados de verificación:")
    print(
        f"  Fase 3 - Generación:      {'✅ OK' if results['fase3_generation'] else '❌ PROBLEMAS'}"
    )
    print(
        f"  Fase 5 - Carga:           {'✅ OK' if results['fase5_loading'] else '❌ PROBLEMAS'}"
    )
    print(
        f"  Flujo de datos:           {'✅ OK' if results['data_flow'] else '❌ PROBLEMAS'}"
    )
    print(
        f"  Variables críticas:       {'✅ OK' if results['critical_variables'] else '❌ PROBLEMAS'}"
    )

    all_ok = all(results.values())

    print("\n" + "=" * 80)

    if all_ok:
        print("✅ VERIFICACIÓN COMPLETA: TODO CORRECTO")
        print("\nTodas las variables están correctamente:")
        print("  ✅ Generadas en Fase 3")
        print("  ✅ Guardadas en archivos")
        print("  ✅ Disponibles para Fase 5")
        print("  ✅ Con valores válidos")
        print("\n🚀 El proyecto está listo para ejecutarse.")
        return 0
    else:
        print("❌ VERIFICACIÓN COMPLETA: HAY PROBLEMAS")
        print("\nProblemas detectados:")

        if not results["fase3_generation"]:
            print("  ❌ Fase 3 no genera todas las variables necesarias")
            print("     Acción: Revisar código de Fase 3")

        if not results["fase5_loading"]:
            print("  ❌ Fase 5 no puede cargar todas las variables")
            print("     Acción: Ejecutar Fase 3 para generar outputs")

        if not results["data_flow"]:
            print("  ❌ Problemas en el flujo de datos entre fases")
            print("     Acción: Revisar formato de archivos")

        if not results["critical_variables"]:
            print("  ❌ Variables críticas tienen problemas")
            print("     Acción: Revisar valores de uncertainty, image_id, is_tp")

        print("\n⚠️  Revisa los detalles arriba para más información.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
