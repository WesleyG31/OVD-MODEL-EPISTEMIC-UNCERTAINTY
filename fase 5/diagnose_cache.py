"""
Script de diagnóstico para verificar que los datos cacheados son diferentes
entre baseline y MC-Dropout.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path("..")
FASE2_BASELINE = BASE_DIR / "fase 2" / "outputs" / "baseline" / "preds_raw.json"
FASE3_MC_DROPOUT_PARQUET = (
    BASE_DIR / "fase 3" / "outputs" / "mc_dropout" / "mc_stats_labeled.parquet"
)

print("=" * 70)
print("DIAGNÓSTICO DE DATOS CACHEADOS")
print("=" * 70)

# 1. Cargar Baseline
print("\n1. BASELINE (Fase 2)")
print("-" * 70)
if FASE2_BASELINE.exists():
    with open(FASE2_BASELINE, "r") as f:
        baseline_data = json.load(f)
    print(f"✅ Archivo encontrado: {FASE2_BASELINE}")
    print(f"   Total predicciones: {len(baseline_data)}")

    # Agrupar por imagen
    baseline_by_img = {}
    for pred in baseline_data:
        img_id = pred.get("image_id")
        if img_id not in baseline_by_img:
            baseline_by_img[img_id] = []
        baseline_by_img[img_id].append(pred)

    print(f"   Total imágenes: {len(baseline_by_img)}")

    # Muestra de datos
    sample_img = list(baseline_by_img.keys())[0]
    sample_preds = baseline_by_img[sample_img][:3]
    print(f"\n   Muestra (imagen {sample_img}, primeras 3 predicciones):")
    for i, pred in enumerate(sample_preds):
        print(
            f"      {i+1}. score={pred.get('score', 'N/A'):.4f}, "
            f"category_id={pred.get('category_id', 'N/A')}, "
            f"bbox={pred.get('bbox', [])[:2]}, "
            f"uncertainty={pred.get('uncertainty', 'N/A')}"
        )
else:
    print(f"❌ No encontrado: {FASE2_BASELINE}")
    baseline_data = []
    baseline_by_img = {}

# 2. Cargar MC-Dropout
print("\n2. MC-DROPOUT (Fase 3)")
print("-" * 70)
if FASE3_MC_DROPOUT_PARQUET.exists():
    mc_df = pd.read_parquet(FASE3_MC_DROPOUT_PARQUET)
    print(f"✅ Archivo encontrado: {FASE3_MC_DROPOUT_PARQUET}")
    print(f"   Total predicciones: {len(mc_df)}")
    print(f"   Columnas: {mc_df.columns.tolist()}")

    # Estadísticas de incertidumbre
    print(f"\n   Estadísticas de incertidumbre:")
    print(f"      Min: {mc_df['uncertainty'].min():.6f}")
    print(f"      Max: {mc_df['uncertainty'].max():.6f}")
    print(f"      Media: {mc_df['uncertainty'].mean():.6f}")
    print(f"      Std: {mc_df['uncertainty'].std():.6f}")

    # Agrupar por imagen
    mc_by_img = {}
    for _, row in mc_df.iterrows():
        img_id = int(row["image_id"])
        if img_id not in mc_by_img:
            mc_by_img[img_id] = []

        # Convertir bbox si es necesario
        bbox = row["bbox"]
        if isinstance(bbox, (list, np.ndarray)) and len(bbox) == 4:
            if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                bbox_xywh = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
            else:
                bbox_xywh = bbox
        else:
            bbox_xywh = bbox

        mc_by_img[img_id].append(
            {
                "image_id": img_id,
                "category_id": int(row["category_id"]) + 1,  # 0-indexed a 1-indexed
                "bbox": bbox_xywh,
                "score": float(row["score_mean"]),
                "uncertainty": float(row["uncertainty"]),
            }
        )

    print(f"   Total imágenes: {len(mc_by_img)}")

    # Muestra de datos
    sample_img = list(mc_by_img.keys())[0]
    sample_preds = mc_by_img[sample_img][:3]
    print(f"\n   Muestra (imagen {sample_img}, primeras 3 predicciones):")
    for i, pred in enumerate(sample_preds):
        print(
            f"      {i+1}. score={pred.get('score', 'N/A'):.4f}, "
            f"category_id={pred.get('category_id', 'N/A')}, "
            f"bbox={pred.get('bbox', [])[:2]}, "
            f"uncertainty={pred.get('uncertainty', 'N/A'):.6f}"
        )
else:
    print(f"❌ No encontrado: {FASE3_MC_DROPOUT_PARQUET}")
    mc_by_img = {}

# 3. Comparación
print("\n3. COMPARACIÓN")
print("-" * 70)
if baseline_by_img and mc_by_img:
    common_imgs = set(baseline_by_img.keys()) & set(mc_by_img.keys())
    print(f"Imágenes en común: {len(common_imgs)}")

    if common_imgs:
        test_img = list(common_imgs)[0]
        baseline_preds = baseline_by_img[test_img]
        mc_preds = mc_by_img[test_img]

        print(f"\nComparando imagen {test_img}:")
        print(f"   Baseline: {len(baseline_preds)} predicciones")
        print(f"   MC-Dropout: {len(mc_preds)} predicciones")

        if baseline_preds and mc_preds:
            print(f"\n   Primera predicción de cada método:")
            print(f"      Baseline:")
            print(f"         score: {baseline_preds[0].get('score', 'N/A')}")
            print(
                f"         category_id: {baseline_preds[0].get('category_id', 'N/A')}"
            )
            print(
                f"         uncertainty: {baseline_preds[0].get('uncertainty', 'N/A')}"
            )

            print(f"      MC-Dropout:")
            print(f"         score: {mc_preds[0].get('score', 'N/A')}")
            print(f"         category_id: {mc_preds[0].get('category_id', 'N/A')}")
            print(f"         uncertainty: {mc_preds[0].get('uncertainty', 'N/A')}")

            same_score = baseline_preds[0].get("score") == mc_preds[0].get("score")
            same_cat = baseline_preds[0].get("category_id") == mc_preds[0].get(
                "category_id"
            )
            same_unc = baseline_preds[0].get("uncertainty", 0) == mc_preds[0].get(
                "uncertainty", 0
            )

            print(f"\n   ¿Son iguales?")
            print(
                f"      Scores: {'❌ SÍ (PROBLEMA)' if same_score else '✅ NO (correcto)'}"
            )
            print(
                f"      Categories: {'✅ Puede ser normal' if same_cat else '⚠️  Diferentes'}"
            )
            print(
                f"      Uncertainties: {'❌ SÍ (PROBLEMA)' if same_unc else '✅ NO (correcto)'}"
            )

            if same_score and same_unc:
                print(f"\n   ❌ ERROR CRÍTICO: Los datos son idénticos!")
                print(f"      Esto causará que las temperaturas sean iguales.")
            else:
                print(f"\n   ✅ Los datos son diferentes (correcto)")

# 4. Resumen
print("\n4. RESUMEN")
print("=" * 70)
if baseline_by_img and mc_by_img:
    print("✅ Ambos archivos cargados correctamente")
    if common_imgs:
        # Calcular estadísticas
        baseline_uncertainties = []
        mc_uncertainties = []

        for img_id in list(common_imgs)[:100]:  # Primeras 100 imágenes
            for pred in baseline_by_img[img_id]:
                baseline_uncertainties.append(pred.get("uncertainty", 0.0))
            for pred in mc_by_img[img_id]:
                mc_uncertainties.append(pred.get("uncertainty", 0.0))

        print(f"\nEstadísticas de incertidumbre (primeras 100 imágenes comunes):")
        print(
            f"   Baseline:   media={np.mean(baseline_uncertainties):.6f}, "
            f"std={np.std(baseline_uncertainties):.6f}"
        )
        print(
            f"   MC-Dropout: media={np.mean(mc_uncertainties):.6f}, "
            f"std={np.std(mc_uncertainties):.6f}"
        )

        if np.mean(baseline_uncertainties) == np.mean(mc_uncertainties) == 0.0:
            print(f"\n   ❌ PROBLEMA: Ambos métodos tienen incertidumbre 0.0")
            print(f"      Las temperaturas calculadas serán idénticas")
        else:
            print(f"\n   ✅ Las incertidumbres son diferentes (correcto)")
else:
    print("❌ Falta al menos uno de los archivos")

print("\n" + "=" * 70)
