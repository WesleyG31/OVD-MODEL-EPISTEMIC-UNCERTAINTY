import json
import pandas as pd
from pathlib import Path

print("=" * 70)
print("ANÁLISIS DE SPLITS POR FASE")
print("=" * 70)

# Fase 2 - Baseline
fase2_file = Path("../fase 2/outputs/baseline/preds_raw.json")
if fase2_file.exists():
    with open(fase2_file, "r") as f:
        fase2_data = json.load(f)
    fase2_imgs = set([p["image_id"] for p in fase2_data])
    print(f"\nFase 2 (Baseline):")
    print(f"  Predicciones: {len(fase2_data):,}")
    print(f"  Imágenes: {len(fase2_imgs):,}")
    print(f"  Archivo: preds_raw.json")

# Fase 3 - MC-Dropout
fase3_file = Path("../fase 3/outputs/mc_dropout/mc_stats_labeled.parquet")
if fase3_file.exists():
    df3 = pd.read_parquet(fase3_file)
    print(f"\nFase 3 (MC-Dropout):")
    print(f"  Predicciones: {len(df3):,}")
    print(f"  Imágenes: {df3['image_id'].nunique():,}")
    print(f"  Archivo: mc_stats_labeled.parquet")

# Fase 4 - Temperature Scaling
fase4_file = Path("../fase 4/outputs/temperature_scaling/calib_detections.csv")
if fase4_file.exists():
    df4 = pd.read_csv(fase4_file)
    print(f"\nFase 4 (Temperature Scaling):")
    print(f"  Predicciones: {len(df4):,}")
    print(f"  Imágenes: N/A (no guardó image_id)")
    print(f"  Archivo: calib_detections.csv")
    print(f"  Split: val_calib (8000 imágenes completo)")

# Verificar splits
print(f"\n{'='*70}")
print("VERIFICACIÓN DE SPLITS")
print("=" * 70)

val_calib = Path("../data/bdd100k_coco/val_calib.json")
val_eval = Path("../data/bdd100k_coco/val_eval.json")

if val_calib.exists():
    with open(val_calib, "r") as f:
        calib_data = json.load(f)
    calib_imgs = set([img["id"] for img in calib_data["images"]])
    print(f"\nval_calib.json: {len(calib_imgs):,} imágenes")

if val_eval.exists():
    with open(val_eval, "r") as f:
        eval_data = json.load(f)
    eval_imgs = set([img["id"] for img in eval_data["images"]])
    print(f"val_eval.json: {len(eval_imgs):,} imágenes")

# Análisis de overlap
print(f"\n{'='*70}")
print("OVERLAP ENTRE FASES Y SPLITS")
print("=" * 70)

if fase2_file.exists() and val_eval.exists():
    overlap_2_eval = fase2_imgs & eval_imgs
    overlap_2_calib = fase2_imgs & calib_imgs
    print(f"\nFase 2 (Baseline) - {len(fase2_imgs):,} imágenes:")
    print(
        f"  ├─ Overlap con val_eval: {len(overlap_2_eval):,} ({len(overlap_2_eval)/len(fase2_imgs)*100:.1f}%)"
    )
    print(
        f"  └─ Overlap con val_calib: {len(overlap_2_calib):,} ({len(overlap_2_calib)/len(fase2_imgs)*100:.1f}%)"
    )

if fase3_file.exists() and val_eval.exists():
    fase3_imgs = set(df3["image_id"].unique())
    overlap_3_eval = fase3_imgs & eval_imgs
    overlap_3_calib = fase3_imgs & calib_imgs
    print(f"\nFase 3 (MC-Dropout) - {len(fase3_imgs):,} imágenes:")
    print(
        f"  ├─ Overlap con val_eval: {len(overlap_3_eval):,} ({len(overlap_3_eval)/len(fase3_imgs)*100:.1f}%)"
    )
    print(
        f"  └─ Overlap con val_calib: {len(overlap_3_calib):,} ({len(overlap_3_calib)/len(fase3_imgs)*100:.1f}%)"
    )

print(f"\nFase 4 (Temperature) - 8000 imágenes (val_calib completo):")
print(f"  ├─ Overlap con val_eval: N/A")
print(f"  └─ Overlap con val_calib: 100% (procesó todo val_calib)")

print(f"\n{'='*70}")
print("CONCLUSIÓN")
print("=" * 70)
print("\n¿Qué debería tener Fase 5?")
print(
    "  - Para calibración (val_calib): Necesita predicciones de baseline y MC-Dropout"
)
print("  - Para evaluación (val_eval): Necesita predicciones de baseline y MC-Dropout")
print("\n¿Qué tiene actualmente?")
print(f"  - Fase 2: {len(fase2_imgs)} imágenes en val_eval (debería ser 2000)")
print(f"  - Fase 3: {len(fase3_imgs)} imágenes en val_eval (debería ser 2000)")
print(f"  - Fase 4: 8000 imágenes en val_calib (correcto)")
print("\n❌ PROBLEMA 1: Fase 2 solo tiene 1988 imágenes, no las 2000 de val_eval")
print("❌ PROBLEMA 2: Fase 3 solo tiene 100 imágenes, no las 2000 de val_eval")
print("❌ PROBLEMA 3: No hay predicciones MC-Dropout para val_calib (0 overlap)")
print("\n💡 SOLUCIÓN:")
print("  Fase 5 debe CALCULAR desde cero las predicciones faltantes:")
print("  - MC-Dropout para 500 imágenes de val_calib (para calibración)")
print("  - MC-Dropout para 2000 imágenes de val_eval (para evaluación)")
print("  O bien, aceptar calcular desde cero y no usar cache.")
