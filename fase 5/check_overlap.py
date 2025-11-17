import pandas as pd
import json
from pathlib import Path

# Cargar MC-Dropout Parquet
mc_file = Path("../fase 3/outputs/mc_dropout/mc_stats_labeled.parquet")
df = pd.read_parquet(mc_file)
mc_img_ids = sorted(df["image_id"].unique())

# Cargar val_calib
val_calib = Path("../data/bdd100k_coco/val_calib.json")
with open(val_calib, "r") as f:
    data = json.load(f)

calib_img_ids = [img["id"] for img in data["images"]][:500]  # Primeras 500

print("MC-Dropout:")
print(f"  Total imágenes: {len(mc_img_ids)}")
print(f"  Rango: {min(mc_img_ids)} - {max(mc_img_ids)}")
print(f"  Primeros 10: {mc_img_ids[:10]}")

print("\nval_calib (primeras 500):")
print(f"  Total imágenes: {len(calib_img_ids)}")
print(f"  Rango: {min(calib_img_ids)} - {max(calib_img_ids)}")
print(f"  Primeros 10: {calib_img_ids[:10]}")

# Verificar overlap
overlap = set(mc_img_ids) & set(calib_img_ids)
print(f"\nOverlap:")
print(f"  Imágenes en común: {len(overlap)}")
print(f"  Porcentaje: {len(overlap) / 500 * 100:.1f}%")

if len(overlap) < 500:
    print(
        f"\n❌ PROBLEMA: Solo {len(overlap)}/500 imágenes de val_calib tienen predicciones MC-Dropout"
    )
    print(
        f"   Las restantes {500 - len(overlap)} imágenes calcularán desde cero o usarán baseline"
    )
