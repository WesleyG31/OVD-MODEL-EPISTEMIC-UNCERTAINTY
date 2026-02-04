import json
from pathlib import Path

val_calib = Path("../data/bdd100k_coco/val_calib.json")
with open(val_calib, "r") as f:
    data = json.load(f)

print(f"Total imágenes en val_calib: {len(data['images'])}")
