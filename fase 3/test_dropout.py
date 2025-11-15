"""
Script de diagnóstico para verificar si el Dropout está activo en GroundingDINO
"""

import sys

sys.path.append("/opt/program/GroundingDINO")

import torch
from groundingdino.util.inference import load_model

print("=" * 70)
print("DIAGNÓSTICO: DROPOUT EN GROUNDINGDINO")
print("=" * 70)

# Cargar modelo
model_config = (
    "/opt/program/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
)
model_weights = "/opt/program/GroundingDINO/weights/groundingdino_swint_ogc.pth"

print("\n1. Cargando modelo...")
model = load_model(model_config, model_weights)
model = model.cuda()
print("   ✓ Modelo cargado")

print("\n2. Inspeccionando módulos Dropout en el modelo...")
dropout_modules = []
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Dropout):
        dropout_modules.append(
            {
                "name": name,
                "training": module.training,
                "p": module.p,
                "in_head": ("class_embed" in name or "bbox_embed" in name),
            }
        )

print(f"\n   Total Dropout modules encontrados: {len(dropout_modules)}")

if len(dropout_modules) == 0:
    print("   ❌ ¡NO HAY MÓDULOS DROPOUT EN EL MODELO!")
    print("   Este es el problema: GroundingDINO no tiene Dropout activo.")
else:
    print("\n   Detalle de módulos Dropout:")
    for dm in dropout_modules:
        status = "✓" if dm["in_head"] else " "
        print(f"   [{status}] {dm['name']}")
        print(f"       - training: {dm['training']}")
        print(f"       - p: {dm['p']}")
        print(f"       - in_head: {dm['in_head']}")

print("\n3. Activando Dropout en la cabeza...")
model.eval()
for name, module in model.named_modules():
    if "class_embed" in name or "bbox_embed" in name:
        if isinstance(module, torch.nn.Dropout):
            module.train()
            print(f"   ✓ Activado: {name}")

print("\n4. Verificando estado después de activar...")
active_dropouts = []
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Dropout) and module.training:
        active_dropouts.append(name)

print(f"   Dropout modules activos (training=True): {len(active_dropouts)}")
for name in active_dropouts:
    print(f"   - {name}")

print("\n" + "=" * 70)
print("CONCLUSIÓN:")
if len(dropout_modules) == 0:
    print("❌ El modelo NO tiene Dropout. MC-Dropout NO FUNCIONARÁ.")
    print("   Solución: Usar otro método de incertidumbre (ensembles, temperatura)")
elif len(active_dropouts) == 0:
    print("⚠️  El modelo tiene Dropout pero NO se pudo activar.")
    print("   Verificar que p > 0 en los módulos Dropout")
else:
    print("✓ Dropout está activo y debería funcionar.")
    print(f"  {len(active_dropouts)} módulos Dropout activos con p > 0")
print("=" * 70)
