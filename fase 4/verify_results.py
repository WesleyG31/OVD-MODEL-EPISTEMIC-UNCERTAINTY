import pandas as pd
import json
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("./outputs/temperature_scaling")

print("=" * 80)
print("VERIFICACIÓN COMPLETA DE RESULTADOS - FASE 4: TEMPERATURE SCALING")
print("=" * 80)

# 1. Cargar métricas principales
with open(OUTPUT_DIR / "temperature.json", "r") as f:
    temp_data = json.load(f)

with open(OUTPUT_DIR / "calibration_metrics.json", "r") as f:
    calib_metrics = json.load(f)

T_optimal = temp_data["T_global"]
nll_calib_before = temp_data["nll_before"]
nll_calib_after = temp_data["nll_after"]

nll_eval_before = calib_metrics["val_eval"]["before"]["nll"]
nll_eval_after = calib_metrics["val_eval"]["after"]["nll"]
ece_eval_before = calib_metrics["val_eval"]["before"]["ece"]
ece_eval_after = calib_metrics["val_eval"]["after"]["ece"]
brier_eval_before = calib_metrics["val_eval"]["before"]["brier"]
brier_eval_after = calib_metrics["val_eval"]["after"]["brier"]

print("\n1. TEMPERATURA OPTIMIZADA")
print("-" * 80)
print(f"   T_optimal = {T_optimal:.4f}")
if T_optimal > 1.1:
    print(f"   ✓ T > 1.0 → Modelo era SOBRECONFIDENTE")
    print(f"   → Temperature Scaling REDUCE las probabilidades")
elif T_optimal < 0.9:
    print(f"   ✓ T < 1.0 → Modelo era SUBCONFIDENTE")
    print(f"   → Temperature Scaling AUMENTA las probabilidades")
else:
    print(f"   ⚠️  T ≈ 1.0 → Modelo YA ESTABA BIEN CALIBRADO")

# 2. Verificar datos de calibración
calib_df = pd.read_csv(OUTPUT_DIR / "calib_detections.csv")
print("\n2. DATOS DE CALIBRACIÓN (val_calib)")
print("-" * 80)
print(f"   Total detecciones: {len(calib_df)}")
print(f"   TP: {calib_df['is_tp'].sum()} ({calib_df['is_tp'].mean()*100:.2f}%)")
print(
    f"   FP: {(~calib_df['is_tp'].astype(bool)).sum()} ({(1-calib_df['is_tp'].mean())*100:.2f}%)"
)
print(f"   Score promedio: {calib_df['score'].mean():.4f}")
print(f"   Score range: [{calib_df['score'].min():.4f}, {calib_df['score'].max():.4f}]")
print(f"\n   NLL en val_calib:")
print(f"     Antes (T=1.0):  {nll_calib_before:.4f}")
print(f"     Después (T={T_optimal:.2f}): {nll_calib_after:.4f}")
print(
    f"     Mejora: {nll_calib_before - nll_calib_after:.4f} ({((nll_calib_before - nll_calib_after)/nll_calib_before)*100:.2f}%)"
)

# 3. Verificar datos de evaluación
eval_df = pd.read_csv(OUTPUT_DIR / "eval_detections.csv")
print("\n3. DATOS DE EVALUACIÓN (val_eval)")
print("-" * 80)
print(f"   Total detecciones: {len(eval_df)}")
print(f"   TP: {eval_df['is_tp'].sum()} ({eval_df['is_tp'].mean()*100:.2f}%)")
print(
    f"   FP: {(~eval_df['is_tp'].astype(bool)).sum()} ({(1-eval_df['is_tp'].mean())*100:.2f}%)"
)

# 4. Métricas de calibración
print("\n4. MÉTRICAS DE CALIBRACIÓN EN VAL_EVAL")
print("-" * 80)
print(f"   {'Métrica':<20} {'Antes':<15} {'Después':<15} {'Mejora':<15} {'% Mejora'}")
print(f"   {'-'*20} {'-'*15} {'-'*15} {'-'*15} {'-'*10}")

nll_mejora = nll_eval_before - nll_eval_after
nll_pct = (nll_mejora / nll_eval_before) * 100
print(
    f"   {'NLL':<20} {nll_eval_before:<15.4f} {nll_eval_after:<15.4f} {nll_mejora:<15.4f} {nll_pct:>9.2f}%"
)

ece_mejora = ece_eval_before - ece_eval_after
ece_pct = (ece_mejora / ece_eval_before) * 100
print(
    f"   {'ECE':<20} {ece_eval_before:<15.4f} {ece_eval_after:<15.4f} {ece_mejora:<15.4f} {ece_pct:>9.2f}%"
)

brier_mejora = brier_eval_before - brier_eval_after
brier_pct = (brier_mejora / brier_eval_before) * 100
print(
    f"   {'Brier Score':<20} {brier_eval_before:<15.4f} {brier_eval_after:<15.4f} {brier_mejora:<15.4f} {brier_pct:>9.2f}%"
)

# 5. Verificar calibración por bins
print("\n5. ANÁLISIS DE CALIBRACIÓN POR BINS (val_eval)")
print("-" * 80)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


logits_eval = eval_df["logit"].values
labels_eval = eval_df["is_tp"].values

# Antes
probs_before = sigmoid(logits_eval)
bins = np.linspace(0, 1, 11)
digitized = np.digitize(probs_before, bins) - 1

print("\n   ANTES de calibrar (T=1.0):")
print(f"   {'Bin':<15} {'Conf':<10} {'Acc':<10} {'Gap':<10} {'Count'}")
for i in range(10):
    mask = digitized == i
    if mask.sum() > 0:
        conf = probs_before[mask].mean()
        acc = labels_eval[mask].mean()
        gap = abs(conf - acc)
        print(
            f"   [{bins[i]:.1f}-{bins[i+1]:.1f}]     {conf:.4f}    {acc:.4f}    {gap:.4f}    {mask.sum():>6}"
        )

# Después
probs_after = sigmoid(logits_eval / T_optimal)
digitized = np.digitize(probs_after, bins) - 1

print("\n   DESPUÉS de calibrar (T={:.2f}):".format(T_optimal))
print(f"   {'Bin':<15} {'Conf':<10} {'Acc':<10} {'Gap':<10} {'Count'}")
for i in range(10):
    mask = digitized == i
    if mask.sum() > 0:
        conf = probs_after[mask].mean()
        acc = labels_eval[mask].mean()
        gap = abs(conf - acc)
        print(
            f"   [{bins[i]:.1f}-{bins[i+1]:.1f}]     {conf:.4f}    {acc:.4f}    {gap:.4f}    {mask.sum():>6}"
        )

# 6. Verificar artefactos
print("\n6. ARTEFACTOS GENERADOS")
print("-" * 80)
artefactos = [
    "temperature.json",
    "calib_detections.csv",
    "eval_detections.csv",
    "calibration_metrics.json",
    "reliability_diagram.png",
    "confidence_distribution.png",
    "risk_coverage.png",
    "temperature_per_class.json",
    "calibration_per_class.csv",
    "final_report.txt",
]

for artefacto in artefactos:
    path = OUTPUT_DIR / artefacto
    if path.exists():
        size = path.stat().st_size
        print(f"   ✓ {artefacto:<35} ({size:>10,} bytes)")
    else:
        print(f"   ✗ {artefacto:<35} (NO EXISTE)")

# 7. Diagnóstico final
print("\n" + "=" * 80)
print("7. DIAGNÓSTICO FINAL")
print("=" * 80)

checks_passed = 0
total_checks = 4

if abs(T_optimal - 1.0) > 0.1:
    print("✓ T es significativamente diferente de 1.0 → Había problema de calibración")
    checks_passed += 1
else:
    print("⚠️  T muy cerca de 1.0 → Modelo ya estaba calibrado")

if nll_mejora > 0:
    print(f"✓ NLL mejoró en {nll_pct:.2f}%")
    checks_passed += 1
else:
    print(f"✗ NLL NO mejoró (empeoró {abs(nll_pct):.2f}%)")

if ece_mejora > 0:
    print(f"✓ ECE mejoró en {ece_pct:.2f}%")
    checks_passed += 1
else:
    print(f"✗ ECE NO mejoró (empeoró {abs(ece_pct):.2f}%)")

if brier_mejora > 0:
    print(f"✓ Brier Score mejoró en {brier_pct:.2f}%")
    checks_passed += 1
else:
    print(f"✗ Brier Score NO mejoró (empeoró {abs(brier_pct):.2f}%)")

print("\n" + "=" * 80)
if checks_passed >= 3:
    print(
        "✅ CALIBRACIÓN EXITOSA ({}/{} checks pasados)".format(
            checks_passed, total_checks
        )
    )
    print("\nLa temperatura ajustó correctamente las probabilidades.")
    print("Las métricas de calibración mejoraron significativamente.")
elif checks_passed >= 2:
    print(
        "⚠️  CALIBRACIÓN PARCIALMENTE EXITOSA ({}/{} checks pasados)".format(
            checks_passed, total_checks
        )
    )
    print("\nAlgunas métricas mejoraron, otras no.")
elif T_optimal > 0.95 and T_optimal < 1.05:
    print(
        "ℹ️  MODELO YA ESTABA CALIBRADO ({}/{} checks pasados)".format(
            checks_passed, total_checks
        )
    )
    print("\nT ≈ 1.0 indica que el modelo baseline ya tenía buena calibración.")
    print("Esto es POSITIVO: no necesitas aplicar temperatura en producción.")
else:
    print(
        "❌ PROBLEMA DETECTADO ({}/{} checks pasados)".format(
            checks_passed, total_checks
        )
    )
    print("\nLa calibración no funcionó como se esperaba.")
    print("Revisa los datos y el proceso de optimización.")

print("=" * 80)
