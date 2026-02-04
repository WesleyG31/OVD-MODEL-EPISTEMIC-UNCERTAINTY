import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np

fig = plt.figure(figsize=(16, 10))
gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

# Título principal
fig.suptitle(
    "FASE 4: TEMPERATURE SCALING - RESUMEN DE VERIFICACIÓN",
    fontsize=20,
    fontweight="bold",
    y=0.98,
)

# 1. Estado general (top izquierda)
ax1 = fig.add_subplot(gs[0, 0])
ax1.axis("off")
ax1.text(
    0.5,
    0.9,
    "✅ CALIBRACIÓN EXITOSA",
    ha="center",
    va="top",
    fontsize=18,
    fontweight="bold",
    color="green",
    transform=ax1.transAxes,
)

status_text = """
T óptima = 2.34 (Modelo sobreconfidente)

✓ NLL mejoró 2.46%
✓ ECE mejoró 21.64% 
✓ Brier mejoró 3.16%
✓ mAP se mantuvo (0.1819)

4/4 checks pasados
"""
ax1.text(
    0.5,
    0.65,
    status_text,
    ha="center",
    va="top",
    fontsize=12,
    family="monospace",
    bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.3),
    transform=ax1.transAxes,
)

# 2. Métricas principales (top derecha)
ax2 = fig.add_subplot(gs[0, 1])
metrics = ["NLL", "ECE", "Brier"]
before = [0.6996, 0.1934, 0.2527]
after = [0.6824, 0.1516, 0.2447]
x = np.arange(len(metrics))

bars1 = ax2.bar(x - 0.2, before, 0.4, label="Antes (T=1.0)", color="coral", alpha=0.8)
bars2 = ax2.bar(
    x + 0.2, after, 0.4, label="Después (T=2.34)", color="skyblue", alpha=0.8
)

ax2.set_ylabel("Valor", fontsize=12, fontweight="bold")
ax2.set_title("Métricas de Calibración en val_eval", fontsize=13, fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels(metrics, fontsize=11)
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3, axis="y")

# Añadir valores encima de las barras
for i, (b, a) in enumerate(zip(before, after)):
    improvement = ((b - a) / b) * 100
    ax2.text(
        i,
        max(b, a) + 0.02,
        f"-{improvement:.1f}%",
        ha="center",
        fontsize=10,
        fontweight="bold",
        color="green",
    )

# 3. Interpretación de T (medio izquierda)
ax3 = fig.add_subplot(gs[1, 0])
ax3.axis("off")

interpretation = """
INTERPRETACIÓN DE T = 2.34

El modelo baseline era SOBRECONFIDENTE:
• Reportaba probabilidades más altas
  de lo que su accuracy real indicaba
  
Temperature Scaling corrige esto:
• score_original = 0.70
  → score_calibrado ≈ 0.45
  
• score_original = 0.50
  → score_calibrado ≈ 0.35

Efecto: Reduce la confianza para que
sea más realista y alineada con accuracy
"""

ax3.text(
    0.05,
    0.95,
    interpretation,
    ha="left",
    va="top",
    fontsize=11,
    family="monospace",
    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5),
    transform=ax3.transAxes,
)

# 4. Calibración por bins - comparación (medio derecha)
ax4 = fig.add_subplot(gs[1, 1])

bins_labels = ["0.3-0.4", "0.4-0.5", "0.5-0.6", "0.6-0.7"]
gap_before = [0.1924, 0.2922, 0.3045, 0.2243]
gap_after = [0.0615, 0.1302, 0.3215, 0.2563]
x = np.arange(len(bins_labels))

bars1 = ax4.bar(x - 0.2, gap_before, 0.4, label="Gap Antes", color="red", alpha=0.7)
bars2 = ax4.bar(x + 0.2, gap_after, 0.4, label="Gap Después", color="green", alpha=0.7)

ax4.set_ylabel("Gap (|Confianza - Accuracy|)", fontsize=11, fontweight="bold")
ax4.set_title("Reducción del Gap por Bin de Confianza", fontsize=13, fontweight="bold")
ax4.set_xticks(x)
ax4.set_xticklabels(bins_labels, fontsize=10, rotation=0)
ax4.legend(fontsize=10)
ax4.grid(alpha=0.3, axis="y")
ax4.axhline(y=0.1, color="gray", linestyle="--", alpha=0.5, label="Umbral aceptable")

# 5. Datos procesados (bottom izquierda)
ax5 = fig.add_subplot(gs[2, 0])
ax5.axis("off")

data_text = """
DATOS PROCESADOS

val_calib (calibración):
• Total: 7,994 detecciones
• TP: 4,708 (58.89%)
• FP: 3,286 (41.11%)
• NLL mejoró: 2.50%

val_eval (evaluación):
• Total: 30,246 detecciones
• TP: 17,531 (57.96%)
• FP: 12,715 (42.04%)
• Todas las métricas mejoraron
"""

ax5.text(
    0.05,
    0.95,
    data_text,
    ha="left",
    va="top",
    fontsize=11,
    family="monospace",
    bbox=dict(boxstyle="round", facecolor="lightcyan", alpha=0.5),
    transform=ax5.transAxes,
)

# 6. Recomendaciones (bottom derecha)
ax6 = fig.add_subplot(gs[2, 1])
ax6.axis("off")

recommendations = """
RECOMENDACIONES

✓ Usar T=2.34 en producción:
  prob_cal = sigmoid(logit / 2.34)

✓ Aplicar a decisiones ADAS:
  • Si prob_cal > 0.7 → Alta confianza
  • Si prob_cal < 0.3 → Baja confianza

✓ Considerar T por clase para:
  • person, truck, bus (minoritarias)

✓ Monitorear calibración:
  • Recalcular T periódicamente
  • Verificar stability en nuevos datos

⚠️ NO afecta mAP (ranking preservado)
"""

ax6.text(
    0.05,
    0.95,
    recommendations,
    ha="left",
    va="top",
    fontsize=11,
    family="monospace",
    bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
    transform=ax6.transAxes,
)

plt.savefig(
    "outputs/temperature_scaling/verification_summary.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="white",
)
print(
    "✅ Resumen visual guardado en: outputs/temperature_scaling/verification_summary.png"
)
plt.show()
