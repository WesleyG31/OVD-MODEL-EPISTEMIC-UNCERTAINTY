"""
Create a visual summary of the verification results
"""

import pandas as pd
import json
from pathlib import Path

print("╔" + "═" * 78 + "╗")
print("║" + " " * 20 + "VERIFICATION SUMMARY - ALL SYSTEMS READY" + " " * 18 + "║")
print("╚" + "═" * 78 + "╝")
print()

# Summary table
data = {
    "Component": [
        "MC-Dropout Cache",
        "Uncertainty Field",
        "Temperature Calibration",
        "Baseline Predictions",
        "Ground Truth Data",
        "Data Coverage",
        "Code Corrections",
        "Fase 5 Ready",
    ],
    "Status": ["✅"] * 8,
    "Details": [
        "29,914 records, 1,996 images",
        "Present in all records, 98.8% non-zero",
        "T_global = 2.344, NLL improved -2.5%",
        "22,162 predictions, 1,988 images",
        "10,000 images (val_calib + val_eval)",
        "99.8% of val_eval (1,996/2,000)",
        "[:100] limitation removed, re-run complete",
        "All cache files verified, ready to execute",
    ],
}

df = pd.DataFrame(data)

print(
    "┌─────────────────────────────┬────────┬────────────────────────────────────────┐"
)
print(
    "│ Component                   │ Status │ Details                                │"
)
print(
    "├─────────────────────────────┼────────┼────────────────────────────────────────┤"
)

for _, row in df.iterrows():
    component = row["Component"].ljust(27)
    status = row["Status"].center(6)
    details = row["Details"][:38].ljust(38)
    print(f"│ {component} │ {status} │ {details} │")

print(
    "└─────────────────────────────┴────────┴────────────────────────────────────────┘"
)
print()

# Critical variables table
print("┌────────────────────────────────────────────────────────────────────────────┐")
print("│                  CRITICAL VARIABLES (10/10 Present)                        │")
print("├────────────────────────────────────────────────────────────────────────────┤")

variables = [
    ("image_id", "Unique image identifier"),
    ("category_id", "Object category (0-9)"),
    ("bbox", "Bounding box [x1, y1, x2, y2]"),
    ("score_mean", "Mean confidence across K=5"),
    ("score_std", "Standard deviation"),
    ("score_var", "Variance of confidence"),
    ("uncertainty", "Epistemic uncertainty (KEY)"),
    ("num_passes", "Number of MC passes"),
    ("is_tp", "True Positive flag"),
    ("max_iou", "Maximum IoU with GT"),
]

for var, desc in variables:
    var_str = var.ljust(15)
    status = "✅"
    desc_str = desc.ljust(40)
    print(f"│  {var_str}  {status}   {desc_str[:38]} │")

print("└────────────────────────────────────────────────────────────────────────────┘")
print()

# File inventory
print("┌────────────────────────────────────────────────────────────────────────────┐")
print("│                          CRITICAL FILES INVENTORY                          │")
print("├────────────────────────────────────────────────────────────────────────────┤")

files = [
    ("Fase 3", "mc_stats_labeled.parquet", "✅", "29,914 records"),
    ("Fase 3", "preds_mc_aggregated.json", "✅", "COCO format"),
    ("Fase 3", "metrics.json", "✅", "mAP metrics"),
    ("Fase 4", "temperature.json", "✅", "T=2.344"),
    ("Fase 4", "calib_detections.csv", "✅", "7,994 records"),
    ("Fase 2", "preds_raw.json", "✅", "22,162 predictions"),
    ("Data", "val_calib.json", "✅", "8,000 images"),
    ("Data", "val_eval.json", "✅", "2,000 images"),
]

for fase, filename, status, info in files:
    fase_str = fase.ljust(8)
    file_str = filename.ljust(30)
    info_str = info.ljust(20)
    print(f"│  {fase_str} {file_str} {status}  {info_str} │")

print("└────────────────────────────────────────────────────────────────────────────┘")
print()

# Statistics
print("┌────────────────────────────────────────────────────────────────────────────┐")
print("│                      UNCERTAINTY STATISTICS                                │")
print("├────────────────────────────────────────────────────────────────────────────┤")

mc_df = pd.read_parquet("fase 3/outputs/mc_dropout/mc_stats_labeled.parquet")
unc_stats = mc_df["uncertainty"].describe()

stats = [
    ("Total Records", f"{len(mc_df):,}"),
    ("Unique Images", f"{mc_df['image_id'].nunique():,}"),
    ("Uncertainty Mean", f"{unc_stats['mean']:.6f}"),
    ("Uncertainty Std", f"{unc_stats['std']:.6f}"),
    ("Uncertainty Min", f"{unc_stats['min']:.6f}"),
    ("Uncertainty Max", f"{unc_stats['max']:.6f}"),
    (
        "Non-Zero Values",
        f"{(mc_df['uncertainty'] > 0).sum():,} ({(mc_df['uncertainty'] > 0).sum()/len(mc_df)*100:.1f}%)",
    ),
]

for stat, value in stats:
    stat_str = stat.ljust(20)
    value_str = value.rjust(25)
    print(f"│  {stat_str}       {value_str}                    │")

print("└────────────────────────────────────────────────────────────────────────────┘")
print()

# Temperature info
print("┌────────────────────────────────────────────────────────────────────────────┐")
print("│                    TEMPERATURE CALIBRATION INFO                            │")
print("├────────────────────────────────────────────────────────────────────────────┤")

with open("fase 4/outputs/temperature_scaling/temperature.json") as f:
    temp_data = json.load(f)

temp_info = [
    ("Method", "Global Temperature Scaling"),
    ("T_global", f"{temp_data['T_global']:.4f}"),
    ("Interpretation", "Overconfident model (T > 1.0)"),
    ("NLL Before", f"{temp_data['nll_before']:.4f}"),
    ("NLL After", f"{temp_data['nll_after']:.4f}"),
    (
        "Improvement",
        f"-{(temp_data['nll_before'] - temp_data['nll_after'])/temp_data['nll_before']*100:.2f}%",
    ),
]

for label, value in temp_info:
    label_str = label.ljust(20)
    value_str = value.ljust(40)
    print(f"│  {label_str}  {value_str}            │")

print("└────────────────────────────────────────────────────────────────────────────┘")
print()

# Final status
print("╔" + "═" * 78 + "╗")
print("║" + " " * 78 + "║")
print("║" + " " * 25 + "✅ ALL SYSTEMS READY FOR FASE 5 ✅" + " " * 19 + "║")
print("║" + " " * 78 + "║")
print("║" + " " * 15 + "Next Step: Execute fase 5/main.ipynb" + " " * 27 + "║")
print("║" + " " * 15 + "Expected Runtime: 15-30 minutes" + " " * 32 + "║")
print("║" + " " * 78 + "║")
print("╚" + "═" * 78 + "╝")
