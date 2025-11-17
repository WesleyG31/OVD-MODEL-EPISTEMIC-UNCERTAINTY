"""
Final Comprehensive Verification Script
Checks all outputs, variables, and cache files for Fase 3-5
"""

import pandas as pd
import json
from pathlib import Path
import numpy as np

print("=" * 80)
print("  FINAL COMPREHENSIVE VERIFICATION - OVD PROJECT")
print("=" * 80)

# 1. Verify Fase 3 MC-Dropout outputs
print("\n1. FASE 3 - MC-DROPOUT VERIFICATION")
print("-" * 80)

mc_parquet = Path("fase 3/outputs/mc_dropout/mc_stats_labeled.parquet")
if mc_parquet.exists():
    mc_df = pd.read_parquet(mc_parquet)
    print(f"✓ MC-Dropout cache file exists")
    print(f"  - Total records: {len(mc_df):,}")
    print(f"  - Unique images: {mc_df['image_id'].nunique():,}")
    print(f"  - Columns: {list(mc_df.columns)}")

    # Check critical variables
    critical_vars = [
        "image_id",
        "category_id",
        "bbox",
        "score_mean",
        "score_std",
        "score_var",
        "uncertainty",
        "num_passes",
        "is_tp",
        "max_iou",
    ]
    missing_vars = [v for v in critical_vars if v not in mc_df.columns]

    if missing_vars:
        print(f"  ✗ MISSING VARIABLES: {missing_vars}")
    else:
        print(f"  ✓ All {len(critical_vars)} critical variables present")

    # Check uncertainty statistics
    print(f"\n  Uncertainty Statistics:")
    print(f"    - Mean: {mc_df['uncertainty'].mean():.6f}")
    print(f"    - Std: {mc_df['uncertainty'].std():.6f}")
    print(f"    - Min: {mc_df['uncertainty'].min():.6f}")
    print(f"    - Max: {mc_df['uncertainty'].max():.6f}")
    print(
        f"    - Non-zero values: {(mc_df['uncertainty'] > 0).sum():,} ({(mc_df['uncertainty'] > 0).sum()/len(mc_df)*100:.1f}%)"
    )

    # Check bbox format
    sample_bbox = mc_df["bbox"].iloc[0]
    print(f"\n  Bbox format check:")
    print(f"    - Sample bbox: {sample_bbox}")
    print(f"    - Type: {type(sample_bbox)}")

else:
    print(f"✗ MC-Dropout cache file NOT FOUND: {mc_parquet}")

# Check other Fase 3 files
fase3_files = {
    "mc_stats.parquet": "fase 3/outputs/mc_dropout/mc_stats.parquet",
    "preds_mc_aggregated.json": "fase 3/outputs/mc_dropout/preds_mc_aggregated.json",
    "metrics.json": "fase 3/outputs/mc_dropout/metrics.json",
    "tp_fp_analysis.json": "fase 3/outputs/mc_dropout/tp_fp_analysis.json",
    "timing_data.parquet": "fase 3/outputs/mc_dropout/timing_data.parquet",
}

print(f"\n  Other Fase 3 files:")
for name, path in fase3_files.items():
    exists = Path(path).exists()
    status = "✓" if exists else "✗"
    print(f"    {status} {name}")
    if exists and path.endswith(".json"):
        with open(path) as f:
            data = json.load(f)
            if isinstance(data, list):
                print(f"       → {len(data)} records")
            elif isinstance(data, dict):
                print(f"       → Keys: {list(data.keys())}")

# 2. Verify Fase 4 Temperature Scaling outputs
print("\n\n2. FASE 4 - TEMPERATURE SCALING VERIFICATION")
print("-" * 80)

temp_file = Path("fase 4/outputs/temperature_scaling/temperature.json")
if temp_file.exists():
    with open(temp_file) as f:
        temps = json.load(f)
    print(f"✓ Temperature file exists")
    print(f"  - Optimal T: {temps.get('optimal_T', 'N/A')}")

    if "per_class_T" in temps:
        print(f"  - Per-class temperatures: {len(temps['per_class_T'])} classes")
        for cls_name, temp in temps["per_class_T"].items():
            print(f"    • {cls_name}: {temp:.4f}")

    # Check if all temperatures are identical (indicates problem)
    if "per_class_T" in temps:
        per_class_temps = list(temps["per_class_T"].values())
        if len(set(per_class_temps)) == 1:
            print(
                f"  ⚠ WARNING: All per-class temperatures are identical ({per_class_temps[0]:.4f})"
            )
            print(f"    This may indicate incomplete calibration data")
        else:
            print(
                f"  ✓ Per-class temperatures are diverse (range: {min(per_class_temps):.4f} - {max(per_class_temps):.4f})"
            )
else:
    print(f"✗ Temperature file NOT FOUND: {temp_file}")

# Check calibration data
calib_csv = Path("fase 4/outputs/temperature_scaling/calib_detections.csv")
if calib_csv.exists():
    calib_df = pd.read_csv(calib_csv)
    print(f"\n  ✓ Calibration detections: {len(calib_df):,} records")
    print(f"    - Columns: {list(calib_df.columns)}")

    # Check for image_id column (may have different name)
    img_id_col = None
    for col in calib_df.columns:
        if "image" in col.lower() and "id" in col.lower():
            img_id_col = col
            break

    if img_id_col:
        print(f"    - Unique images: {calib_df[img_id_col].nunique():,}")

    print(f"    - Has uncertainty: {'uncertainty' in calib_df.columns}")
    if "uncertainty" in calib_df.columns:
        unc_stats = calib_df["uncertainty"].describe()
        print(f"    - Uncertainty mean: {unc_stats['mean']:.6f}")
        print(f"    - Uncertainty std: {unc_stats['std']:.6f}")
        print(
            f"    - Non-zero: {(calib_df['uncertainty'] > 0).sum():,} ({(calib_df['uncertainty'] > 0).sum()/len(calib_df)*100:.1f}%)"
        )
else:
    print(f"  ✗ Calibration detections NOT FOUND")

# 3. Verify Fase 2 Baseline outputs
print("\n\n3. FASE 2 - BASELINE VERIFICATION")
print("-" * 80)

baseline_file = Path("fase 2/outputs/baseline/preds_raw.json")
if baseline_file.exists():
    with open(baseline_file) as f:
        baseline = json.load(f)
    print(f"✓ Baseline predictions exist")
    print(f"  - Total predictions: {len(baseline):,}")

    # Check image coverage
    image_ids = set(pred["image_id"] for pred in baseline)
    print(f"  - Unique images: {len(image_ids):,}")

    # Sample prediction
    sample = baseline[0]
    print(f"  - Sample prediction keys: {list(sample.keys())}")
else:
    print(f"✗ Baseline predictions NOT FOUND: {baseline_file}")

# 4. Check data availability
print("\n\n4. DATA AVAILABILITY")
print("-" * 80)

gt_val_calib = Path("data/bdd100k_coco/val_calib.json")
gt_val_eval = Path("data/bdd100k_coco/val_eval.json")

if gt_val_calib.exists():
    with open(gt_val_calib) as f:
        val_calib = json.load(f)
    print(f"✓ val_calib.json: {len(val_calib.get('images', []))} images")
else:
    print(f"✗ val_calib.json NOT FOUND")

if gt_val_eval.exists():
    with open(gt_val_eval) as f:
        val_eval = json.load(f)
    print(f"✓ val_eval.json: {len(val_eval.get('images', []))} images")
else:
    print(f"✗ val_eval.json NOT FOUND")

# 5. Coverage Analysis
print("\n\n5. COVERAGE ANALYSIS")
print("-" * 80)

if mc_parquet.exists() and gt_val_calib.exists() and gt_val_eval.exists():
    mc_df = pd.read_parquet(mc_parquet)

    with open(gt_val_calib) as f:
        val_calib = json.load(f)
    with open(gt_val_eval) as f:
        val_eval = json.load(f)

    # Get all ground truth image IDs
    calib_img_ids = set(img["id"] for img in val_calib.get("images", []))
    eval_img_ids = set(img["id"] for img in val_eval.get("images", []))
    all_gt_img_ids = calib_img_ids | eval_img_ids

    # Get MC-Dropout image IDs
    mc_img_ids = set(mc_df["image_id"].unique())

    # Calculate coverage against val_eval only (Fase 3 target)
    eval_coverage = (
        len(mc_img_ids & eval_img_ids) / len(eval_img_ids) * 100 if eval_img_ids else 0
    )

    print(f"Ground truth images:")
    print(f"  - Total (val_calib + val_eval): {len(all_gt_img_ids):,}")
    print(f"  - val_calib: {len(calib_img_ids):,}")
    print(f"  - val_eval: {len(eval_img_ids):,}")
    print(f"\nMC-Dropout cache: {len(mc_img_ids):,} images")
    print(
        f"  - Coverage of val_eval: {len(mc_img_ids & eval_img_ids):,}/{len(eval_img_ids):,} ({eval_coverage:.1f}%)"
    )

    if eval_coverage < 99:
        missing = eval_img_ids - mc_img_ids
        print(f"⚠ MISSING {len(missing)} images from val_eval in MC-Dropout cache")
        print(f"  First 10 missing IDs: {sorted(list(missing))[:10]}")
    else:
        print(f"✓ Full val_eval coverage achieved!")

    # Note about val_calib
    calib_overlap = len(mc_img_ids & calib_img_ids)
    if calib_overlap > 0:
        print(
            f"  Note: {calib_overlap} val_calib images also in cache (expected if Fase 3 ran on full dataset)"
        )

# 6. Final Summary
print("\n\n" + "=" * 80)
print("  FINAL SUMMARY")
print("=" * 80)

checks = []

# Check 1: MC-Dropout cache
if mc_parquet.exists():
    mc_df = pd.read_parquet(mc_parquet)
    has_uncertainty = "uncertainty" in mc_df.columns
    has_all_vars = all(v in mc_df.columns for v in critical_vars)
    checks.append(("MC-Dropout cache exists", True))
    checks.append(("MC-Dropout has uncertainty", has_uncertainty))
    checks.append(("MC-Dropout has all variables", has_all_vars))
    # Check coverage against val_eval (which is what Fase 3 processes)
    with open("data/bdd100k_coco/val_eval.json") as f:
        val_eval = json.load(f)
    eval_img_ids = set(img["id"] for img in val_eval["images"])
    mc_img_ids = set(mc_df["image_id"].unique())
    eval_coverage = len(mc_img_ids & eval_img_ids) / len(eval_img_ids) * 100
    checks.append(("MC-Dropout coverage > 99% of val_eval", eval_coverage > 99))
else:
    checks.append(("MC-Dropout cache exists", False))

# Check 2: Temperature file
if temp_file.exists():
    with open(temp_file) as f:
        temps = json.load(f)
    has_t_global = "T_global" in temps
    has_valid_t = has_t_global and temps["T_global"] > 0
    checks.append(("Temperature file exists", True))
    checks.append(("Has T_global", has_t_global))
    checks.append(("T_global is valid", has_valid_t))
else:
    checks.append(("Temperature file exists", False))

# Check 3: Baseline
checks.append(("Baseline predictions exist", baseline_file.exists()))

# Check 4: Ground truth
checks.append(("val_calib.json exists", gt_val_calib.exists()))
checks.append(("val_eval.json exists", gt_val_eval.exists()))

# Print summary
print()
for check_name, passed in checks:
    status = "✓" if passed else "✗"
    print(f"{status} {check_name}")

all_passed = all(passed for _, passed in checks)
print(f"\n{'='*80}")
if all_passed:
    print("✓✓✓ ALL CHECKS PASSED - READY FOR FASE 5 ✓✓✓")
else:
    failed = [name for name, passed in checks if not passed]
    print(f"✗✗✗ {len(failed)} CHECKS FAILED ✗✗✗")
    print(f"Failed checks: {', '.join(failed)}")
print(f"{'='*80}\n")
