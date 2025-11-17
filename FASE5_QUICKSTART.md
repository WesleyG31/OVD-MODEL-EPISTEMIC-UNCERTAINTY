# 🚀 Quick Start Guide - Fase 5 Execution

## You Are Here: Ready to Run Final Analysis ✓

All preceding phases (Fase 2, 3, 4) have been verified and are complete. You're now ready to execute the comprehensive comparison in Fase 5.

---

## What Fase 5 Will Do

Fase 5 compares **6 uncertainty/calibration methods** across 3 dimensions:

### Methods Compared
1. **Baseline** - Standard GroundingDINO (no uncertainty, no calibration)
2. **Baseline + TS** - With temperature scaling
3. **MC-Dropout K=5** - With epistemic uncertainty
4. **MC-Dropout K=5 + TS** - With both uncertainty and calibration
5. **Layer Variance** - Single-pass uncertainty (alternative to MC-Dropout)
6. **Layer Variance + TS** - With calibration

### Evaluation Dimensions
1. **Detection Performance**: mAP@0.5, AP50, AP75, per-class metrics
2. **Calibration Quality**: ECE, NLL, Brier score, reliability diagrams
3. **Risk-Coverage**: Selective prediction using uncertainty

---

## How to Run Fase 5

### Option 1: Run Full Notebook (Recommended)

```bash
cd "c:\Users\SP1VEVW\Desktop\projects\OVD-MODEL-EPISTEMIC-UNCERTAINTY\fase 5"
# Open main.ipynb in Jupyter/VS Code and run all cells
```

**Estimated Time**: 15-30 minutes (using cached results)

### Option 2: Command Line Execution

```bash
cd fase 5
jupyter nbconvert --to notebook --execute main.ipynb --output main_executed.ipynb
```

---

## What to Expect

### Optimized Cache Loading

The notebook has been optimized to **reuse existing results**:

```python
✅ Loading Baseline from: ../fase 2/outputs/baseline/preds_raw.json
✅ Loading MC-Dropout from: ../fase 3/outputs/mc_dropout/mc_stats_labeled.parquet
✅ Loading Temperature from: ../fase 4/outputs/temperature_scaling/temperature.json
```

**Benefits**:
- Reduces runtime from ~2 hours to ~15 minutes
- Guarantees consistency with previous phases
- Avoids re-running expensive MC-Dropout inference

### Progress Indicators

You'll see sections like:

```
═══════════════════════════════════════════════════
   LOADING CACHED RESULTS FROM PREVIOUS PHASES
═══════════════════════════════════════════════════

✓ Fase 2 (Baseline): 22,162 predictions loaded
✓ Fase 3 (MC-Dropout): 29,914 predictions loaded (with uncertainty)
✓ Fase 4 (Temperature): T = 2.344 loaded
```

### Expected Outputs

Fase 5 will generate in `fase 5/outputs/comparison/`:

#### 1. Detection Metrics
- `detection_metrics.json`: mAP, AP50, AP75 for all methods
- `detection_comparison.png`: Side-by-side bar charts

#### 2. Calibration Analysis
- `calibration_metrics.json`: ECE, NLL, Brier scores
- `reliability_diagrams.png`: 6-panel reliability plot
- `calibration_curves.png`: Before/after TS comparison

#### 3. Risk-Coverage Analysis
- `risk_coverage_results.json`: AUC-RC for each method
- `risk_coverage_curves.png`: Selective prediction curves
- `error_vs_uncertainty.png`: Correlation analysis

#### 4. Comparative Reports
- `final_comparison_report.md`: Executive summary
- `method_ranking.csv`: Ranked by combined score
- `summary_table.png`: Publication-ready table

---

## Monitoring Progress

### Cell-by-Cell Breakdown

| Cell | Section | Time | Description |
|------|---------|------|-------------|
| 1-3 | Setup | < 1 min | Imports and configuration |
| 4-6 | Cache Loading | ~2 min | Load Fase 2/3/4 results |
| 7-10 | Detection Eval | ~5 min | mAP calculations per method |
| 11-14 | Calibration | ~5 min | ECE, NLL, reliability diagrams |
| 15-18 | Risk-Coverage | ~5 min | Selective prediction analysis |
| 19-22 | Visualization | ~2 min | Generate all plots |
| 23-25 | Reports | < 1 min | Summary tables and markdown |

**Total**: 15-25 minutes

### Key Checkpoints

Look for these success messages:

```python
✓ Baseline evaluation complete: mAP = 0.XXX
✓ MC-Dropout evaluation complete: mAP = 0.XXX
✓ Temperature scaling applied successfully
✓ Calibration metrics computed
✓ Risk-coverage curves generated
✓ All visualizations saved
```

---

## Troubleshooting

### If Cache Loading Fails

The notebook has **automatic fallback**:

```python
if cached_predictions['baseline'] is None:
    print("⚠️ Cache not found, running full inference...")
    # Falls back to full model inference
```

**Note**: Fallback will take ~2 hours. If this happens, verify:
- [ ] `fase 2/outputs/baseline/preds_raw.json` exists
- [ ] `fase 3/outputs/mc_dropout/mc_stats_labeled.parquet` exists
- [ ] `fase 4/outputs/temperature_scaling/temperature.json` exists

Run `python final_verification.py` to re-check.

### If GPU Memory Issues

If you see `CUDA out of memory`:

1. **Reduce batch size** in cache loading (if fallback triggers)
2. **Use CPU** by setting `device = 'cpu'` in config
3. **Close other GPU processes**

### If Missing Uncertainty

Check that MC-Dropout cache is being loaded from **parquet**, not JSON:

```python
# Good (has uncertainty):
cached_predictions['mc_dropout'] = pd.read_parquet(FASE3_MC_DROPOUT_PARQUET)

# Bad (no uncertainty):
cached_predictions['mc_dropout'] = json.load(FASE3_MC_DROPOUT_JSON)
```

The notebook should prioritize parquet automatically.

---

## Interpreting Results

### Detection Metrics

**Expected mAP ranges**:
- Baseline: ~0.30-0.35
- MC-Dropout: ~0.30-0.35 (similar, uncertainty doesn't change detection)
- All methods: Similar mAP (calibration preserves ranking)

**Key Insight**: Temperature scaling doesn't change mAP because it only re-scales scores, not their relative order.

### Calibration Metrics

**Expected improvements with TS**:
- ECE: -20% to -30% (lower is better)
- NLL: -2% to -5% (lower is better)
- Brier: -3% to -5% (lower is better)

**Reliability Diagrams**: After TS, bars should align closer to the diagonal (perfect calibration).

### Risk-Coverage

**AUC-RC** (Area Under Risk-Coverage Curve):
- Higher = better selective prediction
- MC-Dropout methods should show higher AUC-RC (uncertainty helps)
- Baseline should have lowest AUC-RC (no uncertainty signal)

**Expected Ranking**:
1. MC-Dropout K=5 + TS (best: uncertainty + calibration)
2. MC-Dropout K=5 (good uncertainty, but overconfident)
3. Layer Variance + TS
4. Baseline + TS (calibrated, but no uncertainty)
5. Layer Variance
6. Baseline (worst: overconfident, no uncertainty)

---

## After Execution

### Verification Checklist

Once Fase 5 completes, verify:

- [ ] All output directories created (`outputs/comparison/`)
- [ ] Detection metrics JSON exists and contains 6 methods
- [ ] Calibration metrics show improvement for +TS methods
- [ ] Risk-coverage plots show MC-Dropout > Baseline
- [ ] Final report markdown is generated
- [ ] All visualizations (PNG files) saved

### Next Steps

1. **Review Results**: 
   - Open `final_comparison_report.md`
   - Check `method_ranking.csv`
   - Inspect visualizations

2. **Identify Best Method**:
   - For **accuracy**: All methods similar
   - For **calibration**: Any +TS method
   - For **selective prediction**: MC-Dropout K=5 + TS
   - For **deployment**: Balance complexity vs. gain

3. **Document Findings**:
   - Update project README with key results
   - Prepare presentation/paper materials
   - Archive outputs for reproducibility

---

## Quick Verification Command

Before running Fase 5, one last check:

```bash
python final_verification.py
```

Expected output:
```
✓✓✓ ALL CHECKS PASSED - READY FOR FASE 5 ✓✓✓
```

If you see this, you're good to go! 🚀

---

## Support

If you encounter issues:

1. Re-run `python final_verification.py` to diagnose
2. Check `FINAL_VERIFICATION_REPORT.md` for details
3. Review Fase 3/4 outputs manually if cache loading fails
4. Consider running Fase 3/4 again if outputs are corrupted

---

**Ready?** Open `fase 5/main.ipynb` and run all cells!

Good luck! 🎉
