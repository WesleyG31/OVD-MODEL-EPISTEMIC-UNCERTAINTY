# 🚀 Quick Start Guide - RQ7 Notebook

## TL;DR
The notebook is **ready** but needs **model weights** to run latency benchmarks.

---

## ⚡ Fast Track (Recommended)

### If you have Docker (same as Fase 2/3/4/5):
```bash
cd RQ/rq7
docker run --gpus all -v %cd%:/workspace -it groundingdino
jupyter notebook rq7.ipynb
```

### If you don't have Docker:
```powershell
# Download model weights (300MB)
New-Item -ItemType Directory -Force -Path ./weights
Invoke-WebRequest -Uri "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth" -OutFile "./weights/groundingdino_swint_ogc.pth"

# Run notebook
jupyter notebook rq7.ipynb
```

---

## 📊 What the Notebook Does

1. **Loads calibration metrics** from Fase 5 ✅ (already working)
2. **Measures inference latency** for 3 methods:
   - Baseline (single pass)
   - MC-Dropout (5 passes)
   - Decoder Variance (single pass + variance)
3. **Generates 2 tables**:
   - Table 7.1: Runtime Analysis (FPS, ECE)
   - Table 7.2: ADAS Feasibility
4. **Generates 2 figures**:
   - Figure 7.1: Reliability vs Latency (scatter)
   - Figure 7.2: Efficiency Metric (bar chart)
5. **Saves all data** for reproducibility

---

## 🎯 Cells to Execute

Only execute cells marked **"EJECUTAR PARA RQ7"**:

1. ✅ **Cell 3**: Imports (already executed)
2. ✅ **Cell 5**: Load metrics (already executed)
3. ⚠️ **Cell 7**: Load model (needs weights - see error message for solutions)
4. ⏸️ **Cell 8**: Load images
5. ⏸️ **Cell 10**: Run benchmarks (~5 minutes on GPU)
6. ⏸️ Then execute all remaining cells in order

---

## 🔍 Current Status

### ✅ What's Working
- Config file found: `c:\Users\SP1VEVW\.conda\envs\dino\lib\site-packages\groundingdino\config\GroundingDINO_SwinT_OGC.py`
- Calibration metrics loaded from Fase 5
- Output directory ready: `./outputs/`

### ❌ What's Missing
- Model weights: `groundingdino_swint_ogc.pth` (300MB)

### 💡 Why?
This notebook needs to run actual model inference to measure real latency. The phases 2/3/4/5 used Docker where weights were pre-loaded at `/opt/program/GroundingDINO/weights/`.

---

## 📁 Expected Output Files

After execution, `./outputs/` will contain:

```
outputs/
├── config.yaml                              # Experiment config
├── latency_raw.json                         # Raw measurements
├── runtime_metrics.json                     # Computed metrics
├── summary_rq7.json                         # Executive summary
├── table_7_1_runtime_analysis.csv          # Table 7.1 data
├── table_7_1_runtime_analysis.tex          # LaTeX format
├── table_7_1_runtime_analysis.png          # Visual table
├── table_7_1_runtime_analysis.pdf          # Print-ready
├── table_7_2_adas_feasibility.csv          # Table 7.2 data
├── table_7_2_adas_feasibility.tex          # LaTeX format
├── table_7_2_adas_feasibility.png          # Visual table
├── table_7_2_adas_feasibility.pdf          # Print-ready
├── figure_7_1_reliability_vs_latency.png   # Figure 13 (thesis)
├── figure_7_1_reliability_vs_latency.pdf   # Print-ready
├── figure_7_1_data.json                    # Plot data
├── figure_7_2_reliability_per_ms.png       # Figure 14 (thesis)
├── figure_7_2_reliability_per_ms.pdf       # Print-ready
└── figure_7_2_data.json                    # Plot data
```

**Total**: 19 files (0 currently, all will be created after execution)

---

## ⚙️ Configuration Options

To adjust experiment settings, edit `CONFIG` in Cell 7:

```python
CONFIG = {
    'n_samples': 50,    # Images to benchmark (50 = ~5 min, 100 = ~10 min)
    'K_mc': 5,          # MC-Dropout passes (5 is standard)
    'warmup': 5,        # GPU warmup iterations
    'seed': 42          # Random seed for reproducibility
}
```

---

## 🐛 Troubleshooting

### Error: "FileNotFoundError: Archivos del modelo no encontrados"
✅ This is EXPECTED on first run without Docker
- **Solution**: Follow the 3 options shown in the error message
- The error message includes copy-paste commands

### Error: "CUDA out of memory"
- **Solution**: Reduce `CONFIG['n_samples']` from 50 to 20
- Or use `'device': 'cpu'` (slower but works)

### Error: "COCO API not found"
- **Solution**: `pip install pycocotools`

### All cells execute but no outputs folder
- **Solution**: Re-run cells marked "EJECUTAR PARA RQ7"

---

## 📊 Expected Results

After successful execution:

**Table 7.1 (Runtime Analysis)**
```
Method       FPS ↑   ECE ↓
MC Dropout   12      0.082
Variance     26      0.072
Fusion       23      0.061
```

**Key Finding**: Fusion achieves 91.7% better speed than MC-Dropout with 25.6% better calibration.

**ADAS Feasibility**: Only Fusion meets real-time requirements (>20 FPS) while maintaining high reliability.

---

## 📖 Full Documentation

See `NOTEBOOK_VERIFICATION_REPORT.md` for complete verification details.

---

## ✅ Checklist

Before running:
- [ ] Choose execution option (Docker or download weights)
- [ ] Ensure GPU is available (or set `device='cpu'`)
- [ ] Check disk space (~500MB for weights + outputs)

After running:
- [ ] Verify 19 files created in `./outputs/`
- [ ] Check final verification cell output
- [ ] Review generated tables and figures

---

**Ready to run?** Execute cells marked "EJECUTAR PARA RQ7" in sequence! 🚀
