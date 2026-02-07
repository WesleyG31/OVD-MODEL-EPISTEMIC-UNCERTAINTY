# Details for Reproducing Methodology and Results

## Overview

This document provides comprehensive instructions for reproducing all experimental phases of the epistemic uncertainty quantification and calibration study in open-vocabulary object detection for Advanced Driver Assistance Systems (ADAS). The project evaluates uncertainty estimation methods applied to the GroundingDINO model on the BDD100K dataset, implementing baseline detection, MC-Dropout uncertainty quantification, temperature scaling calibration, and comparative analysis across multiple methods.

## System Requirements

### Hardware Requirements

- **GPU**: NVIDIA GPU with minimum 8GB VRAM (tested on RTX 3090/4090)
- **CUDA**: Version 12.1 or compatible
- **RAM**: Minimum 16GB system memory
- **Storage**: Approximately 50GB for dataset and model weights

### Software Requirements

- **Docker**: Version 20.10 or higher (required for GroundingDINO environment)
- **NVIDIA Docker Runtime**: For GPU access within containers
- **Operating System**: Linux (Ubuntu 20.04/22.04 recommended) or Windows with WSL2

## Environment Setup

### Docker Container Configuration

The entire experimental pipeline is executed within a Docker container specifically configured for GroundingDINO. This containerized approach ensures reproducibility across different systems by isolating dependencies and library versions.

#### Building the Docker Image

Navigate to the `installing_dino` directory and build the image using the provided Dockerfile:

```bash
cd installing_dino
docker build -t groundingdino .
```

The Dockerfile specifies:
- **Base Image**: pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel
- **CUDA Architecture**: Targets compute capabilities 6.0, 6.1, 7.0, 7.5, 8.0, 8.6
- **GroundingDINO Source**: Cloned from official repository (https://github.com/IDEA-Research/GroundingDINO)
- **Model Weights**: groundingdino_swint_ogc.pth downloaded automatically

#### Key Dependencies Installed

The container includes precisely versioned dependencies to ensure reproducibility:

- PyTorch: 2.3.1+cu121
- torchvision: 0.18.1+cu121
- transformers: 4.40.2
- timm: 0.9.16
- opencv-python: 4.9.0.80
- supervision: >=0.22.0
- pycocotools: 2.0.8
- pandas: 2.2.1
- numpy: 1.26.2

#### Running the Container

Launch the container with GPU access and volume mounting:

```bash
docker run --gpus all -it -p 8888:8888 \
  -v /path/to/OVD-MODEL-EPISTEMIC-UNCERTAINTY:/workspace \
  groundingdino
```

**Volume Mounting Explanation**:
- The `-v` flag mounts the project directory into `/workspace` inside the container
- All file paths in the notebooks reference `/workspace` as the base directory
- Model weights are accessed at `/opt/program/GroundingDINO/weights/groundingdino_swint_ogc.pth`
- Configuration file is at `/opt/program/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py`

### Jupyter Notebook Access

Inside the container, launch Jupyter Notebook:

```bash
cd /workspace
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Access the notebook interface at `http://localhost:8888` on your host machine.

## Dataset Preparation

### Downloading BDD100K Dataset

The Berkeley DeepDrive 100K dataset must be downloaded from Kaggle:

**Dataset Source**: https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k

Download and extract the dataset to maintain the following directory structure:

```
data/
├── bdd100k/
│   ├── images/
│   │   └── 100k/
│   │       └── val/          # 10,000 validation images
│   └── labels/
│       └── det_20/
│           └── det_val.json  # Detection annotations
```

### Format Conversion to COCO

Execute the data preprocessing notebook to convert BDD100K annotations to COCO format:

**Notebook**: `data/main.ipynb`

This notebook performs:
1. Parsing BDD100K JSON annotations
2. Converting bounding boxes from (x1, y1, x2, y2) to COCO format (x, y, width, height)
3. Mapping 10 BDD100K categories to consecutive IDs (1-10)
4. Validating bounding box geometry
5. Splitting data into calibration and evaluation sets

**Category Mapping**:
```
ID 1: person
ID 2: rider
ID 3: car
ID 4: truck
ID 5: bus
ID 6: train
ID 7: motorcycle
ID 8: bicycle
ID 9: traffic light
ID 10: traffic sign
```

### Dataset Splitting

The notebook creates two subsets from the 10,000 validation images:

- **val_calib.json**: 8,000 images (80%) for temperature scaling calibration
- **val_eval.json**: 2,000 images (20%) for final evaluation

**Splitting Strategy**:
- Random shuffling with fixed seed (seed=42) for reproducibility
- Image-level splitting (all annotations from same image stay together)
- Stratified to maintain class distribution
- Zero overlap verified between calibration and evaluation sets

**Output Files**:
```
data/bdd100k_coco/
├── val_calib.json    # Calibration split (8,000 images)
└── val_eval.json     # Evaluation split (2,000 images)
```

### Text Prompt Configuration

Create the prompt file for GroundingDINO open-vocabulary detection:

**File**: `data/prompts/bdd100k.txt`

```
person
rider
car
truck
bus
train
motorcycle
bicycle
traffic light
traffic sign
```

**Prompt Design Rationale**:
- Simple, canonical class names without articles or descriptors
- Lowercase, singular form for consistency
- Matches COCO terminology for benchmark alignment
- Concatenated with period separators during inference: "person . rider . car . ..."

## Experimental Phases

### Phase 1: Data Preparation

**Status**: Completed by executing `data/main.ipynb`

**Outputs**:
- COCO-formatted annotations
- Calibration/evaluation splits
- Data quality validation reports

### Phase 2: Baseline Object Detection

**Objective**: Establish baseline detection performance without uncertainty quantification or calibration.

**Notebook**: `fase 2/main.ipynb`

**Key Configuration** (`fase 2/configs/baseline.yaml`):
```yaml
model:
  checkpoint: /opt/program/GroundingDINO/weights/groundingdino_swint_ogc.pth
  config: /opt/program/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py
  device: cuda
  
inference:
  conf_threshold: 0.30
  nms_iou: 0.65
  batch_size: 1
  max_detections: 300
```

**Inference Pipeline**:
1. Load GroundingDINO model with Swin Transformer backbone
2. Process 2,000 evaluation images (val_eval.json)
3. Single-pass inference with dropout disabled (standard evaluation mode)
4. Apply confidence threshold and NMS post-processing
5. Match predictions to ground truth using Hungarian algorithm (IoU ≥ 0.5)

**Evaluation Metrics Computed**:
- mAP@[.50:.95]: Mean Average Precision across IoU thresholds 0.5-0.95
- AP@50: Average Precision at IoU=0.5
- AP@75: Average Precision at IoU=0.75
- Per-category AP for all 10 classes
- AP by object size (small: <32², medium: 32²-96², large: ≥96²)

**Output Files** (`fase 2/outputs/baseline/`):
- `preds_raw.json`: Raw predictions with bounding boxes and scores
- `detection_metrics.json`: COCO evaluation metrics
- `error_analysis.json`: False positive/negative analysis
- `calib_inputs.csv`: Predictions with TP/FP labels for calibration

**Expected Results**:
- mAP@[.50:.95]: ~0.17
- AP@50: ~0.28
- Inference time: ~0.27s per image

### Phase 3: MC-Dropout Uncertainty Estimation

**Objective**: Quantify epistemic uncertainty using Monte Carlo Dropout with multiple stochastic forward passes.

**Notebook**: `fase 3/main.ipynb`

**Theoretical Foundation**:
MC-Dropout approximates Bayesian inference by performing K forward passes with dropout enabled during inference. The variance in predictions across these K passes quantifies epistemic uncertainty (model uncertainty due to limited knowledge).

**Implementation Challenges**:

GroundingDINO's transformer architecture differs fundamentally from traditional CNN-based detectors:

1. **Architecture Analysis**:
   - Backbone: Swin Transformer (hierarchical vision transformer)
   - Encoder/Decoder: Cross-modal transformer layers
   - Detection Heads: Deterministic linear layers (class_embed, bbox_embed)

2. **Dropout Configuration Discovery**:
   Through systematic module inspection, 37 dropout modules were identified:
   - All located in transformer encoder/decoder layers (middle layers)
   - Dropout probability: p=0.1 (10% neurons disabled)
   - Detection heads are deterministic (no dropout by design)

3. **Selective Dropout Activation**:
   ```python
   # Identify dropout modules
   dropout_modules = []
   for name, module in model.named_modules():
       if isinstance(module, torch.nn.Dropout):
           dropout_modules.append((name, module))
   
   # Activate only transformer dropouts during inference
   model.eval()  # Set model to eval mode
   for name, module in dropout_modules:
       if 'transformer' in name:  # Only transformer layers
           module.train()  # Enable dropout
   ```

**MC-Dropout Configuration**:
- **K**: 5 stochastic forward passes
- **Dropout reactivation**: Before each pass (critical, as prediction function resets states)
- **Alignment threshold**: IoU ≥ 0.65 for matching detections across passes

**Inference Protocol**:
1. For each image, perform K=5 forward passes with dropout active
2. Collect all detections from each pass (typically 10-20 detections per pass)
3. Align corresponding detections across passes using IoU-based matching
4. For each aligned detection cluster:
   - Compute mean score: μ = (1/K) Σ scores
   - Compute uncertainty: σ² = (1/K) Σ (score - μ)²
   - Mean bounding box: average of aligned boxes

**Output Files** (`fase 3/outputs/mc_dropout/`):
- `preds_mc_aggregated.json`: Predictions with mean scores and uncertainty
- `mc_stats_labeled.parquet`: Detailed per-detection statistics with TP/FP labels
- `uncertainty_analysis.json`: Correlation between uncertainty and errors
- `auroc_metrics.json`: Uncertainty's ability to discriminate TP from FP

**Expected Results**:
- AUROC (uncertainty discriminates TP/FP): ~0.63
- FP/TP uncertainty ratio: ~2.24x (false positives have higher uncertainty)
- Computational overhead: 5x baseline time (~1.8s per image)

**Validation**:
- 98.9% of detections show non-zero variance, confirming dropout activation
- Uncertainty correlates with detection errors (lower scores → higher uncertainty)

### Phase 4: Temperature Scaling Calibration

**Objective**: Calibrate predicted probabilities to match empirical frequencies using temperature scaling.

**Notebook**: `fase 4/main.ipynb`

**Theoretical Background**:

Temperature scaling is a post-hoc calibration method that adjusts predicted probabilities by dividing logits by a scalar temperature parameter T:

```
z_calibrated = z / T
p_calibrated = sigmoid(z_calibrated) = 1 / (1 + exp(-z/T))
```

Where:
- z = logit(score) = log(score / (1 - score))
- T > 1: Softens probabilities (reduces overconfidence)
- T < 1: Sharpens probabilities (increases confidence)
- T = 1: No change

**Optimization Process**:

1. **Calibration Data Preparation**:
   - Use baseline predictions on val_calib.json (8,000 images)
   - Match predictions to ground truth (IoU ≥ 0.5)
   - Label each detection as TP (true positive) or FP (false positive)

2. **Temperature Optimization**:
   - Objective function: Negative Log-Likelihood (NLL)
   - Optimization method: L-BFGS-B (scipy.optimize)
   - Search bounds: T ∈ [0.1, 10.0]
   - Convergence criterion: Change in NLL < 1e-6

3. **Logit Conversion**:
   ```python
   # Handle edge cases for numerical stability
   epsilon = 1e-7
   score_clipped = np.clip(score, epsilon, 1 - epsilon)
   logit = np.log(score_clipped / (1 - score_clipped))
   ```

4. **NLL Loss Function**:
   ```python
   def nll_loss(T, logits, labels):
       calibrated_probs = sigmoid(logits / T)
       nll = -np.mean(labels * np.log(calibrated_probs) + 
                      (1 - labels) * np.log(1 - calibrated_probs))
       return nll
   ```

**Evaluation on Test Set** (val_eval.json):
- Apply optimized temperature to baseline predictions
- Apply optimized temperature to MC-Dropout predictions
- Compute calibration metrics

**Calibration Metrics**:

1. **Expected Calibration Error (ECE)**:
   - Bins predictions into confidence intervals
   - Measures gap between confidence and accuracy
   - Lower is better (0 = perfect calibration)

2. **Negative Log-Likelihood (NLL)**:
   - Probabilistic loss function
   - Penalizes overconfident incorrect predictions
   - Lower is better

3. **Brier Score**:
   - Mean squared error between probabilities and binary outcomes
   - Combines calibration and discrimination
   - Lower is better

**Output Files** (`fase 4/outputs/temperature_scaling/`):
- `temperature.json`: Optimized T value (~2.34, indicating overconfidence)
- `calib_detections.csv`: Calibration set predictions with logits
- `eval_calibrated.csv`: Test set predictions with calibrated scores
- `calibration_metrics.json`: ECE, NLL, Brier scores before/after
- `reliability_diagram.png`: Visual calibration assessment

**Expected Results**:
- Optimal Temperature: T ≈ 2.34 (model is overconfident)
- ECE reduction: ~21% (0.0716 → 0.0561)
- NLL reduction: ~2.5% (0.1138 → 0.1110)
- mAP preservation: ΔmAP ≈ 0% (calibration doesn't harm discrimination)

### Phase 5: Comprehensive Method Comparison

**Objective**: Compare six methods combining uncertainty estimation and calibration on a common evaluation set.

**Notebook**: `fase 5/main.ipynb`

**Methods Evaluated**:
1. Baseline (no uncertainty, no calibration)
2. Baseline + Temperature Scaling
3. MC-Dropout K=5 (uncertainty via multiple passes)
4. MC-Dropout K=5 + Temperature Scaling
5. Decoder Variance (single-pass uncertainty from 6 decoder layers)
6. Decoder Variance + Temperature Scaling

**Optimization Strategy**:

The notebook implements intelligent caching to reuse computationally expensive results:

```python
# Check for existing results from previous phases
if os.path.exists('../fase 2/outputs/baseline/preds_raw.json'):
    # Load cached baseline predictions (~15 minutes saved)
    
if os.path.exists('../fase 3/outputs/mc_dropout/mc_stats_labeled.parquet'):
    # Load cached MC-Dropout results (~90 minutes saved)
    
if os.path.exists('../fase 4/outputs/temperature_scaling/temperature.json'):
    # Load optimized temperature (~5 minutes saved)
```

**Execution Time**:
- With cached results: ~15-20 minutes
- From scratch: ~2 hours

**Decoder Variance Method** (Novel Contribution):

Instead of multiple forward passes, extract uncertainty from intermediate representations:

1. **Hook Registration**:
   ```python
   layer_outputs = {}
   
   def hook_fn(name):
       def hook(module, input, output):
           layer_outputs[name] = output
       return hook
   
   # Register hooks on all 6 decoder layers
   for i in range(6):
       model.transformer.decoder.layers[i].register_forward_hook(
           hook_fn(f'decoder_layer_{i}')
       )
   ```

2. **Single-Pass Inference**:
   - Perform one forward pass
   - Capture score predictions from all 6 decoder layers
   - Each layer produces slightly different scores for same detection

3. **Uncertainty Computation**:
   ```python
   # For each detection, collect scores from 6 layers
   layer_scores = [layer_1_score, layer_2_score, ..., layer_6_score]
   mean_score = np.mean(layer_scores)
   uncertainty = np.var(layer_scores)  # Variance across layers
   ```

**Advantages**:
- No computational overhead (single pass)
- Captures representational uncertainty
- Fast enough for real-time applications

**Disadvantages**:
- Lower uncertainty quality than MC-Dropout
- Less theoretically grounded

**Comparative Evaluation**:

For each method, compute:

1. **Detection Metrics**:
   - mAP@[.50:.95], AP@50, AP@75
   - Per-class AP
   - Computational time

2. **Calibration Metrics**:
   - ECE, NLL, Brier Score
   - Reliability diagrams (confidence vs accuracy)

3. **Risk-Coverage Curves**:
   - Sort predictions by uncertainty (high to low)
   - Progressively remove high-uncertainty predictions
   - Plot coverage (% predictions retained) vs risk (error rate)
   - Compute AUC (Area Under Risk-Coverage curve)

4. **Uncertainty AUROC**:
   - Treat TP=0, FP=1 as binary classification
   - Use uncertainty as classification score
   - Compute AUROC (ability to discriminate errors)

**Output Files** (`fase 5/outputs/comparison/`):
- `detection_metrics.json`: mAP for all 6 methods
- `calibration_metrics.json`: ECE/NLL/Brier for all methods
- `risk_coverage_auc.json`: Risk-coverage AUC scores
- `uncertainty_auroc.json`: Error discrimination performance
- `comparison_map.png`: Bar chart of mAP across methods
- `comparison_calibration.png`: ECE/NLL comparison
- `reliability_diagrams.png`: 6-panel reliability plot
- `risk_coverage_curves.png`: Risk-coverage curves overlay

**Expected Comparative Results**:
- Best detection: Baseline (mAP ~0.17, no uncertainty overhead)
- Best calibration: Any method + Temperature Scaling (ECE ~0.056)
- Best uncertainty: MC-Dropout K=5 (AUROC ~0.63, FP discrimination)
- Best efficiency: Decoder Variance (single-pass, real-time capable)

### Phase 6: Interactive Demonstration

**Objective**: Provide visual interface for exploring detection, uncertainty, and calibration on individual images.

**Application**: `fase 6/app/demo.py`

**Launch Instructions**:
```bash
cd fase\ 6
streamlit run app/demo.py
```

Access at `http://localhost:8501`

**Features**:
- Upload custom images or select from 9 curated samples
- Choose among 6 methods (baseline, MC-Dropout, decoder variance, with/without TS)
- Adjust confidence threshold slider
- Filter by uncertainty threshold
- View annotated detections with uncertainty labels
- Histogram of uncertainty distribution
- Table of detection details

**Use Cases**:
- Visualizing calibration effects (scores before/after temperature scaling)
- Understanding when model is uncertain
- Demonstrating risk-aware decision making for ADAS

## Research Questions Analysis

Beyond the main experimental phases, the `New_RQ/` directory contains focused analyses addressing specific research questions:

### RQ1: Representation-Level Uncertainty Fusion

**Notebook**: `New_RQ/new_rq1/rq1.ipynb`

**Question**: How can uncertainties from multiple decoder layers be combined for reliable risk assessment?

**Methods Compared**:
- Best single layer (Layer 6 only)
- Mean fusion (average across 6 layers)
- Variance fusion (variance of scores across 6 layers)

**Analysis**:
- Per-layer calibration performance
- Fusion strategy impact on AUROC
- Risk-coverage trade-offs

### RQ2-RQ10: Additional Research Questions

Each subdirectory (`new_rq2/` through `new_rq10/`) contains:
- Dedicated notebook (`rq*.ipynb`)
- Markdown explanation (`new_rq*.md`)
- Output artifacts (figures, tables, metrics)

These analyses explore specific aspects of uncertainty quantification, calibration trade-offs, and ADAS-relevant scenarios.

## Reproducibility Checklist

To ensure full reproducibility, verify the following before starting:

### Environment

- [ ] Docker installed and GPU accessible
- [ ] NVIDIA Docker runtime configured
- [ ] Sufficient disk space (50GB+)
- [ ] CUDA 12.1 compatible GPU

### Dataset

- [ ] BDD100K validation split downloaded (10,000 images)
- [ ] Images located at `data/bdd100k/images/100k/val/`
- [ ] Annotations at `data/bdd100k/labels/det_20/det_val.json`

### Docker Container

- [ ] GroundingDINO image built successfully
- [ ] Model weights downloaded (`groundingdino_swint_ogc.pth`)
- [ ] Container can access GPU (verify with `nvidia-smi`)
- [ ] Project directory mounted at `/workspace`

### Execution Order

1. [ ] Data preparation: `data/main.ipynb`
2. [ ] Phase 2 baseline: `fase 2/main.ipynb`
3. [ ] Phase 3 MC-Dropout: `fase 3/main.ipynb`
4. [ ] Phase 4 temperature scaling: `fase 4/main.ipynb`
5. [ ] Phase 5 comparison: `fase 5/main.ipynb`
6. [ ] Phase 6 demo: `fase 6/app/demo.py`

### Random Seeds

All experiments use fixed random seeds for reproducibility:
- PyTorch: `torch.manual_seed(42)`
- NumPy: `np.random.seed(42)`
- CUDA: `torch.cuda.manual_seed_all(42)`

### Validation

- [ ] Phase 2 mAP@[.50:.95] ≈ 0.17 (±0.01)
- [ ] Phase 3 AUROC ≈ 0.63 (±0.02)
- [ ] Phase 4 optimal T ≈ 2.34 (±0.1)
- [ ] Phase 5 outputs match expected ranges

## Troubleshooting Common Issues

### GPU Memory Errors

If encountering CUDA out-of-memory errors:
- Reduce batch size to 1 (already default)
- Clear GPU cache: `torch.cuda.empty_cache()`
- Restart Docker container to reset GPU state

### Model Loading Failures

If GroundingDINO fails to load:
- Verify model weights exist at `/opt/program/GroundingDINO/weights/groundingdino_swint_ogc.pth`
- Check config path: `/opt/program/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py`
- Rebuild Docker image if files are missing

### Dataset Path Errors

All paths are relative to `/workspace` inside the container:
- Images: `/workspace/data/bdd100k/images/100k/val/`
- Annotations: `/workspace/data/bdd100k_coco/val_eval.json`
- Verify volume mounting: `-v /host/path:/workspace`

### MC-Dropout Not Working

If MC-Dropout shows zero variance:
- Verify dropout modules are in train mode
- Check dropout reactivation before each forward pass
- Inspect module states: `module.training` should be True for transformer dropouts

### Temperature Optimization Failing

If temperature optimization does not converge:
- Verify calibration data has both TP and FP examples
- Check for NaN values in logits (edge case handling)
- Ensure sufficient calibration data (>1000 detections)

## Output Artifacts Summary

After completing all phases, the following outputs should be available:

### Phase 2 Outputs
```
fase 2/outputs/baseline/
├── config.yaml
├── preds_raw.json
├── detection_metrics.json
├── error_analysis.json
└── calib_inputs.csv
```

### Phase 3 Outputs
```
fase 3/outputs/mc_dropout/
├── config.yaml
├── preds_mc_aggregated.json
├── mc_stats_labeled.parquet
├── uncertainty_analysis.json
└── auroc_metrics.json
```

### Phase 4 Outputs
```
fase 4/outputs/temperature_scaling/
├── temperature.json
├── calib_detections.csv
├── eval_calibrated.csv
├── calibration_metrics.json
└── reliability_diagram.png
```

### Phase 5 Outputs
```
fase 5/outputs/comparison/
├── config.yaml
├── detection_metrics.json
├── calibration_metrics.json
├── risk_coverage_auc.json
├── uncertainty_auroc.json
├── comparison_map.png
├── comparison_calibration.png
├── reliability_diagrams.png
└── risk_coverage_curves.png
```

## Performance Benchmarks

Typical execution times on RTX 3090 GPU:

| Phase | Task | Time | Images |
|-------|------|------|---------|
| Data Prep | Format conversion | 5 min | 10,000 |
| Phase 2 | Baseline inference | 10 min | 2,000 |
| Phase 3 | MC-Dropout K=5 | 90 min | 2,000 |
| Phase 4 | Temperature calibration | 5 min | 8,000 |
| Phase 5 | Method comparison | 15 min* | 2,000 |
| Phase 6 | Demo (per image) | 0.3-1.5s | 1 |

*With cached results; 120 minutes from scratch

## Citation and Acknowledgments

If using this methodology in academic work, please reference:

- **GroundingDINO**: Liu et al., "Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection", ECCV 2024
- **BDD100K Dataset**: Yu et al., "BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning", CVPR 2020
- **MC-Dropout**: Gal & Ghahramani, "Dropout as a Bayesian Approximation", ICML 2016
- **Temperature Scaling**: Guo et al., "On Calibration of Modern Neural Networks", ICML 2017

## Contact and Support

For questions regarding methodology reproduction, please refer to:
- Project documentation: `README.md` in each phase directory
- Detailed explanations: `explanation.md` in project root
- Phase-specific reports: `REPORTE_FINAL_FASE*.md` in each phase

All code is provided as-is for academic reproducibility purposes.
