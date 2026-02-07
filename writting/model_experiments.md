# Model Training, Environment, and Evaluation Process

## Overview

This research employs a pre-trained open-vocabulary object detection model in an evaluation-only paradigm. No model training or fine-tuning was conducted as part of this work. Instead, the study focuses on evaluating and enhancing the uncertainty quantification and calibration capabilities of an existing pre-trained model through post-hoc methods. This section details the computational environment, model configuration, and evaluation process employed throughout the experimental phases.

## Computational Environment

### Docker-Based Setup

The entire experimental pipeline was executed within a containerized Docker environment specifically designed for GroundingDINO deployment. This approach ensures reproducibility, dependency isolation, and consistent execution across different hardware configurations.

The Docker environment was configured with the following specifications:

**Base Image**: The container was built on top of the official PyTorch Docker image `pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel`, which provides a pre-configured deep learning environment with CUDA 12.1 support and cuDNN 8 for optimized GPU acceleration.

**CUDA Configuration**: The environment variables were set to enable proper GPU utilization:
- `CUDA_HOME=/usr/local/cuda`: Points to the CUDA installation directory
- `TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"`: Specifies the compute capability architectures to support, ensuring compatibility across multiple GPU generations from Pascal (6.0) through Ampere (8.6)
- `SETUPTOOLS_USE_DISTUTILS=stdlib`: Ensures proper Python package installation

**System Dependencies**: The container includes essential build tools and libraries:
- `build-essential`: Compilation tools for building Python extensions
- `git`: Version control for cloning repositories
- `python3-opencv`: OpenCV library for image processing operations
- `ca-certificates`: SSL certificates for secure downloads
- `wget`: Utility for downloading model weights

**Working Directory**: All operations are conducted within `/opt/program`, which serves as the base directory for the GroundingDINO installation and project execution. This path is consistently referenced throughout the configuration files, as evidenced by the model checkpoint and configuration paths pointing to `/opt/program/GroundingDINO/`.

### GroundingDINO Installation

The GroundingDINO framework was installed directly from the official GitHub repository within the Docker container:

**Repository Clone**: The source code was obtained via:
```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
```

**Model Weights**: Pre-trained weights for the Swin Transformer-based GroundingDINO model (SwinT-OGC variant) were downloaded from the official release:
- **Weight File**: `groundingdino_swint_ogc.pth`
- **Source**: GitHub releases (version v0.1.0-alpha)
- **Location**: `/opt/program/GroundingDINO/weights/groundingdino_swint_ogc.pth`

These weights represent a model pre-trained on large-scale object detection datasets with open-vocabulary capabilities, enabling zero-shot detection of arbitrary object categories without category-specific training.

### Python Environment and Dependencies

The Python environment within the Docker container was configured with specific package versions to ensure compatibility and reproducibility:

**Core Deep Learning Framework**:
- PyTorch 2.3.1 with CUDA 12.1 support
- TorchVision 0.18.1 for computer vision utilities
- NumPy 1.26.2 for numerical computations

**Model-Specific Requirements**:
- Transformers 4.40.2: Hugging Face library for BERT text encoder
- TIMM 0.9.16: PyTorch Image Models library for vision transformers
- Addict 2.4.0: Dictionary manipulation utilities used by GroundingDINO

**Computer Vision and Evaluation**:
- OpenCV-Python 4.9.0.80: Image processing and manipulation
- Supervision 0.22.0+: Detection utilities and visualization
- PycocoTools 2.0.8: COCO dataset format handling and evaluation metrics

**Data Processing and Analysis**:
- Pandas 2.2.1: Data manipulation and CSV handling
- YAPF: Code formatting utilities

This dependency configuration ensures all components of the detection pipeline, from image preprocessing through model inference to metric calculation, operate with tested and compatible versions.

### Hardware Configuration

The experiments were executed on GPU-accelerated hardware to enable efficient inference:

**GPU Requirements**: NVIDIA GPUs with CUDA Compute Capability 6.0 or higher (Pascal architecture or newer) were supported by the Docker configuration. The experimental results were obtained on hardware with the following characteristics:
- **GPU Memory**: Minimum 8GB VRAM recommended for processing BDD100K images at native resolution
- **Observed Memory Usage**: Approximately 1190 MB GPU memory per image during baseline inference
- **CUDA Version**: 12.1 as specified in the Docker base image

**CPU and System Memory**: While CPU specifications were not constrained, adequate system RAM (minimum 16GB recommended) was necessary for loading the dataset annotations and storing intermediate prediction results.

## Model Configuration

### Pre-trained Model Specification

This research employs the GroundingDINO model with the Swin Transformer-OGC (Object-Grounded Context) architecture variant. The model was used exclusively in its pre-trained state without any fine-tuning or weight updates.

**Architecture Details**:
- **Model Name**: Grounding-DINO
- **Variant**: SwinT-OGC (Swin Transformer backbone with Object-Grounded Context)
- **Configuration File**: `/opt/program/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py`
- **Checkpoint**: `/opt/program/GroundingDINO/weights/groundingdino_swint_ogc.pth`

**Input Specifications**:
- **Adaptive Resolution**: Images are resized to maintain aspect ratio while ensuring the shorter dimension reaches 800 pixels and the longer dimension does not exceed 1333 pixels. This adaptive resizing preserves spatial relationships while maintaining computational efficiency.
- **Normalization**: RGB channels are normalized using ImageNet statistics:
  - Means: [0.485, 0.456, 0.406]
  - Standard Deviations: [0.229, 0.224, 0.225]

**Inference Configuration**:
- **Device**: CUDA (GPU acceleration enabled)
- **Batch Size**: 1 (sequential image processing)
- **Maximum Detections**: 300 per image (configurable limit to prevent memory overflow)
- **Confidence Threshold**: 0.30 (baseline), 0.25 (MC-Dropout and comparative methods)
- **NMS IoU Threshold**: 0.65 (non-maximum suppression to remove duplicate detections)

### Text Prompts for Open-Vocabulary Detection

GroundingDINO's open-vocabulary capability requires natural language text prompts specifying the object categories of interest. For the BDD100K dataset, a comma-separated prompt string was constructed containing all ten target categories:

**Prompt Format**:
```
"person. rider. car. truck. bus. train. motorcycle. bicycle. traffic light. traffic sign."
```

This prompt format follows GroundingDINO's expected input convention, where categories are separated by periods and spaces. The model processes this text through its BERT-based text encoder to generate language embeddings that guide the visual detection process through cross-modal attention mechanisms.

**Category Mapping**: To handle potential vocabulary variations in the model's predictions, a synonym normalization dictionary was implemented:
- `bike` → `bicycle`
- `pedestrian` → `person`
- `motor` → `motorcycle`
- `signal` → `traffic sign`

This mapping ensures consistent category assignment when the model generates semantically equivalent but lexically different class labels.

## Evaluation Process

### No Training Paradigm

It is crucial to emphasize that **no model training, fine-tuning, or weight optimization** was performed in this research. The GroundingDINO model was used exclusively in its pre-trained state, obtained from the official release. The research focus was on:

1. **Evaluation**: Assessing the pre-trained model's detection performance on the BDD100K autonomous driving dataset
2. **Uncertainty Quantification**: Implementing methods to estimate epistemic uncertainty using the existing model architecture
3. **Post-Hoc Calibration**: Applying temperature scaling to improve probability calibration without modifying model weights

This evaluation-only approach is appropriate because:
- **Research Objective**: The goal was to investigate uncertainty quantification and calibration methods applicable to existing deployed models, not to develop new detection architectures or improve detection accuracy through training.
- **Zero-Shot Transfer**: Open-vocabulary detectors like GroundingDINO are designed for zero-shot transfer to new domains without fine-tuning, making them suitable for direct evaluation on BDD100K.
- **Computational Efficiency**: Avoiding training significantly reduces computational costs while still enabling meaningful investigation of uncertainty estimation methods.
- **Practical Relevance**: Many real-world deployment scenarios involve using pre-trained models without domain-specific training due to data scarcity or computational constraints.

### Inference Pipeline

The evaluation process consisted of systematic inference across multiple experimental phases, each building upon the baseline configuration:

**Phase 1: Baseline Evaluation**
- **Objective**: Establish reference performance metrics
- **Method**: Standard forward pass with dropout layers in evaluation mode
- **Output**: Bounding boxes, confidence scores, and class predictions
- **Dataset Split**: val_eval subset (1,988 images from BDD100K validation set)

**Phase 2: Monte Carlo Dropout Evaluation**
- **Objective**: Quantify epistemic uncertainty through stochastic inference
- **Method**: Multiple forward passes (K=5) with dropout layers activated during inference
- **Dropout Configuration**: 37 dropout modules in transformer encoder-decoder layers set to training mode (p=0.1)
- **Aggregation**: Predictions across multiple passes are aligned using IoU-based matching, and statistical measures (mean, variance) are computed for confidence scores and bounding boxes
- **Uncertainty Metric**: Score variance across stochastic passes

**Phase 3: Temperature Scaling Calibration**
- **Objective**: Improve probability calibration of confidence scores
- **Method**: Post-hoc recalibration using a learned temperature parameter
- **Optimization Set**: val_calib subset (1,000 images) used to optimize temperature via negative log-likelihood minimization
- **Application**: Optimized temperature applied to transform confidence scores on val_eval subset
- **Temperature Parameter**: Single scalar T optimized via L-BFGS-B algorithm

**Phase 4: Comparative Evaluation**
- **Objective**: Compare multiple uncertainty and calibration methods
- **Methods Evaluated**:
  1. Baseline (no uncertainty, no calibration)
  2. Baseline + Temperature Scaling
  3. MC-Dropout (K=5)
  4. MC-Dropout + Temperature Scaling
  5. Decoder Layer Variance (single-pass uncertainty)
  6. Decoder Layer Variance + Temperature Scaling
- **Analysis**: Detection metrics (mAP), calibration metrics (ECE, NLL, Brier), and uncertainty quality (AUROC for error detection)

### Reproducibility Measures

To ensure reproducibility of the evaluation results, several measures were implemented:

**Random Seed Control**:
```python
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

These settings ensure deterministic behavior in:
- Random weight initialization (not applicable here as weights are pre-loaded)
- Dropout mask generation during MC-Dropout inference
- Data shuffling and sampling operations

**Containerization**: The Docker environment specification ensures dependency versions and system configurations remain consistent across different execution environments.

**Configuration Logging**: All hyperparameters and settings are logged in YAML configuration files for each experimental phase, providing complete documentation of the evaluation setup.

## Evaluation Metrics

### Detection Performance

**COCO-Style Metrics**: The primary detection performance metrics follow the COCO evaluation protocol:
- **mAP@[0.5:0.95]**: Mean Average Precision averaged over IoU thresholds from 0.5 to 0.95 with 0.05 increments
- **AP@50**: Average Precision at IoU threshold of 0.5
- **AP@75**: Average Precision at IoU threshold of 0.75
- **Per-Category AP**: Individual Average Precision for each of the 10 BDD100K categories

**Size-Stratified Metrics**:
- **AP_small**: Performance on objects smaller than 32² pixels
- **AP_medium**: Performance on objects between 32² and 96² pixels
- **AP_large**: Performance on objects larger than 96² pixels

### Calibration Metrics

**Expected Calibration Error (ECE)**: Measures the difference between predicted confidence and actual accuracy, computed over confidence bins.

**Negative Log-Likelihood (NLL)**: Probabilistic loss function quantifying the quality of predicted probability distributions.

**Brier Score**: Mean squared error between predicted probabilities and binary outcome indicators.

### Uncertainty Quality Metrics

**AUROC (Area Under ROC Curve)**: Evaluates the ability of uncertainty estimates to distinguish between correct detections (true positives) and incorrect detections (false positives). Higher AUROC indicates better discrimination capability.

**Risk-Coverage Curves**: Visualize the trade-off between error rate (risk) and dataset coverage when rejecting predictions based on uncertainty thresholds.

## Computational Performance

### Inference Speed

**Baseline Performance**:
- **Time per Image**: 0.275 seconds (average)
- **Frames per Second (FPS)**: 3.64
- **Total Inference Time**: ~9 minutes for 1,988 images (val_eval set)

**MC-Dropout Overhead** (K=5 passes):
- **Time per Image**: 1.84 seconds (average)
- **Speedup Factor**: 0.15× (6.7× slower than baseline)
- **Total Inference Time**: ~52 minutes for 1,988 images

**Temperature Scaling Overhead**:
- **Optimization Time**: < 1 minute on val_calib subset (1,000 images)
- **Inference Overhead**: Negligible (simple mathematical transformation of scores)

These timing measurements reflect the computational trade-offs between uncertainty estimation quality and inference efficiency, with MC-Dropout providing high-quality epistemic uncertainty estimates at the cost of multiple forward passes.

## Summary

This research employs a comprehensive evaluation framework for assessing uncertainty quantification and calibration methods on a pre-trained open-vocabulary object detector. The Docker-based computational environment ensures reproducibility, while the evaluation-only paradigm (no training) focuses the investigation on post-hoc methods applicable to existing deployed models. The systematic inference pipeline across multiple experimental phases, combined with rigorous metric evaluation, provides robust empirical evidence for the effectiveness of MC-Dropout and temperature scaling in autonomous driving perception systems.
