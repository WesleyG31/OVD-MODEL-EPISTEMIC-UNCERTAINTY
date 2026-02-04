# Project Overview: Epistemic Uncertainty and Calibration in Open-Vocabulary Object Detection for ADAS

## 1. What is this project about?

This project investigates how to make artificial intelligence systems for self-driving cars **more reliable and trustworthy**. Specifically, it focuses on improving object detection systems that can recognize any object (not just pre-defined categories) while also **understanding when the system is uncertain** about its predictions.

Imagine a self-driving car that can recognize common objects like cars and pedestrians, but also unusual objects it has never seen before. However, recognizing objects is not enough—the system must also **know how confident it is** in each detection. This project develops methods to measure and improve this confidence, making the system safer for real-world deployment.

---

## 2. Why is this important?

### The Problem:
Modern AI object detectors can identify objects, but they often provide **overconfident or poorly calibrated predictions**. For example:
- The system might be 95% confident about detecting a pedestrian when it should only be 60% confident
- It doesn't know when it's making mistakes
- It can't distinguish between "I'm sure there's no object" vs "I'm uncertain if there's an object"

### The Impact on Safety:
In autonomous driving (ADAS - Advanced Driver Assistance Systems), **wrong predictions can be catastrophic**:
- **False positives**: Detecting objects that don't exist → unnecessary emergency braking
- **False negatives**: Missing real objects → potential collisions
- **Overconfidence**: Acting on uncertain detections → dangerous decisions

This project aims to **quantify and reduce these uncertainties**, making AI decisions more transparent and reliable.

---

## 3. How was the project approached?

### 3.1 The Foundation: Open-Vocabulary Object Detection (OVD)

**Traditional object detectors** can only recognize objects from predefined categories (e.g., 80 categories in COCO dataset: car, person, dog, etc.). If you show them a new object like "scooter" or "construction cone," they fail.

**Open-Vocabulary Detection (OVD)** solves this by allowing detectors to recognize **any object described in natural language**. This project uses **GroundingDINO**, a state-of-the-art OVD model that combines:
- Vision transformers (Swin Transformer) for image understanding
- Language models (BERT) for text understanding
- Cross-modal fusion to connect visual features with text descriptions

### 3.2 The Challenge: Uncertainty Quantification

While GroundingDINO can detect open-vocabulary objects, it doesn't reliably measure its **epistemic uncertainty** (uncertainty due to lack of knowledge). This project develops and evaluates three methods to estimate this uncertainty:

> **Important Note on Transformer Architecture:**
> 
> GroundingDINO is fundamentally different from traditional CNN-based detectors (like YOLO or Faster R-CNN). It's built on a **transformer architecture** with:
> - **Swin Transformer backbone**: Hierarchical vision transformer for image features
> - **Cross-modal transformer encoder/decoder**: Fuses vision and language features
> - **Text encoder (BERT)**: Processes natural language queries
> 
> This architecture impacts how uncertainty methods are implemented:
> - Dropout layers exist only in transformer middle layers (37 modules with p=0.1)
> - Detection heads (`class_embed`, `bbox_embed`) are deterministic linear layers
> - Feature representations are high-dimensional embeddings (not traditional CNN features)
> - Cross-modal attention mechanisms introduce additional complexity
> 
> Each uncertainty method must account for these architectural characteristics, which differ significantly from traditional detection pipelines.

#### **Method 1: Baseline (No Uncertainty)**
- Standard inference without uncertainty estimation
- Provides only confidence scores (not calibrated)
- **Purpose**: Establish baseline performance for comparison

#### **Method 2: MC-Dropout (Monte Carlo Dropout)**
- Runs the model multiple times (K=5 passes) with dropout enabled
- Measures **variance in predictions** across multiple runs
- Higher variance = higher uncertainty
- **Captures epistemic uncertainty** (model's knowledge gaps)
- **Trade-off**: High quality but computationally expensive

**⚠️ Implementation Challenge - GroundingDINO Architecture:**

Unlike traditional CNN-based detectors (like YOLO or Faster R-CNN), GroundingDINO is built on a **transformer architecture** (Swin Transformer backbone + transformer encoder/decoder), which presents unique challenges for MC-Dropout:

- **Traditional approach (doesn't work)**: Classic object detectors have dropout layers in their classification and bounding box prediction heads, making MC-Dropout straightforward to implement.

- **GroundingDINO reality**: After architectural analysis, we discovered that:
  - ❌ **NO dropout in detection heads** (`class_embed` and `bbox_embed` are deterministic)
  - ✅ **37 dropout modules exist** but only in the transformer encoder/decoder (middle layers)
  - ✅ Dropout rate p=0.1 (conservative: 10% neurons disabled per pass)

**What this means:**
The **final prediction layers are deterministic**, so variance comes only from the stochastic features produced by the transformer encoder/decoder. This is different from typical MC-Dropout implementations where the final layers themselves are stochastic.

**Why GroundingDINO was designed this way:**
- Transformer architectures typically use dropout for **regularization during training**
- Detection heads are kept deterministic for **stable predictions**
- This is a design choice in the original GroundingDINO architecture

**Our solution:**
Instead of trying to modify the architecture (which would require retraining), we:
1. **Identified all 37 dropout modules** in the transformer layers through systematic inspection
2. **Activated dropout during inference** by setting these modules to `.train()` mode while keeping the rest of the model in `.eval()` mode
3. **Ensured dropout is re-activated before each pass** (critical because the prediction function can reset module states)

**Code implementation:**
```python
# Diagnostic phase: Find dropout modules
dropout_modules = []
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Dropout) and module.p > 0:
        dropout_modules.append(module)
# Result: 37 modules found in transformer encoder/decoder

# Inference phase: Activate dropout for stochastic passes
model.eval()  # Keep BatchNorm and other layers in eval mode
for module in dropout_modules:
    module.train()  # Only dropout in train mode for stochasticity

# Run K forward passes with different dropout masks
for k in range(K):
    predictions_k = predict(model, image, text_prompt)
```

**Impact on uncertainty quality:**
- ✅ Successfully captures epistemic uncertainty despite architectural limitations
- ✅ Achieves useful correlation with errors (Pearson r=0.43, AUROC=0.65)
- ⚠️ Uncertainty magnitude is lower than if heads also had dropout
- ⚠️ Conservative dropout rate (p=0.1) limits variance range

**Why we didn't modify the architecture:**
- Would require **retraining** the entire model (computationally prohibitive)
- Risk of degrading detection performance (mAP)
- The existing dropout placement still provides **sufficient uncertainty signal** for discrimination

This architectural limitation is actually representative of real-world scenarios—practitioners often work with pre-trained models where modifying the architecture isn't feasible, making our approach more practically relevant.

#### **Method 3: Decoder Variance**
- Extracts uncertainty from the model's internal decoder embeddings
- Calculates variance across different decoder layers
- Only requires **one forward pass** (fast)
- **Trade-off**: Fast but initially less accurate

**🔧 How it works with Transformer Architecture:**

This method is specifically designed to leverage GroundingDINO's **transformer decoder structure**. Unlike convolutional neural networks that produce feature maps, transformers create high-dimensional **embedding representations** at multiple layers.

**Key concept:**
The transformer decoder has **6 layers**, each producing object query embeddings. If the model is uncertain about a detection, these embeddings will show high variance across layers (inconsistent representations).

**Implementation:**
```python
# Extract embeddings from all 6 decoder layers
decoder_embeddings = []  # Shape: [6 layers, num_queries, embedding_dim]

# For each detection (query), calculate variance across layers
for each_detection:
    embedding_variance = variance(decoder_embeddings[:, detection_idx, :])
    # High variance = model is uncertain about this detection
```

**Why this works for transformers:**
- **Self-attention mechanism**: Each layer refines the object representation by attending to other parts of the image and text
- **Layer disagreement**: If different decoder layers produce very different embeddings for the same object query, the model hasn't converged to a confident prediction
- **Natural uncertainty signal**: Unlike MC-Dropout (which artificially injects noise), decoder variance measures the model's **natural internal consistency**

**Advantages for transformer-based OVD:**
- ✅ **Architecture-agnostic**: Works with any transformer decoder, regardless of dropout placement
- ✅ **No architectural modification needed**: Uses existing layer outputs
- ✅ **Single forward pass**: As fast as baseline inference
- ✅ **Interpretable**: Variance directly reflects layer agreement/disagreement

**Limitations:**
- ⚠️ Initially produces **less calibrated** uncertainties than MC-Dropout (ECE=0.206)
- ⚠️ Requires understanding of transformer layer structure for implementation
- ⚠️ Uncertainty magnitude depends on embedding dimensionality and normalization

**Why this is a good match for GroundingDINO:**
Since GroundingDINO's detection heads are deterministic (no dropout), decoder variance provides an alternative way to measure uncertainty by examining the **multi-layer consensus** within the transformer architecture itself. This is more principled than trying to retrofit dropout into layers that weren't designed for it.

#### **Method 4: Temperature Scaling (Calibration)**
- Post-processing technique that **recalibrates confidence scores**
- Learns a temperature parameter to adjust predictions
- Combines with other methods (Baseline+TS, MC-Dropout+TS, Variance+TS)
- **Does not add latency** (post-processing only)

#### **Method 5: Decision Fusion (Proposed Contribution)**
- **Combines** decoder variance uncertainty + temperature scaling calibration
- Implements risk-aware selective prediction
- Filters high-risk detections based on fused uncertainty
- **Best of both worlds**: Fast (like variance) + well-calibrated (like TS)

---

### 3.3 Architectural Insights: Why Traditional Approaches Don't Apply Directly

**The Transformer Difference:**

This project revealed important insights about applying uncertainty quantification to modern transformer-based vision-language models:

1. **Dropout placement matters**: 
   - Traditional detectors: Dropout in final classification/regression layers
   - GroundingDINO: Dropout only in transformer encoder/decoder middle layers
   - **Implication**: MC-Dropout captures feature uncertainty, not prediction-head uncertainty

2. **Multi-layer representations are informative**:
   - Transformers naturally produce **layer-wise refinements** of object representations
   - Decoder variance exploits this by measuring **inter-layer consistency**
   - This is a novel uncertainty signal unavailable in single-layer CNN detectors

3. **Cross-modal fusion adds complexity**:
   - GroundingDINO fuses vision and language through attention mechanisms
   - Uncertainty can arise from **ambiguous text queries** or **ambiguous visual features**
   - Traditional uncertainty methods (designed for vision-only) must account for both modalities

4. **Pre-trained models constrain options**:
   - Modifying architecture requires expensive retraining
   - Practical uncertainty quantification must work with **frozen pre-trained weights**
   - This makes post-hoc methods (like decoder variance and temperature scaling) more valuable

**Key Takeaway:**
Uncertainty quantification for transformer-based open-vocabulary detectors requires **rethinking traditional approaches**. This project demonstrates that:
- ✅ MC-Dropout still works but requires careful identification of stochastic layers
- ✅ Decoder variance provides a transformer-native alternative
- ✅ Temperature scaling is essential regardless of uncertainty estimation method
- ✅ Combining methods (fusion) achieves better results than any single approach

These architectural considerations make the findings of this project particularly relevant for the **new generation of foundation models** (like CLIP-based detectors, SAM, etc.) that increasingly rely on transformer architectures.

---

## 4. What metrics were used and why?

The project uses multiple metrics to evaluate different aspects of system performance:

### 4.1 Detection Performance Metrics

#### **mAP (mean Average Precision)**
- **What it measures**: Overall detection quality
- **Range**: 0-100% (higher is better)
- **Why it matters**: Standard metric for object detection accuracy
- **How it works**: Combines precision (correct detections) and recall (finding all objects)

#### **AR (Average Recall)**
- **What it measures**: Ability to find all objects in images
- **Range**: 0-100% (higher is better)
- **Why it matters**: Critical for safety—missing objects is dangerous in ADAS

### 4.2 Calibration Metrics (Confidence Reliability)

These metrics measure **how well the model's confidence matches reality**:

#### **ECE (Expected Calibration Error)**
- **What it measures**: Difference between predicted confidence and actual accuracy
- **Range**: 0-1 (lower is better, 0 = perfect calibration)
- **Example**: If the model says "90% confident" 100 times, it should be correct 90 times. ECE measures the deviation from this ideal.
- **Why it matters**: Poor calibration = unreliable confidence scores = dangerous decisions
- **How it's calculated**: 
  ```
  ECE = Σ (|accuracy - confidence|) × (# predictions in bin)
  ```

#### **Brier Score**
- **What it measures**: Mean squared difference between predictions and outcomes
- **Range**: 0-1 (lower is better)
- **Why it matters**: Penalizes confident wrong predictions heavily
- **Advantage**: Sensitive to both calibration and sharpness (decisiveness)

#### **NLL (Negative Log-Likelihood)**
- **What it measures**: Probabilistic loss for predicted distributions
- **Range**: 0-∞ (lower is better)
- **Why it matters**: Standard metric in probabilistic machine learning
- **Advantage**: Captures full probability distribution quality

### 4.3 Uncertainty Quality Metrics

#### **Pearson Correlation (r)**
- **What it measures**: Linear relationship between uncertainty and error
- **Range**: -1 to +1 (higher positive correlation is better)
- **Why it matters**: High uncertainty should correlate with high error (model knows when it's wrong)
- **Interpretation**: r > 0.3 = good, r > 0.5 = excellent uncertainty indicator

#### **Spearman Correlation (ρ)**
- **What it measures**: Monotonic relationship between uncertainty and error
- **Why it matters**: Captures non-linear relationships better than Pearson
- **Advantage**: More robust to outliers

#### **AUROC (Area Under ROC Curve)**
- **What it measures**: Ability to classify errors using uncertainty thresholds
- **Range**: 0-1 (0.5 = random, 1.0 = perfect)
- **Why it matters**: Tests if uncertainty can be used to **reject incorrect detections**
- **Practical use**: In ADAS, high-uncertainty detections can trigger human alerts

### 4.4 Risk-Aware Metrics (RQ5 Specific)

#### **Risk Score**
- **What it measures**: Expected harm from incorrect detections
- **Formula**: `Risk = FP_rate × confidence`
- **Why it matters**: Confident false alarms are dangerous in ADAS (unnecessary braking)

#### **Coverage**
- **What it measures**: Proportion of predictions retained after filtering
- **Range**: 0-100%
- **Why it matters**: Higher coverage = more predictions available for decision-making

#### **Risk-Coverage AUC**
- **What it measures**: Trade-off between safety (low risk) and usability (high coverage)
- **Range**: 0-1 (higher is better)
- **Why it matters**: Optimal operating point for selective prediction systems

#### **Selective Precision**
- **What it measures**: Precision after filtering uncertain detections
- **Why it matters**: Shows improvement from uncertainty-based filtering

### 4.5 Computational Efficiency Metrics

#### **FPS (Frames Per Second)**
- **What it measures**: Speed of inference
- **Range**: 0-∞ (higher is better)
- **ADAS requirement**: ≥20 FPS for real-time operation (≤50 ms per frame)
- **Why it matters**: Slow systems are unusable in real-time driving scenarios

#### **Latency (milliseconds)**
- **What it measures**: Time to process one image
- **Range**: Lower is better
- **Why it matters**: Direct impact on reaction time in autonomous systems

#### **Reliability per Millisecond (Proposed Metric)**
- **What it measures**: Calibration quality normalized by computational cost
- **Formula**: `(1 - ECE) / latency_ms`
- **Why it matters**: Fair comparison between fast-but-inaccurate vs slow-but-accurate methods
- **Innovation**: First efficiency-normalized uncertainty metric for OVD

---

## 5. The Dataset: BDD100K

### What is BDD100K?
- Large-scale **driving video dataset** with 100,000 images
- Collected from real-world driving in diverse conditions (day/night, weather, urban/highway)
- Annotated with 10 object categories relevant to ADAS:
  - Vehicles: car, truck, bus, train
  - Vulnerable road users: person, rider, bicycle, motorcycle
  - Infrastructure: traffic light, traffic sign

### Why BDD100K?
- **Realistic**: Captures real-world driving complexity
- **Diverse**: Multiple weather conditions, lighting, scenarios
- **Large**: Sufficient data for robust evaluation
- **ADAS-relevant**: Categories match autonomous driving needs

### Dataset Split:
- **Training**: 70,000 images (not used in this project, model is pre-trained)
- **Validation**: 10,000 images (used for calibration and evaluation)

---

## 6. Project Phases (How the work was structured)

### Phase 1: Foundation Setup (Not documented in detail)
- Environment setup (Docker, GroundingDINO installation)
- Data preprocessing (COCO format conversion)
- Baseline implementation

### Phase 2: Baseline Evaluation
- **Goal**: Establish baseline performance without uncertainty
- **Methods**: Standard GroundingDINO inference
- **Metrics**: mAP, AR, ECE, Brier, NLL
- **Key Output**: Baseline metrics showing **poor calibration** (ECE=0.267)

### Phase 3: MC-Dropout Implementation
- **Goal**: Implement and evaluate epistemic uncertainty via MC-Dropout
- **Methods**: K=5 stochastic passes, variance aggregation
- **Metrics**: Detection + calibration + uncertainty correlation
- **Key Finding**: MC-Dropout **reduces ECE to 0.203** but is **slow** (0.7 FPS)

### Phase 4: Temperature Scaling
- **Goal**: Improve calibration via post-hoc recalibration
- **Methods**: Learn temperature parameter on validation set
- **Metrics**: ECE improvement, computational cost
- **Key Finding**: TS dramatically improves all methods (**Fusion achieves ECE=0.141**)

### Phase 5: Comprehensive Comparison
- **Goal**: Compare all methods on same data split
- **Methods**: Proper train/calibration/test splits to avoid data leakage
- **Metrics**: All detection, calibration, and uncertainty metrics
- **Key Finding**: **Fusion (Variance+TS)** balances speed and quality

### Phase 6: Interactive Demo
- **Goal**: Create user-facing demonstration of uncertainty visualization
- **Methods**: Gradio web interface with real-time inference
- **Features**: Upload images, select methods, visualize uncertainty heatmaps

---

## 7. Key Innovations and Contributions

### 7.1 First comprehensive uncertainty study for OVD
- Previous work focused on closed-set detection
- This project extends uncertainty quantification to **open-vocabulary** setting

### 7.2 Adapting uncertainty methods to transformer architectures
- **Challenge identified**: Traditional MC-Dropout assumes dropout in prediction heads
- **Solution developed**: Systematic identification of dropout modules in transformer layers
- **Novel approach**: Decoder variance method that exploits transformer's multi-layer structure
- **Practical guidance**: How to apply uncertainty quantification to pre-trained transformer models without architectural modification
- **Broader impact**: Findings applicable to other transformer-based vision-language models (CLIP, ALIGN, Florence, etc.)

### 7.3 Novel Decision Fusion Framework (RQ5)
- Combines multiple uncertainty sources
- Implements risk-aware selective prediction
- Achieves better safety-coverage trade-offs than individual methods

### 7.4 Efficiency-normalized metrics
- **Reliability per millisecond**: Fairness in comparing methods with different speeds
- Critical for practical deployment decisions

### 7.5 Real-world feasibility assessment
- Runtime benchmarks on actual hardware (RTX 4060 Laptop GPU)
- Realistic evaluation of ADAS deployment viability

---

## 8. What did the project achieve?

### 8.1 Scientific Findings

**1. Transformer architectures require adapted uncertainty methods**
- **Discovery**: GroundingDINO has NO dropout in detection heads (class_embed, bbox_embed)
- **Finding**: 37 dropout modules exist only in transformer encoder/decoder layers (p=0.1)
- **Implication**: MC-Dropout must target transformer layers, not final prediction layers
- **Innovation**: Decoder variance provides transformer-native uncertainty without dropout
- **Lesson**: Traditional uncertainty techniques need architecture-specific adaptation

**2. Calibration is critical and improvable**
- Baseline model has poor calibration (ECE=0.267)
- Temperature Scaling reduces ECE by **47.2%** (to 0.141 for Fusion)
- Well-calibrated predictions enable safer ADAS decisions

**3. MC-Dropout is effective but expensive**
- Captures epistemic uncertainty well (correlation r=0.43) despite architectural limitations
- Reduces false positive rate when used for filtering
- **5.18× slower** than single-pass methods (infeasible for real-time)
- Works by introducing stochasticity in transformer middle layers, not prediction heads

**4. Decoder Variance is a fast alternative**
- Single forward pass (same speed as baseline)
- Captures uncertainty from internal transformer layer representations
- Exploits multi-layer refinement inherent to transformer architecture
- When combined with TS → **Fusion method** with best trade-offs

**4. Decision Fusion achieves optimal balance**
- **Speed**: 3.7 FPS (same as Variance)
- **Calibration**: ECE=0.141 (best among all methods)
- **Efficiency**: 3.19 reliability/ms (**5.6× better** than MC-Dropout)

**5. Real-time constraints are challenging**
- None of the methods achieve strict real-time (≥20 FPS) on tested hardware
- However, **Fusion is the most deployable** with optimizations
- Projections: With better GPU, Fusion could reach 15-30 FPS

### 8.2 Practical Impact

**For ADAS Deployment:**
- Provides clear guidance on method selection based on constraints
- Fusion method recommended for production systems
- Uncertainty-based filtering reduces false positives by 38.9%

**For AI Safety:**
- Demonstrates importance of calibration for trustworthy AI
- Shows how to measure and visualize model uncertainty
- Enables selective prediction for safety-critical applications

**For Research Community:**
- First comprehensive OVD uncertainty benchmark
- New evaluation metrics (reliability per ms, risk-coverage AUC)
- Reproducible methodology with open-source code

---

## 9. Limitations and Future Work

### Current Limitations:
1. **Speed**: Methods don't meet strict real-time (20 FPS) on laptop GPU
2. **Dataset**: Only evaluated on BDD100K (one domain)
3. **Model**: Only tested with GroundingDINO (one OVD architecture)
4. **Scenarios**: Limited to standard driving conditions

### Future Directions:
1. **Optimization**: Model compression, quantization, faster backbones
2. **Multi-domain**: Test on different datasets (NuScenes, Waymo, KITTI)
3. **Other OVD models**: Evaluate on GLIP, RegionCLIP, etc.
4. **Edge deployment**: Test on actual automotive hardware (NVIDIA Drive)
5. **Online calibration**: Adapt to distribution shifts during deployment

---

## 10. Summary: The Big Picture

This project tackles a fundamental challenge in AI for autonomous driving: **How do we make object detection systems that are not only accurate but also aware of their own limitations?**

More specifically, it addresses a critical gap in applying uncertainty quantification to **modern transformer-based open-vocabulary detectors**, which have fundamentally different architectures than traditional CNN-based detectors.

The answer involves:
- **Understanding architectural constraints**: Discovering that GroundingDINO's dropout placement differs from traditional detectors
- **Adapting classic methods**: Modifying MC-Dropout to work with transformer encoder/decoder layers
- **Developing new methods**: Creating decoder variance to exploit transformer-native multi-layer representations
- **Measuring uncertainty** through multiple complementary methods
- **Calibrating predictions** to match confidence with actual accuracy
- **Fusing information** from different sources for better decisions
- **Balancing trade-offs** between computational speed and prediction quality

The result is a comprehensive framework that:
- ✅ Quantifies uncertainty in open-vocabulary object detection
- ✅ Adapts uncertainty methods to transformer architectures (37 dropout modules in encoder/decoder)
- ✅ Introduces decoder variance as a fast, transformer-native alternative to MC-Dropout
- ✅ Improves calibration by 47% through temperature scaling
- ✅ Achieves 5.6× better efficiency than Monte Carlo methods
- ✅ Enables safer ADAS through risk-aware selective prediction
- ✅ Provides practical guidance for working with pre-trained transformer models

**Most importantly**, this work demonstrates that:
1. **Uncertainty quantification is not just theoretically interesting but practically essential** for deploying AI in safety-critical real-world systems like autonomous vehicles.
2. **Transformer-based vision-language models require rethinking traditional uncertainty approaches**, as architectural differences fundamentally change how uncertainty propagates through the network.
3. **Architectural analysis is crucial**: Understanding where dropout exists (or doesn't exist) in the model is essential for correct implementation of uncertainty methods.

By making AI systems that **know what they don't know** and adapting uncertainty techniques to **modern transformer architectures**, we take a crucial step toward truly reliable and trustworthy autonomous driving.
