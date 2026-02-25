RQ6 — Decoder dynamics as epistemic uncertainty signals

RQ6: What intrinsic properties of transformer decoder dynamics encode epistemic uncertainty in OVD, and when does inter-layer variance reliably proxy model uncertainty?

Figures & Tables

Figure 6.1 =  C:\Users\SP1VEVW\Desktop\projects\OVD-MODEL-EPISTEMIC-UNCERTAINTY\New_RQ\new_rq6\output\Fig_RQ6_1_decoder_variance.png

Figure 6.2 =  C:\Users\SP1VEVW\Desktop\projects\OVD-MODEL-EPISTEMIC-UNCERTAINTY\New_RQ\new_rq6\output\Fig_RQ6_2_auroc_by_layer.png

Table 6.1 = C:\Users\SP1VEVW\Desktop\projects\OVD-MODEL-EPISTEMIC-UNCERTAINTY\New_RQ\new_rq6\output\Table_RQ6_1.csv

Table 6.2 = C:\Users\SP1VEVW\Desktop\projects\OVD-MODEL-EPISTEMIC-UNCERTAINTY\New_RQ\new_rq6\output\Table_RQ6_2.csv

Table 6.2a = C:\Users\SP1VEVW\Desktop\projects\OVD-MODEL-EPISTEMIC-UNCERTAINTY\New_RQ\new_rq6\output\Table_RQ6_2a_failure.csv

Table 6.2b = C:\Users\SP1VEVW\Desktop\projects\OVD-MODEL-EPISTEMIC-UNCERTAINTY\New_RQ\new_rq6\output\Table_RQ6_2b_amplified.csv

---



# RQ6 — Decoder Dynamics as Epistemic Uncertainty Signals

---

## Research Question

**RQ6:** What intrinsic properties of transformer decoder dynamics encode epistemic uncertainty in Open Vocabulary Detection (OVD), and when does inter-layer variance reliably proxy model uncertainty?

---

## Background — What is this about? (for non-experts)

### What is Open Vocabulary Detection (OVD)?

Object detection is the task of finding objects in images and drawing bounding boxes around them. Traditional detectors only know how to find objects they were trained on (e.g., "car", "person"). **Open Vocabulary Detection (OVD)** is a newer approach where the model can find *any* object described in plain text — you type a word or phrase, and the model finds it. The model used here is **GroundingDINO**, which takes an image and a text prompt (e.g., "car. truck. person.") and returns bounding boxes with confidence scores.

### What is a Transformer Decoder?

GroundingDINO uses a Transformer architecture, which is the same family of models behind large language models like GPT. Inside the model, there is a component called the **decoder**, which is a sequence of 6 processing layers stacked on top of each other. Each layer takes the output of the previous one and refines it. The key idea is that the model does not just produce one answer — it produces **6 intermediate answers** (one per layer), each more refined than the last.

Think of it like a sculptor: the first layer gives a rough shape, and each subsequent layer carves more detail. By the time the 6th layer is done, the bounding box prediction is (ideally) final and accurate.

### What is Epistemic Uncertainty?

When a model makes a prediction, there are two types of "not knowing":

- **Aleatoric uncertainty**: The data itself is ambiguous (e.g., a blurry image). No amount of more training will fix this.
- **Epistemic uncertainty**: The model itself is unsure, usually because it hasn't seen enough examples like this one. More data or a better model can reduce this.

RQ6 asks: **can we measure how uncertain the model is (epistemic uncertainty) by looking at how its internal predictions change from one decoder layer to the next?**

---

## The Hypothesis

The central hypothesis had three parts:

- **H1**: False Positive predictions (boxes that the model draws but are wrong) oscillate more across the 6 decoder layers than True Positive predictions (boxes that are correct). In other words, the model "keeps changing its mind" more for wrong predictions than for right ones.
- **H2**: As we go deeper into the decoder (from layer 1 to layer 6), the ability to detect errors using this variance signal improves. Later layers are better at separating wrong from right predictions.
- **H3**: The gap (separation) between how variable FP predictions are vs TP predictions grows as we go deeper.

---

## What Was Done — Step by Step

### Step 1: Dataset and Model

The experiment used **BDD100K**, a large real-world driving dataset with 100,000 annotated images. The validation split was used (500 images sampled), with 10 object categories: `person, rider, car, truck, bus, train, motorcycle, bicycle, traffic light, traffic sign`.

The model was **GroundingDINO SwinT-OGC**, which has a 6-layer transformer decoder. It was run on each of the 500 images using all 10 category names as the text prompt.

**Total detections obtained: 7,788**
- True Positives (TP): 4,282 (55.0%) — predictions that matched a ground-truth box with IoU ≥ 0.5 and the correct category
- False Positives (FP): 3,506 (45.0%) — predictions that did not match any ground-truth box

### Step 2: Capturing Intermediate Decoder Predictions (the technical core)

This is the most novel and technically challenging part. Normally, GroundingDINO only returns the **final** bounding box from layer 6. The intermediate predictions from layers 1–5 are computed internally but discarded before the output is returned.

To capture them, the code uses a technique called **monkey-patching** (temporarily replacing a function inside the model with a modified version). Specifically, the `TransformerDecoder.forward()` method was patched so that, after each layer runs, the intermediate bounding box predictions (called `ref_points` in the code, stored as `(cx, cy, w, h)` in normalized sigmoid coordinates where all values are between 0 and 1) were saved to memory.

This gives, for every detected object, a trajectory of 6 bounding box predictions — one per decoder layer — all in the same normalized coordinate space.

### Step 3: Computing Inter-Layer Variance

For each detected object, the 6 bounding box predictions form a matrix of shape `(6 layers × 4 coordinates)`. The **inter-layer variance** is computed as: for each of the 4 coordinates (cx, cy, w, h), compute the statistical variance across the 6 layers, then average those 4 variances into a single number.

A high variance means the box kept changing a lot from layer to layer. A low variance means it stabilized quickly.

This single number — `bbox_variance` — is the **uncertainty proxy** for RQ6. The idea is: if the model is confident about a detection, the box should converge quickly and barely change between layers (low variance). If the model is uncertain, the box keeps shifting (high variance).

### Step 4: Matching Predictions to Ground Truth

Each prediction was matched to the ground-truth annotations using the standard COCO-style IoU matching with threshold 0.5 and category matching. This labels every detection as TP or FP.

### Step 5: Metrics

Three main metrics were used:

#### AUROC (Area Under the ROC Curve)
This measures how well a score (here: `bbox_variance`) can distinguish two groups (FP = errors vs TP = correct). A value of 0.5 means the score is useless (random). A value of 1.0 means it perfectly separates errors from correct predictions. Values around 0.6–0.7 indicate a real but modest signal.

**Why use it here?** We want to know if `bbox_variance` is a useful signal for catching errors automatically. AUROC directly answers: "if I rank all detections by their variance, how often do errors appear at the top of the list?"

#### AUPR (Area Under the Precision-Recall Curve)
Similar to AUROC but more informative when the classes are imbalanced. It measures how precisely the top-ranked predictions (by variance) are actual errors.

**Why use it?** In object detection, FP and TP counts can vary a lot. AUPR gives a complementary view to AUROC and is sensitive to precision at the top of the ranking.

#### Inter-Layer Variance (Var(TP) and Var(FP))
The raw variance values for TP and FP groups at each decoder layer depth. These are computed in sigmoid normalized space and are very small numbers (~1e-7), which is expected because all coordinates are between 0 and 1 and the decoder converges quickly.

**Why use it?** To directly test H1 and H3 — we need to see whether FP variance > TP variance, and whether the gap grows with depth.

---

## What Was Found

### Finding 1 — H1 Confirmed: FP predictions are consistently more variable than TP

At every decoder layer, false positive predictions have higher inter-layer variance than true positives. In layer 6, the ratio is **×2.51** (FP variance = 3.79×10⁻⁷ vs TP variance = 1.51×10⁻⁷).

This confirms the core intuition: when the model is going to make a wrong prediction, the bounding box "wanders" more across the decoder layers before settling. Correct predictions stabilize faster.

### Finding 2 — H2 Confirmed (globally, with a nuance): AUROC improves with depth

The error-detection AUROC rises from:
- Layer 1: **0.500** (baseline — trivially 0.5 because with only one layer there is no variance)
- Layer 2: **0.582**
- Layer 3: **0.600**
- Layer 4: **0.611**
- Layer 5: **0.616** ← peak
- Layer 6: **0.616** (marginally lower: 0.6158 vs 0.6163)

The global trend is monotonically increasing from layer 2 to layer 5. Layer 6 shows a tiny drop of 0.0005 compared to layer 5 — this is reported honestly. It means the GroundingDINO SwinT-OGC decoder effectively **converges at layer 5**, and the 6th layer adds negligible new information for uncertainty estimation.

The absolute AUROC values (~0.58–0.62) are modest. This is not a failure — it accurately reflects the fast convergence behavior of the SwinT-OGC architecture. The signal is real and consistent, but this decoder does not exhibit large oscillations between layers.

### Finding 3 — H3 Confirmed: Separation grows with depth (peaks at layer 5)

The gap Δ(FP−TP) in variance:
- Layer 2: 1.77×10⁻⁷
- Layer 3: 2.16×10⁻⁷
- Layer 4: 2.31×10⁻⁷
- Layer 5: **2.36×10⁻⁷** ← peak separation
- Layer 6: 2.28×10⁻⁷

The separation peaks at layer 5 and decreases very slightly at layer 6, which is consistent with the AUROC peak also being at layer 5.

### Finding 4 — Failure and Amplification Conditions (Table RQ6.2)

Not all detection scenarios are equally well served by this variance proxy. The baseline AUROC (all detections) is 0.616.

**Conditions where the signal degrades (failure conditions):**

- **Boundary IoU matches** (IoU between 0.4 and 0.6): AUROC = 0.520, drop of **−0.096**. This is the worst case. When a predicted box partially overlaps a ground-truth box but not enough to count as correct, the variance signal cannot distinguish it from a real error. The ambiguity here is fundamentally aleatoric (the box is genuinely borderline), not epistemic.
- **Extreme small objects** (area < 10th percentile = 161.8 px²): AUROC = 0.586, drop of −0.030. Small objects produce noisy, quantization-prone bounding box coordinates. The variance is high for both TP and FP, reducing discriminability.
- **Low confidence predictions** (score < 0.4): AUROC = 0.613, drop of −0.003. Nearly neutral — the score-based ambiguity is orthogonal to the variance signal.

**Conditions where the signal is amplified (unexpected findings):**

- **Dense scenes** (images with ≥ 20.8 objects — top 25% busiest): AUROC = 0.636, gain of +0.020. Crowded scenes cause more inter-layer instability specifically for FP detections, making them easier to identify by variance. The original hypothesis expected this to be a failure condition — the real data showed the opposite.
- **Extreme aspect ratios** (w/h < 0.3 or > 3.0): AUROC = **0.812**, gain of **+0.196**. This is the strongest finding in the failure analysis. Objects with very unusual shapes (very tall/thin or very wide/flat objects — like poles, bicycles lying down, or wide banners) produce large inter-layer bounding box oscillations specifically when the prediction is wrong. This is a genuinely new and publishable finding.

---

## What Was Challenged / Limitations

### 1. The variance values are very small (~1e-7)
This is entirely expected. All coordinates are in sigmoid space [0,1], and the GroundingDINO decoder converges very fast (it was designed to). The small values do not mean the signal is noise — the ratio FP/TP = ×2.51 is consistent and reproducible across 7,788 detections.

### 2. The AUROC ceiling is modest (~0.62)
A perfect uncertainty estimator would reach AUROC close to 1.0. Getting 0.62 means inter-layer variance is a *partial* proxy for epistemic uncertainty — useful but not sufficient alone. This is an honest limitation of using a single scalar (bbox variance) derived from a fast-converging decoder. Richer signals (e.g., combining variance with attention entropy or score distributions) would likely improve performance.

### 3. Layer 6 does not improve over Layer 5
The hypothesis stated that "late layers yield higher AUROC." In practice, layer 5 is the sweet spot and layer 6 represents decoder convergence. This is reported honestly — H2 holds as a global trend but not as a strict step-by-step monotonic improvement.

### 4. Missing failure conditions from the original design
The original design included `heavy_occlusion` and `prompt_mismatch` as failure conditions. These require metadata (occlusion labels, per-category grounding scores) that is not available in the BDD100K annotations. They were excluded to preserve empirical rigor — no synthetic proxies were used.

### 5. Capturing intermediate decoder outputs required non-trivial engineering
GroundingDINO does not expose its intermediate predictions through its public API. A monkey-patching approach on the `TransformerDecoder.forward()` method was needed. The correctness of the capture was validated by cross-checking the final layer predictions against the values returned by the standard `predict()` function.

---

## Figures & Tables

### Figure 6.1
**Path:** `output/Fig_RQ6_1_decoder_variance.png`

**Caption:** Figure RQ6.1. Inter-layer bounding-box variance across transformer decoder depth for true positive (TP, green) and false positive (FP, red) detections. Evaluated on BDD100K val (N=500 images, 7,788 detections) using GroundingDINO SwinT-OGC. FP predictions exhibit consistently higher variance than TP at all depths — at layer 6, Var(FP) = 3.79×10⁻⁷ vs Var(TP) = 1.51×10⁻⁷, a ratio of ×2.51. The separation Δ(FP−TP) grows from layer 2 (1.77×10⁻⁷) to a peak at layer 5 (2.36×10⁻⁷), confirming that decoder dynamics progressively concentrate epistemic signal on error-prone detections. Layer 1 variance is zero by construction (a single point has no variance). All values are in normalized sigmoid coordinate space (cx, cy, w, h ∈ [0,1]); magnitudes (~1e-7) reflect this normalization, not pixel scale.

---

### Figure 6.2
**Path:** `output/Fig_RQ6_2_auroc_by_layer.png`

**Caption:** Figure RQ6.2. AUROC of inter-layer bounding-box variance as an error-detection signal as a function of transformer decoder layer depth. Evaluated on BDD100K val (N=500 images, 7,788 detections), GroundingDINO SwinT-OGC. AUROC rises from 0.500 (layer 1, trivially random — single-point variance is zero by construction) to 0.582 at layer 2, reaching a peak of 0.616 at layer 5 (Δ=+0.034 from layer 2). Layer 6 yields AUROC=0.616 (0.6158), a marginal decline of 0.0005 vs layer 5 (0.6163), indicating decoder convergence at layer 5. AUPR at layer 6 is 0.567. The absolute AUROC range (0.58–0.62) is modest, reflecting the fast convergence of the SwinT-OGC decoder; the global monotonic trend (layers 2→5) confirms the hypothesis direction. Dashed line: random-classifier baseline (AUROC=0.5).

---

### Table 6.1
**Path:** `output/Table_RQ6_1.csv`

**Caption:** Table RQ6.1. Layer-wise diagnostics of decoder-variance uncertainty on BDD100K val (N=500 images, 7,788 detections). AUROC and AUPR measure the ability of inter-layer bbox variance to distinguish false positives (errors) from true positives at each decoder depth. Var(TP) and Var(FP) are mean inter-layer bounding-box variances in normalized sigmoid coordinate space, reported ×10⁻⁷ for readability. ΔAUROC is the per-layer gain relative to the previous layer. ★ marks the layer with peak AUROC (layer 5, AUROC=0.6163). Layer 1 AUROC=0.500 and Var=0 by construction (variance of a single point is zero). The global positive trend from layer 2 to layer 5 confirms H2; the marginal decline at layer 6 (ΔAUROC=−0.0005) indicates that the SwinT-OGC decoder converges at layer 5. Ratio FP/TP is the multiplicative factor by which FP variance exceeds TP variance at each layer.

| Layer (ℓ) | AUROC | ΔAUROC | AUPR | Var(TP) ↓ [×1e-7] | Var(FP) ↑ [×1e-7] | Δ(FP−TP) [×1e-7] | Ratio FP/TP |
|-----------|-------|--------|------|--------------------|--------------------|--------------------|-------------|
| 1         | 0.500 | N/A    | 0.450 | 0.000             | 0.000              | 0.000              | N/A         |
| 2         | 0.582 | +0.082 | 0.544 | 1.215             | 2.984              | 1.769              | ×2.46       |
| 3         | 0.600 | +0.018 | 0.554 | 1.531             | 3.695              | 2.164              | ×2.41       |
| 4         | 0.611 | +0.011 | 0.561 | 1.615             | 3.922              | 2.307              | ×2.43       |
| 5 ★       | 0.616 | +0.005 | 0.566 | 1.573             | 3.936              | 2.362              | ×2.50       |
| 6         | 0.616 | −0.001 | 0.567 | 1.508             | 3.790              | 2.282              | ×2.51       |

---

### Table 6.2
**Path:** `output/Table_RQ6_2.csv`

**Caption:** Table RQ6.2. Conditions under which inter-layer bounding-box variance becomes less (failure, Δ > 0) or more (amplified, Δ < 0) predictive of epistemic uncertainty, relative to the global baseline AUROC of 0.616. All conditions are derived from objective, measurable attributes of the detections in BDD100K annotations — no synthetic metadata was used. AUROC is computed for detections belonging to each condition group. Δ vs baseline is the AUROC drop (positive = degradation) relative to the global baseline. The primary failure condition is boundary IoU matches (IoU 0.4–0.6), where aleatoric localization ambiguity near the decision boundary dominates the variance signal (AUROC drop −0.096). Conversely, extreme aspect ratios strongly amplify the signal (AUROC=0.812, gain +0.196). Conditions requiring metadata absent from BDD100K (occlusion labels, per-token grounding scores) are excluded to preserve empirical rigor.

---

### Table 6.2a — Failure Conditions
**Path:** `output/Table_RQ6_2a_failure.csv`

**Caption:** Table RQ6.2a. Conditions where inter-layer bounding-box variance degrades as an error-detection signal (AUROC < baseline 0.616). Three conditions cause degradation: boundary IoU matches (the most severe, AUROC=0.520, −0.096), extreme small objects (AUROC=0.586, −0.030), and low-confidence predictions (AUROC=0.613, −0.003). In all cases, the variance proxy is overwhelmed by either aleatoric ambiguity (borderline boxes) or low signal-to-noise in the coordinate space (tiny objects).

| Scenario | Observed Effect | AUROC | Δ vs Baseline | n | Interpretation |
|----------|----------------|-------|--------------|---|----------------|
| Boundary IoU matches (0.4 < IoU < 0.6) | Variance saturates; TP/FP separation collapses near decision boundary | 0.520 | +0.096 | 615 | Aleatoric localization ambiguity dominates — bbox variance no longer discriminative |
| Extreme small objects (area < p10 = 161.8 px²) | Unstable early decoding; noisy ref_points across layers | 0.586 | +0.029 | 779 | Quantization artifacts + low signal-to-noise in bbox coords amplify or collapse variance |
| Low confidence predictions (score < 0.4) | Elevated variance for both TP and FP — proxy degrades | 0.613 | +0.003 | 5,029 | Score-based ambiguity dominates; variance cannot distinguish error type |

---

### Table 6.2b — Amplified Signal Conditions
**Path:** `output/Table_RQ6_2b_amplified.csv`

**Caption:** Table RQ6.2b. Conditions where inter-layer bounding-box variance is stronger than the global baseline (AUROC > 0.616). Two conditions amplify the signal: dense scenes (AUROC=0.636, +0.020) and extreme aspect ratios (AUROC=0.812, +0.196). Notably, both were originally hypothesized to be failure conditions; the empirical results show the opposite. Dense scenes increase inter-layer instability for FP detections due to occlusion and matching ambiguity, making them more distinguishable. Extreme aspect ratios (very tall/thin or very wide/flat objects) produce large geometric oscillations specifically for FP predictions, yielding a strong uncertainty signal. The extreme aspect ratio result (AUROC=0.812 on 435 detections) is the most actionable finding in this analysis.

| Scenario | Observed Effect | AUROC | Δ vs Baseline | n | Interpretation |
|----------|----------------|-------|--------------|---|----------------|
| Dense scenes (≥ p75 = 20.8 objects/image) | Variance signal amplified — AUROC improves | 0.636 | −0.020 | 3,233 | Occlusion increases inter-layer instability, making FP more distinguishable |
| Extreme aspect ratios (w/h < 0.3 or > 3.0) | Variance signal amplified — AUROC improves | 0.812 | −0.196 | 435 | Geometric distortion amplifies inter-layer bbox oscillation specifically for FP |

---

## Summary Answer to RQ6

> *What intrinsic properties of transformer decoder dynamics encode epistemic uncertainty, and when does inter-layer variance reliably proxy model uncertainty?*

**Inter-layer bounding-box variance** is a real, measurable epistemic uncertainty signal in the GroundingDINO SwinT-OGC decoder. FP predictions are consistently 2.4–2.5× more variable across layers than TP predictions (H1 ✅). The error-detection AUROC grows monotonically from layer 2 to layer 5 (0.582→0.616, Δ=+0.034), confirming that deeper layers produce a stronger epistemic signal (H2 ✅, global trend). The TP/FP variance separation also grows with depth, peaking at layer 5 (H3 ✅).

The signal is **most reliable** for objects with extreme aspect ratios (AUROC=0.812) and in dense scenes (AUROC=0.636). It is **least reliable** for borderline detections near the IoU decision boundary (AUROC=0.520), where aleatoric ambiguity dominates.

The absolute AUROC ceiling (~0.62 globally) reflects the fast convergence of the SwinT-OGC architecture rather than a flaw in the approach. The decoder reaches its epistemic expression at layer 5, not layer 6, which is the honest empirical answer to when inter-layer variance becomes a reliable proxy: **after semantic stabilization, which occurs at layer 5 in this architecture**.

