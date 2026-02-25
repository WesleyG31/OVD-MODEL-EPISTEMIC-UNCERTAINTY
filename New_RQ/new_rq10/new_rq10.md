# RQ10 — Decision-theoretic Selective Prediction Using Calibrated Uncertainty

**Research Question:** How can calibrated epistemic uncertainty support selective prediction rules that minimize risk under coverage constraints?

---

## Figures & Tables

**Figure 10.1** — `output/Fig_RQ10_1_selective_decision.png`
**Figure 10.2** — `output/Fig_RQ10_2_fp_fn_tradeoff.png`
**Table 10.1** — `output/Table_RQ10_1_operating_points.csv`
**Table 10.2** — `output/Table_RQ10_2_metric_relevance.csv`

---

## Updated Captions

**Figure 10.1** — Risk–coverage trade-off curves for three selective prediction policies (score threshold, uncertainty reject, and joint normalised score + uncertainty) applied to Baseline, Baseline + Temperature Scaling, and MC-Dropout + Temperature Scaling. The joint policy achieves the lowest risk at coverage levels below 0.85; score-only and uncertainty-only policies converge at high coverage (≥ 0.90). All values are derived from real GroundingDINO predictions on the BDD100K validation set.

**Figure 10.2** — False positive (FP) rate and false negative (FN) rate as a function of coverage for the three selective rejection policies applied to MC-Dropout + Temperature Scaling. Contrary to the anticipated asymmetric trade-off, the joint policy reduces both FP and FN rates simultaneously across all coverage levels, while uncertainty-only rejection increases both error types relative to the score baseline at low coverage.

**Table 10.1** — Operating-point comparison at coverage = 0.90 and coverage = 0.80 for MC-Dropout + Temperature Scaling. Risk, FP rate, and FN rate are reported for each policy alongside the signed risk difference (ΔRisk) relative to the score-threshold baseline. Negative ΔRisk indicates improvement; positive ΔRisk indicates degradation.

**Table 10.2** — Pearson absolute correlation (|ρ|) and coefficient of variation (CV) for four evaluation metrics against operational risk (Risk@0.90), computed across six model variants. Contrary to the common assumption, mAP shows the highest correlation (|ρ| = 0.93) with operational risk in this experiment, while AURC shows the lowest (|ρ| = 0.14) and is flagged as statistically unstable due to near-zero variance across methods.

---

## 1. Background and Motivation

### What problem is this trying to solve?

When an AI model makes predictions — in this case, detecting objects in images for autonomous driving — it does not always know when it is wrong. A model might confidently detect something that does not exist (a **false positive**, or FP) or miss a real object entirely (a **false negative**, or FN). Both types of errors are costly in safety-critical settings.

One way to reduce errors is **selective prediction**: instead of always outputting every detection, the model can *choose to abstain* from reporting detections it is uncertain about. The idea is: if you only show the detections the model is most confident about, the ones you do show are more likely to be correct. You accept fewer detections (lower **coverage**), but the ones you do accept carry lower risk.

This RQ investigates whether adding **epistemic uncertainty** — a measure of how much the model "does not know" — on top of the standard confidence score improves this selection process. Epistemic uncertainty specifically captures the model's uncertainty about its own parameters (i.e., structural ignorance), as opposed to randomness in the data.

### What is Coverage and Risk?

- **Coverage** is the fraction of all generated detections that the system chooses to accept and report. Coverage = 1.0 means "report everything"; Coverage = 0.5 means "report only the top 50% of detections".
- **Risk** is the error rate among accepted detections: Risk = (FP + FN) / total detections. A lower risk at a given coverage level means the policy is better at keeping good detections and discarding bad ones.

The core idea is the **risk–coverage trade-off**: as you become more selective (lower coverage), risk should decrease because you are keeping only the best predictions. The question is *which signal* best guides that selection.

---

## 2. How the Study Was Approached

### Models evaluated

Six model variants from prior phases of this project were analysed. All are based on **GroundingDINO**, an open-vocabulary object detection model evaluated on the BDD100K autonomous driving dataset (10 object categories: person, car, bus, truck, etc.):

| Model | What it is |
|---|---|
| Baseline | Standard GroundingDINO, no uncertainty |
| Baseline + Temp. Scaling | Baseline with confidence scores recalibrated using Temperature Scaling |
| MC-Dropout | GroundingDINO with Dropout kept active at inference to generate uncertainty estimates |
| MC-Dropout + Temp. Scaling | MC-Dropout with recalibrated scores |
| Decoder Variance | Uncertainty estimated from variance across decoder attention outputs |
| Decoder Variance + Temp. Scaling | Decoder Variance with recalibrated scores |

**Baseline and Baseline+TS have zero uncertainty by design** — they produce no epistemic uncertainty signal whatsoever. For these models, only score-based selection is possible.

### Three selection policies

For each model, three policies were implemented and compared:

1. **Score threshold**: Sort detections by confidence score (highest first) and accept the top fraction. This is the standard industry approach.
2. **Uncertainty reject**: Sort detections by uncertainty (lowest uncertainty first, since low uncertainty means the model is more confident) and accept the least uncertain ones. For Decoder Variance, the direction is reversed because its uncertainty is positively correlated with correct detections.
3. **Joint (normalised score + uncertainty)**: Combine score and uncertainty into a single ranking signal using the formula `joint = score / (1 + uncertainty_normalised)`. Detections with high score *and* low uncertainty rise to the top; detections with low score *or* high uncertainty get pushed down.

### Why a normalisation step was necessary

MC-Dropout uncertainty values are extremely small in absolute scale — on the order of 1×10⁻⁴. Without any adjustment, the formula `score / (1 + 0.0001)` is virtually identical to `score / 1`, meaning the uncertainty term has essentially no effect on the ranking. To make uncertainty numerically meaningful, a **scale factor** was applied: `unc_scale = 1 / mean_uncertainty ≈ 11,401`. After this normalisation, the average normalised uncertainty is approximately 1.0, so the denominator `(1 + unc_norm)` actually shifts the ranking of detections.

### What data was used

All predictions were loaded from Phase 5 comparison outputs (`eval_baseline.json`, `eval_mc_dropout.json`, etc.). Each detection was matched to ground-truth annotations using **IoU (Intersection over Union) ≥ 0.5** — meaning a detection counts as a true positive only if it overlaps sufficiently with a real annotated object. Detections that do not meet this threshold are false positives.

Coverage levels evaluated: 0.99, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50.

---

## 3. Metrics Used, What They Measure, and Why

### Risk (primary metric)
**What it calculates:** Risk = (number of FP + number of FN) / total detections at a given coverage level.  
**Why it was used:** It directly captures the operational cost of errors at a specific acceptance rate. It is the most deployment-relevant single number: "if I accept this fraction of detections, how wrong am I on average?"

### Coverage
**What it calculates:** The fraction of total detections the system decides to output (accept). Coverage = 0.80 means 80% of detections are reported.  
**Why it was used:** Coverage is the natural constraint in a real deployment — you cannot simply reject everything. The trade-off between coverage and risk defines the operating point of the system.

### FP rate and FN rate (component metrics)
**What they calculate:**  
- FP rate = false positives / accepted detections (how many reported detections are wrong).  
- FN rate = true positives rejected / total real objects (how many real objects the system missed by rejecting them).  
**Why they were used:** They decompose risk into its two components and reveal whether a policy preferentially eliminates one type of error. Safety-critical systems often have asymmetric costs (a missed pedestrian is worse than a spurious detection), so understanding which error type a policy targets matters.

### AURC (Area Under the Risk–Coverage Curve)
**What it calculates:** The integral of the risk curve over all coverage levels (trapezoidal integration). A lower AURC means the policy maintains low risk across all possible operating points, not just at one specific coverage.  
**Why it was used:** A single risk value at one coverage level can be misleading. AURC summarises the quality of the uncertainty ranking across the entire spectrum of acceptance thresholds — it is the standard metric for evaluating selective prediction systems.

### Risk@Coverage (Risk at a specific coverage level)
**What it calculates:** The exact risk value at a fixed coverage level (e.g., Risk@0.90 = risk when 90% of detections are accepted).  
**Why it was used:** It provides a concrete, deployment-facing operating point. A system deployer can say "I need to accept at least 90% of detections — what is my expected error rate?" This is more actionable than AURC alone.

### mAP (Mean Average Precision)
**What it calculates:** The standard object detection accuracy metric. It measures how well the model ranks all detections across all classes and IoU thresholds, averaged across categories.  
**Why it was used:** It is the dominant benchmark metric in the field and serves as a baseline for comparison. The RQ explores whether risk-aware metrics add information beyond what mAP already captures.

### ECE (Expected Calibration Error)
**What it calculates:** The average difference between a model's predicted confidence scores and its actual accuracy at those confidence levels. A perfectly calibrated model with ECE = 0 means "when it says 80% confident, it is right 80% of the time."  
**Why it was used:** Calibration is a prerequisite for meaningful uncertainty-based selection. If confidence scores are poorly calibrated, using them as a selection signal will be unreliable. Temperature Scaling was applied in prior phases specifically to reduce ECE.

### Uncertainty AUROC
**What it calculates:** The Area Under the ROC Curve when using uncertainty values to discriminate between true positives (TP) and false positives (FP). AUROC = 0.5 means uncertainty is random noise; AUROC = 1.0 means perfect separation.  
**Why it was used:** It quantifies how useful the uncertainty signal actually is for identifying incorrect detections. This was computed in Phase 5 and directly informs how much improvement to expect in RQ10.

---

## 4. What Was Found

### 4.1 The uncertainty signal from MC-Dropout is real but modest

MC-Dropout uncertainty achieves an AUROC of **0.648** for discriminating true positives from false positives. This means uncertainty is above random (> 0.50) but far from perfect (< 1.00). Concretely, the model assigns lower uncertainty to correct detections and higher uncertainty to incorrect ones — but the separation is incomplete. Many FPs have low uncertainty and many TPs have high uncertainty, so the signal cannot cleanly separate them.

Decoder Variance produces an uncertainty signal that is *inverted* — higher uncertainty actually correlates with correct detections (AUROC = 0.304, which means if you reverse the direction, AUROC = 0.696). This required special handling in the rejection policy.

### 4.2 Figure 10.1 — Risk–Coverage results

The risk–coverage curves reveal three distinct behaviours:

- **Baseline and Baseline+TS** produce essentially identical curves (they differ only in calibration, not detection order, and have zero uncertainty). Their risk stays high (~0.33–0.40) across all coverage levels.
- **MC-Dropout + TS with score-only** performs similarly to baseline, confirming that the detection model itself is the bottleneck.
- **MC-Dropout + TS with joint policy** achieves noticeably lower risk at coverage ≤ 0.85, dropping to approximately 0.31 at coverage = 0.50. This is the only policy that shows meaningful improvement.
- **Uncertainty-only reject** does *not* improve over score-only and actually increases risk at intermediate coverage levels. The uncertainty signal alone is not strong enough to be used in isolation.

### 4.3 Table 10.1 — Operating points at fixed coverage

| Coverage | Policy | Risk | FP rate | FN rate | ΔRisk vs score |
|---|---|---|---|---|---|
| 0.90 | Score threshold | 0.3792 | 0.3823 | 0.0595 | — |
| 0.90 | Uncertainty reject | 0.3770 | 0.3810 | 0.0577 | **−0.0022** (marginal) |
| 0.90 | Joint + uncertainty | **0.3451** | 0.3633 | 0.0306 | **−0.0342** |
| 0.80 | Score threshold | 0.3493 | 0.3488 | 0.1188 | — |
| 0.80 | Uncertainty reject | 0.3645 | 0.3583 | 0.1316 | **+0.0152 ❌ degradation** |
| 0.80 | Joint + uncertainty | **0.3151** | 0.3275 | 0.0899 | **−0.0342** |

Key findings from Table 10.1:
- The **joint policy consistently reduces risk by ~3.4 percentage points** at both coverage levels tested — a real but modest improvement.
- At coverage = 0.90, uncertainty-only rejection provides almost no benefit (ΔRisk = −0.0022, less than 1%).
- At coverage = 0.80, uncertainty-only rejection **makes things worse** (ΔRisk = +0.0152, marked in red). Sorting by uncertainty alone disrupts the good ordering that confidence scores already provide.
- The absolute risk values are high (~0.31–0.38) because GroundingDINO produces many false positives on BDD100K when run with a low confidence threshold of 0.25 — this is a known characteristic of open-vocabulary detectors on dense driving scenes.

### 4.4 Figure 10.2 — FP/FN trade-off

An important anticipated finding was an *asymmetric* trade-off: that uncertainty-aware rejection would preferentially eliminate false positives while only slightly increasing false negatives. The real results do not match this expectation.

Instead:
- The **joint policy reduces both FP and FN simultaneously** at all coverage levels. At coverage = 0.50, joint achieves FP rate = 0.219 and FN rate = 0.345, compared to score-only at FP = 0.248 and FN = 0.355 — both are lower, not just FP.
- The **uncertainty-only policy increases both FP and FN** at low coverage relative to score-only. It does not selectively filter false positives.
- The expected asymmetric behaviour (big FP reduction, small FN increase) would require a much stronger uncertainty signal — AUROC around 0.85 or higher — to produce a clearly differential effect.

### 4.5 Table 10.2 — Which metric best predicts operational risk?

| Metric | Correlation with Risk@0.90 (|ρ|) | Sensitivity (CV) | Validity |
|---|---|---|---|
| mAP | 0.93 | 0.031 | valid |
| ECE | 0.27 | 0.283 | valid |
| AURC | 0.14 | 0.016 | ⚠ near-zero variance (unstable) |
| Risk@Coverage | 1.00 | 0.060 | tautological (same metric) |

This is the most surprising finding of RQ10. The expectation was that AURC and Risk@Coverage would be more informative than mAP for predicting operational risk. The data shows the opposite:

- **mAP has the highest valid correlation with Risk@0.90 (|ρ| = 0.93)**. Across the six model variants, the models with higher detection accuracy also happen to have lower operational risk. This is not a logical necessity — it reflects the specific structure of this experiment where the methods that improve mAP (MC-Dropout, Decoder Variance) also change the prediction set in ways that affect risk.
- **AURC has essentially no correlation (|ρ| = 0.14) and near-zero variance (CV = 0.016)**. This means AURC is almost identical across all six methods (~0.16), making it useless for ranking methods by their risk profile in this setting. The near-zero variance flag indicates the metric is statistically unstable for this comparison.
- **ECE has low correlation (|ρ| = 0.27)** with risk but high sensitivity (CV = 0.283), meaning ECE varies substantially across methods (ranging from 0.141 to 0.343) but that variation does not track operational risk. Better calibration does not necessarily translate to lower risk.
- **Risk@Coverage is tautologically correlated with itself** (|ρ| = 1.00) — this is not an informative result, just a mathematical identity.

---

## 5. Challenges and Limitations

### The scale problem with uncertainty values
MC-Dropout uncertainty values are on the order of 1×10⁻⁴. This is not an error — it reflects that the model's softmax outputs vary very little across dropout passes, because GroundingDINO's architecture is not designed to be uncertainty-aware. Without the `unc_scale ≈ 11,401` normalisation, the joint formula produces a ranking virtually identical to score-only, and the comparison would be meaningless. The normalisation is methodologically sound but highlights that this model does not naturally produce well-scaled uncertainty estimates.

### The weak uncertainty signal
An AUROC of 0.648 means the uncertainty can separate TPs from FPs only slightly better than chance. In practice, the confidence score (which is the model's own output signal) already encodes much of the TP/FP information. Uncertainty adds a complementary but weak second signal. For the joint policy to produce strong asymmetric FP suppression, AUROC would need to be substantially higher — roughly 0.80 or above.

### The mAP paradox in Table 10.2
The finding that mAP correlates most strongly with operational risk is a consequence of the experimental design, not a general truth. With only six data points (six model variants), two of which are structurally identical to their TS counterparts in terms of prediction content, the statistical conditions for reliable correlation estimation are poor. The result should be interpreted cautiously: it does not mean mAP is a better deployment metric than AURC in general; it means that in this specific comparison, the methods that change mAP also happen to change risk in the same direction.

### High absolute risk values
Risk values around 0.31–0.38 are high. This reflects that GroundingDINO, when used as a zero-shot detector with a confidence threshold of 0.25 on BDD100K, generates many low-quality detections. The selective prediction framework is designed to address this by filtering them — and it does reduce risk — but the absolute floor is constrained by the quality of the base model.

---

## 6. Answer to the Research Question

**How can calibrated epistemic uncertainty support selective prediction rules that minimize risk under coverage constraints?**

Calibrated epistemic uncertainty from MC-Dropout *can* support selective prediction, but only when combined with confidence scores in a normalised joint ranking formula. Using uncertainty alone as the selection signal is ineffective and can degrade performance.

The joint policy `score / (1 + uncertainty_normalised)` consistently reduces risk by approximately 3.4 percentage points compared to score-only thresholding at both coverage = 0.90 and coverage = 0.80. This is a real, measurable improvement derived entirely from real model predictions.

However, the improvement is modest. The fundamental limiting factor is the uncertainty signal strength (AUROC = 0.648 for MC-Dropout). A stronger uncertainty estimator — one that more reliably identifies which detections are wrong — would yield larger risk reductions and could produce the theoretically expected asymmetric FP suppression. With AUROC = 0.648, the uncertainty is informative but not discriminative enough to dramatically change the selection order beyond what confidence already provides.

The practical conclusion is: calibration and uncertainty are worth including in a selective prediction pipeline, but the quality of the uncertainty estimator is the binding constraint. The framework is sound; the signal needs to be stronger to produce the full theoretical benefit.

---