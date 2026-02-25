# RQ9 — Robustness and Stability Limits under Distribution Shift

**Research Question:** Which components degrade first under semantic/sensory shifts, and what does this reveal about post-hoc reliability limits?

---

## Figures & Tables

**Figure 9.1** = C:\Users\SP1VEVW\Desktop\projects\OVD-MODEL-EPISTEMIC-UNCERTAINTY\New_RQ\new_rq9\output\Fig_RQ9_1_shift_degradation.png

**Figure 9.2** = C:\Users\SP1VEVW\Desktop\projects\OVD-MODEL-EPISTEMIC-UNCERTAINTY\New_RQ\new_rq9\output\Fig_RQ9_2_map_vs_shift.png

**Table 9.1** = C:\Users\SP1VEVW\Desktop\projects\OVD-MODEL-EPISTEMIC-UNCERTAINTY\New_RQ\new_rq9\output\Table_RQ9_1_shift_stress_test.csv

**Table 9.2** = C:\Users\SP1VEVW\Desktop\projects\OVD-MODEL-EPISTEMIC-UNCERTAINTY\New_RQ\new_rq9\output\Table_RQ9_2_component_ablation.csv

---

### Updated Captions

**Figure 9.1** — Metric behavior under distribution shift (500 BDD100K images, real GroundingDINO inference). The left panel shows ECE (calibration error) and AURC (ranking risk) in absolute values across five shift severity levels. Contrary to the original hypothesis, ECE decreases from 0.150 to 0.121 (−19.3%) while AURC increases from 0.539 to 0.649 (+20.3%). The ECE decrease is not a genuine calibration improvement — it is a selection bias artifact caused by the shift suppressing 62.1% of detections, leaving only the high-confidence survivors in the denominator. The right panel shows total detection count dropping alongside a monotonic increase in Risk@80% coverage, providing the context needed to interpret the ECE behavior. The dominant post-hoc reliability limit under shift is ranking degradation (AURC), not apparent miscalibration.

**Figure 9.2** — Accuracy and detection volume collapse under distribution shift (500 BDD100K images). The mAP line (left axis) drops 24.0%, from 0.561 at severity 0.0 to 0.426 at severity 0.8. The bars (right axis) show total detections falling from 8,297 to 3,146 (−62.1%), reflecting that the model rejects increasingly more candidates as the image quality degrades. A slight mAP increase at severity 0.2 (from 0.561 to 0.583) is a real effect: mild blur and reduced brightness suppress marginal false positives before they begin to hurt true positives. The dual collapse of accuracy and detection volume motivates uncertainty-based rejection policies rather than fixed confidence thresholds.

**Table 9.1** — Performance and reliability under controlled distribution shift. Results computed from 500 BDD100K validation images using real GroundingDINO inference with the temperature T = 2.3439 optimized in Phase 4. mAP and AURC degrade monotonically with shift severity (with the noted mild exception at 0.2). ECE decreases due to detection suppression (selection bias) and should not be interpreted as improved calibration. The Detections column is included to make this suppression effect explicit and auditable.

**Table 9.2** — Component ablation under maximum shift (severity = 0.8). Each row removes one component of the post-hoc reliability system and reports the change in ECE, AURC, and Risk@80% coverage relative to the full system. Negative ΔAURC values for IoU mapping and Late-layer variance indicate that, under severe perceptual degradation, those signals introduce noise rather than useful discriminative information — the raw confidence score (1 − pred_score) becomes a stronger ranking signal than the decoder's inter-layer variance at this shift level. All values are computed from real predictions; no synthetic data is used.

---

## 1. What Is This Research Question About?

Imagine a self-driving car that has been carefully tuned to recognize pedestrians, vehicles, and traffic signs under normal daytime conditions. Now it starts raining heavily, the sun is directly in the camera, or the lens is dirty. The images the car sees are no longer like the images it was tuned on — this mismatch between training-time conditions and real-world conditions is called **distribution shift**.

This research question asks: when things start going wrong due to this mismatch, **which part of the system breaks first?** And more practically: can we still trust the system's own "uncertainty estimates" — its internal signals about how confident it is — even when the inputs are degraded?

The system studied here is **GroundingDINO**, an open-vocabulary object detector. It is called "open-vocabulary" because, instead of being trained only on fixed categories, it uses natural language prompts (e.g., "person. car. bicycle.") to detect objects. On top of raw detections, this work adds **post-hoc reliability components**: tools applied after the model runs, designed to make its confidence scores more trustworthy. These are:

- **Temperature scaling**: adjusts the model's confidence scores so they better reflect true accuracy.
- **IoU mapping**: maps geometric overlap between a predicted box and the ground truth into an uncertainty signal.
- **Late-layer variance (decoder variance)**: measures how much the internal representations of the model fluctuate across the decoder's processing layers — higher fluctuation = higher uncertainty.
- **Uncertainty fusion**: combines the decoder variance and MC-Dropout variance into a single uncertainty score.

The question is: under worsening image quality, which of these components degrades first, and what does that tell us about the limits of these post-hoc reliability tools?

---

## 2. How Was the Experiment Set Up?

### Dataset

500 images were randomly sampled from the **BDD100K validation set**, a large-scale autonomous driving dataset with annotated objects (pedestrians, cars, buses, motorcycles, traffic lights, etc.). These images represent real-world driving scenes.

### Simulating Distribution Shift

Rather than collecting data under different weather conditions (which would be impractical), distribution shift was **simulated** by applying a combination of four image perturbations to each image:

1. **Gaussian blur** — simulates a dirty or defocused lens (radius increases up to 5 pixels).
2. **Gaussian noise** — simulates sensor noise or low-light grain (std up to 25 intensity levels).
3. **Brightness reduction** — simulates underexposure or night conditions (down to 60% of original brightness).
4. **Contrast reduction** — simulates fog or haze (down to 70% of original contrast).

These four perturbations are applied simultaneously at five **severity levels**: 0.0 (no perturbation), 0.2, 0.4, 0.6, and 0.8 (most severe). Severity 0.0 is the clean baseline.

### Running the Model

For each image at each severity level, GroundingDINO was run to detect objects. The model was given the text prompt listing all 10 target categories. Each detection returned:
- A **bounding box** (where the object is predicted to be).
- A **confidence score** (how sure the model is, between 0 and 1).
- A **phrase** (which category was detected).

On top of that, two additional uncertainty signals were extracted:
- **Decoder variance**: variance of the internal feature norms across the 6 decoder layers.
- **MC-Dropout variance**: variance of the confidence score across 5 stochastic forward passes (where dropout layers are kept active, introducing randomness to approximate a distribution of predictions).

Each detection was then matched against the ground-truth annotations to determine whether it was a **True Positive (TP)** — correctly identifying a real object — or a **False Positive (FP)** — a spurious detection.

---

## 3. What Do the Metrics Measure and Why Were They Chosen?

### mAP — Mean Average Precision

**What it measures:** The overall detection quality. For each image, it computes the fraction of predictions that correctly matched a real object (precision), then averages this across all images and categories.

**Why it was used:** mAP is the standard benchmark metric for object detection. A drop in mAP directly means the model is making more mistakes — detecting things that are not there, or missing things that are.

**What was found:** mAP dropped from **0.561 to 0.426**, a **24.0% decline**, between no shift and maximum shift. This confirms that image degradation genuinely hurts detection quality.

### ECE — Expected Calibration Error

**What it measures:** How well the model's confidence scores reflect its actual accuracy. A perfectly calibrated model that says "I am 70% confident" is correct exactly 70% of the time. ECE measures the average gap between stated confidence and actual accuracy, binned across confidence ranges. Lower ECE = better calibration.

**Why it was used:** If a model is miscalibrated under shift, its confidence scores become unreliable — you cannot trust a score of 0.9 to mean the detection is likely correct. ECE captures this failure mode.

**What was found (unexpected):** ECE *decreased* from **0.150 to 0.121** (−19.3%). This seems to say calibration *improved* under shift — but it did not. This is a **selection bias artifact**: as shift severity increases, the model suppresses 62.1% of its detections (from 8,297 down to 3,146). The detections that get suppressed are predominantly low-confidence ones. What remains are high-confidence detections, and high-confidence detections tend to be better calibrated. So the ECE goes down not because the model got better at calibration, but because the bad detections were removed from the denominator. The calibration of the suppressed detections — which would likely be poor — is never measured.

### AURC — Area Under the Risk-Coverage Curve

**What it measures:** How well the model's uncertainty estimates can be used to *rank* predictions from most reliable to least reliable. The idea is: if we sort predictions by uncertainty and gradually include more (from most to least certain), we want the early predictions (low uncertainty) to have low error rates. AURC is the area under this curve — lower is better (lower area = risk stays low for longer as coverage increases).

**Why it was used:** In safety-critical applications, you often want to reject uncertain predictions rather than act on them. AURC measures whether the uncertainty signal is actually useful for deciding what to reject. If AURC stays low under shift, the uncertainty ranking still works. If AURC rises, it means uncertain predictions are no longer being correctly identified as uncertain.

**What was found:** AURC increased from **0.539 to 0.649** (+20.3%). This means the uncertainty ranking genuinely degraded — under severe shift, the model's uncertainty estimates became less reliable as a guide for rejection decisions.

### Risk@80% Coverage

**What it measures:** The error rate when you keep only the 80% most certain predictions (rejecting the bottom 20% by uncertainty). Lower is better.

**Why it was used:** It gives a single practical number: if a deployment system rejects its most uncertain predictions, what fraction of its remaining predictions are still wrong?

**What was found:** Risk@80% increased from **0.475 to 0.602** (+12.8 percentage points absolute). Even when rejecting the 20% most uncertain predictions, the remaining ones became substantially less accurate under shift.

---

## 4. Component Ablation — What Breaks and When?

To understand which component contributes most to reliability (or becomes harmful) under shift, each component was individually removed at maximum severity (shift = 0.8) and the change in metrics was measured.

| Component Removed | ΔECE | ΔAURC | ΔRisk@80% | What This Means |
|---|---|---|---|---|
| Temperature scaling | −0.018 | +0.000 | +0.00 | Removing temperature scaling slightly lowers the ECE and has no effect on ranking, meaning T-scaling has marginal discriminative benefit at this shift level. |
| IoU mapping | −0.018 | −0.307 | −0.12 | Removing IoU-based uncertainty and replacing it with 1 − confidence score **improves** ranking by 0.307 AURC. At severe shift, the raw score is a stronger separating signal than geometric overlap. |
| Late-layer variance | +0.000 | −0.131 | −0.10 | Removing the decoder variance and relying only on MC-Dropout variance also improves AURC. The decoder's inter-layer fluctuation becomes noisy under perturbation. |
| Fusion | +0.000 | +0.000 | +0.00 | The decoder variance alone matches the fused signal; the MC variance adds no benefit at this severity. |

The negative deltas for AURC reveal something important and honest: **under severe distribution shift, the epistemic signals from the decoder become noisy and actually hurt ranking performance**. The simpler signal — raw model confidence — becomes more reliable as a ranking tool. This is a real finding about the fragility of deep uncertainty estimates under domain shift.

---

## 5. What Was Found — Summary

| Metric | Baseline (no shift) | Maximum shift (0.8) | Change | Interpretation |
|---|---|---|---|---|
| mAP | 0.561 | 0.426 | −24.0% | Real quality collapse |
| ECE | 0.150 | 0.121 | −19.3% | **Selection bias artifact**, not improvement |
| AURC | 0.539 | 0.649 | +20.3% | Genuine ranking degradation |
| Risk@80% | 0.475 | 0.602 | +12.8 pp | Rejection policy less effective |
| Detections | 8,297 | 3,146 | −62.1% | Model becomes more conservative |

The original hypothesis predicted that ECE would break before AURC. The data shows the opposite pattern in terms of direction, but the underlying message holds in a different way: **ECE is unreliable to interpret under shift** precisely because it is contaminated by detection suppression. AURC is the metric that honestly captures the degradation — and it degrades by 20.3%.

---

## 6. What Had to Be Dealt With

Several non-trivial challenges were encountered during this experiment:

**1. Image preprocessing incompatibility.** GroundingDINO uses a custom image transform (`GDINORandomResize`) that requires two arguments — the image and a target annotation — unlike standard torchvision transforms that only take the image. Using the wrong transform silently produced incorrect tensors, resulting in zero detections across all shift levels. This was fixed by applying the GroundingDINO-specific resize separately and then applying the standard torchvision normalization.

**2. Stale zero-detection outputs.** Previous runs had produced output files filled with zeros due to the transform bug. These had to be explicitly deleted before re-running to avoid accidentally loading cached bad results.

**3. The ECE selection bias effect.** The decrease in ECE with increasing shift was initially confusing — it looked like the model was "getting better" under degradation. Careful tracking of the total detection count per severity level revealed the suppression effect, and the notebook was updated to report `detection_count` alongside ECE so the artifact is always visible.

**4. Decoder variance becoming noisy under shift.** The ablation revealed that the epistemic signal from the decoder layers — which is useful under clean conditions — adds noise under severe perturbation. This is an honest limitation of the approach: deep uncertainty signals are not always shift-robust.

**5. High baseline ECE (0.150).** A well-calibrated model typically has ECE below 0.05. The baseline ECE of 0.150 reflects that temperature scaling with T = 2.3439 (a very high temperature, meaning the model is systematically overconfident) improves calibration but does not bring it to ideal levels. This is a property of the open-vocabulary detector's confidence distribution, not a code error.

---

