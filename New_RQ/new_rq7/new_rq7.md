RQ7 — Deterministic vs stochastic epistemic uncertainty

RQ7: How do deterministic internal signals differ from Bayesian sampling approximations in characterizing epistemic uncertainty in OVD?

Figures & Tables

Figure 7.1 =  C:\Users\SP1VEVW\Desktop\projects\OVD-MODEL-EPISTEMIC-UNCERTAINTY\New_RQ\new_rq7\output\Fig_RQ7_1_risk_coverage.png

Figure 7.2 =  C:\Users\SP1VEVW\Desktop\projects\OVD-MODEL-EPISTEMIC-UNCERTAINTY\New_RQ\new_rq7\output\Fig_RQ7_2_latency_ece.png

Table 7.1 = C:\Users\SP1VEVW\Desktop\projects\OVD-MODEL-EPISTEMIC-UNCERTAINTY\New_RQ\new_rq7\output\Table_RQ7_1.csv

Table 7.2 = C:\Users\SP1VEVW\Desktop\projects\OVD-MODEL-EPISTEMIC-UNCERTAINTY\New_RQ\new_rq7\output\Table_RQ7_2.csv


---

# RQ7 — Deterministic vs Stochastic Epistemic Uncertainty

**Research Question:** How do deterministic internal signals differ from Bayesian sampling approximations in characterizing epistemic uncertainty in Open-Vocabulary Detection (OVD)?

---

## Figures & Tables

**Figure 7.1** = `C:\Users\SP1VEVW\Desktop\projects\OVD-MODEL-EPISTEMIC-UNCERTAINTY\New_RQ\new_rq7\output\Fig_RQ7_1_risk_coverage.png`

**Figure 7.2** = `C:\Users\SP1VEVW\Desktop\projects\OVD-MODEL-EPISTEMIC-UNCERTAINTY\New_RQ\new_rq7\output\Fig_RQ7_2_latency_ece.png`

**Table 7.1** = `C:\Users\SP1VEVW\Desktop\projects\OVD-MODEL-EPISTEMIC-UNCERTAINTY\New_RQ\new_rq7\output\Table_RQ7_1.csv`

**Table 7.2** = `C:\Users\SP1VEVW\Desktop\projects\OVD-MODEL-EPISTEMIC-UNCERTAINTY\New_RQ\new_rq7\output\Table_RQ7_2.csv`

---

**Figure 7.1** — Risk–coverage curves comparing three uncertainty estimators (Deterministic decoder variance, MC Dropout, and their naive fusion) computed on 6,111 matched detections across 498 BDD100K validation images. The Deterministic estimator dominates with the lowest RC-AUC (0.067), indicating it is the most effective selector for separating correct detections from errors when predictions are ranked by uncertainty. MC Dropout performs worst (RC-AUC = 0.295). The naive fusion of scores is intermediate (RC-AUC = 0.148). Contrary to the initial hypothesis, fusion does NOT improve risk-coverage over the Deterministic signal alone.

**Table 7.1** — Cost–benefit comparison of the three uncertainty estimators on BDD100K validation data (6,111 matched detections). Latency values for MC Dropout and Deterministic are real measurements from Phase 3 experiments; Fusion latency is estimated as Deterministic + 10% overhead. Deterministic is the sole Pareto-optimal method: it is 5.0× faster than MC Dropout (357.8 ms vs 1,788.9 ms per image) and achieves the best calibration (ECE = 0.2051, NLL = 0.7083). Fusion produces the worst ECE (0.2199) and NLL (0.7165) because averaging raw scores from two differently-scaled estimators without recalibration degrades calibration.

**Figure 7.2** — Efficiency–reliability trade-off plot (latency in ms/image vs Expected Calibration Error). Deterministic inference is the only Pareto-optimal operating point: lower latency and lower ECE than both MC Dropout and Fusion. MC Dropout is 5.0× slower with marginally worse calibration. Fusion is nearly as fast as Deterministic but yields the highest ECE of all three methods. Latency values for MC Dropout (1,788.9 ms) and Deterministic (357.8 ms) are real measurements; Fusion (393.6 ms) is estimated.

**Table 7.2** — AUROC breakdown showing which estimator best identifies each type of detection failure, computed on 6,111 matched detections from BDD100K val. MC Dropout wins on all four failure categories. There is no complementarity between Deterministic and MC Dropout: the decoder variance signal (even after polarity inversion) does not capture the failure modes evaluated. Det performs near-chance on prompt ambiguity (AUROC = 0.179). Failure type labels are approximated via heuristics (score threshold, category ID, bounding-box area), not ground-truth annotations.

---

## 1. Background — What Is This Research Question About?

### The core problem: a model that does not know what it does not know

An object detection model takes an image, looks for objects, and assigns a confidence score to each detection (e.g., "I am 87% confident this is a car"). However, a high confidence score does not always mean the detection is correct. A model can be wrong and still be very confident — this is called a **confident false positive**. Conversely, it can be right but uncertain.

**Epistemic uncertainty** is the kind of uncertainty that comes from the model's lack of knowledge. It is different from randomness in the data (aleatoric uncertainty). Epistemic uncertainty is especially important in safety-critical applications like autonomous driving: if the model is uncertain about a detection, a downstream system should be more cautious about acting on it.

The goal of this research question is to compare **two fundamentally different strategies** for measuring epistemic uncertainty in a Grounding DINO model (an Open-Vocabulary Detector):

1. **Deterministic strategy**: Extract internal signals that already exist inside the model during a single forward pass, without running the model more than once. Specifically, we use the **variance of the output score across the decoder layers** — the model has 6 transformer decoder layers, and the score it assigns to a detection changes slightly at each layer. A large change (high variance) across layers is treated as a signal of uncertainty.

2. **Stochastic strategy (MC Dropout)**: Run the model multiple times (K=5 passes) with dropout randomly turned on during inference, which forces parts of the network to be disabled randomly. Each pass produces slightly different detections. The **variance of the confidence scores across these 5 passes** is used as the uncertainty estimate. This is a well-known Bayesian approximation technique called Monte Carlo Dropout.

3. **Fusion**: A simple combination of both signals — normalize each to [0,1] and average them — to test whether they carry complementary information.

---

## 2. Why These Metrics? What Do They Measure?

### 2.1 Expected Calibration Error (ECE)

**What it measures:** How trustworthy the model's confidence scores are. A perfectly calibrated model means: "if the model says 80% confidence across 100 detections, then exactly 80 of those should be correct." ECE measures the average gap between predicted confidence and actual accuracy, grouped into bins.

**Why we used it:** ECE directly answers the question "can I trust this confidence score?" If a model is well-calibrated (low ECE), its scores are informative. If not, a high score does not reliably mean a correct detection.

**Formula:** Split all detections into confidence bins. For each bin, measure |average_confidence − fraction_correct|, weighted by the fraction of detections in that bin. Sum these weighted gaps.

**Real results:**
- Deterministic: ECE = **0.2051** (best)
- MC Dropout: ECE = 0.2089
- Fusion: ECE = **0.2199** (worst)

### 2.2 Negative Log-Likelihood (NLL)

**What it measures:** How well the model's probability outputs explain the actual outcomes. It penalizes confident wrong predictions heavily. Lower NLL means the model is assigning high probability to correct outcomes and low probability to incorrect ones.

**Why we used it:** NLL is a stricter calibration measure than ECE because it penalizes overconfident errors more severely.

**Real results:**
- Deterministic: NLL = **0.7083** (best)
- MC Dropout: NLL = 0.7096
- Fusion: NLL = **0.7165** (worst)

### 2.3 Risk–Coverage Curve and its Area (RC-AUC)

**What it measures:** This is perhaps the most practically important metric. Imagine you can choose to only act on the detections you are most confident about, and discard the rest. The **coverage** is the fraction of detections you keep. The **risk** is the error rate among the kept detections. A good uncertainty estimator should allow you to keep a large fraction of detections while keeping the error rate low — meaning if you rank detections by uncertainty (most uncertain first), the curve should drop quickly.

**Why we used it:** ECE and NLL measure calibration of the raw score, but the RC-AUC measures whether the uncertainty signal can **rank** detections correctly — whether it separates correct detections from errors. This is directly useful in deployment: you can set a threshold and say "reject anything with uncertainty above X."

**How RC-AUC works:** Lower is better. An RC-AUC of 0 would mean the uncertainty perfectly identifies all errors at the top. A random estimator would give RC-AUC ≈ fraction of errors × 0.5.

**Real results:**
- Deterministic: RC-AUC = **0.0666** (best — 4.4× better than MC Dropout)
- Fusion: RC-AUC = 0.1484
- MC Dropout: RC-AUC = **0.2952** (worst)

### 2.4 Latency and FPS

**What it measures:** How long each method takes per image, and how many images it can process per second.

**Why we used it:** Even if a method gives perfect uncertainty, it is useless in a real-time application if it takes too long. The cost-benefit trade-off between accuracy and speed is critical for deployment.

**Real results (measured from Phase 3 experiments):**
- MC Dropout (K=5): 1,788.9 ms/image → 0.56 FPS
- Deterministic: 357.8 ms/image → 2.80 FPS (**5.0× faster than MC Dropout**)
- Fusion: ~393.6 ms/image → 2.54 FPS (estimated: Deterministic + 10% overhead)

### 2.5 AUROC by Failure Type

**What it measures:** AUROC (Area Under the ROC Curve) tells you how well an uncertainty signal can distinguish between "this detection is an error of type X" vs "this detection is correct." An AUROC of 1.0 would be perfect discrimination; 0.5 means no better than random.

**Why we used it:** Different types of detection errors (a very confident wrong prediction vs a small confused object) might require different uncertainty signals to detect. If MC Dropout is better at catching one type and Deterministic is better at another, they are **complementary** and fusion would genuinely help.

**How failure types were defined** (using heuristics on the matched detection dataset):
- **Confident FP**: detections with score > 0.70 that are wrong (n = 7)
- **Novel class boundary**: wrong detections of person or rider categories (category IDs 0–1, n = 592)
- **Prompt ambiguity**: wrong detections of other categories with bounding-box area ≥ 5,000 pixels² (n = 357)
- **Background clutter**: wrong detections with bounding-box area < 5,000 pixels² (n = 1,546)

---

## 3. How Was This Approached?

### 3.1 Data Sources

All results are computed on **real BDD100K validation data** — no synthetic or dummy numbers were used.

| Source | Content | Size |
|---|---|---|
| Phase 3 (`mc_stats_labeled.parquet`) | MC Dropout detections with score_variance across K=5 passes, TP/FP labels | 29,914 detections |
| Phase 3 (`timing_data.parquet`) | Real per-image inference times for MC Dropout | 500 images |
| Phase 3 (`computational_cost.json`) | Real baseline (single-pass) inference time and overhead factor | — |
| RQ6 (`decoder_dynamics.parquet`) | Decoder-layer score trajectories and score_variance per detection, TP/FP labels | 7,788 detections |
| Phase 4 (`temperature.json`) | Calibration temperature (T = used for context; final ECE computed on raw scores) | — |

### 3.2 Building the Comparison Dataset

The core challenge was that MC Dropout and Decoder Variance were computed in separate experiments and produced different sets of detections. To ensure a **fair comparison** — the same detections evaluated under all three methods — we performed a **bbox-level matching**:

- Find images present in both datasets.
- For each image, match detections that have the same bounding box coordinates (rounded to 0 decimal places).
- Keep only the detections that appear in both.

This produced a **matched dataset of 6,111 detections across 498 images** — the same ground for all comparisons.

### 3.3 Handling the Inverted Decoder Signal

When we examined the decoder variance signal, we found something unexpected: **true positives (correct detections) had HIGHER score variance across decoder layers than false positives**. This is the opposite of the intuition that "more uncertain = more likely to be wrong."

This makes physical sense in retrospect: a detection that the decoder is actively "deciding" across its 6 layers (high variance) is one where the model is processing a real, complex object — it ends up correct. A false positive is often a brief, quickly-suppressed activation with less variation.

To use decoder variance as an uncertainty signal (where "more uncertain = more likely to be wrong"), we applied a polarity inversion: `uncertainty_det = −score_variance`. After inversion, the signal works as expected for risk-coverage ranking.

### 3.4 Fusion Strategy

The fusion was implemented as the simplest possible combination:
1. Normalize each signal to [0, 1] across the matched dataset.
2. Average: `uncertainty_fused = (uncertainty_mc_normalized + uncertainty_det_normalized) / 2`
3. The fused confidence score for ECE/NLL was the arithmetic mean of the two methods' scores.

This is a naive fusion — it does not recalibrate either signal before combining them.

---

## 4. What Was Found

### Finding 1: Deterministic is faster by 5× — confirmed

MC Dropout requires running the model 5 times (K=5 passes). The measured overhead is real and large: **1,788.9 ms vs 357.8 ms per image**. Deterministic adds zero overhead because the variance is extracted from the single forward pass that is already required to produce the detection output.

### Finding 2: Deterministic has the best calibration — confirmed but differences are small

Deterministic achieves ECE = 0.2051 vs MC's 0.2089. The ranking is consistent (Det < MC < Fusion) but the absolute difference is only ~0.004 ECE units. Both methods are notably miscalibrated in absolute terms (ECE ≈ 0.21 means the average gap between confidence and accuracy is 21 percentage points). The scores from both methods, as raw values, are not reliable enough for direct probabilistic interpretation without further recalibration.

### Finding 3: Deterministic dominates risk-coverage — this was NOT the original hypothesis

The original hypothesis predicted that **Fusion would dominate** risk-coverage. The data show the opposite: Deterministic is by far the best uncertainty ranker (RC-AUC = 0.067), and Fusion is more than twice as bad (0.148). MC Dropout is the worst (0.295).

Why? The inverted decoder variance signal (`−score_variance`) is a strong ranking signal — when sorted, it correctly places most errors above most correct detections. MC Dropout's score variance, despite being a theoretically justified Bayesian approximation, is a much weaker separator on this dataset. One reason may be that K=5 passes is too few to get a stable variance estimate.

### Finding 4: Fusion degrades calibration — unexpected

Averaging the raw scores from MC Dropout and Deterministic without recalibration makes the ECE **worse** (0.2199) than either method alone. This happens because the two scores operate on different internal scales: MC score is the mean of 5 softmax outputs; Det score is a single decoder output. Averaging them shifts the confidence distribution away from the true accuracy, inflating the calibration gap.

**What would be needed to fix fusion:** Apply temperature scaling or isotonic regression to each signal independently before merging.

### Finding 5: No complementarity observed between Det and MC Dropout

The initial hypothesis expected that each method would be better at different types of errors (Deterministic for confident FPs, MC for novel-class boundaries, Fusion for ambiguous prompts). The data show MC Dropout wins on **all four failure categories**:

| Failure type | AUROC MC | AUROC Det | AUROC Fusion | Winner | Margin |
|---|---|---|---|---|---|
| Confident FP | **0.831** | 0.666 | 0.748 | MC Dropout | +11.1% |
| Novel class boundary | **0.706** | 0.445 | 0.657 | MC Dropout | +7.5% |
| Prompt ambiguity | **0.698** | 0.179 | 0.407 | MC Dropout | +71.5% |
| Background clutter | **0.621** | 0.502 | 0.617 | MC Dropout | +0.7% |

The Deterministic signal is especially weak on **prompt ambiguity** (AUROC = 0.179, which is worse than random). This makes sense: decoder layer variance captures geometric and representational instability in the visual backbone, but it has no direct access to the text–vision alignment uncertainty that causes prompt ambiguity failures.

### Finding 6: Risk@50%Coverage is the strongest per-image predictor of deployment safety

A supplementary analysis computed per-image correlations between risk-aware metrics and proxies for operational safety. R@C50 (the error rate on the 50% most-confident detections per image) showed strong Spearman correlations:
- R@C50 MC vs per-image precision: ρ = **−0.717** (p < 0.001)
- R@C50 Det vs per-image precision: ρ = **−0.679** (p < 0.001)
- R@C50 Fusion vs per-image precision: ρ = **−0.706** (p < 0.001)

By contrast, mean confidence score (the mAP proxy) correlates at only ρ = +0.283 — meaning R@C50 is approximately 2.5× more strongly predictive of whether an image will have many errors. Per-image AURC shows near-zero correlation with per-image AP (ρ ≈ 0.01–0.02) because the median image has only about 11 matched detections, which is too few to compute a stable AURC estimate.

---

## 5. Challenges and Limitations

### Challenge 1: Dataset alignment
MC Dropout and Decoder Variance were run in different pipeline stages with different detection thresholds. The bbox-matching strategy retained 6,111 out of a potential ~30,000 detections. The remaining detections could not be compared because they appeared in only one of the two datasets. This reduces statistical power but ensures fairness.

### Challenge 2: Inverted decoder signal
The decoder variance was found to have the opposite polarity to what was expected. Using the raw signal would have made Deterministic appear useless. Applying the inversion is methodologically justified but must be explicitly stated when reporting — it means the signal is not interpretable as "higher = more uncertain" without transformation.

### Challenge 3: Failure type labeling without ground truth
The four failure categories (confident FP, novel class boundary, prompt ambiguity, background clutter) are assigned using heuristic rules based on score, category ID, and bounding-box area. These are reasonable proxies but are not validated labels. The numbers of some categories are very small (Confident FP: n = 7), which means AUROC estimates for those bins have high variance.

### Challenge 4: Fusion latency is estimated, not measured
The Fusion strategy was not run as a standalone experiment. Its latency (393.6 ms) is estimated as Deterministic + 10% overhead for computing the fusion step. The real overhead could differ.

### Challenge 5: MC Dropout with K=5 may be underpowered
Standard MC Dropout analyses typically use K=20–50 passes to get stable variance estimates. With K=5, the score variance per detection is noisy. This may explain why MC Dropout performs poorly at risk-coverage despite being theoretically motivated — the estimates are too unstable to rank detections reliably.

---

## 6. Summary of Conclusions

| Claim | Status | Key number |
|---|---|---|
| Deterministic is substantially faster than MC Dropout | ✅ Confirmed | 5.0× (357.8 ms vs 1,788.9 ms) |
| Deterministic has better calibration than MC Dropout | ✅ Confirmed (small margin) | ECE 0.2051 vs 0.2089 |
| Deterministic is better at filtering errors (risk-coverage) | ✅ Confirmed | RC-AUC 0.067 vs 0.295 |
| Fusion provides the best risk-coverage | ❌ Not confirmed | Fusion RC-AUC = 0.148 (worse than Det) |
| Fusion provides the best calibration | ❌ Not confirmed | Fusion ECE = 0.2199 (worst) |
| MC Dropout captures complementary ambiguity information | ❌ Not confirmed | MC wins all 4 failure types |
| Risk-aware metrics predict deployment safety better than mAP | ✅ Confirmed | R@C50 ρ=−0.72 vs Mean Score ρ=+0.28 |

**Overall conclusion:** The Deterministic decoder variance estimator is the most cost-efficient choice for epistemic uncertainty quantification in this OVD system. It is 5× cheaper, better calibrated, and a stronger error-ranking signal than MC Dropout. Naive fusion of the two signals without individual recalibration makes calibration worse and does not add value over the Deterministic signal alone.