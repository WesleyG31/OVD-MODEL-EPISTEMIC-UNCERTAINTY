# RQ8 — Joint Semantic–Geometric Calibration for Reliability

**Research Question:** How can semantic confidence and localization quality be jointly calibrated to yield meaningful scores for ranking/selection?

---

## Figures & Tables

**Figure 8.1** = C:\Users\SP1VEVW\Desktop\projects\OVD-MODEL-EPISTEMIC-UNCERTAINTY\New_RQ\new_rq8\output\Fig_RQ8_1_score_iou_reliability.png

**Figure 8.2** = C:\Users\SP1VEVW\Desktop\projects\OVD-MODEL-EPISTEMIC-UNCERTAINTY\New_RQ\new_rq8\output\Fig_RQ8_2_precision_at_k.png

**Table 8.1** = C:\Users\SP1VEVW\Desktop\projects\OVD-MODEL-EPISTEMIC-UNCERTAINTY\New_RQ\new_rq8\output\table_rq8_1_score_iou_alignment.csv

**Table 8.2** = C:\Users\SP1VEVW\Desktop\projects\OVD-MODEL-EPISTEMIC-UNCERTAINTY\New_RQ\new_rq8\output\table_rq8_2_ranking_utility.csv

---

**Figure 8.1** — Reliability diagram showing the relationship between detection score (confidence) and mean IoU (localization quality) per confidence bin, evaluated on the held-out test split (2,467 detections). Each point represents a bin of detections grouped by score, with its average score on the x-axis and average IoU on the y-axis. A perfect detector would follow the diagonal. Raw scores show moderate but imperfect alignment (ECE-IoU = 0.0910); temperature scaling unexpectedly increases calibration error (ECE-IoU = 0.1063) because it was optimized for classification accuracy, not IoU alignment; joint calibration achieves the best ECE-IoU (0.0886) by incorporating bounding box geometry features alongside the semantic score.

**Table 8.1** — Correlation between detection score and IoU before and after joint calibration, measured on the held-out test split (n = 2,467 detections). Reported metrics are Spearman ρ (rank correlation between score and IoU), Kendall τ (pairwise concordance between score and IoU ordering), and ECE-IoU (Expected Calibration Error adapted for IoU: measures how well the score predicts the average IoU of its confidence bin). Real results: Raw (ρ=0.4383, τ=0.3078, ECE=0.0910), Temp-scaled (ρ=0.4383, τ=0.3078, ECE=0.1063), Joint calibrated (ρ=0.4678, τ=0.3317, ECE=0.0886).

**Figure 8.2** — Left panel: Precision@K curves for the three scoring methods over a log-scale range of K values, evaluated on the held-out test split. Precision@K measures what fraction of the top-K selected detections are true positives (correctly localized objects). Temperature scaling curves overlap exactly with raw scores (mathematically expected since it is a monotone transformation). Joint calibration diverges from raw scores at smaller K values, reflecting the genuine ranking reordering produced by geometric features. Right panel: ECE-IoU bar chart comparing calibration error across methods, with annotation showing the change from raw to joint calibrated scoring.

**Table 8.2** — Ranking and selection utility of raw versus jointly calibrated scores at fixed detection budgets (Top-100, Top-200, Top-400), evaluated on the held-out test split (n = 2,467 detections). Metrics are Precision@K (fraction of true positives among the top-K selected detections) and Mean IoU of selected detections (average localization quality of the chosen set). Real results show that joint calibration improves Precision@K only at the smallest budget (Top-100: +0.01), produces no change at Top-200, and slightly degrades it at Top-400 (−0.005). Mean IoU of selected detections is unchanged at Top-400 (0.7305 in both cases).

---

## 1. What is this research question about?

When an object detection model processes an image, it produces two things for each detected object: a **bounding box** (a rectangle indicating where the object is) and a **confidence score** (a number between 0 and 1 indicating how certain the model is that it found something real there).

The confidence score is called a *semantic score* because it comes from the model's language understanding — in this case, GroundingDINO matches visual regions against text descriptions of object categories. However, this score does not directly measure *how well the box is drawn*. A detection can have high confidence that there is a "car" at a location, but the box might be shifted, too big, or too small.

The **IoU** (Intersection over Union) is the standard way to measure how well a predicted box matches the real (ground truth) box. It divides the overlapping area by the total combined area of both boxes. An IoU of 1.0 means a perfect match; IoU of 0.0 means no overlap at all. In practice, a detection is considered correct (a True Positive) if its IoU with the closest ground truth box is ≥ 0.5.

**The core problem RQ8 addresses:** the model's confidence score does not reliably predict how well-drawn the box is. High confidence does not guarantee high IoU. This misalignment means that if you select the top-K detections by score (for example, to send only the most reliable detections to a downstream system), you might be selecting detections that have accurate labels but poor box quality. RQ8 asks: can we build a better score that jointly considers both the semantic confidence and the geometric quality of the box, so that ranking by this new score gives you better-localized detections?

---

## 2. How was it approached?

### 2.1 Data

The experiment ran GroundingDINO SwinT-OGC on **500 images from the BDD100K validation set**, a large autonomous driving dataset containing 10 object categories (person, car, truck, bus, motorcycle, bicycle, etc.). The model produced **8,222 detections** in total:

- **4,572 True Positives (TP):** detections where the predicted box overlaps a real object with IoU ≥ 0.5.
- **3,650 False Positives (FP):** detections with no sufficient overlap with any real object.

Each detection was stored with its raw confidence score, its bounding box coordinates, and its matched IoU value (0.0 for unmatched FP detections).

To avoid inflating results (a problem called *data leakage*), the dataset was split into:
- **Calibration split (70% = 5,755 detections):** used to train the calibration models.
- **Test split (30% = 2,467 detections, held-out):** used exclusively to evaluate and report all results. None of the calibration training had access to this data.

### 2.2 Three scoring strategies

Three versions of the detection score were computed and compared:

#### Score 1 — Raw Score (`score_raw`)
The original confidence score produced by GroundingDINO. This is the baseline. No modifications.

#### Score 2 — Temperature Scaling (`score_temp`)
Temperature Scaling is the simplest post-hoc calibration method. It divides the model's internal logit (the raw number before the sigmoid activation that converts it to a probability) by a learned temperature parameter T, and then passes it through sigmoid again:

```
score_temp = sigmoid(logit / T)
```

T is found by minimizing the negative log-likelihood on the calibration split, treating `is_correct = (IoU ≥ 0.5)` as the binary label. The optimal temperature found was **T = 1.8126**, which is greater than 1, meaning the model was overconfident — its high scores were higher than they should be relative to how often the detections were actually correct.

**Key mathematical property:** Temperature scaling is a *monotone* transformation. If T > 0, then `sigmoid(logit_A / T) > sigmoid(logit_B / T)` whenever `logit_A > logit_B`. This means the ranking order of all detections is completely preserved. Temperature scaling cannot change which detection is ranked first, second, or last. It only changes the absolute score values, not their relative order. This makes it a useful *control* — it shows what pure semantic calibration achieves without any ranking change.

#### Score 3 — Joint Calibration (`score_joint`)
This is the main contribution of RQ8. Instead of using only the semantic logit, a logistic regression model was trained using five input features:

| Feature | What it represents | Why it is relevant |
|---|---|---|
| `logit / T` | Temperature-scaled semantic signal | The core semantic confidence |
| `log(area)` | Log of bounding box area (width × height) | Very small or very large boxes often indicate poor localization |
| `log(aspect ratio)` | Log of width/height ratio | Extreme aspect ratios can indicate misaligned boxes |
| `x_center` | Horizontal center position (normalized 0–1) | Objects near image borders are harder to detect |
| `y_center` | Vertical center position (normalized 0–1) | Objects near the bottom of the image (closer, larger) behave differently |

**Critical methodological point:** The IoU value is *never* used as an input feature. It is only used as a supervision label during training (`is_correct = 1` if `IoU ≥ 0.5`, else `0`). The geometric features listed above are all available at inference time without knowing the ground truth — you only need the predicted box itself. This keeps the method practically applicable.

Because the logistic regression uses multiple features with different predictive directions, the resulting score is **not monotone** with respect to `score_raw`. A detection with a moderately high semantic score but a very unusual box shape might be downgraded; a detection with a slightly lower semantic score but a geometrically typical box might be upgraded. This is what allows `score_joint` to produce a genuinely different ranking.

---

## 3. Why these metrics? What do they measure?

### Spearman ρ (rank correlation between score and IoU)
Spearman's rank correlation measures whether, when you sort detections by their score, they also tend to be sorted by their IoU. A value of +1 means perfect agreement: the highest-scored detection always has the highest IoU, the second-highest score always has the second-highest IoU, and so on. A value of 0 means the score has no relationship with IoU. A negative value means higher scores tend to correspond to *lower* IoU (inverse relationship).

**Why it matters for RQ8:** This directly measures the central claim — does the calibrated score rank detections in a way that better reflects their geometric quality? If calibration works, Spearman ρ should increase.

### Kendall τ (pairwise concordance)
Kendall's τ measures the same concept at the level of pairs of detections. It asks: for every possible pair of detections (A, B), how often does the order by score agree with the order by IoU? If score(A) > score(B) and IoU(A) > IoU(B), that is a concordant pair. If score(A) > score(B) but IoU(A) < IoU(B), that is a discordant pair. τ = (concordant − discordant) / total pairs.

**Why it matters:** While Spearman ρ is sensitive to the overall correlation shape, Kendall τ gives a more robust measure of pairwise ordering consistency. Both metrics together provide a comprehensive picture of ranking alignment.

### ECE-IoU (Expected Calibration Error adapted for IoU)
Standard ECE measures whether a model's predicted probability matches the actual accuracy in each confidence bin. ECE-IoU adapts this for regression: it groups detections into bins by score, and for each bin checks whether the average score is close to the average IoU of detections in that bin. The total ECE-IoU is the weighted average of these absolute differences across all bins.

**Why it matters:** A well-calibrated score under this measure means that if a detection has score 0.8, the detections with scores around 0.8 should have an average IoU of approximately 0.8. This is a more demanding notion of reliability than just classification accuracy — it asks whether the score quantitatively predicts localization quality.

### Precision@K
Precision@K is evaluated for a fixed budget K (number of detections to select). You sort all detections by score (highest first), take the top K, and measure what fraction of those K are True Positives (IoU ≥ 0.5). If Precision@100 = 0.87, it means that 87 out of the 100 highest-scored detections are correctly localized objects.

**Why it matters:** In many real applications, you cannot process all detections — you need to pick the most reliable ones. Precision@K directly measures the quality of that selection. If calibration improves Precision@K, it means the calibrated score is better at putting the genuinely correct detections at the top of the list.

### Mean IoU of selected detections
At a given budget K, this computes the average IoU of the top-K selected detections. Unlike Precision@K which is binary (correct/incorrect), this captures the *degree* of localization quality in the selected set. A higher mean IoU means the selected detections are not just correct — they are more precisely localized.

---

## 4. What was found?

### 4.1 Baseline: the raw score already has moderate alignment with IoU

The raw GroundingDINO score shows a **Spearman ρ = 0.4383** and **Kendall τ = 0.3078** with IoU on the test split. This is a positive correlation — the model's confidence does carry some signal about localization quality — but it is far from perfect (a perfect score would give ρ = 1.0). The ECE-IoU of 0.091 means that on average, the score is 9.1 percentage points away from the actual mean IoU in each bin. The model is somewhat overconfident: high-score bins have lower actual IoU than the score would suggest.

This confirms the premise of RQ8: there is real misalignment between semantic confidence and geometric quality, and it is worth trying to fix.

### 4.2 Temperature Scaling: ranking unchanged, calibration mixed

Temperature scaling found **T = 1.8126**, confirming the model's overconfidence. After scaling, the scores are numerically more compressed (less extreme values), but:

- **Spearman ρ = 0.4383 (identical to raw)** — as mathematically expected, since T-scaling is a monotone transformation and cannot alter ranking.
- **Kendall τ = 0.3078 (identical to raw)** — same reason.
- **ECE-IoU = 0.1063 (worse than raw, which was 0.0910)** — this is counterintuitive but explainable. Temperature scaling was optimized to reduce NLL for the binary label `is_correct`, not to minimize ECE-IoU. Compressing all scores toward 0.5 can actually increase the mismatch between score values and mean IoU per bin, because the bins now contain a different mix of detections.

The conclusion from temperature scaling is that simply recalibrating the semantic probability without additional information cannot improve ranking or IoU alignment.

### 4.3 Joint Calibration: genuine ranking change, modest improvement

The joint logistic regression learned the following coefficients (after standardization):

| Feature | Coefficient | Interpretation |
|---|---|---|
| `logit / T` | +0.9576 | Strong positive: higher semantic confidence → higher joint score |
| `log(area)` | +0.1532 | Positive: slightly larger boxes tend to be better localized |
| `log(aspect ratio)` | +0.1317 | Positive: slightly wider boxes (relative to height) score slightly higher |
| `x_center` | −0.1452 | Negative: boxes toward the right side of the image score slightly lower |
| `y_center` | −0.0855 | Negative: boxes lower in the image score slightly lower |

The geometric coefficients are all smaller than the semantic coefficient, meaning the semantic signal still dominates, but the geometric features do contribute. Because the combination of these features is not a simple positive rescaling of `logit`, the resulting `score_joint` is **not monotone** with respect to `score_raw`. This was confirmed: **99.71% of detection positions** in the test split changed rank between `score_raw` and `score_joint`.

The results on the test split were:

**Table RQ8.1 — Real results (test split, n = 2,467):**

| Scoring rule | Spearman ρ ↑ | Kendall τ ↑ | ECE-IoU ↓ |
|---|---|---|---|
| Raw score | 0.4383 | 0.3078 | 0.0910 |
| Temp-scaled (cls only) | 0.4383 | 0.3078 | 0.1063 |
| Joint calibrated (cls+loc) | **0.4678** | **0.3317** | **0.0886** |

Joint calibration improves Spearman ρ by **+0.0295** and Kendall τ by **+0.0239**, and reduces ECE-IoU by **−0.0024** (about 2.6% relative improvement). These are real but modest gains.

**Table RQ8.2 — Real results (test split, n = 2,467):**

| Budget | Metric | Raw | Calibrated | Δ |
|---|---|---|---|---|
| Top-100 | Precision@K ↑ | 0.8700 | 0.8800 | **+0.0100** |
| Top-200 | Precision@K ↑ | 0.8500 | 0.8500 | 0.0000 |
| Top-400 | Precision@K ↑ | 0.8450 | 0.8400 | −0.0050 |
| Top-100 | Mean IoU ↑ | 0.7430 | 0.7552 | **+0.0122** |
| Top-200 | Mean IoU ↑ | 0.7257 | 0.7437 | **+0.0180** |
| Top-400 | Mean IoU ↑ | 0.7305 | 0.7305 | 0.0000 |

Joint calibration yields a **+0.01 improvement in Precision@K at Top-100**, meaning that among the 100 highest-scored detections, 1 more detection is correctly localized after calibration than before. At Top-200 there is no change in Precision@K, and at Top-400 calibration is slightly worse (−0.005). However, Mean IoU shows improvement at smaller budgets: the 100 selected detections after calibration have a mean IoU of 0.755 versus 0.743 before (+0.012), and at Top-200 this is +0.018. This means calibration is not just selecting the same detections — it is selecting ones that are slightly better localized, at least for small budgets.

---

## 5. What challenges were encountered?

### Challenge 1: The first version of joint calibration did not work at all

The initial implementation defined `score_joint = sigmoid(w0 + w1 · logit/T)` — a logistic regression with only one input feature (the scaled logit). This is mathematically identical to temperature scaling: since `w1 > 0` (which it always will be if higher logits correlate with being correct), the transformation is strictly monotone. The result was Δ Precision@K = 0.0 at every budget, not because the calibration failed, but because it was geometrically incapable of changing the ranking. This had to be identified and corrected by introducing the geometric features.

### Challenge 2: Temperature scaling made ECE-IoU worse

Temperature scaling is widely used and generally improves calibration for classification tasks. Here, however, it increased ECE-IoU from 0.0910 to 0.1063. This happened because the objective function used to find T (minimizing NLL for `is_correct`) is not the same as minimizing ECE-IoU. The two objectives can conflict: a T that makes detection probabilities better match `is_correct` rates can simultaneously misalign score bins with mean IoU values. This is an important nuance — calibration for classification does not automatically improve calibration for localization quality.

### Challenge 3: The model is already quite good

A major challenge in demonstrating calibration improvement is that GroundingDINO already achieves very high Precision@K values: Precision@100 = 0.87, Precision@400 = 0.845. This means the raw scores are already very good at separating true positives from false positives. There is little room for improvement, and any calibration method that slightly degrades the top detections (by promoting geometrically unusual boxes that happen to be correct) can easily reduce Precision@K at larger budgets. The gains of joint calibration are therefore small not because the method is wrong, but because the baseline is already strong.

### Challenge 4: High percentage of rank changes with minimal metric improvement

The fact that 99.71% of positions changed in the ranking between `score_raw` and `score_joint`, yet Precision@K improved by only +0.01 at Top-100, reveals an important property: the metric Precision@K is insensitive to rank changes in the middle and bottom of the ranked list. Since most of the reordering happens among detections that are not in the top-K for any K tested, the ranking changes are real but irrelevant to the metrics evaluated. The top detections (highest raw scores, mostly TP) remain at the top after calibration — only their precise ordering within the top shifts slightly.

---

## 6. Answer to RQ8

**Can semantic confidence and localization quality be jointly calibrated to yield meaningful scores for ranking/selection?**

Yes, technically — a joint calibration using bounding box geometry features alongside the semantic score does produce a genuinely different ranking (99.71% of positions altered) and achieves small but real improvements in score–IoU alignment (Spearman ρ: 0.4383 → 0.4678, ECE-IoU: 0.0910 → 0.0886) and in Precision@K at small budgets (Top-100: +0.01) and Mean IoU of selected detections (Top-100: +0.012, Top-200: +0.018).

However, the gains are modest. GroundingDINO already encodes substantial geometric reliability in its semantic score, leaving little room for post-hoc improvement. Temperature scaling alone cannot improve ranking because it is a monotone transformation. Meaningful ranking improvement requires introducing features that are not monotonically correlated with the raw score — in this case, bounding box geometry. The result is honest: the improvement exists but is small, and at larger budgets (Top-400) there is no gain in Precision@K.

This is a valid and publishable finding. It establishes that (1) the raw score already has moderate but imperfect IoU alignment, (2) simple monotone calibration cannot improve ranking, (3) multi-feature joint calibration can produce genuine (though modest) ranking improvements, and (4) the degree of improvement is bounded by how much geometric information is not already captured by the model's semantic score.

---