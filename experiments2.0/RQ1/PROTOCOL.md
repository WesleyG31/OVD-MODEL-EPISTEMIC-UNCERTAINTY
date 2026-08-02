# RQ1 frozen protocol

Pre-confirmatory revision 3 was frozen on 2026-07-31 before confirmatory
feature extraction or test evaluation. Its final rules are incorporated
directly in this protocol: one-sided superiority tests use the paired
sequence-cluster bootstrap centered at the zero-boundary null;
non-inferiority tests use their frozen negative margin; and uncentered
replicates are used only for percentile intervals. The deterministic
reference universe retains all 900 eligible queries, while every MC pool
contains the nominal top 300 plus all eligible reference query IDs.

The empirical scope is limited to the pinned GroundingDINO Swin-T checkpoint,
ten fixed BDD prompts, the frozen BDD100K partition, and recorded hardware.
RQ1 is a separately prespecified family, its outcome must be reported, and it
cannot rescue or be rescued by another RQ. Diagnostic MINI results are not
scientific evidence.

## Research question

> How can epistemic uncertainty extracted from multiple internal
> representations of a transformer-based open-vocabulary detector be fused
> into a reliable uncertainty signal for risk-aware ADAS perception?

## Claim boundary

The experiment distinguishes:

1. **MC epistemic proxies**: between-pass variability produced by stochastic
   depth in the trained Swin backbone and vision-language fusion blocks while
   all learned weights remain fixed.
2. **Decoder-dynamics proxies**: within-pass changes across decoder layers.

Decoder dynamics are not called epistemic uncertainty merely because they vary.
They qualify as useful epistemic proxies only if they predict held-out errors
and degrade coherently under held-out conditions.

## Data partitions

- The canonical source is Kaggle dataset
  `solesensei/solesensei_bdd100k/versions/2`.
- The original BDD100K validation labels are converted deterministically to
  COCO, including the historical aliases `motor -> motorcycle` and
  `bike -> bicycle`.
- The legacy paper split is reproduced with Python `random.Random(42)`: 80%
  development and a 20% evaluation pool. Eight evaluation images used by the
  automated mini run are frozen as a diagnostic partition and excluded by
  source group. The remaining 1,992 images form the untouched confirmatory
  test.
- Development source groups are split into fusion training (70%) and
  validation (30%). The BDD `videoName` is the group when present; because the
  detection release has one annotated keyframe per source video, the image
  stem is the documented fallback group.
- Detection-level random splitting is forbidden.

Data preparation must pass the version, checksum, image, bounding-box,
category, split-overlap and source-group audit. The RQ1 manifest refuses to run
without that audit. Model selection, feature orientation and calibration use
no confirmatory-test labels. Diagnostic labels cannot be used to change the
frozen confirmatory protocol.

Detector inference is a label-free paper-level artifact shared with compatible
RQs. Schema v1 performs one deterministic pass and the same 10 seeded DropPath
passes per image, using seed
`20260731 + image_id + (pass_index + 1) * 1009`. Atomic NPZ/JSON shards bind
the source image, model, environment, inference/association configuration and
array schema by SHA-256. Ground-truth matching, uncertainty features, fusion,
calibration and evaluation remain RQ1-specific. A future incompatible tensor
requirement must create a coexisting schema version rather than altering v1.

## Detection universe and correctness

- A deterministic pass defines the reference detection universe.
- Candidate generation uses a low score threshold (`0.05`) and retains at most
  300 detections per image. Confirmatory risk metrics use the pre-specified
  operational threshold `0.20`; thresholds `0.05`, `0.10`, `0.20` and `0.30`
  form a mandatory sensitivity analysis.
- Predictions are matched to ground truth by image and category with
  score-ordered, one-to-one IoU matching at IoU >= 0.5.
- A ground-truth object can be assigned to at most one prediction.
- False negatives are computed separately from unmatched ground-truth objects;
  rejected true positives are never presented as all false negatives.

An image-level artifact is written even when the detector returns no
detections. A secondary safety analysis labels images containing at least one
false negative at the operational threshold and evaluates the maximum
detection-level uncertainty in the scene (uncertainty 1.0 when no operational
detection exists). This does not turn a missed object into a scored detection;
it explicitly tests the limitation of a detection-conditioned signal.

## Stochastic association

Each stochastic pass is associated with the deterministic reference detections
using a one-to-one linear assignment. Association uses category agreement and
IoU, with a minimum IoU of 0.30. Missing detections remain explicitly missing
and contribute to `absence_rate`; they are never replaced with deterministic
values.

Association IoU values 0.20, 0.30 and 0.50 are repeated on a frozen
validation subset as a robustness analysis. They cannot select the
confirmatory method.

## Internal representations

- **Semantic**: predictive entropy, mutual information, class disagreement and
  confidence variance across stochastic passes.
- **Geometric**: box covariance, pairwise IoU loss, reference-point variance
  and decoder reference displacement.
- **Representation**: hidden-state covariance, cosine instability and
  decoder-layer hidden displacement.
- **Presence**: stochastic absence rate.
- **Confidence control**: `1 - deterministic confidence`.

Every signal is stored independently before fusion so that failures and
direction reversals remain visible.

The primary estimator uses 10 stochastic passes. Prefix estimates at 2, 5 and
10 passes are computed from the same stochastic sequence and evaluated as a
mandatory convergence/cost sensitivity analysis; no extra detector calls are
needed. The Swin-T configuration has zero ordinary/text/fusion dropout and
`fusion_droppath=0.1`; only non-zero `DropPath` modules are enabled. With the
pinned model this comprises 11 Swin-backbone and 6 fusion-block modules.
Runtime metadata records every enabled module and probability.

PyTorch 2.5.1 reports that the CUDA `cumsum` used by GroundingDINO positional
encoding has no guaranteed deterministic implementation. Deterministic mode is
therefore `warn_only`; seeds, cuDNN and cuBLAS settings remain fixed, and the
paper must not claim bitwise equality across different GPU architectures.
Two independent schema-v1 executions were bitwise identical on the verified
local software/GPU stack, including every stored array and complete shard hash.

## Fusion

Fusion models are fitted only on detections at the frozen operational score
threshold 0.20. Regularized logistic models are trained on fusion-training
detections. Validation source groups are deterministically divided 50:50: the
selection fold chooses regularization, the selected model is refitted on
training plus selection, and untouched calibration groups fit isotonic error
probabilities. Selection and calibration groups never overlap. Confirmatory
labels are used once for final reporting. A fixed random-forest fusion over
all internal signals is the nonlinear sensitivity comparator.

Required ablations:

- confidence only;
- individual conventional proxies: semantic mutual information, predictive
  entropy, box variance and embedding variance;
- each internal representation family;
- semantic + geometric;
- all internal signals without confidence;
- nonlinear random-forest fusion of all internal signals;
- all signals including confidence.

The paper must report if the confidence-only control is superior.

The frozen detector is reported independently using COCO bbox mAP
(IoU 0.50:0.95), AP50, AP75 and average recall on the untouched test images.
Uncertainty quality is never presented without this detector-quality context.

## Primary outcomes

1. Error-detection AUROC with sequence-clustered bootstrap confidence interval.
2. Error-detection AUPRC.
3. Area under the risk-coverage curve (lower is better), retaining the least
   uncertain detections first.
4. Brier score, log loss and ECE for calibrated error probability, each with
   a sequence-clustered interval; ECE is repeated with 10, 15 and 20 bins.
5. Risk at fixed coverage and coverage at fixed risk.

All confidence intervals are resampled by source video/image group, not by
detection.

Every fusion is also compared with confidence-only using a paired clustered
bootstrap; positive deltas always mean improvement. One-sided tests use the
paired cluster-bootstrap distribution centered at the boundary null; the
uncentered distribution is used only for percentile confidence intervals.
Confirmatory results are
stratified by BDD100K time of day, weather, scene, predicted category and
COCO-style predicted object size when a stratum has at least 200 operational
detections and both outcomes.

The frozen primary claim requires `all_internal` to improve both AUROC and
AURC over confidence in one-sided, paired sequence-clustered bootstrap tests
after Holm correction at familywise alpha 0.05. Its Brier improvement interval
must also remain above the non-inferiority boundary -0.02. This rule can only
be satisfied on the confirmatory partition. A validation-only robustness
subset additionally measures independent MC seeds, association IoU,
canonical/synonymous prompts and fixed blur/brightness corruptions. These
diagnostics delimit sensitivity but cannot rescue a failed primary claim.

The enabled backbone and fusion DropPath modules are perturbed together. The
study therefore attributes performance to signal families and the combined
stochastic-depth intervention, not causally to an individual layer or block.

## Prohibited analyses

- Oracle IoU or TP/FP labels as inference-time fusion inputs.
- Test-set normalization, direction selection, weight selection or threshold
  tuning.
- Reusing ground-truth objects for multiple true-positive predictions.
- Calling per-image precision “mAP”.
- Dropping unmatched stochastic detections from the uncertainty computation.
- Replacing failed or missing measurements with simulated values.
