# RQ2 frozen protocol

Pre-confirmatory revision 3 is defined in
[`METHODOLOGICAL_AMENDMENTS.md`](../METHODOLOGICAL_AMENDMENTS.md).

Protocol freeze date: 2026-07-31. This protocol was specified before any RQ2
confirmatory output was generated or inspected. Results under `New_RQ/` are
legacy exploratory material and are not evidence for this experiment.

## Research question and claim boundary

> How does fusing deterministic and stochastic uncertainty estimators improve
> reliability in open-vocabulary object detection?

The experiment concerns the pinned GroundingDINO Swin-T model on the frozen
BDD100K partition. A single-pass decoder-dynamics score is called a
**deterministic proxy**, not epistemic uncertainty by definition. Between-pass
variation induced only in trained non-zero stochastic-depth modules is called
an **MC epistemic proxy**. Neither is asserted to represent every source of
predictive uncertainty. “Improvement” means better held-out error ranking and
selective risk at the same frozen detector predictions; it does not mean higher
detector mAP or guaranteed ADAS safety.

## Hypotheses and variables

- H1: learned estimator-level fusion has higher error-detection AUROC than
  capacity-matched learned deterministic-only and stochastic-only estimators.
- H2: learned fusion has lower area under the risk-coverage curve (AURC) than
  both learned standalone estimators.
- H3 (secondary): fusion improves AUPRC and calibrated Brier/NLL/ECE, and its
  advantage remains directionally coherent across MC-pass, score-threshold,
  object-size, category, time-of-day, weather and scene analyses.
- H4 (secondary complementarity): deterministic and stochastic scores are not
  redundant on validation data, assessed by Spearman correlation and by the
  incremental performance of fusion. Correlation alone is not evidence of
  usefulness.

The independent variable is estimator method. The dependent variables are
AUROC, AUPRC, AURC, risk at fixed coverage, coverage at fixed risk, Brier
score, log loss, ECE and inference cost. Detection correctness is defined by
score-ordered one-to-one class-consistent matching at IoU >= 0.50.

## Frozen data and controls

RQ2 reuses the passing data audit, checkpoint, local BERT snapshot, processed
COCO annotations and group-disjoint manifests already stored directly under
`experiments2.0`. It never reads `RQ1/models`, `RQ1/outputs`, RQ1 metrics or RQ1
figures. The shared confirmatory benchmark contains 1,992 images; the eight
diagnostic images are excluded from it by source group. Train has 5,600 images
and validation has 2,400.

RQ2 consumes the neutral schema-v1 detector cache under
`data/derived/groundingdino_mc_v1`. It uses exactly the same deterministic
reference and 10-pass sequence as RQ1, with seed
`20260731 + image_id + (pass_index + 1) * 1009`, so canonical detector inference
occurs once rather than once per RQ. Each atomic NPZ/JSON shard is tied to the
source image, weights, packages, code, inference/association configuration,
manifest request and array schema by fingerprints and SHA-256. RQ2 performs
its own ground-truth matching, feature materialization, estimation and
evaluation; the shared layer contains no labels or RQ1 result.

The deterministic reference pass defines one common detection universe at
score 0.05, capped at 300 predictions per image. Operational analysis uses
score 0.20. Sensitivity thresholds are 0.05, 0.10, 0.20 and 0.30. False
negatives remain a separate detector-quality limitation and are not converted
into synthetic scored detections.

## Estimators and ablations

The deterministic family contains, from the one reference pass:

1. decoder reference-point variance across layers;
2. mean decoder reference-point displacement across adjacent layers;
3. mean cosine displacement of the matched query hidden state across layers.

The stochastic family contains, from 10 associated stochastic passes:
semantic mutual information, predictive entropy, class disagreement, score
variance, box variance, pairwise IoU loss, final hidden-state variance/cosine
instability and absence rate. Association is one-to-one with minimum IoU 0.30;
missing detections stay missing and contribute to absence rate. Only non-zero
`DropPath` modules are enabled and every enabled module is recorded.

Prespecified methods are:

- deterministic confidence uncertainty (`1 - score`);
- fixed deterministic and fixed stochastic composites, constructed by
  averaging train-empirical CDF transforms of positively oriented features;
- fixed equal fusion, the arithmetic mean of the two fixed composites;
- regularized logistic deterministic-only, stochastic-only and fused models;
- logistic fusion plus confidence, to test whether fusion adds information
  beyond ordinary confidence;
- a fixed random-forest nonlinear fusion sensitivity comparator.

Train and validation detections are first restricted to the same frozen score
0.20 population used by the primary evaluation. Validation source groups are
then split 50/50 into disjoint model-selection and probability-calibration
folds. Logistic regularization is chosen from C = 0.01, 0.1, 1 and 10 by
selection-fold AUROC, with smaller C breaking ties. All preprocessing is fit
on train only. Error-probability calibration is isotonic and fit only on the
held-out calibration groups. Feature
directions, the equal weight, feature groups and random-forest settings are
fixed here; confirmatory labels cannot alter them. MC prefix estimates at 2, 5
and 10 passes reuse the same stochastic sequence.

## Outcomes and statistical analysis

Co-primary endpoints are AUROC (higher is better) and AURC (lower is better)
for learned fusion versus each learned standalone estimator: four paired
comparisons. Source sequence/image groups, rather than detections, are sampled
in 2,000 paired bootstrap repetitions. Positive deltas always mean the fusion
is better. Null-centered paired cluster-bootstrap p-values are Holm-adjusted at familywise alpha
0.05; percentile 95% intervals are also reported. The success criterion is
that all four adjusted primary comparisons reject no-improvement and have the
prespecified favorable point-estimate direction. Mixed or negative results are
reported without retuning.

Secondary outcomes include AUPRC, calibration metrics, fixed-risk/coverage
operating points, fixed versus learned fusion, the nonlinear sensitivity and
score complementarity. Intervals for method metrics use the same clustered
bootstrap. Subgroups are descriptive and are reported only with at least 200
operational detections and both outcome classes. No subgroup is used to choose
the primary model.

Computational cost records synchronized deterministic-pass time, total MC time,
per-image aggregation time, enabled stochastic modules, GPU/CUDA/cuDNN and
package versions. Throughput is reported both for the deterministic proxy and
the full ten-pass stochastic/fused path; fusion arithmetic is not misreported
as avoiding the MC passes it consumes.

## Failure criteria and sensitivity

The run fails technically on a missing/failed audit, manifest/hash mismatch,
CPU fallback, checkpoint mismatch, invalid or non-finite required outputs,
missing image shards, one-class train/validation inputs, or fewer than the
frozen minimum validation detections. Corrupt or stale shards are recomputed
atomically. A scientific failure is a non-favorable or multiplicity-adjusted
non-significant primary comparison; this remains a publishable negative/mixed
result, not a reason to tune on test.

Mandatory sensitivities are 2/5/10 MC prefixes, the four score thresholds,
random-forest fusion, confidence augmentation, category, COCO object size and
BDD time-of-day/weather/scene. The diagnostic mini run is only a technical
test and cannot change the protocol.

## Limitations

The study uses one detector/checkpoint, one prompt vocabulary and one dataset
mirror. Stochastic depth is an approximate perturbation of a frozen model, not
a posterior sample. Decoder dynamics may reflect refinement rather than model
uncertainty. Detection-conditioned uncertainty cannot score wholly missed
objects. Isotonic calibration and model selection use disjoint validation
groups. The common RQ1/RQ2 benchmark creates a paper-level multiplicity and
selective-reporting obligation beyond the within-RQ2 Holm family. Cross-domain,
corruption and second-detector claims are out of scope.

## Prohibited analyses

- confirmatory normalization, feature orientation, threshold or model tuning;
- oracle IoU, TP/FP labels or subgroup identity as inference-time features;
- per-detection resampling, many-to-one ground-truth matching or dropping
  unmatched MC detections;
- use of diagnostic/legacy/simulated values as paper evidence;
- claiming that a fused signal improves detector mAP when predictions are
  unchanged, or that the confirmatory question is answered before the full
  real run is completed and interpreted.
