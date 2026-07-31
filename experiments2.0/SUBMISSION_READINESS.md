# Applied Soft Computing submission readiness

Assessment date: 2026-07-31.

## Current decision

**Not yet ready for submission.** The implementation is structured for a
paper, but no simulated or expected result can be used as evidence. Readiness
requires a passing data audit, a passing end-to-end smoke test, the complete
frozen run, and a human interpretation of the resulting confirmatory metrics.

**Technical FULL-run decision: ready.** The pre-confirmatory revision 3,
98-test suite, model/layer/checkpoint gates, five MINI workflows, cache reuse,
candidate audit, output isolation and confirmatory manifest checks pass. This
does not predict or guarantee favorable FULL scientific results.

## Automated gates

- [x] Standard `venv`/`pip` workflow; no Conda environment and no Docker.
- [x] Direct and transitive environment inventory recorded.
- [x] `groundingdino-py==0.4.0`, CUDA tensor operation and deformable
  attention verified on GPU.
- [x] GroundingDINO checkpoint and BERT revision fixed and SHA256 verified.
- [x] Non-zero stochastic operators identified and recorded.
- [x] Project-scoped unit tests pass.
- [x] Kaggle BDD100K version 2 data audit passes (10,000 images, 185,526
  annotations, all image SHA256 values recorded).
- [x] RQ1 two-image GPU smoke test passes; two independent extractions match
  exactly at both DataFrame and Parquet SHA256 level.
- [x] Isolated RQ1 mini end-to-end run passes on 6 train, 4 validation and 8
  test images: extraction, 16 fitted models, group-disjoint
  selection/calibration, robustness, evaluation, integrity checks and report
  generation. These diagnostic outputs are not confirmatory evidence.
- [x] RQ1 fitting is aligned with the frozen 0.20 operational threshold;
  hyperparameter selection and isotonic calibration use disjoint source
  groups.
- [x] RQ1 includes a frozen nonlinear random-forest comparator, independent MC
  seed repeats, association-IoU sensitivity, prompt sensitivity and controlled
  blur/brightness shifts on validation only.
- [x] RQ1 reports clustered intervals for Brier/NLL/ECE, ECE-bin sensitivity,
  category/object-size subgroups, a reliability diagram and synchronized
  latency/VRAM.
- [x] RQ1 freezes a Holm-corrected AUROC/AURC primary rule plus Brier
  non-inferiority; diagnostic results cannot satisfy it.
- [x] The 8 evaluation images inspected by the mini run are frozen as a
  diagnostic partition and excluded by group from the 1,992-image
  confirmatory test manifest.
- [x] RQ2 hypotheses, feature directions, estimator families, primary
  comparisons, Holm correction and failure criteria were frozen before any
  RQ2 confirmatory execution.
- [x] RQ2 two-image GPU smoke passes with exact DataFrame and Parquet SHA-256
  equality across independent model loads.
- [x] Neutral schema-v1 inference is versioned, label-free, atomic and
  hash-validated; two independent one-image GPU executions match in every
  array and complete NPZ SHA-256. RQ1/RQ2 use the same frozen MC seed sequence.
- [x] The detector fingerprint is isolated from receipts and downstream CPU
  materialization, so future feature/report work cannot trigger unnecessary
  canonical GPU recomputation; inference/schema/data/model changes still do.
- [x] Isolated RQ2 mini end-to-end passes on 6 train, 4 validation and 8
  diagnostic images: hash-valid resumable extraction, 12 primary/sensitivity
  estimators, COCO context, clustered bootstrap, multiplicity adjustment,
  finite-metric audit and report generation. These outputs are not evidence.
- [x] RQ3 protocol freezes its claim boundary, targets, eight spatial features,
  equal-capacity control, seven methods, six Holm-adjusted comparisons,
  calibration, sensitivity, subgroup and `not_estimable` rules before any RQ3
  confirmatory output exists.
- [x] The pre-RQ4 shared/RQ1/RQ2/RQ3 baseline passes 45 tests, including RQ3
  formulas, empty/single/absent cases, matching, hashes, schemas, fingerprints,
  incompatibility, leakage, models, metrics, corruption and future contracts.
- [x] RQ3 two-image GPU smoke passes across distinct diagnostic namespaces
  with exact DataFrame and Parquet SHA-256 equality.
- [x] Isolated RQ3 mini end-to-end passes on 6/4/8 images, fits seven primary
  methods plus 2/5/10 sensitivities, calibrates, bootstraps, re-ranks COCO,
  generates reports and marks all output diagnostic-only.
- [x] RQ3 mini reuses 6/4/8 canonical shared shards with computed=0, validates
  immutable request receipts, exactly aligns 1,183/822/1,521 common rows with
  both RQ1 and RQ2, and reuses all 18 RQ3 feature shards on a second run.
- [x] RQ4 protocol freezes the within-BDD domain definition, claim boundary,
  three component targets, feature groups, ten methods, capacity control,
  six Holm-adjusted primary comparisons and `not_estimable` rules before any
  RQ4 confirmatory output exists.
- [x] The combined shared/RQ1/RQ2/RQ3/RQ4 suite passes 62 tests, including RQ4
  formulas, empty/single/absent cases, domain/target leakage, group splitting,
  models, metrics, manifests, fingerprints, corruption and schema extension.
- [x] RQ4 two-image GPU smoke passes across distinct namespaces with exact
  DataFrame and Parquet SHA-256 equality.
- [x] Isolated RQ4 mini passes on 6/4/8 images, fits ten primary methods plus
  2/5/10 sensitivities, calibrates, evaluates frozen shifts, bootstraps,
  generates CSV/PNG/PDF reports and marks all outputs diagnostic-only.
- [x] RQ4 mini reuses 6/4/8 canonical shards with computed=0, exactly aligns
  1,183/822/1,521 common rows with RQ1, RQ2 and RQ3, validates all report
  hashes and reuses all 18 RQ4 feature shards on a second run.
- [x] RQ5 protocol freezes the claim boundary, `accept`/`defer` semantics,
  criticality formula, validation folds, methods, capacity controls, MC
  prefixes, six-test Holm family, Brier interval and offline latency frontier before any RQ5
  confirmatory output exists.
- [x] The combined shared/RQ1/RQ2/RQ3/RQ4/RQ5 suite passes 98 tests, including
  RQ5 formulas, empty/single/absent/missing cases, group splitting, leakage,
  models, weighted metrics, hashes, corruption and future schema contracts.
- [x] RQ5 two-image GPU smoke passes across distinct diagnostic namespaces
  with exact DataFrame and Parquet SHA-256 equality.
- [x] Isolated RQ5 mini passes on 6/4/8 images, fits seven frozen methods plus
  mc05/mc10 sensitivities, calibrates, evaluates, bootstraps and generates
  hash-validated CSV/PNG/PDF reports marked diagnostic-only.
- [x] RQ5 mini reuses 6/4/8 canonical shards with computed=0, validates
  immutable receipts, exactly aligns 1,183/822/1,521 common rows with RQ1--RQ3,
  reports the current RQ4 train/validation artifacts as a different image
  universe, and reuses all 18 RQ5 feature shards on a second run.
- [x] RQ5 preserves the negative mini outcome without tuning. The decision
  overhead gate passed (~1.21 ms p95); mc02 misses every reported deployment
  budget and remains an offline feasibility result. This is not evidence.
- [x] The deterministic reference universe is uncapped within all 900 model
  queries; MINI observed zero reference truncation. MC retains top-300 plus
  every eligible reference query and records all pre/post-cap counts.
- [x] Top-level Windows and Linux orchestration includes all five implemented
  questions, RQ1--RQ5.
- [x] The refactored mini end-to-end run confirms shared-cache reuse on 6/4/8
  images, immutable request receipts, exact RQ1/RQ2 identities and common MC
  features, and zero additional detector shards for RQ2 after RQ1.
- [x] Shared feature materialization was reduced from 2.58 to 0.54 seconds per
  image for RQ1 and from 2.10 to 0.30 seconds per image for RQ2 in the final
  eight-image diagnostic timing, without changing scientific outputs.
- [ ] Complete shared train/validation/test inference and all RQ-specific
  materializations finish without missing shards.
- [ ] RQ1--RQ5 confirmatory fitting, evaluation and report generation
  finish.
- [ ] Every generated table/figure is regenerated from real artifacts.

## Frozen RQ1 evidence

The run must report:

- detector COCO mAP, AP50, AP75 and AR;
- confidence-only and individual uncertainty baselines;
- semantic, geometric, representation and presence ablations;
- internal-only and confidence-augmented fusion;
- paired sequence-clustered improvement intervals versus confidence;
- AUROC, AUPRC, AURC, Brier, NLL, ECE, risk at coverage and coverage at risk;
- 2/5/10 MC-pass sensitivity;
- score-threshold sensitivity;
- time-of-day, weather and scene strata;
- secondary image-level false-negative safety results;
- elapsed time, package/CUDA/cuDNN/GPU metadata and all data/model hashes;
- sequence-disjoint selection/calibration inventories and model/report hashes;
- nonlinear fusion, feature complementarity and category/object-size strata;
- independent seeds, association IoU, prompts and corruption robustness;
- synchronized warm-model throughput and peak GPU memory.

## Frozen RQ2 evidence

The RQ2 run must report:

- confidence, fixed deterministic, fixed stochastic and equal fixed fusion;
- capacity-matched learned deterministic-only, stochastic-only and fused
  estimators, plus confidence augmentation and nonlinear sensitivity;
- AUROC, AUPRC, AURC, Brier, NLL, ECE and fixed risk/coverage outcomes;
- four paired sequence-clustered fusion-versus-standalone primary comparisons
  with Holm-adjusted one-sided p-values and percentile intervals;
- 2/5/10 MC-pass and score-threshold sensitivity;
- category, predicted object-size, time-of-day, weather and scene subgroups;
- validation-only deterministic/stochastic complementarity;
- synchronized single-pass and ten-pass MC runtime, enabled stochastic modules
  and environment/model/data/source/artifact hashes.

RQ2 diagnostic estimates must not be copied into the manuscript. A mixed or
negative confirmatory result remains valid and must not trigger test retuning.

## Frozen RQ3 evidence

The RQ3 confirmatory run must report:

- confidence, fixed spatial agreement, learned spatial quality, equal fusion,
  product fusion, capacity-matched non-spatial product and direct spatial
  fusion;
- AUROC, AUPRC, AURC, Brier, NLL, ECE and fixed risk/coverage outcomes;
- the six product-versus-confidence/capacity-control AUROC/AURC/Brier paired
  comparisons with Holm-adjusted one-sided p-values;
- COCO AP50/AP75 and AP/AR context under each fixed-candidate ranking score;
- 2/5/10 MC prefix, score-threshold, localization-IoU and ECE-bin sensitivity;
- localization/detection error taxonomy and time/weather/scene/category/size
  subgroups;
- group-disjoint model-selection/calibration inventories, runtime/VRAM,
  neutral-cache reuse and all data/model/source/artifact hashes.

RQ3 mini estimates are diagnostics only. They do not satisfy the frozen
success rule and must not be copied into the manuscript or used for tuning.

## Frozen RQ4 evidence

The RQ4 confirmatory run must report:

- raw and calibrated confidence; each class/localization/uncertainty level;
  all pairwise products; multilevel; and the flat same-feature control;
- Brier, NLL, ECE/MCE, AUROC, AUPRC, AURC and fixed risk/coverage outcomes on
  the frozen shifted subset;
- six multilevel-versus-confidence/flat Brier/NLL/AURC paired comparisons with
  Holm-adjusted one-sided p-values and sequence-bootstrap intervals;
- reference, shifted-axis, severity and observed-value strata, plus
  category/object-size subgroups and domain reliability gaps;
- coefficient capacity, 2/5/10 MC-prefix, score-threshold and ECE-bin
  sensitivities;
- group-disjoint selection/calibration inventories, runtime/VRAM, neutral
  cache reuse and all data/model/source/artifact/report hashes.

RQ4 mini estimates are diagnostic only. Its natural, overlapping BDD strata
must not be described as an external domain benchmark, and mixed/negative
confirmatory outcomes must not trigger retuning.

## Frozen RQ5 evidence

The RQ5 confirmatory run must report:

- raw/calibrated confidence, uncertainty-only, criticality-only, unweighted
  late fusion, risk-aware fusion and the flat same-feature capacity control;
- weighted AURC, coverage at weighted risk 0.10, ordinary AUROC/AUPRC/AURC,
  Brier/NLL/ECE/MCE and fixed operating coverage/risk;
- four risk-aware-versus-confidence/flat superiority comparisons plus two
  Brier non-inferiority comparisons in one six-test Holm family, using
  null-centered paired sequence bootstrap tests and percentile intervals;
- Brier non-inferiority lower confidence bounds and decision p95 <= 5 ms as
  gates; mc02/mc05 estimated and mc10 measured end-to-end latency remain an
  explicitly offline feasibility frontier rather than a 10 Hz claim;
- score, 2/5/10 MC prefix, criticality-coefficient,
  category/size/criticality/time/weather/scene sensitivities, coefficient
  capacity, GPU memory, cache reuse and hashes;
- deferred criticality/error composition while stating that fallback outcomes
  and false negatives are not evaluated.

RQ5 mini estimates are diagnostic only. Negative quality or latency-feasibility
results are retained without retuning and must not be copied into the paper.

## Scientific limitations to resolve or delimit

1. **External validity.** Controlled blur/brightness shifts are now included,
   but one Kaggle mirror, one BDD100K partition and one GroundingDINO Swin-T
   checkpoint still cannot support a universal claim about open-vocabulary
   detectors. Add a second detector/backbone or explicitly restrict the claim
   to this setting.
2. **Robustness sample.** Independent MC seeds, association IoU, prompt and
   corruption tests are implemented on a frozen validation subset; their full
   outputs still need to complete and must remain diagnostic rather than tune
   the confirmatory test.
3. **Missed objects.** The primary signal is detection-conditioned and cannot
   assign uncertainty to an object that produced no detection. The image-level
   false-negative analysis must be presented as a limitation, not as
   object-level uncertainty.
4. **Dataset identity and rights.** `solesensei/solesensei_bdd100k` is a
   third-party Kaggle mirror whose license is listed as “Other.” Confirm that
   its contents match the authorized BDD100K release and that use,
   redistribution and reviewer access comply with the original license.
5. **Image-level diagnostic endpoint.** All eight diagnostic test images had
   at least one operational false negative, so image-level AUROC/AUPRC were
   correctly marked `not_estimable`. The confirmatory run must establish
   whether both image-level outcome classes exist; otherwise this secondary
   endpoint must be reported as non-estimable rather than replaced or tuned.

## Journal-specific actions

The current Applied Soft Computing guide places machine/deep learning and
vision/pattern recognition within scope and describes technical papers as
normally 20–30 pages (maximum 50). Before submission:

- use an editable Word or LaTeX source and the journal template;
- prepare the separate graphical abstract requested by the guide;
- archive code/configuration manifests with a persistent software identifier
  such as a Zenodo DOI and cite both software and dataset versions;
- include a research-data availability statement;
- disclose any generative-AI assistance according to Elsevier policy;
- provide competing-interest, funding, author-contribution and permission
  statements;
- do not upload generated, simulated or selectively chosen confirmatory
  results.

Journal guide:
https://www.sciencedirect.com/journal/applied-soft-computing/publish/guide-for-authors
