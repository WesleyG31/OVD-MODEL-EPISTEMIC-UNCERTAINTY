# RQ5 frozen protocol / Protocolo congelado de RQ5

Pre-confirmatory revision 3 / Revisión preconfirmatoria 3:
[`METHODOLOGICAL_AMENDMENTS.md`](../METHODOLOGICAL_AMENDMENTS.md).

Protocol status: **frozen before implementation and before any confirmatory
RQ5 result was inspected (2026-07-31)**.

## English

### Research question and claim boundary

> How can fused uncertainty and calibrated detection outputs be integrated
> into a decision-level fusion layer to achieve risk-aware selective
> perception under real-time ADAS constraints?

RQ5 studies a detection-conditioned `accept`/`defer` decision for the fixed
GroundingDINO candidates. `Defer` means route the candidate to a hypothetical
safe fallback, temporal tracker, or additional sensor; it does not mean that
the object is absent and it is never counted as a demonstrated safe action.
RQ5 does not alter boxes/classes, recover missed objects, implement a fallback,
fuse sensors, or establish vehicle safety. The claim is limited to the pinned
GroundingDINO Swin-T, prompt vocabulary, BDD100K split, and recorded hardware.

The real-time analysis has two evidence levels. Ten-pass latency is measured
from synchronized canonical extraction. Two/five-pass latency is an explicitly
labelled linear prefix estimate from the recorded ten-pass stochastic block;
decision-layer CPU overhead is measured directly. Therefore RQ5 can establish
an offline latency/quality frontier, not deployed deadline compliance.

### Frozen data flow and leakage controls

- Train (5,600 images) fits uncertainty and capacity-control models.
- Validation (2,400 images) is split deterministically by `sequence_id` into
  model selection (40%), component calibration (30%), and operating-policy
  calibration (30%). No source group crosses these folds.
- Diagnostic test (8 images) is only for smoke/mini engineering checks.
- Confirmatory test (1,992 images) is evaluated once after authorization.

The diagnostic and confirmatory namespaces, manifests, predictions, models,
and reports are isolated. Confirmatory labels may not select features,
regularization, calibration, fusion weights, thresholds, risk targets,
hypotheses, or subgroups. RQ5 never reads RQ1--RQ4 models or outputs.

RQ5 consumes `groundingdino_mc_v1` after
`validate_consumer_compatibility(config, "rq5")` and obtains its identity from
`shared_identity(config)`. Schema v1 already contains every required tensor;
it must not be extended or reinterpreted. The detector produces one
deterministic and ten canonical DropPath passes per image. The operational MC
prefix is 2; prefixes 5 and 10 are mandatory sensitivities and are sliced from
the same immutable arrays without new GPU inference.

### Outcome, features, and criticality

The binary outcome is the frozen class-aware detection error at score >= 0.20
and IoU >= 0.50. It is a label only. False negatives remain in image summaries
and are reported as a limitation; they are not converted into detection rows.

The confidence component uses `1 - score`. The epistemic component uses the
canonical semantic, geometric, and representation reductions:

- mutual information, predictive entropy, class disagreement, score variance;
- reference-IoU mean/std, pairwise-IoU loss, box variance, absence rate;
- embedding variance, cosine instability, deterministic reference/hidden
  trajectory statistics, bounding-box area fraction, and mean MC score.

Missing MC observations remain explicit. Train-learned median imputation and
missingness indicators are allowed; infinities are forbidden. Targets,
matched IoU/indices, false-negative counts, sequence IDs, test statistics, and
oracle/domain labels are prohibited model inputs.

Risk criticality is label-free and frozen from predicted class and geometry.
Class severity is 2.0 for person/rider/motorcycle/bicycle, 1.5 for
truck/bus/train, 1.25 for car, and 1.0 for traffic light/sign. Geometry is
`1 + 0.5 * bottomness + 0.5 * centrality`, where normalized bottomness and
centrality lie in [0,1]. The final weight is their product (range [1,4]). It is
used for decisions and weighted risk, never as an error-model feature.

### Methods, baselines, ablations, and capacity controls

All learned binary models are regularized logistic pipelines with median
imputation, missing indicators, standardization, deterministic seeds, and
`C in {0.01, 0.1, 1, 10}`. C is selected on the validation selection fold by
AUROC, with smaller C breaking ties. Isotonic mappings are fit only on the
component-calibration fold. The final late-fusion mapping and operating
threshold are fit only on the policy-calibration fold.

- `raw_confidence`: uncalibrated `1 - score`.
- `calibrated_confidence`: calibrated detector confidence.
- `uncertainty_only`: calibrated logistic fusion of epistemic features.
- `criticality_only`: criticity ranking without an error estimate.
- `late_fusion_unweighted`: equal 0.5/0.5 late fusion of calibrated confidence
  and epistemic error probabilities, without criticality weighting.
- `risk_aware_fusion` (primary): final-calibrated equal late fusion multiplied
  by the frozen criticality weight for the decision score.
- `flat_joint`: one calibrated logistic model over confidence plus the same
  epistemic raw features, multiplied by the same criticality weight. It is the
  feature-access/capacity control.
- `risk_aware_fusion_mc05` and `_mc10`: mandatory MC-prefix sensitivities;
  mc02 is the primary method.

The equal late-fusion weight is fixed, not selected. Every risk-aware error
baseline receives the same criticality multiplier. This separates gains due
to uncertainty fusion from ordinary confidence, extra feature access,
calibration, criticality weighting, and additional MC passes.

For each method, the policy fold chooses the largest acceptance threshold
whose criticality-weighted selective error risk is <= 0.10. If no non-empty
set is feasible, coverage is zero and all candidates are deferred. This rule
is frozen before test.

### Metrics, hypotheses, and success rule

Primary quality metrics are criticality-mass-weighted AURC and maximum
criticality-mass coverage at weighted risk 0.10. Secondary metrics include
ordinary count coverage, ordinary AUROC/AUPRC/AURC,
Brier, NLL, ECE (10/15/20 bins), weighted risk at 0.5/0.7/0.8/0.9/1.0
coverage, coverage at weighted risk 0.05/0.10/0.20, operating coverage/risk,
defer rate, and error/criticality composition of accepted/deferred candidates.

- H1: `risk_aware_fusion` lowers weighted AURC versus
  `calibrated_confidence` and `flat_joint`.
- H2: it increases coverage at weighted risk 0.10 versus both baselines.
- H3 (gate): its Brier score is no worse than either baseline by more than the
  absolute non-inferiority margin 0.01.
- H4 (systems gate): measured p95 decision overhead is <= 5 ms/image. The
  mc02 end-to-end p95 versus 33.3/50/100 ms budgets is reported as an offline
  feasibility result and is not a deployed 10 Hz claim.

The four H1--H2 comparisons and two H3 non-inferiority comparisons form one
six-test family. A paired bootstrap resamples `sequence_id` groups 2,000 times.
Improvements are oriented positive; the distribution is centered at zero for
superiority and at -0.01 for non-inferiority before one-sided testing. Holm
controls familywise alpha 0.05. H3 additionally requires the percentile lower
confidence bound to remain above -0.01. Overall confirmatory success requires
all six corrected tests plus the decision-overhead gate. A mini run can never satisfy
the rule. Negative or mixed results are preserved without retuning.

Latency sensitivities report 33.3, 50, and 100 ms budgets, prefixes 0/2/5/10,
throughput, p50/p95, and measured GPU memory. The primary scientific policy
uses mc02. A deterministic-only row is a compute baseline, not fused RQ5.

### Subgroups, sensitivity, and not-estimable rules

Prespecified subgroups are predicted category, object size, criticality tier,
timeofday, weather, and scene. Score thresholds 0.05/0.10/0.20/0.30 and MC
prefixes 2/5/10 and geometry coefficients 0/0, 0.25/0.25, 0.5/0.5 and 1/1
are mandatory sensitivities. Subgroup and threshold analyses
are secondary and cannot replace the primary result.

A metric is `not_estimable` when it has fewer than the frozen minimum rows,
fewer than two outcome classes, no source groups, no finite predictions, zero
criticality mass, or no accepted detections where acceptance is required. No
synthetic replacement is allowed. Mini lowers only technical sample gates and
bootstrap repetitions.

### Integrity, resumption, and prohibited analyses

Every image feature shard records image ID, schema, materialization/source
fingerprints, consumed shared-shard SHA-256, feature-shard SHA-256, and
image-summary SHA-256. Combined features, model index, predictions, bootstrap,
metrics, tables, figures, and report manifest are hash-validated. Resumption
accepts only compatible artifacts with valid hashes and recomputes stale,
missing, or corrupt downstream artifacts atomically.

Prohibited analyses include confirmatory tuning; per-detection bootstrap;
using target/IoU/group/test/domain columns as model inputs; fitting on test;
claiming deferred candidates are safe or recovered; silently treating prefix
latency estimates as measurements; hiding failures/baselines; presenting mini
values as paper evidence; changing canonical seeds/thresholds/association;
and running `full` without explicit authorization.

## Español

### Pregunta y límite del claim

> ¿Cómo pueden integrarse la incertidumbre fusionada y las salidas calibradas
> del detector en una capa de fusión a nivel de decisión para lograr percepción
> selectiva sensible al riesgo bajo restricciones ADAS de tiempo real?

RQ5 estudia una decisión `aceptar`/`diferir` condicionada a los candidatos
fijos de GroundingDINO. Diferir significa enviar el candidato a un fallback
hipotético, tracker temporal o sensor adicional; no significa ausencia del
objeto ni constituye una acción segura demostrada. RQ5 no cambia cajas/clases,
no recupera falsos negativos, no implementa el fallback ni demuestra seguridad
vehicular. El claim se limita al modelo, prompts, BDD100K y hardware fijados.

La latencia de diez pasadas se mide con sincronización CUDA. Dos/cinco pasadas
son estimaciones lineales explícitas del bloque estocástico de diez; el overhead
CPU de decisión sí se mide. Por tanto se obtiene una frontera offline de
latencia/calidad, no cumplimiento desplegado de deadlines.

### Flujo congelado y prevención de leakage

Train (5.600) ajusta modelos. Validation (2.400) se divide por `sequence_id`
en selección 40%, calibración de componentes 30% y calibración de
política/threshold 30%. Diagnostic test (8) solo valida técnicamente; el
confirmatory test (1.992) se usa una vez tras autorización. No hay grupos
compartidos, tuning confirmatorio ni lectura de modelos/salidas de RQ1--RQ4.

RQ5 consume `groundingdino_mc_v1`, valida compatibilidad y obtiene el
fingerprint con `shared_identity(config)`. V1 ya contiene todos los tensores.
El prefijo operativo es 2 MC; 5/10 son sensibilidades derivadas de los mismos
arrays sin inferencia GPU nueva.

### Outcome, features y criticidad

El outcome es error class-aware con score >= 0,20 e IoU >= 0,50. Los falsos
negativos quedan en resúmenes, no se inventan filas. La confianza usa
`1-score`; incertidumbre fusiona las reducciones semánticas, geométricas y de
representación canónicas, trayectorias, área y score MC medio. La imputación
usa solo train y conserva indicadores de ausencia. Targets, IoU, índices
matched, conteos FN, grupos, estadísticas de test y oráculos no son inputs.

La criticidad usa solo clase/caja predichas. La severidad es 2,0 para usuarios
vulnerables, 1,5 para vehículos grandes/tren, 1,25 para coche y 1,0 para
señales/semáforos. El factor geométrico es
`1 + 0,5*bottomness + 0,5*centrality`; el peso final está en [1,4]. Se usa para
decidir y ponderar riesgo, nunca para predecir el error.

### Métodos y controles

Los modelos son logísticos regularizados con imputación, indicadores,
estandarización, semillas fijas y C en `{0.01,0.1,1,10}`. La selección,
calibración isotónica y threshold usan los folds separados ya descritos.

Los métodos congelados son confianza raw/calibrada, incertidumbre sola,
criticidad sola, fusión tardía no ponderada, `risk_aware_fusion` primario y un
`flat_joint` con los mismos features como control de capacidad. La fusión es
la media 0,5/0,5 fija de las dos probabilidades calibradas; el score de decisión
la multiplica por criticidad. Todos los baselines sensibles al riesgo reciben
el mismo peso. mc05/mc10 son sensibilidades. El threshold maximiza cobertura
en validation con riesgo ponderado <= 0,10; si no existe conjunto no vacío,
se difiere todo.

### Hipótesis, métricas y éxito

H1 exige menor AURC ponderado que confianza calibrada y `flat_joint`; H2 mayor
cobertura a riesgo 0,10 frente a ambos. Las cuatro comparaciones usan bootstrap
pareado por secuencia (2.000) y Holm unilateral con alfa familiar 0,05. H3
exige no inferioridad Brier con margen 0,01. H4 exige overhead p95 <= 5 ms y
latencia end-to-end mc02 estimada p95 <= 100 ms. El éxito confirmatorio requiere
todos los gates; mini nunca puede satisfacerlos.

Se reportan además AUROC/AUPRC/AURC, Brier/NLL/ECE, curvas riesgo-cobertura,
acciones operativas, defer rate, subgrupos congelados, thresholds
0,05/0,10/0,20/0,30, prefijos 2/5/10, presupuestos 33,3/50/100 ms, memoria y
throughput. Bootstrap es por grupo, nunca por detección.

Una métrica queda `not_estimable` sin filas, ambas clases, grupos, predicciones
finitas, masa de criticidad o aceptados necesarios. No se simula evidencia.
Cada shard y artefacto downstream se valida con fingerprints y SHA-256; al
reanudar solo se reutiliza contenido compatible e íntegro.

Se prohíben tuning sobre confirmatory, inputs oracle/test/grupo, ajuste sobre
test, interpretar `defer` como seguro, presentar estimaciones de prefijo como
mediciones, ocultar negativos, usar mini como evidencia, cambiar el contrato
canónico y ejecutar `full` sin autorización explícita.
