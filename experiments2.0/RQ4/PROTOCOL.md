# RQ4 frozen protocol / Protocolo congelado de RQ4

Pre-confirmatory revision 3 / Revisión preconfirmatoria 3:
[`METHODOLOGICAL_AMENDMENTS.md`](../METHODOLOGICAL_AMENDMENTS.md).

Protocol freeze date / Fecha de congelación: 2026-07-31.
Pre-confirmatory revision / Revisión preconfirmatoria: 2026-07-31, revision 2.

This protocol was written before implementing RQ4 estimators and before
reading or generating any RQ4 confirmatory output. RQ1--RQ3 artifacts are not
inputs or evidence for RQ4. The only cross-RQ dependency is the immutable,
label-free shared extraction contract.

Revision 2 was frozen after diagnostic validation revealed that mixed-domain
development would measure subgroup performance rather than calibration
transport. No confirmatory output existed. Revision 2 therefore requires
source-only component fitting and calibration, category-conditioned class
calibration, and target-shift evaluation. Earlier diagnostic RQ4 artifacts
are superseded and are not evidence.

Este protocolo se escribió antes de implementar los estimadores de RQ4 y
antes de leer o generar cualquier resultado confirmatorio de RQ4. Los
artefactos de RQ1--RQ3 no son entradas ni evidencia para RQ4. La única
dependencia entre RQ es el contrato inmutable de extracción compartida sin
etiquetas.

La revisión 2 se congeló cuando la validación diagnóstica mostró que desarrollo
con dominios mezclados mediría subgrupos, no transporte de calibración. No
existía ningún output confirmatorio. La revisión exige ajuste y calibración
solo en source, calibración condicionada por categoría y evaluación en target
shift. Los artefactos diagnósticos anteriores quedan sustituidos y no son
evidencia.

## English

### Research question and claim boundary

> To what extent does multi-level post-hoc calibration--combining class-level,
> localization-level, and uncertainty-level calibration--enhance detection
> reliability under domain shifts?

The claim is restricted to the pinned GroundingDINO Swin-T checkpoint, prompt
vocabulary and frozen BDD100K partition. “Domain shift” means prespecified
within-dataset covariate strata in BDD time of day, weather and scene. It does
not mean an unseen dataset, sensor, geography or open-set vocabulary shift.
The experiment measures reliability of fixed detector candidates; it does not
change boxes/classes, recover false negatives or establish ADAS safety.

“To what extent” is quantified by absolute and relative changes with
sequence-clustered confidence intervals in Brier score, negative log
likelihood (NLL), expected calibration error (ECE), error AUROC/AUPRC, AURC,
risk at coverage and coverage at risk. Confirmatory success is narrower than
the descriptive answer and is frozen below.

### Frozen data, detector and shared extraction

- The source-domain subset of train (within the frozen 5,600 images) fits the
  three component models. Source IDs are selected from frozen BDD metadata
  before detector extraction, so shifted development images consume neither
  GPU inference nor feature materialization for RQ4. Shifted development rows
  never fit a model.
- The source-domain subset of validation (within the frozen 2,400 images) is
  split by source sequence into disjoint selection and final-calibration
  folds. Shifted validation rows are descriptive only and cannot tune RQ4.
- Diagnostic test (8 images) is only for mini technical validation.
- Confirmatory test (1,992 images) is evaluated once after authorization.

RQ4 consumes schema v1 at `data/derived/groundingdino_mc_v1`, with one
deterministic pass and ten DropPath passes per image. The v1 arrays provide all
required scores, categories, boxes, presence masks, decoder trajectories and
representations; no schema extension is authorized. RQ4 calls
`validate_consumer_compatibility(config, "rq4")` before extraction and obtains
the shared fingerprint through `shared_identity(config)`.

The operational score threshold is 0.20. Detection correctness is the frozen,
score-ordered, class-consistent one-to-one match at IoU >= 0.50. False
negatives remain image-level context and are not synthesized as scored rows.

### Prespecified domain shifts

The reference/source stratum is the conjunction `daytime + clear + city street`.
For each row, the time, weather and scene axes are marked shifted when their
value differs from that reference. Undefined/unknown metadata are reported as
unknown and are excluded from axis-specific confirmatory claims; they remain
in the overall “any shifted axis” analysis so missing metadata cannot be used
to discard difficult detections.

Component fitting, hyperparameter selection and isotonic calibration use only
the exact reference conjunction. The primary target domain is all detections
with at least one shifted axis. Secondary
descriptive strata are reference, each shifted axis, exact shift severity
(zero, one, two or three axes), and observed attribute values. These strata
were selected semantically and from development metadata counts only, without
reading confirmatory labels or RQ4 predictions. Shift descriptors are never
model features.

### Targets and feature groups

The class-level target is agreement between the predicted class and the class
of the maximum-IoU non-crowd ground-truth object, provided maximum IoU is at
least 0.10. The localization-level target is class-agnostic maximum IoU >=
0.50. The uncertainty-level and final target is primary one-to-one detection
correctness. These targets are training/evaluation labels only.

The class level uses detector score plus predicted category as a one-hot
condition. Categories below the train-only minimum count and categories
unseen at fit time use a shared global fallback. The localization level uses eight
frozen spatial features: reference-to-MC IoU mean/std, pairwise MC IoU loss,
MC coordinate variance, absence rate, deterministic reference-point variance
and step, and normalized box area. The uncertainty level uses eleven frozen
semantic, geometric and representation features: mutual information,
predictive entropy, class disagreement, score variance, pairwise IoU loss,
box variance, absence, embedding variance, embedding cosine instability,
deterministic hidden-state step and mean MC score.

Canonical MC reductions are imported from `adas_ovd.mc_features`. Complete
stochastic absence remains explicit; missing values use train-fitted median
imputation plus missingness indicators. Ground-truth targets, matched IoU,
file/image/sequence IDs, domain attributes and test-derived statistics are
prohibited inference inputs.

### Methods, baselines, ablations and capacity controls

All learned component models are regularized logistic regressions with median
imputation, missingness indicators and standardization. Predicted category is
one-hot encoded for the class and flat controls, with a train-only rare-class
fallback. C is selected from
`{0.01, 0.1, 1, 10}` by AUROC on the validation selection fold, with smaller C
breaking ties, then refit on train plus selection. Method-level isotonic maps
are fit only on the disjoint validation calibration fold.

- `raw_confidence`: `1 - score`, without an isotonic map.
- `confidence_calibrated`: the same rank with a final isotonic map.
- `class_only`: `1 - P(class correct)`.
- `localization_only`: `1 - P(well localized)`.
- `uncertainty_only`: `1 - P(detection correct | uncertainty features)`.
- `class_localization`, `class_uncertainty`, and
  `localization_uncertainty`: prespecified pairwise product ablations.
- `multilevel` (primary): `1 - P(class correct) * P(well localized) *
  P(detection correct | uncertainty features)`.
- `flat_joint`: a single logistic model over the union of the same raw score,
  localization and uncertainty features. It controls feature access, fitting
  data, estimator family, hyperparameter grid and comparable coefficient
  capacity without the proposed factorization.

Except `raw_confidence`, every method receives the same final isotonic
calibration treatment. Ranking metrics use pre-isotonic rank scores because
isotonic ties can alter ordering; probability metrics use calibrated outputs.
Parameter counts are reported to expose any residual capacity difference.
Prefixes of 2, 5 and 10 MC passes are a mandatory multilevel sensitivity and
reuse the first passes of the frozen ten-pass cache without new GPU inference.

### Hypotheses, metrics and success rule

- H1: on shifted detections, `multilevel` has lower Brier than
  `confidence_calibrated` and `flat_joint`.
- H2: on shifted detections, `multilevel` has lower NLL than both baselines.
- H3: on shifted detections, `multilevel` has lower AURC than both baselines.
- H4 (secondary): multilevel reduces ECE and worst-stratum calibration error
  without a materially adverse reference-domain reliability gap.

The six H1--H3 comparisons form one confirmatory family. Paired bootstrap
resamples source sequences 2,000 times. Differences are oriented so positive
means improvement. Null-centered paired cluster-bootstrap p-values receive Holm correction at
familywise alpha 0.05. Overall success requires favorable point estimates and
rejection of no improvement in all six comparisons on the confirmatory shifted
subset. A diagnostic mini run can never satisfy this rule. Negative or mixed
results are retained without confirmatory retuning.

The bootstrap draws are shared across methods and metrics. Cluster resampling
is implemented exactly through integer cluster-multiplicity weights rather
than repeatedly expanding the same detection rows; unit tests verify equality
with explicit cluster duplication. This changes runtime and memory use, not
the bootstrap estimand.

Secondary metrics are ECE at 10/15/20 bins, maximum calibration error, AUROC,
AUPRC, risk at 0.5/0.7/0.8/0.9/1.0 coverage, coverage at risk 0.05/0.10/0.20,
domain reliability gaps, per-axis/severity/value results, object size and
predicted-category subgroups, and CPU/GPU cost. Score thresholds
0.05/0.10/0.20/0.30 and 2/5/10 MC prefixes are mandatory sensitivities.

### Not-estimable, integrity and resumption rules

A metric is `not_estimable` with fewer than the frozen minimum rows, fewer
than two outcome classes, no source groups, no finite outputs or no
predictions. No synthetic replacement is allowed. Mini lowers only technical
sample gates and bootstrap repetitions.

Each RQ4 image shard records image ID, materialization fingerprint,
source-code hash, schema, consumed shared-shard SHA-256, feature-shard SHA-256
and image-summary SHA-256. Combined features, model index, predictions,
bootstrap, metrics and report outputs are hash-validated. Resumption accepts
only artifacts whose identities and hashes pass. RQ4 must fail on a failed
data audit, manifest/shared-contract mismatch, group overlap, target/domain
leakage, stale/corrupt hashes or a confirmatory/diagnostic namespace mix.

### Limitations and prohibited analyses

These are natural BDD covariate strata rather than externally induced or
unseen-domain shifts, and they overlap. Component probabilities are correlated,
so their product is a structured score rather than an independence claim; the
final calibrator absorbs scale mismatch but cannot guarantee transportability.
The class target depends on nearest-object assignment, MC association can
fail, detection-conditioned calibration cannot assess missed objects, and one
detector/dataset cannot establish general OVD or ADAS safety.

Prohibited analyses include confirmatory feature/hyperparameter/threshold or
hypothesis selection; fitting normalization/calibration on test; using oracle
targets, IoU or domain attributes as inputs; per-detection bootstrap; dropping
unknown domains or absent MC observations after seeing outcomes; hiding
baselines/negative results; presenting mini values as paper evidence; and
running `full` without explicit authorization.

---

## Español

### Pregunta de investigación y límite del claim

> ¿En qué medida la calibración post-hoc multinivel —combinando calibración a
> nivel de clase, localización e incertidumbre— mejora la fiabilidad de las
> detecciones bajo cambios de dominio?

El claim se limita al checkpoint GroundingDINO Swin-T, al vocabulario de
prompts y a la partición BDD100K congelados. “Cambio de dominio” significa
estratos covariables internos y preespecificados de hora, clima y escena en
BDD. No significa otro dataset, sensor, geografía ni vocabulario open-set. El
experimento mide la fiabilidad de candidatos fijos; no cambia cajas/clases,
recupera falsos negativos ni demuestra seguridad ADAS.

La magnitud se cuantifica mediante cambios absolutos y relativos con intervalos
agrupados por secuencia en Brier, NLL, ECE, AUROC/AUPRC de error, AURC, riesgo
a cobertura y cobertura a riesgo. El éxito confirmatorio es más estrecho que
la descripción y queda congelado abajo.

### Datos, detector y extracción compartida

- Solo el subconjunto source de train dentro de las 5.600 imágenes ajusta los
  tres componentes. Los IDs fuente se eligen con metadatos BDD congelados
  antes de extraer, por lo que imágenes desplazadas de desarrollo no consumen
  inferencia GPU ni materialización RQ4; esas filas nunca ajustan modelos.
- Solo el subconjunto source de validation dentro de las 2.400 imágenes se
  divide por secuencia en selección y calibración final. Validation desplazado
  es únicamente descriptivo y no puede ajustar RQ4.
- Diagnostic test (8) solo sirve para validación técnica mini.
- Confirmatory test (1.992) se evalúa una vez tras autorización.

RQ4 consume schema v1 en `data/derived/groundingdino_mc_v1`, con una pasada
determinista y diez DropPath por imagen. Los arrays v1 contienen scores,
clases, cajas, máscaras de presencia, trayectorias y representaciones; no se
autoriza extender el esquema. RQ4 valida compatibilidad antes de extraer y
obtiene el fingerprint mediante `shared_identity(config)`.

El threshold operativo es 0.20. La corrección de detección usa matching
uno-a-uno sensible a clase, ordenado por score, con IoU >= 0.50. Los falsos
negativos permanecen como contexto por imagen.

### Cambios de dominio preespecificados

El estrato de referencia es `daytime + clear + city street`. Cada eje difiere
si su valor no coincide. Los metadatos indefinidos se reportan como unknown y
se excluyen de claims por eje, pero permanecen en “algún eje desplazado” para
impedir que la ausencia de metadata descarte detecciones difíciles.

El ajuste de componentes, la selección y la isotónica usan exclusivamente la
conjunción source. El dominio target primario incluye detecciones con al menos
un eje desplazado. Los
estratos secundarios son referencia, cada eje, severidad exacta de cero a tres
ejes y valores observados. Se eligieron semánticamente y usando solo conteos de
metadata de desarrollo, sin etiquetas confirmatorias ni predicciones RQ4. Los
descriptores de shift nunca son features del modelo.

### Targets y grupos de características

El target de clase exige acuerdo con la clase del objeto no-crowd de máximo
IoU y que ese IoU sea al menos 0.10. El target de localización es IoU máximo
sin clase >= 0.50. El target de incertidumbre y final es la corrección
uno-a-uno. Solo se usan como etiquetas.

El nivel de clase usa score y categoría predicha one-hot. Las categorías por
debajo del mínimo contado solo en train y las no vistas usan un fallback
global compartido. Localización usa ocho features espaciales:
media/std de IoU referencia-MC, pérdida IoU entre pares, varianza de cajas,
ausencia, varianza/paso deterministas de puntos de referencia y área
normalizada. Incertidumbre usa once features semánticos, geométricos y de
representación: información mutua, entropía, desacuerdo, varianza de score,
pérdida IoU, varianza de caja, ausencia, varianza/inestabilidad coseno de
embedding, paso de hidden state y score MC medio.

Las reducciones canónicas se importan de `adas_ovd.mc_features`. La ausencia
total sigue explícita; los valores ausentes usan mediana aprendida en train e
indicadores. Targets, IoU matched, IDs, atributos de dominio y estadísticas de
test están prohibidos como entradas.

### Métodos, baselines, ablaciones y capacidad

Los componentes son regresiones logísticas regularizadas con imputación,
indicadores y estandarización; clase y control plano codifican categoría
one-hot con fallback de clases raras aprendido solo en train. C se selecciona
entre `{0.01, 0.1, 1, 10}` por
AUROC en selección de validation, desempatando por menor C, y se reajusta con
train + selección. Los mappings isotónicos se ajustan solo en el fold disjunto
de calibración.

- `raw_confidence`: `1 - score`, sin isotónica.
- `confidence_calibrated`: mismo ranking con isotónica final.
- `class_only`: `1 - P(clase correcta)`.
- `localization_only`: `1 - P(bien localizado)`.
- `uncertainty_only`: `1 - P(detección correcta | features de incertidumbre)`.
- Ablaciones por pares: `class_localization`, `class_uncertainty` y
  `localization_uncertainty`.
- `multilevel` (primario): uno menos el producto de las tres probabilidades.
- `flat_joint`: una sola logística sobre la unión de los mismos features;
  controla acceso a features, datos, familia, grid y capacidad comparable sin
  la factorización propuesta.

Salvo `raw_confidence`, todos reciben idéntica calibración isotónica final. Las
métricas de ranking usan scores pre-isotónica y las probabilísticas la salida
calibrada. Se reportan parámetros. Los prefijos MC 2/5/10 reutilizan la caché
congelada y no ejecutan inferencia GPU adicional.

### Hipótesis, métricas y regla de éxito

- H1: en detecciones desplazadas, `multilevel` reduce Brier frente a
  `confidence_calibrated` y `flat_joint`.
- H2: reduce NLL frente a ambos.
- H3: reduce AURC frente a ambos.
- H4 secundaria: reduce ECE y el peor error por estrato sin deterioro material
  del gap en referencia.

Las seis comparaciones H1--H3 forman una familia. El bootstrap pareado
remuestrea secuencias 2.000 veces; positivo siempre significa mejora. Los
p-values unilaterales reciben Holm con alpha familiar 0.05. El éxito global
exige dirección favorable y rechazo en las seis comparaciones sobre el
subconjunto confirmatorio desplazado. Mini nunca satisface esta regla. Se
preservan resultados negativos o mixtos sin retuning.

Los draws bootstrap son comunes entre métodos y métricas. El remuestreo por
cluster usa pesos enteros de multiplicidad exactamente equivalentes a duplicar
filas por secuencia, igualdad cubierta por tests. Esto reduce tiempo y memoria,
pero no cambia el estimando bootstrap.

Son secundarios ECE 10/15/20, máximo error de calibración, AUROC, AUPRC,
puntos de riesgo/cobertura, gaps de dominio, ejes/severidad/valores, tamaño,
categoría y coste. Son obligatorias sensibilidades de score
0.05/0.10/0.20/0.30 y prefijos MC 2/5/10.

### Reglas no estimables, integridad y reanudación

Una métrica es `not_estimable` sin filas mínimas, ambas clases, grupos fuente,
salidas finitas o predicciones. No se simula. Mini solo reduce gates y
repeticiones.

Cada shard registra image ID, fingerprint, hash de código, schema y SHA-256
del shard compartido, shard específico y resumen. Features, índice de modelos,
predicciones, bootstrap, métricas y reportes se validan por hash. Reanudar solo
acepta identidades y hashes válidos. RQ4 falla ante data audit/manifest/contrato
incompatibles, leakage de grupos/targets/dominio, hashes corruptos o mezcla de
namespaces diagnósticos y confirmatorios.

### Limitaciones y análisis prohibidos

Son estratos covariables BDD naturales y solapados, no dominios externos. Las
probabilidades componentes están correlacionadas; su producto es un score
estructurado, no un supuesto de independencia. La calibración final corrige
escala, pero no garantiza transportabilidad. El target de clase depende de
asignación al objeto más cercano, la asociación MC puede fallar y el análisis
condicionado a detección no evalúa objetos omitidos.

Se prohíbe seleccionar features, hiperparámetros, thresholds o hipótesis con
confirmatory; ajustar normalización/calibración en test; usar targets oracle,
IoU o dominio como inputs; bootstrap por detección; eliminar unknown o
ausencias tras ver resultados; ocultar baselines/negativos; presentar mini como
evidencia; y ejecutar `full` sin autorización explícita.
