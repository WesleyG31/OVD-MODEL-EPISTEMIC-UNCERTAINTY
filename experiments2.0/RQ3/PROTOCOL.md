# RQ3 frozen protocol / Protocolo congelado de RQ3

Pre-confirmatory revision 3 / Revisión preconfirmatoria 3:
The final rules are incorporated directly in this protocol. One-sided
superiority tests use the paired sequence-cluster bootstrap centered at the
zero-boundary null; non-inferiority tests use their frozen negative margin;
and uncentered replicates are used only for percentile intervals. The
deterministic reference universe retains all 900 eligible queries, while every
MC pool contains the nominal top 300 plus all eligible reference query IDs.
The empirical scope is limited to the pinned checkpoint, prompts, BDD100K
partition, and recorded hardware. RQ3 is a separately prespecified family and
cannot rescue or be rescued by another RQ. MINI results are not evidence.

Las reglas finales están integradas directamente en este protocolo. Los tests
unilaterales usan bootstrap pareado por secuencia centrado en la frontera nula;
la no inferioridad usa su margen negativo congelado; y las réplicas no
centradas se reservan para intervalos percentiles. El universo determinista
conserva las 900 queries elegibles y cada pool MC contiene el top 300 nominal
más todas las queries de referencia elegibles. El alcance se limita al
checkpoint, prompts, partición BDD100K y hardware registrados. RQ3 es una
familia separada y no rescata ni puede ser rescatada por otra RQ. MINI no
constituye evidencia.

Protocol freeze date / Fecha de congelación: 2026-07-31.

This protocol was written before implementing the RQ3 estimators and before
reading or generating any RQ3 confirmatory output. Existing RQ1/RQ2 diagnostic
artifacts and legacy exploratory folders are not evidence for RQ3.

Este protocolo se escribió antes de implementar los estimadores de RQ3 y antes
de leer o generar cualquier resultado confirmatorio de RQ3. Los artefactos
diagnósticos existentes de RQ1/RQ2 y las carpetas exploratorias heredadas no
son evidencia para RQ3.

## English

### Research question and claim boundary

> How does fusing classification confidence with spatial localization quality
> improve the reliability, ranking, and calibration of open-vocabulary
> detections in safety-critical driving scenarios?

The claim is restricted to the pinned GroundingDINO Swin-T checkpoint, prompt
vocabulary and frozen BDD100K partition. “Classification confidence” is the
detector's maximum prompted category score; it is not asserted to be a pure
class posterior. “Spatial localization quality” is an inference-time estimate
of whether a predicted box has class-agnostic maximum IoU at least 0.50 with a
non-crowd ground-truth object. Ground-truth IoU is a training/evaluation target
only and is prohibited as an inference-time input.

Improvement means better ranking of fixed detector candidates, lower selective
detection risk and better calibrated probability of class-aware one-to-one
detection error. RQ3 does not claim to increase recall, recover missed objects,
change the detector boxes/classes, or establish ADAS safety.

### Frozen data, detector and shared extraction

- Train (5,600 images) fits quality estimators.
- Validation (2,400 images) is divided deterministically by source sequence
  into disjoint selection and calibration folds.
- Diagnostic test (8 images) is used only by mini technical validation.
- Confirmatory test (1,992 images) is used once for final evaluation.

RQ3 consumes the label-free schema-v1 cache at
`data/derived/groundingdino_mc_v1`. It uses the same deterministic reference
and ten DropPath passes as RQ1/RQ2, including the frozen seed sequence,
candidate threshold and association policy. RQ3 does not read RQ1/RQ2 models,
outputs, metrics or reported results. The v1 arrays are sufficient; no schema
extension is authorized by this protocol.

The extraction threshold remains 0.05 and the primary operational score
threshold is 0.20. Detection correctness is the existing score-ordered,
one-to-one, class-consistent match at IoU >= 0.50. False negatives are reported
as detector context but are not converted into synthetic scored detections.

### Variables and targets

The independent variable is ranking/calibration method. Dependent variables
are error AUROC, AUPRC, AURC, risk at coverage, coverage at risk, Brier score,
negative log likelihood, ECE, COCO AP/AR under method-specific re-ranking, and
CPU/GPU cost.

For each fixed reference detection, RQ3 records three label-only targets:

1. `localization_iou`: maximum IoU with any non-crowd ground-truth box,
   ignoring class;
2. `is_well_localized`: `localization_iou >= 0.50`;
3. the existing class-aware `is_error` target from one-to-one matching.

The class-agnostic target isolates box alignment from category correctness.
It can assign high IoU to duplicate proposals, whereas the primary one-to-one
error target penalizes duplicates; this distinction is retained and reported.

### Frozen feature groups

The eight spatial localization features are:

1. mean reference-to-associated-MC box IoU;
2. standard deviation of reference-to-MC IoU;
3. mean pairwise MC IoU loss;
4. MC box-coordinate variance;
5. MC absence rate;
6. deterministic decoder reference-point variance;
7. deterministic decoder reference-point step length;
8. predicted box area divided by image area.

Canonical MC reductions are imported from `adas_ovd.mc_features`; RQ3 adds
only the reference-to-MC agreement and normalized area formulas that are not
available in schema v1 helpers. Missing stochastic matches remain missing and
absence stays explicit.

The capacity control uses exactly eight non-spatial features with the same
preprocessing, estimator family and hyperparameter grid: semantic mutual
information, predictive entropy, class disagreement, MC score variance,
embedding variance, embedding cosine instability, deterministic hidden-state
step and mean present-pass score.

No ground-truth target, matched IoU, subgroup, filename, sequence identity or
test-derived statistic may enter an inference-time feature set.

### Prespecified methods, baselines and ablations

- `confidence`: uncertainty `1 - classification score`.
- `spatial_agreement`: fixed uncertainty `1 - mean reference-to-MC IoU`, with
  complete stochastic absence assigned maximum uncertainty.
- `learned_spatial_quality`: uncertainty `1 - P(well localized)` from the
  spatial-only logistic quality estimator.
- `equal_fusion`: equal arithmetic mean of confidence uncertainty and learned
  spatial-quality uncertainty.
- `product_fusion` (primary): `1 - score * P(well localized)`.
- `capacity_control_product`: the same product and logistic capacity as the
  primary method, but the quality model uses the eight non-spatial controls.
- `direct_spatial_fusion`: regularized logistic error model using score plus
  the eight spatial features; this is a higher-flexibility sensitivity, not
  the primary contribution.

Both quality estimators use median imputation with missingness indicators,
standardization and logistic regression. Regularization is selected from
`C = {0.01, 0.1, 1, 10}` by localization AUROC on the validation selection
fold, with smaller C breaking ties. The chosen estimator is refit on train plus
selection data. The direct fusion uses the identical grid, selected by
detection-error AUROC.

This design controls ordinary confidence, spatial quality alone, a fixed equal
fusion, added feature count, equal estimator capacity and extra nonlinear
composition. Prefix models at 2, 5 and 10 MC passes test whether an effect is
only caused by using more stochastic passes; all prefixes reuse the frozen
ten-pass sequence and require no extra GPU inference.

### Calibration

Every method, including confidence, receives the same post-hoc isotonic mapping
from its uncertainty rank to detection-error probability. Isotonic calibration
is fit only on validation calibration groups, which are disjoint from model
selection groups and from train/test groups. Confirmatory labels never fit or
alter a calibrator. Calibration is measured with Brier, NLL and ECE; ECE is
repeated with 10, 15 and 20 bins. Isotonic ties may change ranking, so ranking
metrics always use the pre-calibration uncertainty and probability metrics use
the calibrated output.

### Hypotheses, comparisons and success rule

- H1: `product_fusion` has higher error AUROC than `confidence` and
  `capacity_control_product`.
- H2: `product_fusion` has lower AURC than both baselines.
- H3: calibrated `product_fusion` has lower Brier score than both baselines.
- H4 (secondary): spatial fusion improves AUPRC/risk operating points and
  re-ranking is directionally more beneficial at stricter localization IoU
  (for example AP75) without hiding AP50 or recall changes.

The six H1-H3 comparisons form one confirmatory family: three metrics times
two baselines. Paired source-sequence bootstrap differences use 2,000
repetitions and are oriented so positive always means improvement. One-sided
null-centered paired cluster-bootstrap p-values receive Holm correction at
familywise alpha 0.05. The
frozen overall success rule requires favorable point estimates and rejection
of no improvement in all six comparisons. Ranking success (four comparisons)
and calibration success (two comparisons) are also reported separately, but
neither may be relabeled after seeing test results. Mixed or negative results
must be preserved without confirmatory retuning.

Method intervals use sequence-clustered percentile 95% bootstrap intervals.
COCO AP/AR under alternative scores is secondary descriptive context and is
not used to rescue a failed primary family.

### Sensitivity, subgroups and not-estimable rules

Mandatory sensitivities are 2/5/10 MC prefixes, score thresholds 0.05, 0.10,
0.20 and 0.30, localization at IoU 0.50 versus 0.75, and ECE bin counts
10/15/20. Descriptive subgroups are time of day, weather, scene, predicted
category and predicted object size.

A metric or subgroup is `not_estimable` when it has fewer than the frozen
minimum rows, fewer than two outcome classes, no source groups, no predictions,
or no finite estimator output. It must not be replaced by a simulated value.
The mini configuration lowers only technical sample-count gates and bootstrap
repetitions; it does not alter hypotheses or scientific conclusions.

### Integrity, resumption and leakage controls

Each RQ3 image shard records image ID, materialization fingerprint, source-code
hash, schema, consumed shared-shard SHA-256, feature-shard SHA-256 and
image-summary SHA-256. Combined features, model index, predictions, bootstrap,
metrics and report artifacts are hash-validated. A resume reuses only artifacts
whose identity and hashes pass; missing, stale or corrupt RQ3 artifacts are
recomputed without deleting valid shared/RQ1/RQ2 artifacts.

The implementation must fail on a non-passing data audit, manifest mismatch,
shared-contract mismatch, CPU fallback during required extraction, stale hash,
overlapping validation groups, target leakage into features, one-class fitting
fold, non-finite required output or confirmatory/diagnostic namespace mix.

### Limitations and prohibited analyses

The spatial estimator is learned on one detector/dataset and is not a direct
IoU head trained jointly with the detector. MC DropPath agreement is a proxy,
association errors can affect it, and class-agnostic maximum IoU does not
penalize duplicate proposals by itself. Detection-conditioned methods cannot
score wholly missed objects. Post-hoc calibration may not transfer across
domains, and one checkpoint/prompt vocabulary cannot support a universal OVD
or safety claim.

Prohibited analyses include confirmatory feature selection, normalization,
hyperparameter tuning, threshold tuning or hypothesis changes; oracle IoU or
TP/FP as inputs; per-detection bootstrap; many-to-one matching for the primary
error target; dropping absent MC observations; hiding confidence-only or
capacity-control results; presenting mini/diagnostic values as evidence;
claiming changed recall when boxes are fixed; and running `full` without
explicit user authorization.

---

## Español

### Pregunta de investigación y límite del claim

> ¿Cómo mejora la fiabilidad, el ranking y la calibración de las detecciones
> open-vocabulary en escenarios críticos de conducción la fusión de confianza
> de clasificación con calidad de localización espacial?

El claim se limita al checkpoint GroundingDINO Swin-T, al vocabulario de
prompts y a la partición BDD100K congelados. La “confianza de clasificación” es
el máximo score de categoría del detector condicionado por prompts; no se
afirma que sea un posterior puro de clase. La “calidad de localización
espacial” es una estimación en inferencia de si una caja predicha alcanza IoU
máximo, sin considerar clase, de al menos 0.50 con un objeto ground truth no
crowd. El IoU ground truth solo es target de entrenamiento/evaluación y está
prohibido como entrada en inferencia.

Mejora significa mejor ranking de los candidatos fijos del detector, menor
riesgo selectivo y mejor probabilidad calibrada del error de detección definido
mediante matching uno-a-uno y sensible a clase. RQ3 no afirma aumentar recall,
recuperar objetos omitidos, cambiar cajas/clases ni establecer seguridad ADAS.

### Datos, detector y extracción compartida congelados

- Train (5.600 imágenes) ajusta los estimadores de calidad.
- Validation (2.400 imágenes) se divide de forma determinista por secuencia
  fuente en folds disjuntos de selección y calibración.
- Diagnostic test (8 imágenes) solo sirve para validación técnica mini.
- Confirmatory test (1.992 imágenes) se usa una vez en la evaluación final.

RQ3 consume la caché sin etiquetas schema-v1 en
`data/derived/groundingdino_mc_v1`. Usa la misma referencia determinista y las
diez pasadas DropPath de RQ1/RQ2, incluida la secuencia de semillas, umbral de
candidatos y política de asociación congelados. RQ3 no lee modelos, outputs,
métricas ni resultados de RQ1/RQ2. Los arrays v1 son suficientes; este
protocolo no autoriza extender el esquema.

El umbral de extracción sigue siendo 0.05 y el umbral operativo primario es
0.20. La corrección de detección usa el matching existente, ordenado por score,
uno-a-uno, sensible a clase y con IoU >= 0.50. Los falsos negativos se reportan
como contexto del detector, pero no se convierten en detecciones sintéticas.

### Variables y targets

La variable independiente es el método de ranking/calibración. Las variables
dependientes son AUROC y AUPRC de error, AURC, riesgo a cobertura, cobertura a
riesgo, Brier, log loss, ECE, AP/AR COCO al reordenar por cada método y coste
CPU/GPU.

Para cada detección de referencia fija, RQ3 registra tres targets que solo son
labels:

1. `localization_iou`: IoU máximo con cualquier caja ground truth no crowd,
   ignorando la clase;
2. `is_well_localized`: `localization_iou >= 0.50`;
3. el target `is_error` existente, sensible a clase y uno-a-uno.

El target independiente de clase separa alineación espacial y corrección de
categoría. Puede asignar IoU alto a propuestas duplicadas, mientras el target
primario uno-a-uno penaliza duplicados; esta distinción se conserva y reporta.

### Grupos de características congelados

Las ocho características espaciales son:

1. IoU medio entre caja de referencia y cajas MC asociadas;
2. desviación estándar de ese IoU;
3. pérdida IoU media entre pares de cajas MC;
4. varianza de coordenadas de cajas MC;
5. tasa de ausencia MC;
6. varianza determinista de puntos de referencia del decoder;
7. desplazamiento determinista de puntos de referencia del decoder;
8. área de caja predicha dividida por el área de imagen.

Las reducciones MC canónicas se importan desde `adas_ovd.mc_features`; RQ3 solo
añade el acuerdo referencia-MC y el área normalizada, cuyas fórmulas no existen
en los helpers v1. Los matches estocásticos ausentes permanecen ausentes y la
ausencia queda explícita.

El control de capacidad usa exactamente ocho características no espaciales con
el mismo preprocesamiento, familia de estimador y grid: información mutua
semántica, entropía predictiva, desacuerdo de clase, varianza de score MC,
varianza de embedding, inestabilidad coseno de embedding, paso determinista de
hidden state y score medio de las pasadas presentes.

Ningún target ground truth, IoU matched, subgrupo, nombre de archivo, identidad
de secuencia o estadístico derivado de test puede entrar en un feature set de
inferencia.

### Métodos, baselines y ablaciones preespecificados

- `confidence`: incertidumbre `1 - score de clasificación`.
- `spatial_agreement`: incertidumbre fija `1 - IoU medio referencia-MC`; la
  ausencia estocástica total recibe incertidumbre máxima.
- `learned_spatial_quality`: `1 - P(bien localizado)` del estimador espacial.
- `equal_fusion`: media aritmética de incertidumbre de confianza y calidad
  espacial aprendida.
- `product_fusion` (primario): `1 - score * P(bien localizado)`.
- `capacity_control_product`: mismo producto y capacidad logística, pero el
  modelo de calidad usa los ocho controles no espaciales.
- `direct_spatial_fusion`: modelo logístico de error con score y las ocho
  características espaciales; es una sensibilidad más flexible, no la
  contribución primaria.

Ambos estimadores de calidad usan imputación por mediana con indicadores de
ausencia, estandarización y regresión logística. Se selecciona regularización
entre `C = {0.01, 0.1, 1, 10}` por AUROC de localización en el fold de selección
de validation, con el menor C para desempatar. El estimador elegido se reajusta
con train más selección. La fusión directa usa el mismo grid y AUROC de error.

El diseño controla confianza ordinaria, calidad espacial aislada, fusión fija
igual, cantidad de features, capacidad equivalente y composición aprendida.
Los prefijos de 2, 5 y 10 pasadas comprueban si un efecto proviene solo de más
pasadas; reutilizan la secuencia de diez pasadas y no requieren inferencia GPU
adicional.

### Calibración

Cada método, incluida confianza, recibe el mismo mapping isotónico desde su
ranking de incertidumbre hasta probabilidad de error. Se ajusta solo con los
grupos del fold de calibración de validation, disjuntos de selección,
train y test. Las etiquetas confirmatorias nunca ajustan un calibrador. Se
miden Brier, NLL y ECE; ECE se repite con 10, 15 y 20 bins. Como los empates
isotónicos pueden alterar ranking, las métricas de ranking usan incertidumbre
precalibrada y las métricas probabilísticas usan la salida calibrada.

### Hipótesis, comparaciones y regla de éxito

- H1: `product_fusion` tiene mayor AUROC de error que `confidence` y
  `capacity_control_product`.
- H2: `product_fusion` tiene menor AURC que ambos baselines.
- H3: `product_fusion` calibrado tiene menor Brier que ambos baselines.
- H4 (secundaria): la fusión espacial mejora AUPRC/puntos operativos y el
  re-ranking beneficia direccionalmente más la localización estricta (por
  ejemplo AP75) sin ocultar AP50 ni cambios de recall.

Las seis comparaciones H1-H3 forman una sola familia confirmatoria: tres
métricas por dos baselines. Las diferencias bootstrap pareadas por secuencia
fuente usan 2.000 repeticiones y se orientan para que positivo siempre sea
mejora. Los p-values unilaterales reciben corrección de Holm con alpha familiar
0.05. La regla global exige dirección favorable y rechazo de no mejora en las
seis comparaciones. También se reporta éxito de ranking (cuatro) y calibración
(dos), pero no se pueden redefinir tras ver test. Los resultados mixtos o
negativos se preservan sin retuning confirmatorio.

Los intervalos por método usan bootstrap percentil 95% agrupado por secuencia.
AP/AR COCO bajo scores alternativos es contexto descriptivo secundario y no
rescata una familia primaria fallida.

### Sensibilidad, subgrupos y reglas `not_estimable`

Las sensibilidades obligatorias son prefijos MC 2/5/10, thresholds de score
0.05/0.10/0.20/0.30, localización a IoU 0.50 frente a 0.75 y ECE con
10/15/20 bins. Los subgrupos descriptivos son hora, clima, escena, categoría
predicha y tamaño predicho.

Una métrica o subgrupo es `not_estimable` si no alcanza el mínimo congelado de
filas, no contiene ambas clases, no tiene grupos fuente, no tiene predicciones
o no produce salida finita. No se reemplaza por simulación. La configuración
mini solo reduce gates técnicos y repeticiones bootstrap; no cambia hipótesis
ni conclusiones científicas.

### Integridad, reanudación y controles de leakage

Cada shard RQ3 registra ID de imagen, fingerprint de materialización, hash de
código fuente, esquema, SHA-256 del shard compartido consumido, SHA-256 del
feature shard y del resumen. Features combinados, índice de modelos,
predicciones, bootstrap, métricas y reportes se validan por hash. Al reanudar
solo se reutilizan artefactos cuya identidad y hashes pasan; los ausentes,
stale o corruptos se recalculan sin borrar artefactos válidos compartidos o de
RQ1/RQ2.

La implementación debe fallar ante data audit fallido, manifiesto incompatible,
contrato compartido incompatible, fallback CPU durante extracción requerida,
hash stale, solapamiento de grupos de validación, leakage de targets a features,
fold de ajuste con una sola clase, salida requerida no finita o mezcla de
namespaces diagnósticos/confirmatorios.

### Limitaciones y análisis prohibidos

El estimador espacial se aprende con un detector/dataset y no es un head IoU
entrenado junto al detector. El acuerdo MC DropPath es un proxy, los errores de
asociación pueden afectarlo y el IoU máximo sin clase no penaliza por sí mismo
las propuestas duplicadas. Los métodos condicionados a detección no puntúan
objetos completamente omitidos. La calibración post-hoc puede no transferir a
otros dominios y un checkpoint/vocabulario no permite un claim OVD o de
seguridad universal.

Se prohíbe seleccionar features, normalizar, ajustar hiperparámetros/thresholds
o cambiar hipótesis con confirmatory; usar IoU oracle o TP/FP como entradas;
bootstrap por detección; matching muchos-a-uno para el error primario; eliminar
ausencias MC; ocultar confianza o el control de capacidad; presentar mini como
evidencia; afirmar cambios de recall con cajas fijas; y ejecutar `full` sin
autorización explícita del usuario.
