# Pre-confirmatory methodological amendments

Status: frozen on 2026-07-31 before any confirmatory feature extraction or
test evaluation. This document supersedes conflicting wording in the original
RQ protocols; diagnostic MINI results are not scientific evidence.

## English

1. One-sided superiority tests use the paired sequence-cluster bootstrap
   distribution centered at the zero boundary null. Non-inferiority tests are
   centered at their frozen negative margin. Uncentered bootstrap replicates
   are used only for percentile confidence intervals. Holm is applied to every
   explicitly listed test in an RQ family.
2. RQ2 train/validation fitting and primary evaluation use the same score
   threshold 0.20 population. Validation sequence groups are split 50/50 into
   disjoint model-selection and isotonic-calibration folds.
3. The deterministic reference universe retains all 900 eligible GroundingDINO
   queries. Every MC association pool contains the nominal top 300 candidates
   plus every eligible deterministic reference query ID. Candidate counts
   before/after the nominal cap are recorded per image.
4. RQ5 coverage is measured over criticality mass and weighted AURC integrates
   over that mass. Its family contains four superiority and two Brier
   non-inferiority tests. The latter must pass Holm and keep their lower 95%
   confidence bound above -0.01. Criticality geometry coefficients are tested
   at 0, 0.25, 0.5 and 1.0.
5. RQ5 makes an offline latency-quality frontier claim. Decision-layer p95 <=
   5 ms is a gate; total mc02/mc05/mc10 latency and 33.3/50/100 ms budgets are
   feasibility results, not a deployed 10 Hz claim.
6. The empirical scope is the pinned GroundingDINO Swin-T checkpoint, ten fixed
   BDD prompts, frozen BDD100K partition and recorded hardware. The study does
   not claim base-to-novel generalization, detector/backbone generality, causal
   attribution to individual layers, external-domain validity, or uncertainty
   for objects with no detection.
7. Each RQ is a separately prespecified family and all five outcomes must be
   reported. No omnibus “all RQs supported” claim is made, and a positive RQ
   cannot rescue a negative RQ.

## Español

Estado: congelado el 2026-07-31 antes de extraer features confirmatorios o
evaluar el test. Este documento reemplaza cualquier texto incompatible de los
protocolos originales; MINI no constituye evidencia científica.

1. Los tests unilaterales de superioridad usan el bootstrap pareado por
   secuencia centrado en la frontera nula cero. La no inferioridad se centra en
   su margen negativo fijado. El bootstrap no centrado se usa solo para
   intervalos percentiles. Holm incluye todos los tests declarados de cada RQ.
2. RQ2 usa la misma población con score >= 0,20 en ajuste y evaluación. Los
   grupos de validación se dividen 50/50, sin solapamiento, entre selección del
   modelo y calibración isotónica.
3. El universo determinista conserva las 900 queries elegibles. Cada pase MC
   incluye el top 300 y toda query de referencia elegible; se registran los
   conteos antes y después del límite nominal.
4. En RQ5 la cobertura y AURC se definen sobre masa de criticidad. La familia
   contiene cuatro tests de superioridad y dos de no inferioridad Brier; estos
   deben superar Holm y mantener el límite inferior del IC 95% sobre -0,01.
   También se evalúan coeficientes geométricos 0, 0,25, 0,5 y 1,0.
5. RQ5 estudia una frontera offline calidad-latencia. El overhead decisional
   p95 <= 5 ms es un gate; la latencia total y los presupuestos son resultados
   de factibilidad, no una afirmación de despliegue a 10 Hz.
6. El alcance se limita a GroundingDINO Swin-T, diez prompts BDD, la partición
   BDD100K congelada y el hardware registrado. No se afirma generalización a
   categorías nuevas, otros detectores/backbones, causalidad de capas, dominio
   externo ni incertidumbre de objetos que no producen detección.
7. Cada RQ es una familia confirmatoria separada y se publican las cinco. No se
   formula una conclusión ómnibus de que todas fueron apoyadas ni una RQ
   positiva rescata otra negativa.
