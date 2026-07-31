# RQ5 risk-aware selective perception / Percepción selectiva sensible al riesgo RQ5

## English

### Scope and status

RQ5 asks how fused epistemic uncertainty and calibrated detector outputs can
feed an `accept`/`defer` decision layer under ADAS latency constraints. The
implementation has passed a two-image, two-namespace GPU repeatability smoke
test and a diagnostic 6/4/8 mini end-to-end run. It is technically ready for
an explicitly authorized confirmatory run, but it has **no confirmatory
scientific answer**. Mini values must not be used for tuning or paper claims.

`Defer` requests a hypothetical fallback, tracker, or additional sensor. It
does not suppress an object as absent and RQ5 does not evaluate fallback
outcomes, missed objects, sensor fusion, or vehicle safety. See
[PROTOCOL.md](PROTOCOL.md) for the frozen claim boundary, hypotheses, feature
groups, controls, correction family, success rule, and prohibited analyses.

### Reused environment, data, model, and cache

RQ5 reuses `experiments2.0/.venv`, the audited Kaggle BDD100K v2 copy, frozen
5,600/2,400/8/1,992 partitions, pinned GroundingDINO Swin-T checkpoint, and
exact local BERT revision. It calls the shared compatibility gate before any
detector work and computes the cache identity with `shared_identity(config)`;
the fingerprint is never hardcoded.

All required tensors already exist in neutral schema v1 under
`data/derived/groundingdino_mc_v1/canonical/<fingerprint>`. The operational
policy uses the first two MC DropPath passes; five and ten are sensitivities
sliced from the same arrays. RQ5 does not read RQ1--RQ4 models or outputs and
does not duplicate the canonical 11 GPU passes.

### Frozen decision layer

The detector-confidence error estimate and a logistic fusion of canonical
semantic/geometric/representation uncertainties are isotonic-calibrated on a
sequence-disjoint validation fold. Their equal 0.5/0.5 late fusion receives a
final validation-only calibration. A frozen criticality weight from predicted
class, bottomness, and centrality turns the error probability into expected
decision risk. The policy accepts the largest low-risk prefix selected on a
separate validation fold and defers the rest.

Controls are raw/calibrated confidence, uncertainty alone, criticality alone,
late fusion without criticality, and a flat logistic model with identical raw
feature access. Prefixes 2/5/10, score thresholds 0.05/0.10/0.20/0.30, fixed
subgroups, and capacity counts are reported.

The primary family compares mc02 risk-aware fusion with calibrated confidence
and the flat control for weighted AURC and coverage at weighted risk 0.10.
Four paired sequence-bootstrap comparisons receive Holm correction. Brier
non-inferiority (margin 0.01), p95 decision overhead <= 5 ms, and estimated
mc02 p95 end-to-end latency <= 100 ms are additional gates.

### Exact commands

From the repository root on Windows:

```powershell
$python = ".\experiments2.0\.venv\Scripts\python.exe"
& $python -m pip check
& $python ".\experiments2.0\scripts\verify_environment.py"
& $python ".\experiments2.0\scripts\verify_model.py"
& $python ".\experiments2.0\scripts\run_rq5.py" --mode smoke
& $python ".\experiments2.0\scripts\run_rq5.py" --mode mini
```

Individual stages are:

```powershell
& $python -m rq5.cli validate --config ".\experiments2.0\RQ5\configs\rq5_mini.yaml"
& $python -m rq5.cli extract --config ".\experiments2.0\RQ5\configs\rq5_mini.yaml" --split train --limit 6
& $python -m rq5.cli fit --config ".\experiments2.0\RQ5\configs\rq5_mini.yaml"
& $python -m rq5.cli evaluate --config ".\experiments2.0\RQ5\configs\rq5_mini.yaml"
& $python -m rq5.cli report --config ".\experiments2.0\RQ5\configs\rq5_mini.yaml"
```

After explicit authorization, the confirmatory command is:

```powershell
& $python ".\experiments2.0\scripts\run_rq5.py" --mode full
```

Linux uses `experiments2.0/.venv/bin/python` with the same arguments.

### Diagnostic validation (not paper evidence)

The smoke test completed in 36.5 s and produced exact DataFrame and Parquet
identity across `rq5_smoke_a`/`rq5_smoke_b` (SHA-256
`e06646a182dd0a1e5d59fe4f02e1d4824def5de27071b5ae92214e88bafeaf07`).
The mini reused 6/4/8 canonical shards with `computed=0`, aligned the common
universe exactly with available RQ1--RQ3 consumers, and reused all 18 RQ5
feature shards on a second run. Immutable request receipts and the 13 report
artifacts pass SHA-256 validation.

The mini result is deliberately retained even though it is negative:
risk-aware mc02 fusion did not improve the frozen primary comparisons and did
not pass Brier non-inferiority. Its directly measured decision p95 was about
3.58 ms, but estimated mc02 end-to-end p95 was about 2,497 ms, so the 100 ms
systems gate failed. The deterministic-only estimate also exceeded 100 ms.
These eight diagnostic images cannot establish effect size, significance, or
confirmatory feasibility and did not trigger retuning.

Ten-pass GPU blocks are measured; mc02/mc05 values assume linear stochastic
pass cost and are labelled estimates. Peak mini GPU memory was 2,030,990,848
bytes (about 1.89 GiB). The existing project estimate remains 25--35 hours for
complete shared extraction on this laptop, roughly 17.3 GiB of shared cache,
and at least 55 GB free disk.

### Outputs, resumption, and integrity

Mini artifacts are isolated under `RQ5/outputs/mini_e2e` and
`RQ5/models/mini_e2e`; confirmatory artifacts use `RQ5/outputs` and
`RQ5/models`. Outputs include split feature/image-summary Parquet files,
per-image shards and sidecars, metadata, fitted models and model index,
calibrated long-form predictions, sequence bootstrap, metrics JSON, CSV
tables, PNG/PDF figures, captions, and `report_manifest.json`. The notebook
only reads completed artifacts.

Re-running validates every shared and RQ5 shard hash. Valid feature shards are
reused; missing, stale, or corrupt downstream shards are rebuilt atomically.
Each shared request gets a new immutable receipt while canonical NPZ/JSON
sidecars remain unchanged when `shared_shards_computed == 0`.

## Español

### Alcance y estado

RQ5 pregunta cómo integrar incertidumbre epistémica fusionada y salidas
calibradas del detector en una capa `aceptar`/`diferir` bajo restricciones
ADAS de latencia. La implementación superó smoke GPU repetible de dos imágenes
y mini diagnóstico 6/4/8. Está técnicamente lista para una corrida
confirmatoria explícitamente autorizada, pero **no existe respuesta científica
confirmatoria**. Mini no debe usarse para tuning ni claims del paper.

`Diferir` solicita un fallback, tracker o sensor hipotético; no declara que el
objeto esté ausente. RQ5 no evalúa el fallback, objetos omitidos, fusión de
sensores ni seguridad vehicular. [PROTOCOL.md](PROTOCOL.md) congela el claim,
hipótesis, features, controles, corrección, éxito y análisis prohibidos.

### Entorno, datos, modelo y caché reutilizados

RQ5 reutiliza `.venv`, BDD100K v2 auditado, particiones
5.600/2.400/8/1.992 y GroundingDINO/BERT fijados. Valida compatibilidad antes
del detector y obtiene la identidad mediante `shared_identity(config)`.

Schema v1 ya contiene todos los tensores. La política usa el prefijo de dos
pasadas DropPath; cinco/diez son sensibilidades del mismo shard. RQ5 no lee
modelos/salidas de RQ1--RQ4 ni duplica las 11 pasadas canónicas.

### Capa de decisión congelada

La confianza y una logística que fusiona incertidumbre semántica, geométrica y
de representación se calibran con isotónica en validation separada por
secuencia. La fusión tardía 0,5/0,5 se calibra nuevamente. Un peso de criticidad
fijo por clase/caja predichas produce riesgo esperado. Otro fold de validation
elige la máxima cobertura con riesgo ponderado <= 0,10.

Los controles son confianza raw/calibrada, incertidumbre sola, criticidad sola,
fusión sin criticidad y logística plana con los mismos features. Se reportan
prefijos 2/5/10, thresholds 0,05/0,10/0,20/0,30, subgrupos y capacidad.

La familia primaria compara fusión mc02 con confianza calibrada y control
plano en AURC ponderado y cobertura a riesgo 0,10, con cuatro bootstrap por
secuencia y Holm. También se exigen no inferioridad Brier 0,01, overhead p95
<= 5 ms y latencia mc02 p95 estimada <= 100 ms.

### Comandos exactos

Desde la raíz en Windows:

```powershell
$python = ".\experiments2.0\.venv\Scripts\python.exe"
& $python -m pip check
& $python ".\experiments2.0\scripts\verify_environment.py"
& $python ".\experiments2.0\scripts\verify_model.py"
& $python ".\experiments2.0\scripts\run_rq5.py" --mode smoke
& $python ".\experiments2.0\scripts\run_rq5.py" --mode mini
```

Etapas individuales:

```powershell
& $python -m rq5.cli validate --config ".\experiments2.0\RQ5\configs\rq5_mini.yaml"
& $python -m rq5.cli extract --config ".\experiments2.0\RQ5\configs\rq5_mini.yaml" --split train --limit 6
& $python -m rq5.cli fit --config ".\experiments2.0\RQ5\configs\rq5_mini.yaml"
& $python -m rq5.cli evaluate --config ".\experiments2.0\RQ5\configs\rq5_mini.yaml"
& $python -m rq5.cli report --config ".\experiments2.0\RQ5\configs\rq5_mini.yaml"
```

Tras autorización explícita:

```powershell
& $python ".\experiments2.0\scripts\run_rq5.py" --mode full
```

Linux usa `experiments2.0/.venv/bin/python` con los mismos argumentos.

### Validación diagnóstica (no es evidencia)

Smoke tardó 36,5 s y produjo identidad exacta de DataFrame/Parquet en dos
namespaces (SHA-256
`e06646a182dd0a1e5d59fe4f02e1d4824def5de27071b5ae92214e88bafeaf07`).
Mini reutilizó 6/4/8 shards canónicos con `computed=0`, alineó exactamente el
universo común disponible con RQ1--RQ3 y reutilizó los 18 feature shards en la
segunda corrida. Recibos inmutables y 13 artefactos del reporte validaron hash.

Se conserva el resultado mini negativo: mc02 no mejoró las comparaciones
primarias ni superó no inferioridad Brier. El overhead p95 medido fue ~3,58 ms,
pero la latencia end-to-end mc02 estimada p95 fue ~2.497 ms y falló el gate de
100 ms; incluso la estimación determinista lo excedió. Ocho imágenes no
establecen efecto, significancia ni factibilidad confirmatoria y no provocaron
tuning.

El bloque GPU de diez pasadas sí es medido; mc02/mc05 son estimaciones lineales
etiquetadas. El pico mini fue 2.030.990.848 bytes (~1,89 GiB). Se mantiene la
planificación de 25--35 horas, ~17,3 GiB de caché y al menos 55 GB libres.

### Salidas, reanudación e integridad

Mini queda en `RQ5/outputs/mini_e2e` y `RQ5/models/mini_e2e`; full usa las
raíces RQ5. Se generan Parquet de features/resúmenes, shards/sidecars, metadata,
modelos e índice, predicciones calibradas, bootstrap, métricas, tablas,
figuras PNG/PDF y `report_manifest.json`. El notebook solo lee artefactos.

Al reanudar se validan hashes compartidos y RQ5. Shards íntegros se reutilizan;
ausentes, stale o corruptos se regeneran atómicamente. Cada solicitud produce
un recibo inmutable y la caché canónica permanece sin cambios con
`shared_shards_computed == 0`.
