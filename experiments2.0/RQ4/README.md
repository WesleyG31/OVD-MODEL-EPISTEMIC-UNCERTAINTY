# RQ4 multi-level calibration / Calibración multinivel RQ4

## English

### Scope and status

RQ4 asks: *To what extent does multi-level post-hoc calibration--combining
class-level, localization-level, and uncertainty-level calibration--enhance
detection reliability under domain shifts?*

The implementation is technically validated through a two-image repeatability
smoke test and a diagnostic 6/4/8 mini end-to-end run. It is ready to start the
authorized full run, but it has **no confirmatory scientific answer yet**.
Mini values must not be copied into the paper or used for tuning. The frozen
design and prohibited analyses are in [PROTOCOL.md](PROTOCOL.md).

“Domain shift” is deliberately limited to prespecified BDD100K covariate
strata in time of day, weather and scene relative to `daytime + clear + city
street`. This is not evidence for an external dataset, sensor or geography.

### Reused environment, data and detector

RQ4 reuses the existing `experiments2.0/.venv`, audited Kaggle BDD100K v2
copy, frozen 5,600/2,400/8/1,992 split, pinned GroundingDINO Swin-T checkpoint
and exact local BERT revision. Do not create another environment or download
these assets again when their audits pass.

RQ4 consumes the neutral schema-v1 cache at
`data/derived/groundingdino_mc_v1/canonical/<fingerprint>`. It validates
compatibility before extraction and computes the fingerprint with
`shared_identity(config)`; no fingerprint is hardcoded. The required tensors
already exist in v1, so RQ4 does not change detector inference or execute a
second canonical set of 11 GPU passes.

### Frozen methods

All component fitting, hyperparameter selection and isotonic calibration are
restricted to the frozen source domain `daytime + clear + city street`.
Shifted development rows are never used to fit or select a model. The
confirmatory target is the prespecified shifted subset only.

Three logistic component models estimate class correctness, class-agnostic
localization quality and detection correctness from epistemic features. Class
calibration uses score plus a train-only categorical encoding; rare and unseen
classes fall back to the global score effect. The primary score is one minus
the product of these three probabilities. Controls include raw/calibrated
confidence, every single-level and pairwise ablation, and a flat logistic model
with the same raw features and comparable parameter capacity. Except raw
confidence, all methods receive the same final isotonic mapping fit on a
sequence-disjoint source-validation calibration fold.

The confirmatory family compares multilevel calibration with calibrated
confidence and the flat control on shifted detections for Brier, NLL and AURC.
Six one-sided paired sequence-bootstrap comparisons receive Holm correction.

### Exact commands

From the repository root:

```powershell
$python = ".\experiments2.0\.venv\Scripts\python.exe"
& $python -m pip check
& $python ".\experiments2.0\scripts\verify_environment.py"
& $python ".\experiments2.0\scripts\verify_model.py"
& $python ".\experiments2.0\scripts\run_rq4.py" --mode smoke
& $python ".\experiments2.0\scripts\run_rq4.py" --mode mini
```

Run individual stages with:

```powershell
& $python -m rq4.cli validate --config ".\experiments2.0\RQ4\configs\rq4_mini.yaml"
& $python -m rq4.cli extract --config ".\experiments2.0\RQ4\configs\rq4_mini.yaml" --split train --limit 6
& $python -m rq4.cli fit --config ".\experiments2.0\RQ4\configs\rq4_mini.yaml"
& $python -m rq4.cli evaluate --config ".\experiments2.0\RQ4\configs\rq4_mini.yaml"
& $python -m rq4.cli report --config ".\experiments2.0\RQ4\configs\rq4_mini.yaml"
```

After explicit authorization, the exact confirmatory command is:

```powershell
& $python ".\experiments2.0\scripts\run_rq4.py" --mode full
```

Linux uses `experiments2.0/.venv/bin/python` with the same script arguments.

### Outputs, resumption and integrity

Mini artifacts are isolated under `RQ4/outputs/mini_e2e` and
`RQ4/models/mini_e2e`; confirmatory artifacts use `RQ4/outputs` and
`RQ4/models`. Outputs include split feature/image-summary Parquet files,
per-image shards and sidecars, metadata, fitted models and model index,
calibrated predictions, bootstrap records, metrics JSON, CSV tables, PNG/PDF
figures and `report_manifest.json`. The notebook only reads these completed
artifacts.

Re-running validates every shared and RQ4 shard hash. Valid feature shards are
reused; missing, stale or corrupt shards are recomputed atomically. Model,
evaluation and report artifacts are also reused only when their source,
configuration and input hashes still match. MC=10 sensitivity reuses the
primary ten-pass features and fitted components rather than materializing or
fitting duplicates. Each consumer invocation receives a new immutable request
receipt, while canonical NPZ/JSON shard sidecars remain unchanged when
`shared_shards_computed == 0`.

The full RQ4 request is pruned by frozen metadata to 519 source-train, 243
source-validation and 1,992 confirmatory-test images: 2,754 rather than 9,992
images. At the measured mini throughput, a cold first run is conservatively
expected to require roughly 4--8 GPU hours plus CPU evaluation; this is an
estimate, not a deadline. Hash-validated per-image resumption preserves all
completed work after interruption. Reserve at least 55 GB for the environment,
dataset, model, cache and outputs. The diagnostic peak was about 1.9 GiB VRAM.

CUDA warns that a GroundingDINO `cumsum` kernel is not guaranteed bitwise
deterministic across GPU architectures. Exact equality was verified on this
same software/hardware stack; cross-GPU binary identity is not promised.

## Español

### Alcance y estado

RQ4 pregunta: *¿En qué medida la calibración post-hoc multinivel —combinando
calibración a nivel de clase, localización e incertidumbre— mejora la
fiabilidad de las detecciones bajo cambios de dominio?*

La implementación está validada técnicamente mediante smoke repetible de dos
imágenes y mini end-to-end diagnóstico 6/4/8. Está lista para iniciar el full
autorizado, pero **todavía no existe una respuesta científica confirmatoria**.
Los valores mini no deben copiarse al paper ni usarse para tuning. El diseño y
los análisis prohibidos están congelados en [PROTOCOL.md](PROTOCOL.md).

“Cambio de dominio” se limita a estratos covariables BDD100K preespecificados
de hora, clima y escena respecto a `daytime + clear + city street`. No es
evidencia para otro dataset, sensor o geografía.

### Entorno, datos y detector reutilizados

RQ4 reutiliza `experiments2.0/.venv`, la copia auditada Kaggle BDD100K v2, la
partición 5.600/2.400/8/1.992 y GroundingDINO/BERT fijados. No se debe crear
otro entorno ni descargar otra vez estos activos si sus auditorías pasan.

RQ4 consume la caché neutral schema-v1 en
`data/derived/groundingdino_mc_v1/canonical/<fingerprint>`. Valida
compatibilidad y obtiene el fingerprint con `shared_identity(config)`; no lo
hardcodea. Los tensores necesarios ya existen en v1, por lo que RQ4 no cambia
la inferencia ni repite las 11 pasadas GPU canónicas.

### Métodos congelados

El ajuste de componentes, la selección de hiperparámetros y la calibración
isotónica se restringen al dominio fuente congelado `daytime + clear + city
street`. Ninguna fila desplazada de desarrollo se usa para ajustar o elegir
modelos; el objetivo confirmatorio es únicamente el subconjunto desplazado.

Tres modelos logísticos estiman corrección de clase, calidad de localización
sin clase y corrección de detección desde incertidumbre epistémica. La
calibración de clase usa score y una codificación categórica aprendida sólo en
train; las clases raras o no vistas usan el efecto global del score. El score
primario es uno menos el producto de esas probabilidades. Los controles son
confianza raw/calibrada, niveles individuales, ablaciones por pares y una
logística plana con los mismos features y capacidad comparable. Salvo la
confianza raw, todos reciben la misma isotónica final ajustada en un fold
fuente de calibración separado por secuencia.

La familia confirmatoria compara multinivel con confianza calibrada y el
control plano en Brier, NLL y AURC sobre detecciones desplazadas. Las seis
comparaciones bootstrap pareadas reciben Holm.

### Comandos exactos

Desde la raíz:

```powershell
$python = ".\experiments2.0\.venv\Scripts\python.exe"
& $python -m pip check
& $python ".\experiments2.0\scripts\verify_environment.py"
& $python ".\experiments2.0\scripts\verify_model.py"
& $python ".\experiments2.0\scripts\run_rq4.py" --mode smoke
& $python ".\experiments2.0\scripts\run_rq4.py" --mode mini
```

Las etapas individuales son:

```powershell
& $python -m rq4.cli validate --config ".\experiments2.0\RQ4\configs\rq4_mini.yaml"
& $python -m rq4.cli extract --config ".\experiments2.0\RQ4\configs\rq4_mini.yaml" --split train --limit 6
& $python -m rq4.cli fit --config ".\experiments2.0\RQ4\configs\rq4_mini.yaml"
& $python -m rq4.cli evaluate --config ".\experiments2.0\RQ4\configs\rq4_mini.yaml"
& $python -m rq4.cli report --config ".\experiments2.0\RQ4\configs\rq4_mini.yaml"
```

Tras autorización explícita, el comando confirmatorio exacto es:

```powershell
& $python ".\experiments2.0\scripts\run_rq4.py" --mode full
```

Linux usa `experiments2.0/.venv/bin/python` con los mismos argumentos.

### Salidas, reanudación e integridad

Mini queda aislado en `RQ4/outputs/mini_e2e` y `RQ4/models/mini_e2e`; full usa
`RQ4/outputs` y `RQ4/models`. Se generan Parquet de features/resúmenes, shards
y sidecars por imagen, metadata, modelos e índice, predicciones calibradas,
bootstrap, métricas JSON, tablas CSV, figuras PNG/PDF y
`report_manifest.json`. El notebook solo lee artefactos terminados.

Al repetir se validan hashes compartidos y RQ4. Se reutilizan shards válidos y
se recalculan atómicamente los ausentes, stale o corruptos. Modelos, evaluación
y reporte sólo se reutilizan cuando siguen coincidiendo sus hashes de código,
configuración e inputs. La sensibilidad MC=10 reutiliza features y componentes
primarios de diez pasadas. Cada invocación recibe un recibo inmutable nuevo;
los NPZ/JSON canónicos permanecen sin cambios cuando
`shared_shards_computed == 0`.

El full RQ4 se poda por metadatos congelados a 519 imágenes source-train, 243
source-validation y 1.992 confirmatory-test: 2.754 en vez de 9.992 imágenes.
Con el rendimiento mini medido, un primer run sin caché se estima de forma
conservadora en unas 4--8 horas GPU más la evaluación CPU; es una estimación,
no un plazo garantizado. La reanudación por imagen preserva todo trabajo
completo tras una interrupción. Reserve al menos 55 GB. El pico diagnóstico
fue aproximadamente 1,9 GiB VRAM.

CUDA advierte que un kernel `cumsum` no garantiza identidad binaria entre
arquitecturas GPU. Se verificó igualdad exacta en el mismo stack; no se promete
identidad binaria entre GPUs.
