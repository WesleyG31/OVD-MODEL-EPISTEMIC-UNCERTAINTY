# RQ3 — Localization-aware confidence fusion / Fusión de confianza sensible a localización

[English](#english) | [Español](#español)

## English

RQ3 asks: **How does fusing classification confidence with spatial
localization quality improve the reliability, ranking, and calibration of
open-vocabulary detections in safety-critical driving scenarios?** The Python
package in `src/rq3` is the scientific source of truth. The notebook only reads
completed artifacts.

### Reuse and scope

RQ3 reuses the existing `experiments2.0/.venv`, audited BDD100K data, pinned
GroundingDINO/BERT files, frozen group-disjoint manifests and neutral
`groundingdino_mc_v1` shards. It calls the shared compatibility gate before
requesting any shard and does not read RQ1/RQ2 models, outputs or results.

Schema v1 is sufficient: classification scores, deterministic boxes,
associated MC boxes/presence and decoder reference dynamics produce all RQ3
features. Ground-truth matching, class-agnostic localization targets, fitting,
calibration and evaluation remain under `RQ3/`; no label enters the shared
cache.

The primary method multiplies raw classification confidence by a learned
spatial probability of IoU >= 0.50. It is compared with confidence alone and a
non-spatial control having the same eight-feature logistic capacity. Fixed
spatial agreement, localization-only, equal fusion and direct learned fusion
are ablations. See `PROTOCOL.md` for the frozen six-comparison Holm family and
success rule.

### Verify the reused environment

From the repository root on Windows:

```powershell
$python = ".\experiments2.0\.venv\Scripts\python.exe"
& $python -m pip check
& $python ".\experiments2.0\scripts\verify_environment.py"
& $python ".\experiments2.0\scripts\verify_model.py"
```

Do not recreate the environment, redownload audited files or change the
PyTorch GroundingDINO backend.

### Smoke, mini and full

The smoke command uses two images in two independent diagnostic cache
namespaces and requires exact DataFrame and Parquet equality:

```powershell
& $python ".\experiments2.0\scripts\run_rq3.py" --mode smoke
```

The mini command uses 6 train, 4 validation and all 8 diagnostic-test images.
It exercises shared reuse, feature shards, fitting, disjoint selection and
calibration, bootstrap, COCO re-ranking, tables, PNG/PDF figures and hash
validation:

```powershell
& $python ".\experiments2.0\scripts\run_rq3.py" --mode mini
```

Mini artifacts are isolated under `RQ3/outputs/mini_e2e` and
`RQ3/models/mini_e2e`. They are diagnostics, never paper evidence.

The confirmatory command is implemented but must only be launched with explicit
authorization after reviewing mini integrity:

```powershell
& $python ".\experiments2.0\scripts\run_rq3.py" --mode full
```

On Linux use `experiments2.0/.venv/bin/python` with the same arguments.

### Resume, outputs and resources

RQ3 validates each neutral NPZ/JSON shard and writes separate atomic
detection/image-summary Parquet shards. Every sidecar binds image ID, source
hash, schema, materialization fingerprint and shared/feature/summary SHA-256.
Rerunning a command reuses valid shards and recomputes only missing, stale or
corrupt RQ3 artifacts.

Outputs include train/validation/test features and metadata, image summaries,
fitted models and `model_index.json`, calibrated predictions, clustered
bootstrap Parquet, `metrics.json`, CSV tables, PNG/PDF figures and
`report_manifest.json`. The shared cache projects to about 17.3 GiB from the
verified mini mean; reserve at least 55 GB. The previously observed mini GPU
peak is about 1.9 GiB. A full shared extraction on this laptop is estimated at
25–35 hours, although RQ3 adds no detector passes when canonical shards exist.

### Interpretation limits

The eight-image mini cannot answer RQ3. Only the untouched 1,992-image
confirmatory run can evaluate the frozen hypotheses, and any mixed or negative
outcome must remain reported without test retuning. The method is
detection-conditioned, cannot recover missed objects, uses one detector and
domain, and estimates localization with post-hoc proxies rather than a jointly
trained IoU head.

---

## Español

RQ3 pregunta: **¿Cómo mejora la fiabilidad, el ranking y la calibración de las
detecciones open-vocabulary en escenarios críticos de conducción la fusión de
confianza de clasificación con calidad de localización espacial?** El paquete
Python en `src/rq3` es la fuente científica de verdad. El notebook solo lee
artefactos terminados.

### Reutilización y alcance

RQ3 reutiliza `experiments2.0/.venv`, los datos BDD100K auditados, los archivos
GroundingDINO/BERT fijados, los manifiestos congelados sin grupos compartidos y
los shards neutrales `groundingdino_mc_v1`. Ejecuta la validación de
compatibilidad antes de solicitar shards y no lee modelos, outputs ni
resultados de RQ1/RQ2.

El schema v1 es suficiente: scores de clasificación, cajas deterministas,
cajas/presencia MC asociadas y dinámica de referencias del decoder generan
todas las características RQ3. El matching ground truth, targets de
localización sin clase, ajuste, calibración y evaluación permanecen bajo
`RQ3/`; ninguna etiqueta entra en la caché compartida.

El método primario multiplica confianza raw por una probabilidad espacial
aprendida de IoU >= 0.50. Se compara con confianza sola y con un control no
espacial de la misma capacidad logística y ocho features. Acuerdo espacial
fijo, localización sola, fusión igual y fusión directa son ablaciones.
`PROTOCOL.md` congela la familia Holm de seis comparaciones y la regla de éxito.

### Verificar el entorno reutilizado

Desde la raíz del repositorio en Windows:

```powershell
$python = ".\experiments2.0\.venv\Scripts\python.exe"
& $python -m pip check
& $python ".\experiments2.0\scripts\verify_environment.py"
& $python ".\experiments2.0\scripts\verify_model.py"
```

No recree el entorno, no descargue de nuevo archivos auditados ni cambie el
backend PyTorch de GroundingDINO.

### Smoke, mini y full

Smoke usa dos imágenes en dos namespaces diagnósticos independientes y exige
igualdad exacta de DataFrame y Parquet:

```powershell
& $python ".\experiments2.0\scripts\run_rq3.py" --mode smoke
```

Mini usa 6 imágenes train, 4 validation y las 8 de diagnostic test. Comprueba
reutilización compartida, shards, ajuste, selección/calibración disjuntas,
bootstrap, re-ranking COCO, tablas, figuras PNG/PDF y hashes:

```powershell
& $python ".\experiments2.0\scripts\run_rq3.py" --mode mini
```

Los artefactos mini quedan aislados en `RQ3/outputs/mini_e2e` y
`RQ3/models/mini_e2e`. Son diagnósticos, nunca evidencia del paper.

El comando confirmatorio está implementado, pero solo debe iniciarse con
autorización explícita tras revisar la integridad mini:

```powershell
& $python ".\experiments2.0\scripts\run_rq3.py" --mode full
```

En Linux use `experiments2.0/.venv/bin/python` con los mismos argumentos.

### Reanudación, outputs y recursos

RQ3 valida cada shard neutral NPZ/JSON y escribe shards Parquet atómicos propios
de detecciones/resúmenes. Cada sidecar vincula ID de imagen, hash de fuente,
schema, fingerprint y SHA-256 compartido/de features/de resumen. Repetir un
comando reutiliza shards válidos y solo recalcula artefactos RQ3 ausentes,
stale o corruptos.

Las salidas incluyen features y metadata de train/validation/test, resúmenes,
modelos e `model_index.json`, predicciones calibradas, bootstrap Parquet,
`metrics.json`, tablas CSV, figuras PNG/PDF y `report_manifest.json`. La caché
compartida se proyecta a unos 17.3 GiB; reserve al menos 55 GB. El pico mini GPU
observado previamente es ~1.9 GiB. La extracción compartida full se estima en
25–35 horas en este portátil, aunque RQ3 no añade pasadas si los shards
canónicos ya existen.

### Límites de interpretación

El mini de ocho imágenes no responde RQ3. Solo la corrida confirmatoria intacta
de 1.992 imágenes puede evaluar las hipótesis congeladas, y todo resultado
mixto o negativo debe conservarse sin retuning de test. El método está
condicionado a detecciones, no recupera objetos omitidos, usa un detector y
dominio, y estima localización con proxies post-hoc en vez de un head IoU
entrenado conjuntamente.

