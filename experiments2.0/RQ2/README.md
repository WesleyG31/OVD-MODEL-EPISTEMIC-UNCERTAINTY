# RQ2 — Deterministic/stochastic uncertainty fusion / Fusión de incertidumbre determinista/estocástica

[English](#english) | [Español](#español)

## English

RQ2 asks: **How does fusing deterministic and stochastic uncertainty
estimators improve reliability in open-vocabulary object detection?** The
scientific source of truth is the typed Python package in `src/rq2`; the
notebook only reads completed JSON/Parquet/CSV artifacts.

### Reused and RQ2-specific components

RQ2 reuses the existing `experiments2.0/.venv`, audited BDD100K files, pinned
GroundingDINO checkpoint, local BERT snapshot, frozen manifests, and the shared
detector/data/matching/metrics/provenance code in `src/adas_ovd`. It also
validates and reuses the label-free detector shards in
`data/derived/groundingdino_mc_v1`; it does not read `RQ1/models`,
`RQ1/outputs`, or RQ1 results. Deterministic/stochastic feature materialization,
fusion models, statistical tests, reports, models and outputs are isolated
under `RQ2/`.

No additional dependency is required. Do not recreate the environment or
redownload data/models. To verify the reused stack:

```powershell
$python = ".\experiments2.0\.venv\Scripts\python.exe"
& $python -m pip check
& $python ".\experiments2.0\scripts\verify_environment.py"
& $python ".\experiments2.0\scripts\verify_model.py"
```

### Smoke, mini and full commands

The two-image GPU smoke test performs two independent extractions and requires
an exact DataFrame match:

```powershell
& $python ".\experiments2.0\scripts\run_rq2.py" --mode smoke
```

The isolated mini run uses 6 train images, 4 validation images and all 8
diagnostic images. It exercises extraction, estimator fitting, calibration,
COCO evaluation, clustered bootstrap, multiplicity adjustment and reporting:

```powershell
& $python ".\experiments2.0\scripts\run_rq2.py" --mode mini
```

Mini outputs go only to `RQ2/outputs/mini_e2e` and models to
`RQ2/models/mini_e2e`. They are technical diagnostics, not paper evidence.

The confirmatory command is intentionally not run during development:

```powershell
& $python ".\experiments2.0\scripts\run_rq2.py" --mode full
```

On Linux, use `experiments2.0/.venv/bin/python` with the same script and
arguments. The top-level `reproduce.ps1 -Mode full` and `reproduce.sh full`
run RQ1 followed by RQ2 after shared preparation.

### Resume and artifacts

Each image has one neutral atomic NPZ/JSON inference shard shared with RQ1 and
separate RQ2 detection/image-summary feature shards. A rerun validates source,
configuration, checkpoint, manifest, schema and SHA-256, reuses valid shards
and recomputes missing or corrupt ones. When RQ1 has completed canonical
inference, RQ2 performs no additional detector passes. Combined Parquet
artifacts and fitted models are also SHA-256 validated before use.

Expected outputs include train/validation/test features and metadata, image
timings, estimator models and index, calibrated prediction Parquet, bootstrap
Parquet, `metrics.json`, metric/parameter CSV tables, PNG/PDF figures and
`report_manifest.json`. `notebooks/01_results.ipynb` reads those completed
artifacts only.

### Scientific scope

The primary test compares learned fusion with capacity-matched learned
deterministic-only and stochastic-only estimators for AUROC and AURC using a
paired source-group bootstrap and Holm correction. Confidence, fixed equal
fusion, confidence augmentation and a random forest are controls/sensitivities.
See `PROTOCOL.md` for frozen hypotheses, subgroups and failure criteria.

One detector, checkpoint, prompt vocabulary and dataset cannot establish a
universal OVD result. MC stochastic depth is only an epistemic proxy, decoder
dynamics can reflect refinement, and detection-conditioned uncertainty cannot
score wholly missed objects. A passing mini run establishes technical
readiness only; the RQ is not scientifically answered until the full real
confirmatory run is completed and interpreted without test-set retuning.

---

## Español

RQ2 pregunta: **¿Cómo mejora la fiabilidad de la detección de objetos de
vocabulario abierto la fusión de estimadores de incertidumbre deterministas y
estocásticos?** La fuente científica de verdad es el paquete Python tipado en
`src/rq2`; el notebook solo lee artefactos JSON/Parquet/CSV ya terminados.

### Componentes reutilizados y específicos de RQ2

RQ2 reutiliza `experiments2.0/.venv`, los archivos BDD100K auditados, el
checkpoint fijado de GroundingDINO, el snapshot BERT local, los manifiestos
congelados y el código compartido de detector/datos/matching/métricas/procedencia
en `src/adas_ovd`. También valida y reutiliza los shards del detector sin
etiquetas en `data/derived/groundingdino_mc_v1`. No lee `RQ1/models`,
`RQ1/outputs` ni resultados de RQ1. La materialización de características
deterministas/estocásticas, las fusiones, pruebas estadísticas, reportes,
modelos y salidas permanecen aislados en `RQ2/`.

No se requiere ninguna dependencia adicional. No recree el entorno ni vuelva
a descargar datos/modelos. Para verificar el stack reutilizado:

```powershell
$python = ".\experiments2.0\.venv\Scripts\python.exe"
& $python -m pip check
& $python ".\experiments2.0\scripts\verify_environment.py"
& $python ".\experiments2.0\scripts\verify_model.py"
```

### Comandos smoke, mini y full

El smoke GPU de dos imágenes ejecuta dos extracciones independientes y exige
igualdad exacta de los DataFrames:

```powershell
& $python ".\experiments2.0\scripts\run_rq2.py" --mode smoke
```

La corrida mini aislada usa 6 imágenes de train, 4 de validation y las 8
imágenes diagnósticas. Comprueba extracción, ajuste, calibración, evaluación
COCO, bootstrap agrupado, corrección por multiplicidad y reporte:

```powershell
& $python ".\experiments2.0\scripts\run_rq2.py" --mode mini
```

Las salidas mini se guardan solo en `RQ2/outputs/mini_e2e` y los modelos en
`RQ2/models/mini_e2e`. Son diagnósticos técnicos, no evidencia del paper.

El comando confirmatorio no se ejecuta durante el desarrollo:

```powershell
& $python ".\experiments2.0\scripts\run_rq2.py" --mode full
```

En Linux, use `experiments2.0/.venv/bin/python` con el mismo script y
argumentos. Los comandos superiores `reproduce.ps1 -Mode full` y
`reproduce.sh full` ejecutan RQ1 y después RQ2 tras la preparación compartida.

### Reanudación y artefactos

Cada imagen tiene un shard neutral NPZ/JSON compartido con RQ1 y shards
separados de características/resumen propios de RQ2. Al reanudar se validan
fuente, configuración, checkpoint, manifiesto, esquema y SHA-256; se
reutilizan shards válidos y se recalculan los ausentes o corruptos. Si RQ1 ya
terminó la inferencia canónica, RQ2 no realiza pasadas adicionales del
detector. Los Parquet combinados y modelos ajustados también se validan por
SHA-256 antes de usarse.

Las salidas esperadas incluyen características y metadatos de
train/validation/test, tiempos por imagen, modelos e índice, predicciones
calibradas, bootstrap, `metrics.json`, tablas CSV de métricas/parámetros, figuras PNG/PDF y
`report_manifest.json`. `notebooks/01_results.ipynb` solo lee artefactos
terminados.

### Alcance científico

La prueba primaria compara la fusión aprendida con estimadores aprendidos
determinista-only y stochastic-only de capacidad comparable para AUROC y AURC,
mediante bootstrap pareado por grupo fuente y corrección de Holm. Confianza,
fusión fija igual, aumento con confianza y random forest son controles o
sensibilidades. `PROTOCOL.md` contiene hipótesis, subgrupos y criterios de fallo.

Un solo detector, checkpoint, vocabulario y dataset no permite una conclusión
universal sobre OVD. La profundidad estocástica MC es solo un proxy epistémico,
la dinámica del decoder puede reflejar refinamiento y la incertidumbre
condicionada a detecciones no puntúa objetos completamente omitidos. Un mini
correcto solo demuestra preparación técnica; la RQ no queda respondida hasta
terminar e interpretar la corrida confirmatoria real sin reajustar con el test.
