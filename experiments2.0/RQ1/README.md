# RQ1 — Multi-representation uncertainty fusion / Fusión de incertidumbre multi-representación

[English](#english) | [Español](#español)

## English

The `.py` implementation is the scientific source of truth. The notebook only
reads completed artifacts.

### One-time preparation

From the repository root:

```powershell
.\experiments2.0\setup_env.ps1
$python = ".\experiments2.0\.venv\Scripts\python.exe"
& $python ".\experiments2.0\scripts\prepare_data.py"
& $python ".\experiments2.0\scripts\prepare_model.py"
```

The dataset is pinned to Kaggle version 2. Data preparation must finish with
`"status": "pass"`; RQ1 must not run after a failed audit. The first download
is approximately 9.9 GB, and later executions reuse verified local content.
Canonical GroundingDINO inference is stored as neutral, versioned shards under
`data/derived/groundingdino_mc_v1`; RQ2 and later compatible RQs validate and
reuse those shards without reading any RQ1 result.

### Staged execution

```powershell
$rq1 = ".\experiments2.0\.venv\Scripts\rq1.exe"
$config = ".\experiments2.0\RQ1\configs\rq1.yaml"

& $rq1 manifest --config $config
& $rq1 smoke --config $config --images 2
& $rq1 extract --config $config --split train
& $rq1 extract --config $config --split validation
& $rq1 fit --config $config
& $rq1 robustness --config $config
& $rq1 extract --config $config --split test
& $rq1 evaluate --config $config
& $rq1 report --config $config
```

Confirmatory test extraction and inspection occur only after the variables and
fusion procedure have been frozen with train/validation. After `report`, open
`notebooks/01_results.ipynb` with the `.venv` kernel; it only regenerates tables
and figures from Parquet/JSON artifacts.

### Automated checks and full run

Validate repeatability on two images:

```powershell
& $python ".\experiments2.0\scripts\run_rq1.py" --mode smoke
```

Exercise the complete fitting, calibration, evaluation and reporting workflow
on the isolated diagnostic partition:

```powershell
& $python ".\experiments2.0\scripts\run_rq1.py" `
    --config ".\experiments2.0\RQ1\configs\rq1_mini.yaml" `
    --mode mini
```

Mini artifacts are stored under `RQ1/outputs/mini_e2e` and
`RQ1/models/mini_e2e`. The eight diagnostic images are permanently excluded
by group from the confirmatory test, so running or inspecting mini mode does
not contaminate the paper results.

The workflow fits only detections at the frozen 0.20 operating threshold,
uses sequence-disjoint validation folds for hyperparameter selection and
probability calibration, and evaluates a nonlinear random-forest comparator.
Its validation-only robustness stage tests independent MC seeds, association
IoU, synonymous prompts and controlled corruptions. Mini output is always
labelled diagnostic and cannot set `rq1_answer_supported=true`.

After freezing the protocol, regenerate the complete RQ1 evidence with:

```powershell
& $python ".\experiments2.0\scripts\run_rq1.py" --mode full
```

The recommended clean-machine entry point is documented in
`../REPRODUCIBILITY.md`; the shared schema and RQ3+ contract are documented in
`../SHARED_EXTRACTION.md`.

---

## Español

La implementación `.py` es la fuente científica de verdad. El notebook solo
lee artefactos terminados.

### Preparación única

Desde la raíz del repositorio:

```powershell
.\experiments2.0\setup_env.ps1
$python = ".\experiments2.0\.venv\Scripts\python.exe"
& $python ".\experiments2.0\scripts\prepare_data.py"
& $python ".\experiments2.0\scripts\prepare_model.py"
```

El dataset está fijado a Kaggle versión 2. La preparación debe terminar con
`"status": "pass"`; RQ1 no debe ejecutarse con una auditoría fallida. La
primera descarga ocupa aproximadamente 9.9 GB y las ejecuciones posteriores
reutilizan el contenido local verificado.
La inferencia canónica de GroundingDINO se guarda como shards neutrales y
versionados en `data/derived/groundingdino_mc_v1`; RQ2 y las RQ futuras
compatibles los validan y reutilizan sin leer resultados de RQ1.

### Ejecución por etapas

```powershell
$rq1 = ".\experiments2.0\.venv\Scripts\rq1.exe"
$config = ".\experiments2.0\RQ1\configs\rq1.yaml"

& $rq1 manifest --config $config
& $rq1 smoke --config $config --images 2
& $rq1 extract --config $config --split train
& $rq1 extract --config $config --split validation
& $rq1 fit --config $config
& $rq1 robustness --config $config
& $rq1 extract --config $config --split test
& $rq1 evaluate --config $config
& $rq1 report --config $config
```

El test confirmatorio no se extrae ni inspecciona hasta congelar las variables
y el método de fusión con train/validation. Después de `report`, abra
`notebooks/01_results.ipynb` con el kernel de `.venv`; solo regenera tablas y
figuras a partir de artefactos Parquet/JSON.

### Comprobaciones automáticas y corrida completa

Valide la repetibilidad con dos imágenes:

```powershell
& $python ".\experiments2.0\scripts\run_rq1.py" --mode smoke
```

Compruebe el ciclo completo de ajuste, calibración, evaluación y reporte sobre
la partición diagnóstica aislada:

```powershell
& $python ".\experiments2.0\scripts\run_rq1.py" `
    --config ".\experiments2.0\RQ1\configs\rq1_mini.yaml" `
    --mode mini
```

Los artefactos mini se escriben en `RQ1/outputs/mini_e2e` y
`RQ1/models/mini_e2e`. Las ocho imágenes diagnósticas están excluidas
permanentemente por grupo del test confirmatorio; ejecutar o inspeccionar el
modo mini no contamina los resultados del paper.

El flujo ajusta solo detecciones en el umbral operativo congelado 0.20, usa
folds de validación separados por secuencia para seleccionar hiperparámetros y
calibrar probabilidades, y evalúa un comparador Random Forest no lineal. La
robustez, limitada a validación, prueba semillas MC independientes, IoU de
asociación, prompts sinónimos y corrupciones controladas. La salida mini
siempre se marca como diagnóstica y no puede establecer
`rq1_answer_supported=true`.

Después de congelar el protocolo, regenere toda la evidencia de RQ1 con:

```powershell
& $python ".\experiments2.0\scripts\run_rq1.py" --mode full
```

El punto de entrada recomendado desde una máquina limpia está documentado en
`../REPRODUCIBILITY.md`; el esquema compartido y el contrato para RQ3+ están en
`../SHARED_EXTRACTION.md`.
