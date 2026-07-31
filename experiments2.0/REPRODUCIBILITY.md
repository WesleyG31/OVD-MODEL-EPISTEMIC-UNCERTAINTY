# Experiments 2.0 Reproducibility / Reproducibilidad

[English](#english) | [Español](#español)

## English

This guide rebuilds the environment, obtains and audits the data, verifies
GroundingDINO on a GPU and reproduces RQ1--RQ5 from a clean machine. The official
workflow uses Python `venv` and `pip`; Conda and Docker are not required.

The current one-command workflow reproduces all implemented evidence, which
currently consists of RQ1--RQ5. Later questions must be registered in
the top-level orchestrator when their frozen implementations are added.

### 1. Verified data status

The source is pinned to Kaggle dataset version 2:
`solesensei/solesensei_bdd100k`.

<https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k>

The complete local audit finished with `"status": "pass"` and verified:

- 10,000 images, all 1280 x 720;
- 185,526 valid annotations;
- 8,000 development images and a 2,000-image evaluation pool;
- normalization of `motor -> motorcycle` and `bike -> bicycle`;
- SHA-256 for every one of the 10,000 used images;
- zero shared files and zero shared source groups between development and
  evaluation;
- zero shared groups between train and validation;
- all ten expected ADAS categories, including 452 `motorcycle` annotations.

The frozen RQ1 partition is:

| Partition | Images | Purpose |
|---|---:|---|
| train | 5,600 | Fusion fitting |
| validation | 2,400 | Selection and calibration |
| diagnostic test | 8 | Mini technical check; not evidence |
| confirmatory test | 1,992 | Final paper evaluation |

The eight diagnostic images are excluded from the confirmatory test by source
group. Both manifests report zero overlap. Mini results must never be reported
as paper results.

Auditable evidence:

- `artifacts/data_audit.json`: counts, categories and audit status;
- `artifacts/data_provenance.json`: source, hashes and image inventory;
- `artifacts/bdd100k_split_manifest.json`: confirmatory partition;
- `artifacts/bdd100k_diagnostic_manifest.json`: isolated mini partition;
- `artifacts/model_provenance.json`: pinned checkpoint and text encoder.

Data and weights are not committed to Git because of size and licensing. Each
replicator downloads them from their sources and verifies identity locally.

### 2. Requirements

- Windows 10/11 with PowerShell, or Linux with Bash;
- CPython 3.12;
- Internet access for the initial installation and download;
- at least 55 GB of free disk space recommended (plan 17–25 GiB for the shared
  10,000-image inference cache; the 18-shard mini mean projects to 17.3 GiB);
- an NVIDIA GPU and driver compatible with the pinned PyTorch CUDA 11.8 build
  for the full run.

A CPU installation supports unit tests, but it is not an approved or practical
path for complete RQ1--RQ5 detector extraction.

If Kaggle requests authentication, set `KAGGLE_API_TOKEN` or store the token in
`~/.kaggle/access_token`. Never commit tokens to this repository.

### 3. Quick reproduction on Windows

Open PowerShell at the repository root:

```powershell
cd "C:\All_files\Projects\OVD-MODEL-EPISTEMIC-UNCERTAINTY"
```

On a clean machine, run the complete smoke workflow first:

```powershell
.\experiments2.0\reproduce.ps1
```

This command:

1. creates `experiments2.0/.venv`;
2. installs the pinned packages with `pip`;
3. runs the project tests;
4. verifies CUDA and a real deformable-attention GPU operation;
5. downloads, converts and audits BDD100K;
6. downloads and verifies GroundingDINO and BERT;
7. performs two independent smoke extractions and requires an exact match.

After all smoke checks pass, start the confirmatory run only with explicit
authorization:

```powershell
.\experiments2.0\reproduce.ps1 -Mode full
```

Before FULL, run all five isolated diagnostic workflows with:

```powershell
.\experiments2.0\reproduce.ps1 -Mode mini -SkipSetup
```

If the environment is already prepared and verified, skip package setup:

```powershell
.\experiments2.0\reproduce.ps1 -Mode full -SkipSetup
```

If PowerShell blocks local scripts, enable them only for the current process:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

### 4. Quick reproduction on Linux

From the repository root:

```bash
bash experiments2.0/reproduce.sh smoke
bash experiments2.0/reproduce.sh mini
bash experiments2.0/reproduce.sh full
```

To select another Python 3.12 executable:

```bash
PYTHON_BIN=/path/to/python3.12 bash experiments2.0/setup_env.sh gpu
```

### 5. Mini verification before the long run

Mini mode exercises extraction, group-disjoint selection/calibration,
nonlinear comparison, validation-only robustness, COCO evaluation, clustered
bootstrap and table/figure generation with isolated artifacts:

```powershell
.\experiments2.0\reproduce.ps1 -Mode mini -SkipSetup
```

It uses 6 train images, 4 validation images and all 8 diagnostic images. Its
outputs are written to each RQ's `outputs/mini_e2e` and models to each RQ's
`models/mini_e2e`; they never mix with confirmatory artifacts.

### 6. Staged execution

To inspect every gate separately on Windows:

```powershell
.\experiments2.0\setup_env.ps1 -Target gpu
$python = ".\experiments2.0\.venv\Scripts\python.exe"

& $python ".\experiments2.0\scripts\verify_environment.py"
& $python ".\experiments2.0\scripts\prepare_data.py"
& $python ".\experiments2.0\scripts\prepare_model.py"
& $python ".\experiments2.0\scripts\verify_model.py"
& $python ".\experiments2.0\scripts\run_rq1.py" --mode smoke
& $python ".\experiments2.0\scripts\run_rq2.py" --mode smoke
& $python ".\experiments2.0\scripts\run_rq3.py" --mode smoke
```

`prepare_data.py` must print `"status": "pass"`. Environment verification must
report at least:

```text
environment_type: venv
groundingdino-py: 0.4.0
cuda_available: true
cuda_tensor_operation_passed: true
deformable_attention_device: cuda:0
deformable_attention_test_passed: true
```

An existing offline dataset copy can be supplied as follows:

```powershell
& $python ".\experiments2.0\scripts\prepare_data.py" `
    --source-dir "D:\path\to\dataset"
```

The copy is processed and audited with the same rules; existence alone is not
accepted as proof of validity.

### 7. Resuming and GPU use

Canonical detector extraction writes one atomic compressed shard and JSON
sidecar per image under `data/derived/groundingdino_mc_v1`. RQ1--RQ5
validate its schema, configuration fingerprint, source image, split request
and SHA-256. They then write separate RQ-specific feature shards. If execution
is interrupted, rerun the same `-Mode full` command: every valid shared and
RQ-specific shard is reused, while a missing, stale or corrupt shard is
recomputed.

The canonical path executes one deterministic plus ten stochastic GPU passes
once per image; RQ2, RQ3, RQ4 and RQ5 reuse the same neutral shards and do not repeat
those 11 passes. RQ1's bounded validation-only robustness conditions perform additional
inference because they deliberately alter seeds, prompts, association or
images.

Close games, other model-training jobs and VRAM-heavy applications before a
long run. Do not change configurations, thresholds, manifests or versions
between resumptions.

PyTorch reports that the CUDA `cumsum` used by GroundingDINO does not guarantee
bitwise equality across different GPU architectures. The project pins seeds,
cuDNN, cuBLAS, packages, data and weights. Exact repetition was verified on the
same stack, but the paper must not promise binary identity across GPUs.

### 8. Expected outputs

Neutral inference shards and request manifests are stored in
`experiments2.0/data/derived/groundingdino_mc_v1`. They are inputs shared by
compatible RQs, not RQ1 results.

The confirmatory run writes to `experiments2.0/RQ1/outputs`:

- train, validation and test feature Parquet files with metadata and hashes;
- `metrics.json` and `bootstrap_metrics.parquet`;
- fused and image-safety prediction Parquet files;
- CSV tables for results, coefficients, detector performance, sensitivity,
  category/object-size/environment subgroups, primary inference, calibration,
  feature complementarity, robustness and computational cost;
- AUROC, risk-coverage and reliability figures in PNG/PDF;
- validation-only robustness Parquet/JSON for MC seeds, association IoU,
  prompts and controlled corruptions.

Fitted models are stored in `experiments2.0/RQ1/models`. The `.py` files are
the scientific source of truth; `RQ1/notebooks/01_results.ipynb` only reads
completed artifacts.

RQ2 writes its isolated feature, prediction, bootstrap, metric, CSV,
PNG/PDF and report-manifest artifacts to `experiments2.0/RQ2/outputs`, and its
fitted estimators to `experiments2.0/RQ2/models`. The corresponding mini paths
end in `mini_e2e`. `RQ2/notebooks/01_results.ipynb` is also read-only with
respect to scientific computation.

RQ3 writes localization targets/features, fitted spatial and capacity-control
models, calibrated predictions, paired bootstrap results, COCO re-ranking
tables, CSV/PNG/PDF reports and a hash manifest under `RQ3/outputs` and
`RQ3/models`. Its notebook is also a completed-artifact reader only.

RQ4 writes class/localization/uncertainty calibration targets and features,
component/ablation/capacity-control models, calibrated predictions, paired
sequence bootstrap, domain tables, PNG/PDF figures and a hash manifest under
`RQ4/outputs` and `RQ4/models`. Its notebook is also read-only.

RQ5 writes label-free criticality descriptors, 2/5/10-prefix epistemic
features, calibrated `accept`/`defer` policies, long-form predictions,
sequence bootstrap, latency/quality tables, PNG/PDF figures and a hash
manifest under `RQ5/outputs` and `RQ5/models`. Its notebook only reads
completed artifacts. mc10 GPU latency is measured; mc02/mc05 are explicitly
labelled linear prefix estimates.

### 9. Reuse across RQ1--RQ5 and later questions

RQ2 reuses the verified raw/processed dataset, pinned detector and text
encoder, frozen manifests, shared code under `src/adas_ovd`, and the neutral
versioned detector shards under `data/derived/groundingdino_mc_v1`. Its
feature/model/evaluation implementation remains isolated from RQ1.

RQ-specific outputs remain isolated:

- RQ1 code, fitted models and results stay under `RQ1/`;
- RQ2 code, models and results are created under `RQ2/`;
- RQ3 code, models and results are created under `RQ3/`;
- RQ4 code, models and results are created under `RQ4/`;
- RQ5 code, models and results are created under `RQ5/`;
- RQ2 must not depend implicitly on `RQ1/outputs` or `RQ1/models`;
- each current or future consumer must pass the shared compatibility gate;
  an incompatible tensor requirement creates a coexisting schema version and
  must never silently reinterpret or overwrite v1;
- using an RQ1 fusion/model as an RQ2 input is allowed only when the RQ2
  protocol explicitly defines that dependency;
- freeze RQ2/RQ3/RQ4/RQ5 hypotheses, inputs and metrics before inspecting confirmatory
  labels. Reusing the same benchmark requires disclosure and appropriate
  multiple-comparison interpretation.

Therefore the data may be reused; RQ1's reported answers should not be treated
as generic shared data.

### 10. RQ1--RQ5 completion criteria

A successful command does not automatically make the hypothesis positive.
After the run, audit and interpret:

- internal fusion versus confidence with paired bootstrap;
- AUROC, AUPRC, AURC, Brier, NLL and ECE;
- risk-coverage and coverage at fixed risk;
- semantic, geometric, representation and presence ablations;
- sensitivity to 2/5/10 MC passes and score thresholds;
- time-of-day, weather, scene, category and object-size subgroups;
- independent MC seeds, association IoU, prompt and corruption robustness;
- sequence-clustered Brier/NLL/ECE intervals and ECE-bin sensitivity;
- the frozen AUROC/AURC Holm-corrected success rule and Brier
  non-inferiority;
- synchronized warm-model latency, throughput and peak GPU memory;
- image-level false negatives and any `not_estimable` result.

Do not tune the method using the confirmatory test. A fusion that does not beat
confidence is still a valid result and must be reported honestly.

For RQ2, also audit learned fusion against learned deterministic-only and
stochastic-only estimators for both AUROC and AURC, the four Holm-adjusted
primary comparisons, fixed equal fusion, confidence augmentation, nonlinear
fusion, validation complementarity, category/object-size subgroups and the
synchronized deterministic-versus-MC runtime. A passing technical command does
not establish a positive hypothesis.

For RQ3, audit product fusion against confidence and the capacity-matched
non-spatial product control for AUROC, AURC and Brier in the frozen six-test
Holm family. Also report localization-only/equal/direct ablations, 2/5/10-pass
and score/IoU/ECE-bin sensitivities, COCO AP50/AP75 re-ranking, error taxonomy,
subgroups, calibration, shared-shard reuse and all `not_estimable` outcomes.
Mini output remains diagnostic even when nominal comparisons are favorable.

For RQ4, audit multilevel calibration against calibrated confidence and the
flat same-feature control on the frozen shifted subset for Brier, NLL and AURC
in the six-test Holm family. Report all single/pair ablations, coefficient
capacity, 2/5/10 MC and score-threshold sensitivities, reference/axis/severity
domain strata, ECE bins, risk/coverage, hashes and every `not_estimable`
outcome. Within-BDD shifts must not be described as external validation.

For RQ5, audit mc02 risk-aware late fusion against calibrated confidence and
the flat same-feature control for weighted AURC and coverage at weighted risk
0.10 in the frozen six-test Holm family, including two Brier non-inferiority tests,
operating coverage/risk, criticality and no-criticality controls, 2/5/10 MC
prefixes, score/subgroup sensitivities, measured decision overhead, the
measured-versus-estimated latency distinction, offline budgets, hashes and all
`not_estimable` outcomes. `Defer` must not be described as a safe outcome.

### 11. Troubleshooting

- **`CUDA is not available`**: verify the NVIDIA driver and run
  `scripts/verify_environment.py`; do not continue the full run on CPU.
- **Kaggle/authentication failure**: configure the token and rerun the same
  command. A partial download cannot pass the audit.
- **Hash mismatch**: do not edit the audit. Replace only the identified file
  and rerun the corresponding preparation step.
- **Interrupted run**: rerun `reproduce.ps1 -Mode full`; do not delete
  either RQ's output shards.
- **Mini results exist**: never copy them into the manuscript.

Publication readiness is tracked in `SUBMISSION_READINESS.md`; the frozen
protocols are in `RQ1/PROTOCOL.md`, `RQ2/PROTOCOL.md`, `RQ3/PROTOCOL.md`,
`RQ4/PROTOCOL.md` and `RQ5/PROTOCOL.md`.

---

## Español

Esta guía permite reconstruir el entorno, obtener y auditar los datos,
verificar GroundingDINO sobre GPU y reproducir RQ1--RQ5 desde una máquina limpia.
La ruta oficial usa Python `venv` y `pip`; no requiere Conda ni Docker.

El flujo actual de un solo comando reproduce toda la evidencia implementada,
que actualmente corresponde a RQ1--RQ5. Las preguntas posteriores deberán
registrarse en el orquestador superior al añadir sus implementaciones congeladas.

### 1. Estado de los datos verificados

La fuente está fijada a la versión 2 del dataset de Kaggle
`solesensei/solesensei_bdd100k`:

<https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k>

La auditoría local completa terminó con `"status": "pass"` y comprobó:

- 10,000 imágenes, todas de 1280 x 720;
- 185,526 anotaciones válidas;
- 8,000 imágenes en desarrollo y 2,000 en el pool de evaluación;
- normalización de los alias BDD históricos `motor -> motorcycle` y
  `bike -> bicycle`;
- SHA-256 de cada una de las 10,000 imágenes utilizadas;
- cero archivos y cero grupos fuente compartidos entre desarrollo y
  evaluación;
- cero grupos compartidos entre train y validation;
- las diez categorías ADAS esperadas, incluida `motorcycle` con 452
  anotaciones.

La partición congelada para RQ1 es:

| Partición | Imágenes | Uso |
|---|---:|---|
| train | 5,600 | Ajuste de las fusiones |
| validation | 2,400 | Selección y calibración |
| diagnostic test | 8 | Comprobación técnica mini; no es evidencia |
| confirmatory test | 1,992 | Evaluación final del paper |

Las 8 imágenes diagnósticas están excluidas del test confirmatorio por grupo.
Los manifiestos registran cero solapamiento. Las pruebas mini nunca deben
presentarse como resultados del paper.

Evidencia auditable:

- `artifacts/data_audit.json`: conteos, categorías y resultado de auditoría;
- `artifacts/data_provenance.json`: origen, hashes y archivos de imagen;
- `artifacts/bdd100k_split_manifest.json`: partición confirmatoria;
- `artifacts/bdd100k_diagnostic_manifest.json`: partición mini aislada;
- `artifacts/model_provenance.json`: checkpoint y text encoder fijados.

Los datos y pesos no se incluyen en Git por tamaño y licencia; cada replicador
los obtiene desde sus fuentes y verifica su identidad automáticamente.

### 2. Requisitos

- Windows 10/11 con PowerShell, o Linux con Bash;
- CPython 3.12;
- conexión a Internet para la primera instalación y descarga;
- al menos 55 GB de espacio libre recomendado (planifique 17–25 GiB para la
  caché compartida de 10,000 imágenes; la media de 18 shards mini proyecta
  17.3 GiB);
- para la corrida completa, GPU NVIDIA y driver compatible con PyTorch
  CUDA 11.8.

Una instalación CPU permite ejecutar pruebas unitarias, pero no es una ruta
práctica ni aprobada para la extracción completa del detector de RQ1--RQ5.

Si Kaggle solicita autenticación, configure `KAGGLE_API_TOKEN` o coloque el
token en `~/.kaggle/access_token`. No guarde tokens dentro del repositorio.

### 3. Reproducción rápida en Windows

Abra PowerShell en la raíz del repositorio:

```powershell
cd "C:\All_files\Projects\OVD-MODEL-EPISTEMIC-UNCERTAINTY"
```

En una máquina limpia, ejecute primero el smoke test completo:

```powershell
.\experiments2.0\reproduce.ps1
```

Este único comando:

1. crea `experiments2.0/.venv`;
2. instala las versiones fijadas con `pip`;
3. ejecuta las pruebas del proyecto;
4. comprueba CUDA y una operación real de atención deformable en GPU;
5. descarga, convierte y audita BDD100K;
6. descarga y verifica GroundingDINO y BERT mediante SHA-256/revisión;
7. ejecuta dos extracciones smoke independientes y exige igualdad exacta.

Cuando todos los smoke finalicen correctamente, inicie la corrida
confirmatoria solo con autorización explícita:

```powershell
.\experiments2.0\reproduce.ps1 -Mode full
```

Antes de FULL, ejecute los cinco workflows diagnósticos aislados:

```powershell
.\experiments2.0\reproduce.ps1 -Mode mini -SkipSetup
```

Si el entorno ya fue preparado y verificado, puede evitar reinstalar paquetes:

```powershell
.\experiments2.0\reproduce.ps1 -Mode full -SkipSetup
```

Si PowerShell bloquea scripts locales, habilítelos solo para la sesión actual:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

### 4. Reproducción rápida en Linux

Desde la raíz del repositorio:

```bash
bash experiments2.0/reproduce.sh smoke
bash experiments2.0/reproduce.sh mini
bash experiments2.0/reproduce.sh full
```

Para seleccionar otro ejecutable Python 3.12:

```bash
PYTHON_BIN=/ruta/a/python3.12 bash experiments2.0/setup_env.sh gpu
```

### 5. Verificación mini antes de la corrida larga

El modo mini comprueba extracción, selección/calibración separadas por grupo,
comparación no lineal, robustez limitada a validación, evaluación COCO,
bootstrap agrupado y generación de tablas/figuras con artefactos separados:

```powershell
.\experiments2.0\reproduce.ps1 -Mode mini -SkipSetup
```

Utiliza 6 imágenes train, 4 validation y las 8 imágenes diagnósticas. Sus
salidas se guardan en `outputs/mini_e2e` de cada RQ y sus modelos en
`models/mini_e2e` de cada RQ, sin mezclarse con la corrida confirmatoria.

### 6. Ejecución por etapas

Para inspeccionar cada gate por separado en Windows:

```powershell
.\experiments2.0\setup_env.ps1 -Target gpu
$python = ".\experiments2.0\.venv\Scripts\python.exe"

& $python ".\experiments2.0\scripts\verify_environment.py"
& $python ".\experiments2.0\scripts\prepare_data.py"
& $python ".\experiments2.0\scripts\prepare_model.py"
& $python ".\experiments2.0\scripts\verify_model.py"
& $python ".\experiments2.0\scripts\run_rq1.py" --mode smoke
& $python ".\experiments2.0\scripts\run_rq2.py" --mode smoke
& $python ".\experiments2.0\scripts\run_rq3.py" --mode smoke
& $python ".\experiments2.0\scripts\run_rq4.py" --mode smoke
```

`prepare_data.py` debe imprimir `"status": "pass"`. La verificación del
entorno debe informar, como mínimo:

```text
environment_type: venv
groundingdino-py: 0.4.0
cuda_available: true
cuda_tensor_operation_passed: true
deformable_attention_device: cuda:0
deformable_attention_test_passed: true
```

También puede usar una copia del dataset ya descargada:

```powershell
& $python ".\experiments2.0\scripts\prepare_data.py" `
    --source-dir "D:\ruta\al\dataset"
```

La copia se procesa y audita con las mismas reglas; no se acepta por el solo
hecho de existir.

### 7. Reanudación y uso de la GPU

La extracción canónica crea por imagen un shard comprimido y atómico y una
sidecar JSON dentro de `data/derived/groundingdino_mc_v1`. RQ1--RQ5 validan su
esquema, fingerprint de configuración, imagen fuente, solicitud de partición y
SHA-256. Después escriben shards de características separados para cada RQ. Si
la ejecución se interrumpe, repita el mismo comando `-Mode full`: se reutiliza
cada shard compartido y específico válido, y se recalcula cualquier shard
ausente, obsoleto o corrupto.

La ruta canónica ejecuta una pasada determinista más diez estocásticas una sola
vez por imagen; RQ2, RQ3, RQ4 y RQ5 reutilizan los mismos shards neutrales y no repiten
esas 11 pasadas. Las condiciones acotadas de robustez de RQ1 sobre validación
realizan inferencia adicional porque modifican intencionalmente semillas,
prompts, asociación o imágenes.

Antes de una corrida larga, cierre juegos, entrenamiento de otros modelos y
aplicaciones que ocupen mucha VRAM. No cambie configuración, umbrales,
manifiestos ni versiones entre reanudaciones.

PyTorch advierte que el `cumsum` CUDA usado por GroundingDINO no garantiza
igualdad bit a bit entre arquitecturas GPU distintas. El proyecto fija
semillas, cuDNN, cuBLAS, paquetes, datos y pesos; la repetición exacta fue
comprobada en el mismo stack, pero el paper no debe prometer identidad binaria
entre GPUs diferentes.

### 8. Salidas esperadas

Los shards neutrales de inferencia y sus manifiestos de solicitud se guardan
en `experiments2.0/data/derived/groundingdino_mc_v1`. Son entradas compartidas
por RQs compatibles, no resultados de RQ1.

La corrida confirmatoria escribe en `experiments2.0/RQ1/outputs`:

- `features_train.parquet`, `features_validation.parquet` y
  `features_test.parquet`, con sus metadatos y hashes;
- `metrics.json` y `bootstrap_metrics.parquet`;
- `fused_predictions.parquet` e `image_safety_predictions.parquet`;
- tablas CSV de resultados, coeficientes, detector, sensibilidad, subgrupos de
  categoría/tamaño/entorno, inferencia primaria, calibración,
  complementariedad, robustez y coste computacional;
- figuras PNG/PDF de AUROC, riesgo-cobertura y fiabilidad;
- Parquet/JSON de robustez sobre validación para semillas MC, IoU de
  asociación, prompts y corrupciones controladas.

Los modelos ajustados se guardan en `experiments2.0/RQ1/models`. Los archivos
`.py` son la fuente científica de verdad; el notebook
`RQ1/notebooks/01_results.ipynb` solo consume artefactos terminados.

RQ2 guarda sus características, predicciones, bootstrap, métricas, tablas CSV,
figuras PNG/PDF y manifiesto de reporte aislados en
`experiments2.0/RQ2/outputs`, y los estimadores ajustados en
`experiments2.0/RQ2/models`. Las rutas mini terminan en `mini_e2e`.
`RQ2/notebooks/01_results.ipynb` tampoco contiene cómputo científico.

RQ3 guarda targets/features de localización, modelos espaciales y controles de
capacidad, predicciones calibradas, bootstrap pareado, re-ranking COCO,
tablas/figuras y manifiesto con hashes bajo `RQ3/outputs` y `RQ3/models`. Su
notebook también solo lee artefactos terminados.

RQ4 guarda targets/features de calibración de clase, localización e
incertidumbre, modelos de componentes/ablaciones/control de capacidad,
predicciones calibradas, bootstrap por secuencia, tablas de dominio,
figuras PNG/PDF y manifiesto con hashes bajo `RQ4/outputs` y `RQ4/models`. Su
notebook también es solo lector.

RQ5 guarda descriptores de criticidad sin etiquetas, features epistémicos con
prefijos 2/5/10, políticas calibradas `aceptar`/`diferir`, predicciones,
bootstrap por secuencia, tablas de latencia/calidad, figuras PNG/PDF y
manifiesto de hashes bajo `RQ5/outputs` y `RQ5/models`. Su notebook solo lee
artefactos. La latencia GPU mc10 es medida; mc02/mc05 son estimaciones lineales
etiquetadas.

### 9. Reutilización entre RQ1--RQ5 y preguntas posteriores

RQ2 reutiliza el dataset raw/procesado verificado, el detector y text encoder
fijados, los manifiestos congelados, el código de `src/adas_ovd` y los shards
neutrales y versionados de `data/derived/groundingdino_mc_v1`. Su
implementación de características/modelos/evaluación permanece aislada de RQ1.

Los resultados específicos permanecen aislados:

- el código, los modelos ajustados y los resultados de RQ1 permanecen en
  `RQ1/`;
- el código, los modelos y los resultados de RQ2 se crean en `RQ2/`;
- el código, los modelos y los resultados de RQ3 se crean en `RQ3/`;
- el código, los modelos y los resultados de RQ4 se crean en `RQ4/`;
- el código, los modelos y los resultados de RQ5 se crean en `RQ5/`;
- RQ2 no debe depender implícitamente de `RQ1/outputs` o `RQ1/models`;
- cada consumidor actual o futuro debe superar la validación de compatibilidad
  compartida; un tensor incompatible exige una versión de esquema coexistente
  y nunca debe reinterpretar ni sobrescribir v1 silenciosamente;
- utilizar una fusión o modelo de RQ1 como entrada de RQ2 solo es válido cuando
  el protocolo RQ2 define explícitamente esa dependencia;
- las hipótesis, entradas y métricas de RQ2/RQ3/RQ4/RQ5 deben congelarse antes de
  inspeccionar las etiquetas confirmatorias. La reutilización del mismo
  benchmark debe declararse y considerar comparaciones múltiples.

Por tanto, la data sí puede reutilizarse; las respuestas reportadas por RQ1 no
deben tratarse como datos compartidos genéricos.

### 10. Criterios de finalización de RQ1--RQ5

Que el comando termine no implica automáticamente que la hipótesis sea
positiva. Después de la corrida se debe auditar e interpretar:

- fusión interna frente a confianza, con bootstrap pareado;
- AUROC, AUPRC, AURC, Brier, NLL y ECE;
- riesgo-cobertura y cobertura a riesgo fijo;
- ablaciones semántica, geométrica, representación y presencia;
- sensibilidad con 2/5/10 pasadas MC y con distintos umbrales;
- resultados por hora, clima, escena, categoría y tamaño de objeto;
- robustez a semillas MC independientes, IoU de asociación, prompts y
  corrupciones;
- intervalos agrupados de Brier/NLL/ECE y sensibilidad de bins de ECE;
- regla congelada de AUROC/AURC con Holm y no-inferioridad Brier;
- latencia sincronizada con modelo caliente, throughput y pico de memoria GPU;
- falsos negativos a nivel de imagen y cualquier resultado `not_estimable`.

No ajuste el método usando el test confirmatorio. Una fusión que no supera la
confianza sigue siendo un resultado válido y debe reportarse como tal.

Para RQ2, audite además la fusión aprendida frente a los estimadores aprendidos
determinista-only y stochastic-only para AUROC y AURC, las cuatro comparaciones
primarias corregidas por Holm, la fusión fija igual, el aumento con confianza,
la fusión no lineal, complementariedad en validación, subgrupos de
categoría/tamaño y el tiempo sincronizado determinista frente a MC. Un comando
técnicamente correcto no convierte la hipótesis en positiva.

Para RQ3, audite `product_fusion` frente a confianza y al control no espacial
de capacidad equivalente para AUROC, AURC y Brier en la familia Holm de seis
tests. Reporte también ablaciones de localización/fusión igual/directa,
sensibilidad 2/5/10 y de score/IoU/bins ECE, re-ranking COCO AP50/AP75,
taxonomía de error, subgrupos, calibración, reutilización de shards y todo
`not_estimable`. Mini sigue siendo diagnóstico aunque sus comparaciones
nominales sean favorables.

Para RQ4, audite multinivel frente a confianza calibrada y el control plano de
mismos features sobre el subconjunto desplazado para Brier, NLL y AURC en la
familia Holm de seis tests. Reporte niveles/pares, capacidad, sensibilidades MC
2/5/10 y de score, estratos de referencia/eje/severidad, bins ECE,
riesgo/cobertura, hashes y cada `not_estimable`. Los shifts internos BDD no
deben describirse como validación externa.

Para RQ5, audite fusión tardía mc02 frente a confianza calibrada y el control
plano de mismos features en AURC ponderado y cobertura a riesgo 0,10 dentro de
la familia Holm de seis tests, incluidos dos tests de no inferioridad Brier,
cobertura/riesgo operativo, controles de criticidad, prefijos 2/5/10,
sensibilidades, overhead medido, distinción entre latencia medida/estimada,
presupuestos offline, hashes y cada `not_estimable`. `Diferir` no es un resultado
seguro demostrado.

### 11. Solución de problemas

- **`CUDA is not available`**: confirme driver NVIDIA, reinicie la terminal y
  ejecute `scripts/verify_environment.py`; no continúe el full en CPU.
- **Fallo de Kaggle/autenticación**: configure el token y repita el mismo
  comando. Una descarga parcial no pasa la auditoría.
- **Hash incorrecto**: no edite el artefacto; elimine o reemplace únicamente
  el archivo señalado y vuelva a ejecutar la preparación correspondiente.
- **Interrupción de la corrida**: repita `reproduce.ps1 -Mode full`; no borre
  los shards de salida de ninguna RQ.
- **Resultados mini presentes**: no los copie al manuscrito; son diagnósticos.

El estado de preparación para publicación se mantiene en
`SUBMISSION_READINESS.md` y los protocolos congelados en `RQ1/PROTOCOL.md` y
`RQ2/PROTOCOL.md`, `RQ3/PROTOCOL.md`, `RQ4/PROTOCOL.md` y `RQ5/PROTOCOL.md`.
