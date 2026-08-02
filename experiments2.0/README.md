# Experiments 2.0

[English](#english) | [Español](#español)

## English

Reproducible code for the paper on epistemic uncertainty in open-vocabulary
detection for ADAS. The official workflow uses Python `venv` and `pip`; Conda
and Docker are not required.

The empirical claim is restricted to the pinned GroundingDINO Swin-T
checkpoint, ten fixed BDD prompts, the frozen BDD100K partition and recorded
hardware. “Open-vocabulary” describes the architecture; this experiment does
not establish base-to-novel or cross-backbone generalization. The final frozen
methodological rules are incorporated directly in each RQ-specific
`PROTOCOL.md`.

Each RQ is a separately prespecified family and all five outcomes are reported;
there is no paper-level omnibus “all hypotheses supported” claim.

Documentation policy: every README and reproducibility guide must keep its
English and Spanish sections synchronized in the same change.

The `.py` modules are the scientific source of truth. They provide tested,
versioned and non-interactive execution. Notebooks only read completed
artifacts to regenerate tables and figures.

Current scope: the top-level reproduction command runs RQ1--RQ5. RQ2 fuses
deterministic and stochastic uncertainty; RQ3 fuses classification confidence
with learned spatial quality; RQ4 evaluates class/localization/uncertainty
post-hoc calibration under frozen BDD covariate shifts; RQ5 adds calibrated
risk-aware `accept`/`defer` decisions and an offline latency frontier. No RQ reads another
RQ's models, outputs or results. All five consume one neutral, hash-validated
detector extraction: 11 GPU passes per image in total.

### Quick start

From the repository root on Windows, run the complete installation, data,
model and the RQ-specific two-image repeatability checks:

```powershell
.\experiments2.0\reproduce.ps1
```

After all smoke tests pass, start the full confirmatory RQ1--RQ5 runs
only with explicit authorization:

```powershell
.\experiments2.0\reproduce.ps1 -Mode full
```

The isolated five-RQ diagnostic workflow is:

```powershell
.\experiments2.0\reproduce.ps1 -Mode mini -SkipSetup
```

On Linux:

```bash
bash experiments2.0/reproduce.sh smoke
bash experiments2.0/reproduce.sh mini
bash experiments2.0/reproduce.sh full
```

To create only the isolated environment:

```powershell
.\experiments2.0\setup_env.ps1 -Target gpu
```

The setup creates only `experiments2.0/.venv`, installs pinned versions,
checks `groundingdino-py==0.4.0`, runs a real CUDA tensor operation and tests
the package's PyTorch deformable-attention implementation on the GPU. It also
runs the project-scoped tests and writes the resolved environment to
`requirements-lock.txt`.

The full run requires an NVIDIA GPU and a driver compatible with the pinned
CUDA 11.8 PyTorch build. Setup fails explicitly when CUDA is unavailable, so a
full run cannot silently fall back to CPU.

The shared layer can also be precomputed explicitly with
`scripts/run_shared_extraction.py`; normal RQ commands request and resume it
automatically. See `SHARED_EXTRACTION.md` for its schema and the RQ3+ contract.

RQ1 freezes an explicit success rule: the internal-only fusion must improve
both AUROC and AURC over confidence after Holm correction and remain
non-inferior in Brier score. Selection and isotonic calibration use separate,
sequence-disjoint validation folds. Calibration intervals, category/object
size subgroups, synchronized runtime/VRAM, a nonlinear comparator and
validation-only robustness to MC seeds, association, prompts and corruptions
are generated automatically.

RQ3 freezes a localization-aware product score, confidence and
capacity-matched controls, group-disjoint selection/calibration, 2/5/10-pass
sensitivity and a six-comparison Holm family over AUROC, AURC and calibrated
Brier score. Mini RQ3 outputs are diagnostic only.

RQ4 freezes a three-level product of class, localization and epistemic
reliability probabilities. It compares against calibrated confidence and a
flat same-feature capacity control on shifted detections using Brier, NLL and
AURC with a six-comparison Holm family. Component fitting, selection and
calibration use only the frozen source domain; shifted rows are evaluation
only. Class calibration is category-conditioned with a global fallback for
rare or unseen classes. Within-BDD shifts are not external domain validation,
and mini outputs are diagnostic only.

RQ5 freezes equal late fusion of calibrated detector confidence and fused
epistemic error probability, followed by a label-free ADAS criticality weight
and validation-selected `accept`/`defer` policy. Its mc02 primary method is
compared with calibrated confidence and a flat same-feature capacity control
using weighted selective risk, group bootstrap, calibration, a decision-cost
gate and an offline latency frontier. Prefix latency is labelled measured (mc10) or estimated
(mc02/mc05); mini outputs are diagnostic only.

### Data and model identity

The canonical input is the pinned Kaggle dataset version
`solesensei/solesensei_bdd100k/versions/2`. Data preparation converts the
BDD100K validation release to COCO, normalizes historical category aliases,
creates deterministic group-disjoint partitions and audits images, boxes,
categories, groups and SHA-256 values.

GroundingDINO, its checkpoint and the exact BERT snapshot are also pinned and
verified before extraction. Large data and model files remain under
`experiments2.0/data/` and are ignored by Git; their provenance is stored in
`experiments2.0/artifacts/`.

### Reuse across research questions

The project is intentionally divided into shared and RQ-specific layers:

- `src/adas_ovd`, `data`, the pinned model and dataset/manifests are shared by
  RQ1, RQ2, RQ3, RQ4, RQ5 and later research questions;
- `data/derived/groundingdino_mc_v1` contains the implemented immutable,
  label-free detector shards used by every compatible consumer;
- `RQ1/configs`, `RQ1/src`, `RQ1/models` and `RQ1/outputs` are specific to RQ1;
- `RQ2/configs`, `RQ2/src`, `RQ2/models` and `RQ2/outputs` are specific to RQ2;
- `RQ3/configs`, `RQ3/src`, `RQ3/models` and `RQ3/outputs` are specific to RQ3;
- `RQ4/configs`, `RQ4/src`, `RQ4/models` and `RQ4/outputs` are specific to RQ4;
- `RQ5/configs`, `RQ5/src`, `RQ5/models` and `RQ5/outputs` are specific to RQ5;
- future RQs must live in their own `RQ*/` folders and import shared behavior
  from `adas_ovd`;
- an RQ must not silently read another RQ's fitted models or reported results;
- every consumer must pass the common compatibility gate before using the
  versioned shared artifact; a future incompatible tensor requirement creates
  a coexisting schema version instead of modifying v1 in place.

Raw/processed BDD100K data and the frozen split may therefore be reused. RQ1
metrics and fitted fusion models may be reused only when the scientific design
explicitly defines them as inputs. All RQ2/RQ3/RQ4/RQ5 hypotheses and analysis choices
must be frozen before inspecting confirmatory labels to avoid test leakage.

### Layout

```text
experiments2.0/
|-- setup_env.ps1 / setup_env.sh
|-- reproduce.ps1 / reproduce.sh
|-- requirements-*.txt
|-- configs/base.yaml
|-- scripts/
|-- src/adas_ovd/          # shared by all RQs
|-- tests/
|-- data/                  # downloads + versioned shared inference; ignored by Git
|-- SHARED_EXTRACTION.md   # schema and extension contract for future RQs
|-- artifacts/             # provenance, audits and frozen manifests
|-- RQ1/
|   |-- configs/rq1.yaml
|   |-- PROTOCOL.md
|   |-- src/rq1/
|   |-- tests/
|   |-- notebooks/         # completed-artifact readers only
|   |-- models/
|   `-- outputs/
|-- RQ2/                   # same isolated RQ layout
|-- RQ3/                   # same isolated RQ layout
|-- RQ4/                   # same isolated RQ layout
`-- RQ5/
    |-- configs/rq5.yaml
    |-- PROTOCOL.md
    |-- src/rq4/
    |-- tests/
    |-- notebooks/         # completed-artifact readers only
    |-- models/
    `-- outputs/
```

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the clean-machine procedure
and the RQ-specific `PROTOCOL.md` and `README.md` files for frozen designs and
commands.

---

## Español

Código reproducible del paper sobre incertidumbre epistémica en detección
open-vocabulary para ADAS. La ruta oficial usa Python `venv` y `pip`; no
requiere Conda ni Docker.

El alcance empírico se limita al checkpoint GroundingDINO Swin-T, diez prompts
BDD fijados, la partición BDD100K congelada y el hardware registrado.
“Open-vocabulary” describe la arquitectura; no se demuestra generalización a
categorías nuevas ni a otros backbones. Las reglas metodológicas
preconfirmatorias finales están integradas directamente en el `PROTOCOL.md`
de cada RQ.

Cada RQ es una familia preespecificada separada y se informan las cinco; no se
formula una conclusión ómnibus de que todas las hipótesis fueron apoyadas.

Política de documentación: cada README y guía de reproducibilidad debe
mantener sincronizadas sus secciones en inglés y español en el mismo cambio.

Los módulos `.py` son la fuente científica de verdad. Permiten una ejecución
probada, versionada y no interactiva. Los notebooks solo leen artefactos
terminados para regenerar tablas y figuras.

Alcance actual: el comando superior ejecuta RQ1--RQ5. RQ2 fusiona incertidumbre
determinista y estocástica; RQ3 fusiona confianza con calidad espacial; RQ4
evalúa calibración post-hoc de clase/localización/incertidumbre bajo shifts
covariables BDD congelados; RQ5 añade decisiones `aceptar`/`diferir` calibradas
y sensibles al riesgo con una frontera offline de latencia. Ninguna lee modelos, salidas ni
resultados de otra RQ. Las cinco consumen una sola extracción neutral: 11
pasadas GPU por imagen en total.

### Inicio rápido

Desde la raíz del repositorio en Windows, ejecute la instalación, preparación
de datos/modelo y comprobaciones repetibles por RQ con dos imágenes:

```powershell
.\experiments2.0\reproduce.ps1
```

Cuando todos los smoke tests terminen correctamente, inicie las corridas
confirmatorias completas de RQ1--RQ5 solo con autorización explícita:

```powershell
.\experiments2.0\reproduce.ps1 -Mode full
```

El workflow diagnóstico aislado de las cinco RQ es:

```powershell
.\experiments2.0\reproduce.ps1 -Mode mini -SkipSetup
```

En Linux:

```bash
bash experiments2.0/reproduce.sh smoke
bash experiments2.0/reproduce.sh mini
bash experiments2.0/reproduce.sh full
```

Para crear solamente el entorno aislado:

```powershell
.\experiments2.0\setup_env.ps1 -Target gpu
```

La instalación crea únicamente `experiments2.0/.venv`, instala versiones
fijadas, comprueba `groundingdino-py==0.4.0`, ejecuta una operación tensorial
CUDA real y prueba en GPU la atención deformable PyTorch incluida por el
paquete. También ejecuta las pruebas del proyecto y escribe el entorno resuelto
en `requirements-lock.txt`.

La corrida completa requiere una GPU NVIDIA y un driver compatible con el
build PyTorch CUDA 11.8 fijado. La instalación falla explícitamente si CUDA no
está disponible; por tanto, el full no puede pasar silenciosamente a CPU.

La capa compartida también puede precalcularse explícitamente con
`scripts/run_shared_extraction.py`; los comandos normales de cada RQ la
solicitan y reanudan automáticamente. `SHARED_EXTRACTION.md` documenta su
esquema y el contrato para RQ3+.

RQ1 congela una regla explícita de éxito: la fusión solo-interna debe mejorar
AUROC y AURC frente a confianza tras la corrección de Holm y mantener no
inferioridad en Brier. La selección y calibración isotónica usan folds de
validación distintos y separados por secuencia. El flujo genera
automáticamente intervalos de calibración, subgrupos de categoría/tamaño,
tiempo/VRAM sincronizados, un comparador no lineal y robustez limitada a
validación frente a semillas MC, asociación, prompts y corrupciones.

RQ3 congela un producto sensible a localización, controles de confianza y
capacidad equivalente, selección/calibración disjuntas por grupo,
sensibilidad 2/5/10 y una familia Holm de seis comparaciones para AUROC, AURC
y Brier calibrado. Las salidas mini de RQ3 solo son diagnósticas.

RQ4 congela un producto de tres probabilidades de fiabilidad de clase,
localización e incertidumbre. Lo compara con confianza calibrada y un control
plano de mismos features en Brier, NLL y AURC sobre detecciones desplazadas,
con seis contrastes Holm. Ajuste, selección y calibración usan sólo el dominio
fuente congelado; las filas desplazadas son únicamente evaluación. La
calibración de clase está condicionada por categoría y usa fallback global
para clases raras o no vistas. Sus shifts internos BDD no son validación de
dominio externo y mini solo es diagnóstico.

RQ5 congela fusión tardía equitativa de confianza calibrada y probabilidad de
error epistémico fusionado, seguida por criticidad ADAS sin etiquetas y una
política `aceptar`/`diferir` seleccionada en validation. Compara mc02 con
confianza calibrada y un control plano de mismos features mediante riesgo
selectivo ponderado, bootstrap por grupo, calibración, un gate de coste
decisional y una frontera offline de latencia. La
latencia se etiqueta como medida (mc10) o estimada (mc02/mc05); mini solo es
diagnóstico.

### Identidad de datos y modelo

La entrada canónica es la versión fijada de Kaggle
`solesensei/solesensei_bdd100k/versions/2`. La preparación convierte el release
de validación BDD100K a COCO, normaliza alias históricos, crea particiones
deterministas sin grupos compartidos y audita imágenes, cajas, categorías,
grupos y SHA-256.

GroundingDINO, su checkpoint y el snapshot exacto de BERT también están fijados
y se verifican antes de la extracción. Los datos y modelos grandes permanecen
en `experiments2.0/data/` y Git los ignora; su procedencia se registra en
`experiments2.0/artifacts/`.

### Reutilización entre preguntas de investigación

El proyecto está dividido intencionalmente en capas compartidas y específicas:

- `src/adas_ovd`, `data`, el modelo fijado y los datos/manifiestos son
  compartidos por RQ1, RQ2, RQ3, RQ4, RQ5 y las preguntas posteriores;
- `data/derived/groundingdino_mc_v1` contiene los shards inmutables y sin
  etiquetas que consume toda RQ compatible;
- `RQ1/configs`, `RQ1/src`, `RQ1/models` y `RQ1/outputs` son específicos de
  RQ1;
- `RQ2/configs`, `RQ2/src`, `RQ2/models` y `RQ2/outputs` son específicos de
  RQ2;
- `RQ3/configs`, `RQ3/src`, `RQ3/models` y `RQ3/outputs` son específicos de
  RQ3;
- `RQ4/configs`, `RQ4/src`, `RQ4/models` y `RQ4/outputs` son específicos de
  RQ4;
- `RQ5/configs`, `RQ5/src`, `RQ5/models` y `RQ5/outputs` son específicos de
  RQ5;
- cada RQ futura debe vivir en su propia carpeta `RQ*/` e importar el
  comportamiento compartido desde `adas_ovd`;
- una RQ no debe leer silenciosamente los modelos ajustados o resultados
  publicados de otra RQ;
- cada consumidor debe superar la validación común de compatibilidad; si una
  RQ futura requiere un tensor incompatible, debe crear una versión de esquema
  coexistente en vez de modificar v1.

Por tanto, sí se pueden reutilizar los datos BDD100K raw/procesados y la
partición congelada. Las métricas y fusiones ajustadas de RQ1 solo deben
reutilizarse cuando el diseño científico las defina explícitamente como
entradas. Todas las hipótesis y decisiones de RQ2/RQ3/RQ4/RQ5 deben congelarse antes de
inspeccionar etiquetas confirmatorias para evitar fuga del test.

### Organización

```text
experiments2.0/
|-- setup_env.ps1 / setup_env.sh
|-- reproduce.ps1 / reproduce.sh
|-- requirements-*.txt
|-- configs/base.yaml
|-- scripts/
|-- src/adas_ovd/          # compartido por todas las RQ
|-- tests/
|-- data/                  # descargas + inferencia compartida; ignoradas por Git
|-- SHARED_EXTRACTION.md   # esquema y contrato para RQ futuras
|-- artifacts/             # procedencia, auditorías y manifiestos congelados
|-- RQ1/
|   |-- configs/rq1.yaml
|   |-- PROTOCOL.md
|   |-- src/rq1/
|   |-- tests/
|   |-- notebooks/         # solo lectores de artefactos terminados
|   |-- models/
|   `-- outputs/
|-- RQ2/                   # misma estructura aislada por RQ
|-- RQ3/                   # misma estructura aislada por RQ
|-- RQ4/                   # misma estructura aislada por RQ
`-- RQ5/
    |-- configs/rq5.yaml
    |-- PROTOCOL.md
    |-- src/rq4/
    |-- tests/
    |-- notebooks/         # solo lectores de artefactos terminados
    |-- models/
    `-- outputs/
```

Consulte [REPRODUCIBILITY.md](REPRODUCIBILITY.md) para el procedimiento desde
una máquina limpia y los archivos `PROTOCOL.md` y `README.md` de cada RQ para
los diseños congelados y comandos.
