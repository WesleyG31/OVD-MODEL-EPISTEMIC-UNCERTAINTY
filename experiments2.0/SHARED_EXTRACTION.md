# Shared detector extraction / Extracción compartida del detector

## English

### Purpose

GroundingDINO inference is a neutral, versioned paper artifact. RQ1--RQ5
and future compatible research questions reuse one deterministic pass plus the
same ten seeded DropPath passes per image. They do not reuse another RQ's
features, fitted models, metrics or reported results.

The canonical cache is
`data/derived/groundingdino_mc_v1/canonical/<fingerprint>/`. Every image has an
atomic compressed NPZ shard and a JSON sidecar containing the image hash,
schema, configuration fingerprint, seeds, runtime, GPU memory, enabled
stochastic modules and shard SHA-256. Request manifests additionally bind the
cache to the frozen split manifest and ordered image IDs.

The shared inference is label-free. Ground-truth matching and every
RQ-specific feature, model, calibration, metric and report are computed in the
consumer RQ. This prevents test labels from influencing detector extraction.

### Commands

RQ1--RQ5 request the cache automatically, so the normal command remains:

```powershell
.\experiments2.0\reproduce.ps1 -Mode full -SkipSetup
```

RQ1 creates missing canonical shards; RQ2 validates and reuses them. To
precompute or audit the neutral layer explicitly:

```powershell
$python = ".\experiments2.0\.venv\Scripts\python.exe"
& $python ".\experiments2.0\scripts\run_shared_extraction.py" `
    --mode mini --consumer rq1
& $python ".\experiments2.0\scripts\run_shared_extraction.py" `
    --mode full --consumer rq1
```

The mini command defaults to the diagnostic manifest and 6/4/8 images. Full
mode refuses a non-confirmatory manifest. Re-running a command validates
SHA-256 values, reuses valid shards and recomputes only missing or corrupt
ones.

### RQ3/RQ4/RQ5 use and contract for later questions

RQ3 is a schema-v1 consumer. It derives localization stability from the
reference box, associated MC boxes/presence and decoder reference dynamics;
it derives class-agnostic IoU targets only inside RQ3 after loading ground
truth. No additional detector tensor or schema reinterpretation was needed,
and RQ3 calls `validate_consumer_compatibility(config, "rq3")` before every
shared request.

RQ4 is also a schema-v1 consumer. It derives class, localization and
epistemic calibration features entirely from existing reference scores,
boxes, presence masks, MC category scores/boxes/embeddings and decoder
trajectories. Domain descriptors and all targets are added only inside RQ4
after materialization. It calls
`validate_consumer_compatibility(config, "rq4")`; no detector tensor,
inference setting or schema-v1 meaning changed.

RQ5 consumes the same v1 scores, boxes, MC presence/category/box/embedding
arrays and decoder trajectories. It derives 2/5/10-pass uncertainty prefixes,
label-free criticality, calibration targets and decision policies only after
materialization inside `RQ5/`. It calls
`validate_consumer_compatibility(config, "rq5")`; its latency analysis reads
the existing synchronized timing sidecars and does not change inference.

A new RQ must:

1. live under its own `RQn/` directory;
2. extend `configs/base.yaml` and repeat the frozen inference keys in
   `rqn.extraction`;
3. call `validate_consumer_compatibility(config, "rqn")` before GPU work;
4. call `ensure_shared_split(...)` and read arrays with
   `materialize_common(...)` or `load_shared_shard(...)`;
5. write only its feature/model/metric/report artifacts under `RQn/`;
6. register its package/tests in `pyproject.toml` and both setup scripts;
7. add its runner to `reproduce.ps1` and `reproduce.sh` only after its protocol
   and confirmatory analysis are frozen.

The compatibility gate requires identical MC passes, seed stride, stochastic
module types, candidate threshold and association policy. If a future RQ needs
an internal tensor absent from `ARRAY_SCHEMA`, do not mutate schema v1. Add a
coexisting versioned root and reader/writer (for example
`groundingdino_mc_v2`) and document which RQs consume it. Existing v1 hashes
and confirmatory artifacts must remain interpretable.

Across the 18 verified mini shards, the measured mean is 1.77 MiB; a linear
10,000-image projection is 17.3 GiB. Detection counts vary, so plan 17–25 GiB
for the shared cache and reserve at least 55 GB for the environment, model,
downloaded/extracted dataset, cache and RQ outputs.

Canonical MC reductions are implemented once in `src/adas_ovd/mc_features.py`.
RQ1 and RQ2 naming wrappers were verified bit-for-bit on all 3,487 mini
detections, including the 2/5/10-pass prefixes. New RQs should use these
primitives rather than duplicate numerical formulas.

The shared fingerprint hashes only code that can change detector shard
contents, plus the external adapter/matching/reproducibility sources. Receipt,
label matching, feature, model and report changes invalidate their own CPU
artifacts without discarding valid GPU inference. Any change to inference,
association, array schema, weights, packages, data identity or frozen
parameters still creates a new fingerprint.

The bounded RQ1 robustness study intentionally performs extra inference on a
validation-only subset because it changes seeds, prompts, association or image
corruptions. The 11-pass saving applies to the canonical RQ1--RQ5 inference,
not to those scientifically distinct robustness conditions.

## Español

### Propósito

La inferencia de GroundingDINO es un artefacto neutral y versionado del paper.
RQ1--RQ5 y las preguntas futuras compatibles reutilizan una pasada
determinista más las mismas diez pasadas DropPath con semilla por imagen. No
reutilizan características, modelos ajustados, métricas ni resultados
reportados de otra RQ.

La caché canónica está en
`data/derived/groundingdino_mc_v1/canonical/<fingerprint>/`. Cada imagen tiene
un shard NPZ comprimido y atómico y una sidecar JSON con hash de imagen,
esquema, fingerprint de configuración, semillas, tiempo, memoria GPU, módulos
estocásticos habilitados y SHA-256 del shard. Los manifiestos de solicitud
también vinculan la caché al manifiesto de partición y a los IDs ordenados.

La inferencia compartida no usa etiquetas. El matching con ground truth y cada
característica, modelo, calibración, métrica y reporte específico se calculan
dentro de la RQ consumidora. Así las etiquetas de test no influyen en la
extracción del detector.

### Comandos

RQ1--RQ5 solicitan la caché automáticamente, por lo que el comando normal
sigue siendo:

```powershell
.\experiments2.0\reproduce.ps1 -Mode full -SkipSetup
```

RQ1 crea los shards canónicos faltantes; RQ2 los valida y reutiliza. Para
precalcular o auditar explícitamente la capa neutral:

```powershell
$python = ".\experiments2.0\.venv\Scripts\python.exe"
& $python ".\experiments2.0\scripts\run_shared_extraction.py" `
    --mode mini --consumer rq1
& $python ".\experiments2.0\scripts\run_shared_extraction.py" `
    --mode full --consumer rq1
```

El modo mini usa por defecto el manifiesto diagnóstico y 6/4/8 imágenes. El
modo full rechaza un manifiesto que no sea confirmatorio. Al repetir un
comando se validan los SHA-256, se reutilizan shards válidos y solo se
recalculan los ausentes o corruptos.

### Uso de RQ3/RQ4/RQ5 y contrato para preguntas posteriores

RQ3 consume schema v1. Deriva estabilidad de localización desde la caja de
referencia, cajas/presencia MC asociadas y dinámica de referencias del decoder;
los targets IoU sin clase solo se derivan dentro de RQ3 tras cargar ground
truth. No hizo falta ningún tensor adicional ni reinterpretar el esquema, y
RQ3 llama `validate_consumer_compatibility(config, "rq3")` antes de cada
solicitud compartida.

RQ4 también consume schema v1. Deriva features de calibración de clase,
localización e incertidumbre desde scores, cajas, presencia, scores de clase,
embeddings MC y trayectorias del decoder ya existentes. Los descriptores de
dominio y targets solo se añaden dentro de RQ4 tras materializar. Llama
`validate_consumer_compatibility(config, "rq4")`; no cambió ningún tensor,
parámetro de inferencia ni significado de v1.

RQ5 consume los mismos scores, cajas, presencia, clases, embeddings MC y
trayectorias v1. Los prefijos 2/5/10, criticidad sin etiquetas, calibración y
políticas se derivan solo tras materializar dentro de `RQ5/`. Llama
`validate_consumer_compatibility(config, "rq5")`; el análisis de latencia lee
los tiempos sincronizados existentes y no cambia la inferencia.

Una RQ nueva debe:

1. vivir en su propia carpeta `RQn/`;
2. extender `configs/base.yaml` y repetir las claves de inferencia congeladas
   en `rqn.extraction`;
3. llamar `validate_consumer_compatibility(config, "rqn")` antes de usar GPU;
4. llamar `ensure_shared_split(...)` y leer arrays mediante
   `materialize_common(...)` o `load_shared_shard(...)`;
5. escribir sus artefactos de características/modelos/métricas/reportes solo
   dentro de `RQn/`;
6. registrar su paquete/pruebas en `pyproject.toml` y ambos scripts de setup;
7. añadir su runner a `reproduce.ps1` y `reproduce.sh` únicamente después de
   congelar su protocolo y análisis confirmatorio.

La validación exige igualdad de pasadas MC, stride de semillas, tipos de
módulo estocástico, umbral de candidatos y política de asociación. Si una RQ
futura necesita un tensor que no está en `ARRAY_SCHEMA`, no se debe modificar
v1. Debe crearse una raíz y lector/escritor versionados coexistentes, por
ejemplo `groundingdino_mc_v2`, y documentar qué RQs lo consumen. Los hashes y
artefactos confirmatorios v1 deben seguir siendo interpretables.

En los 18 shards mini verificados, la media medida es 1.77 MiB; una proyección
lineal para 10,000 imágenes es 17.3 GiB. El número de detecciones varía, por lo
que se deben planificar 17–25 GiB para la caché y reservar al menos 55 GB para
entorno, modelo, dataset descargado/extraído, caché y salidas de las RQ.

Las reducciones MC canónicas se implementan una sola vez en
`src/adas_ovd/mc_features.py`. Los wrappers de nombres de RQ1 y RQ2 se
verificaron bit a bit sobre las 3,487 detecciones mini, incluidos los prefijos
de 2/5/10 pasadas. Las RQ nuevas deben usar estas primitivas en lugar de
duplicar fórmulas numéricas.

El fingerprint compartido incluye solo el código capaz de cambiar el contenido
de los shards del detector y las fuentes externas de
adapter/matching/reproducibilidad. Los cambios de recibos, matching con
etiquetas, características, modelos o reportes invalidan sus propios
artefactos CPU sin descartar inferencia GPU válida. Cualquier cambio de
inferencia, asociación, esquema, pesos, paquetes, identidad de datos o
parámetros congelados sí genera un fingerprint nuevo.

El estudio acotado de robustez de RQ1 realiza intencionalmente inferencia extra
sobre un subconjunto exclusivo de validación porque cambia semillas, prompts,
asociación o corrupciones de imagen. El ahorro de 11 pasadas se aplica a la
inferencia canónica compartida por RQ1--RQ5, no a esas condiciones científicas
distintas.
