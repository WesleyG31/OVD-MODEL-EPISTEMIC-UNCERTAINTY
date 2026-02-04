# RQ6 - Arquitectura Técnica y Flujo de Datos

## Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                    GroundingDINO Model                          │
│                                                                 │
│  ┌─────────────┐      ┌──────────────────────────────┐        │
│  │   Backbone  │ ───> │      Transformer             │        │
│  │   (SwinT)   │      │                              │        │
│  └─────────────┘      │  ┌────────────────────────┐  │        │
│                       │  │   Decoder Layers       │  │        │
│                       │  │   (6 layers)           │  │        │
│                       │  │                        │  │        │
│                       │  │  Layer 0 ──> emb[0]    │  │◄──┐    │
│                       │  │  Layer 1 ──> emb[1]    │  │   │    │
│                       │  │  Layer 2 ──> emb[2]    │  │   │    │
│                       │  │  Layer 3 ──> emb[3]    │  │   │ Hooks
│                       │  │  Layer 4 ──> emb[4]    │  │   │    │
│                       │  │  Layer 5 ──> emb[5]    │  │   │    │
│                       │  │                        │  │◄──┘    │
│                       │  └────────────────────────┘  │        │
│                       │                              │        │
│                       │  ┌────────────────────────┐  │        │
│                       │  │   Detection Heads      │  │        │
│                       │  │   (bbox + class)       │  │        │
│                       │  └────────────────────────┘  │        │
│                       └──────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
            ┌────────────────────────────────────────┐
            │        Detection Output                │
            │  - Boxes: [N, 4]                       │
            │  - Scores: [N]                         │
            │  - Labels: [N]                         │
            │  - Layer embeddings: [6, N, 256]      │
            └────────────────────────────────────────┘
                                  │
                                  ▼
            ┌────────────────────────────────────────┐
            │    Varianza Inter-Capa                 │
            │  Para cada detección i:                │
            │    layer_scores[i] = [s₀, s₁, ..., s₅]│
            │    variance[i] = Var(layer_scores[i])  │
            └────────────────────────────────────────┘
```

## Flujo de Datos Detallado

### 1. Input Processing
```
Imagen (1280x720, RGB)
    │
    ├──> load_image() ──> Tensor (3, H, W)
    │
    └──> Ground Truth Annotations
         - Bboxes: [M, 4] (xyxy format)
         - Categories: [M]
```

### 2. Forward Pass con Hooks
```python
# Pseudo-código del flujo
layer_embeddings = {}  # Storage for hooks

# Register hooks
for layer_idx, layer_module in decoder_layers:
    hook = layer_module.register_forward_hook(
        lambda module, input, output: 
            layer_embeddings[layer_idx] = output[0].detach()
    )

# Forward pass
boxes, scores, phrases = model.predict(image, text_prompt)

# Now layer_embeddings contains:
# {
#   0: Tensor([900, 1, 256]),  # Layer 0 output
#   1: Tensor([900, 1, 256]),  # Layer 1 output
#   ...
#   5: Tensor([900, 1, 256])   # Layer 5 output
# }
```

### 3. Extraction per Detection
```python
# Para cada detección idx (de N detecciones totales):
for idx in range(N):
    layer_scores = []
    
    # Extraer embedding de esta query en cada capa
    for layer_idx in [0, 1, 2, 3, 4, 5]:
        emb = layer_embeddings[layer_idx]  # [900, 1, 256]
        query_emb = emb[idx, 0, :]          # [256]
        
        # Calcular score basado en norma
        emb_norm = torch.norm(query_emb)
        layer_score = sigmoid(emb_norm / 10.0)
        
        layer_scores.append(layer_score)
    
    # Calcular varianza inter-capa
    variance = np.var(layer_scores)  # Incertidumbre epistémica
```

### 4. Ground Truth Matching
```python
# Para cada predicción, encontrar mejor GT match
for pred in predictions:
    best_iou = 0
    best_gt = None
    
    for gt in ground_truth:
        if pred.category != gt.category:
            continue
        
        iou = compute_iou(pred.bbox, gt.bbox)
        if iou > best_iou:
            best_iou = iou
            best_gt = gt
    
    is_correct = (best_iou >= 0.5)  # TP if IoU >= 0.5
```

### 5. Data Structure Evolution

#### Nivel 1: Raw Detection
```python
{
    'bbox': [x1, y1, x2, y2],
    'score': 0.87,
    'category': 'car',
    'category_id': 3
}
```

#### Nivel 2: Con Layer Scores
```python
{
    'bbox': [x1, y1, x2, y2],
    'score': 0.87,
    'category': 'car',
    'category_id': 3,
    'layer_scores': [0.82, 0.84, 0.85, 0.86, 0.87, 0.87],  # 6 capas
    'num_layers': 6
}
```

#### Nivel 3: Con Varianza
```python
{
    'bbox': [x1, y1, x2, y2],
    'score': 0.87,
    'category': 'car',
    'category_id': 3,
    'layer_scores': [0.82, 0.84, 0.85, 0.86, 0.87, 0.87],
    'score_variance': 0.0003,  # np.var(layer_scores)
    'bbox_variance': 0.12,     # Varianza espacial
    'num_layers': 6
}
```

#### Nivel 4: Con GT Matching
```python
{
    'bbox': [x1, y1, x2, y2],
    'score': 0.87,
    'category_id': 3,
    'layer_scores': [0.82, 0.84, 0.85, 0.86, 0.87, 0.87],
    'score_variance': 0.0003,
    'bbox_variance': 0.12,
    'is_correct': True,        # TP
    'iou': 0.78,              # Con mejor GT match
    'image_id': 12345
}
```

## Cálculos de Métricas

### 1. Varianza Acumulada por Capa

Para analizar cómo mejora la discriminación con profundidad:

```python
def compute_cumulative_variance(detection, up_to_layer):
    """
    Calcula varianza usando solo capas 0..up_to_layer
    """
    layer_scores = detection['layer_scores'][:up_to_layer+1]
    
    if len(layer_scores) > 1:
        return np.var(layer_scores)
    else:
        return 0.0

# Ejemplo:
# detection['layer_scores'] = [0.82, 0.84, 0.85, 0.86, 0.87, 0.87]
# 
# up_to_layer=0 → var([0.82]) = 0.0
# up_to_layer=1 → var([0.82, 0.84]) = 0.0001
# up_to_layer=2 → var([0.82, 0.84, 0.85]) = 0.0002
# ...
# up_to_layer=5 → var([0.82, 0.84, 0.85, 0.86, 0.87, 0.87]) = 0.0003
```

### 2. AUROC por Capa

Para cada capa ℓ:

```python
# Preparar datos
uncertainties = []
is_errors = []

for detection in all_detections:
    # Varianza acumulada hasta capa ℓ
    unc = compute_cumulative_variance(detection, up_to_layer=ℓ)
    uncertainties.append(unc)
    
    # Error = FP (not is_correct)
    is_errors.append(not detection['is_correct'])

# Calcular AUROC
# uncertainty alta debe predecir error (FP)
auroc = roc_auc_score(is_errors, uncertainties)
```

### 3. Separación TP vs FP

Para cada capa ℓ:

```python
# Agrupar por correctitud
tp_detections = [d for d in all_detections if d['is_correct']]
fp_detections = [d for d in all_detections if not d['is_correct']]

# Calcular varianza media
tp_variances = [compute_cumulative_variance(d, ℓ) for d in tp_detections]
fp_variances = [compute_cumulative_variance(d, ℓ) for d in fp_detections]

tp_mean_var = np.mean(tp_variances)
fp_mean_var = np.mean(fp_variances)

# Separación (queremos que sea positiva y creciente)
separation = fp_mean_var - tp_mean_var
```

## Estructura de Datos en Disco

### decoder_dynamics.parquet
```python
DataFrame con columnas:
- image_id: int
- bbox: list[float] (4 elementos)
- score: float
- category_id: int
- is_correct: bool
- iou: float
- score_variance: float
- bbox_variance: float
- layer_scores: list[float] (6 elementos)
- num_layers: int

Tamaño aproximado: ~50MB para 10k detecciones
```

### layer_variance_stats.csv
```python
DataFrame con columnas:
- layer: int (1-6)
- tp_variance: float
- fp_variance: float
- separation: float

Tamaño: ~1KB
```

### auroc_by_layer.csv
```python
DataFrame con columnas:
- layer: int (1-6)
- auroc: float
- aupr: float

Tamaño: ~1KB
```

## Optimizaciones Implementadas

### 1. Hook Efficiency
```python
# ❌ Ineficiente: Guardar todo
def hook(module, input, output):
    layer_embeddings.append(output.clone())  # Copia todo

# ✅ Eficiente: Guardar solo lo necesario
def hook(module, input, output):
    layer_embeddings.append(output[0].detach().cpu())  # Detach + CPU
```

### 2. Batch Processing
```python
# En lugar de procesar imagen por imagen:
for img in images:
    process(img)  # Slow

# Se podría hacer (futuro):
batch = images[:batch_size]
process_batch(batch)  # Faster
```

### 3. Memory Management
```python
# Limpiar después de cada imagen
torch.cuda.empty_cache()

# Detach tensors inmediatamente
embedding = output.detach().cpu()
```

## Parámetros Configurables

### En CONFIG (Celda 1)
```python
CONFIG = {
    'seed': 42,                    # Reproducibilidad
    'device': 'cuda',              # cuda/cpu
    'categories': [...],           # 10 categorías BDD100K
    'iou_matching': 0.5,           # Threshold para TP/FP
    'conf_threshold': 0.25,        # Confianza mínima
    'num_layers': 6,               # Capas del decoder
    'sample_size': 500             # Imágenes a procesar
}
```

### Ajustes Comunes

#### Más rápido (desarrollo)
```python
'sample_size': 50  # Solo 50 imágenes
```

#### Más preciso (producción)
```python
'sample_size': 2000  # Todas las imágenes de val_eval
```

#### Más estricto en matching
```python
'iou_matching': 0.7  # Solo matches muy buenos son TP
```

## Complejidad Computacional

### Tiempo por Imagen
```
T_image = T_forward + T_hooks + T_postprocess

T_forward ≈ 100ms     (inferencia del modelo)
T_hooks ≈ 10ms        (captura de embeddings)
T_postprocess ≈ 20ms  (cálculo de varianzas, matching)

Total ≈ 130ms/imagen con GPU
```

### Memoria
```
M_model ≈ 2GB         (pesos del modelo)
M_embeddings ≈ 50MB   (6 layers × 900 queries × 256 dim × 4 bytes)
M_image ≈ 10MB        (imagen + activaciones)

Total ≈ 2.5GB GPU
```

### Escalabilidad
```
500 imágenes × 130ms = 65s ≈ 1 minuto (solo inferencia)
+ postprocesamiento ≈ 2 minutos
+ análisis y plots ≈ 2 minutos

Total ≈ 5 minutos (en práctica 15-20 min por overhead)
```

## Verificación de Correctitud

### Sanity Checks Implementados

1. **Número de capas capturadas**
   ```python
   assert len(layer_embeddings) == 6, "Should capture 6 decoder layers"
   ```

2. **Dimensionalidad de embeddings**
   ```python
   assert emb.shape == (900, 1, 256), "Wrong embedding shape"
   ```

3. **Varianza no negativa**
   ```python
   assert variance >= 0, "Variance cannot be negative"
   ```

4. **AUROC en rango válido**
   ```python
   assert 0 <= auroc <= 1, "AUROC out of range"
   ```

5. **TP + FP = Total**
   ```python
   assert num_tp + num_fp == len(detections), "Mismatch in counts"
   ```

## Conclusión Técnica

Esta implementación:
- ✅ Captura correctamente las dinámicas del decoder
- ✅ Calcula varianza inter-capa como proxy de incertidumbre
- ✅ Matchea con GT para determinar TP/FP
- ✅ Analiza progresión por profundidad
- ✅ Identifica condiciones de falla
- ✅ Es eficiente en memoria y tiempo
- ✅ Es reproducible y verificable

La arquitectura es sólida y lista para generar resultados reales para RQ6.
