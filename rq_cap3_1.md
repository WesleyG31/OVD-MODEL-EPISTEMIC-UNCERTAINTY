# CAPÍTULO 3: METODOLOGÍA

## 3.1 Diseño del Pipeline Experimental

### 3.1.1 Arquitectura General del Sistema

El pipeline experimental desarrollado para cuantificar y calibrar la incertidumbre epistémica en detección de objetos de vocabulario abierto (OVD) se estructura en cinco fases secuenciales e interdependientes, diseñadas para establecer una línea base de rendimiento, implementar métodos de estimación de incertidumbre, aplicar técnicas de calibración y realizar una comparación exhaustiva de enfoques.

La arquitectura del sistema se fundamenta en el modelo **Grounding-DINO** (Liu et al., 2023), un detector de objetos estado del arte que combina:

1. **Vision Transformer (ViT)** como backbone visual (arquitectura Swin-T con 800×1333 píxeles de entrada)
2. **Transformer Decoder** para procesamiento de características multimodales
3. **Language Model** para embeddings textuales de prompts open-vocabulary
4. **Cross-attention mechanisms** para alineación visual-lingüística

La elección de este modelo se justifica por tres razones fundamentales:
- **Capacidad Open-Vocabulary**: Permite detección de clases no vistas durante el entrenamiento mediante prompts textuales
- **Arquitectura Transformer**: Incluye módulos de Dropout que posibilitan la aplicación de MC-Dropout sin modificaciones arquitectónicas
- **Rendimiento competitivo**: Reporta mAP@0.5 de 52.5% en COCO (superior a métodos tradicionales como Faster R-CNN)

#### Pipeline Multi-Fase

```
┌──────────────────────────────────────────────────────────────────────┐
│                   ARQUITECTURA EXPERIMENTAL COMPLETA                 │
└──────────────────────────────────────────────────────────────────────┘

 FASE 2: BASELINE                    FASE 3: MC-DROPOUT
┌─────────────────────┐             ┌──────────────────────┐
│ Grounding-DINO      │             │ Grounding-DINO       │
│ (modo eval estándar)│             │ (Dropout activo)     │
│                     │             │                      │
│ ViT Backbone        │             │ K=5 pases forward    │
│ Transformer Decoder │───────────▶ │ estocásticos         │
│ Detection Head      │             │                      │
│                     │             │ μ = (1/K)Σf(x;Wₖ)   │
│ Output: Scores raw  │             │ σ² = Var(scores)     │
└─────────────────────┘             └──────────────────────┘
         │                                    │
         │                                    │
         ▼                                    ▼
    22,162 preds                         29,914 preds
    (val_eval)                           (val_eval)
         │                                    │
         └────────┬───────────────────────────┘
                  │
                  ▼
         FASE 4: TEMPERATURE SCALING
         ┌──────────────────────────┐
         │ Calibración Post-Hoc     │
         │                          │
         │ z_calib = z / T          │
         │ Optimización NLL         │
         │                          │
         │ val_calib: 8,000 imgs    │
         │ T_optimal = 2.344        │
         └──────────────────────────┘
                  │
                  ▼
         FASE 5: COMPARACIÓN
         ┌──────────────────────────┐
         │ 6 Métodos evaluados:     │
         │ 1. Baseline              │
         │ 2. Baseline + TS         │
         │ 3. MC-Dropout            │
         │ 4. MC-Dropout + TS       │
         │ 5. Decoder Variance      │
         │ 6. Decoder Variance + TS │
         │                          │
         │ Métricas:                │
         │ • Detección (mAP)        │
         │ • Calibración (ECE, NLL) │
         │ • Risk-Coverage (AUC)    │
         └──────────────────────────┘
```

### 3.1.2 Dataset y Particiones de Datos

#### BDD100K: Berkeley DeepDrive Dataset

El dataset seleccionado es **BDD100K** (Yu et al., 2020), uno de los benchmarks más extensos para conducción autónoma, que comprende:

- **100,000 imágenes** de alta resolución (1280×720 píxeles)
- **10 categorías de objetos** relevantes para ADAS (Advanced Driver Assistance Systems)
- **Escenarios diversos**: condiciones climáticas variables, iluminación nocturna/diurna, entornos urbanos/autopista
- **1.8M bounding boxes** anotadas manualmente

**Clases de objetos (alineadas con COCO format)**:

| ID | Categoría | Distribución | Características |
|----|-----------|--------------|-----------------|
| 1  | person    | 16.2%        | Peatones en vías urbanas |
| 2  | rider     | 2.8%         | Ciclistas/motociclistas |
| 3  | car       | 52.4%        | Vehículos mayoritarios |
| 4  | truck     | 3.1%         | Vehículos pesados |
| 5  | bus       | 1.2%         | Transporte público |
| 6  | train     | 0.1%         | Ferrocarriles urbanos |
| 7  | motorcycle| 1.4%         | Motocicletas |
| 8  | bicycle   | 2.3%         | Bicicletas |
| 9  | traffic light | 7.8%    | Semáforos |
| 10 | traffic sign  | 12.7%   | Señales viales |

#### Estrategia de Partición de Datos

A diferencia de particiones tradicionales (train/val/test), este trabajo utiliza un esquema específico para calibración, dividiendo el conjunto de validación original en dos subconjuntos disjuntos:

```python
# Configuración de splits (definida en fase 2/configs/baseline.yaml)
dataset:
  val_calib_json: ../data/bdd100k_coco/val_calib.json   # 8,000 imágenes
  val_eval_json:  ../data/bdd100k_coco/val_eval.json    # 2,000 imágenes
```

**Justificación de la partición**:

1. **val_calib (8,000 imágenes, 80%)**: 
   - **Propósito**: Optimización de hiperparámetros de calibración (temperatura T)
   - **Uso**: Matching predicción-ground truth para calcular NLL y optimizar T
   - **Prevención de overfitting**: Al separar datos de calibración de evaluación final

2. **val_eval (2,000 imágenes, 20%)**:
   - **Propósito**: Evaluación final imparcial de métodos calibrados
   - **Uso**: Cálculo de métricas de detección (mAP), calibración (ECE, NLL, Brier) y risk-coverage
   - **Garantía de generalización**: Datos nunca vistos por procesos de optimización

**Distribución estadística**:

| Split | Imágenes | Detecciones GT | Detecciones/img | Uso |
|-------|----------|----------------|----------------|-----|
| val_calib | 8,000 | ~144,000 | ~18.0 | Optimización T |
| val_eval | 2,000 | ~36,000 | ~18.0 | Evaluación final |

Esta estrategia es consistente con metodología de calibración de Guo et al. (2017), que recomienda mantener conjuntos separados para ajustar parámetros de post-calibración.

### 3.1.3 Configuración del Modelo Base

#### Parámetros de Grounding-DINO

```yaml
# Configuración baseline (fase 2/configs/baseline.yaml)
model:
  name: Grounding-DINO
  architecture: SwinT-OGC  # Swin Transformer - Object-Grounded Cross-attention
  checkpoint: groundingdino_swint_ogc.pth  # Pesos pre-entrenados
  config: GroundingDINO_SwinT_OGC.py
  input_size: [800, 1333]  # (H, W) formato estándar COCO
  device: cuda
```

**Características arquitectónicas**:

1. **Backbone: Swin Transformer**
   - Ventanas de atención de tamaño 7×7
   - 4 stages con downsampling progresivo (×4, ×8, ×16, ×32)
   - Feature Pyramid Network (FPN) para detección multi-escala
   - 28M parámetros en backbone

2. **Language Encoder**
   - BERT-base pre-entrenado
   - Embeddings de 768 dimensiones para prompts textuales
   - Procesamiento de vocabulario open-vocabulary

3. **Detection Head**
   - 6 capas de Transformer Decoder
   - **3 módulos de Dropout** (p=0.1) en cada capa del decoder
   - 300 queries de objeto (máximo de detecciones por imagen)
   - Predicción simultánea de: {clase, bbox, score}

#### Prompts Textuales (Open-Vocabulary)

El modelo opera mediante prompts textuales que definen las clases de interés:

```python
# Definido en fase 2/main.ipynb
PROMPTS = [
    'person', 'rider', 'car', 'truck', 'bus', 'train',
    'motorcycle', 'bicycle', 'traffic light', 'traffic sign'
]

TEXT_PROMPT = '. '.join(PROMPTS) + '.'
# Resultado: "person. rider. car. truck. bus. train. motorcycle. 
#             bicycle. traffic light. traffic sign."
```

**Ventaja de Open-Vocabulary**: A diferencia de detectores de vocabulario cerrado (Faster R-CNN, YOLO), Grounding-DINO puede generalizar a nuevas categorías simplemente modificando el prompt textual, sin necesidad de reentrenamiento.

#### Parámetros de Inferencia

```python
# Configuración de detección (fase 2/configs/baseline.yaml)
inference:
  conf_threshold: 0.30      # Umbral de confianza mínimo
  nms_iou: 0.65             # IoU para Non-Maximum Suppression
  batch_size: 1             # Inferencia secuencial (limitación GPU)
  max_detections: 300       # Máximo de detecciones por imagen
```

**Justificación de hiperparámetros**:

- **conf_threshold=0.30**: Seleccionado tras análisis de sensibilidad (ver Fase 2, sección 5.3) que identificó un plateau de rendimiento entre 0.25-0.30. Valores inferiores no mejoran mAP pero incrementan falsos positivos.

- **nms_iou=0.65**: Threshold estándar de COCO que balancea supresión de duplicados vs retención de objetos densos (e.g., múltiples autos en atasco).

### 3.1.4 Reproducibilidad y Control de Variabilidad

#### Semillas Aleatorias

Para garantizar reproducibilidad exacta de resultados:

```python
# Inicialización en todas las fases (fase 2-5/main.ipynb)
import torch
import numpy as np

CONFIG = {'seed': 42}

torch.manual_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed(CONFIG['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

**Impacto en resultados**:
- **PyTorch**: Controla inicialización de pesos, dropout, operaciones aleatorias en GPU
- **NumPy**: Controla shuffling, muestreo, generación de ruido
- **CUDA**: Determinismo en operaciones de convolución y reducción

#### Hardware y Entorno

```yaml
# Infraestructura utilizada
GPU: NVIDIA RTX 3090 / 4090 (24GB VRAM)
CUDA: 11.7
PyTorch: 2.0.1
Sistema: Ubuntu 22.04 LTS / Docker container
```

**Mediciones de rendimiento**:
- **Baseline**: 0.275s/imagen (3.64 FPS)
- **MC-Dropout (K=5)**: 1.84s/imagen (0.54 FPS, overhead 5x)
- **Memoria GPU**: 1,190 MB (baseline), 1,450 MB (MC-Dropout)

---

## 3.2 Configuración de Experimentos por Fase

### 3.2.1 Fase 2: Baseline de Detección sin Calibrar

#### Objetivos

1. Establecer métricas de referencia (mAP, AP por clase, AP por tamaño)
2. Caracterizar distribución de confianzas (scores) del modelo pre-entrenado
3. Identificar patrones de error (falsos positivos/negativos, matriz de confusión)
4. Determinar umbral operativo óptimo mediante análisis de sensibilidad

#### Configuración Experimental

```yaml
# fase 2/configs/baseline.yaml
model:
  checkpoint: groundingdino_swint_ogc.pth
  inference_mode: eval  # Modo estándar (sin dropout estocástico)

inference:
  conf_threshold: 0.30
  nms_iou: 0.65
  
dataset:
  split: val_eval  # 2,000 imágenes para evaluación
```

**Proceso de inferencia**:

```python
# Pseudocódigo (fase 2/main.ipynb, sección 6)
for image_id in val_eval:
    # 1. Cargar y preprocesar imagen
    image, (orig_w, orig_h) = load_image(image_path)
    
    # 2. Forward pass (modo eval, sin dropout)
    with torch.no_grad():
        boxes, logits, phrases = model(image, TEXT_PROMPT)
    
    # 3. Post-procesamiento
    scores = logits.sigmoid().cpu()  # Conversión logit → probabilidad
    boxes_xywh = box_ops.box_cxcywh_to_xyxy(boxes) * [orig_w, orig_h, orig_w, orig_h]
    
    # 4. Filtrado y NMS
    mask = scores >= conf_threshold
    boxes, scores, labels = apply_nms(boxes[mask], scores[mask], labels[mask], 
                                       iou_threshold=nms_iou)
    
    # 5. Almacenar predicciones en formato COCO
    predictions.append({
        "image_id": image_id,
        "category_id": cat_id,
        "bbox": [x, y, w, h],  # formato COCO
        "score": float(score)
    })
```

#### Post-procesamiento de Detecciones

**1. Normalización de Etiquetas**:

```python
# Manejo de variabilidad léxica de Grounding-DINO
PROMPT_SYNONYMS = {
    'bike': 'bicycle',
    'motorbike': 'motorcycle',
    'pedestrian': 'person',
    'vehicle': 'car',
    'stop sign': 'traffic sign',
    'red light': 'traffic light'
}

def normalize_label(predicted_label):
    """Mapea variantes textuales a clases canónicas de BDD100K"""
    label_lower = predicted_label.lower().strip()
    return PROMPT_SYNONYMS.get(label_lower, label_lower)
```

**Justificación**: Grounding-DINO puede generar sinónimos o variaciones de los prompts originales, requiriendo normalización para matching correcto con ground truth.

**2. Non-Maximum Suppression (NMS)**:

```python
def apply_nms(boxes, scores, labels, iou_threshold=0.65):
    """
    NMS por clase independiente (Class-Agnostic NMS sería subóptimo)
    
    Algoritmo:
    1. Para cada clase c:
        a. Ordenar detecciones por score descendente
        b. Seleccionar detección con mayor score
        c. Eliminar detecciones con IoU > threshold
        d. Repetir hasta agotar detecciones
    """
    keep_indices = []
    for label_id in torch.unique(labels):
        mask = labels == label_id
        class_boxes = boxes[mask]
        class_scores = scores[mask]
        
        # Calcular IoU matricial
        iou_matrix = compute_iou_matrix(class_boxes)
        
        # Greedy NMS
        order = class_scores.argsort(descending=True)
        while order.numel() > 0:
            i = order[0]
            keep_indices.append(i)
            iou = iou_matrix[i, order[1:]]
            order = order[1:][iou <= iou_threshold]
    
    return boxes[keep_indices], scores[keep_indices], labels[keep_indices]
```

**3. Clipping de Bounding Boxes**:

```python
def clip_bbox(bbox, img_w, img_h):
    """Asegurar que bboxes estén dentro de límites de imagen"""
    x, y, w, h = bbox
    x = max(0, min(x, img_w))
    y = max(0, min(y, img_h))
    w = max(0, min(w, img_w - x))
    h = max(0, min(h, img_h - y))
    return [x, y, w, h]
```

#### Métricas de Evaluación

**1. Métricas de Detección (COCO API)**:

```python
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Cargar ground truth y predicciones
coco_gt = COCO(val_eval_annotations)
coco_dt = coco_gt.loadRes(predictions)

# Evaluación mAP
coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

# Métricas extraídas:
mAP_50_95 = coco_eval.stats[0]  # mAP@[0.5:0.95]
AP_50 = coco_eval.stats[1]      # AP@IoU=0.5
AP_75 = coco_eval.stats[2]      # AP@IoU=0.75
AP_small = coco_eval.stats[3]   # AP objetos < 32² px
AP_medium = coco_eval.stats[4]  # AP objetos 32²-96² px
AP_large = coco_eval.stats[5]   # AP objetos > 96² px
```

**2. Análisis de Errores**:

```python
def analyze_errors(predictions, ground_truth, iou_threshold=0.5):
    """
    Clasificación de errores:
    - True Positives (TP): IoU >= threshold
    - False Positives (FP): Detección sin GT correspondiente
    - False Negatives (FN): GT sin detección correspondiente
    """
    tp, fp, fn = 0, 0, 0
    confusion_matrix = defaultdict(int)
    
    for image_id in images:
        preds = predictions[image_id]
        gts = ground_truth[image_id]
        
        # Hungarian matching (algoritmo de asignación óptima)
        cost_matrix = compute_iou_matrix(preds['boxes'], gts['boxes'])
        pred_indices, gt_indices = linear_sum_assignment(-cost_matrix)
        
        for pred_idx, gt_idx in zip(pred_indices, gt_indices):
            if cost_matrix[pred_idx, gt_idx] >= iou_threshold:
                tp += 1
                if preds['labels'][pred_idx] != gts['labels'][gt_idx]:
                    confusion_matrix[(preds['labels'][pred_idx], 
                                     gts['labels'][gt_idx])] += 1
            else:
                fp += 1
                fn += 1
        
        # FP no asignados
        fp += len(preds['boxes']) - len(pred_indices)
        # FN no asignados
        fn += len(gts['boxes']) - len(gt_indices)
    
    return {
        'tp': tp, 'fp': fp, 'fn': fn,
        'precision': tp / (tp + fp),
        'recall': tp / (tp + fn),
        'confusion_matrix': confusion_matrix
    }
```

#### Resultados Principales (Fase 2)

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **mAP@[0.5:0.95]** | 0.1705 | Baseline razonable para OVD sin fine-tuning |
| **AP@50** | 0.2785 | 27.85% precisión con IoU≥0.5 |
| **AP@75** | 0.1705 | Caída significativa en localizaciones precisas |
| **AP (small)** | 0.0633 | Problema crítico: objetos pequeños no detectados |
| **AP (medium)** | 0.1821 | Rendimiento aceptable en objetos medianos |
| **AP (large)** | 0.3770 | Mejor rendimiento en objetos grandes |

**Hallazgos críticos**:
1. **Alta tasa de falsos negativos**: 988 FN vs 26 FP en 100 imágenes analizadas
2. **Problema de recall**: El modelo falla en detectar objetos presentes (especialmente cars: 598 FN, traffic signs: 223 FN)
3. **Distribución de scores desbalanceada**: Scores promedio de detecciones correctas (0.42) vs incorrectas (0.38) con poca separación
4. **Sobreconfianza del modelo**: Reliability diagrams muestran confianza predicha > precisión real

### 3.2.2 Fase 3: Incertidumbre Epistémica con MC-Dropout

#### Fundamentación Teórica

Monte Carlo Dropout (Gal & Ghahramani, 2016) interpreta Dropout como aproximación variacional de inferencia bayesiana:

**Objetivo**: Aproximar la distribución posterior de los pesos del modelo:

```
p(W | D) ≈ q(W) = ∏ᵢ Bernoulli(pᵢ)
```

Donde:
- `W`: parámetros del modelo
- `D`: datos de entrenamiento
- `pᵢ`: probabilidad de mantener neurona i (keep probability)

**Predicción Bayesiana**:

```
p(y | x, D) ≈ (1/K) ∑ₖ₌₁ᴷ p(y | x, Wₖ)    donde Wₖ ~ q(W)
```

**Estimación de Incertidumbre**:

```
Varianza predictiva: σ²(x) = (1/K) ∑ₖ [f(x; Wₖ) - μ(x)]²
donde μ(x) = (1/K) ∑ₖ f(x; Wₖ)
```

La varianza σ²(x) cuantifica la **incertidumbre epistémica**: qué tan incierto está el modelo debido a su conocimiento limitado.

#### Configuración Experimental

```yaml
# fase 3/outputs/mc_dropout/config.yaml
K: 5  # Número de pases estocásticos (trade-off computación-calidad)
seed: 42
device: cuda
conf_threshold: 0.25  # Reducido vs baseline para aumentar recall
iou_threshold_nms: 0.5
iou_threshold_alignment: 0.65  # Para alinear detecciones entre pases K
```

**Justificación de K=5**:
- Estudios previos (Gal & Ghahramani, 2016; Kendall & Gal, 2017) reportan estabilización de estimaciones con K≥5
- Trade-off: K=5 → 5x overhead computacional (1.84s vs 0.37s por imagen)
- K>10 muestra rendimientos decrecientes en mejora de estimación

#### Modificación Arquitectónica: Activación Selectiva de Dropout

**Desafío**: GroundingDINO tiene Dropout solo en el **Transformer Decoder** (3 módulos por capa × 6 capas = 18 módulos de Dropout), no en el backbone Swin-T.

**Estrategia implementada**: Dropout selectivo

```python
def enable_dropout_in_decoder(model):
    """
    Estrategia híbrida:
    - Backbone (Swin-T): modo eval → sin estocásticidad
    - Decoder (Transformer): Dropout activo → estocásticidad controlada
    
    Justificación:
    1. Backbone pre-entrenado es estable → no necesita variación
    2. Decoder aprende alineación visual-texto → mayor incertidumbre
    3. Dropout en decoder tiene p=0.1 (tasa razonable)
    """
    model.eval()  # Modo eval global (BatchNorm estable)
    
    for name, module in model.named_modules():
        if 'decoder' in name or 'class_embed' in name or 'bbox_embed' in name:
            if isinstance(module, torch.nn.Dropout):
                module.train()  # Activar Dropout en inferencia
    
    return model
```

**Diagnóstico de Dropout**:

```python
# Verificación de módulos activos (fase 3/main.ipynb, sección 2)
dropout_modules = []
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Dropout):
        dropout_modules.append({
            'name': name,
            'training': module.training,  # True si activo
            'p': module.p                 # Dropout rate
        })

# Resultado: 18 módulos con p=0.1, training=True en decoder
```

#### Proceso de Inferencia Estocástica

```python
def mc_dropout_inference(model, image, text_prompt, K=5):
    """
    Realiza K forward passes con Dropout activo
    
    Returns:
        boxes_ensemble: Lista de K conjuntos de bboxes
        scores_ensemble: Lista de K conjuntos de scores
        labels_ensemble: Lista de K conjuntos de labels
    """
    boxes_ensemble, scores_ensemble, labels_ensemble = [], [], []
    
    for k in range(K):
        with torch.no_grad():  # No acumular gradientes (inferencia)
            # Dropout genera variación estocástica diferente en cada pase
            boxes, logits, phrases = model(image, text_prompt)
            
            scores = logits.sigmoid().cpu()
            boxes_xywh = box_ops.box_cxcywh_to_xyxy(boxes)
            labels = [normalize_label(p) for p in phrases]
            
            # NMS por pase individual
            boxes_k, scores_k, labels_k = apply_nms(
                boxes_xywh, scores, labels, 
                iou_threshold=CONFIG['iou_threshold_nms']
            )
            
            boxes_ensemble.append(boxes_k)
            scores_ensemble.append(scores_k)
            labels_ensemble.append(labels_k)
    
    return boxes_ensemble, scores_ensemble, labels_ensemble
```

#### Alineación de Detecciones entre Pases

**Desafío**: Los K pases generan detecciones con bounding boxes ligeramente diferentes para el mismo objeto. Es necesario agrupar detecciones correspondientes al mismo objeto real.

**Solución**: Algoritmo de clustering por IoU

```python
def align_detections_across_passes(boxes_ensemble, scores_ensemble, 
                                    labels_ensemble, iou_threshold=0.65):
    """
    Agrupa detecciones del mismo objeto a través de K pases
    
    Algoritmo:
    1. Para cada pase k=1,...,K:
        - Comparar detecciones con clusters existentes
        - Si IoU > threshold Y misma clase → asignar a cluster
        - Si no match → crear nuevo cluster
    2. Para cada cluster:
        - Calcular estadísticas: mean(scores), std(scores), count
    
    Returns:
        aligned_detections: DataFrame con columnas:
            - bbox: [x, y, w, h] promedio
            - score_mean: μ(score) a través de K pases
            - score_std: σ(score) (incertidumbre epistémica)
            - count: cuántos pases detectaron el objeto
            - label: clase predicha
    """
    clusters = []  # Lista de clusters de detecciones
    
    for k in range(len(boxes_ensemble)):
        for i, box in enumerate(boxes_ensemble[k]):
            label = labels_ensemble[k][i]
            score = scores_ensemble[k][i]
            
            # Buscar cluster compatible
            matched = False
            for cluster in clusters:
                if cluster['label'] == label:
                    # Calcular IoU con bbox promedio del cluster
                    iou = compute_iou(box, cluster['bbox_mean'])
                    if iou >= iou_threshold:
                        # Agregar detección al cluster
                        cluster['boxes'].append(box)
                        cluster['scores'].append(score)
                        cluster['count'] += 1
                        matched = True
                        break
            
            if not matched:
                # Crear nuevo cluster
                clusters.append({
                    'boxes': [box],
                    'scores': [score],
                    'label': label,
                    'count': 1
                })
    
    # Calcular estadísticas finales
    aligned = []
    for cluster in clusters:
        aligned.append({
            'bbox': np.mean(cluster['boxes'], axis=0),
            'label': cluster['label'],
            'score_mean': np.mean(cluster['scores']),
            'score_std': np.std(cluster['scores']),  # ← Incertidumbre
            'count': cluster['count']
        })
    
    return pd.DataFrame(aligned)
```

#### Métricas de Incertidumbre

**1. Varianza de Scores (σ²)**:

```python
# Para cada detección alineada
uncertainty = score_std ** 2  # Varianza de scores a través de K pases

# Interpretación:
# - Baja varianza (σ² < 0.01): Modelo seguro de la predicción
# - Alta varianza (σ² > 0.05): Modelo incierto (posible FP o objeto ambiguo)
```

**2. Coeficiente de Variación (CV)**:

```python
CV = score_std / score_mean

# Normaliza varianza por magnitud de score
# CV > 0.2 indica alta incertidumbre relativa
```

**3. AUROC de Discriminación TP vs FP**:

```python
# Evaluar si incertidumbre separa errores de aciertos
from sklearn.metrics import roc_auc_score

# Etiquetar detecciones como TP/FP (matching con GT)
y_true = [1 if is_true_positive(det) else 0 for det in detections]

# Usar incertidumbre como score de clasificación
y_score = [det['score_std'] for det in detections]

auroc = roc_auc_score(y_true, y_score)

# AUROC = 0.5: No discrimina (aleatorio)
# AUROC > 0.6: Discriminación moderada
# AUROC > 0.7: Discriminación fuerte
```

#### Resultados Principales (Fase 3)

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **mAP@0.5** | 0.1823 | +6.9% mejora vs baseline (0.1705) |
| **AUROC (TP vs FP)** | 0.634 | Incertidumbre discrimina errores |
| **Detecciones variables** | 98.9% | 29,591/29,914 detecciones varían entre pases |
| **Ratio σ(FP)/σ(TP)** | 2.24x | Falsos positivos tienen 2.24× más incertidumbre |
| **Tiempo/imagen** | 1.84s | Overhead 5x vs baseline (0.37s) |

**Hallazgos críticos**:
1. **MC-Dropout funcional**: 98.9% de detecciones muestran variación estocástica
2. **Incertidumbre predictiva**: σ²_mean = 0.0142 para TPs vs 0.0681 para FPs
3. **Correlación con errores**: Pearson r = 0.42 entre σ² y probabilidad de FP
4. **Limitación de K=5**: Algunos objetos no detectados en todos los pases (count < 5)

### 3.2.3 Fase 4: Calibración de Probabilidades con Temperature Scaling

#### Fundamentación Teórica

Un modelo está **calibrado** si sus probabilidades predichas reflejan con precisión la probabilidad real de correctitud:

```
P(Ŷ = y | P̂ = p) = p
```

**Ejemplo**: Si el modelo predice 100 objetos con score=0.8, esperamos que ~80 sean verdaderos positivos.

**Temperature Scaling** (Guo et al., 2017) es un método de calibración post-hoc que modifica scores mediante un parámetro escalar T:

```
Conversión score → logit:
z = logit(s) = log(s / (1 - s))

Aplicación de temperatura:
z_calib = z / T

Conversión logit → score calibrado:
s_calib = sigmoid(z_calib) = 1 / (1 + exp(-z/T))
```

**Interpretación de T**:
- **T = 1.0**: Sin cambio (modelo original)
- **T > 1.0**: **Suavizado** (smoothing) → reduce confianza → modelo sobreconfiado
- **T < 1.0**: **Agudizado** (sharpening) → aumenta confianza → modelo subconfiado

#### Configuración Experimental

```yaml
# fase 4/outputs/temperature_scaling/config.yaml
seed: 42
iou_matching: 0.5  # Para matching predicción-GT
device: cuda
categories: [person, rider, car, truck, bus, train, motorcycle, 
             bicycle, traffic light, traffic sign]

# Optimización
method: L-BFGS-B  # Optimizador quasi-Newton
bounds: [0.1, 10.0]  # Restricciones en T
```

**Proceso en dos etapas**:

```
Etapa 1: CALIBRACIÓN (val_calib, 8,000 imgs)
┌─────────────────────────────────────────┐
│ 1. Inferencia baseline en val_calib    │
│ 2. Matching predicción-GT (IoU≥0.5)    │
│ 3. Extraer pares (score, label_tp_fp)  │
│ 4. Convertir scores → logits           │
│ 5. Optimizar T minimizando NLL         │
│                                         │
│ Función objetivo:                       │
│ NLL(T) = -∑ yᵢlog(σ(zᵢ/T)) +           │
│           (1-yᵢ)log(1-σ(zᵢ/T))         │
│                                         │
│ Resultado: T_optimal = 2.344            │
└─────────────────────────────────────────┘
               │
               ▼
Etapa 2: EVALUACIÓN (val_eval, 2,000 imgs)
┌─────────────────────────────────────────┐
│ 1. Inferencia baseline en val_eval     │
│ 2. Aplicar T_optimal a scores           │
│ 3. Calcular métricas de calibración:   │
│    - ECE (Expected Calibration Error)  │
│    - NLL (Negative Log-Likelihood)     │
│    - Brier Score                        │
│ 4. Verificar preservación de mAP       │
└─────────────────────────────────────────┘
```

#### Matching Predicción-Ground Truth

```python
def match_predictions_to_gt(predictions, ground_truth, iou_threshold=0.5):
    """
    Asigna cada predicción a un GT (si existe) para etiquetar TP/FP
    
    Algoritmo:
    1. Para cada imagen:
        a. Computar matriz de IoU entre pred y GT
        b. Hungarian algorithm para asignación óptima
        c. Etiquetar: IoU >= threshold → TP, sino → FP
    2. Ground truth sin match → FN (no se usan en calibración)
    
    Returns:
        calibration_data: DataFrame con columnas:
            - score: Score raw del modelo
            - label: 1 si TP, 0 si FP
            - category: Clase de objeto
            - image_id: ID de imagen
    """
    calib_data = []
    
    for img_id in images:
        preds = predictions[img_id]  # {boxes, scores, labels}
        gts = ground_truth[img_id]
        
        # Matriz de IoU (N_pred × N_gt)
        iou_matrix = compute_iou_matrix(preds['boxes'], gts['boxes'])
        
        # Asignación óptima (maximiza suma de IoUs)
        from scipy.optimize import linear_sum_assignment
        pred_idx, gt_idx = linear_sum_assignment(-iou_matrix)
        
        matched_preds = set()
        for p_idx, g_idx in zip(pred_idx, gt_idx):
            score = preds['scores'][p_idx]
            pred_label = preds['labels'][p_idx]
            gt_label = gts['labels'][g_idx]
            iou = iou_matrix[p_idx, g_idx]
            
            # Determinar TP/FP
            is_tp = (iou >= iou_threshold) and (pred_label == gt_label)
            
            calib_data.append({
                'score': score,
                'label': 1 if is_tp else 0,
                'category': pred_label,
                'image_id': img_id,
                'iou': iou
            })
            matched_preds.add(p_idx)
        
        # Predicciones no matcheadas → FP
        for p_idx in range(len(preds['boxes'])):
            if p_idx not in matched_preds:
                calib_data.append({
                    'score': preds['scores'][p_idx],
                    'label': 0,  # FP
                    'category': preds['labels'][p_idx],
                    'image_id': img_id,
                    'iou': 0.0
                })
    
    return pd.DataFrame(calib_data)
```

#### Optimización de Temperatura

```python
def optimize_temperature(logits, labels):
    """
    Encuentra T que minimiza Negative Log-Likelihood
    
    Args:
        logits: Array de logits = log(score / (1-score))
        labels: Array binario (1=TP, 0=FP)
    
    Returns:
        T_optimal: Temperatura óptima
        nll_optimal: NLL mínimo alcanzado
    """
    def nll_loss(T):
        """
        Negative Log-Likelihood como función de T
        
        NLL = -∑ᵢ yᵢ·log(σ(zᵢ/T)) + (1-yᵢ)·log(1-σ(zᵢ/T))
        
        Equivalente a Binary Cross-Entropy sobre probabilidades calibradas
        """
        T = T[0]  # Scipy pasa array 1D
        z_scaled = logits / T
        probs = torch.sigmoid(torch.tensor(z_scaled))
        
        # Binary Cross-Entropy
        eps = 1e-7  # Evitar log(0)
        loss = -torch.mean(
            labels * torch.log(probs + eps) + 
            (1 - labels) * torch.log(1 - probs + eps)
        )
        return loss.item()
    
    # Optimización con L-BFGS-B (quasi-Newton con bounds)
    from scipy.optimize import minimize
    
    result = minimize(
        nll_loss,
        x0=[1.0],  # Inicialización T=1.0
        method='L-BFGS-B',
        bounds=[(0.1, 10.0)],  # Restricciones razonables
        options={'maxiter': 100}
    )
    
    T_optimal = result.x[0]
    nll_optimal = result.fun
    
    return T_optimal, nll_optimal
```

**Proceso de conversión score ↔ logit**:

```python
def score_to_logit(score, eps=1e-7):
    """
    Inverse sigmoid: logit = log(p / (1-p))
    
    Manejo de edge cases:
    - score ≈ 0 → logit ≈ -10 (muy negativo)
    - score ≈ 1 → logit ≈ +10 (muy positivo)
    """
    score = np.clip(score, eps, 1 - eps)  # Evitar log(0) y división por 0
    return np.log(score / (1 - score))

def logit_to_score(logit):
    """Sigmoid: σ(z) = 1 / (1 + exp(-z))"""
    return 1 / (1 + np.exp(-logit))

# Aplicación de temperatura
def apply_temperature(score, T):
    """Pipeline completo: score → logit → scale → score"""
    logit = score_to_logit(score)
    logit_scaled = logit / T
    score_calibrated = logit_to_score(logit_scaled)
    return score_calibrated
```

#### Métricas de Calibración

**1. Expected Calibration Error (ECE)**:

```python
def compute_ece(scores, labels, n_bins=10):
    """
    ECE = ∑ᵇ (nᵇ/n) |acc(b) - conf(b)|
    
    Donde:
    - b: bin de confianzas [i/B, (i+1)/B)
    - nᵇ: número de predicciones en bin b
    - acc(b): precisión real en bin b
    - conf(b): confianza promedio en bin b
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        # Seleccionar predicciones en bin
        in_bin = (scores >= bin_boundaries[i]) & (scores < bin_boundaries[i+1])
        
        if in_bin.sum() > 0:
            acc = labels[in_bin].mean()  # Precisión real
            conf = scores[in_bin].mean()  # Confianza promedio
            ece += (in_bin.sum() / len(scores)) * abs(acc - conf)
    
    return ece
```

**Interpretación**:
- ECE = 0: Modelo perfectamente calibrado
- ECE < 0.05: Calibración buena
- ECE > 0.10: Calibración pobre (scores no confiables)

**2. Negative Log-Likelihood (NLL)**:

```python
def compute_nll(scores, labels):
    """
    NLL = -∑ᵢ yᵢ·log(pᵢ) + (1-yᵢ)·log(1-pᵢ)
    
    Mide la calidad de probabilidades predichas
    """
    eps = 1e-7
    scores = np.clip(scores, eps, 1 - eps)
    nll = -np.mean(
        labels * np.log(scores) + 
        (1 - labels) * np.log(1 - scores)
    )
    return nll
```

**3. Brier Score**:

```python
def compute_brier(scores, labels):
    """
    Brier = (1/n) ∑ᵢ (pᵢ - yᵢ)²
    
    Error cuadrático medio de probabilidades
    """
    return np.mean((scores - labels) ** 2)
```

#### Reliability Diagrams

Visualización de calibración que compara confianza predicha vs precisión real:

```python
def plot_reliability_diagram(scores, labels, n_bins=10, title=''):
    """
    Eje X: Confianza predicha (binned)
    Eje Y: Precisión real (accuracy en bin)
    Línea diagonal: Calibración perfecta
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    
    accuracies = []
    confidences = []
    counts = []
    
    for i in range(n_bins):
        in_bin = (scores >= bin_boundaries[i]) & (scores < bin_boundaries[i+1])
        if in_bin.sum() > 0:
            accuracies.append(labels[in_bin].mean())
            confidences.append(scores[in_bin].mean())
            counts.append(in_bin.sum())
        else:
            accuracies.append(0)
            confidences.append(bin_centers[i])
            counts.append(0)
    
    # Graficar
    plt.figure(figsize=(8, 8))
    plt.bar(confidences, accuracies, width=1/n_bins, alpha=0.7, 
            edgecolor='black', label='Accuracy')
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'reliability_{title}.png', dpi=150)
```

#### Resultados Principales (Fase 4)

| Métrica | Antes (T=1.0) | Después (T=2.344) | Mejora |
|---------|---------------|-------------------|--------|
| **Temperatura Óptima** | - | **2.344** | Modelo sobreconfiado |
| **ECE** | 0.0716 | **0.0561** | **-21.64%** ✓ |
| **NLL** | 0.1138 | **0.1110** | **-2.46%** ✓ |
| **Brier Score** | 0.0742 | **0.0719** | **-3.16%** ✓ |
| **mAP@0.5** | 0.1705 | **0.1709** | **+0.24%** (preservado) |

**Hallazgos críticos**:
1. **Sobreconfianza confirmada**: T=2.344 > 1.0 indica que el modelo asigna scores demasiado altos
2. **Calibración mejorada**: ECE reducido en 21.64%, aproximando scores a probabilidades reales
3. **Rendimiento discriminativo preservado**: mAP prácticamente idéntico (Temperature Scaling no cambia ranking)
4. **Generalización**: Temperatura optimizada en val_calib funciona bien en val_eval

### 3.2.4 Fase 5: Comparación Sistemática de Métodos

#### Diseño Experimental Completo

La Fase 5 integra todos los métodos anteriores en un marco de comparación unificado, evaluando **6 configuraciones**:

| ID | Método | Incertidumbre | Calibración | Código |
|----|--------|---------------|-------------|--------|
| 1  | Baseline | ❌ No | ❌ No | `baseline` |
| 2  | Baseline + TS | ❌ No | ✅ Sí (T=2.344) | `baseline_ts` |
| 3  | MC-Dropout (K=5) | ✅ Epistémica (σ²) | ❌ No | `mc_dropout` |
| 4  | MC-Dropout + TS | ✅ Epistémica (σ²) | ✅ Sí (T=2.89) | `mc_dropout_ts` |
| 5  | Decoder Variance | ✅ Multi-capa | ❌ No | `decoder_var` |
| 6  | Decoder Variance + TS | ✅ Multi-capa | ✅ Sí (T=2.21) | `decoder_var_ts` |

#### Configuración Unificada

```yaml
# fase 5/outputs/comparison/config.yaml
seed: 42
device: cuda
categories: [person, rider, car, truck, bus, train, motorcycle, 
             bicycle, traffic light, traffic sign]

# Parámetros de inferencia
conf_threshold: 0.25
nms_threshold: 0.65
K_mc: 5  # Pases MC-Dropout

# Parámetros de evaluación
iou_matching: 0.5  # Para matching TP/FP
n_bins: 10  # Para ECE y reliability diagrams

# Reutilización de resultados previos (optimización)
reuse_fase2: true   # Cargar preds de ../fase 2/outputs/baseline/
reuse_fase3: true   # Cargar preds de ../fase 3/outputs/mc_dropout/
reuse_fase4: true   # Cargar temperaturas de ../fase 4/outputs/temperature_scaling/
```

#### Optimización: Reutilización de Resultados

```python
# Estrategia de cacheo para reducir tiempo de ejecución
# Original: ~2 horas | Optimizado: ~15 minutos (-87.5%)

# Cargar predicciones baseline (Fase 2)
FASE2_PREDS = Path('../fase 2/outputs/baseline/preds_raw.json')
if FASE2_PREDS.exists():
    print("✓ Reutilizando predicciones baseline de Fase 2")
    with open(FASE2_PREDS) as f:
        baseline_preds = json.load(f)
else:
    print("⚠ Ejecutando inferencia baseline completa...")
    baseline_preds = run_baseline_inference(model, val_eval)

# Cargar predicciones MC-Dropout (Fase 3)
FASE3_PREDS = Path('../fase 3/outputs/mc_dropout/preds_mc_aggregated.json')
if FASE3_PREDS.exists():
    print("✓ Reutilizando predicciones MC-Dropout de Fase 3")
    with open(FASE3_PREDS) as f:
        mc_preds = json.load(f)
else:
    print("⚠ Ejecutando inferencia MC-Dropout completa (K=5)...")
    mc_preds = run_mc_dropout_inference(model, val_eval, K=5)

# Cargar temperaturas optimizadas (Fase 4)
FASE4_TEMPS = Path('../fase 4/outputs/temperature_scaling/temperature.json')
if FASE4_TEMPS.exists():
    print("✓ Reutilizando temperaturas de Fase 4")
    with open(FASE4_TEMPS) as f:
        temperatures = json.load(f)
else:
    print("⚠ Optimizando temperaturas desde cero...")
    temperatures = optimize_temperatures_all_methods(val_calib)
```

#### Matriz de Evaluación Completa

Para cada uno de los 6 métodos, se calculan:

**1. Métricas de Detección**:
```python
detection_metrics = {
    'mAP_50_95': coco_eval.stats[0],
    'AP_50': coco_eval.stats[1],
    'AP_75': coco_eval.stats[2],
    'AP_small': coco_eval.stats[3],
    'AP_medium': coco_eval.stats[4],
    'AP_large': coco_eval.stats[5],
    'per_class_AP': {cat: ap for cat, ap in zip(categories, per_class_aps)}
}
```

**2. Métricas de Calibración**:
```python
calibration_metrics = {
    'ECE': compute_ece(scores, labels, n_bins=10),
    'NLL': compute_nll(scores, labels),
    'Brier': compute_brier(scores, labels),
    'reliability_diagram': plot_reliability(scores, labels)
}
```

**3. Risk-Coverage Analysis**:
```python
# Ordenar predicciones por incertidumbre (de menos a más incierta)
sorted_preds = sort_by_uncertainty(predictions, uncertainty_scores)

risks, coverages = [], []
for threshold_percentile in range(0, 101, 5):
    # Retener solo predicciones con incertidumbre <= threshold
    retained = sorted_preds[:int(len(sorted_preds) * threshold_percentile / 100)]
    
    # Calcular riesgo (tasa de error) en predicciones retenidas
    risk = compute_error_rate(retained)
    coverage = len(retained) / len(sorted_preds)
    
    risks.append(risk)
    coverages.append(coverage)

# Calcular AUC (área bajo la curva risk-coverage)
auc = compute_auc(coverages, risks)
```

**Interpretación de Risk-Coverage**:
- **Eje X (Coverage)**: Porcentaje de predicciones retenidas
- **Eje Y (Risk)**: Tasa de error en predicciones retenidas
- **AUC alto**: Buena capacidad para identificar predicciones confiables
- **Curva ideal**: Risk bajo a coverages altos

#### Comparación de Incertidumbres

```python
# Análisis de capacidad discriminativa (TP vs FP)
for method in methods:
    uncertainty_tp = uncertainties[is_true_positive]
    uncertainty_fp = uncertainties[is_false_positive]
    
    # AUROC: capacidad de separar TP de FP usando incertidumbre
    auroc = roc_auc_score(
        [1]*len(uncertainty_tp) + [0]*len(uncertainty_fp),
        np.concatenate([uncertainty_tp, uncertainty_fp])
    )
    
    # Ratio de incertidumbres
    ratio_fp_tp = uncertainty_fp.mean() / uncertainty_tp.mean()
    
    results[method] = {
        'auroc': auroc,
        'ratio_fp_tp': ratio_fp_tp,
        'uncertainty_tp_mean': uncertainty_tp.mean(),
        'uncertainty_fp_mean': uncertainty_fp.mean()
    }
```

#### Artefactos Generados

```
fase 5/outputs/comparison/
├── config.yaml                      # Configuración completa
│
├── Predicciones (formato CSV)
│   ├── eval_baseline.csv           # Método 1
│   ├── eval_baseline_ts.csv        # Método 2
│   ├── eval_mc_dropout.csv         # Método 3
│   ├── eval_mc_dropout_ts.csv      # Método 4
│   ├── eval_decoder_variance.csv   # Método 5
│   └── eval_decoder_variance_ts.csv# Método 6
│
├── Métricas (formato JSON)
│   ├── detection_metrics.json      # mAP por método
│   ├── calibration_metrics.json    # ECE, NLL, Brier
│   ├── risk_coverage_auc.json      # AUCs de risk-coverage
│   └── uncertainty_analysis.json   # AUROC, ratios FP/TP
│
├── Visualizaciones (PNG)
│   ├── reliability_diagram_all.png # 6 subplots
│   ├── risk_coverage_curves.png    # Comparación curvas
│   ├── calibration_comparison.png  # Barras ECE/NLL/Brier
│   └── detection_comparison.png    # Barras mAP por método
│
└── Reportes (TXT/MD)
    ├── summary.txt                  # Resumen ejecutivo
    ├── best_method_recommendation.md# Recomendación final
    └── trade_offs_analysis.md       # Análisis de compromisos
```

#### Resultados Finales (Fase 5)

**Mejor método global: MC-Dropout + Temperature Scaling**

| Dimensión | Método Ganador | Métrica | Valor | Justificación |
|-----------|----------------|---------|-------|---------------|
| **Detección** | MC-Dropout + TS | mAP@0.5 | **0.1823** | +6.9% vs baseline |
| **Incertidumbre** | MC-Dropout + TS | AUROC | **0.6335** | Mejor discriminación TP/FP |
| **Calibración** | Decoder Var + TS | ECE | **0.1409** | Mejor estimación probabilística |
| **Eficiencia** | Baseline + TS | Tiempo/img | **0.37s** | Sin overhead computacional |

**Trade-offs identificados**:

1. **MC-Dropout**: Alta incertidumbre útil pero costosa computacionalmente (5x overhead)
2. **Temperature Scaling**: Mejora calibración universalmente sin coste computacional
3. **Decoder Variance**: Baja calibración pero eficiente (single-pass, no overhead)

---

**Conclusión Metodológica**:

El pipeline experimental diseñado permite una evaluación exhaustiva de métodos de cuantificación de incertidumbre y calibración en detección de objetos open-vocabulary. La arquitectura modular (Fase 2→3→4→5) facilita la reproducibilidad, comparación justa y identificación de compromisos entre rendimiento, incertidumbre y coste computacional.
