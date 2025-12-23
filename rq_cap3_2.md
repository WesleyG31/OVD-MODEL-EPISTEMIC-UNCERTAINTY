# CAPÍTULO 3: METODOLOGÍA (Continuación)

## 3.3 Fase 2: Establecimiento del Baseline de Detección

### 3.3.1 Objetivos de la Fase Baseline

La Fase 2 constituye el fundamento experimental del proyecto, con cuatro objetivos principales:

1. **Establecer métricas de referencia**: Obtener valores base de mAP@[0.5:0.95], AP50, AP75 y desempeño por clase que permitan cuantificar mejoras de fases posteriores
2. **Caracterizar distribución de confianzas**: Analizar el rango, media y distribución de scores del modelo pre-entrenado para identificar patrones de sobreconfianza/subconfianza
3. **Identificar patrones de error**: Clasificar errores en falsos positivos (FP), falsos negativos (FN) y generar matriz de confusión para entender limitaciones del modelo
4. **Optimizar umbral operativo**: Realizar análisis de sensibilidad a umbrales de confianza para determinar el punto operativo que maximiza mAP

### 3.3.2 Pipeline de Inferencia Baseline

El proceso de inferencia en la Fase 2 sigue una arquitectura secuencial de 8 etapas:

```
┌─────────────────────────────────────────────────────────────────┐
│              PIPELINE DE INFERENCIA BASELINE                    │
└─────────────────────────────────────────────────────────────────┘

ENTRADA: val_eval (2,000 imágenes), Modelo Grounding-DINO

ETAPA 1: CARGA DE IMAGEN
├─ Cargar imagen RGB desde disco
├─ Extraer dimensiones originales (W, H)
└─ Preprocesar: resize a [800, 1333] + normalización

ETAPA 2: FORWARD PASS (Modo Determinista)
├─ model.eval() → BatchNorm en modo inferencia
├─ Dropout desactivado (p=0, sin estocásticidad)
├─ with torch.no_grad() → no cálculo de gradientes
└─ Output: {boxes_cxcywh, logits, phrases}

ETAPA 3: CONVERSIÓN DE COORDENADAS
├─ boxes_cxcywh → boxes_xyxy (formato [x1, y1, x2, y2])
├─ Desnormalización: boxes * [W, H, W, H]
└─ boxes_xyxy → boxes_xywh (formato COCO [x, y, w, h])

ETAPA 4: NORMALIZACIÓN DE ETIQUETAS
├─ phrases raw → normalized_labels
├─ Aplicar diccionario de sinónimos (bike→bicycle, etc.)
└─ Mapear a category_ids de BDD100K COCO

ETAPA 5: FILTRADO POR CONFIANZA
├─ scores = sigmoid(logits)
├─ Aplicar conf_threshold = 0.30
└─ Retener solo detecciones con score ≥ threshold

ETAPA 6: NON-MAXIMUM SUPPRESSION (NMS)
├─ Agrupar por clase (NMS independiente por categoría)
├─ Ordenar por score descendente
├─ Eliminar duplicados con IoU > 0.65
└─ Output: detecciones sin duplicados

ETAPA 7: LIMITACIÓN DE DETECCIONES
├─ Ordenar por score (mayor a menor)
├─ Seleccionar top-K (K=300)
└─ Descartar detecciones de menor confianza

ETAPA 8: CLIPPING Y FORMATO COCO
├─ Clip bboxes a límites de imagen
├─ Convertir a formato JSON COCO
└─ Append a lista global de predicciones

SALIDA: preds_raw.json (22,162 detecciones)
```

#### Pseudocódigo Completo

```python
ALGORITMO: InferenciaBaseline(model, dataset, config)

ENTRADA:
    model: Grounding-DINO pre-entrenado
    dataset: val_eval (2,000 imágenes)
    config: {conf_threshold=0.30, nms_iou=0.65, max_detections=300}

SALIDA:
    predictions: Lista de detecciones en formato COCO

BEGIN
    predictions ← []
    coco_gt ← LoadCOCO(dataset.annotations)
    image_ids ← coco_gt.getImageIDs()
    
    PARA CADA img_id EN image_ids:
        // 1. Cargar y preprocesar imagen
        img_info ← coco_gt.loadImages(img_id)
        img_path ← dataset.image_dir / img_info.file_name
        image_rgb ← LoadImage(img_path)
        W, H ← image_rgb.size
        image_transformed ← Preprocess(image_rgb)  // Resize + normalize
        
        // 2. Forward pass determinista
        model.eval()  // Modo evaluación (Dropout OFF, BatchNorm fixed)
        CON torch.no_grad():
            boxes, logits, phrases ← model.predict(
                image_transformed,
                text_prompt="person. rider. car. truck. bus. ...",
                box_threshold=config.conf_threshold,
                text_threshold=0.25
            )
        FIN CON
        
        // 3. Post-procesamiento de coordenadas
        boxes_xyxy ← ConvertCxCyWhToXyXy(boxes)
        boxes_xyxy ← boxes_xyxy * [W, H, W, H]  // Desnormalizar
        boxes_xywh ← ConvertXyXyToXyWh(boxes_xyxy)
        scores ← Sigmoid(logits)
        
        // 4. Normalización de etiquetas
        normalized_labels ← []
        category_ids ← []
        PARA CADA phrase EN phrases:
            label ← NormalizeLabel(phrase)  // Sinónimos + canonización
            SI label EN BDD_CATEGORIES:
                normalized_labels.append(label)
                category_ids.append(LabelToID(label))
        FIN PARA
        
        // 5. Filtrado por confianza (ya aplicado en predict)
        // boxes, scores, labels ya filtrados por box_threshold
        
        // 6. Non-Maximum Suppression por clase
        boxes_nms, scores_nms, labels_nms ← ApplyNMS(
            boxes_xywh, scores, category_ids,
            iou_threshold=config.nms_iou
        )
        
        // 7. Limitación a max_detections
        SI len(boxes_nms) > config.max_detections:
            sorted_indices ← ArgsortDescending(scores_nms)
            sorted_indices ← sorted_indices[:config.max_detections]
            boxes_nms ← boxes_nms[sorted_indices]
            scores_nms ← scores_nms[sorted_indices]
            labels_nms ← labels_nms[sorted_indices]
        FIN SI
        
        // 8. Formato COCO y almacenamiento
        PARA i = 0 HASTA len(boxes_nms):
            bbox ← ClipBBox(boxes_nms[i], W, H)
            prediction ← {
                "image_id": img_id,
                "category_id": labels_nms[i],
                "bbox": bbox,  // [x, y, w, h] formato COCO
                "score": scores_nms[i]
            }
            predictions.append(prediction)
        FIN PARA
    FIN PARA
    
    // Guardar predicciones
    GuardarJSON(predictions, "outputs/baseline/preds_raw.json")
    
    RETORNAR predictions
FIN
```

### 3.3.3 Algoritmo de Non-Maximum Suppression (NMS)

NMS es crítico para eliminar detecciones duplicadas del mismo objeto. La implementación utiliza **NMS por clase independiente** (class-agnostic sería subóptimo).

```python
ALGORITMO: ApplyNMS(boxes, scores, labels, iou_threshold)

ENTRADA:
    boxes: Array N×4 de bounding boxes [x, y, w, h]
    scores: Array N de scores de confianza
    labels: Array N de category_ids
    iou_threshold: Umbral IoU para considerar duplicado (default 0.65)

SALIDA:
    boxes_keep, scores_keep, labels_keep: Detecciones sin duplicados

BEGIN
    keep_indices ← []
    unique_labels ← Unique(labels)
    
    // NMS independiente por cada clase
    PARA CADA label_id EN unique_labels:
        // 1. Seleccionar detecciones de esta clase
        mask ← (labels == label_id)
        class_boxes ← boxes[mask]
        class_scores ← scores[mask]
        class_indices ← Indices(mask)  // Índices originales
        
        SI len(class_boxes) == 0:
            CONTINUAR
        FIN SI
        
        // 2. Calcular áreas de bounding boxes
        areas ← class_boxes[:, 2] * class_boxes[:, 3]  // w × h
        
        // 3. Ordenar por score descendente
        order ← ArgsortDescending(class_scores)
        
        // 4. Algoritmo Greedy de NMS
        MIENTRAS order no esté vacío:
            // Seleccionar detección con mayor score
            i ← order[0]
            keep_indices.append(class_indices[i])
            
            SI len(order) == 1:
                ROMPER  // Última detección
            FIN SI
            
            // Calcular IoU con detecciones restantes
            xx1 ← max(class_boxes[i, 0], class_boxes[order[1:], 0])
            yy1 ← max(class_boxes[i, 1], class_boxes[order[1:], 1])
            xx2 ← min(class_boxes[i, 0] + class_boxes[i, 2],
                      class_boxes[order[1:], 0] + class_boxes[order[1:], 2])
            yy2 ← min(class_boxes[i, 1] + class_boxes[i, 3],
                      class_boxes[order[1:], 1] + class_boxes[order[1:], 3])
            
            // Área de intersección
            w_inter ← max(0, xx2 - xx1)
            h_inter ← max(0, yy2 - yy1)
            intersection ← w_inter * h_inter
            
            // IoU = Intersección / Unión
            union ← areas[i] + areas[order[1:]] - intersection
            iou ← intersection / union
            
            // Filtrar detecciones con IoU bajo (no duplicados)
            mask_keep ← (iou <= iou_threshold)
            order ← order[1:][mask_keep]
        FIN MIENTRAS
    FIN PARA
    
    // Retornar detecciones seleccionadas
    keep_indices ← Sort(keep_indices)  // Mantener orden original
    boxes_keep ← boxes[keep_indices]
    scores_keep ← scores[keep_indices]
    labels_keep ← labels[keep_indices]
    
    RETORNAR boxes_keep, scores_keep, labels_keep
FIN

// Función auxiliar: Cálculo de IoU
FUNCIÓN ComputeIoU(box1, box2):
    x1 ← max(box1[0], box2[0])
    y1 ← max(box1[1], box2[1])
    x2 ← min(box1[0] + box1[2], box2[0] + box2[2])
    y2 ← min(box1[1] + box1[3], box2[1] + box2[3])
    
    intersection ← max(0, x2 - x1) * max(0, y2 - y1)
    area1 ← box1[2] * box1[3]
    area2 ← box2[2] * box2[3]
    union ← area1 + area2 - intersection
    
    RETORNAR intersection / union SI union > 0 SINO 0
FIN FUNCIÓN
```

**Complejidad computacional**:
- **Peor caso**: O(N² × C), donde N = número de detecciones, C = número de clases
- **Caso promedio**: O(N log N × C) debido a ordenamiento inicial por scores
- **Optimización**: NMS por clase reduce comparaciones innecesarias entre objetos de diferentes categorías

### 3.3.4 Evaluación con COCO API

```python
ALGORITMO: EvaluarCOCO(predictions, ground_truth)

ENTRADA:
    predictions: Lista de detecciones en formato COCO
    ground_truth: Anotaciones COCO val_eval

SALIDA:
    metrics: Diccionario con mAP, AP50, AP75, etc.

BEGIN
    coco_gt ← COCO(ground_truth)
    coco_dt ← coco_gt.loadRes(predictions)
    
    // Evaluación global
    coco_eval ← COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()    // Calcula TP/FP/FN por IoU threshold
    coco_eval.accumulate()  // Agrega resultados por clase y área
    coco_eval.summarize()   // Genera métricas finales
    
    // Extraer métricas principales
    metrics ← {
        'mAP_50_95': coco_eval.stats[0],  // mAP@[.50:.95]
        'AP_50': coco_eval.stats[1],      // AP@IoU=0.5
        'AP_75': coco_eval.stats[2],      // AP@IoU=0.75
        'AP_small': coco_eval.stats[3],   // AP objetos < 32² px
        'AP_medium': coco_eval.stats[4],  // AP objetos 32²-96² px
        'AP_large': coco_eval.stats[5],   // AP objetos > 96² px
        'AR_max_1': coco_eval.stats[6],   // Recall máx con 1 det
        'AR_max_10': coco_eval.stats[7],  // Recall máx con 10 det
        'AR_max_100': coco_eval.stats[8]  // Recall máx con 100 det
    }
    
    // Evaluación por clase
    per_class_metrics ← {}
    PARA CADA (cat_id, cat_name) EN BDD_CATEGORIES:
        coco_eval_class ← COCOeval(coco_gt, coco_dt, iouType='bbox')
        coco_eval_class.params.catIds ← [cat_id]
        coco_eval_class.evaluate()
        coco_eval_class.accumulate()
        
        per_class_metrics[cat_name] ← {
            'mAP': coco_eval_class.stats[0],
            'AP50': coco_eval_class.stats[1],
            'AP75': coco_eval_class.stats[2]
        }
    FIN PARA
    
    metrics['per_class'] ← per_class_metrics
    
    RETORNAR metrics
FIN
```

### 3.3.5 Resultados y Análisis de Baseline

**Métricas globales obtenidas**:

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| mAP@[.50:.95] | 0.1705 | Baseline razonable para OVD sin fine-tuning |
| AP@50 | 0.2785 | 27.85% de precisión con IoU≥0.5 |
| AP@75 | 0.1705 | Caída del 38.8% en localizaciones precisas |
| AP (small) | 0.0633 | **Problema crítico**: objetos pequeños mal detectados |
| AP (medium) | 0.1821 | Rendimiento aceptable en objetos medianos |
| AP (large) | 0.3770 | 2.2× mejor que baseline global |

**Análisis de errores (100 imágenes analizadas)**:

- **Falsos Negativos**: 988 (97.4% de errores totales)
- **Falsos Positivos**: 26 (2.6% de errores totales)
- **Ratio FN/FP**: 38.0 (desbalance extremo)

**Distribución de FN por clase**:
```
car            598 (60.5%)  ← Clase más frecuente, mayoría no detectada
traffic sign   223 (22.6%)  ← Objetos pequeños pasados por alto
traffic light  108 (10.9%)  ← Semáforos distantes problemáticos
person          31 (3.1%)   ← Mejor detección en personas
truck           12 (1.2%)   ← Confusión con cars
```

**Conclusiones de Fase 2**:
1. Modelo tiene problema de **baja recall**, no de baja precisión
2. Necesidad de reducir `conf_threshold` para aumentar detecciones (explorado en análisis de sensibilidad)
3. Scores del modelo potencialmente **mal calibrados** (motivación para Fase 4)
4. Variabilidad esperada en detecciones si se introduce estocásticidad (motivación para Fase 3)

---

## 3.4 Fase 3: Cuantificación de Incertidumbre Epistémica mediante MC-Dropout

### 3.4.1 Objetivos y Fundamento Teórico

**Objetivos**:
1. Implementar inferencia estocástica con K=5 pases forward
2. Cuantificar incertidumbre epistémica mediante varianza de scores
3. Evaluar capacidad discriminativa (TP vs FP) usando incertidumbre
4. Analizar overhead computacional (K × tiempo_baseline)

**Fundamento matemático**:

Inferencia bayesiana aproximada mediante Dropout:

```
Objetivo: p(y | x, D) = ∫ p(y | x, W) p(W | D) dW

Aproximación MC-Dropout:
p(y | x, D) ≈ (1/K) Σₖ₌₁ᴷ p(y | x, Wₖ)   donde Wₖ ~ q(W)

Estimadores:
μ(x) = (1/K) Σₖ f(x; Wₖ)           (predicción media)
σ²(x) = (1/K) Σₖ [f(x; Wₖ) - μ(x)]²  (incertidumbre epistémica)
```

### 3.4.2 Pipeline de MC-Dropout

```
┌─────────────────────────────────────────────────────────────────┐
│              PIPELINE MC-DROPOUT (K=5 PASES)                    │
└─────────────────────────────────────────────────────────────────┘

PARA CADA imagen I en val_eval:

  ┌─────────────────────────────────────────────────────────────┐
  │ ETAPA 1: K PASES ESTOCÁSTICOS                              │
  └─────────────────────────────────────────────────────────────┘
  all_passes ← []
  
  PARA k = 1 HASTA K=5:
    ├─ Activar Dropout en Transformer Decoder (p=0.1)
    ├─ model.eval() + dropout.train() (híbrido)
    ├─ Forward pass → boxes_k, scores_k, labels_k
    ├─ Normalizar etiquetas (sinónimos → canónicas)
    ├─ Aplicar NMS por clase (iou_threshold=0.5)
    └─ Almacenar: all_passes.append({boxes_k, scores_k, labels_k})
  FIN PARA
  
  ┌─────────────────────────────────────────────────────────────┐
  │ ETAPA 2: ALINEACIÓN (Hungarian Matching)                   │
  └─────────────────────────────────────────────────────────────┘
  clusters ← AlignDetectionsHungarian(all_passes, iou_threshold=0.65)
  // Agrupa detecciones del mismo objeto a través de K pases
  
  ┌─────────────────────────────────────────────────────────────┐
  │ ETAPA 3: AGREGACIÓN ESTADÍSTICA                            │
  └─────────────────────────────────────────────────────────────┘
  PARA CADA cluster EN clusters:
    bbox_mean ← Mean(cluster.boxes)
    score_mean ← Mean(cluster.scores)
    score_std ← StdDev(cluster.scores)    ← Incertidumbre
    score_var ← Variance(cluster.scores)
    num_passes ← Count(cluster.scores)
  FIN PARA
  
  ┌─────────────────────────────────────────────────────────────┐
  │ ETAPA 4: FORMATO COCO Y ALMACENAMIENTO                     │
  └─────────────────────────────────────────────────────────────┘
  Convertir a formato COCO con campos adicionales:
    - score_mean (confianza promedio)
    - score_std (desviación estándar)
    - score_var (varianza = incertidumbre epistémica)
    - num_passes (robustez del cluster)

FIN PARA CADA imagen
```

### 3.4.3 Algoritmo de Alineación Hungarian Matching

Este es el **algoritmo crítico** de la Fase 3 que resuelve el problema de correspondencia de detecciones entre K pases estocásticos.

**Problema**: Dados K conjuntos de detecciones {D₁, D₂, ..., Dₖ}, identificar cuáles corresponden al mismo objeto real.

**Desafíos**:
1. Bounding boxes varían ligeramente entre pases (estocásticidad de Dropout)
2. Número de detecciones puede variar (algunas predichas solo en k < K pases)
3. Misma clase puede tener múltiples instancias en imagen (e.g., varios autos)

**Solución**: Clustering basado en IoU con matching greedy secuencial.

```python
ALGORITMO: AlignDetectionsHungarian(all_passes, iou_threshold)

ENTRADA:
    all_passes: Lista de K diccionarios, cada uno con:
                {boxes: Array Nₖ×4, scores: Array Nₖ, labels: Array Nₖ}
    iou_threshold: Umbral IoU para considerar mismo objeto (default 0.65)

SALIDA:
    clusters: Lista de clusters, cada cluster = {boxes: [], scores: [], label: int}

BEGIN
    // Inicialización: Usar pase 0 como referencia
    reference ← all_passes[0]
    clusters ← []
    
    // Crear un cluster inicial por cada detección del pase 0
    PARA i = 0 HASTA len(reference.boxes):
        cluster ← {
            boxes: [reference.boxes[i]],
            scores: [reference.scores[i]],
            label: reference.labels[i],
            matched_in_pass: [0]  // Lista de pases que detectaron objeto
        }
        clusters.append(cluster)
    FIN PARA
    
    // Matching secuencial: Para cada pase k=1,...,K-1
    PARA k = 1 HASTA K-1:
        pass_k ← all_passes[k]
        
        SI len(pass_k.boxes) == 0:
            CONTINUAR  // Pase sin detecciones
        FIN SI
        
        // Crear matriz de asignación
        matched_detections_k ← Set()  // Detecciones ya asignadas en pase k
        
        // Intentar asignar cada cluster existente
        PARA CADA cluster EN clusters:
            SI cluster.label NO EN pass_k.labels:
                CONTINUAR  // Pase k no tiene esta clase
            FIN SI
            
            // Calcular bbox promedio actual del cluster
            bbox_cluster_mean ← Mean(cluster.boxes)
            
            // Buscar mejor match en pase k
            best_iou ← 0.0
            best_idx ← -1
            
            PARA j = 0 HASTA len(pass_k.boxes):
                SI j EN matched_detections_k:
                    CONTINUAR  // Ya asignada a otro cluster
                FIN SI
                
                SI pass_k.labels[j] != cluster.label:
                    CONTINUAR  // Clase diferente
                FIN SI
                
                // Calcular IoU con detección j del pase k
                iou ← ComputeIoU(bbox_cluster_mean, pass_k.boxes[j])
                
                SI iou > best_iou Y iou >= iou_threshold:
                    best_iou ← iou
                    best_idx ← j
                FIN SI
            FIN PARA
            
            // Si encontró match, agregar al cluster
            SI best_idx >= 0:
                cluster.boxes.append(pass_k.boxes[best_idx])
                cluster.scores.append(pass_k.scores[best_idx])
                cluster.matched_in_pass.append(k)
                matched_detections_k.add(best_idx)
            FIN SI
        FIN PARA
        
        // Crear nuevos clusters para detecciones no asignadas en pase k
        PARA j = 0 HASTA len(pass_k.boxes):
            SI j NO EN matched_detections_k:
                // Detección nueva (no apareció en pases anteriores)
                new_cluster ← {
                    boxes: [pass_k.boxes[j]],
                    scores: [pass_k.scores[j]],
                    label: pass_k.labels[j],
                    matched_in_pass: [k]
                }
                clusters.append(new_cluster)
            FIN SI
        FIN PARA
    FIN PARA
    
    RETORNAR clusters
FIN

// Función auxiliar: Agregación estadística
ALGORITMO: AggregateClusterStatistics(clusters)

ENTRADA:
    clusters: Lista de clusters con detecciones alineadas

SALIDA:
    aggregated: Lista de detecciones agregadas con estadísticos

BEGIN
    aggregated ← []
    
    PARA CADA cluster EN clusters:
        SI len(cluster.scores) == 0:
            CONTINUAR  // Cluster vacío (no debería ocurrir)
        FIN SI
        
        boxes_array ← Array(cluster.boxes)
        scores_array ← Array(cluster.scores)
        
        detection ← {
            // Geometría promedio
            bbox: Mean(boxes_array, axis=0),  // [x_mean, y_mean, w_mean, h_mean]
            
            // Categoría (idéntica para todo el cluster)
            category_id: cluster.label,
            
            // Estadísticos de score (confianza)
            score_mean: Mean(scores_array),
            score_median: Median(scores_array),
            score_std: StdDev(scores_array),     // ← Incertidumbre epistémica
            score_var: Variance(scores_array),
            score_min: Min(scores_array),
            score_max: Max(scores_array),
            
            // Robustez del cluster
            num_passes: len(scores_array),      // Cuántos pases detectaron objeto
            detection_rate: len(scores_array) / K,  // Tasa de detección (0-1)
            
            // Información adicional
            scores_all: scores_array.tolist(),  // Scores de todos los pases
            matched_passes: cluster.matched_in_pass  // IDs de pases
        }
        
        aggregated.append(detection)
    FIN PARA
    
    RETORNAR aggregated
FIN
```

**Ejemplo ilustrativo**:

```
Pase 0: [{car, [100,100,50,50], s=0.85}, {person, [200,150,30,40], s=0.72}]
Pase 1: [{car, [102,98,52,48], s=0.81}, {person, [201,149,31,41], s=0.68}]
Pase 2: [{car, [99,101,51,49], s=0.87}]  ← person no detectado
Pase 3: [{car, [101,100,50,51], s=0.83}, {person, [200,151,30,40], s=0.70}]
Pase 4: [{car, [100,99,49,50], s=0.84}, {person, [199,150,32,41], s=0.69}]

Clusters resultantes:
1. car: {
     boxes: [[100,100,50,50], [102,98,52,48], [99,101,51,49], 
             [101,100,50,51], [100,99,49,50]],
     scores: [0.85, 0.81, 0.87, 0.83, 0.84],
     bbox_mean: [100.4, 99.6, 50.4, 49.6],
     score_mean: 0.840,
     score_std: 0.0224  ← Incertidumbre BAJA (detección robusta)
   }

2. person: {
     boxes: [[200,150,30,40], [201,149,31,41], [200,151,30,40], 
             [199,150,32,41]],
     scores: [0.72, 0.68, 0.70, 0.69],
     bbox_mean: [200.0, 150.0, 30.75, 40.5],
     score_mean: 0.6975,
     score_std: 0.0171  ← Incertidumbre BAJA
     detection_rate: 4/5 = 0.80  ← No detectado en pase 2
   }
```

**Parámetros críticos**:

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| `iou_threshold` | 0.65 | Suficientemente alto para garantizar mismo objeto, tolerante a variación estocástica |
| `K` (pases) | 5 | Trade-off computación-calidad: overhead 5×, estabilización de varianza |
| `nms_threshold` | 0.5 | Aplicado **antes** de alineación para reducir duplicados por pase |

**Complejidad computacional**:
- **Por imagen**: O(K × N × M), donde N = detecciones por pase, M = clusters existentes
- **Peor caso**: O(K × N²) si todas las detecciones son de la misma clase
- **Caso típico**: O(K × N log N) debido a filtros por clase y early stopping

### 3.4.4 Métricas de Incertidumbre

**1. Varianza de Scores (σ²)**:
```python
σ²(detección) = (1/K) Σₖ [score_k - score_mean]²
```
- **Interpretación**: Cuantifica cuánto varía la confianza del modelo entre pases
- **Valores esperados**:
  - σ² < 0.01: Incertidumbre baja (detección confiable)
  - σ² > 0.05: Incertidumbre alta (detección dudosa, posible FP)

**2. Coeficiente de Variación (CV)**:
```python
CV(detección) = σ / score_mean
```
- **Interpretación**: Incertidumbre normalizada por magnitud del score
- **Ventaja**: Permite comparar detecciones con diferentes niveles de confianza

**3. AUROC (Discriminación TP vs FP)**:
```python
AUROC = ∫₀¹ TPR(FPR) d(FPR)

donde:
- FPR = FP / (FP + TN)  (eje X)
- TPR = TP / (TP + FN)  (eje Y)
- Threshold variable: incertidumbre σ²
```
- **Interpretación**: Capacidad de la incertidumbre para identificar errores
- **Valores objetivo**:
  - AUROC = 0.5: Random (incertidumbre no discrimina)
  - AUROC > 0.6: Discriminación moderada
  - AUROC > 0.7: Discriminación fuerte

### 3.4.5 Resultados de Fase 3

**Métricas de detección**:
- mAP@0.5: 0.1823 (+6.9% mejora vs baseline 0.1705)
- Overhead computacional: 5× (1.84s vs 0.37s por imagen)

**Métricas de incertidumbre**:
- Detecciones con variación: 98.9% (29,591/29,914)
- σ²_mean(TP): 0.0142
- σ²_mean(FP): 0.0681
- **Ratio FP/TP**: 2.24× (falsos positivos tienen 2.24× más incertidumbre)
- **AUROC**: 0.634 (discriminación moderada)

**Conclusión**: MC-Dropout funcional y efectivo para cuantificar incertidumbre epistémica en OVD.

---

## 3.5 Fase 4: Calibración de Probabilidades con Temperature Scaling

### 3.5.1 Objetivos y Fundamento Teórico

**Objetivos**:
1. Corregir sobreconfianza del modelo mediante parámetro escalar T
2. Mejorar alineación entre confianza predicha y precisión real
3. Preservar capacidad discriminativa (mAP debe mantenerse)
4. Evaluar generalización de T optimizado en val_calib → val_eval

**Problema de calibración**:

Un modelo está calibrado si:
```
P(Ŷ = y | P̂ = p) = p
```

**Ejemplo de descalibración**: Si el modelo predice 100 objetos con score=0.8, pero solo 60 son correctos, el modelo está **sobreconfiado** (80% predicho vs 60% real).

**Temperature Scaling**: Método post-hoc que escala logits por temperatura T:

```
Conversión score → logit:
z = log(s / (1 - s))     (inverse sigmoid)

Escalado por temperatura:
z_calib = z / T

Conversión logit → score calibrado:
s_calib = 1 / (1 + exp(-z_calib))
```

**Interpretación de T**:
- T = 1.0: Sin cambio (modelo original)
- **T > 1.0**: Suavizado (smoothing) → reduce confianza → modelo **sobreconfiado**
- T < 1.0: Agudizado (sharpening) → aumenta confianza → modelo subconfiado

### 3.5.2 Pipeline de Temperature Scaling

```
┌─────────────────────────────────────────────────────────────────┐
│              PIPELINE TEMPERATURE SCALING                       │
└─────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────┐
│ ETAPA 1: INFERENCIA EN VAL_CALIB (8,000 imágenes)           │
└───────────────────────────────────────────────────────────────┘
predictions_calib ← InferenciaBaseline(model, val_calib)
// Reutilizar modelo y pipeline de Fase 2
// Output: {image_id, bbox, score, category_id}

┌───────────────────────────────────────────────────────────────┐
│ ETAPA 2: MATCHING PREDICCIÓN-GROUND TRUTH                    │
└───────────────────────────────────────────────────────────────┘
calib_data ← []
PARA CADA img_id EN val_calib:
    preds ← predictions_calib[img_id]
    gts ← ground_truth_calib[img_id]
    
    // Hungarian matching (asignación óptima)
    matches ← HungarianMatching(preds, gts, iou_threshold=0.5)
    
    PARA CADA match EN matches:
        is_tp ← (match.iou >= 0.5) AND (match.pred_label == match.gt_label)
        calib_data.append({
            score: match.pred_score,
            logit: log(match.pred_score / (1 - match.pred_score)),
            is_tp: is_tp,
            category: match.pred_label,
            iou: match.iou
        })
    FIN PARA
FIN PARA

// Resultado: DataFrame con N detecciones × {score, logit, is_tp}

┌───────────────────────────────────────────────────────────────┐
│ ETAPA 3: OPTIMIZACIÓN DE TEMPERATURA                         │
└───────────────────────────────────────────────────────────────┘
logits ← calib_data.logit.values
labels ← calib_data.is_tp.values  // 1=TP, 0=FP

// Función objetivo: Negative Log-Likelihood
FUNCIÓN NLL(T):
    probs ← sigmoid(logits / T)
    nll ← -mean(labels * log(probs) + (1-labels) * log(1-probs))
    RETORNAR nll
FIN FUNCIÓN

// Optimización con L-BFGS-B
T_optimal ← minimize(NLL, x0=1.0, bounds=[0.01, 10.0], method='L-BFGS-B')

┌───────────────────────────────────────────────────────────────┐
│ ETAPA 4: EVALUACIÓN EN VAL_EVAL (2,000 imágenes)             │
└───────────────────────────────────────────────────────────────┘
predictions_eval ← InferenciaBaseline(model, val_eval)

// Aplicar temperatura a scores
PARA CADA pred EN predictions_eval:
    logit ← log(pred.score / (1 - pred.score))
    logit_calib ← logit / T_optimal
    pred.score_calibrated ← sigmoid(logit_calib)
FIN PARA

// Calcular métricas de calibración
metrics_before ← ComputeCalibration(predictions_eval, T=1.0)
metrics_after ← ComputeCalibration(predictions_eval, T=T_optimal)

┌───────────────────────────────────────────────────────────────┐
│ ETAPA 5: EVALUACIÓN DE DETECCIÓN (mAP)                       │
└───────────────────────────────────────────────────────────────┘
// Verificar que mAP no cambia (Temperature Scaling preserva ranking)
mAP_before ← EvaluarCOCO(predictions_eval con score_original)
mAP_after ← EvaluarCOCO(predictions_eval con score_calibrated)

ASSERT abs(mAP_before - mAP_after) < 0.01  // Cambio < 1%
```

### 3.5.3 Algoritmo de Hungarian Matching para Calibración

Este algoritmo asigna cada predicción a un ground truth (si existe) para etiquetar detecciones como TP/FP, necesario para optimizar T.

```python
ALGORITMO: HungarianMatchingForCalibration(predictions, ground_truth, iou_threshold)

ENTRADA:
    predictions: Lista de detecciones {bbox, score, label, image_id}
    ground_truth: Anotaciones COCO {bbox, label, image_id}
    iou_threshold: Umbral IoU para considerar TP (default 0.5)

SALIDA:
    calibration_data: Lista de {score, logit, is_tp, category, iou}

BEGIN
    calibration_data ← []
    images ← Unique([p.image_id for p in predictions])
    
    PARA CADA img_id EN images:
        preds_img ← Filtrar(predictions, image_id == img_id)
        gts_img ← Filtrar(ground_truth, image_id == img_id)
        
        SI len(preds_img) == 0:
            CONTINUAR  // Sin predicciones en imagen
        FIN SI
        
        // PASO 1: Construir matriz de coste (negativa de IoU)
        N_pred ← len(preds_img)
        N_gt ← len(gts_img)
        cost_matrix ← Zeros(N_pred, N_gt)
        
        PARA i = 0 HASTA N_pred:
            PARA j = 0 HASTA N_gt:
                iou ← ComputeIoU(preds_img[i].bbox, gts_img[j].bbox)
                cost_matrix[i, j] ← -iou  // Negativo porque minimizamos coste
            FIN PARA
        FIN PARA
        
        // PASO 2: Algoritmo Húngaro (asignación óptima)
        // Encuentra asignación que maximiza suma de IoUs
        pred_indices, gt_indices ← linear_sum_assignment(cost_matrix)
        
        // PASO 3: Etiquetar detecciones como TP/FP
        matched_preds ← Set()
        
        PARA CADA (pred_idx, gt_idx) EN zip(pred_indices, gt_indices):
            iou ← -cost_matrix[pred_idx, gt_idx]  // Recuperar IoU positivo
            pred ← preds_img[pred_idx]
            gt ← gts_img[gt_idx]
            
            // Determinar si es True Positive
            is_tp ← (iou >= iou_threshold) AND (pred.label == gt.label)
            
            // Convertir score a logit
            score ← pred.score
            logit ← log(score / (1 - score + 1e-7))  // +epsilon para estabilidad
            
            calibration_data.append({
                score: score,
                logit: logit,
                is_tp: 1 SI is_tp SINO 0,
                category: pred.label,
                iou: iou,
                image_id: img_id
            })
            
            matched_preds.add(pred_idx)
        FIN PARA
        
        // PASO 4: Agregar predicciones no matcheadas como FP
        PARA pred_idx = 0 HASTA N_pred:
            SI pred_idx NO EN matched_preds:
                pred ← preds_img[pred_idx]
                score ← pred.score
                logit ← log(score / (1 - score + 1e-7))
                
                calibration_data.append({
                    score: score,
                    logit: logit,
                    is_tp: 0,  // Falso Positivo
                    category: pred.label,
                    iou: 0.0,
                    image_id: img_id
                })
            FIN SI
        FIN PARA
    FIN PARA
    
    RETORNAR calibration_data
FIN

// Función auxiliar: Linear Sum Assignment (Hungarian Algorithm)
// Implementación simplificada del algoritmo
ALGORITMO: linear_sum_assignment(cost_matrix)

ENTRADA:
    cost_matrix: Matriz M×N de costes

SALIDA:
    row_ind, col_ind: Índices de asignación óptima

BEGIN
    // Implementación basada en algoritmo Kuhn-Munkres
    // (en práctica se usa scipy.optimize.linear_sum_assignment)
    
    M, N ← cost_matrix.shape
    
    // Reducción de filas: restar mínimo de cada fila
    PARA i = 0 HASTA M:
        cost_matrix[i, :] ← cost_matrix[i, :] - Min(cost_matrix[i, :])
    FIN PARA
    
    // Reducción de columnas: restar mínimo de cada columna
    PARA j = 0 HASTA N:
        cost_matrix[:, j] ← cost_matrix[:, j] - Min(cost_matrix[:, j])
    FIN PARA
    
    // Iterar hasta encontrar asignación completa
    MIENTRAS no todas las filas asignadas:
        // Buscar ceros en matriz reducida
        // Usar algoritmo de matching bipartito (e.g., Hopcroft-Karp)
        // Actualizar matriz con pasos Hungarian
    FIN MIENTRAS
    
    RETORNAR row_indices, col_indices
FIN
```

**Ejemplo ilustrativo**:

```
Predicciones:  [{car, [100,100,50,50], s=0.85}, 
                {car, [300,200,60,40], s=0.62}, 
                {person, [200,150,30,40], s=0.55}]

Ground Truth:  [{car, [102,98,52,48]}, 
                {person, [201,149,31,41]}]

Matriz de IoU:
           GT_car  GT_person
Pred_car1   0.87      0.0
Pred_car2   0.12      0.0
Pred_person 0.0       0.82

Matriz de Coste (negativa IoU):
           GT_car  GT_person
Pred_car1  -0.87     0.0
Pred_car2  -0.12     0.0
Pred_person 0.0     -0.82

Hungarian Assignment:
- Pred_car1 → GT_car (IoU=0.87 ≥ 0.5, misma clase) → TP
- Pred_person → GT_person (IoU=0.82 ≥ 0.5, misma clase) → TP
- Pred_car2 → sin match → FP

Calibration Data:
[
  {score: 0.85, logit: 1.735, is_tp: 1, category: car, iou: 0.87},
  {score: 0.62, logit: 0.497, is_tp: 0, category: car, iou: 0.12},
  {score: 0.55, logit: 0.201, is_tp: 1, category: person, iou: 0.82}
]
```

### 3.5.4 Algoritmo de Optimización de Temperatura

```python
ALGORITMO: OptimizeTemperature(calibration_data)

ENTRADA:
    calibration_data: DataFrame con {logit, is_tp} de val_calib

SALIDA:
    T_optimal: Temperatura óptima que minimiza NLL

BEGIN
    logits ← calibration_data.logit.values
    labels ← calibration_data.is_tp.values  // 1=TP, 0=FP
    
    // FUNCIÓN OBJETIVO: Negative Log-Likelihood
    FUNCIÓN NLL(T):
        // 1. Escalar logits por temperatura
        T ← max(T, 0.01)  // Evitar división por cero
        logits_scaled ← logits / T
        
        // 2. Convertir a probabilidades
        probs ← 1 / (1 + exp(-logits_scaled))
        probs ← clip(probs, 1e-7, 1 - 1e-7)  // Evitar log(0)
        
        // 3. Calcular Binary Cross-Entropy
        nll ← -mean(labels * log(probs) + (1 - labels) * log(1 - probs))
        
        RETORNAR nll
    FIN FUNCIÓN
    
    // OPTIMIZACIÓN CON L-BFGS-B
    // (Quasi-Newton con bounds)
    
    // Parámetros de optimización
    T_init ← 1.0
    bounds ← [(0.01, 10.0)]  // T ∈ [0.01, 10.0]
    
    // Llamar optimizador scipy
    result ← minimize(
        fun=NLL,
        x0=[T_init],
        method='L-BFGS-B',
        bounds=bounds,
        options={
            'maxiter': 100,
            'ftol': 1e-6,
            'gtol': 1e-5
        }
    )
    
    T_optimal ← result.x[0]
    nll_optimal ← result.fun
    
    // Diagnóstico de convergencia
    SI result.success:
        IMPRIMIR "✓ Optimización convergió"
    SINO:
        IMPRIMIR "⚠ Optimización no convergió:", result.message
    FIN SI
    
    // Calcular NLL antes de calibrar (T=1.0)
    nll_before ← NLL(1.0)
    
    IMPRIMIR "NLL antes (T=1.0):", nll_before
    IMPRIMIR "NLL después (T=", T_optimal, "):", nll_optimal
    IMPRIMIR "Mejora en NLL:", nll_before - nll_optimal
    
    RETORNAR T_optimal
FIN
```

**Detalles del optimizador L-BFGS-B**:

- **L-BFGS** (Limited-memory Broyden-Fletcher-Goldfarb-Shanno): Método quasi-Newton
  - Aproxima matriz Hessiana sin calcularla explícitamente
  - Memoria limitada: almacena solo últimas m iteraciones (m=10 típicamente)
  - Convergencia rápida para funciones suaves

- **B** (Bounded): Permite restricciones en variables
  - En nuestro caso: T ∈ [0.01, 10.0]
  - T < 0.01: inestabilidad numérica (probs → 0 o 1)
  - T > 10.0: suavizado extremo (probs → 0.5)

- **Convergencia**: Típicamente 10-30 iteraciones
  - ftol=1e-6: Tolerancia en cambio de función objetivo
  - gtol=1e-5: Tolerancia en norma del gradiente

### 3.5.5 Métricas de Calibración

```python
ALGORITMO: ComputeCalibrationMetrics(predictions, ground_truth, T, n_bins=10)

ENTRADA:
    predictions: Detecciones con scores
    ground_truth: Anotaciones COCO
    T: Temperatura (1.0 = sin calibrar, T_optimal = calibrado)
    n_bins: Número de bins para ECE (default 10)

SALIDA:
    metrics: {nll, brier, ece, bin_data}

BEGIN
    // 1. Matching y conversión a logits
    calib_data ← HungarianMatchingForCalibration(predictions, ground_truth, 0.5)
    logits ← calib_data.logit.values
    labels ← calib_data.is_tp.values
    
    // 2. Aplicar temperatura
    logits_scaled ← logits / T
    probs ← sigmoid(logits_scaled)
    probs ← clip(probs, 1e-7, 1 - 1e-7)
    
    // 3. NEGATIVE LOG-LIKELIHOOD (NLL)
    nll ← -mean(labels * log(probs) + (1 - labels) * log(1 - probs))
    
    // 4. BRIER SCORE (Error cuadrático medio)
    brier ← mean((probs - labels)²)
    
    // 5. EXPECTED CALIBRATION ERROR (ECE)
    bin_boundaries ← linspace(0, 1, n_bins + 1)
    ece ← 0.0
    bin_data ← []
    
    PARA i = 0 HASTA n_bins:
        bin_lower ← bin_boundaries[i]
        bin_upper ← bin_boundaries[i + 1]
        
        // Seleccionar predicciones en bin
        in_bin ← (probs >= bin_lower) AND (probs < bin_upper)
        prop_in_bin ← mean(in_bin)
        
        SI prop_in_bin > 0:
            // Accuracy real en bin
            accuracy_in_bin ← mean(labels[in_bin])
            
            // Confianza promedio en bin
            avg_confidence_in_bin ← mean(probs[in_bin])
            
            // Contribución al ECE
            gap ← abs(avg_confidence_in_bin - accuracy_in_bin)
            ece ← ece + gap * prop_in_bin
            
            bin_data.append({
                bin_lower: bin_lower,
                bin_upper: bin_upper,
                confidence: avg_confidence_in_bin,
                accuracy: accuracy_in_bin,
                gap: gap,
                count: sum(in_bin)
            })
        SINO:
            // Bin vacío
            bin_data.append({
                bin_lower: bin_lower,
                bin_upper: bin_upper,
                confidence: (bin_lower + bin_upper) / 2,
                accuracy: 0,
                gap: 0,
                count: 0
            })
        FIN SI
    FIN PARA
    
    metrics ← {
        nll: nll,
        brier: brier,
        ece: ece,
        bin_data: bin_data
    }
    
    RETORNAR metrics
FIN
```

### 3.5.6 Resultados de Fase 4

**Temperatura óptima obtenida**: T = 2.344

**Interpretación**: Modelo **sobreconfidente** (T > 1.0), necesita suavizado para alinear confianza con precisión real.

**Mejoras en calibración (val_eval)**:

| Métrica | Antes (T=1.0) | Después (T=2.344) | Mejora |
|---------|---------------|-------------------|--------|
| **NLL** | 0.1138 | **0.1110** | **-2.46%** ✓ |
| **Brier Score** | 0.0742 | **0.0719** | **-3.16%** ✓ |
| **ECE** | 0.0716 | **0.0561** | **-21.64%** ✓ |

**Preservación de mAP**:
- mAP@0.5 antes: 0.1705
- mAP@0.5 después: 0.1709
- **Diferencia**: +0.24% (prácticamente idéntico, dentro del ruido estadístico)

**Conclusión**: Temperature Scaling calibra efectivamente el modelo sin sacrificar capacidad discriminativa, validando su aplicabilidad en OVD para ADAS.

---

**Resumen de secciones 3.3-3.5**:

Estas secciones detallan exhaustivamente las Fases 2-4 del pipeline experimental, incluyendo:

1. **Fase 2 (Baseline)**: Pipeline de inferencia determinista, NMS por clase, evaluación COCO
2. **Fase 3 (MC-Dropout)**: Inferencia estocástica K=5, **Hungarian matching** para alineación de detecciones, agregación estadística de incertidumbre
3. **Fase 4 (Temperature Scaling)**: **Hungarian matching** para calibración, optimización de T con L-BFGS-B, métricas ECE/NLL/Brier

El pseudocódigo proporcionado es suficientemente detallado para implementación y reproduce fielmente los algoritmos del proyecto.
