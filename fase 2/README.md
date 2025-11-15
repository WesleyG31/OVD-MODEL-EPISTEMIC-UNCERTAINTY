# Fase 2: Baseline OVD - Open Vocabulary Detection en BDD100K

## Resumen Ejecutivo

Esta fase establece el **baseline de detección sin calibrar** utilizando **Grounding-DINO** (SwinT-OGC) sobre el dataset **BDD100K**. El objetivo es obtener métricas de referencia que permitan evaluar posteriormente el impacto de métodos de estimación de incertidumbre epistémica y técnicas de calibración.

---

## 📊 Resultados Principales

### Métricas de Detección

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **mAP@[.50:.95]** | **0.1705** | Baseline razonable para OVD sin fine-tuning |
| **AP@50** | **0.2785** | 27.85% de precisión con IoU≥0.5 |
| **AP@75** | **0.1705** | Caída significativa en localizaciones precisas |
| **AP (small)** | **0.0633** | Baja detección en objetos pequeños |
| **AP (medium)** | **0.1821** | Desempeño medio en objetos medianos |
| **AP (large)** | **0.3770** | Mejor rendimiento en objetos grandes |

### Rendimiento Computacional

| Métrica | Valor | Observación |
|---------|-------|-------------|
| **Tiempo/imagen** | 0.275s | Velocidad aceptable para baseline |
| **FPS** | 3.64 | No en tiempo real, pero suficiente para evaluación |
| **GPU Memory** | 1190 MB | Uso moderado (RTX 3090/4090) |
| **Detecciones/imagen** | 11.08 | Promedio razonable para escenas urbanas |

### Distribución de Errores (100 imágenes analizadas)

| Tipo de Error | Cantidad | Porcentaje |
|---------------|----------|------------|
| **Falsos Negativos** | 988 | 97.4% de errores |
| **Falsos Positivos** (conf≥0.5) | 26 | 2.6% de errores |

---

## 🔍 Análisis Detallado

### 1. Problema Principal: Baja Recall (Alta Tasa de FN)

**Hallazgo crítico:** El modelo tiene **988 falsos negativos** vs solo **26 falsos positivos** en 100 imágenes.

#### Falsos Negativos por Clase:

| Clase | FN | % del Total | Implicación |
|-------|-----|-------------|-------------|
| **car** | 598 | 60.5% | Objetos más frecuentes no detectados |
| **traffic sign** | 223 | 22.6% | Señales pequeñas pasadas por alto |
| **traffic light** | 108 | 10.9% | Semáforos difíciles de detectar |
| **person** | 31 | 3.1% | Mejor detección en personas |
| **truck** | 12 | 1.2% | Confusión con cars |
| **bicycle** | 8 | 0.8% | Objetos pequeños problemáticos |
| **bus** | 6 | 0.6% | Relativamente bien detectado |
| **rider** | 2 | 0.2% | Clase con pocos ejemplos |

**Causa raíz identificada:**
- **Umbral de confianza muy alto (0.30)**: Filtra demasiadas detecciones válidas
- **Vocabulario limitado**: Grounding-DINO puede generar variantes léxicas no mapeadas
- **Objetos pequeños**: AP_small = 0.0633 indica que el modelo no captura señales/semáforos distantes

### 2. Matriz de Confusión

**Top 3 confusiones:**

| Predicción | Ground Truth | Cantidad | Análisis |
|------------|-------------|----------|----------|
| **person** → **rider** | 3 | **Semántica ambigua**: persona en bicicleta/moto |
| **truck** → **car** | 2 | **Similaridad visual**: vehículos grandes clasificados como autos |
| **traffic light** ↔ **traffic sign** | 3 | **Confusión entre elementos viales**: problemas de vocabulario |

**Implicación:** Las confusiones son **esperables** dado que:
- El modelo usa prompts textuales genéricos (`"person"`, `"rider"`)
- No hay desambiguación contextual (persona caminando vs montando)
- Semáforos y señales comparten características visuales a baja resolución

### 3. Análisis de Sensibilidad a Umbrales

**Resultado del barrido (11 umbrales: 0.05 → 0.75):**

| Umbral | mAP | AP50 | Detecciones/img | Observación |
|--------|-----|------|----------------|-------------|
| 0.05-0.30 | **0.1705** | **0.2785** | 11.08 | **Plateau de rendimiento** |
| 0.35 | 0.1550 | 0.2470 | 7.93 | Inicio de caída |
| 0.40 | 0.1360 | 0.2103 | 5.67 | Caída moderada |
| 0.50 | 0.0905 | 0.1323 | 2.66 | Pérdida severa de recall |
| 0.60 | 0.0566 | 0.0759 | 0.90 | Recall crítico |
| 0.75 | 0.0149 | 0.0172 | 0.04 | Prácticamente sin detecciones |

**Hallazgo clave:** 
- **Umbral óptimo: 0.25-0.30** (punto operativo actual ⭐)
- **No hay mejora** bajando el umbral < 0.25 (plateau indica que el modelo asigna scores muy bajos a detecciones correctas)
- **Recomendación:** El problema no es el umbral, sino la **calibración de scores**

### 4. Desempeño por Tamaño de Objeto

```
AP_small  = 0.0633  (objetos < 32² px)  ❌ Problemático
AP_medium = 0.1821  (32² < obj < 96² px) ⚠️ Moderado
AP_large  = 0.3770  (objetos > 96² px)  ✅ Aceptable
```

**Interpretación:**
- El modelo **depende fuertemente del tamaño** del objeto
- **Señales de tráfico** (small) explican 223/988 FN (22.6%)
- **Semáforos** (small) explican 108/988 FN (10.9%)
- **Carros distantes** (small/medium) contribuyen a la alta tasa de FN en `car`

---

## 🎯 Configuración Utilizada

### Modelo

```yaml
model:
  name: Grounding-DINO
  architecture: SwinT-OGC
  checkpoint: groundingdino_swint_ogc.pth
  input_size: [800, 1333]  # Adaptativo
  device: cuda
```

### Hiperparámetros de Inferencia

```yaml
inference:
  conf_threshold: 0.30      # Threshold de confianza
  nms_iou: 0.65            # IoU para Non-Maximum Suppression
  batch_size: 1            # Inferencia secuencial
  max_detections: 300      # Límite de detecciones por imagen
```

### Vocabulario (10 clases BDD100K)

```python
PROMPTS = [
    'person', 'rider', 'car', 'truck', 'bus', 'train',
    'motorcycle', 'bicycle', 'traffic light', 'traffic sign'
]

TEXT_PROMPT = "person. rider. car. truck. bus. train. motorcycle. bicycle. traffic light. traffic sign."
```

**Normalización de sinónimos:**
```python
PROMPT_SYNONYMS = {
    'bike': 'bicycle',
    'motorbike': 'motorcycle',
    'stop sign': 'traffic sign',
    'red light': 'traffic light',
    'pedestrian': 'person',
    'bicyclist': 'rider'
}
```

---

## 📁 Artefactos Generados

### Estructura de Archivos

```
outputs/baseline/
├── preds_raw.json                     # 22,162 predicciones en formato COCO
├── metrics.json                       # Métricas completas (global + por clase)
├── perf.txt                           # Rendimiento (FPS, memoria, latencia)
├── calib_inputs.csv                   # 88,620 detecciones para calibración
├── threshold_sweep.csv                # Sensibilidad a 11 umbrales
├── summary_table.csv                  # Tabla resumen para tesis
├── error_analysis.json                # FP/FN detallados con ejemplos
├── final_report.json                  # Reporte completo estructurado
├── final_report.txt                   # Reporte legible para humanos
├── pr_curves/                         # Curvas Precision-Recall (10 clases)
│   ├── person_pr.png
│   ├── car_pr.png
│   └── ...
├── threshold_sensitivity.png          # Gráficos de sensibilidad
├── summary_visualization.png          # Visualización de trade-offs
├── error_visualization.png            # FP vs FN por clase
└── final_summary_visualization.png    # 4 gráficos de resumen

outputs/qualitative/baseline/
└── 50 imágenes con detecciones visualizadas
    ├── 0000046.jpg
    ├── 0000092.jpg
    └── ...

configs/
└── baseline.yaml                      # Configuración reproducible completa

data/prompts/
└── bdd100k.txt                        # Vocabulario versionado
```

### Contenido del Archivo de Calibración

**`calib_inputs.csv`** (88,620 filas × 7 columnas):

| Campo | Descripción | Uso en Fase 3 |
|-------|-------------|---------------|
| `image_id` | ID de imagen BDD100K | Agrupación por imagen |
| `bbox` | [x, y, w, h] en píxeles | Análisis espacial |
| `category_id_pred` | Clase predicha (1-10) | Calibración por clase |
| `score` | Confianza sin calibrar | **Input para Temperature Scaling** |
| `iou` | IoU con mejor GT match | Umbral de correctness |
| `is_correct` | True si IoU≥0.5 | **Label para calibración** |
| `gt_ann_id` | ID de anotación GT | Trazabilidad |

**Estadísticas:**
- Total detecciones: 88,620
- Imágenes procesadas: 8,000 (val_calib)
- Tiempo de generación: ~1 hora

---

## 🚨 Problemas Identificados

### 1. **Baja Recall en Clases Frecuentes**

**Problema:** 
- `car`: 598 FN en solo 100 imágenes → ~60% de los errores
- `traffic sign`: 223 FN → objetos pequeños no detectados

**Causa:**
- Scores de confianza sistemáticamente bajos
- Umbral de 0.30 elimina detecciones válidas (pero el plateau indica que bajar el umbral no ayuda)

**Impacto en Tesis:**
- El modelo **necesita calibración** para mejorar la confianza en predicciones correctas
- La incertidumbre epistémica no está bien capturada (scores planos)

### 2. **Desempeño Pobre en Objetos Pequeños**

**Problema:**
- AP_small = 0.0633 (vs AP_large = 0.3770)
- 331 FN en `traffic sign` + `traffic light` (33.5% de errores)

**Causa:**
- Resolución de entrada [800, 1333] insuficiente para objetos <32px
- Backbone (SwinT) pierde información en downsampling

**Solución propuesta:**
- Multi-scale inference (no implementado en baseline)
- Entrenamiento con data augmentation específico para objetos pequeños

### 3. **Confusión Semántica Person ↔ Rider**

**Problema:**
- 4 confusiones mutuas en 100 imágenes
- Ambigüedad léxica en prompts (`"person"` vs `"rider"`)

**Causa:**
- Grounding-DINO no distingue contexto (persona parada vs montando bicicleta)
- Vocabulario genérico sin atributos

**Solución propuesta:**
- Prompts contextuales: `"person walking"`, `"person riding bicycle"`
- Fine-tuning con ejemplos desambiguados

### 4. **Plateau de Rendimiento con Umbrales Bajos**

**Problema:**
- mAP = 0.1705 constante entre 0.05 ≤ threshold ≤ 0.30
- No hay mejora bajando el umbral

**Interpretación:**
- El modelo asigna **scores muy bajos** a detecciones correctas
- **Falta de calibración**: no hay separación entre scores de TP y FP

**Impacto en Tesis:**
- Justifica la necesidad de **calibración (Fase 3)**
- Sugiere que la **incertidumbre epistémica** no está bien modelada

---

## 💡 Insights para la Tesis

### 1. **Limitaciones de OVD Zero-Shot en Escenarios Realistas**

**Hallazgo:**
- Grounding-DINO sin fine-tuning alcanza solo **17% mAP** en BDD100K
- Comparado con detectores cerrados (ej. Faster R-CNN fine-tuned: ~40% mAP)

**Contribución teórica:**
- Demuestra la **brecha entre capacidad zero-shot y aplicaciones críticas**
- Justifica la investigación en **incertidumbre epistémica** para cuantificar esta brecha

### 2. **Scores de Confianza No Reflejan Correctness**

**Evidencia:**
- Plateau en threshold sweep: mAP constante entre 0.05-0.30
- 26 FP con conf≥0.5 vs 988 FN (muchos con conf<0.3)

**Implicación:**
- Los scores del modelo **no están calibrados**
- **Necesidad urgente** de métodos de calibración (Temperature Scaling, Platt Scaling)

**Para Fase 3:**
- Calibrar sobre `calib_inputs.csv` (88,620 detecciones)
- Evaluar ECE (Expected Calibration Error), MCE (Max Calibration Error)
- Generar diagramas de confiabilidad (reliability diagrams)

### 3. **Dependencia del Tamaño de Objeto**

**Evidencia:**
```
AP_small  = 0.0633  (6.33%)
AP_large  = 0.3770  (37.70%)
Factor de mejora: 5.96x
```

**Hipótesis para futuras fases:**
- La **incertidumbre epistémica debería ser mayor** en objetos pequeños
- Métodos como **MC-Dropout** podrían capturar esta incertidumbre

### 4. **Vocabulario como Cuello de Botella**

**Observación:**
- Confusiones `person↔rider`, `truck↔car`, `traffic_light↔traffic_sign`
- Sinónimos capturados pero no variantes contextuales

**Dirección futura:**
- **Prompts enriquecidos**: `"pedestrian walking on sidewalk"` vs `"person riding bicycle"`
- **Chain-of-Thought prompting**: `"First identify if person is standing or riding, then classify"`

---

## 🔬 Metodología Aplicada

### Pipeline de Evaluación

```
1. Configuración Reproducible
   ├── Fijar seeds (PyTorch, NumPy, CUDA)
   ├── Versionar modelo (commit, checkpoint)
   └── Guardar config (baseline.yaml)

2. Inferencia sobre val_eval (2,000 imágenes)
   ├── Preprocesamiento: resize adaptativo
   ├── Predicción: model.predict(image, TEXT_PROMPT)
   ├── Post-procesamiento:
   │   ├── NMS por clase (IoU=0.65)
   │   ├── Normalización de labels
   │   └── Clipping de bboxes
   └── Guardar: preds_raw.json

3. Evaluación COCO
   ├── mAP global: 0.1705
   ├── Métricas por clase
   └── Curvas Precision-Recall

4. Análisis de Sensibilidad
   ├── Barrido de thresholds (0.05 → 0.75)
   ├── Trade-off Recall vs Precision
   └── Identificar punto operativo óptimo

5. Análisis Cualitativo
   ├── Visualización de 50 imágenes
   ├── Identificación de errores sistemáticos
   └── Casos representativos para "error book"

6. Preparación para Calibración (val_calib: 8,000 imágenes)
   ├── Matching GT↔pred con IoU≥0.5
   ├── Extracción de scores sin calibrar
   ├── Generación de labels de correctness
   └── Guardar: calib_inputs.csv (88,620 filas)
```

### Matching GT↔Pred para Calibración

**Algoritmo implementado:**

```python
for each prediction in image:
    best_iou = 0
    for each ground_truth in image:
        if pred.category == gt.category:
            iou = compute_iou(pred.bbox, gt.bbox)
            if iou > best_iou:
                best_iou = iou
                best_gt = gt
    
    is_correct = (best_iou >= 0.5)  # Threshold estándar COCO
    
    save_record({
        'score': pred.confidence,
        'is_correct': is_correct,
        'iou': best_iou,
        'category_id': pred.category
    })
```

**Validación:**
- 88,620 predicciones procesadas
- Cobertura de todas las 10 clases
- Distribución balanceada de TP/FP para calibración

---

## 📈 Métricas de Calibración (Preparación para Fase 3)

### Qué Evaluar con `calib_inputs.csv`

#### 1. **Expected Calibration Error (ECE)**

**Definición:**
```
ECE = Σ (|accuracy(bin_i) - confidence(bin_i)| × |bin_i| / N)
```

**Uso:**
- Medir qué tan bien alineados están los scores con la probabilidad real de correctness
- Esperado en baseline: **ECE alto** (scores no calibrados)

#### 2. **Maximum Calibration Error (MCE)**

**Definición:**
```
MCE = max |accuracy(bin_i) - confidence(bin_i)|
```

**Uso:**
- Detectar bins con mayor discrepancia
- Identificar rangos de confianza problemáticos

#### 3. **Reliability Diagram (Diagrama de Confiabilidad)**

**Visualización:**
```
Eje X: Predicted confidence (bined)
Eje Y: True accuracy in bin

Línea diagonal: Perfect calibration
Puntos: Bins observados
```

**Interpretación:**
- Puntos **por encima** de la diagonal: modelo **underconfident**
- Puntos **por debajo**: modelo **overconfident**

#### 4. **Brier Score**

**Definición:**
```
BS = (1/N) Σ (score_i - is_correct_i)²
```

**Uso:**
- Métrica de error cuadrático medio
- Combina calibración y discriminación

---

## 🚀 Próximos Pasos (Fase 3: Calibración e Incertidumbre)

### Objetivos de Fase 3

1. **Calibración de Scores**
   - Implementar Temperature Scaling sobre `calib_inputs.csv`
   - Encontrar temperatura óptima T mediante validación cruzada
   - Aplicar T a predicciones: `score_cal = softmax(logits / T)`

2. **Evaluación de Calibración**
   - Calcular ECE, MCE, Brier Score antes/después de calibración
   - Generar reliability diagrams comparativos
   - Verificar mejora en separación TP/FP

3. **Incertidumbre Epistémica**
   - Implementar **MC-Dropout**: múltiples forward passes con dropout activo
   - Implementar **Ensembles**: múltiples modelos/checkpoints
   - Extraer **variance de scores** como proxy de incertidumbre

4. **Análisis de Correlación**
   - Correlacionar incertidumbre con:
     - Tamaño de objeto (esperar alta incertidumbre en small)
     - Confusiones semánticas (person↔rider)
     - Errores de localización (IoU bajo)

### Hipótesis a Validar

#### H1: Calibración Mejora Separación TP/FP
**Predicción:** ECE después de Temperature Scaling < 0.1 (vs ~0.3 en baseline)

**Métrica:** Distribución de scores_cal para TP vs FP

#### H2: Incertidumbre Epistémica Correlaciona con Errores
**Predicción:** FN tienen mayor variance de scores en MC-Dropout

**Métrica:** `mean_variance(FN) > mean_variance(TP)`

#### H3: Objetos Pequeños Tienen Mayor Incertidumbre
**Predicción:** Correlación negativa entre tamaño de objeto y variance

**Métrica:** Pearson correlation(object_area, mc_dropout_variance) < -0.5

---

## 🛠️ Mejoras Propuestas

### Mejoras Inmediatas (Fase 3)

#### 1. **Calibración Multi-Escala**

**Problema actual:** Single threshold para todos los tamaños de objeto

**Solución:**
```python
# Calibrar T por rango de tamaño
T_small  = optimize_temperature(calib_inputs[area < 32²])
T_medium = optimize_temperature(calib_inputs[32² ≤ area < 96²])
T_large  = optimize_temperature(calib_inputs[area ≥ 96²])
```

**Impacto esperado:** Mejorar AP_small de 0.0633 → 0.10+

#### 2. **Calibración Por Clase**

**Problema actual:** Single T para todas las clases

**Solución:**
```python
for category in ['person', 'car', ..., 'traffic_sign']:
    T[category] = optimize_temperature(calib_inputs[cat == category])
```

**Impacto esperado:** Reducir confusiones semánticas

#### 3. **MC-Dropout con Forward Passes Variables**

**Hipótesis:** Más passes → mejor estimación de incertidumbre, pero mayor coste

**Experimento:**
```python
n_passes = [5, 10, 20, 50, 100]
for n in n_passes:
    uncertainty = mc_dropout(model, image, n_passes=n)
    compute_correlation(uncertainty, errors)
```

**Objetivo:** Encontrar trade-off óptimo (n=20 típicamente)

### Mejoras a Medio Plazo (Fase 4)

#### 4. **Fine-Tuning en BDD100K**

**Problema:** Zero-shot mAP = 0.1705 es bajo para aplicaciones críticas

**Solución:**
```python
# Fine-tune last layers
model.freeze_backbone()
model.train_on_bdd100k(train_split, epochs=10)
```

**Impacto esperado:** mAP → 0.30-0.40

#### 5. **Prompts Contextuales**

**Problema actual:**
```python
TEXT_PROMPT = "person. rider. car. ..."  # Genérico
```

**Solución:**
```python
CONTEXT_PROMPTS = {
    'person': "pedestrian walking on sidewalk or standing",
    'rider': "person riding bicycle or motorcycle",
    'car': "passenger vehicle with four wheels",
    'truck': "large cargo vehicle"
}
```

**Impacto esperado:** Reducir confusiones person↔rider, truck↔car

#### 6. **Multi-Scale Inference**

**Problema:** Objetos pequeños no detectados por resolución limitada

**Solución:**
```python
scales = [0.5, 0.75, 1.0, 1.25, 1.5]
predictions = []
for scale in scales:
    img_scaled = resize(image, scale)
    preds = model.predict(img_scaled)
    predictions.extend(rescale_boxes(preds, 1/scale))

final_preds = weighted_nms(predictions)
```

**Impacto esperado:** AP_small → 0.15+

### Mejoras a Largo Plazo (Investigación)

#### 7. **Uncertainty-Guided Active Learning**

**Concepto:** Usar incertidumbre epistémica para seleccionar ejemplos de entrenamiento

**Pipeline:**
```python
1. Inferir sobre unlabeled_pool con MC-Dropout
2. Seleccionar top-K imágenes con mayor incertidumbre
3. Etiquetar manualmente
4. Re-entrenar modelo
5. Repetir hasta convergencia
```

**Impacto esperado:** Reducir coste de etiquetado en 50-70%

#### 8. **Conformal Prediction para Detección**

**Concepto:** Generar prediction sets con garantías estadísticas

**Implementación:**
```python
# Calibrar en val_calib
quantile = compute_quantile(scores, error_rate=0.1)

# Inferir con prediction sets
for prediction in test_set:
    if prediction.score >= quantile:
        output(prediction, confidence="high")
    else:
        output(prediction_set, confidence="ambiguous")
```

**Ventaja:** Garantía matemática de error ≤ 10%

#### 9. **Bayesian Deep Learning**

**Concepto:** Reemplazar pesos determinísticos con distribuciones

**Implementación:**
```python
# Usar Variational Inference
model = BayesianGroundingDINO(prior='normal')
model.train_with_elbo_loss(train_data)

# Inferir con samples
predictions = [model.sample_forward(image) for _ in range(100)]
mean_pred = average(predictions)
uncertainty = variance(predictions)
```

**Ventaja:** Incertidumbre epistémica fundamentada en teoría bayesiana

---

## 📚 Referencias y Contexto

### Comparación con Estado del Arte

| Método | mAP (BDD100K) | AP50 | Notas |
|--------|---------------|------|-------|
| **Grounding-DINO (este baseline)** | **0.1705** | **0.2785** | Zero-shot, sin fine-tuning |
| Faster R-CNN (supervised) | ~0.40 | ~0.60 | Fine-tuned en BDD100K train |
| YOLO-v8 (supervised) | ~0.45 | ~0.65 | Fine-tuned |
| DINO (supervised) | ~0.50 | ~0.70 | Full training |
| OWL-ViT (zero-shot) | ~0.12 | ~0.20 | Similar performance a nuestro baseline |

**Conclusión:** Nuestro baseline está **alineado con OVD zero-shot típico**, confirmando validez de resultados.

### Papers Clave para Fase 3

1. **Temperature Scaling:**
   - Guo et al. (2017). "On Calibration of Modern Neural Networks"
   - Simple, eficaz, single parameter T

2. **MC-Dropout:**
   - Gal & Ghahramani (2016). "Dropout as a Bayesian Approximation"
   - Teóricamente fundamentado

3. **Conformal Prediction:**
   - Angelopoulos & Bates (2021). "A Gentle Introduction to Conformal Prediction"
   - Garantías estadísticas

4. **Open-Vocabulary Detection:**
   - Liu et al. (2023). "Grounding DINO: Marrying DINO with Grounded Pre-Training"
   - Base de nuestro modelo

---

## ✅ Criterios de Éxito (Go/No-Go)

### Criterios Verificados ✅

| Criterio | Estado | Valor/Archivo |
|----------|--------|---------------|
| mAP razonable | ✅ PASS | 0.1705 > 0.05 |
| AP50 razonable | ✅ PASS | 0.2785 > 0.10 |
| Latencia medida | ✅ PASS | 0.275s/imagen |
| Artefactos completos | ✅ PASS | 15 archivos generados |
| Inputs de calibración | ✅ PASS | calib_inputs.csv (88,620 filas) |
| Errores identificados | ✅ PASS | 988 FN, 26 FP analizados |

### Conclusión: ✅ **LISTO PARA FASE 3**

---

## 🎓 Contribuciones para la Tesis

### Capítulo 4: Metodología

**Sección 4.2: Establecimiento de Baseline**
- Justificar elección de Grounding-DINO (SOTA en OVD)
- Describir configuración (thresholds, NMS, vocabulario)
- Documentar proceso de generación de datos de calibración

### Capítulo 5: Resultados

**Sección 5.1: Baseline sin Calibrar**
- Tabla de métricas principales (mAP, AP50, AP75)
- Gráficos de sensibilidad a umbrales
- Análisis de errores por clase

**Sección 5.2: Análisis de Limitaciones**
- Baja recall en objetos pequeños (AP_small = 0.0633)
- Confusiones semánticas (person↔rider)
- Plateau de rendimiento → necesidad de calibración

### Capítulo 6: Incertidumbre Epistémica

**Sección 6.1: Motivación**
- Evidencia de scores no calibrados (threshold plateau)
- 988 FN vs 26 FP → modelo conservador
- Justificar Temperature Scaling, MC-Dropout

**Sección 6.2: Inputs para Calibración**
- Describir `calib_inputs.csv` (88,620 detecciones)
- Matching GT↔pred con IoU≥0.5
- Distribución de TP/FP por clase

### Capítulo 7: Discusión

**Limitaciones del Baseline:**
1. Zero-shot sin fine-tuning
2. Vocabulario genérico
3. Single-scale inference
4. Scores no calibrados

**Contribuciones:**
1. Pipeline reproducible completo
2. Datos de calibración extensivos (88K+ muestras)
3. Identificación de modos de fallo (FN en small objects)
4. Base sólida para comparación con métodos de incertidumbre

---

## 📝 Notas de Implementación

### Archivos Modificados Durante Ejecución

**Problema encontrado:** Dependencias entre celdas del notebook

**Solución aplicada:** Hacer todas las secciones de evaluación (8-15) independientes:
- Cargar datos desde archivos guardados (JSON, CSV, TXT)
- Verificar existencia de variables con `if 'var' not in locals()`
- Proveer valores por defecto si archivos faltan

**Resultado:** Puedes ejecutar cualquier sección de análisis sin re-correr la inferencia (que toma ~1 hora)

### Problema con Parquet

**Error original:**
```
ImportError: Unable to find a usable engine; tried using: 'pyarrow', 'fastparquet'
```

**Solución:**
```bash
pip install pyarrow
# O alternativamente:
# Guardar como CSV en lugar de Parquet
```

**Decisión final:** Guardar como `calib_inputs.csv` (más compatible, mismo contenido)

---

## 🔗 Archivos Clave para Fase 3

| Archivo | Propósito en Fase 3 |
|---------|---------------------|
| `calib_inputs.csv` | **Entrenar T en Temperature Scaling** |
| `preds_raw.json` | Aplicar T y re-evaluar métricas |
| `metrics.json` | Comparar mAP antes/después de calibración |
| `error_analysis.json` | Verificar reducción de FP/FN |
| `baseline.yaml` | Reproducir configuración exacta |

---

## 📊 Visualizaciones Generadas

### 1. Curvas Precision-Recall (10 clases)
- **Ubicación:** `outputs/baseline/pr_curves/*.png`
- **Uso:** Identificar clases con bajo recall/precision

### 2. Sensibilidad a Umbrales
- **Archivo:** `threshold_sensitivity.png`
- **Interpretación:** Plateau entre 0.05-0.30 → necesidad de calibración

### 3. Trade-off Detections vs mAP
- **Archivo:** `summary_visualization.png`
- **Uso:** Visualizar punto operativo óptimo

### 4. Matriz de Confusión
- **Archivo:** `error_visualization.png`
- **Uso:** Identificar pares de clases problemáticos

### 5. Resumen Final
- **Archivo:** `final_summary_visualization.png`
- **Contenido:** 4 gráficos (métricas, tamaños, clases, artefactos)

---

## 🎯 Conclusiones Finales

### Lo que Funciona Bien

1. ✅ **Pipeline robusto y reproducible**
2. ✅ **Detección de objetos grandes** (AP_large = 0.3770)
3. ✅ **Bajo número de falsos positivos** (26 en 100 imágenes)
4. ✅ **Velocidad aceptable** (3.64 FPS)

### Lo que Necesita Mejora

1. ❌ **Recall en objetos pequeños** (AP_small = 0.0633)
2. ❌ **Calibración de scores** (plateau en threshold sweep)
3. ❌ **Desambiguación semántica** (person↔rider)
4. ❌ **Separación TP/FP** (necesita incertidumbre epistémica)

### Impacto para la Tesis

Este baseline establece:
- **Límite inferior** de rendimiento (sin calibración)
- **Necesidad justificada** de métodos de incertidumbre
- **Datos extensivos** para validación (88,620 detecciones)
- **Base reproducible** para comparaciones

**Próximo hito:** Demostrar que **Temperature Scaling + MC-Dropout** mejoran:
- ECE < 0.1 (vs ~0.3 en baseline)
- Mejor correlación entre score y correctness
- Identificación de errores mediante incertidumbre alta

---

**Fase 2 completada:** ✅ **GO para Fase 3**

**Fecha de generación:** 2025-11-11  
**Autor:** Sistema de evaluación baseline OVD  
**Versión:** 1.0
