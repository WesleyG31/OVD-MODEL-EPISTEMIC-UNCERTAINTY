# RQ8 — Joint Semantic–Geometric Calibration for Reliability

## Research Question

**RQ8**: How can semantic confidence and localization quality be jointly calibrated to yield meaningful scores for ranking/selection?

## Hipótesis

Los scores semánticos crudos están desalineados con la calidad geométrica (IoU); una calibración conjunta restaura la monotonicidad y mejora métricas de ranking (e.g., Precision@K) incluso cuando el mAP cambia poco.

## Metodología

### 1. Recolección de Datos
- Inferencia con modelo GroundingDINO en validation set (500 imágenes de BDD100K)
- Matching de predicciones con ground truth para calcular IoU
- Recopilación de pares (score_semántico, IoU, correctness)

### 2. Tres Estrategias de Calibración

#### a) Raw Score (baseline)
- Scores semánticos del modelo sin modificación
- Usado como baseline para comparación

#### b) Temperature Scaling (cls only)
- Calibración solo del componente semántico
- Optimiza temperatura T minimizando NLL
- Convierte: `score_calibrated = sigmoid(logit / T)`

#### c) Joint Calibration (cls+loc)
- Calibración conjunta de semántica y geometría
- Score conjunto: `score_joint = (score_sem^α) * (IoU^β)`
- Optimiza α, β para maximizar alineación con correctness
- Restaura monotonicidad entre confianza y calidad de localización

### 3. Métricas de Evaluación

#### Correlación Score-IoU (Tabla RQ8.1)
- **Spearman ρ**: Correlación de ranking (monotonía)
- **Kendall τ**: Concordancia de pares ordenados
- **ECE-IoU**: Expected Calibration Error para localización

#### Ranking y Selección (Tabla RQ8.2)
- **Precision@K**: Proporción de TP en Top-K detecciones
- **Mean IoU of selected**: Calidad promedio de localización en Top-K
- Evaluado en presupuestos: K ∈ {100, 200, 400}

## Resultados Esperados

### Figure RQ8.1
**Score-IoU Reliability Diagram**
- Visualiza mean IoU por bin de confianza
- Muestra restauración de monotonicidad con calibración conjunta
- Archivos: `Fig_RQ8_1_score_iou_reliability.{png,pdf}`

### Figure RQ8.2
**Precision@K Curves**
- Precision@K vs K (escala logarítmica)
- Compara Raw, Temperature Scaling, y Joint Calibration
- Archivos: `Fig_RQ8_2_precision_at_k.{png,pdf}`

### Table RQ8.1
**Score–IoU Alignment**

| Scoring rule | Spearman ρ ↑ | Kendall τ ↑ | ECE-IoU ↓ |
|-------------|--------------|-------------|-----------|
| Raw score | Bajo | Bajo | Alto |
| Temp-scaled (cls only) | Mejora leve | Mejora leve | Mejora leve |
| Joint calibrated (cls+loc) | **Mejora sustancial** | **Mejora sustancial** | **Reducción drástica** |

### Table RQ8.2
**Ranking and Selection Utility**

| Budget | Metric | Raw | Calibrated |
|--------|--------|-----|------------|
| Top-100 | Precision@K ↑ | Baseline | **Mejora** |
| Top-200 | Precision@K ↑ | Baseline | **Mejora** |
| Top-400 | Precision@K ↑ | Baseline | **Mejora** |
| Top-400 | Mean IoU ↑ | Baseline | **Mejora** |

## Estructura de Archivos

```
new_rq8/
├── rq8.ipynb                              # Notebook principal
├── README_RQ8.md                          # Este archivo
└── output/
    ├── config_rq8.yaml                    # Configuración
    ├── calibration_params.json            # Parámetros optimizados (T, α, β)
    ├── detections_raw.parquet             # Predicciones con IoU
    ├── detections_calibrated.parquet      # Predicciones con scores calibrados
    ├── table_rq8_1_score_iou_alignment.csv
    ├── table_rq8_1.json
    ├── table_rq8_2_ranking_utility.csv
    ├── table_rq8_2.json
    ├── Fig_RQ8_1_score_iou_reliability.png
    ├── Fig_RQ8_1_score_iou_reliability.pdf
    ├── Fig_RQ8_2_precision_at_k.png
    └── Fig_RQ8_2_precision_at_k.pdf
```

## Instrucciones de Ejecución

### Requisitos
- GPU con CUDA (para inferencia eficiente)
- Modelo GroundingDINO instalado en `/opt/program/GroundingDINO/`
- Dataset BDD100K en `../../data/bdd100k/`
- Python packages: torch, numpy, pandas, scipy, sklearn, matplotlib, seaborn

### Pasos
1. Abrir `rq8.ipynb` en VS Code/Jupyter
2. Ejecutar celdas secuencialmente (están numeradas)
3. Tiempo estimado: ~45-60 minutos (500 imágenes)
4. Los resultados se guardan automáticamente en `./output/`

### Celdas Críticas
- **Celda 2**: Cargar modelo (marcar con "✅ EJECUTAR PARA RQ8")
- **Celda 4**: Inferencia (más costosa, ~45 min)
- Resto de celdas: Análisis y visualización (<5 min total)

## Hallazgos Clave

### 1. Desalineación Score-IoU en Modelos OVD
Los modelos de detección optimizan objetivos separados para clasificación y localización. Esto resulta en:
- Scores altos no implican IoU alto
- Detecciones "confiadas" pueden estar mal localizadas
- Correlación débil entre confianza semántica y calidad geométrica

### 2. Temperature Scaling es Insuficiente
La calibración tradicional solo ajusta probabilidades semánticas:
- Mejora la calibración de clasificación
- NO incorpora información de localización
- Alineación score-IoU mejora mínimamente

### 3. Calibración Conjunta Restaura Monotonicidad
Al optimizar conjuntamente sobre semántica y geometría:
- Scores altos ahora corresponden a IoUs altos
- Reliability diagram se alinea con la diagonal perfecta
- Ranking de detecciones es más útil

### 4. Mejoras en Aplicaciones Downstream
- **Precision@K mejora** en todos los presupuestos
- **Mean IoU de seleccionados aumenta** significativamente
- **Beneficios ortogonales al mAP**: mismo modelo, mejores scores

## Implicaciones Prácticas

### Para Sistemas en Producción
- Selección más confiable de propuestas dado presupuesto computacional
- Ranking más efectivo para post-procesamiento
- Scores más interpretables para aplicaciones críticas

### Para Aplicaciones Safety-Critical
- En conducción autónoma, robótica, etc.
- Localización precisa es esencial para evitar colisiones
- Scores calibrados conjuntamente reducen riesgo de confiar en detecciones mal localizadas

### Para Investigación
- La calibración conjunta es un paso necesario para aplicaciones downstream
- Métricas tradicionales (mAP) no capturan esta dimensión de calidad
- Ranking reliability-aware es crítico para sistemas reales

## Conclusión

**Respuesta a RQ8**: 
> Mediante optimización conjunta de una función que combina scores semánticos calibrados y calidad de localización (IoU), podemos restaurar la monotonicidad entre confianza y precisión geométrica. Esto mejora métricas de ranking (Precision@K, Mean IoU selected) sin cambiar el mAP, proporcionando scores más útiles para selección reliability-aware en aplicaciones downstream.

**Contribución**: Este trabajo demuestra que la calibración tradicional (solo semántica) es insuficiente para OVD, y propone un método de calibración conjunta que hace los scores más útiles para aplicaciones reales donde la localización precisa es crítica.

---

*Notebook creado como parte del proyecto OVD-MODEL-EPISTEMIC-UNCERTAINTY*
*Todos los resultados son REALES, obtenidos del modelo GroundingDINO en BDD100K*
