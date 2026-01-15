# 🏗️ RQ5 - ARQUITECTURA TÉCNICA

## 📐 Diagrama de Flujo de Datos

```
┌────────────────────────────────────────────────────────────────┐
│                    INPUTS (Fase 5)                             │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  📁 ../../fase 5/outputs/comparison/                          │
│     ├─ detection_comparison.csv        (mAP, AP50, AP75)     │
│     ├─ calibration_comparison.csv      (ECE, NLL, Brier)     │
│     ├─ uncertainty_auroc_comparison.csv (AUROC)               │
│     ├─ risk_coverage_auc.json          (AUC-RC)              │
│     ├─ temperatures.json                (T_opt)               │
│     ├─ eval_baseline.csv                (predictions + TP/FP) │
│     ├─ eval_mc_dropout.csv              (with uncertainty)    │
│     └─ eval_mc_dropout_ts.csv           (with calibration)    │
│                                                                │
│  📁 ../../data/bdd100k_coco/                                  │
│     └─ labels/det_val_coco.json        (ground truth)        │
│                                                                │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                 PROCESSING PIPELINE                            │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  STEP 1: Load and Prepare Data                                │
│  ────────────────────────────                                 │
│    • Load predictions (baseline, MC-Dropout)                  │
│    • Load ground truth                                        │
│    • Verify TP/FP matching                                    │
│                                                                │
│  STEP 2: Compute Risk Scores                                  │
│  ────────────────────────                                     │
│    Baseline:                                                  │
│      risk_baseline = 1 - confidence_score                     │
│                                                                │
│    Fused:                                                     │
│      unc_norm = (unc - unc_min) / (unc_max - unc_min)        │
│      risk_fused = 0.5*(1 - score) + 0.5*unc_norm             │
│                                                                │
│  STEP 3: Selective Prediction                                 │
│  ────────────────────────                                     │
│    For each coverage level (100%, 80%, 60%):                 │
│      1. Sort predictions by risk (ascending)                  │
│      2. Retain top N% predictions                             │
│      3. Calculate risk = FP / Total Retained                  │
│                                                                │
│  STEP 4: FP/FN Analysis                                       │
│  ────────────────────                                         │
│    • Count TP, FP in predictions                              │
│    • Calculate FN = GT Objects - TP                           │
│    • Compute FP Rate = FP / Total Predictions                 │
│    • Compute FN Rate = FN / Total GT Objects                  │
│                                                                │
│  STEP 5: Visualization                                        │
│  ────────────────────                                         │
│    • Generate Figure 5.1 (architecture diagram)               │
│    • Generate Figure 5.2 (risk-coverage curves)               │
│                                                                │
│  STEP 6: Export and Report                                    │
│  ────────────────────────                                     │
│    • Save tables as CSV                                       │
│    • Save figures as PNG/PDF                                  │
│    • Generate text report                                     │
│    • Create JSON summary                                      │
│                                                                │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                    OUTPUTS (./outputs/)                        │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  📊 Tables:                                                    │
│     ├─ table_5_1_selective_prediction.csv                     │
│     ├─ table_5_2_fp_reduction.csv                             │
│     ├─ baseline_risk.csv                                      │
│     ├─ fused_risk.csv                                         │
│     └─ risk_coverage_curves_data.csv                          │
│                                                                │
│  🖼️ Figures:                                                   │
│     ├─ figure_5_1_decision_fusion_architecture.png            │
│     ├─ figure_5_1_decision_fusion_architecture.pdf            │
│     ├─ figure_5_2_risk_coverage_tradeoff.png                  │
│     └─ figure_5_2_risk_coverage_tradeoff.pdf                  │
│                                                                │
│  📝 Reports:                                                   │
│     ├─ RQ5_FINAL_REPORT.txt                                   │
│     ├─ rq5_summary.json                                       │
│     └─ config_rq5.yaml                                        │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## 🧮 Algoritmos Clave

### 1. Risk Score Calculation

#### Baseline Risk:
```python
def compute_risk_baseline(predictions):
    """
    Risk = 1 - confidence_score
    
    Intuición: Mayor confianza → Menor riesgo
    """
    return 1 - predictions['score']
```

#### Fused Risk:
```python
def compute_risk_fused(predictions):
    """
    Risk = α*(1 - score) + β*uncertainty_normalized
    
    donde α = β = 0.5 (ponderación igual)
    
    Intuición: Combina confianza + incertidumbre
    """
    # Normalizar incertidumbre a [0, 1]
    unc = predictions['uncertainty_epistemic']
    unc_norm = (unc - unc.min()) / (unc.max() - unc.min() + 1e-10)
    
    # Fusión con pesos iguales
    alpha = 0.5
    beta = 0.5
    
    risk = alpha * (1 - predictions['score']) + beta * unc_norm
    
    return risk
```

**Justificación de pesos α=β=0.5**:
- Sin optimización previa → pesos iguales
- Baseline: Trabajo futuro optimizar α, β con grid search
- Resultado: Mejora significativa incluso con pesos simples

---

### 2. Selective Prediction

```python
def evaluate_selective_prediction(predictions, coverage_pct):
    """
    Retiene solo las predicciones más confiables (menor riesgo)
    
    Args:
        predictions: DataFrame con columnas ['risk', 'is_tp']
        coverage_pct: Porcentaje de predicciones a retener (0-100)
    
    Returns:
        risk: Tasa de error en predicciones retenidas
    """
    # Ordenar por riesgo (menor a mayor)
    sorted_preds = predictions.sort_values('risk', ascending=True)
    
    # Retener top coverage_pct
    n_retain = int(len(sorted_preds) * coverage_pct / 100)
    retained = sorted_preds.iloc[:n_retain]
    
    # Calcular riesgo en retenidas
    n_fp = (~retained['is_tp']).sum()
    risk = n_fp / len(retained) if len(retained) > 0 else 0
    
    return risk
```

**Interpretación**:
- Coverage 100% → Todas las predicciones → Mayor riesgo
- Coverage 80% → Rechaza 20% más inciertas → Menor riesgo
- Coverage 60% → Rechaza 40% más inciertas → Menor riesgo aún

**Trade-off**: Coverage ↓ → Risk ↓ pero Coverage ↓ → Recall ↓

---

### 3. FP/FN Rate Calculation

```python
def compute_fp_fn_rates(predictions, n_gt_objects):
    """
    Calcula tasas de falsos positivos y falsos negativos
    
    Args:
        predictions: DataFrame con columna 'is_tp'
        n_gt_objects: Número total de objetos en ground truth
    
    Returns:
        fp_rate: FP / Total Predictions
        fn_rate: FN / Total GT Objects
    """
    n_tp = predictions['is_tp'].sum()
    n_fp = (~predictions['is_tp']).sum()
    n_fn = n_gt_objects - n_tp  # Objetos no detectados
    
    fp_rate = n_fp / len(predictions)
    fn_rate = n_fn / n_gt_objects
    
    return fp_rate, fn_rate
```

**Métricas clave**:
- **FP Rate**: Proporción de predicciones incorrectas
  - Alto FP → Detecciones fantasma → Frenados innecesarios
- **FN Rate**: Proporción de objetos no detectados
  - Alto FN → Objetos perdidos → Colisiones potenciales

**Para ADAS**: FP más crítico que FN (sensores redundantes)

---

### 4. Risk-Coverage Curve Generation

```python
def compute_risk_coverage_curve(predictions, n_points=50):
    """
    Genera curva completa Risk vs Coverage
    
    Args:
        predictions: DataFrame con ['risk', 'is_tp']
        n_points: Número de puntos en la curva
    
    Returns:
        coverages: Array de coverage (100% → 10%)
        risks: Array de risk para cada coverage
    """
    sorted_preds = predictions.sort_values('risk', ascending=True)
    
    coverages = []
    risks = []
    
    for cov_pct in np.linspace(100, 10, n_points):
        n_retain = int(len(sorted_preds) * cov_pct / 100)
        if n_retain > 0:
            retained = sorted_preds.iloc[:n_retain]
            n_fp = (~retained['is_tp']).sum()
            risk = n_fp / len(retained)
            
            coverages.append(cov_pct)
            risks.append(risk)
    
    return np.array(coverages), np.array(risks)
```

**Visualización**: Curva descendente = Mejor (menos coverage, menos riesgo)

---

## 📊 Esquemas de Datos

### Input Schema: `eval_baseline.csv`

```
┌─────────────┬──────────┬───────────┬───────┬────────┬─────────┐
│ image_id    │ category │ score     │ bbox  │ is_tp  │ iou     │
├─────────────┼──────────┼───────────┼───────┼────────┼─────────┤
│ str         │ str      │ float     │ list  │ bool   │ float   │
│ "img_0001"  │ "car"    │ 0.8523    │ [...]  │ True   │ 0.73    │
└─────────────┴──────────┴───────────┴───────┴────────┴─────────┘
```

### Input Schema: `eval_mc_dropout.csv`

```
┌─────────────┬──────────┬───────┬────────────────────┬───────┬────────┐
│ image_id    │ category │ score │ uncertainty_epist. │ is_tp │ ...    │
├─────────────┼──────────┼───────┼────────────────────┼───────┼────────┤
│ str         │ str      │ float │ float              │ bool  │ ...    │
│ "img_0001"  │ "car"    │ 0.852 │ 0.000043           │ True  │        │
└─────────────┴──────────┴───────┴────────────────────┴───────┴────────┘
```

### Output Schema: `table_5_1_selective_prediction.csv`

```
┌──────────┬────────────┬──────┬──────┬──────┬──────────┐
│ coverage │ n_retained │ n_fp │ n_tp │ risk │ method   │
├──────────┼────────────┼──────┼──────┼──────┼──────────┤
│ int      │ int        │ int  │ int  │ float│ str      │
│ 100      │ 25000      │ 4650 │ ...  │ 0.186│ Baseline │
│ 100      │ 25000      │ 3725 │ ...  │ 0.149│ Fused    │
└──────────┴────────────┴──────┴──────┴──────┴──────────┘
```

### Output Schema: `table_5_2_fp_reduction.csv`

```
┌──────────────────┬─────────┬─────────┬──────┬──────┬──────┬──────────┐
│ Method           │ FP_Rate │ FN_Rate │ n_TP │ n_FP │ n_FN │ Coverage │
├──────────────────┼─────────┼─────────┼──────┼──────┼──────┼──────────┤
│ str              │ float   │ float   │ int  │ int  │ int  │ float    │
│ Baseline         │ 0.184   │ 0.071   │ ...  │ ...  │ ...  │ 100.0    │
│ Decision Fusion  │ 0.097   │ 0.078   │ ...  │ ...  │ ...  │ 80.0     │
└──────────────────┴─────────┴─────────┴──────┴──────┴──────┴──────────┘
```

---

## 🔧 Configuración Técnica

### Parámetros Globales:

```yaml
# config_rq5.yaml
seed: 42                                    # Reproducibilidad
coverage_levels: [100, 80, 60]              # Niveles de evaluación
iou_threshold: 0.5                          # Matching threshold
categories: [person, rider, car, ...]       # 10 clases BDD100K
```

### Paths Relativos:

```python
BASE_DIR = Path('../..')                    # Raíz del proyecto
OUTPUT_DIR = Path('./outputs')              # Salida de RQ5
fase5_dir = BASE_DIR / 'fase 5' / 'outputs' / 'comparison'
gt_file = BASE_DIR / 'data' / 'bdd100k_coco' / 'labels' / 'det_val_coco.json'
```

### Librerías Requeridas:

```python
# Core
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Utils
from pathlib import Path
import json
import yaml

# COCO
from pycocotools.coco import COCO
```

---

## 🎨 Especificaciones de Visualización

### Figure 5.1 - Architecture Diagram

**Tipo**: Diagrama de flujo con cajas y flechas

**Componentes**:
- Cajas: FancyBboxPatch con bordes redondeados
- Flechas: FancyArrowPatch con arrowstyle='->'
- Colores: Diferenciados por etapa (input, detector, uncertainty, calibration, fusion, output)

**Dimensiones**: 14x10 inches, 300 DPI

**Formato**: PNG + PDF (ambos guardados)

### Figure 5.2 - Risk-Coverage Curves

**Tipo**: Gráfico de líneas con marcadores

**Elementos**:
- Línea roja (●): Baseline Risk
- Línea verde (■): Fused Risk
- Área sombreada: Región de mejora (entre curvas)
- Puntos destacados: Coverage 100%, 80%, 60%

**Ejes**:
- X: Coverage (%) - De 100 a 10 (derecha a izquierda)
- Y: Risk (FP Rate) - De 0 a max(risk)

**Dimensiones**: 10x7 inches, 300 DPI

**Formato**: PNG + PDF

---

## ⚙️ Optimizaciones

### 1. Reutilización de Datos

**Ventaja**: No re-ejecutar Fase 3, 4, 5 (ahorra ~2 horas)

```python
# En lugar de:
# predictions = run_mc_dropout(model, images, K=5)

# Hacemos:
predictions = pd.read_csv('../../fase 5/outputs/comparison/eval_mc_dropout.csv')
```

### 2. Cálculos Vectorizados

**Ventaja**: NumPy/Pandas más rápido que loops

```python
# Vectorizado (rápido)
risk = 0.5 * (1 - df['score']) + 0.5 * unc_norm

# vs Loop (lento)
# for i in range(len(df)):
#     risk[i] = 0.5 * (1 - df['score'][i]) + 0.5 * unc_norm[i]
```

### 3. Caching de Figuras

**Ventaja**: No regenerar si ya existen (desarrollo iterativo)

```python
fig_path = OUTPUT_DIR / 'figure_5_1_*.png'
if not fig_path.exists():
    # Generar figura
    plt.savefig(fig_path)
```

---

## 🧪 Testing y Validación

### Checks Automáticos:

```python
# 1. Verificar que Fused < Baseline en todos los coverage
assert all(risk_fused < risk_baseline for risk_fused, risk_baseline in zip(...))

# 2. Verificar que FP Rate disminuyó
assert fp_fused < fp_baseline

# 3. Verificar que archivos se generaron
assert (OUTPUT_DIR / 'table_5_1_selective_prediction.csv').exists()
assert (OUTPUT_DIR / 'figure_5_1_*.png').exists()
```

### Validación Manual:

1. **Inspección visual de figuras**
   - ¿Línea verde debajo de roja? ✅
   - ¿Área sombreada visible? ✅

2. **Revisión de tablas**
   - ¿Valores razonables? ✅
   - ¿Mejora consistente? ✅

3. **Comparación con esperados**
   - ¿Diferencia < 10%? ✅ Aceptable
   - ¿Diferencia > 50%? ❌ Revisar

---

## 📈 Complejidad Computacional

### Temporal:

- **Carga datos**: O(n) donde n = número de predicciones
- **Cálculo risk**: O(n)
- **Selective prediction**: O(n log n) por el sorting
- **FP/FN rates**: O(n)
- **Figuras**: O(k) donde k = número de puntos en curva

**Total**: O(n log n) ≈ Lineal para n típico (~25K predicciones)

### Espacial:

- **Datos cargados**: ~100 MB (predicciones + GT)
- **Intermedios**: ~50 MB (risk scores, sorted)
- **Outputs**: ~5 MB (tablas + figuras)

**Total**: ~150-200 MB RAM

---

## 🔒 Reproducibilidad

### Seeds Fijadas:

```python
CONFIG['seed'] = 42
np.random.seed(42)
torch.manual_seed(42)
```

### Versiones de Librerías:

```
numpy==1.24.0
pandas==2.0.0
matplotlib==3.7.0
seaborn==0.12.0
```

### Determinismo:

- ✅ Carga de datos: Orden fijo
- ✅ Cálculos: Determinísticos (no hay muestreo)
- ✅ Visualizaciones: Reproducciones idénticas

---

**✅ Arquitectura técnica documentada completamente**
