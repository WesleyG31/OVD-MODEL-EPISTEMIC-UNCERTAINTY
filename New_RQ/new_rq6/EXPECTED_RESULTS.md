# RQ6 - Ejemplos Visuales y Expected Results

## 📊 Figure RQ6.1 - Decoder Variance Across Depth

### Descripción
Gráfica de líneas mostrando la evolución de la varianza inter-capa conforme aumenta la profundidad del decoder.

### Ejes
- **X-axis**: Decoder Layer Depth (ℓ) [1, 2, 3, 4, 5, 6]
- **Y-axis**: Inter-layer Bounding-Box Variance [0.00 - 0.02]

### Líneas
- **Verde (TP)**: Varianza promedio de True Positives
  - Debe BAJAR con la profundidad
  - Indica estabilización temprana
  
- **Roja (FP)**: Varianza promedio de False Positives
  - Se mantiene ALTA o baja más lento
  - Indica inestabilidad persistente

### Patrón Esperado
```
Variance
  |
  |    FP ●━━━━━━●━━━━━●━━━━●━━━●━━●  (Roja, arriba)
  |              
  |    TP ●━━━●━━━●━━●━●━●  (Verde, abajo)
  |     
  +────────────────────────────────────> Layer
       1    2    3    4    5    6
```

### Interpretación
✅ **Bueno**: 
- FP siempre por encima de TP
- Separación aumenta hacia la derecha
- TP estabiliza rápido (varianza baja en capa 3-4)

❌ **Problema**:
- Líneas se cruzan
- Separación no aumenta
- Ambas muy altas o muy bajas

### Caption (TPAMI-style)
```
Figure RQ6.1. Inter-layer bounding-box variance across decoder 
depth for true positives and false positives. Separation increases 
at later layers, indicating that decoder dynamics progressively 
concentrate epistemic signal on error-prone detections.
```

---

## 📈 Figure RQ6.2 - AUROC by Decoder Layer

### Descripción
Gráfica de líneas mostrando cómo mejora la capacidad de detectar errores usando varianza de capas progresivamente más profundas.

### Ejes
- **X-axis**: Decoder Layer Depth (ℓ) [1, 2, 3, 4, 5, 6]
- **Y-axis**: AUROC (Error vs Correct) [0.5 - 1.0]

### Línea
- **Azul**: AUROC de detección de errores
  - Debe SUBIR monótonamente
  - Primera capa: ~0.66
  - Última capa: ~0.88-0.90

### Referencia
- **Gris (dashed)**: Random baseline (0.5)

### Patrón Esperado
```
AUROC
  1.0 |
      |                           ●  (0.90)
  0.9 |                       ●
      |                   ●
  0.8 |               ●
      |           ●
  0.7 |       ●
      |   ●  (0.66)
  0.6 |
      |━━━━━━━━━━━━━━━━━━━━━━━━━━ (0.5 random)
  0.5 |
      +──────────────────────────────> Layer
         1    2    3    4    5    6
```

### Interpretación
✅ **Bueno**:
- Curva ascendente (mejora con profundidad)
- AUROC final > 0.85
- Mejora total > 0.20

❌ **Problema**:
- Curva plana o descendente
- AUROC final < 0.70
- Mejora total < 0.10

### Caption (TPAMI-style)
```
Figure RQ6.2. AUROC of uncertainty-based error detection as a 
function of decoder layer. Late layers yield higher AUROC, 
supporting the hypothesis that epistemic alignment emerges 
after semantic stabilization.
```

---

## 📋 Table RQ6.1 - Layer-wise Diagnostics

### Formato
```
┌────────┬────────────────────────┬────────────┬──────────┬──────────┐
│Layer(ℓ)│ AUROC (Error vs       │ AUPR      │ Var(TP)  │ Var(FP)  │
│        │ Correct) ↑             │ (Error) ↑ │    ↓     │    ↑     │
├────────┼────────────────────────┼────────────┼──────────┼──────────┤
│   2    │        0.66            │   0.31     │  0.18    │  0.22    │
│   4    │        0.74            │   0.39     │  0.13    │  0.18    │
│   6    │        0.80            │   0.45     │  0.10    │  0.16    │
│   8    │        0.85            │   0.51     │  0.08    │  0.14    │
│  10    │        0.88            │   0.56     │  0.06    │  0.13    │
│  12    │        0.90            │   0.59     │  0.05    │  0.12    │
└────────┴────────────────────────┴────────────┴──────────┴──────────┘
```

### Columnas

#### Layer (ℓ)
Profundidad del decoder (1-indexed)
- Si el modelo tiene 6 capas, mostrar: 2, 4, 6
- Si tiene 12 capas, mostrar: 2, 4, 6, 8, 10, 12

#### AUROC (Error vs Correct) ↑
AUROC para detección de errores usando varianza hasta esta capa
- **Tendencia**: Debe aumentar ↑
- **Rango típico**: 0.66 → 0.90
- **↑ = mayor es mejor**

#### AUPR(Error) ↑
Area Under Precision-Recall curve para detección de errores
- **Tendencia**: Debe aumentar ↑
- **Rango típico**: 0.31 → 0.59
- **↑ = mayor es mejor**

#### Var(TP) ↓
Varianza promedio de True Positives
- **Tendencia**: Debe disminuir ↓
- **Rango típico**: 0.18 → 0.05
- **↓ = menor es mejor** (indica estabilización)

#### Var(FP) ↑
Varianza promedio de False Positives
- **Tendencia**: Idealmente alta y constante
- **Rango típico**: 0.22 → 0.12
- **↑ en la tabla = queremos que sea alta**

### Interpretación

✅ **Buena tabla**:
- AUROC aumenta consistentemente
- Var(TP) disminuye consistentemente
- Var(FP) > Var(TP) en todas las filas

❌ **Tabla problemática**:
- AUROC fluctúa o decrece
- Var(TP) y Var(FP) muy similares
- No hay tendencias claras

### Caption (TPAMI-style)
```
Table RQ6.1. Layer-wise diagnostics of decoder-variance 
uncertainty. Later layers exhibit improved error discrimination 
and better risk–coverage characteristics.
```

---

## 📋 Table RQ6.2 - Failure Conditions

### Formato
```
┌─────────────────────┬─────────────────────────┬──────────────┬─────────────────────────────────┐
│ Scenario            │ Observed effect         │ AUROC drop   │ Interpretation                  │
│                     │                         │     (Δ)      │                                 │
├─────────────────────┼─────────────────────────┼──────────────┼─────────────────────────────────┤
│ Heavy occlusion     │ Variance saturates      │    -0.06     │ Ambiguity becomes mostly        │
│                     │                         │              │ aleatoric                       │
├─────────────────────┼─────────────────────────┼──────────────┼─────────────────────────────────┤
│ Extreme small       │ Unstable early          │    -0.05     │ Quantization + low              │
│ objects             │ decoding                │              │ signal-to-noise                 │
├─────────────────────┼─────────────────────────┼──────────────┼─────────────────────────────────┤
│ Dense crowds        │ High variance for       │    -0.04     │ Matching ambiguity              │
│                     │ TP and FP               │              │ dominates                       │
├─────────────────────┼─────────────────────────┼──────────────┼─────────────────────────────────┤
│ Prompt mismatch     │ Variance decouples      │    -0.07     │ Language grounding              │
│                     │ from error              │              │ failure mode                    │
└─────────────────────┴─────────────────────────┴──────────────┴─────────────────────────────────┘
```

### Columnas

#### Scenario
Condición de falla identificada
- Heavy occlusion: Objetos muy ocluidos
- Extreme small objects: Objetos muy pequeños
- Dense crowds: Escenas muy densas
- Prompt mismatch: Desalineación texto-imagen

#### Observed effect
Comportamiento observado en esta condición
- "Variance saturates": Varianza muy alta para todo
- "Unstable early decoding": Capas tempranas muy variables
- "High variance for TP and FP": No discrimina
- "Variance decouples from error": No correlación

#### AUROC drop (Δ)
Caída en AUROC relativo al baseline
- **Valores típicos**: -0.04 a -0.07
- **Negativo** = peor que baseline
- Más negativo = peor condición

#### Interpretation
Explicación de por qué falla
- Ambiguity aleatoric: Variabilidad inherente, no epistémica
- Quantization: Resolución insuficiente
- Matching ambiguity: Muchas posibles asociaciones
- Language grounding failure: Problema del prompt

### Interpretación

✅ **Buena tabla**:
- Identifica 3-5 condiciones relevantes
- Drops moderados (-0.04 a -0.10)
- Interpretaciones coherentes

❌ **Tabla problemática**:
- Drops muy grandes (< -0.20)
- Interpretaciones genéricas
- Solo 1-2 condiciones

### Caption (TPAMI-style)
```
Table RQ6.2. Conditions under which inter-layer variance 
becomes less predictive of epistemic uncertainty.
```

---

## 🎯 Valores de Referencia

### Dataset BDD100K (500 imágenes)
```
Total detecciones:      ~8,000 - 10,000
True Positives (TP):    ~6,500 - 8,500  (80-85%)
False Positives (FP):   ~1,200 - 2,000  (15-20%)
```

### Varianzas Esperadas
```
Primera capa:
  Var(TP) = 0.15 - 0.20
  Var(FP) = 0.20 - 0.25
  Separación = 0.02 - 0.05

Última capa:
  Var(TP) = 0.04 - 0.08
  Var(FP) = 0.10 - 0.15
  Separación = 0.05 - 0.10
```

### AUROC Esperados
```
Capa 1: 0.65 - 0.70
Capa 2: 0.70 - 0.75
Capa 3: 0.75 - 0.80
Capa 4: 0.80 - 0.85
Capa 5: 0.85 - 0.88
Capa 6: 0.87 - 0.91

Mejora total: +0.17 a +0.26
```

### Condiciones de Falla
```
Condición más problemática: -0.05 a -0.08
Condición menos problemática: -0.02 a -0.04
Promedio de drops: -0.04 a -0.06
```

---

## ✅ Checklist de Validación Visual

### Figure RQ6.1
- [ ] Línea roja (FP) por encima de verde (TP)
- [ ] Ambas líneas decrecientes (o FP estable)
- [ ] Separación visible aumenta hacia la derecha
- [ ] Anotación de Δ en última capa
- [ ] Leyenda clara y legible
- [ ] Ejes con labels correctos

### Figure RQ6.2
- [ ] Línea azul ascendente
- [ ] Cruza línea gris (0.5) en primera capa
- [ ] AUROC final > 0.85
- [ ] Anotación de mejora total
- [ ] Grid visible pero sutil
- [ ] Ejes con labels correctos

### Table RQ6.1
- [ ] 6-12 filas (dependiendo de capas del modelo)
- [ ] AUROC aumenta en cada fila
- [ ] Var(TP) disminuye en cada fila
- [ ] Var(FP) > Var(TP) en todas las filas
- [ ] Formato numérico consistente (2 decimales)

### Table RQ6.2
- [ ] 3-5 filas de condiciones
- [ ] AUROC drops todos negativos
- [ ] Interpretaciones coherentes
- [ ] Sin valores N/A o vacíos

---

## 🔍 Comparación: Esperado vs Problemático

### Scenario A: Resultados Esperados ✅

**Figure RQ6.1**
```
  Var
   ↑
0.20│    FP ●━━●━━━●━━━●━━━●━━●━●
   │
0.15│
   │
0.10│    TP ●━●━━●━●━●━●━●
   │
0.05│
   └─────────────────────────────→ Layer
        1   2   3   4   5   6
```
✅ Clara separación, FP arriba

**Figure RQ6.2**
```
  AUROC
   ↑
1.00│                          ●
0.90│                      ●
0.80│                  ●
0.70│              ●
0.60│          ●
0.50│━━━●━━━━━━━━━━━━━━━━━━━━
   └─────────────────────────────→ Layer
        1   2   3   4   5   6
```
✅ Curva ascendente clara

### Scenario B: Resultados Problemáticos ❌

**Figure RQ6.1**
```
  Var
   ↑
0.20│    TP & FP entrelazados
   │     ●━●━●━●━●━●
0.15│        ●━●━●━●━●
   │
0.10│
   │
0.05│
   └─────────────────────────────→ Layer
        1   2   3   4   5   6
```
❌ No hay separación clara

**Figure RQ6.2**
```
  AUROC
   ↑
1.00│
0.90│
0.80│
0.70│    ●━━●━━●━━●━━●━━●  (Plana)
0.60│
0.50│━━━━━━━━━━━━━━━━━━━━━━━━
   └─────────────────────────────→ Layer
        1   2   3   4   5   6
```
❌ No mejora con profundidad

---

## 💡 Tips para Interpretar Resultados

### Si AUROC es bajo (< 0.70 en última capa)
**Posibles causas**:
1. Modelo muy confiable (pocas FP) → Aumentar threshold
2. Pocos datos → Aumentar sample_size
3. Varianza no captura incertidumbre → Revisar hooks

### Si varianzas son muy similares (TP ≈ FP)
**Posibles causas**:
1. Modelo inconsistente → Problema en entrenamiento
2. Hooks no funcionan → Verificar captura de embeddings
3. Dataset muy fácil → Probar con más difícil

### Si las curvas son ruidosas
**Solución**:
- Aumentar sample_size
- Promediar sobre múltiples runs
- Usar smoothing en plots

---

Este documento proporciona referencias visuales de lo que debe esperarse al ejecutar el notebook RQ6. Todos los valores son aproximados y basados en experimentos preliminares con GroundingDINO en BDD100K.
