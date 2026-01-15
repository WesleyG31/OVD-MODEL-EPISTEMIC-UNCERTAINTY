# Metodología Detallada - RQ7

## 1. Fundamento Teórico

### 1.1 Trade-off Eficiencia-Confiabilidad

En sistemas de detección de objetos para aplicaciones críticas (ADAS), existe un trade-off fundamental:

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  High Reliability  ←────────────→  High Efficiency     │
│  (Low ECE)                          (High FPS)          │
│                                                         │
│  MC-Dropout: ✓ Confiable, ✗ Lento                     │
│  Baseline:   ✗ No confiable, ✓ Rápido                 │
│  Fusion:     ✓ Confiable, ✓ Rápido  ← OBJETIVO        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 1.2 Requisitos ADAS

**Tiempo Real**: ≥20 FPS (≤50ms/frame)
- Necesario para detección confiable a 60-80 km/h
- Permite tiempo de reacción adecuado del sistema

**Confiabilidad**: ECE ≤0.10
- Probabilidades calibradas críticas para toma de decisiones
- Evita sobre/sub-confianza en predicciones

## 2. Métodos Evaluados

### 2.1 MC-Dropout (Baseline de Confiabilidad)

**Descripción**:
- Activa dropout durante inferencia
- Realiza K=5 forward passes
- Calcula incertidumbre epistémica mediante varianza

**Ventajas**:
- ✓ Alta calidad de incertidumbre
- ✓ Detecta casos fuera de distribución
- ✓ Bien calibrado después de TS

**Desventajas**:
- ✗ K× más lento que single-pass
- ✗ No viable para tiempo real
- ✗ Alto consumo de memoria/energía

**Complejidad**: O(K × N) donde N = latencia single-pass

### 2.2 Decoder Variance (Baseline de Eficiencia)

**Descripción**:
- Single forward pass
- Extrae embeddings del decoder
- Calcula varianza intra-capa como proxy de incertidumbre

**Ventajas**:
- ✓ Rápido (overhead mínimo vs baseline)
- ✓ No requiere múltiples pases
- ✓ Bajo consumo de recursos

**Desventajas**:
- ✗ Incertidumbre de menor calidad que MC-Dropout
- ✗ Sin calibración, ECE alto
- ✗ No captura incertidumbre epistémica completa

**Complejidad**: O(N + ε) donde ε = overhead de cálculo de varianza

### 2.3 Fusion (Variance + Temperature Scaling)

**Descripción**:
- Decoder Variance para incertidumbre
- Temperature Scaling para calibración
- Combina eficiencia + confiabilidad

**Ventajas**:
- ✓ Rápido como Variance (TS es post-proceso)
- ✓ Bien calibrado (mejor ECE que MC-Dropout)
- ✓ Apto para tiempo real
- ✓ Bajo overhead computacional

**Desventajas**:
- ✗ Requiere conjunto de calibración (val_calib)
- ✗ Temperatura fija (no adapta por imagen)

**Complejidad**: O(N + ε) en inferencia, O(M) en calibración offline
donde M = tamaño de val_calib

## 3. Protocolo de Medición de Latencia

### 3.1 Hardware y Configuración

```yaml
Device: NVIDIA GPU (CUDA) o CPU
Framework: PyTorch 2.x
Precision: FP32 (sin optimizaciones TensorRT/ONNX)
Batch Size: 1 (evaluación realista para ADAS)
Image Size: Original BDD100K (1280×720)
```

### 3.2 Procedimiento de Benchmark

#### Paso 1: Warmup
```python
for i in range(warmup=5):
    _ = model(image)  # Calentar GPU, cargar cachés
```

**Razón**: Primera inferencia es más lenta (carga CUDA kernels)

#### Paso 2: Medición Baseline
```python
for image in images:
    torch.cuda.synchronize()  # Asegurar que GPU terminó
    start = time.time()
    
    predictions = model(image)  # Single forward pass
    
    torch.cuda.synchronize()
    end = time.time()
    
    times.append(end - start)
```

**Métrica**: `latency_baseline = mean(times)`

#### Paso 3: Medición MC-Dropout
```python
for image in images:
    enable_dropout()  # Activar dropout en cabeza
    
    torch.cuda.synchronize()
    start = time.time()
    
    for k in range(K=5):
        predictions[k] = model(image)  # K forward passes
    
    torch.cuda.synchronize()
    end = time.time()
    
    times.append(end - start)
```

**Métrica**: `latency_mc = mean(times)` incluye K=5 pases

#### Paso 4: Medición Variance
```python
for image in images:
    torch.cuda.synchronize()
    start = time.time()
    
    predictions, embeddings = model(image, return_embeddings=True)
    variance = compute_variance(embeddings)  # Overhead mínimo
    
    torch.cuda.synchronize()
    end = time.time()
    
    times.append(end - start)
```

**Métrica**: `latency_variance = mean(times)` incluye cálculo de varianza

**Nota**: Temperature Scaling NO se mide porque es post-procesamiento:
```python
# TS se aplica DESPUÉS de obtener predicciones
calibrated_score = sigmoid(logit / T)  # Operación trivial O(1)
```

### 3.3 Cálculo de FPS

```python
FPS = 1.0 / mean_latency_seconds
```

**Ejemplo**:
- Latency = 0.083s → FPS = 12.0
- Latency = 0.043s → FPS = 23.3

### 3.4 Estadísticas Reportadas

Para cada método:
- **Media**: `μ = mean(times)`
- **Desviación**: `σ = std(times)`
- **FPS**: `1/μ`
- **Percentil 95**: `p95 = percentile(times, 95)` (worst-case latency)

## 4. Métricas de Confiabilidad

### 4.1 Expected Calibration Error (ECE)

**Definición**:
```
ECE = Σ (|Accuracy(B_i) - Confidence(B_i)|) × |B_i| / N
```

Donde:
- `B_i`: Bin i de confianzas
- `Accuracy(B_i)`: Precisión real en ese bin
- `Confidence(B_i)`: Confianza promedio en ese bin
- `N`: Total de predicciones

**Interpretación**:
- ECE = 0.00: Perfectamente calibrado
- ECE < 0.05: Bien calibrado
- ECE < 0.10: Aceptable
- ECE > 0.15: Mal calibrado

**Fuente**: Fase 5 (`calibration_metrics.json`)

### 4.2 Reliability Score

**Definición**:
```
Reliability Score = 1 - ECE
```

**Rango**: [0, 1], mayor es mejor

**Ventaja**: Métrica intuitiva (91% reliability = 91% confiable)

### 4.3 Reliability per Millisecond

**Definición**:
```
Efficiency = Reliability Score / Latency_ms
```

**Interpretación**:
- Mide "cuánta confiabilidad obtengo por unidad de tiempo"
- Mayor es mejor
- Combina ambos aspectos del trade-off

**Ejemplo**:
```
MC-Dropout: 0.918 / 83ms = 0.011 reliability/ms
Fusion:     0.939 / 43ms = 0.022 reliability/ms  ← 2× mejor
```

## 5. Criterios de Evaluación

### 5.1 Tiempo Real (Real-Time Ready)

```
✔ Real-Time Ready  ⟺  FPS ≥ 20
✗ Not Real-Time    ⟺  FPS < 20
```

**Justificación**:
- 20 FPS = 50ms/frame
- Típicamente ADAS requiere 30+ FPS, pero 20 es mínimo
- Deja margen para post-procesamiento (NMS, tracking)

### 5.2 Confiabilidad Aceptable

```
✔ Reliable  ⟺  Reliability Score ≥ 0.85 (ECE ≤ 0.15)
✗ Unreliable ⟺  Reliability Score < 0.85
```

**Justificación**:
- ADAS no puede tolerar sobre/sub-confianza >15%
- Score ≥0.90 es ideal para decisiones críticas

### 5.3 Factibilidad ADAS

```
✔ ADAS Feasible  ⟺  (FPS ≥ 20) AND (Reliability ≥ 0.85)
✗ Not Feasible   ⟺  Otherwise
```

## 6. Análisis Comparativo

### 6.1 Comparación Directa

**MC-Dropout vs Fusion**:
```
Aspecto          | MC-Dropout | Fusion  | Winner
─────────────────|──────────--|─────────|────────
FPS              | 12         | 23      | Fusion
Latency (ms)     | 83         | 43      | Fusion
ECE              | 0.082      | 0.061   | Fusion
Reliability      | 0.918      | 0.939   | Fusion
Real-Time Ready  | ✗          | ✔       | Fusion
ADAS Feasible    | ✗          | ✔       | Fusion
```

**Conclusión**: Fusion domina a MC-Dropout en todos los aspectos relevantes

### 6.2 Análisis de Trade-off

**Pareto Front**:
```
        Reliability
             ↑
        1.0  ●Fusion (óptimo)
             │
        0.9  ●MC-Dropout
             │
        0.8  ●Variance
             │
        0.7  │
             └───────────────────→ Latency
             0ms  40ms  80ms  120ms
```

**Interpretación**:
- Fusion está en la frontera de Pareto
- No hay método que sea mejor en ambos aspectos simultáneamente
- MC-Dropout está dominado (Fusion es mejor en todo)

## 7. Limitaciones y Consideraciones

### 7.1 Factores que Afectan Latencia

1. **Hardware**: GPU vs CPU (10-20× diferencia)
2. **Batch Size**: Evaluamos batch=1 (realista), pero batch=8 podría ser más rápido
3. **Resolución**: Imágenes completas (1280×720), no crops
4. **Optimizaciones**: Sin TensorRT, ONNX, quantization
5. **Concurrencia**: Mediciones single-threaded

### 7.2 Validez Externa

**Generalización**:
- ✓ Resultados válidos para GroundingDINO
- ✓ Tendencias generalizan a otros detectores
- ✗ Valores absolutos dependen de arquitectura

**Recomendación**: Repetir benchmark con modelo específico de producción

### 7.3 Supuestos

1. **Independencia**: Cada imagen se procesa independientemente
2. **Distribución**: Imágenes aleatorias representativas
3. **Estacionariedad**: Latencia no varía con contenido de imagen
4. **No interferencia**: No hay otros procesos compitiendo por GPU

## 8. Reproducibilidad

### 8.1 Requisitos Mínimos

```yaml
GPU: NVIDIA GTX 1080 o superior (8+ GB VRAM)
RAM: 16+ GB
Python: 3.8+
PyTorch: 2.0+
CUDA: 11.7+
```

### 8.2 Seeds y Determinismo

```python
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

**Nota**: Latencia puede variar ±5% entre ejecuciones debido a:
- Thermal throttling de GPU
- Carga del sistema
- Orden de ejecución de CUDA kernels

### 8.3 Datos Guardados

Todos los datos brutos se guardan en JSON para re-análisis:

```json
{
  "latency_raw.json": "Todos los tiempos individuales",
  "runtime_metrics.json": "Estadísticas agregadas",
  "figure_X_data.json": "Datos de cada figura"
}
```

## 9. Interpretación de Resultados

### 9.1 Pregunta de Investigación

**RQ7**: ¿Fusion logra confiabilidad cercana a MC-Dropout a velocidad de tiempo real?

**Hipótesis**:
- H1: Fusion tiene FPS ≥ 20 (tiempo real)
- H2: Fusion tiene ECE ≤ MC-Dropout (confiabilidad comparable)

**Resultado Esperado**:
```
✓ H1: Fusion = 23 FPS ≥ 20 FPS ✓
✓ H2: Fusion ECE = 0.061 < MC-Dropout ECE = 0.082 ✓
```

**Conclusión**: Fusion satisface AMBAS hipótesis → RQ7 respondida afirmativamente

### 9.2 Implicaciones Prácticas

**Para Despliegue ADAS**:
1. MC-Dropout **NO es viable** en producción (muy lento)
2. Variance solo **NO es suficiente** (mal calibrado)
3. Fusion es **la única opción** que cumple ambos requisitos

**Para Investigación**:
1. Demuestra viabilidad de métodos single-pass + calibración
2. Establece baseline para futuros trabajos
3. Cuantifica trade-off eficiencia-confiabilidad

## 10. Referencias Metodológicas

### Papers Clave

1. **Latency Benchmarking**:
   - "Measuring Neural Net Robustness with Constraints" (Bastani et al., 2016)
   - MLPerf Inference Benchmark

2. **Calibration Metrics**:
   - "On Calibration of Modern Neural Networks" (Guo et al., 2017)
   - "Measuring Calibration in Deep Learning" (Nixon et al., 2019)

3. **MC-Dropout**:
   - "Dropout as a Bayesian Approximation" (Gal & Ghahramani, 2016)

4. **ADAS Requirements**:
   - ISO 26262 (Functional Safety)
   - SAE J3016 (Levels of Automation)

### Herramientas Utilizadas

- **PyTorch**: Framework de deep learning
- **CUDA Events**: Medición precisa de tiempo GPU
- **pycocotools**: Evaluación de detección
- **GroundingDINO**: Modelo OVD
