# Fase 5: Comparación Completa de Métodos ⚡ (Optimizado)

## 🎯 Objetivo

Comparar 6 métodos de incertidumbre y calibración lado a lado:
1. **Baseline** (sin incertidumbre, sin calibración)
2. **Baseline + TS** (Temperature Scaling)
3. **MC-Dropout K=5** (incertidumbre epistémica)
4. **MC-Dropout K=5 + TS** (incertidumbre + calibración)
5. **Varianza entre capas** (single-pass, decoder layers)
6. **Varianza entre capas + TS**

---

## ⚡ NUEVO: Optimización de Rendimiento

**Este notebook ha sido optimizado para reutilizar resultados de fases anteriores.**

### ✅ Tiempo de Ejecución

| Escenario | Tiempo Original | Tiempo Optimizado | Ahorro |
|-----------|----------------|-------------------|--------|
| **Con archivos previos** | ~2 horas | **~15-20 minutos** | ⚡ 85% |
| **Sin archivos previos** | ~2 horas | ~2 horas | - |

### 📦 Archivos Requeridos

Para máximo beneficio, asegúrate de tener:

```
fase 2/outputs/baseline/preds_raw.json           ← Predicciones baseline
fase 3/outputs/mc_dropout/preds_mc_aggregated.json ← Predicciones MC-Dropout
fase 4/outputs/temperature_scaling/temperature.json ← Temperaturas optimizadas
```

### 🔍 Verificar Optimización

Ejecuta el script de verificación antes de correr el notebook:

```bash
python verify_optimization.py
```

Esto te dirá:
- ✅ Qué archivos están disponibles
- ⏱️ Cuánto tiempo ahorrarás
- 📋 Qué necesitas ejecutar primero (si algo falta)

---

## 📊 Métricas Evaluadas

### 1. Detección
- **mAP@[0.5:0.95]**: Métrica principal de COCO
- **AP50**: Precisión a IoU=0.5
- **AP75**: Precisión a IoU=0.75
- **Por clase**: Métricas individuales para cada categoría

### 2. Calibración
- **NLL** (Negative Log-Likelihood): Pérdida probabilística
- **Brier Score**: Error cuadrático de predicciones probabilísticas
- **ECE** (Expected Calibration Error): Diferencia entre confianza y precisión
- **Reliability Diagrams**: Visualización de calibración

### 3. Risk-Coverage
- **Curvas Risk-Coverage**: Trade-off entre riesgo y cobertura
- **AUC**: Área bajo la curva (mayor es mejor)
- **Uncertainty AUROC**: Capacidad de discriminar TP vs FP usando incertidumbre

---

## 🗂️ Estructura de Datos

### Splits de Validación
```python
val_calib.json  # 500 imágenes → Ajustar temperaturas
val_eval.json   # ~10,000 imágenes → Evaluación final
```

### Outputs Generados
```
outputs/comparison/
├── config.yaml                      # Configuración usada
│
├── calib_baseline.csv               # Datos de calibración
├── calib_mc_dropout.csv
├── calib_decoder_variance.csv
│
├── temperatures.json                # Temperaturas optimizadas
│
├── eval_baseline.csv                # Predicciones en val_eval
├── eval_baseline_ts.csv
├── eval_mc_dropout.csv
├── eval_mc_dropout_ts.csv
├── eval_decoder_variance.csv
├── eval_decoder_variance_ts.csv
│
├── detection_metrics.json           # Métricas mAP por método
├── calibration_metrics.json         # Métricas de calibración
├── risk_coverage_auc.json          # AUCs de risk-coverage
├── uncertainty_auroc.json          # AUROC por método
│
└── visualizations/
    ├── comparison_map.png           # Comparación de mAP
    ├── comparison_calibration.png   # Comparación de NLL/ECE
    ├── reliability_diagrams.png     # 6 reliability diagrams
    ├── risk_coverage_curves.png     # Curvas risk-coverage
    └── uncertainty_auroc.png        # AUROC de incertidumbre
```

---

## 🚀 Cómo Ejecutar

### Opción A: Con Optimización (RECOMENDADO)

```bash
# 1. Verificar que tienes los archivos previos
python verify_optimization.py

# 2. Si sale ✅, ejecutar el notebook
jupyter notebook main.ipynb

# Tiempo: ~15-20 minutos ⚡
```

### Opción B: Primera Vez (Sin archivos previos)

```bash
# Ejecuta directamente
jupyter notebook main.ipynb

# Tiempo: ~2 horas 🐌
# Pero generará archivos para futuras ejecuciones
```

### Opción C: Ejecutar Fases Previas Primero

```bash
# 1. Ejecutar Fase 2
cd "../fase 2"
jupyter notebook main.ipynb

# 2. Ejecutar Fase 3
cd "../fase 3"
jupyter notebook main.ipynb

# 3. Ejecutar Fase 4
cd "../fase 4"
jupyter notebook main.ipynb

# 4. Ahora Fase 5 será rápida
cd "../fase 5"
jupyter notebook main.ipynb
```

---

## 📝 Detalles de Implementación

### Métodos de Inferencia

#### 1. Baseline
```python
def inference_baseline(model, image_path, text_prompt, conf_thresh, device):
    # Single-pass, dropout desactivado
    # Incertidumbre = 0.0 (no tiene)
```

#### 2. MC-Dropout
```python
def inference_mc_dropout(model, image_path, text_prompt, conf_thresh, device, K=5):
    # K pases con dropout activo
    # Incertidumbre = varianza de scores entre pases
    # Alineación de detecciones con IoU >= 0.5
```

#### 3. Decoder Variance
```python
def inference_decoder_variance(model, image_path, text_prompt, conf_thresh, device):
    # Single-pass con hooks en capas del decoder
    # Incertidumbre = varianza de scores entre capas
    # Rápido (sin múltiples pases)
```

### Temperature Scaling

```python
# Optimización por NLL
T_opt = minimize(lambda T: nll_loss(T, logits, labels), x0=1.0)

# Aplicación
score_calibrated = sigmoid(logit / T_opt)
```

---

## 📈 Análisis de Resultados

### Métricas Clave a Comparar

1. **Detección (mAP)**:
   - ¿Mejora la calibración la detección?
   - ¿Afecta el MC-Dropout el rendimiento?

2. **Calibración (NLL, ECE)**:
   - ¿Qué método tiene mejor calibración inicial?
   - ¿Mejora significativamente el Temperature Scaling?

3. **Risk-Coverage (AUC)**:
   - ¿Qué método permite mejor trade-off riesgo/cobertura?
   - ¿La incertidumbre epistémica ayuda?

4. **Uncertainty Quality (AUROC)**:
   - ¿Puede la incertidumbre discriminar TP vs FP?
   - ¿MC-Dropout vs Decoder Variance?

---

## 🔧 Configuración

```yaml
seed: 42
device: cuda  # o cpu
categories: [person, rider, car, truck, bus, train, motorcycle, 
            bicycle, traffic light, traffic sign]
iou_matching: 0.5
conf_threshold: 0.25
nms_threshold: 0.65
K_mc: 5
n_bins: 10
```

---

## 🐛 Troubleshooting

### Problema: "ModuleNotFoundError: groundingdino"
```bash
# Asegúrate de estar en el entorno correcto
source /opt/program/venv/bin/activate  # Linux
# O
/opt/program/venv/Scripts/activate  # Windows
```

### Problema: "CUDA out of memory"
```python
# Reducir batch size implícito o usar CPU
CONFIG['device'] = 'cpu'
```

### Problema: "Archivos no encontrados"
```bash
# Verificar paths
python verify_optimization.py

# Si faltan, ejecuta las fases anteriores
```

### Problema: "Resultados no coinciden"
```python
# Verifica que uses el mismo split
# Todas las fases deben usar:
#   - val_calib.json (mismo)
#   - val_eval.json (mismo)
```

---

## 📚 Referencias

- **Baseline**: GroundingDINO estándar
- **MC-Dropout**: [Gal & Ghahramani, 2016](https://arxiv.org/abs/1506.02142)
- **Temperature Scaling**: [Guo et al., 2017](https://arxiv.org/abs/1706.04599)
- **Risk-Coverage**: [Geifman & El-Yaniv, 2017](https://arxiv.org/abs/1705.08500)

---

## 📖 Documentación Adicional

- **[OPTIMIZACIONES.md](OPTIMIZACIONES.md)**: Detalles técnicos de las optimizaciones
- **[verify_optimization.py](verify_optimization.py)**: Script de verificación
- **main.ipynb**: Notebook principal (con comentarios extensos)

---

## ✅ Checklist de Ejecución

- [ ] ¿Tienes los archivos de fases anteriores? → Ejecuta `verify_optimization.py`
- [ ] ¿Configuraste correctamente los paths? → Revisa `BASE_DIR`, `DATA_DIR`
- [ ] ¿Tienes GPU disponible? → Verifica `CONFIG['device']`
- [ ] ¿Instalaste dependencias? → `torch`, `groundingdino`, `pycocotools`
- [ ] ¿Activaste el entorno virtual? → `source /opt/program/venv/bin/activate`

---

## 🎓 Resultados Esperados

Al finalizar, tendrás:

1. ✅ Comparación cuantitativa de 6 métodos
2. ✅ Visualizaciones de comparación
3. ✅ Análisis de calibración
4. ✅ Curvas risk-coverage
5. ✅ Evaluación de calidad de incertidumbre
6. ✅ Reporte final con recomendaciones

**Pregunta clave**: ¿Vale la pena el costo computacional del MC-Dropout comparado con métodos single-pass?

---

## 📧 Soporte

Si encuentras problemas o tienes preguntas, revisa:
1. La documentación en `OPTIMIZACIONES.md`
2. Los comentarios en el notebook
3. Los mensajes de error del script de verificación

**Última actualización**: 2024
**Versión**: 2.0 (Optimizado)
**Estado**: ✅ Probado y funcional
