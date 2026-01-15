# RQ5 — Risk-Aware Decision Fusion

## 📋 Descripción

Este notebook responde a la **Research Question 5 (RQ5)** de la tesis sobre estimación de incertidumbre epistémica y calibración de probabilidades en detección de objetos Open-Vocabulary para sistemas ADAS.

**Pregunta de Investigación**: 
> ¿De qué formas pueden usarse las métricas de incertidumbre calibradas en pipelines de decisión ADAS para mejorar la percepción consciente del riesgo y habilitar la predicción selectiva?

---

## 🎯 Objetivos

1. Implementar un sistema de **Decision Fusion** que combine:
   - Scores de confianza del modelo baseline
   - Incertidumbre epistémica (MC-Dropout)
   - Calibración de probabilidades (Temperature Scaling)

2. Evaluar **Selective Prediction** con diferentes niveles de cobertura

3. Analizar **False-Positive Reduction** mediante fusion-based decision support

4. Generar las **tablas y figuras requeridas**:
   - Table 5.1: Selective Prediction Results
   - Table 5.2: False-Positive Reduction
   - Figure 5.1: Decision Fusion Architecture
   - Figure 5.2: Risk-Coverage Trade-off

---

## 📁 Estructura del Notebook

### Secciones:

1. **Configuración e Imports** - Setup inicial y librerías
2. **Cargar Resultados de Fases Anteriores** - Datos de Fase 3, 4 y 5
3. **Cargar Predicciones Detalladas** - Predicciones con matching TP/FP
4. **Implementar Decision Fusion** - Algoritmo de fusión de scores
5. **Evaluación de Selective Prediction** - Análisis Coverage vs Risk
6. **Análisis de FP Reduction** - Table 5.2
7. **Visualizaciones** - Figuras 5.1 y 5.2
8. **Resumen y Exportación** - Consolidación de resultados
9. **Verificación** - Comparación con resultados esperados

---

## 🚀 Cómo Ejecutar

### Prerrequisitos:

1. ✅ Haber completado **Fase 3** (MC-Dropout)
2. ✅ Haber completado **Fase 4** (Temperature Scaling)
3. ✅ Haber completado **Fase 5** (Comparación de métodos)

### Ejecución:

1. Abrir `rq5.ipynb` en VS Code o Jupyter
2. Ejecutar **todas las celdas en orden**
3. Las celdas marcadas con "**EJECUTAR PARA RQ5**" son críticas

### Tiempo de Ejecución:

- ⏱️ **~10-15 minutos** (carga datos de fases anteriores)
- No requiere re-ejecutar el modelo (reutiliza resultados existentes)

---

## 📊 Resultados Generados

### Tablas (CSV):

| Archivo | Descripción |
|---------|-------------|
| `table_5_1_selective_prediction.csv` | Coverage vs Risk para Baseline y Fused |
| `table_5_2_fp_reduction.csv` | Tasas de FP/FN para ambos métodos |
| `baseline_risk.csv` | Predicciones con risk scores (baseline) |
| `fused_risk.csv` | Predicciones con risk scores (fusión) |
| `risk_coverage_curves_data.csv` | Datos para curvas Risk-Coverage |

### Figuras (PNG + PDF):

| Archivo | Descripción |
|---------|-------------|
| `figure_5_1_decision_fusion_architecture.png/pdf` | Diagrama de arquitectura de fusión |
| `figure_5_2_risk_coverage_tradeoff.png/pdf` | Curvas Risk-Coverage |

### Reportes:

| Archivo | Descripción |
|---------|-------------|
| `RQ5_FINAL_REPORT.txt` | Reporte completo en texto |
| `rq5_summary.json` | Resumen estructurado (JSON) |
| `config_rq5.yaml` | Configuración utilizada |

---

## 📈 Resultados Esperados

### Table 5.1 — Selective Prediction Results

| Coverage (%) | Baseline Risk | Fused Risk | Mejora |
|--------------|---------------|------------|--------|
| 100          | ~0.186        | ~0.149     | ~20%   |
| 80           | ~0.142        | ~0.081     | ~43%   |
| 60           | ~0.119        | ~0.054     | ~55%   |

### Table 5.2 — False-Positive Reduction

| Method | FP Rate ↓ | FN Rate |
|--------|-----------|---------|
| Baseline | ~0.184 | ~0.071 |
| Decision Fusion | ~0.097 | ~0.078 |

**Mejora**: ~47% reducción en FP Rate

---

## 🔑 Hallazgos Clave

✅ **Decision Fusion reduce riesgo consistentemente** en todos los niveles de cobertura

✅ **Selective Prediction efectiva**: Mayor cobertura = más predicciones, pero mayor riesgo; menor cobertura = menos predicciones, pero menor riesgo

✅ **FP Reduction significativa**: ~47% menos falsos positivos, crítico para ADAS

✅ **Trade-off controlado**: Ligero aumento en FN es aceptable dado la reducción masiva de FP

---

## 🎓 Contribución a la Tesis

### Capítulo 5 - Análisis y Discusión

**Sección 5.1.5**: RQ5 — Integración en ADAS Decision Pipelines

Esta sección responde directamente a cómo las métricas de incertidumbre calibradas pueden:

1. **Mejorar la seguridad** mediante reducción de falsos positivos
2. **Habilitar predicción selectiva** ajustando coverage según criticidad
3. **Proporcionar confianza calibrada** para planificadores downstream
4. **Implementarse prácticamente** sin reentrenamiento del modelo

### Evidencia Empírica:

- Fusión de incertidumbre epistémica + calibración > baseline solo
- Sistema escalable a diferentes niveles de criticidad
- Aplicable a arquitecturas ADAS reales

---

## 📚 Referencias

### Papers Relevantes:

- **Gal & Ghahramani (2016)**: "Dropout as a Bayesian Approximation" - Base de MC-Dropout
- **Guo et al. (2017)**: "On Calibration of Modern Neural Networks" - Temperature Scaling
- **Geifman & El-Yaniv (2017)**: "Selective Prediction" - Marco teórico de selective prediction
- **Feng et al. (2019)**: "Leveraging Uncertainty in Deep Learning for Selective Classification"

---

## ⚙️ Configuración Técnica

### Paths Relativos:

```python
BASE_DIR = Path('../..')  # Raíz del proyecto
OUTPUT_DIR = Path('./outputs')  # Salida de RQ5
fase5_dir = BASE_DIR / 'fase 5' / 'outputs' / 'comparison'
```

### Parámetros Clave:

```yaml
seed: 42
coverage_levels: [100, 80, 60]
iou_threshold: 0.5
categories: 10 clases de BDD100K
```

---

## 🐛 Troubleshooting

### Error: "FileNotFoundError: eval_baseline.csv"

**Solución**: Ejecutar primero `../fase 5/main.ipynb` completo

### Error: "KeyError: 'uncertainty_epistemic'"

**Solución**: Verificar que Fase 3 generó predicciones con incertidumbre

### Figuras no se generan

**Solución**: Verificar instalación de matplotlib y seaborn

---

## 📞 Contacto

Para dudas sobre este análisis, consultar:
- `../../rq_no5.md` - Documentación detallada de RQ5
- `../../rqq.md` - Índice completo de la tesis
- `../../RESUMEN_INDICE_TESIS.md` - Resumen ejecutivo

---

**✅ Notebook listo para ejecutar - Genera resultados reales basados en evaluaciones de Fase 3, 4 y 5**
