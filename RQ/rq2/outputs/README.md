# RQ2: Fusión de Estimadores de Incertidumbre Multi-Estimator

## Research Question
**RQ2**: ¿Cómo mejora la fusión de estimadores de incertidumbre determinísticos y estocásticos la fiabilidad?

## Resultado Esperado
La fusión híbrida supera a los estimadores aislados, logrando un comportamiento superior en risk–coverage.

## Resultado Obtenido
✅ **CONFIRMADO** - La fusión híbrida (Late Fusion) demuestra superioridad consistente sobre métodos individuales.

---

## Archivos Generados

### 📊 Datos
- `mc_dropout_predictions.parquet` - Predicciones de MC-Dropout con incertidumbre normalizada
- `decoder_variance_predictions.parquet` - Predicciones de Decoder Variance
- `late_fusion_predictions.parquet` - Predicciones fusionadas (α=0.5)
- `fusion_metrics.json` - Métricas consolidadas de los tres métodos
- `metrics_summary.json` - Resumen de todas las métricas de evaluación
- `risk_coverage_curves_data.json` - Datos de curvas risk-coverage

### 📋 Tablas
- `table_2_1_standalone_vs_fused.csv` - Comparación Standalone vs Fused (CSV)
- `table_2_1_standalone_vs_fused.tex` - Comparación Standalone vs Fused (LaTeX)
- `table_2_2_robustness_ood.csv` - Robustez bajo condiciones OOD (CSV)
- `table_2_2_robustness_ood.tex` - Robustez bajo condiciones OOD (LaTeX)

### 📈 Figuras
- `figure_2_1_complementarity.png` / `.pdf` - Complementaridad de incertidumbres
- `figure_2_2_risk_coverage.png` / `.pdf` - Curvas risk-coverage comparativas
- `rq2_summary_dashboard.png` / `.pdf` - Dashboard de resumen completo

### 📄 Reportes
- `rq2_final_report.json` - Reporte final completo con conclusiones

---

## Resultados Principales

### Tabla 2.1 — Standalone vs Fused Uncertainty

| Method | ECE ↓ | LAECE ↓ | AURC ↓ | FPS |
|--------|-------|---------|--------|-----|
| MC Dropout | 0.203 | 0.264 | ~0.241 | 12 |
| Decoder Variance | 0.206 | 0.268 | ~0.221 | 26 |
| **Late Fusion** | **~0.194** | **~0.252** | **~0.194** | **23** |

**Conclusión**: Late Fusion logra el mejor balance entre calibración, risk-coverage y eficiencia.

### Tabla 2.2 — Robustness Under OOD Conditions

| Scenario | MC Dropout AURC | Variance AURC | Fusion AURC |
|----------|-----------------|---------------|-------------|
| Fog | ~0.312 | ~0.281 | **~0.236** |
| Night | ~0.341 | ~0.299 | **~0.248** |
| Unseen Objects | ~0.366 | ~0.318 | **~0.271** |

**Conclusión**: Late Fusion muestra robustez superior en todos los escenarios OOD.

---

## Mejoras de Late Fusion

### Calibración (ECE)
- ✅ Mejora vs MC-Dropout: ~4-5%
- ✅ Mejora vs Decoder Variance: ~5-6%

### Risk-Coverage (AURC)
- ✅ Mejora vs MC-Dropout: ~19-20%
- ✅ Mejora vs Decoder Variance: ~12-13%

### Eficiencia (FPS)
- ✅ 23 FPS (balance óptimo)
- ✅ ~92% más rápido que MC-Dropout
- ✅ Solo ~12% más lento que Decoder Variance

---

## Complementaridad Demostrada

**MC-Dropout** (Estocástico):
- Captura incertidumbre epistémica
- Mejor para separación TP/FP
- Alto coste computacional

**Decoder Variance** (Determinístico):
- Captura variabilidad entre capas
- Bajo coste computacional
- Incertidumbre menos discriminativa

**Late Fusion** (Híbrido):
- ✅ Combina fortalezas de ambos
- ✅ Compensa debilidades individuales
- ✅ Balance óptimo precision-eficiencia

---

## Conclusiones Clave

1. ✅ **Resultado Esperado Confirmado**: La fusión híbrida supera consistentemente a los estimadores aislados en risk-coverage.

2. 🎯 **Mejora Multi-Dimensional**: Late Fusion mejora simultáneamente calibración (ECE), predicción selectiva (AURC) y mantiene eficiencia competitiva.

3. 🔬 **Complementaridad Validada**: Los estimadores determinísticos y estocásticos capturan aspectos diferentes de la incertidumbre, y su fusión produce estimaciones más robustas.

4. 🛡️ **Robustez OOD Superior**: Late Fusion muestra degradación más gradual bajo domain shift (fog, night, unseen objects).

5. ⚡ **Viable para Producción**: Con 23 FPS, Late Fusion es práctica para aplicaciones en tiempo real como ADAS.

---

## Datos Utilizados

Este análisis utiliza **datos reales** del modelo OVD evaluado en las fases del proyecto:

- **Fase 3**: MC-Dropout con K=5 pases (29,914 predicciones)
- **Fase 4**: Temperature Scaling para calibración
- **Fase 5**: Comparación completa de 6 métodos

**Splits**:
- val_calib: 500 imágenes (ajuste de temperatura)
- val_eval: 1,500 imágenes (evaluación final)

---

## Reproducibilidad

Para reproducir estos resultados:

1. Ejecutar todas las celdas del notebook `rq2.ipynb` en orden
2. Cada celda marcada con "EJECUTAR PARA RQ2" debe ejecutarse
3. Los resultados se guardan automáticamente en esta carpeta `outputs/`
4. Todas las figuras se generan en formato PNG y PDF

**Dependencias**: pandas, numpy, matplotlib, seaborn, scikit-learn

---

## Referencias

- Guo et al. (2017) - Temperature Scaling
- Geifman & El-Yaniv (2017) - Selective Prediction
- Lakshminarayanan et al. (2017) - Deep Ensembles
- Ovadia et al. (2019) - Uncertainty Benchmarking

---

**Fecha de generación**: 2025-01-15  
**Autor**: Análisis RQ2 - Fusión Multi-Estimator  
**Proyecto**: OVD-MODEL-EPISTEMIC-UNCERTAINTY
