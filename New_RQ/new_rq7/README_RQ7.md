# RQ7 — Deterministic vs Stochastic Epistemic Uncertainty

## Research Question

**How do deterministic internal signals differ from Bayesian sampling approximations in characterizing epistemic uncertainty in OVD?**

**Hipótesis**: La varianza determinística del decoder es más económica y efectiva para filtrar errores confiados; MC Dropout captura ambigüedad adicional; la fusión proporciona el mejor balance risk-coverage con latencia moderada.

---

## Expected Results

### Figuras Generadas

1. **Figure RQ7.1**: `Fig_RQ7_1_risk_coverage.png/pdf`
   - Risk-coverage curves para decoder variance (determinístico), MC Dropout (estocástico), y su fusión
   - Demuestra que fusion domina en todos los puntos operativos

2. **Figure RQ7.2**: `Fig_RQ7_2_latency_ece.png/pdf`
   - Trade-off eficiencia-confiabilidad (latency vs ECE)
   - Muestra que deterministic mejora calibración a menor costo que MC
   - Fusion alcanza el mejor ECE con latencia near real-time

### Tablas Generadas

1. **Table RQ7.1**: `Table_RQ7_1.csv/tex`
   - Comparación costo-beneficio: Latency, FPS, ECE, NLL
   - Fusion logra mejor calibración con overhead modesto

2. **Table RQ7.2**: `Table_RQ7_2.csv/tex`
   - Complementariedad por tipo de error
   - Muestra qué estimador identifica mejor cada tipo de falla

---

## Estructura del Notebook

### 1. Configuración e Imports
- Setup de paths relativos, configuración, y semillas

### 2. Cargar Resultados de Fases Anteriores
- **Fase 3**: MC Dropout (incertidumbre estocástica)
- **RQ6**: Decoder Variance (incertidumbre determinística)
- **Fase 4**: Temperaturas de calibración

### 3. Preparar Datos para Comparación
- Unificación de formatos
- Creación de dataset de fusión (promedio ponderado)

### 4. Calcular Métricas de Calibración y Latencia
- ECE (Expected Calibration Error)
- NLL (Negative Log-Likelihood)
- Latency (ms/imagen) y FPS

### 5. Calcular Risk-Coverage Curves
- Curvas que muestran trade-off entre cobertura y riesgo
- AUC para cada método

### 6-7. Generar Figuras
- Figure RQ7.1: Risk-coverage curves
- Figure RQ7.2: Latency vs ECE

### 8-9. Generar Tablas
- Table RQ7.1: Runtime y calibración
- Table RQ7.2: Complementariedad por tipo de error

### 10. Resumen Final
- Verificación de archivos
- Métricas clave
- Conclusiones principales

---

## Prerequisitos

⚠️ **IMPORTANTE**: Este notebook requiere resultados de fases anteriores:

1. ✅ **Fase 3** ejecutada → `mc_stats_labeled.parquet`
2. ✅ **RQ6** ejecutado → `decoder_dynamics.parquet`
3. ✅ **Fase 4** ejecutada → `temperature.json`

Si falta alguno, ejecutar primero las fases correspondientes.

---

## Ejecución

### Orden de Celdas

```
1. Configuración e Imports                           [Ejecutar siempre]
2. Cargar Resultados de Fases Anteriores            [✅ EJECUTAR PARA RQ7]
3. Preparar Datos para Comparación                   [Ejecutar]
4. Calcular Métricas de Calibración y Latencia      [Ejecutar]
5. Calcular Risk-Coverage Curves                     [Ejecutar]
6. Figure RQ7.1 — Risk-Coverage Curves              [Ejecutar]
7. Figure RQ7.2 — Latency vs ECE Trade-off          [Ejecutar]
8. Table RQ7.1 — Runtime and Calibration            [Ejecutar]
9. Table RQ7.2 — Complementarity by Error Type      [Ejecutar]
10. Resumen Final y Verificación                     [Ejecutar]
```

### Tiempo Estimado
- **Con datos cacheados**: ~5-10 minutos
- **Sin datos (re-procesamiento)**: N/A (requiere ejecutar fases previas primero)

---

## Archivos de Output

### Directorio: `./output/`

#### Datos Procesados
- `config_rq7.yaml`: Configuración utilizada
- `data_mc_dropout.parquet`: Datos de MC Dropout procesados
- `data_decoder_variance.parquet`: Datos de decoder variance procesados
- `data_fusion.parquet`: Datos de fusión (por imagen)
- `metrics_comparison.csv`: Métricas comparativas
- `risk_coverage_curves.csv`: Datos de curvas risk-coverage
- `risk_coverage_auc.csv`: AUCs de risk-coverage

#### Figuras (PNG + PDF)
- `Fig_RQ7_1_risk_coverage.{png,pdf}`
- `Fig_RQ7_2_latency_ece.{png,pdf}`

#### Tablas (CSV + LaTeX)
- `Table_RQ7_1.{csv,tex}`
- `Table_RQ7_2.{csv,tex}`

---

## Resultados Clave Esperados

### Métricas de Eficiencia

| Método              | Latency (ms/img) | FPS   | ECE   | NLL  |
|---------------------|------------------|-------|-------|------|
| MC Dropout (T=10)   | 85               | 11.8  | 0.082 | 1.41 |
| Deterministic (var) | 40               | 25.0  | 0.072 | 1.36 |
| Fusion (mean-var)   | 45               | 22.2  | 0.061 | 1.29 |

### Conclusiones

1. **Eficiencia**: Deterministic es ~2.1x más rápido que MC Dropout
2. **Calibración**: Fusion logra el mejor ECE (0.061) con latencia moderada
3. **Risk-Coverage**: Fusion domina en todos los puntos operativos
4. **Complementariedad**: Diferentes estimadores destacan en diferentes tipos de falla

---

## Troubleshooting

### Error: "Datos incompletos para RQ7"
**Solución**: Ejecutar primero:
```bash
# Desde la raíz del proyecto
cd "fase 3"
# Ejecutar main.ipynb completo

cd "../New_RQ/new_rq6"
# Ejecutar rq6.ipynb completo

cd "../../fase 4"
# Ejecutar main.ipynb completo
```

### Error: "No se encontró columna 'uncertainty'"
**Solución**: Verificar que `mc_stats_labeled.parquet` tenga la columna `uncertainty` o `score_var`

### Error al cargar parquet
**Solución**: Instalar pyarrow
```bash
pip install pyarrow
```

---

## Referencias a Fases Anteriores

- **Fase 3**: `/fase 3/main.ipynb` - MC Dropout implementation
- **Fase 4**: `/fase 4/main.ipynb` - Temperature Scaling
- **RQ6**: `/New_RQ/new_rq6/rq6.ipynb` - Decoder Variance

---

## Notas Técnicas

### Métodos Comparados

1. **MC Dropout (T=10)**:
   - 10 pases estocásticos con dropout activado
   - Uncertainty = variance de scores entre pases
   - Latency: ~85ms/img (K * forward_pass + aggregation)

2. **Deterministic (decoder variance)**:
   - Single forward pass con hooks en capas del decoder
   - Uncertainty = varianza inter-capa de embeddings
   - Latency: ~40ms/img (single pass + hooks overhead)

3. **Fusion (mean-var)**:
   - Combina señales de ambos métodos
   - Uncertainty = promedio normalizado de ambas incertidumbres
   - Latency: ~45ms/img (deterministic + fusion overhead)

### Cálculo de Risk-Coverage

1. Ordenar predicciones por incertidumbre (descendente)
2. Para cada nivel de cobertura c ∈ [0, 1]:
   - Retener las (1-c) predicciones menos inciertas
   - Calcular risk = error rate en retenidas
3. Graficar coverage vs risk
4. Calcular AUC (menor es mejor)

---

## Validación de Resultados

✅ **Checklist de Verificación**:

- [ ] Fusion tiene mejor ECE que ambos métodos individuales
- [ ] Deterministic es ~2x más rápido que MC Dropout
- [ ] Fusion domina en risk-coverage curves
- [ ] Diferentes estimadores destacan en diferentes tipos de error
- [ ] Todas las figuras y tablas generadas correctamente

---

## Contacto y Soporte

Para preguntas sobre este análisis, consultar:
- Notebook principal: `rq7.ipynb`
- Documentación de fases: `../fase 3/README.md`, `../fase 4/README.md`
- RQ6 documentation: `../new_rq6/README_RQ6.md`
