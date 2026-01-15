# 📋 INSTRUCCIONES DE EJECUCIÓN - RQ5

## ⚠️ IMPORTANTE: Leer antes de ejecutar

Este notebook **NO genera datos simulados**. Utiliza **resultados reales** de las evaluaciones previas (Fase 3, 4 y 5).

---

## ✅ Pre-requisitos

Antes de ejecutar `rq5.ipynb`, asegúrate de que existen estos archivos:

### Fase 5 - Comparación Completa:
```
../../fase 5/outputs/comparison/
├── detection_comparison.csv
├── calibration_comparison.csv  
├── uncertainty_auroc_comparison.csv
├── risk_coverage_auc.json
├── temperatures.json
├── eval_baseline.csv           ← CRÍTICO
├── eval_mc_dropout.csv          ← CRÍTICO
└── eval_mc_dropout_ts.csv       ← CRÍTICO
```

### Verificación Rápida:

Ejecutar en terminal:

```powershell
# Verificar archivos de Fase 5
Test-Path "../../fase 5/outputs/comparison/eval_baseline.csv"
Test-Path "../../fase 5/outputs/comparison/eval_mc_dropout.csv"
Test-Path "../../fase 5/outputs/comparison/eval_mc_dropout_ts.csv"
```

Si alguno devuelve `False`, debes ejecutar primero:

```
../../fase 5/main.ipynb
```

---

## 🚀 Pasos de Ejecución

### Opción 1: Ejecutar Todo (Recomendado)

1. Abrir `rq5.ipynb` en VS Code
2. **Kernel > Restart Kernel and Run All Cells**
3. Esperar ~10-15 minutos
4. Verificar outputs en `./outputs/`

### Opción 2: Ejecutar Paso a Paso

Ejecutar celdas en orden:

#### Paso 1: Configuración (Celdas 1-2)
```
✅ Imports y setup
✅ Configuración guardada en config_rq5.yaml
```

#### Paso 2: Cargar Datos (Celdas 3-4)
```
✅ Carga resultados de Fase 5
✅ Carga predicciones detalladas con TP/FP
```

#### Paso 3: Decision Fusion (Celda 5)
```
✅ Calcula risk scores baseline
✅ Calcula risk scores fusionados
✅ Guarda baseline_risk.csv y fused_risk.csv
```

#### Paso 4: Selective Prediction (Celda 6)
```
✅ Evalúa coverage 100%, 80%, 60%
✅ Genera Table 5.1
✅ Guarda table_5_1_selective_prediction.csv
```

#### Paso 5: FP Reduction (Celda 7)
```
✅ Calcula FP/FN rates
✅ Genera Table 5.2
✅ Guarda table_5_2_fp_reduction.csv
```

#### Paso 6: Figuras (Celdas 8-9)
```
✅ Figure 5.1: Decision Fusion Architecture
✅ Figure 5.2: Risk-Coverage Trade-off
✅ Guarda PNG y PDF
```

#### Paso 7: Resumen (Celda 10)
```
✅ Consolida resultados
✅ Genera RQ5_FINAL_REPORT.txt
✅ Guarda rq5_summary.json
```

#### Paso 8: Verificación (Celda 11)
```
✅ Compara esperado vs obtenido
✅ Muestra status de cada métrica
```

---

## 📊 Verificación de Resultados

Después de ejecutar, verificar que existen:

### Archivos CSV:
- [ ] `outputs/table_5_1_selective_prediction.csv`
- [ ] `outputs/table_5_2_fp_reduction.csv`
- [ ] `outputs/baseline_risk.csv`
- [ ] `outputs/fused_risk.csv`
- [ ] `outputs/risk_coverage_curves_data.csv`

### Figuras:
- [ ] `outputs/figure_5_1_decision_fusion_architecture.png`
- [ ] `outputs/figure_5_1_decision_fusion_architecture.pdf`
- [ ] `outputs/figure_5_2_risk_coverage_tradeoff.png`
- [ ] `outputs/figure_5_2_risk_coverage_tradeoff.pdf`

### Reportes:
- [ ] `outputs/RQ5_FINAL_REPORT.txt`
- [ ] `outputs/rq5_summary.json`
- [ ] `outputs/config_rq5.yaml`

### Comando de Verificación:

```powershell
Get-ChildItem ./outputs/ | Select-Object Name
```

Deberías ver **12 archivos** en total.

---

## 🔍 Interpretación de Resultados

### Table 5.1 - Selective Prediction Results

**Qué muestra**: Riesgo (FP Rate) a diferentes niveles de cobertura

**Interpretación**:
- **Coverage 100%**: Todas las predicciones retenidas → mayor riesgo
- **Coverage 80%**: Rechaza 20% más inciertas → menor riesgo
- **Coverage 60%**: Rechaza 40% más inciertas → menor riesgo aún

**Esperado**: Fused Risk < Baseline Risk en todos los niveles

### Table 5.2 - False-Positive Reduction

**Qué muestra**: Tasas de FP y FN para Baseline vs Decision Fusion

**Interpretación**:
- **FP Rate ↓**: Queremos que baje (menos falsos positivos)
- **FN Rate**: Puede subir ligeramente (trade-off aceptable)

**Esperado**: FP Rate reducción ~40-50%, FN Rate aumento <10%

### Figure 5.1 - Architecture

**Qué muestra**: Diagrama del pipeline de Decision Fusion

**Componentes**:
1. Camera Input → GroundingDINO
2. Branches: MC-Dropout + Temperature Scaling
3. Fusion Layer: Combina señales
4. Decision: Rechaza alto riesgo, acepta bajo riesgo

### Figure 5.2 - Risk-Coverage Trade-off

**Qué muestra**: Curvas de riesgo vs cobertura

**Interpretación**:
- **Línea verde (Fused) debajo de línea roja (Baseline)**: ✅ Mejora
- **Área sombreada verde**: Región de mejora
- **Puntos marcados**: Coverage 100%, 80%, 60%

---

## ⚠️ Errores Comunes

### Error 1: FileNotFoundError

```
FileNotFoundError: eval_baseline.csv not found
```

**Causa**: Fase 5 no ejecutada o incompleta

**Solución**: 
```bash
cd ../../fase\ 5/
# Ejecutar main.ipynb completo
```

### Error 2: KeyError en columnas

```
KeyError: 'uncertainty_epistemic'
```

**Causa**: Predicciones sin incertidumbre

**Solución**: Verificar que Fase 3 generó correctamente las predicciones con MC-Dropout

### Error 3: Figuras vacías

```
plt.show() no muestra nada
```

**Causa**: Backend de matplotlib no configurado

**Solución**:
```python
import matplotlib
matplotlib.use('Agg')  # Antes de plt.show()
```

### Error 4: Valores NaN en tablas

```
Risk = NaN
```

**Causa**: División por cero (no hay predicciones retenidas)

**Solución**: Verificar que eval_*.csv tienen datos válidos

---

## 📈 Valores Esperados Aproximados

### Table 5.1:

| Coverage | Baseline Risk | Fused Risk | Mejora |
|----------|---------------|------------|--------|
| 100%     | 0.15-0.20     | 0.12-0.16  | 15-25% |
| 80%      | 0.12-0.16     | 0.07-0.10  | 35-45% |
| 60%      | 0.10-0.14     | 0.04-0.07  | 50-60% |

### Table 5.2:

| Métrica | Baseline | Fused | Cambio |
|---------|----------|-------|--------|
| FP Rate | 0.16-0.20 | 0.08-0.12 | -40 a -50% |
| FN Rate | 0.06-0.08 | 0.07-0.09 | +5 a +15% |

**Nota**: Valores exactos dependen de los resultados de Fase 5.

---

## 🎯 Criterios de Éxito

Para considerar RQ5 completada:

- [x] ✅ Table 5.1 generada con 3 niveles de cobertura
- [x] ✅ Table 5.2 generada con FP/FN rates
- [x] ✅ Figure 5.1 muestra arquitectura clara
- [x] ✅ Figure 5.2 muestra curvas con mejora visible
- [x] ✅ Fused Risk < Baseline Risk en todos los casos
- [x] ✅ FP Rate reducción significativa (>30%)
- [x] ✅ Reporte final generado
- [x] ✅ Todos los archivos guardados

---

## 📞 Ayuda

Si encuentras problemas:

1. Revisar `README.md` en esta carpeta
2. Consultar `../../rq_no5.md` para contexto
3. Verificar que Fase 3, 4 y 5 estén completas
4. Comprobar paths relativos

---

**✅ Con estas instrucciones, puedes ejecutar RQ5 y generar todos los resultados requeridos**

**⏱️ Tiempo total: ~15 minutos**

**💾 Output: 12 archivos (tablas, figuras, reportes)**
