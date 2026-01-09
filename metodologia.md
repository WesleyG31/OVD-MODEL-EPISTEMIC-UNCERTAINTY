# Proceso del Proyecto: Detección de Objetos y Cuantificación de Incertidumbre

## Descripción General

Este documento describe el **proceso paso a paso** de cómo se desarrolló el proyecto de detección de objetos con vocabulario abierto (OVD) y cuantificación de incertidumbre epistémica para conducción autónoma. El proyecto se ejecutó en **6 fases**, desde la preparación de datos hasta la comparación final de métodos.

---

## **FASE 1: Preparación de Datos**

### ¿Qué se hizo?
Se preparó el dataset BDD100K para ser utilizado con el modelo OWLv2.

### Pasos realizados:
1. **Descarga del dataset**: Se obtuvo BDD100K con 70,000 imágenes de conducción y sus anotaciones.
2. **Conversión de formato**: Se convirtieron las anotaciones de BDD100K al formato COCO (más compatible con modelos modernos).
3. **División de datos**: Se crearon tres conjuntos:
   - **Train**: 60% de las imágenes (para entrenamiento si fuera necesario)
   - **Validation**: 20% (para ajustes y validación)
   - **Test**: 20% (para evaluación final)
4. **Preparación de vocabulario**: Se definieron las 10 clases de objetos a detectar:
   - Peatones, ciclistas, vehículos, semáforos, señales de tráfico, motos, autobuses, camiones, trenes, y jinetes.

### Resultado:
Dataset listo para ser procesado por el modelo OWLv2.

---

## **FASE 2: Evaluación Baseline (Línea Base)**

### ¿Qué se hizo?
Se evaluó el rendimiento del modelo OWLv2 **sin ninguna modificación**, para establecer una línea base de comparación.

### Pasos realizados:
1. **Carga del modelo**: Se cargó OWLv2 (google/owlv2-large-patch14-ensemble) preentrenado.
2. **Inferencia estándar**: Se procesaron 5,000 imágenes del conjunto de validación, generando predicciones (cajas delimitadoras, clases, y puntajes de confianza).
3. **Cálculo de métricas de detección**:
   - **mAP** (mean Average Precision): 22.68%
   - **AP50**: 36.03%
   - **AP75**: 24.13%
4. **Análisis de rendimiento por clase**: Se identificó qué clases se detectan mejor (ej. vehículos) y cuáles peor (ej. jinetes).

### Resultado:
- Se estableció el **rendimiento base** del modelo.
- **mAP = 22.68%**: Este es el valor de referencia para comparar mejoras futuras.

---

## **FASE 3: Implementación de MC-Dropout**

### ¿Qué se hizo?
Se implementó **MC-Dropout** para cuantificar la incertidumbre epistémica del modelo.

### Pasos realizados:

#### 3.1. Activación de Dropout
1. Se modificó el modelo OWLv2 para **mantener el dropout activo durante la inferencia** (normalmente está desactivado).
2. Se identificaron las capas con dropout en el transformer visual y de texto.

#### 3.2. Inferencia Estocástica
1. **Se ejecutaron 50 pasadas forward** (forward passes) para cada imagen.
2. En cada pasada, el dropout enmascara diferentes neuronas aleatoriamente, generando predicciones ligeramente diferentes.
3. **Resultado por imagen**: 50 conjuntos de predicciones (cajas, clases, scores).

#### 3.3. Cálculo de Incertidumbre
Para cada predicción final, se calculó:
1. **Media de los scores**: Promedio de los 50 scores de confianza.
2. **Varianza de los scores**: Dispersión de los 50 scores (indica incertidumbre epistémica).
3. **Decoder Variance**: Varianza en las posiciones de las cajas predichas (indica incertidumbre espacial).

#### 3.4. Análisis de Resultados
Se realizó un análisis exhaustivo:
1. **True Positives (TP)**: Detecciones correctas.
2. **False Positives (FP)**: Detecciones incorrectas.
3. **Distribuciones de incertidumbre**:
   - FPs tienen **mayor incertidumbre** que TPs.
   - Media de varianza en FPs: 0.000127
   - Media de varianza en TPs: 0.000063
4. **AUROC** (capacidad de discriminar TP/FP usando incertidumbre):
   - **Score Variance**: AUROC = 0.614 (útil, pero no excelente)
   - **Decoder Variance**: AUROC = 0.604

### Resultado:
- **MC-Dropout funciona**: La incertidumbre distingue parcialmente entre TP y FP.
- **mAP = 22.68%**: Se mantuvo igual que baseline (MC-Dropout no mejora precisión, solo añade incertidumbre).
- **Datos guardados**: Se guardaron todas las predicciones con sus incertidumbres en archivos Parquet.

---

## **FASE 4: Calibración con Temperature Scaling**

### ¿Qué se hizo?
Se calibró el modelo **baseline** usando Temperature Scaling para mejorar la confiabilidad de los scores de confianza.

### Pasos realizados:

#### 4.1. Cálculo de ECE Inicial
1. Se evaluó el **Expected Calibration Error (ECE)** del modelo baseline:
   - **ECE = 18.64%**: El modelo es bastante mal calibrado (los scores no reflejan bien la probabilidad real de acierto).

#### 4.2. Búsqueda de Temperatura Óptima
1. Se probaron diferentes valores de temperatura (T) en un rango de 0.1 a 5.0.
2. Se dividió el conjunto de validación en:
   - **Calibration set**: Para encontrar la mejor T.
   - **Test set**: Para evaluar la mejora.
3. **Temperatura óptima encontrada**: T = 1.52
   - Minimiza el ECE en el calibration set.

#### 4.3. Aplicación de Temperature Scaling
1. Se ajustaron los scores de confianza dividiendo los logits por T = 1.52.
2. **Efecto**: Los scores se "suavizan" (menos extremos), mejorando la calibración.

#### 4.4. Evaluación de Resultados
1. **ECE después de TS**: 5.29% (reducción de 71.6% respecto al baseline).
2. **mAP**: Se mantuvo en 22.68% (TS no cambia la precisión, solo calibra los scores).
3. **Diagrama de confiabilidad**: Se observó que las predicciones calibradas están más cerca de la diagonal ideal.

### Resultado:
- **Baseline + TS está bien calibrado**: ECE = 5.29%.
- Los scores de confianza ahora reflejan mejor la probabilidad real de acierto.

---

## **FASE 5: Comparación Final de Métodos**

### ¿Qué se hizo?
Se compararon tres métodos:
1. **Baseline**: Modelo original sin modificaciones.
2. **Baseline + TS**: Modelo calibrado con Temperature Scaling (T = 1.52).
3. **MC-Dropout**: Modelo con cuantificación de incertidumbre (50 pasadas).

### Pasos realizados:

#### 5.1. Evaluación de Detección (mAP)
Se midió la precisión de cada método:
- **Baseline**: mAP = 22.68%
- **Baseline + TS**: mAP = 22.68% (igual, TS no cambia detección)
- **MC-Dropout**: mAP = 22.68% (igual, MC-Dropout no mejora precisión)

**Conclusión**: Todos los métodos tienen la misma precisión de detección.

#### 5.2. Evaluación de Calibración (ECE)
Se midió qué tan bien calibrados están los scores:
- **Baseline**: ECE = 18.64% (mal calibrado)
- **Baseline + TS**: ECE = 5.29% (bien calibrado)
- **MC-Dropout**: ECE = 18.82% (mal calibrado, similar a baseline)

**Conclusión**: Solo Baseline + TS está bien calibrado.

#### 5.3. Cuantificación de Incertidumbre (AUROC)
Se midió la capacidad de identificar FPs usando incertidumbre:
- **Baseline**: No tiene medida de incertidumbre explícita.
- **Baseline + TS**: No añade incertidumbre, solo calibra.
- **MC-Dropout**: AUROC = 0.614 (puede identificar FPs moderadamente bien).

**Conclusión**: Solo MC-Dropout cuantifica incertidumbre epistémica.

#### 5.4. Análisis de Costo Computacional
Se midió el tiempo de inferencia:
- **Baseline**: ~1.5 segundos/imagen (1x)
- **Baseline + TS**: ~1.5 segundos/imagen (1x, solo ajusta scores después)
- **MC-Dropout**: ~75 segundos/imagen (50x más lento, por las 50 pasadas)

**Conclusión**: MC-Dropout es muy costoso computacionalmente.

#### 5.5. Recomendaciones Prácticas
Se generaron recomendaciones basadas en el escenario de uso:

**Para sistemas de alerta al conductor** (requieren rapidez):
- Usar **Baseline + TS** (bien calibrado y rápido).
- Filtrar predicciones con score < 0.3.

**Para sistemas críticos de seguridad** (requieren confiabilidad):
- Usar **MC-Dropout** (cuantifica incertidumbre).
- Filtrar predicciones con score_variance > 0.00009.
- Combinar con Baseline + TS si se necesita calibración.

**Para mapeo/percepción no crítica**:
- Usar **Baseline** (rápido, sin calibración necesaria).

### Resultado:
- **Comparación completa** de los tres métodos.
- **Recomendaciones claras** para cada escenario de uso.
- **Visualizaciones generadas**: Diagramas de confiabilidad, comparación de ECE, distribuciones de incertidumbre, etc.

---

## **FASE 6: Verificación y Documentación**

### ¿Qué se hizo?
Se verificó que todo el proyecto estuviera completo y se documentaron todos los resultados.

### Pasos realizados:
1. **Verificación de variables**: Se confirmó que todas las variables clave estuvieran guardadas correctamente.
2. **Verificación de resultados**: Se validó que todos los archivos de resultados (JSON, Parquet, imágenes) existieran.
3. **Documentación completa**: Se crearon múltiples archivos markdown con:
   - Explicaciones detalladas de conceptos.
   - Resultados numéricos y visualizaciones.
   - Recomendaciones prácticas.
4. **Resumen ejecutivo**: Se generó un informe final consolidando todos los hallazgos.

### Resultado:
- Proyecto completado y documentado.
- Todos los resultados verificados y reproducibles.

---

## Resumen del Flujo de Trabajo

```
1. PREPARACIÓN DE DATOS (Fase 1)
   └─> Dataset BDD100K listo (70K imágenes, 10 clases)

2. EVALUACIÓN BASELINE (Fase 2)
   └─> mAP = 22.68%, ECE = 18.64%
   
3. MC-DROPOUT (Fase 3)
   └─> 50 pasadas forward por imagen
   └─> Incertidumbre calculada (varianza de scores y decoder)
   └─> AUROC = 0.614 (identifica FPs moderadamente)
   └─> mAP = 22.68% (sin cambios en precisión)

4. CALIBRACIÓN (Fase 4)
   └─> Temperature Scaling aplicado (T = 1.52)
   └─> ECE = 5.29% (mejora del 71.6%)
   └─> mAP = 22.68% (sin cambios en precisión)

5. COMPARACIÓN FINAL (Fase 5)
   └─> Baseline: Rápido, mal calibrado, sin incertidumbre
   └─> Baseline + TS: Rápido, bien calibrado, sin incertidumbre
   └─> MC-Dropout: Lento, mal calibrado, con incertidumbre
   └─> Recomendaciones por escenario de uso

6. VERIFICACIÓN (Fase 6)
   └─> Proyecto completo, documentado y verificado
```

---

## Archivos Clave Generados

### Datos y Métricas
- `fase 2/outputs/baseline_results.json`: Métricas del baseline
- `fase 3/outputs/mc_dropout/mc_stats_labeled.parquet`: Predicciones con incertidumbre
- `fase 3/outputs/mc_dropout/tp_fp_analysis.json`: Análisis TP/FP
- `fase 4/outputs/calibration/calibration_metrics.json`: Métricas de calibración
- `fase 5/outputs/comparison/final_report.json`: Comparación final de métodos
- `fase 5/outputs/comparison/temperatures.json`: Temperaturas óptimas

### Visualizaciones
- `fase 5/outputs/comparison/final_comparison_summary.png`: Resumen visual de métodos
- `fase 5/outputs/comparison/reliability_diagrams.png`: Diagramas de confiabilidad
- `fase 5/outputs/comparison/ece_comparison.png`: Comparación de ECE
- `fase 3/outputs/mc_dropout/uncertainty_distributions.png`: Distribuciones de incertidumbre

### Documentación
- `README.md`: Descripción general del proyecto
- `FINAL_SUMMARY.md`: Resumen ejecutivo final
- `PROYECTO_COMPLETADO_FINAL.md`: Informe completo de resultados
- `resultados.md`: Resultados completos con explicaciones detalladas
- `resultados_2.md`: Este documento (proceso del proyecto)

---

## Conclusiones del Proceso

### ✅ Lo que funcionó bien:
1. **División del trabajo en fases**: Facilitó el seguimiento y la validación.
2. **MC-Dropout**: Cuantifica incertidumbre de manera efectiva (AUROC = 0.614).
3. **Temperature Scaling**: Mejora significativamente la calibración (ECE: 18.64% → 5.29%).
4. **Documentación exhaustiva**: Todos los pasos están documentados y reproducibles.

### ⚠️ Limitaciones encontradas:
1. **MC-Dropout es muy lento**: 50x más tiempo de inferencia (no viable para tiempo real).
2. **MC-Dropout mal calibrado**: ECE alto (18.82%), requeriría calibración adicional.
3. **mAP modesto**: 22.68% indica que el modelo podría mejorar con fine-tuning.

### 💡 Aprendizajes clave:
1. **Calibración ≠ Precisión**: TS mejora calibración pero no mAP.
2. **Incertidumbre ≠ Precisión**: MC-Dropout añade incertidumbre pero no mejora mAP.
3. **Trade-offs importantes**: Velocidad vs. incertidumbre, calibración vs. costo computacional.
4. **Importancia del contexto**: La elección del método depende del escenario de uso específico.

---

## Próximos Pasos Potenciales

Si se quisiera extender este proyecto:

1. **Fine-tuning del modelo**: Entrenar OWLv2 específicamente en BDD100K para mejorar mAP.
2. **Optimización de MC-Dropout**: Reducir el número de pasadas (ej. 10 en lugar de 50) para balance entre velocidad e incertidumbre.
3. **Métodos híbridos**: Combinar MC-Dropout con Temperature Scaling para tener tanto incertidumbre como calibración.
4. **Ensembles**: Usar múltiples modelos en lugar de MC-Dropout para cuantificar incertidumbre.
5. **Pruebas en tiempo real**: Implementar en hardware de conducción autónoma real para evaluar viabilidad práctica.
6. **Análisis de casos extremos**: Evaluar rendimiento en condiciones adversas (lluvia, noche, oclusiones).

---

**Fecha de finalización del proyecto**: Enero 2025  
**Autor**: Proyecto de detección de objetos y cuantificación de incertidumbre para conducción autónoma  
**Objetivo alcanzado**: ✅ Evaluación completa de métodos de cuantificación de incertidumbre y calibración en detección de objetos con vocabulario abierto.
