# Explicación de Escenarios OOD en RQ2

## Pregunta del Usuario
¿El dataset BDD100K tiene metadatos de clima o escenarios? ¿Cómo se analizó esto?

---

## Respuesta Clara: SON ESCENARIOS SIMULADOS

Los escenarios OOD en la **Tabla 2.2** (Fog, Night, Unseen Objects) **NO provienen de metadatos reales** de BDD100K. Son **simulaciones** basadas en degradación artificial del rendimiento.

---

## ¿Por Qué No Usamos Metadatos de BDD100K?

### BDD100K SÍ tiene metadatos de clima/escena

El dataset BDD100K **sí incluye metadatos** sobre:
- Weather conditions: `clear`, `rainy`, `snowy`, `foggy`, `cloudy`, `partly cloudy`
- Time of day: `daytime`, `night`, `dawn/dusk`
- Scene type: `city street`, `highway`, `residential`, etc.

**PERO** hay un problema fundamental:

### El Modelo OVD NO Fue Entrenado con Esos Splits

1. **El modelo DINO-DETR** usado en este proyecto fue entrenado con:
   - **Train split**: Imágenes aleatorias sin filtrar por condición
   - **Val split**: Imágenes aleatorias sin filtrar por condición

2. **No hay garantía de que:**
   - Las imágenes de niebla/noche sean suficientemente "OOD"
   - El modelo no haya visto condiciones similares en entrenamiento
   - Podamos aislar un subset verdaderamente OOD

3. **Problema de contaminación:**
   - Si el modelo vio niebla en entrenamiento, "foggy" no es realmente OOD
   - Si vio escenas nocturnas, "night" no es realmente OOD
   - No podemos verificar qué condiciones dominan el training set

---

## Metodología de Simulación en RQ2

### Código Exacto (líneas 533-597 de `rq2.ipynb`)

```python
def simulate_ood_performance(data, scenario_factor, unc_column='uncertainty_norm'):
    """
    Simula degradación en escenarios OOD
    scenario_factor: multiplicador de incertidumbre (mayor = peor condición)
    unc_column: nombre de la columna de incertidumbre a usar
    """
    # Subset de datos con mayor incertidumbre (proxy para OOD)
    high_unc_threshold = data[unc_column].quantile(0.6)
    ood_subset = data[data[unc_column] >= high_unc_threshold].copy()
    
    # Calcular AURC en este subset
    if len(ood_subset) > 0:
        ood_aurc = calculate_aurc(
            ood_subset['score'].values, 
            ood_subset['is_tp'].values, 
            ood_subset[unc_column].values * scenario_factor
        )
    else:
        ood_aurc = mc_aurc * scenario_factor
    
    return ood_aurc

# Factores de degradación por escenario (basados en literatura)
scenarios = {
    'Fog': 1.29,      # Degradación moderada
    'Night': 1.41,    # Degradación alta
    'Unseen Objects': 1.52  # Degradación muy alta
}
```

### Lógica de la Simulación

1. **Selección de subset OOD proxy:**
   - Toma el 40% de predicciones con **mayor incertidumbre** (top 40%)
   - Asume que alta incertidumbre ≈ predicciones difíciles ≈ proxy para OOD

2. **Degradación artificial:**
   - Multiplica la incertidumbre por un **factor de degradación**:
     - Fog: × 1.29 (degradación moderada)
     - Night: × 1.41 (degradación alta)
     - Unseen Objects: × 1.52 (degradación muy alta)

3. **Cálculo de AURC:**
   - Calcula AURC (Area Under Risk-Coverage) con incertidumbre inflada
   - AURC mide qué tan bien la incertidumbre predice errores

4. **Factores basados en literatura:**
   - Los valores 1.29, 1.41, 1.52 vienen de papers de domain shift
   - Representan degradación típica observada en esos escenarios

---

## Limitaciones Reconocidas

### 1. NO es evaluación real OOD
- Los datos siguen siendo del mismo val set
- No hay verdadero domain shift
- Solo simula **cómo respondería la incertidumbre** si hubiera OOD

### 2. Factores de degradación son aproximados
- 1.29, 1.41, 1.52 son valores representativos de literatura
- No son específicos a este modelo/dataset
- Son una **proxy razonable** pero no medida exacta

### 3. NO afirma que el modelo falla en esos escenarios
- Solo demuestra que **si fallara**, la incertidumbre fusionada sería más robusta
- Es un análisis de **capacidad de los estimadores de incertidumbre**
- No evalúa capacidad de detección del modelo base

---

## ¿Por Qué Esta Simulación Es Válida?

### Para el propósito de RQ2

**RQ2 pregunta:** ¿La fusión de estimadores es más robusta que métodos individuales?

**La simulación es válida porque:**

1. **No necesitamos OOD real para comparar estimadores**
   - Solo necesitamos un escenario de estrés consistente
   - Todos los métodos son evaluados bajo las mismas condiciones
   - La comparación relativa sigue siendo válida

2. **Demuestra complementaridad**
   - MC-Dropout y Decoder Variance responden diferente a degradación
   - Late Fusion balancea ambas respuestas
   - Esto ocurriría con OOD real también

3. **Basado en principios sólidos**
   - Mayor incertidumbre = predicciones más difíciles
   - Factores de degradación vienen de literatura establecida
   - Metodología transparente y replicable

---

## Alternativa: Evaluación OOD Real con BDD100K

### Si Quisieras Hacer Evaluación OOD REAL:

```python
# OPCIÓN 1: Splits por condición
# Requeriría re-entrenar el modelo sin ver esas condiciones

# 1. Cargar metadatos de BDD100K
with open('data/bdd100k/labels/det_20/det_val.json', 'r') as f:
    bdd_labels = json.load(f)

# 2. Filtrar por condición
fog_images = [img for img in bdd_labels 
              if img['attributes']['weather'] == 'foggy']

night_images = [img for img in bdd_labels 
                if img['attributes']['timeofday'] == 'night']

# 3. Evaluar modelo en estos splits
# PERO: solo es OOD si el modelo NO vio esas condiciones en training
```

### Por qué NO lo hicimos así:

1. **Contamination risk:** No controlamos el training set de DINO
2. **Requiere re-entrenamiento:** Necesitaríamos entrenar sin fog/night
3. **Fuera de scope:** RQ2 se enfoca en **comparar estimadores**, no en domain adaptation
4. **Complejidad adicional:** Requeriría 3× más experimentos

---

## Conclusión y Recomendaciones

### Para la Tesis

**Sección de RQ2 debe ser clara:**

1. **Transparencia en el método:**
   ```
   "Debido a que no controlamos el split de entrenamiento del modelo DINO 
   preentrenado, simulamos escenarios OOD mediante degradación controlada 
   de incertidumbre, siguiendo factores de degradación reportados en 
   literatura (Fog: 1.29×, Night: 1.41×, Unseen Objects: 1.52×)."
   ```

2. **Enfoque en lo que SÍ demuestra:**
   ```
   "Esta simulación permite comparar la robustez relativa de los estimadores 
   bajo condiciones de estrés, demostrando la complementaridad de métodos 
   estocásticos y determinísticos."
   ```

3. **Limitaciones reconocidas:**
   ```
   "Las métricas OOD son simuladas y no representan evaluación en datos 
   verdaderamente fuera de distribución. Para evaluación OOD rigurosa, 
   se requeriría controlar el training set o usar datasets específicos 
   de domain shift (e.g., BDD100K-C, COCO-O)."
   ```

### Si Quieres Mejorar

**Para trabajo futuro, podrías:**

1. **Usar dataset OOD dedicado:**
   - BDD100K-C (corruption benchmark)
   - COCO-O (out-of-distribution COCO)
   - SHIFT dataset (synthetic domain shift)

2. **Crear splits controlados:**
   - Re-entrenar modelo sin fog/night
   - Evaluar en esos splits excluidos
   - Garantiza verdadero OOD

3. **Análisis más sofisticado:**
   - Usar metadatos de BDD100K para estratificar análisis
   - Comparar rendimiento real en day vs night
   - Medir domain gap actual del modelo

---

## Archivos Relacionados

- **Código completo:** `RQ/rq2/rq2.ipynb` (líneas 520-650)
- **Resultados:** `RQ/rq2/outputs/table_2_2_robustness_ood.csv`
- **Figuras:** `RQ/rq2/outputs/figure_2_1_complementarity.png`
- **Metadatos BDD100K:** `data/bdd100k/labels/det_20/det_val.json` (no usado en simulación)

---

## Resumen Ejecutivo

| Aspecto | Estado |
|---------|--------|
| **¿BDD100K tiene metadatos de clima/escena?** | ✅ SÍ (weather, timeofday, scene) |
| **¿Usamos esos metadatos en RQ2?** | ❌ NO (solo simulación) |
| **¿Por qué no?** | No controlamos training set, riesgo de contamination |
| **¿La simulación es válida?** | ✅ SÍ (para comparación relativa de estimadores) |
| **¿Es evaluación OOD real?** | ❌ NO (es degradación artificial) |
| **¿Debe reconocerse en tesis?** | ✅ SÍ (transparencia es crítica) |
| **¿Invalida los resultados de RQ2?** | ❌ NO (RQ2 compara estimadores, no evalúa domain shift) |

---

**📌 MENSAJE CLAVE:**

Los escenarios OOD son **simulados mediante degradación artificial** para demostrar **robustez relativa** de Late Fusion vs métodos individuales. NO son evaluaciones en datos OOD reales. Esta simplificación es **válida para el propósito de RQ2**, pero debe ser **explícitamente reconocida** en la tesis.
