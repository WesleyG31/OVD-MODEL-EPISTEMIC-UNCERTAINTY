# 📋 GUÍA EXACTA: QUÉ CELDAS EJECUTAR EN FASE 5 PARA RQ1

## 🎯 OBJETIVO
Regenerar SOLO el archivo `eval_decoder_variance.json` con el campo `layer_uncertainties` para RQ1.

---

## ✅ CELDAS A EJECUTAR (EN ORDEN)

### PASO 1: Configuración Inicial

| # Celda | Título | ¿Ejecutar? | Tiempo |
|---------|--------|------------|--------|
| 1 | "⚡ OPTIMIZACIÓN: Reutilizar Resultados..." | ✅ SÍ | 1 seg |
| 2 | "Fase 5: Comparación Completa..." | ✅ SÍ | 1 seg |
| 3 | "## 1. Configuración e Imports" | ✅ SÍ | 5 seg |

**Contenido de Celda 3**:
```python
import os
import sys
import json
import yaml
...
```

---

### PASO 2: Cargar Modelo

| # Celda | Título | ¿Ejecutar? | Tiempo |
|---------|--------|------------|--------|
| ~5-10 | Celdas de carga del modelo GroundingDINO | ✅ SÍ | 30 seg |

**Busca la celda que contiene**:
```python
from groundingdino.util.inference import load_model
model = load_model(model_config, model_weights)
```

---

### PASO 3: Definir Funciones de Inferencia

| # Celda | Título | ¿Ejecutar? | Tiempo |
|---------|--------|------------|--------|
| ~15-20 | Función `inference_baseline()` | ⚠️ OPCIONAL | 1 seg |
| ~21-25 | Función `inference_mc_dropout()` | ⚠️ OPCIONAL | 1 seg |
| **~26-30** | **Función `inference_decoder_variance()`** | **✅ SÍ (CRÍTICO)** | **1 seg** |

**Busca la celda que contiene**:
```python
def inference_decoder_variance(model, image_path, text_prompt, conf_thresh, device):
    """Método 5: Varianza entre capas del decoder (single-pass)"""
    # ...
```

⚠️ **IMPORTANTE**: Esta función DEBE incluir:
```python
detections.append({
    'bbox': box.tolist(),
    'score': score_clipped,
    'logit': logit,
    'category': cat,
    'uncertainty': uncertainty,
    'layer_uncertainties': layer_uncertainties_list,  # ← DEBE EXISTIR
    'layer_count': len(layer_uncertainties_list)      # ← DEBE EXISTIR
})
```

---

### PASO 4: Inferencia en val_calib (SOLO decoder_variance)

| Sección | ¿Ejecutar? | Tiempo |
|---------|------------|--------|
| "## 4. Inferencia en val_calib..." | **✅ SÍ, pero SOLO las líneas de decoder_variance** | 10-15 min |

**Busca esta parte del código**:
```python
for img_id in tqdm(img_ids_calib, desc="Procesando calibración"):
    # ...
    
    # ========================================================================
    # Método 5: Decoder Variance  ← EJECUTAR ESTA PARTE
    # ========================================================================
    preds_dec = inference_decoder_variance(model, img_path, TEXT_PROMPT, ...)
    
    for pred in preds_dec:
        methods_calib_data['decoder_variance'].append({...})
```

⚠️ **NO EJECUTES**:
- La parte de `inference_baseline()` (ya tienes esos datos)
- La parte de `inference_mc_dropout()` (ya tienes esos datos)

---

### PASO 5: Ajustar Temperatura para decoder_variance

| Sección | ¿Ejecutar? | Tiempo |
|---------|------------|--------|
| "## 5. Ajustar Temperaturas" - SOLO decoder_variance | ✅ SÍ | 1 min |

**Busca**:
```python
for method_name in ['mc_dropout', 'decoder_variance']:  # ← Solo ejecuta decoder_variance
    # ...
```

---

### PASO 6: Inferencia en val_eval_final (SOLO decoder_variance)

| Sección | ¿Ejecutar? | Tiempo |
|---------|------------|--------|
| "## 6. Inferencia en val_eval_final..." | **✅ SÍ, pero SOLO decoder_variance** | 20-25 min |

**Busca esta parte del código**:
```python
for img_id in tqdm(img_ids_eval_final, desc="Evaluación final"):
    # ...
    
    # ========================================================================
    # Método 5: Decoder Variance  ← EJECUTAR ESTA PARTE
    # ========================================================================
    preds_dec = inference_decoder_variance(model, img_path, TEXT_PROMPT, ...)
    
    for pred in preds_dec:
        methods_results['decoder_variance'].append({...})
```

---

### PASO 7: Guardar Resultados

| Sección | ¿Ejecutar? | Tiempo |
|---------|------------|--------|
| Celda que guarda `eval_decoder_variance.json` | ✅ SÍ | 5 seg |

**Busca**:
```python
with open(OUTPUT_DIR / 'eval_decoder_variance.json', 'w') as f:
    json.dump(methods_results['decoder_variance'], f)
```

---

## 📝 RESUMEN: CELDAS EXACTAS A EJECUTAR

### MÉTODO SIMPLE (Si conoces los números de celda):

```
Ejecutar celdas: 1, 2, 3, [cargar modelo], [inference_decoder_variance], 
                  [val_calib - decoder_variance], [temperatura], 
                  [val_eval - decoder_variance], [guardar JSON]
```

### MÉTODO DETALLADO (Buscar por contenido):

1. **Celda 1-3**: Toda la configuración inicial
2. **Busca**: `from groundingdino.util.inference import load_model` → Ejecutar
3. **Busca**: `def inference_decoder_variance(` → **Ejecutar (CRÍTICO)**
4. **Busca**: `# Método 5: Decoder Variance` en sección `val_calib` → Ejecutar solo esa parte
5. **Busca**: Ajuste de temperatura para `decoder_variance` → Ejecutar
6. **Busca**: `# Método 5: Decoder Variance` en sección `val_eval` → Ejecutar solo esa parte
7. **Busca**: `eval_decoder_variance.json` (guardar) → Ejecutar

---

## ⚠️ LO QUE NO DEBES EJECUTAR

| ❌ NO Ejecutar | Razón |
|----------------|-------|
| `inference_baseline()` | Ya tienes esos datos de Fase 2 |
| `inference_mc_dropout()` | Ya tienes esos datos de Fase 3 (y es MUY lento) |
| Celdas de métricas y visualización | No son necesarias para RQ1 |
| Secciones 7-10 (si existen) | Solo necesitas hasta guardar el JSON |

---

## 🔍 CÓMO IDENTIFICAR LAS CELDAS

### En Jupyter Notebook:

1. **Abre**: `fase 5/main.ipynb`
2. **Usa Ctrl+F** para buscar:
   - `"def inference_decoder_variance"` → Ejecuta esa celda
   - `"Método 5: Decoder Variance"` → Ejecuta esas secciones
   - `"eval_decoder_variance.json"` → Ejecuta cuando guarda

3. **Scroll manual**:
   - Inicio → Ejecuta todo hasta cargar el modelo
   - Busca `inference_decoder_variance` → Ejecuta
   - Busca sección 4 (val_calib) → Ejecuta SOLO decoder_variance
   - Busca sección 6 (val_eval) → Ejecuta SOLO decoder_variance
   - Busca donde guarda JSON → Ejecuta

---

## ✅ VERIFICACIÓN POST-EJECUCIÓN

Después de ejecutar, verifica:

```python
import json

# Cargar el archivo generado
with open('outputs/comparison/eval_decoder_variance.json', 'r') as f:
    data = json.load(f)

# Verificar estructura
print("Total predicciones:", len(data))
print("Campos:", data[0].keys())

# CRÍTICO: Verificar layer_uncertainties
if 'layer_uncertainties' in data[0]:
    print("✅ layer_uncertainties EXISTE")
    print("Ejemplo:", data[0]['layer_uncertainties'])
else:
    print("❌ layer_uncertainties NO EXISTE - Algo salió mal")
```

---

## 💡 TIPS

### Si te pierdes:

1. **Busca comentarios**: Las secciones tienen títulos claros como "## 4. Inferencia en val_calib"
2. **Busca "decoder_variance"**: Cada vez que veas este término, es relevante para RQ1
3. **Lee los prints**: El código imprime mensajes como "Procesando decoder_variance..."

### Si sale error:

1. **Verifica GPU**: `torch.cuda.is_available()` debe ser `True`
2. **Verifica modelo**: Debe estar cargado en memoria
3. **Verifica paths**: Los directorios de datos deben existir

### Para ahorrar tiempo:

- **NO re-ejecutes** baseline ni MC-dropout
- **Solo** ejecuta las líneas que mencionan `decoder_variance`
- **Usa** los datos cacheados de Fase 2 y 3

---

## ⏰ TIEMPO ESTIMADO TOTAL

| Actividad | Tiempo |
|-----------|--------|
| Configuración + cargar modelo | 1 min |
| Definir `inference_decoder_variance()` | 1 seg |
| Inferencia en val_calib | 10-15 min |
| Ajustar temperatura | 1 min |
| Inferencia en val_eval | 20-25 min |
| Guardar JSON | 5 seg |
| **TOTAL** | **~35-40 min** |

---

## 📞 SIGUIENTE PASO

Una vez ejecutadas las celdas:

1. Ejecuta el script de verificación:
   ```bash
   cd RQ/rq1
   python verificar_datos_reales.py
   ```

2. Si ves `✅✅✅ DATOS REALES VERIFICADOS`:
   - Dime: "Datos verificados, actualiza RQ1"
   - Yo actualizaré el notebook RQ1

3. Si ves `❌ ERROR`:
   - Comparte el error completo
   - Te ayudaré a resolverlo

---

¿Listo para comenzar? Abre `fase 5/main.ipynb` y sigue esta guía paso a paso.
