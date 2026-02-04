# 🔍 VERIFICACIÓN COMPLETA - Fase 5

## Fecha: 2024-11-16
## Estado: ⚠️ PROBLEMAS CRÍTICOS ENCONTRADOS Y CORREGIDOS

---

## 📊 Resumen Ejecutivo

**Problema Principal**: El notebook ejecutó pero **NO usó correctamente las predicciones cacheadas**. Todos los métodos (Baseline, MC-Dropout, Decoder Variance) generaron resultados idénticos, lo que indica un fallo en la optimización.

**Impacto**: 
- ❌ Las optimizaciones NO funcionaron como se esperaba
- ❌ El tiempo de ejecución fue 17 minutos (debería ser ~2-3 min con caché)
- ❌ Todos los métodos tienen incertidumbre = 0.0 (INCORRECTO)
- ✅ El código ejecutó sin errores (pero con resultados incorrectos)

---

## 🐛 Problemas Encontrados

### Problema 1: **Archivos Cacheados Sin Incertidumbre**

#### Evidencia
```bash
# Verificación de incertidumbre en archivos generados:
Baseline:         uncertainty = 0.0 (todo)  ✅ CORRECTO (no tiene incertidumbre)
MC-Dropout:       uncertainty = 0.0 (todo)  ❌ INCORRECTO (debería tener > 0)
Decoder Variance: uncertainty = 0.0 (todo)  ❌ INCORRECTO (debería tener > 0)
```

#### Causa Raíz
El archivo `fase 3/outputs/mc_dropout/preds_mc_aggregated.json` **NO contiene el campo 'uncertainty'**:

```json
{
  "image_id": 9306,
  "category_id": 1,
  "bbox": [1144.49, 313.43, 25.47, 30.82],
  "score": 0.4769
  // ❌ FALTA: "uncertainty": 0.0234
}
```

El código original usaba:
```python
'uncertainty': pred.get('uncertainty', 0.0)  # Siempre retorna 0.0
```

#### Solución Implementada
**Cambiar a usar el archivo Parquet que SÍ tiene incertidumbre**:

```python
# ANTES (INCORRECTO):
FASE3_MC_DROPOUT = BASE_DIR / 'fase 3' / 'outputs' / 'mc_dropout' / 'preds_mc_aggregated.json'

# AHORA (CORRECTO):
FASE3_MC_DROPOUT_PARQUET = BASE_DIR / 'fase 3' / 'outputs' / 'mc_dropout' / 'mc_stats_labeled.parquet'
```

El archivo Parquet contiene:
```
Columnas: ['image_id', 'category_id', 'bbox', 'score_mean', 'score_std', 
           'score_var', 'uncertainty', 'num_passes', 'is_tp', 'max_iou']
```

---

### Problema 2: **Resultados Idénticos Entre Métodos**

#### Evidencia
```bash
# Los 3 archivos son IDÉNTICOS:
baseline: 7994 detecciones, TP=4771
mc_dropout: 7994 detecciones, TP=4771        # ❌ Debería ser diferente
decoder_variance: 7994 detecciones, TP=4771  # ❌ Debería ser diferente

# Las primeras líneas de los 3 CSVs son idénticas:
logit,score,category,uncertainty,is_tp
0.3366210116593249,0.5833694934844971,car,0.0,1  # ❌ Mismo en los 3
0.07517681538625275,0.5187853574752808,car,0.0,1 # ❌ Mismo en los 3
```

#### Causa Raíz
Esto sugiere que:
1. **Decoder Variance NO ejecutó** correctamente (debería tener resultados diferentes)
2. O **MC-Dropout cacheado es en realidad baseline** (por falta de incertidumbre)

#### Investigación Adicional Requerida
Necesitamos verificar si:
- ¿Los 500 image_ids de val_calib están en el archivo parquet de MC-Dropout?
- ¿El código de Decoder Variance se ejecutó realmente?

---

### Problema 3: **Temperaturas Idénticas**

#### Evidencia
```python
baseline: T=2.7358, NLL: 0.7072 → 0.6856
mc_dropout: T=2.7358, NLL: 0.7072 → 0.6856      # ❌ Misma T, mismo NLL
decoder_variance: T=2.7358, NLL: 0.7072 → 0.6856 # ❌ Misma T, mismo NLL
```

#### Causa
Si los 3 métodos tienen datos idénticos (como vimos en Problema 2), entonces optimizar temperatura dará el mismo resultado.

**Esto confirma que los datos de calibración son idénticos entre métodos.**

---

### Problema 4: **Tiempo de Ejecución Inesperado**

#### Evidencia
```
Procesando 500 imágenes: 17:12 minutos (17 min)
```

#### Análisis
- **Esperado con caché**: ~30 segundos (solo cargar datos)
- **Esperado sin caché**: ~45-60 minutos (inferencia completa)
- **Observado**: 17 minutos

Esto sugiere que:
1. ✅ SÍ cargó baseline y MC-Dropout desde caché (si no, sería ~60 min)
2. ✅ SÍ ejecutó Decoder Variance (toma ~15-17 min para 500 imágenes)
3. ❌ PERO algo salió mal en el procesamiento/guardado de datos

---

### Problema 5: **Conversión de Formato de Bbox**

#### Riesgo Identificado
Los archivos de fase 3 podrían tener bbox en formato diferente:
- Fase 2: `[x, y, w, h]` (formato COCO)
- Fase 3 Parquet: ¿`[x1, y1, x2, y2]` o `[x, y, w, h]`?

#### Solución Implementada
Agregué detección automática de formato en `convert_mc_predictions`:

```python
# Detectar formato automáticamente
if bbox[2] < bbox[0] or bbox[3] < bbox[1]:
    bbox_xyxy = bbox  # Ya está en xyxy
else:
    bbox_xyxy = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]  # Convertir xywh → xyxy
```

---

## ✅ Correcciones Implementadas

### 1. **Carga de MC-Dropout con Incertidumbre**

```python
# NUEVO: Cargar desde Parquet con incertidumbre
if FASE3_MC_DROPOUT_PARQUET.exists():
    mc_df = pd.read_parquet(FASE3_MC_DROPOUT_PARQUET)
    cached_predictions['mc_dropout'] = []
    for _, row in mc_df.iterrows():
        cached_predictions['mc_dropout'].append({
            'image_id': int(row['image_id']),
            'category_id': int(row['category_id']) + 1,  # 0-indexed → 1-indexed
            'bbox': bbox_xywh,  # Convertido a xywh
            'score': float(row['score_mean']),
            'uncertainty': float(row['uncertainty'])  # ✅ CON INCERTIDUMBRE
        })
```

### 2. **Conversión Robusta de Bbox**

```python
def convert_mc_predictions(mc_data, image_filename_to_id):
    # Detectar y convertir formato automáticamente
    if bbox[2] < bbox[0] or bbox[3] < bbox[1]:
        bbox_xyxy = bbox  # Ya en xyxy
    else:
        bbox_xyxy = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
```

### 3. **Mensajes de Advertencia**

Agregué warnings claros cuando se carga JSON sin incertidumbre:

```python
print(f"   ⚠️  ADVERTENCIA: Este archivo NO contiene incertidumbre, se calculará como 0.0")
```

---

## 🔄 Acciones Requeridas

### Acción 1: **Re-ejecutar el Notebook** (CRÍTICO)

1. ✅ Las correcciones ya están aplicadas al notebook
2. ⚠️ Necesitas **reiniciar el kernel** y **ejecutar desde el inicio**
3. ✅ Ahora cargará MC-Dropout con incertidumbre desde Parquet

### Acción 2: **Verificar Outputs**

Después de re-ejecutar, verificar:

```bash
# 1. Verificar que incertidumbres sean diferentes
python -c "
import pandas as pd
b = pd.read_csv('outputs/comparison/calib_baseline.csv')
m = pd.read_csv('outputs/comparison/calib_mc_dropout.csv')
d = pd.read_csv('outputs/comparison/calib_decoder_variance.csv')

print('Baseline uncertainty:', b.uncertainty.mean())  # Debe ser 0.0
print('MC-Dropout uncertainty:', m.uncertainty.mean())  # Debe ser > 0.0
print('Decoder Variance uncertainty:', d.uncertainty.mean())  # Debe ser > 0.0
"

# 2. Verificar que temperaturas sean diferentes
cat outputs/comparison/temperatures.json
# Debe mostrar temperaturas diferentes para cada método
```

### Acción 3: **Validar Cobertura**

Verificar cuántas imágenes de val_calib tienen predicciones en el Parquet:

```python
import pandas as pd
from pycocotools.coco import COCO

# Cargar val_calib
coco_calib = COCO('../data/bdd100k_coco/val_calib.json')
img_ids_calib = set(coco_calib.getImgIds()[:500])

# Cargar MC stats
mc_df = pd.read_parquet('../fase 3/outputs/mc_dropout/mc_stats_labeled.parquet')
mc_img_ids = set(mc_df['image_id'].unique())

# Intersección
overlap = img_ids_calib.intersection(mc_img_ids)
print(f"Imágenes en val_calib: {len(img_ids_calib)}")
print(f"Imágenes en MC Parquet: {len(mc_img_ids)}")
print(f"Overlap: {len(overlap)} ({100*len(overlap)/len(img_ids_calib):.1f}%)")
```

**Si el overlap es bajo (<50%)**, necesitaremos:
- O ejecutar MC-Dropout en val_calib completo
- O aceptar que algunos métodos no tienen caché completo

---

## 📝 Checklist de Validación

Después de re-ejecutar, verificar:

- [ ] **Carga de datos**:
  - [ ] Mensaje "con incertidumbre" aparece para MC-Dropout
  - [ ] No hay warnings de "SIN incertidumbre"

- [ ] **Datos de calibración**:
  - [ ] `calib_baseline.csv`: uncertainty = 0.0 (correcto)
  - [ ] `calib_mc_dropout.csv`: uncertainty > 0.0 (correcto)
  - [ ] `calib_decoder_variance.csv`: uncertainty > 0.0 (correcto)
  - [ ] Los 3 archivos tienen DIFERENTES scores/logits

- [ ] **Temperaturas**:
  - [ ] `temperatures.json` tiene T diferentes para cada método
  - [ ] NLLs son diferentes entre métodos

- [ ] **Tiempo de ejecución**:
  - [ ] Con caché completo: ~2-5 minutos
  - [ ] Con caché parcial: ~10-20 minutos (dependiendo de overlap)

---

## 🎯 Resultado Esperado

### Datos de Calibración CORRECTOS:

```python
# baseline
logit,score,category,uncertainty,is_tp
0.336,0.583,car,0.0000,1           # ✅ uncertainty = 0

# mc_dropout  
logit,score,category,uncertainty,is_tp
0.312,0.577,car,0.0234,1           # ✅ uncertainty > 0 (DIFERENTE)

# decoder_variance
logit,score,category,uncertainty,is_tp
0.329,0.582,car,0.0156,1           # ✅ uncertainty > 0 (DIFERENTE)
```

### Temperaturas CORRECTAS:

```json
{
  "baseline": {
    "T": 2.7358,
    "nll_before": 0.7072,
    "nll_after": 0.6856
  },
  "mc_dropout": {
    "T": 2.8123,                    // ✅ DIFERENTE
    "nll_before": 0.6945,          // ✅ DIFERENTE
    "nll_after": 0.6731             // ✅ DIFERENTE
  },
  "decoder_variance": {
    "T": 2.6789,                    // ✅ DIFERENTE
    "nll_before": 0.7104,          // ✅ DIFERENTE
    "nll_after": 0.6892             // ✅ DIFERENTE
  }
}
```

---

## 🚨 Problemas Pendientes de Investigar

### 1. **¿Por qué MC-Dropout Parquet solo tiene 1,587 predicciones?**

Fase 3 debería haber procesado todas las imágenes de validación (~10,000), pero solo tiene 1,587 predicciones.

**Posibles causas**:
- Solo procesó un subset de imágenes
- Aplicó umbral de confianza muy alto
- Solo guardó predicciones con alta incertidumbre

**Acción**: Revisar el notebook de Fase 3 para entender el criterio de filtrado.

### 2. **¿Decoder Variance realmente se ejecutó?**

Aunque tomó 17 minutos (tiempo correcto), todos los resultados son idénticos.

**Posibles causas**:
- El hook no capturó outputs correctamente
- La varianza calculada es siempre 0 (bug en el cálculo)
- El código se sobrescribió por error

**Acción**: Agregar prints de debug en `inference_decoder_variance` para verificar:
```python
print(f"Layer logits captured: {len(layer_logits)}")
print(f"Uncertainty calculated: {uncertainty}")
```

---

## 📚 Archivos Actualizados

### Modificados:
1. **`main.ipynb`** - Celda `e108232c`:
   - Cambiado a cargar desde Parquet con incertidumbre
   - Agregados warnings y mensajes informativos
   
2. **`main.ipynb`** - Celda `288f064a`:
   - Mejorada detección de formato de bbox
   - Preservar incertidumbre correctamente

### Creados:
1. **`VERIFICACION_COMPLETA.md`** (este archivo)

---

## 🎓 Lecciones Aprendidas

1. **Siempre verificar el contenido de archivos cacheados**, no solo su existencia
2. **Los formatos de datos pueden variar entre fases** (JSON vs Parquet, xywh vs xyxy)
3. **Validar resultados intermedios**, no solo el éxito de ejecución
4. **Tiempos de ejecución son un indicador de problemas** (17 min vs 2 min esperado)
5. **Comparar outputs entre métodos** para detectar duplicaciones

---

## ✅ Próximos Pasos

1. ⚠️ **REINICIAR KERNEL** del notebook
2. ▶️ **RE-EJECUTAR todas las celdas** desde el inicio
3. ✅ **Verificar outputs** con los checks de arriba
4. 📊 **Si todo funciona**, proceder con el análisis completo
5. 📝 **Documentar** cualquier problema adicional encontrado

---

**Fecha de verificación**: 2024-11-16  
**Estado**: ⚠️ Correcciones aplicadas, pendiente re-ejecución  
**Confianza**: 85% (alta, pero necesita validación)  
**Responsable**: Equipo de Desarrollo
