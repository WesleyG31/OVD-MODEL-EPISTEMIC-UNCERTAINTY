# INSTRUCCIONES: Opción 2 - Correr Fase 3 con todos los datos

## Resumen
La Fase 3 actualmente **solo procesa 100 imágenes** de val_eval. Esto causa que el cache de MC-Dropout sea insuficiente para la Fase 5. La solución es correr la Fase 3 con **todas las 2,000 imágenes** de val_eval.

---

## 🔧 Cambio Realizado

### Archivo modificado: `fase 3/main.ipynb`

**Línea modificada:**
```python
# ANTES (línea ~1336):
for img_id in tqdm(image_ids[:100], desc="Procesando imágenes"):

# DESPUÉS:
for img_id in tqdm(image_ids, desc="Procesando imágenes"):
```

**Cambio adicional:**
Se eliminó el mensaje de advertencia:
```python
# ELIMINADO:
print(f"⚠️  Procesando primeras 100 imágenes para prueba rápida\n")
```

---

## 📋 Pasos para Ejecutar

### Paso 1: Abrir Fase 3
```
1. Abrir el archivo: fase 3/main.ipynb
2. Verificar que tiene la modificación (sin [:100])
```

### Paso 2: Ejecutar todas las celdas
```
1. En VS Code: Menú "Run" > "Run All"
2. O usar Ctrl+Shift+Enter repetidamente
3. O en Jupyter: "Cell" > "Run All"
```

### Paso 3: Tiempo estimado
```
- Con 100 imágenes: ~15-20 minutos
- Con 2,000 imágenes: ~6-7 horas (estimado)
```

### Paso 4: Verificar resultados
Al finalizar, debe generar:
```
fase 3/outputs/mc_dropout/
├── mc_stats_labeled.parquet     ← Debe tener ~2,000 imágenes (no solo 100)
├── preds_mc_aggregated.json     ← Predicciones para todas las imágenes
├── metrics.json                 ← Métricas de detección
├── timing_data.parquet          ← Tiempos de inferencia
└── ...
```

**Verificación del cache:**
```powershell
# En terminal PowerShell:
cd "fase 3/outputs/mc_dropout"
python -c "import pandas as pd; df = pd.read_parquet('mc_stats_labeled.parquet'); print(f'Imágenes en cache: {df.image_id.nunique()}')"
```

**Resultado esperado:**
```
Imágenes en cache: 2000  ← ¡Debe ser 2000, no 100!
```

---

## 🚀 Paso 5: Correr Fase 5

Una vez que Fase 3 termine (y genere el cache completo), puedes correr Fase 5:

```
1. Abrir: fase 5/main.ipynb
2. Ejecutar todas las celdas ("Run All")
3. Tiempo estimado: ~30-45 minutos
```

### Verificar temperaturas diferentes
Al final de Fase 5, verificar:
```powershell
cat "outputs/comparison/temperatures.json"
```

**Resultado esperado (temperaturas DIFERENTES):**
```json
{
  "baseline": 1.53,
  "mc_dropout": 1.67,       ← ¡Debe ser diferente!
  "decoder_variance": 1.42  ← ¡Debe ser diferente!
}
```

**Si sale IGUAL (error):**
```json
{
  "baseline": 1.53,
  "mc_dropout": 1.53,       ← ¡ERROR! igual que baseline
  "decoder_variance": 1.53  ← ¡ERROR! igual que baseline
}
```

---

## 📊 Salidas esperadas de Fase 5

Después de correr Fase 5, deberías tener:

```
outputs/comparison/
├── temperatures.json               ← Temperaturas (diferentes por método)
├── calib_baseline.csv              ← Calibración baseline (1,500 imgs)
├── calib_mc_dropout.csv            ← Calibración MC-Dropout (1,500 imgs)
├── calib_decoder_variance.csv      ← Calibración decoder (1,500 imgs)
├── eval_baseline.csv               ← Evaluación baseline (500 imgs)
├── eval_mc_dropout.csv             ← Evaluación MC-Dropout (500 imgs)
├── eval_decoder_variance.csv       ← Evaluación decoder (500 imgs)
├── final_report.txt                ← Reporte final
└── ...
```

---

## ⚠️ Notas Importantes

### Si Fase 3 falla o se interrumpe:
1. **NO reiniciar desde cero**: El código guarda resultados parciales
2. **Checkpoint manual**: Si se interrumpe, puedes modificar el notebook para:
   ```python
   # Procesar desde la imagen N en adelante
   for img_id in tqdm(image_ids[N:], desc="Procesando imágenes"):
   ```

### Si no quieres esperar 6-7 horas:
**Opción alternativa**: Usar un subset más grande (ej: 500 imágenes)
```python
# En fase 3/main.ipynb, línea ~1336:
for img_id in tqdm(image_ids[:500], desc="Procesando imágenes"):
```

Esto te dará mejor cobertura que 100 imágenes, pero terminará más rápido que 2,000.

### Recursos computacionales:
- **GPU necesaria**: Sí (CUDA debe estar disponible)
- **Memoria GPU**: ~6-8 GB recomendados
- **RAM**: ~16 GB recomendados
- **Almacenamiento**: ~2-3 GB para los outputs

---

## 🎯 Resumen de la solución

| Estado | Descripción |
|--------|-------------|
| ✅ **Fase 2** | Ya completa (baseline cache: 1,988 imágenes) |
| ⏳ **Fase 3** | **Debes correr** (generar cache para 2,000 imágenes) |
| ✅ **Fase 4** | Ya completa (temperatura scaling) |
| ⏳ **Fase 5** | **Correr después** de Fase 3 |

---

## 📞 Validación Final

Después de correr todo, puedes usar los scripts de diagnóstico:

```powershell
# 1. Verificar cobertura de cache
python diagnose_cache.py

# 2. Verificar overlap entre splits
python check_overlap.py

# 3. Contar imágenes por split
python count_images.py

# 4. Análisis completo
python analyze_splits.py
```

---

## 📝 Conclusión

**Lo que debes hacer:**
1. ✅ Verificar que el notebook de Fase 3 tiene el cambio (sin `[:100]`)
2. 🚀 Correr Fase 3 completo (esperar ~6-7 horas)
3. ✅ Verificar que el cache tiene 2,000 imágenes
4. 🚀 Correr Fase 5
5. ✅ Verificar que las temperaturas son diferentes

**Si todo sale bien:**
- Temperaturas diferentes ✅
- Cache completo ✅
- Resultados de calibración correctos ✅

---

¡Éxito! 🎉
