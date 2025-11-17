# 🚀 GUÍA DE EJECUCIÓN - Opción 2 (IDEAL)

## ✅ Cambios Implementados

He modificado **Fase 5** para:
1. ✅ Usar **val_eval** en lugar de val_calib
2. ✅ Split inteligente: 500 para calibración, 1500 para evaluación
3. ✅ Reutilizar cache de Fase 2 y Fase 3
4. ✅ Diagnósticos detallados de cache usage

## 📋 INSTRUCCIONES EXACTAS

### PASO 1: Re-ejecutar Fase 3 (MC-Dropout Completo)

**📂 Archivo**: `fase 3/main.ipynb`

**🎯 Objetivo**: Generar predicciones MC-Dropout para las 2,000 imágenes de val_eval

**⏱️ Tiempo estimado**: 2-3 horas

**📝 Qué hacer**:

1. Abre el notebook: `fase 3/main.ipynb`

2. **VERIFICA** que procese **2,000 imágenes** de val_eval:
   - Busca la celda que dice algo como:
     ```python
     img_ids_eval = coco_eval.getImgIds()
     # Procesar solo primeras 100 para prueba
     img_ids_eval = img_ids_eval[:100]  # ❌ CAMBIAR ESTO
     ```
   
   - **CÁMBIALO** a:
     ```python
     img_ids_eval = coco_eval.getImgIds()
     # Procesar las 2000 imágenes completas
     print(f"Procesando {len(img_ids_eval)} imágenes de val_eval")
     ```

3. **EJECUTA** todo el notebook de arriba a abajo:
   - `Run All` o `Ctrl+Shift+Enter` en cada celda
   - Espera a que termine (~2-3 horas)

4. **VERIFICA** que se genere:
   - `fase 3/outputs/mc_dropout/mc_stats_labeled.parquet` con **~40,000 predicciones** (20 pred/img × 2000 imgs)
   - El archivo debe tener la columna `uncertainty` con valores > 0

---

### PASO 2: Re-ejecutar Fase 5 (Comparación Completa)

**📂 Archivo**: `fase 5/main.ipynb` (YA MODIFICADO)

**🎯 Objetivo**: Comparar los 6 métodos con cache completo

**⏱️ Tiempo estimado**: 15-30 minutos

**📝 Qué hacer**:

1. Abre el notebook: `fase 5/main.ipynb`

2. **EJECUTA** todo el notebook de arriba a abajo:
   - `Run All` o ejecuta celda por celda
   - **NO NECESITAS MODIFICAR NADA**, ya está corregido

3. **OBSERVA** la salida de la Celda 8 (Sección 4 - Calibración):
   ```
   🔍 OVERLAP CON CALIBRACIÓN (primeras 500 de val_eval):
      Baseline cacheado: XXX/500 imágenes (XX.X%)
      MC-Dropout cacheado: 500/500 imágenes (100.0%)  ← Debería ser 100%
   
   📊 ESTADÍSTICAS DE PROCESAMIENTO:
      Baseline: 500 cacheadas, 0 calculadas
      MC-Dropout: 500 cacheadas, 0 calculadas  ← Todo desde cache
   ```

4. **VERIFICA** que las temperaturas sean DIFERENTES:
   ```
   temperatures.json:
   {
     "baseline": {"T": 2.XXXX},      ← Diferente
     "mc_dropout": {"T": 3.YYYY},    ← Diferente
     "decoder_variance": {"T": 2.ZZZZ}  ← Diferente
   }
   ```

---

## 🔍 Verificación de Éxito

### Después de Fase 3:
```bash
cd "fase 3/outputs/mc_dropout"
python -c "import pandas as pd; df = pd.read_parquet('mc_stats_labeled.parquet'); print(f'Total predicciones: {len(df)}'); print(f'Imágenes únicas: {df[\"image_id\"].nunique()}'); print(f'Uncertainty media: {df[\"uncertainty\"].mean():.6f}')"
```

**Salida esperada**:
```
Total predicciones: ~40000
Imágenes únicas: 2000
Uncertainty media: 0.000086  (o similar, > 0)
```

### Después de Fase 5:
```bash
cd "fase 5/outputs/comparison"
python -c "import json; temps = json.load(open('temperatures.json')); print('Temperaturas:'); [print(f'  {k}: {v[\"T\"]:.4f}') for k, v in temps.items()]; print('\nSon diferentes?', len(set([v['T'] for v in temps.values()])) == 3)"
```

**Salida esperada**:
```
Temperaturas:
  baseline: 2.XXXX
  mc_dropout: 3.YYYY
  decoder_variance: 2.ZZZZ

Son diferentes? True  ← IMPORTANTE
```

---

## ⚠️ IMPORTANTE: Qué NO hacer

❌ **NO ejecutes Fase 2** (Baseline) - Ya está correcto
❌ **NO ejecutes Fase 4** (Temperature Scaling) - Ya está correcto
❌ **NO modifiques Fase 5** - Ya está corregido

---

## 🎯 Resumen de Ejecución

```
1. Modificar Fase 3 para procesar 2000 imágenes
   └─ Ejecutar Fase 3 completa (~2-3 horas)
   
2. Ejecutar Fase 5 (ya modificada)
   └─ Ejecutar Fase 5 completa (~15-30 minutos)
   
3. Verificar resultados
   └─ Temperaturas diferentes ✅
```

---

## 📊 Resultados Esperados

Después de completar ambos pasos:

✅ **Fase 3**: 
- mc_stats_labeled.parquet con 2,000 imágenes
- Uncertainty > 0 para predicciones MC-Dropout

✅ **Fase 5**:
- Temperaturas DIFERENTES para cada método
- Cache usage: 100% para calibración, ~100% para evaluación
- CSVs con datos diferentes entre métodos
- Métricas de detección, calibración y uncertainty

---

## 💡 Consejos

1. **Ejecuta en horario nocturno**: Fase 3 toma 2-3 horas
2. **Monitorea el progreso**: Verifica que no haya errores en Fase 3
3. **Guarda los outputs**: Importante para reproducibilidad
4. **Verifica cada paso**: Usa los scripts de verificación

---

## 🆘 Si algo falla

**Si Fase 3 falla a mitad de camino**:
- No pierdas el progreso, puede continuar desde donde quedó
- Verifica espacio en disco
- Verifica memoria GPU

**Si Fase 5 sigue dando temperaturas iguales**:
- Verifica que Fase 3 haya terminado correctamente
- Ejecuta los scripts de verificación
- Revisa que mc_stats_labeled.parquet tenga 2000 imágenes

---

¿Listo para comenzar? 

**Empieza con Paso 1** (Fase 3) y cuando termine, continúa con **Paso 2** (Fase 5).
