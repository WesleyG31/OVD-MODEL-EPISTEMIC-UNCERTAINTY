# 🔍 DIAGNÓSTICO: Temperaturas Idénticas Entre Métodos

## Problema Identificado

Los archivos de calibración (`calib_baseline.csv`, `calib_mc_dropout.csv`, `calib_decoder_variance.csv`) son **idénticos**, lo que causa que las temperaturas calculadas sean las mismas (2.735) para los 3 métodos.

## Causa Raíz

**NO HAY OVERLAP** entre las predicciones cacheadas de MC-Dropout y las imágenes procesadas en val_calib:

- **MC-Dropout Parquet** (Fase 3): Tiene 100 imágenes (IDs: 136-9857)
- **val_calib (primeras 500)**: Imágenes con IDs diferentes (4-9961)
- **Overlap**: 0% (¡ninguna imagen en común!)

Esto significa que:
1. Aunque el código carga correctamente el Parquet con incertidumbre
2. **NINGUNA** de las 500 imágenes procesadas tiene predicciones MC-Dropout cacheadas
3. Todas las 500 imágenes deben calcular MC-Dropout desde cero (K=5 pases, muy costoso)
4. **HIPÓTESIS**: El código puede estar fallando silenciosamente y usando baseline en su lugar

## Verificación

He agregado código de diagnóstico al notebook que mostrará:
- Cuántas imágenes tienen overlap con el cache
- Cuántas predicciones vienen de caché vs cálculo desde cero
- Comparación de los CSVs generados (logits, scores, uncertainties)

## Soluciones Posibles

### Opción 1: Ejecutar Fase 3 Completa (RECOMENDADO)
Volver a ejecutar Fase 3 (MC-Dropout) en las **primeras 500 imágenes de val_calib** para generar el cache correcto.

**Ventajas**:
- Tendrás predicciones MC-Dropout correctas con incertidumbre real
- Las temperaturas serán diferentes para cada método
- Resultados más precisos

**Desventajas**:
- Toma ~1.5 horas ejecutar MC-Dropout en 500 imágenes (K=5 pases)

### Opción 2: Reducir Imágenes Procesadas
Procesar solo las **100 imágenes** que SÍ tienen MC-Dropout cacheado.

**Cómo**:
Cambiar en el notebook:
```python
# De:
for img_id in tqdm(img_ids_calib[:500]):

# A:
mc_cached_ids = set(mc_by_img.keys())
calib_subset = [img_id for img_id in img_ids_calib if img_id in mc_cached_ids][:100]
for img_id in tqdm(calib_subset):
```

**Ventajas**:
- Rápido, usa solo cache
- Las temperaturas serán diferentes

**Desventajas**:
- Solo 100 imágenes para calibración (menos robusto)

### Opción 3: Permitir Cálculo desde Cero (ACTUAL)
El notebook ya está configurado para calcular MC-Dropout cuando no hay cache.

**Lo que deberías verificar**:
1. Ejecutar el notebook con el nuevo código de diagnóstico
2. Verificar que los contadores muestren:
   - `MC-Dropout: 0 cacheadas, 500 calculadas`
3. Verificar que los CSVs sean diferentes

**Si los CSVs siguen siendo idénticos**, significa que `inference_mc_dropout` no se está ejecutando correctamente y hay un bug en el código.

## Próximos Pasos

1. **Ejecuta el notebook** con el código de diagnóstico actualizado
2. **Revisa los mensajes** en la sección 4 (val_calib):
   - ¿Cuántas imágenes de overlap?
   - ¿Cuántas predicciones cacheadas vs calculadas?
   - ¿Los CSVs son diferentes?

3. **Comparte los resultados** y decidimos:
   - Si los CSVs siguen idénticos → hay un bug, necesito verlo
   - Si los CSVs son diferentes → ¡perfecto! Las temperaturas serán diferentes

## Archivos de Diagnóstico Creados

- `diagnose_cache.py`: Verifica que los datos cacheados sean diferentes
- `check_overlap.py`: Verifica overlap entre cache y val_calib
- `count_images.py`: Cuenta imágenes en val_calib

## Cambios Aplicados al Notebook

✅ Agregado diagnóstico de overlap antes de procesar
✅ Contadores de predicciones cacheadas vs calculadas  
✅ Verificación de diferencias en CSVs generados
✅ Mensajes claros sobre qué está pasando

---

**Ejecuta el notebook y comparte la salida de la sección 4 para continuar.**
