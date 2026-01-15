# 🚀 Guía de Ejecución - RQ7 Notebook

## ⚠️ IMPORTANTE: Celdas que DEBEN Ejecutarse

Para obtener los resultados completos de RQ7, debes ejecutar las siguientes celdas marcadas con **"EJECUTAR PARA RQ7"**:

### Fase 1: Preparación (Obligatoria)
✅ **Celda 2**: Imports y verificación de archivos
✅ **Celda 3**: Cargar métricas de calibración de Fase 5

### Fase 2: Medición de Latencia (⚠️ CRÍTICA - EJECUTAR PARA RQ7)
🔴 **Celda 5**: Cargar modelo GroundingDINO
   - Tiempo: ~30 segundos
   - Requiere: GPU recomendada (CPU funciona pero más lento)

🔴 **Celda 6**: Cargar imágenes de validación
   - Tiempo: ~10 segundos
   - Selecciona 50 imágenes aleatorias para benchmark

🔴 **Celda 7**: Definir funciones de medición de latencia
   - Instantáneo (solo definiciones)

🔴 **Celda 8**: Ejecutar benchmarks de latencia
   - ⏱️ Tiempo: 15-20 minutos (GPU) / 45-60 minutos (CPU)
   - ⚠️ ESTA ES LA CELDA MÁS IMPORTANTE
   - Mide latencia real de:
     - Baseline (50 imágenes)
     - MC-Dropout K=5 (50 imágenes × 5 pases = 250 inferencias)
     - Variance (50 imágenes)
   - Guarda resultados en `latency_raw.json`

### Fase 3: Análisis y Visualización (Automática)
✅ **Celda 9**: Calcular métricas de runtime
✅ **Celda 10**: Generar Tabla 7.1
✅ **Celda 11**: Generar Tabla 7.2
✅ **Celda 12**: Generar Figura 7.1
✅ **Celda 13**: Generar Figura 7.2
✅ **Celda 14**: Generar resumen ejecutivo
✅ **Celda 15**: Verificar archivos generados

---

## 📋 Checklist de Ejecución

### Antes de Empezar
- [ ] Verificar que Fase 5 está completa
- [ ] Confirmar que existe `../../fase 5/outputs/comparison/calibration_metrics.json`
- [ ] Tener GPU disponible (opcional pero recomendado)
- [ ] ~20 GB RAM disponible
- [ ] ~30 minutos de tiempo disponible

### Ejecución Paso a Paso

#### 1. Preparación (2 minutos)
```
Celda 1 → Celda 2 → Celda 3
```
**Salida esperada**: 
- "✅ Todos los archivos necesarios están disponibles"
- Tabla con métricas ECE de Fase 5

#### 2. Cargar Modelo (1 minuto)
```
Celda 4 → Celda 5
```
**Salida esperada**:
- "✅ Modelo cargado en cuda/cpu"
- "✅ Módulos dropout encontrados: X"

#### 3. Preparar Imágenes (30 segundos)
```
Celda 6
```
**Salida esperada**:
- "✅ Cargadas 50 imágenes de validación para benchmark"

#### 4. Definir Funciones (instantáneo)
```
Celda 7
```
**Salida esperada**:
- "✅ Funciones de medición de latencia definidas"

#### 5. ⚠️ EJECUTAR BENCHMARKS (15-60 minutos)
```
Celda 8 ⚠️ ESTA ES LA CELDA CRÍTICA
```
**Salida esperada**:
- Progress bars para cada método:
  - "Baseline latency: 100%|██████████| 50/50"
  - "MC-Dropout K=5 latency: 100%|██████████| 50/50"
  - "Variance latency: 100%|██████████| 50/50"
- Tiempos medios por método
- "✅ Resultados guardados en outputs/latency_raw.json"

**⏱️ Tiempos estimados**:
- GPU (RTX 3090/4090): ~15 minutos
- GPU (GTX 1080): ~25 minutos
- CPU: ~45-60 minutos

**⚠️ Si falla esta celda**:
- Reducir `n_samples` de 50 a 20 en la celda 2
- Verificar memoria GPU disponible
- Cerrar otros procesos que usen GPU

#### 6. Generar Resultados (5 minutos)
```
Celda 9 → Celda 10 → Celda 11 → Celda 12 → Celda 13 → Celda 14 → Celda 15
```
**Salida esperada**:
- Tabla 7.1 mostrada y guardada
- Tabla 7.2 mostrada y guardada
- Figura 7.1 mostrada y guardada
- Figura 7.2 mostrada y guardada
- Resumen ejecutivo impreso
- "✅ TODOS LOS ARCHIVOS GENERADOS EXITOSAMENTE"

---

## 🔍 Verificación de Resultados

### Archivos que DEBEN Existir

Después de ejecutar todas las celdas, verifica:

```powershell
ls outputs/
```

Debes tener **18 archivos**:
```
✅ config.yaml
✅ latency_raw.json              # ⚠️ CRÍTICO - Prueba de ejecución real
✅ runtime_metrics.json
✅ table_7_1_runtime_analysis.csv
✅ table_7_1_runtime_analysis.tex
✅ table_7_1_runtime_analysis.png
✅ table_7_1_runtime_analysis.pdf
✅ table_7_2_adas_feasibility.csv
✅ table_7_2_adas_feasibility.tex
✅ table_7_2_adas_feasibility.png
✅ table_7_2_adas_feasibility.pdf
✅ figure_7_1_reliability_vs_latency.png
✅ figure_7_1_reliability_vs_latency.pdf
✅ figure_7_1_data.json
✅ figure_7_2_reliability_per_ms.png
✅ figure_7_2_reliability_per_ms.pdf
✅ figure_7_2_data.json
✅ summary_rq7.json
```

### Valores Esperados (Aproximados)

**latency_raw.json**:
```json
{
  "baseline": [0.038, 0.039, ...],      # ~50 valores, media ~38ms
  "mc_dropout": [0.082, 0.084, ...],    # ~50 valores, media ~83ms
  "decoder_variance": [0.043, 0.041, ...] # ~50 valores, media ~43ms
}
```

**runtime_metrics.json**:
```json
{
  "mc_dropout": {
    "fps": 12.0,
    "ece": 0.082,
    "reliability_score": 0.918
  },
  "fusion": {
    "fps": 23.0,
    "ece": 0.061,
    "reliability_score": 0.939
  }
}
```

---

## 🐛 Troubleshooting

### Error: "CUDA out of memory"
**Solución**:
```python
# En celda 2, cambiar:
CONFIG = {
    'n_samples': 20,  # ← Reducir de 50 a 20
    'warmup': 3       # ← Reducir de 5 a 3
}
```

### Error: "Model not found"
**Solución**:
```python
# Verificar rutas en celda 5
model_config = '/opt/program/GroundingDINO/...'  # ← Ajustar ruta
model_weights = '/opt/program/GroundingDINO/...' # ← Ajustar ruta
```

### Error: "calibration_metrics.json not found"
**Solución**:
```bash
# Ejecutar primero la Fase 5 completa
cd "../../fase 5"
jupyter nbconvert --execute main.ipynb
```

### Latencia muy alta (>200ms)
**Posible causa**:
- Ejecutando en CPU en lugar de GPU
- GPU ocupada por otro proceso
- Imágenes muy grandes

**Solución**:
```python
# Verificar device
print(torch.cuda.is_available())  # Debe ser True
print(CONFIG['device'])           # Debe ser 'cuda'
```

---

## ⏱️ Timeline Completo

| Fase | Tiempo GPU | Tiempo CPU | Crítico |
|------|-----------|-----------|---------|
| 1. Setup | 2 min | 2 min | No |
| 2. Cargar modelo | 1 min | 1 min | No |
| 3. Cargar imágenes | 0.5 min | 0.5 min | No |
| 4. Funciones | 0 min | 0 min | No |
| 5. **Benchmarks** | **15 min** | **45 min** | **SÍ** |
| 6. Análisis | 2 min | 2 min | No |
| 7. Tablas | 1 min | 1 min | No |
| 8. Figuras | 2 min | 2 min | No |
| **TOTAL** | **23.5 min** | **53.5 min** | - |

---

## 📊 Interpretación de Resultados

### Tabla 7.1 - Runtime Analysis
- **FPS ↑**: Mayor es mejor (capacidad de procesamiento)
- **ECE ↓**: Menor es mejor (mejor calibración)
- **Conclusión**: Fusion tiene mejor ECE con FPS aceptable

### Tabla 7.2 - ADAS Feasibility
- **Real-Time Ready**: ✔ si FPS ≥ 20
- **Reliability Score**: 1 - ECE (mayor es mejor)
- **Conclusión**: Solo Fusion es viable para ADAS

### Figura 7.1 - Reliability vs Latency
- **Eje X**: Latencia (ms) → Menor es mejor
- **Eje Y**: Reliability → Mayor es mejor
- **Zona verde**: Región de tiempo real (<50ms)
- **Conclusión**: Fusion está en zona óptima

### Figura 7.2 - Reliability per Millisecond
- **Métrica**: Efficiency = Reliability / Latency
- **Mayor es mejor**: Más confiabilidad por unidad de tiempo
- **Conclusión**: Fusion es el más eficiente

---

## ✅ Checklist Final

Antes de considerar RQ7 completo, verifica:

- [ ] Ejecutaste la celda 8 (benchmarks) completamente
- [ ] `latency_raw.json` existe y contiene datos reales
- [ ] Las 4 figuras se generaron (2 tablas + 2 gráficos)
- [ ] Todos los archivos están en `outputs/`
- [ ] Los valores de FPS son realistas (~12-26 FPS)
- [ ] ECE de Fusion < ECE de MC-Dropout
- [ ] `summary_rq7.json` muestra conclusiones correctas

---

## 📞 Soporte

Si después de seguir esta guía sigues teniendo problemas:

1. Verifica que todas las fases anteriores (2-5) estén completas
2. Revisa los logs de error completos
3. Confirma versiones de librerías compatibles
4. Intenta con `n_samples=10` para testing rápido

**Tiempo mínimo requerido**: 20-25 minutos con GPU
**No hay atajos**: La celda 8 DEBE ejecutarse para tener datos reales
