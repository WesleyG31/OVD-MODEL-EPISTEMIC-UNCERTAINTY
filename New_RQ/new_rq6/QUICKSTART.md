# RQ6 - Quick Start Guide

## Inicio Rápido (5 pasos)

### 1. Verificar Prerrequisitos
```bash
# Verificar que el modelo está instalado
ls /opt/program/GroundingDINO/weights/groundingdino_swint_ogc.pth

# Verificar que el dataset está disponible
ls ../data/bdd100k_coco/val_eval.json

# Verificar que estamos en el directorio correcto
pwd  # Debe terminar en .../New_RQ/new_rq6/
```

### 2. Abrir el Notebook
```bash
# Desde VSCode o Jupyter
jupyter notebook rq6.ipynb
```

### 3. Ejecutar Celdas Clave

#### ⚡ Celda 1: Configuración (ejecutar siempre)
```python
# Configuración e imports
# Tiempo: ~10 segundos
```

#### ⚡ Celda 2: Cargar Modelo (ejecutar siempre)
```python
# ✅ EJECUTAR PARA RQ6 - Cargar modelo GroundingDINO
# Tiempo: ~30 segundos
```

#### ⚡ Celda 5: Inferencia (LENTA - ejecutar una vez)
```python
# ✅ EJECUTAR PARA RQ6 - Procesar dataset
# Tiempo: ~15-20 minutos con GPU
# Genera: decoder_dynamics.parquet
```

#### ⚡ Celdas 6-14: Análisis y Visualización (ejecutar después de celda 5)
```python
# Tiempo: ~2 minutos
# Genera: Todas las figuras y tablas
```

### 4. Verificar Outputs
```bash
ls output/
# Debe mostrar 14 archivos
```

### 5. Ver Resultados
```bash
# Ver resumen
cat output/summary_rq6.json

# Ver tablas
cat output/Table_RQ6_1.csv
cat output/Table_RQ6_2.csv

# Ver figuras
open output/Fig_RQ6_1_decoder_variance.png
open output/Fig_RQ6_2_auroc_by_layer.png
```

## Comandos de Ejecución

### Opción A: Ejecución Completa (Primera Vez)
```python
# En Jupyter/VSCode:
# 1. Ejecutar todas las celdas: Ctrl+Shift+Enter repetidas veces
# 2. O usar "Run All Cells"
# Tiempo total: ~20-25 minutos
```

### Opción B: Re-análisis (Si ya tienes decoder_dynamics.parquet)
```python
# En Jupyter/VSCode:
# 1. Ejecutar celdas 1-4 (configuración y carga de modelo)
# 2. SALTAR celda 5 (inferencia - ya está hecha)
# 3. Ejecutar celdas 6-14 (análisis y visualización)
# Tiempo total: ~3 minutos
```

### Opción C: Solo Verificar Resultados
```bash
# Si ya ejecutaste todo antes:
python -c "
import json
with open('output/summary_rq6.json') as f:
    s = json.load(f)
    print('Total detecciones:', s['dataset']['total_detections'])
    print('AUROC mejora:', s['key_findings']['auroc_improvement']['total_improvement'])
"
```

## Configuración Rápida

### Reducir Tiempo de Ejecución (Para Pruebas)
En **Celda 1**, modificar:
```python
CONFIG = {
    # ...otras configuraciones...
    'sample_size': 50  # Cambiar de 500 a 50
}
```
Tiempo de inferencia: ~2 minutos en lugar de 15-20 minutos

### Aumentar Precisión (Para Resultados Finales)
En **Celda 1**, modificar:
```python
CONFIG = {
    # ...otras configuraciones...
    'sample_size': 2000  # Usar todo val_eval
}
```
Tiempo de inferencia: ~50-60 minutos

## Checklist de Ejecución

### Antes de Empezar
- [ ] GroundingDINO instalado
- [ ] Dataset BDD100K disponible
- [ ] GPU disponible (recomendado)
- [ ] Espacio en disco: ~500MB para outputs

### Durante la Ejecución
- [ ] Celda 1 ejecutada sin errores
- [ ] Celda 2 cargó el modelo correctamente
- [ ] Celda 5 procesó las imágenes (ver barra de progreso)
- [ ] No hay errores en las celdas de análisis

### Después de la Ejecución
- [ ] Directorio `output/` existe
- [ ] 14 archivos generados
- [ ] Figuras PNG y PDF visibles
- [ ] Tablas CSV y LaTeX legibles
- [ ] summary_rq6.json con resultados

## Troubleshooting Rápido

### ❌ "CUDA out of memory"
```python
# Solución: Reducir sample_size
CONFIG['sample_size'] = 50  # En lugar de 500
```

### ❌ "Model not found"
```bash
# Solución: Verificar paths
ls /opt/program/GroundingDINO/weights/
# Si no existe, revisar instalación de GroundingDINO
```

### ❌ "Dataset not found"
```bash
# Solución: Verificar path relativo
ls ../../data/bdd100k_coco/val_eval.json
# Ajustar BASE_DIR si es necesario
```

### ❌ "No module named 'groundingdino'"
```bash
# Solución: Agregar al PYTHONPATH
export PYTHONPATH="/opt/program/GroundingDINO:$PYTHONPATH"
```

### ❌ Celda 5 toma demasiado tiempo
```python
# Verificar:
print(f"Device: {CONFIG['device']}")  # Debe ser 'cuda'
print(f"GPU disponible: {torch.cuda.is_available()}")
# Si es 'cpu', la ejecución será MUY lenta
```

## Outputs Esperados

### Terminal (durante ejecución)
```
✓ Configuración cargada
  Device: cuda
  Output: ./output
  Categorías: 10
✓ Config guardada en ./output/config.yaml
✓ Modelo cargado en cuda
✓ Prompt: person. rider. car. ...
✓ Capas del decoder encontradas: 6
✓ Funciones auxiliares definidas
✓ Función de inferencia con captura de capas definida

Procesando 500 imágenes...
100%|████████████████████| 500/500 [15:23<00:00,  1.85s/it]

✓ Procesamiento completado: 8234 detecciones

Resumen de resultados:
  Total detecciones: 8234
  True Positives (TP): 6891
  False Positives (FP): 1343
  Capas capturadas por detección: 6.0

✓ Resultados guardados en ./output/decoder_dynamics.parquet
...
```

### Figuras Generadas
```
output/
├── Fig_RQ6_1_decoder_variance.png   [Varianza TP vs FP]
├── Fig_RQ6_1_decoder_variance.pdf   
├── Fig_RQ6_2_auroc_by_layer.png     [AUROC por capa]
└── Fig_RQ6_2_auroc_by_layer.pdf     
```

### Tablas Generadas
```
output/
├── Table_RQ6_1.csv    [Layer-wise diagnostics]
├── Table_RQ6_1.tex    
├── Table_RQ6_2.csv    [Failure conditions]
└── Table_RQ6_2.tex    
```

## Interpretación de Resultados

### Figure RQ6.1 - ¿Qué Buscar?
- ✅ **Línea verde (TP) más baja que roja (FP)** → TP se estabilizan antes
- ✅ **Separación aumenta hacia la derecha** → Mejora con profundidad
- ❌ Si las líneas se cruzan o están muy juntas → Problema

### Figure RQ6.2 - ¿Qué Buscar?
- ✅ **AUROC aumenta de izquierda a derecha** → Capas tardías mejores
- ✅ **AUROC final > 0.70** → Buena discriminación
- ❌ Si AUROC es plano o decrece → Problema

### Table RQ6.1 - ¿Qué Buscar?
- ✅ **AUROC ↑ con la capa** → Mejora progresiva
- ✅ **Var(TP) ↓ con la capa** → TP se estabilizan
- ✅ **Var(FP) relativamente alta** → FP más inciertos

### Table RQ6.2 - ¿Qué Buscar?
- ✅ **AUROC drops negativos** → Condiciones problemáticas
- ✅ **Interpretaciones coherentes** → Explicación de fallas

## Validación de Hipótesis

El notebook valida automáticamente tres hipótesis. En el output final verás:

```
3. Validación de hipótesis:
   - H1 (TP estabilizan antes que FP): ✓ CONFIRMADA
   - H2 (Capas tardías mejor AUROC): ✓ CONFIRMADA
   - H3 (Separación aumenta con profundidad): ✓ CONFIRMADA
```

Si ves ✗ en alguna, revisar:
1. ¿Suficientes datos? (aumentar sample_size)
2. ¿Modelo funciona bien? (verificar mAP en fase 2)
3. ¿Capas correctamente capturadas? (verificar hooks)

## Siguiente Paso

Una vez ejecutado exitosamente:

1. **Revisar figuras**: ¿Muestran el patrón esperado?
2. **Revisar tablas**: ¿Números coherentes?
3. **Leer summary_rq6.json**: Resumen completo
4. **Usar en paper**: Figuras y tablas están listas para TPAMI

## Tiempo Total Esperado

| Configuración | sample_size | Tiempo Inferencia | Tiempo Total |
|--------------|-------------|-------------------|--------------|
| Prueba rápida | 50 | ~2 min | ~5 min |
| Estándar | 500 | ~15 min | ~20 min |
| Completo | 2000 | ~60 min | ~70 min |

*Tiempos con GPU NVIDIA RTX 3090 o superior*

## Notas Finales

- 🚀 **Primera vez**: Ejecutar todo (~20 min)
- ⚡ **Re-análisis**: Saltar celda 5 (~3 min)
- 📊 **Visualización**: Solo celdas 7-14 (~1 min)
- 🔍 **Debugging**: Reducir sample_size a 10-20

**¡Listo para generar resultados para RQ6!** 🎉
