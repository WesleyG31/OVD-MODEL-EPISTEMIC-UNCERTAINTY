# RQ6 — Decoder Dynamics as Epistemic Uncertainty Signals

## 📋 Resumen

Este notebook responde la **Research Question 6** del proyecto sobre incertidumbre epistémica en modelos de detección de objetos open-vocabulary (OVD).

**RQ6**: ¿Qué propiedades intrínsecas de la dinámica del decoder transformer codifican incertidumbre epistémica en OVD, y cuándo la varianza inter-capa sirve de proxy confiable para la incertidumbre del modelo?

**Hipótesis**: La incertidumbre se alinea más con los errores conforme aumenta la profundidad del decoder: las predicciones TP se estabilizan antes que las FP; la varianza en capas tardías separa mejor los errores y mejora el AUROC de detección de errores.

## 🎯 Objetivos

1. **Capturar dinámicas del decoder**: Extraer embeddings de cada capa del decoder de GroundingDINO
2. **Calcular incertidumbre inter-capa**: Usar varianza entre capas como proxy de incertidumbre epistémica
3. **Analizar evolución por profundidad**: Verificar que la discriminación mejora en capas tardías
4. **Identificar condiciones de falla**: Encontrar escenarios donde la varianza es menos predictiva

## 📊 Deliverables

### Figuras (TPAMI-style)
- ✅ **Figure RQ6.1**: Varianza inter-capa de bounding-box por profundidad (TP vs FP)
- ✅ **Figure RQ6.2**: AUROC de detección de errores por capa del decoder

### Tablas (TPAMI-style)
- ✅ **Table RQ6.1**: Diagnósticos de efectividad de incertidumbre por capa
- ✅ **Table RQ6.2**: Condiciones de falla donde la varianza es menos predictiva

### Datos
- ✅ `decoder_dynamics.parquet`: Detecciones con varianzas por capa
- ✅ `layer_variance_stats.csv`: Estadísticas de varianza por capa
- ✅ `auroc_by_layer.csv`: AUROC por profundidad del decoder
- ✅ `summary_rq6.json`: Resumen completo de resultados

## 🚀 Quick Start

### Ejecución Rápida (3 comandos)
```bash
# 1. Abrir notebook
jupyter notebook rq6.ipynb

# 2. Ejecutar celdas clave (marcar "✅ EJECUTAR PARA RQ6")
#    - Celda 1: Configuración
#    - Celda 2: Cargar modelo
#    - Celda 5: Inferencia (15-20 min)

# 3. Verificar outputs
ls output/  # Debe mostrar 14 archivos
```

### Tiempo de Ejecución
- **Primera vez**: ~20-25 minutos (con GPU)
- **Re-análisis**: ~3 minutos (si ya existe decoder_dynamics.parquet)

## 📁 Estructura de Archivos

```
new_rq6/
├── rq6.ipynb                    # 📓 Notebook principal (30 celdas)
├── output/                      # 📂 Directorio de resultados
│   ├── Fig_RQ6_1_*.png/pdf     # 🖼️ Figura 1: Varianza TP vs FP
│   ├── Fig_RQ6_2_*.png/pdf     # 🖼️ Figura 2: AUROC por capa
│   ├── Table_RQ6_1.csv/.tex    # 📋 Tabla 1: Layer-wise diagnostics
│   ├── Table_RQ6_2.csv/.tex    # 📋 Tabla 2: Failure conditions
│   ├── decoder_dynamics.parquet # 💾 Datos crudos
│   ├── layer_variance_stats.csv # 📊 Estadísticas por capa
│   ├── auroc_by_layer.csv       # 📊 AUROC por capa
│   ├── summary_rq6.json         # 📄 Resumen JSON
│   └── figure_captions.txt      # 📝 Captions TPAMI
├── README_RQ6.md                # 📖 Documentación completa
├── QUICKSTART.md                # ⚡ Guía de inicio rápido
├── RESUMEN_EJECUTIVO.md         # 📋 Resumen del notebook
└── ARQUITECTURA_TECNICA.md      # 🏗️ Detalles técnicos
```

## 🔬 Metodología

### 1. Captura de Embeddings del Decoder
- Se registran **hooks** en cada una de las 6 capas del decoder de GroundingDINO
- Durante inferencia, se capturan los embeddings de cada query en cada capa
- Formato: `[num_queries, batch, embed_dim]` → `[900, 1, 256]`

### 2. Cálculo de Varianza Inter-Capa
Para cada detección:
```python
layer_scores = [score_layer_0, score_layer_1, ..., score_layer_5]
uncertainty = np.var(layer_scores)  # Incertidumbre epistémica
```

### 3. Matching con Ground Truth
- Cada predicción se matchea con GT usando IoU
- TP (True Positive): IoU ≥ 0.5 y categoría correcta
- FP (False Positive): IoU < 0.5 o categoría incorrecta

### 4. Análisis Progresivo por Profundidad
Para cada capa ℓ ∈ {1, 2, 3, 4, 5, 6}:
- Calcular varianza acumulada usando capas 1..ℓ
- Computar AUROC para detección de errores
- Separar estadísticas de TP vs FP

## 📈 Resultados Esperados

### Hipótesis 1: TP se estabilizan antes que FP
**Métrica**: Var(TP) < Var(FP) en capas tardías
- TP alcanzan consensus rápido (baja varianza)
- FP mantienen alta varianza (incertidumbre)

### Hipótesis 2: Capas tardías mejoran AUROC
**Métrica**: AUROC(capa 6) > AUROC(capa 1)
- Primera capa: AUROC ≈ 0.65-0.70
- Última capa: AUROC ≈ 0.85-0.90
- Mejora: +0.15 a +0.25

### Hipótesis 3: Separación aumenta con profundidad
**Métrica**: (Var(FP) - Var(TP)) aumenta con la capa
- Capas tempranas: Separación baja
- Capas tardías: Separación alta
- Indica concentración progresiva de señal epistémica

## 📋 Prerequisitos

### Software
- Python 3.8+
- PyTorch 1.12+ con CUDA
- GroundingDINO instalado en `/opt/program/GroundingDINO/`
- Librerías: pandas, numpy, matplotlib, seaborn, sklearn, pycocotools

### Datos
- Dataset BDD100K en `../../data/bdd100k_coco/`
- Split: `val_eval.json` (2,000 imágenes)
- Se procesan las primeras 500 por defecto

### Hardware
- **GPU**: NVIDIA con ≥8GB VRAM (recomendado)
- **CPU**: Funciona pero ~10x más lento
- **RAM**: ≥16GB
- **Disco**: ~500MB para outputs

## 🔧 Configuración

### Parámetros Principales (Celda 1)
```python
CONFIG = {
    'seed': 42,                  # Reproducibilidad
    'device': 'cuda',            # cuda o cpu
    'sample_size': 500,          # Imágenes a procesar
    'iou_matching': 0.5,         # Threshold para TP/FP
    'conf_threshold': 0.25,      # Confianza mínima
    'num_layers': 6              # Capas del decoder
}
```

### Ajustes Comunes

#### Ejecución rápida (pruebas)
```python
'sample_size': 50  # 2 minutos en lugar de 15
```

#### Ejecución completa (paper)
```python
'sample_size': 2000  # Todo val_eval
```

## 📚 Documentación Adicional

- **[QUICKSTART.md](QUICKSTART.md)**: Guía de inicio rápido (5 minutos de lectura)
- **[README_RQ6.md](README_RQ6.md)**: Documentación completa con troubleshooting
- **[RESUMEN_EJECUTIVO.md](RESUMEN_EJECUTIVO.md)**: Resumen del notebook (30 celdas)
- **[ARQUITECTURA_TECNICA.md](ARQUITECTURA_TECNICA.md)**: Detalles técnicos y flujo de datos

## ✅ Validación

### Checklist de Resultados
- [ ] Figure RQ6.1 muestra FP > TP en varianza
- [ ] Figure RQ6.2 muestra AUROC creciente
- [ ] Table RQ6.1 tiene valores coherentes
- [ ] Table RQ6.2 lista condiciones de falla
- [ ] summary_rq6.json confirma las 3 hipótesis

### Validación Automática
El notebook incluye validación automática de hipótesis:
```
✓ H1 (TP estabilizan antes que FP): CONFIRMADA
✓ H2 (Capas tardías mejor AUROC): CONFIRMADA
✓ H3 (Separación aumenta con profundidad): CONFIRMADA
```

## 🐛 Troubleshooting

### Problemas Comunes

#### "CUDA out of memory"
**Solución**: Reducir `sample_size` a 50 o 100

#### "Model not found"
**Solución**: Verificar instalación de GroundingDINO
```bash
ls /opt/program/GroundingDINO/weights/groundingdino_swint_ogc.pth
```

#### "Dataset not found"
**Solución**: Verificar path relativo al dataset
```bash
ls ../../data/bdd100k_coco/val_eval.json
```

#### Ejecución muy lenta
**Causa**: Ejecutando en CPU en lugar de GPU
**Solución**: Verificar `torch.cuda.is_available() == True`

## 📊 Métricas y KPIs

### Métricas de Dataset
- Imágenes procesadas: 500 (configurable)
- Detecciones esperadas: ~8,000-10,000
- TP rate esperado: ~80-85%
- FP rate esperado: ~15-20%

### Métricas de Calidad
- AUROC primera capa: 0.65-0.70
- AUROC última capa: 0.85-0.90
- Mejora en AUROC: +0.15 a +0.25
- Separación Var(FP)-Var(TP): Positiva y creciente

## 🎓 Contexto del Proyecto

Este notebook es parte de un proyecto más amplio sobre incertidumbre epistémica en OVD:

- **Fase 2**: Baseline sin incertidumbre
- **Fase 3**: MC-Dropout para incertidumbre
- **Fase 4**: Temperature scaling para calibración
- **Fase 5**: Comparación de métodos
- **RQ6**: Análisis de dinámicas del decoder (este notebook)

## 🤝 Contribuciones

### Estructura del Código
- Código bien documentado en español
- Contenido de figuras en inglés (TPAMI-style)
- Funciones modulares y reutilizables
- Seeds fijados para reproducibilidad

### Extensiones Posibles
1. Analizar más capas (si el modelo tiene más de 6)
2. Probar otros transformers (DETR, etc.)
3. Agregar análisis por categoría
4. Estudiar varianza temporal (video)

## 📄 Licencia

Este código es parte del proyecto de investigación sobre incertidumbre epistémica en OVD. Uso académico y de investigación.

## 📞 Soporte

### Recursos
- Documentación: Ver archivos .md en este directorio
- Issues: Revisar troubleshooting en README_RQ6.md
- Logs: Revisar outputs del notebook

### Contacto
Para preguntas específicas sobre RQ6, revisar primero:
1. QUICKSTART.md (inicio rápido)
2. README_RQ6.md (troubleshooting)
3. ARQUITECTURA_TECNICA.md (detalles técnicos)

---

## 🎉 ¡Listo para Ejecutar!

```bash
# Paso 1: Abrir notebook
jupyter notebook rq6.ipynb

# Paso 2: Ejecutar celdas marcadas "✅ EJECUTAR PARA RQ6"
# (Celdas 1, 2, 5)

# Paso 3: Ver resultados
ls output/
```

**Tiempo total: ~20 minutos** ⏱️

**Outputs esperados: 14 archivos** 📁

**Figuras listas para paper: 2 (PNG + PDF)** 🖼️

**Tablas listas para paper: 2 (CSV + LaTeX)** 📋

---

*Generado para responder RQ6 del proyecto OVD Epistemic Uncertainty*  
*Última actualización: 2026-02-04*
