# 🚀 Quickstart - RQ8

## Ejecución Rápida (3 pasos)

### 1️⃣ Abrir Notebook
```bash
cd New_RQ/new_rq8
# Abrir rq8.ipynb en VS Code
```

### 2️⃣ Ejecutar Celdas Secuencialmente
- ▶️ **Run All** en VS Code
- ⏱️ Tiempo total: ~50-60 minutos
- ☕ Tomar un café durante la inferencia (celda 4)

### 3️⃣ Verificar Resultados
```bash
# Los archivos estarán en ./output/
ls output/
```

## ⚠️ Celdas Críticas

### Celda 2: Cargar Modelo
```python
# ✅ EJECUTAR PARA RQ8 - Cargar modelo GroundingDINO
```
- Tiempo: ~10 segundos
- Requiere: GPU con CUDA

### Celda 4: Inferencia
```python
# ✅ EJECUTAR PARA RQ8 - Inferencia y matching con ground truth
```
- Tiempo: ~40-50 minutos ⚠️
- Procesa: 500 imágenes
- Genera: `detections_raw.parquet`

## 📊 Archivos Generados

Al finalizar tendrás:
```
output/
├── 📊 2 Tablas (CSV + JSON)
│   ├── table_rq8_1_score_iou_alignment.csv
│   ├── table_rq8_1.json
│   ├── table_rq8_2_ranking_utility.csv
│   └── table_rq8_2.json
│
├── 📈 2 Figuras (PNG + PDF)
│   ├── Fig_RQ8_1_score_iou_reliability.png
│   ├── Fig_RQ8_1_score_iou_reliability.pdf
│   ├── Fig_RQ8_2_precision_at_k.png
│   └── Fig_RQ8_2_precision_at_k.pdf
│
└── 💾 Datos intermedios
    ├── detections_raw.parquet
    ├── detections_calibrated.parquet
    ├── calibration_params.json
    └── config_rq8.yaml
```

## ✅ Resultados Esperados

### Tabla RQ8.1
```
Scoring rule              | Spearman ρ ↑ | Kendall τ ↑ | ECE-IoU ↓
-------------------------|--------------|-------------|----------
Raw score                 | ~0.34        | ~0.23       | ~0.091
Temp-scaled (cls only)    | ~0.38        | ~0.26       | ~0.083
Joint calibrated (cls+loc)| ~0.62        | ~0.47       | ~0.051
```

**Mejoras esperadas**:
- 📈 Spearman ρ: **+82%**
- 📈 Kendall τ: **+104%**
- 📉 ECE-IoU: **-44%**

### Tabla RQ8.2
```
Budget  | Metric          | Raw   | Calibrated | Mejora
--------|-----------------|-------|------------|-------
Top-100 | Precision@K ↑   | 0.71  | 0.76       | +7.0%
Top-200 | Precision@K ↑   | 0.67  | 0.71       | +6.0%
Top-400 | Precision@K ↑   | 0.62  | 0.65       | +4.8%
Top-400 | Mean IoU ↑      | 0.58  | 0.62       | +6.9%
```

### Figura RQ8.1 - Reliability Diagram
![Expected](https://via.placeholder.com/600x400.png?text=Score+vs+Mean+IoU)

**Qué esperar**:
- 🔴 **Raw**: Curva errática, sin monotonicidad
- 🟡 **Temp Scaling**: Mejora leve
- 🟢 **Joint Calibration**: Curva casi perfecta, cerca de la diagonal

### Figura RQ8.2 - Precision@K
![Expected](https://via.placeholder.com/600x400.png?text=Precision@K+vs+K)

**Qué esperar**:
- 🟢 **Joint Calibration** mantiene precision más alta
- 📈 Separación clara entre métodos
- 📊 Mejora consistente en todo el rango de K

## 🔧 Troubleshooting

### Error: "No CUDA device"
```bash
# Verificar GPU
nvidia-smi
# Si no hay GPU, cambiar en celda 1:
CONFIG['device'] = 'cpu'  # Advertencia: MUY lento (~4 horas)
```

### Error: "Model not found"
```bash
# Verificar path del modelo
ls /opt/program/GroundingDINO/weights/groundingdino_swint_ogc.pth
# Ajustar path en celda 2 si es necesario
```

### Error: "Dataset not found"
```bash
# Verificar dataset
ls ../../data/bdd100k/bdd100k/images/100k/val/
# Ajustar path si es necesario
```

### Inferencia muy lenta
```bash
# Reducir sample_size en celda 1
CONFIG['sample_size'] = 100  # En lugar de 500
# Resultados serán menos robustos pero más rápidos
```

## 📚 Documentación Completa

Para más detalles, consulta:
- 📖 `README_RQ8.md` - Documentación técnica completa (inglés)
- 📊 `RESUMEN_EJECUTIVO_RQ8.md` - Resumen ejecutivo (español)
- ✅ `VERIFICACION_RQ8.md` - Checklist de verificación

## 🎯 TL;DR

1. **Abrir** `rq8.ipynb`
2. **Ejecutar** todas las celdas (Run All)
3. **Esperar** ~1 hora
4. **Verificar** archivos en `./output/`
5. **Listo!** ✅

---

**Tiempo total**: ~50-60 minutos
**Dificultad**: Media (requiere GPU)
**Requisitos**: GroundingDINO + BDD100K dataset
