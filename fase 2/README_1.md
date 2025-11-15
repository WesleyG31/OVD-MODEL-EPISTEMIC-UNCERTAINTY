# Fase 2: Baseline OVD - BDD100K - Detection

Pipeline completo para establecer el baseline de detección con Grounding-DINO sobre BDD100K.

## Descripción

Este notebook implementa todas las etapas necesarias para crear un baseline reproducible de Open-Vocabulary Detection, incluyendo:

1. **Configuración del modelo** (Grounding-DINO SwinT-OGC)
2. **Definición del vocabulario** (10 clases BDD100K)
3. **Inferencia sobre val_eval** con post-procesamiento (NMS, normalización)
4. **Evaluación COCO** (mAP, AP50, AP75, por clase)
5. **Análisis de sensibilidad** a umbrales
6. **Visualización cualitativa** (50 imágenes de muestra)
7. **Preparación para calibración** (generación de inputs sobre val_calib)
8. **Análisis de errores** (FP, FN, pares de confusión)

## Requisitos

### Hardware
- GPU con CUDA (mínimo 8GB VRAM)
- Recomendado: RTX 3090/4090 o superior

### Software
```bash
pip install torch torchvision
pip install pycocotools
pip install pandas matplotlib seaborn
pip install Pillow tqdm pyyaml
```

### Modelo Grounding-DINO
El modelo debe estar instalado en `../installing_dino/GroundingDINO/` con:
- Pesos: `weights/groundingdino_swint_ogc.pth`
- Config: `groundingdino/config/GroundingDINO_SwinT_OGC.py`

## Estructura de Dataset

```
../data/
├── bdd100k/bdd100k/bdd100k/images/100k/val/  # Imágenes
└── bdd100k_coco/
    ├── val_eval.json   # Anotaciones para evaluación
    └── val_calib.json  # Anotaciones para calibración (opcional)
```

## Ejecución

### Opción 1: Ejecutar todo el notebook
Ejecuta todas las celdas secuencialmente desde el principio.

### Opción 2: Ejecución por secciones
1. **Secciones 1-4**: Setup y carga del modelo (~2 min)
2. **Sección 6**: Inferencia sobre val_eval (~30-60 min dependiendo del tamaño)
3. **Sección 8**: Evaluación COCO (~2 min)
4. **Secciones 9-11**: Análisis y visualización (~10 min)
5. **Sección 12**: Preparación calibración (~30 min si val_calib existe)

## Artefactos Generados

### `outputs/baseline/`
- `preds_raw.json` - Predicciones en formato COCO
- `metrics.json` - Métricas de evaluación completas
- `perf.txt` - Métricas de rendimiento (FPS, memoria)
- `threshold_sweep.csv` - Sensibilidad a umbrales
- `summary_table.csv` - Tabla resumen para la tesis
- `error_analysis.json` - Análisis de errores común
- `calib_inputs.parquet` - Inputs para calibración
- `final_report.json` - Reporte completo del baseline

### `outputs/baseline/pr_curves/`
- Curvas Precision-Recall por cada clase

### `outputs/qualitative/baseline/`
- 50 imágenes con detecciones visualizadas

### `configs/`
- `baseline.yaml` - Configuración completa reproducible

### `../data/prompts/`
- `bdd100k.txt` - Lista de prompts para las 10 clases

## Configuración Baseline

### Modelo
- **Nombre**: Grounding-DINO
- **Arquitectura**: SwinT-OGC
- **Input size**: 800×1333 (adaptativo)
- **Checkpoint**: groundingdino_swint_ogc.pth

### Inferencia
- **conf_threshold**: 0.30
- **nms_iou**: 0.65
- **max_detections**: 300
- **batch_size**: 1

### Vocabulario (10 clases)
1. person
2. rider
3. car
4. truck
5. bus
6. train
7. motorcycle
8. bicycle
9. traffic light
10. traffic sign

## Métricas Esperadas

Basándose en reportes de BDD100K con modelos similares:

- **mAP@[.50:.95]**: ~0.15-0.30
- **AP@50**: ~0.30-0.50
- **AP@75**: ~0.15-0.30

Nota: OVD suele tener menor rendimiento que modelos cerrados debido a la generalización.

## Análisis de Rendimiento

El notebook mide automáticamente:
- **Latencia**: tiempo promedio por imagen
- **Throughput**: FPS
- **Memoria GPU**: pico de uso
- **Detecciones/imagen**: promedio

## Siguientes Pasos (Fase 3)

Con el baseline completado, estarás listo para:

1. **Calibración de scores**: Temperature Scaling sobre `calib_inputs.parquet`
2. **Métricas de incertidumbre**: ECE, Brier Score, NLL
3. **Métodos de estimación**: MC-Dropout, Ensembles, etc.
4. **Comparación**: baseline vs. métodos calibrados

## Criterios Go/No-Go

El notebook verifica automáticamente:
- ✅ mAP y AP50 razonables
- ✅ Latencia medida
- ✅ Todos los artefactos generados
- ✅ Inputs para calibración creados
- ✅ Errores identificados

## Troubleshooting

### Error: CUDA out of memory
- Reducir tamaño de imagen en el config de Grounding-DINO
- Usar batch_size=1

### Error: pycocotools no encuentra archivo
- Verificar que las rutas en `baseline.yaml` sean correctas
- Verificar que `val_eval.json` use IDs numéricos (no strings)

### Advertencia: No se encuentra val_calib.json
- Normal si solo tienes val_eval
- La sección de calibración se saltará automáticamente

### Resultados muy bajos
- Verificar que el vocabulario incluya todas las clases
- Revisar normalización de labels (sinónimos)
- Ajustar conf_threshold y nms_iou en el barrido

## Referencias

- [Grounding-DINO Paper](https://arxiv.org/abs/2303.05499)
- [BDD100K Dataset](https://www.bdd100k.com/)
- [COCO Evaluation](https://cocodataset.org/#detection-eval)

## Versión y Commit

Generado automáticamente el: 2025-11-09

Para reproducibilidad, asegúrate de:
1. Usar el mismo commit de Grounding-DINO
2. Usar los mismos pesos del modelo
3. Fijar las seeds (ya implementado en el código)
