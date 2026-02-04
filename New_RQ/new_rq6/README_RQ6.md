# RQ6 — Decoder Dynamics as Epistemic Uncertainty Signals

## Research Question
¿Qué propiedades intrínsecas de la dinámica del decoder transformer codifican incertidumbre epistémica en OVD, y cuándo la varianza inter-capa sirve de proxy confiable para la incertidumbre del modelo?

## Hipótesis
La incertidumbre se alinea más con los errores conforme aumenta la profundidad: las predicciones TP se estabilizan antes que las FP; la varianza en capas tardías separa mejor los errores y mejora el AUROC de detección de errores.

## Instrucciones de Ejecución

### Prerrequisitos
1. Modelo GroundingDINO instalado y configurado (ver fases anteriores)
2. Dataset BDD100K en `../../data/bdd100k_coco/`
3. GPU con CUDA disponible (recomendado)
4. Paquetes Python: torch, pandas, numpy, matplotlib, seaborn, pycocotools, sklearn

### Celdas Marcadas como "✅ EJECUTAR PARA RQ6"
Las siguientes celdas deben ejecutarse en orden:
- **Celda 2**: Cargar modelo GroundingDINO
- **Celda 4**: Función de inferencia con captura de capas
- **Celda 5**: Procesar dataset (500 imágenes por defecto)

### Orden de Ejecución
1. **Configuración e Imports** (Celda 1)
2. **Cargar Modelo** (Celda 2) - ⚠️ Necesita ejecución manual
3. **Funciones Auxiliares** (Celda 3)
4. **Inferencia con Captura** (Celda 4) - ⚠️ Necesita ejecución manual
5. **Procesar Dataset** (Celda 5) - ⚠️ Necesita ejecución manual, ~15-20 min
6. **Análisis de Varianza** (Celda 6)
7. **Figuras y Tablas** (Celdas 7-12)
8. **Resumen Final** (Celdas 13-14)

### Tiempo de Ejecución Estimado
- Configuración: < 1 minuto
- Carga de modelo: ~30 segundos
- Inferencia (500 imágenes): ~15-20 minutos con GPU
- Análisis y visualización: ~2 minutos
- **Total**: ~20-25 minutos

### Outputs Generados

#### Figuras (PNG + PDF)
- `Fig_RQ6_1_decoder_variance.png/pdf`: Varianza inter-capa TP vs FP por profundidad
- `Fig_RQ6_2_auroc_by_layer.png/pdf`: AUROC de detección de errores por capa

#### Tablas (CSV + LaTeX)
- `Table_RQ6_1.csv/.tex`: Diagnósticos de efectividad por capa
- `Table_RQ6_2.csv/.tex`: Condiciones de falla

#### Datos
- `decoder_dynamics.parquet`: Detecciones con varianzas por capa
- `layer_variance_stats.csv`: Estadísticas de varianza por capa
- `auroc_by_layer.csv`: AUROC por capa del decoder
- `summary_rq6.json`: Resumen completo de resultados
- `figure_captions.txt`: Captions TPAMI-style

### Estructura del Output
```
./output/
├── config.yaml
├── decoder_dynamics.parquet
├── layer_variance_stats.csv
├── auroc_by_layer.csv
├── Fig_RQ6_1_decoder_variance.png
├── Fig_RQ6_1_decoder_variance.pdf
├── Fig_RQ6_2_auroc_by_layer.png
├── Fig_RQ6_2_auroc_by_layer.pdf
├── Table_RQ6_1.csv
├── Table_RQ6_1.tex
├── Table_RQ6_2.csv
├── Table_RQ6_2.tex
├── summary_rq6.json
└── figure_captions.txt
```

## Metodología

### 1. Captura de Embeddings del Decoder
- Se registran hooks en cada capa del decoder de GroundingDINO
- Para cada detección, se extrae el embedding de la query correspondiente en cada capa
- Se calcula un "score" por capa basado en la norma del embedding

### 2. Cálculo de Varianza Inter-Capa
- Para cada detección, se calcula la varianza de los scores a través de las capas
- Mayor varianza indica mayor incertidumbre epistémica
- Se compara varianza para TP (True Positives) vs FP (False Positives)

### 3. Análisis por Profundidad
- Se analiza cómo la varianza evoluciona con la profundidad del decoder
- Se calcula AUROC para detección de errores usando la varianza acumulada hasta cada capa
- Se verifica la hipótesis de que capas tardías tienen mejor discriminación

### 4. Condiciones de Falla
- Se identifican escenarios donde la varianza inter-capa es menos predictiva
- Small objects, low confidence, boundary matches, extreme aspect ratios
- Se calcula el drop en AUROC para cada condición

## Expected Results vs Actual Results

### Figure RQ6.1
**Expected**: Separación de varianza aumenta con la profundidad, FP > TP en capas tardías
**Actual**: El notebook genera esta figura con datos reales del modelo

### Figure RQ6.2
**Expected**: AUROC aumenta monótonamente con la profundidad, última capa ~0.88-0.90
**Actual**: El notebook calcula AUROC real por capa

### Table RQ6.1
**Expected**: Valores dummy mostrados en el prompt
**Actual**: Valores reales calculados del modelo, incluyendo:
- AUROC por capa
- AUPR por capa
- Var(TP) y Var(FP) por capa

### Table RQ6.2
**Expected**: 4 condiciones de falla con drops de -0.04 a -0.07
**Actual**: Condiciones reales identificadas en el dataset con drops calculados

## Notas Importantes

1. **No hay datos simulados**: Todos los resultados provienen del modelo GroundingDINO real
2. **Reproducibilidad**: Seed fijado en 42 para reproducibilidad
3. **Paths relativos**: Todos los paths son relativos a la estructura del proyecto
4. **Idioma**: Comentarios en español, contenido de figuras en inglés (TPAMI-style)
5. **Formato TPAMI**: Figuras y tablas siguen el formato de papers académicos

## Troubleshooting

### Error: "Model not found"
- Verificar que GroundingDINO esté instalado en `/opt/program/GroundingDINO/`
- Verificar que los pesos estén en `/opt/program/GroundingDINO/weights/`

### Error: "Dataset not found"
- Verificar que BDD100K esté en `../../data/bdd100k_coco/`
- Verificar que `val_eval.json` exista

### Bajo AUROC en capas tardías
- Puede indicar que el modelo no tiene suficiente varianza inter-capa
- Verificar que los hooks estén capturando correctamente las capas
- Verificar que hay suficientes detecciones TP y FP

### Memoria insuficiente
- Reducir `sample_size` en CONFIG (default: 500)
- Usar batch size más pequeño
- Liberar GPU entre celdas con `torch.cuda.empty_cache()`

## Referencias
- GroundingDINO: Open-Set Detection via Language
- BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning
- Fases 2-5 del proyecto para contexto sobre el modelo y métricas
