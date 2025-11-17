# 🚗 Demo Interactiva: OVD con Calibración e Incertidumbre

## 📋 Descripción

Aplicación web interactiva que demuestra:
- **Detección Open-Vocabulary** en escenas ADAS (BDD100K)
- **Calibración de probabilidades** mediante Temperature Scaling
- **Incertidumbre epistémica** mediante MC-Dropout y varianza decoder
- **Filtrado inteligente** basado en incertidumbre

## 🚀 Ejecución

```bash
cd fase\ 6
streamlit run app/demo.py
```

La aplicación se abrirá en `http://localhost:8501`

## 🎯 Funcionalidades

### Métodos Disponibles

1. **Baseline**: Detección estándar sin calibración ni incertidumbre
2. **Baseline + TS**: Con calibración de probabilidades
3. **MC-Dropout K=5**: 5 pases estocásticos para incertidumbre
4. **MC-Dropout K=5 + TS**: Con calibración
5. **Varianza Decoder**: Incertidumbre desde múltiples capas (single-pass)
6. **Varianza Decoder + TS**: Con calibración

### Controles

- **Umbral de confianza**: Filtrar detecciones por probabilidad
- **Umbral de incertidumbre**: Filtrar detecciones inciertas
- **Carga de imagen**: Subir propia o usar muestras pre-seleccionadas
- **Métricas globales**: Ver rendimiento general del método

### Visualización

- **Cajas de detección** coloreadas por clase
- **Etiquetas** con clase, confianza y nivel de incertidumbre
- **Tabla de detecciones** con valores numéricos
- **Histograma** de distribución de incertidumbre

## 📊 Interpretación

### Calibración
- **Sin TS**: El modelo puede ser sobreconfiado (p=0.95 pero accuracy real 70%)
- **Con TS**: Probabilidades ajustadas a frecuencia real de aciertos

### Incertidumbre
- **Baja (< 0.05)**: El modelo está seguro, decisión confiable
- **Media (0.05-0.1)**: Cierta duda, usar con precaución
- **Alta (> 0.1)**: Modelo muy incierto, requiere verificación

### Uso en ADAS
- **Modo seguro**: Filtrar por umbral de incertidumbre
- **Detecciones de alta incertidumbre**: Alertar al conductor
- **Detecciones de baja incertidumbre**: Actuar automáticamente

## 🎨 Casos de Uso

La demo incluye 9 imágenes pre-seleccionadas:

- **Casos fáciles (3)**: Pocos objetos, buena iluminación
- **Casos medios (3)**: Tráfico moderado, condiciones normales  
- **Casos difíciles (3)**: Muchos objetos, oclusión, condiciones adversas

## 📈 Métricas Mostradas

- **mAP**: Precisión media del método
- **ECE**: Error de calibración esperado
- **Total detecciones**: Número de objetos detectados
- **Alta incertidumbre**: Detecciones que requieren atención

## 🔧 Requisitos

- Python 3.8+
- GroundingDINO instalado
- CUDA (opcional, acelera inferencia)
- Resultados de Fases 4 y 5 disponibles

## 📝 Notas

- La inferencia con MC-Dropout (K=5) toma ~5x más tiempo que baseline
- Varianza decoder es más rápido pero menos preciso
- Temperature Scaling requiere resultados de Fase 4
- Las métricas globales provienen de la Fase 5

## 🎓 Para la Defensa

Esta demo permite:
1. Mostrar visualmente el efecto de la calibración
2. Demostrar cuándo el modelo es incierto
3. Explicar cómo usar incertidumbre para decisiones seguras en ADAS
4. Comparar métodos en tiempo real

## 📸 Capturas de Pantalla

Ejecutar la demo y tomar capturas de:
- Caso fácil con baja incertidumbre
- Caso difícil con alta incertidumbre
- Comparación antes/después de calibración
- Efecto del filtrado por incertidumbre
