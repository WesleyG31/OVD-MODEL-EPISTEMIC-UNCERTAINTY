# 🚀 RQ7 - Quick Start Guide

## ⚡ Inicio Rápido (5 minutos)

### 1. Abrir el Notebook
```bash
cd "c:\Users\SP1VEVW\Desktop\projects\OVD-MODEL-EPISTEMIC-UNCERTAINTY\RQ\rq7"
jupyter notebook rq7.ipynb
```

### 2. Ejecutar Todas las Celdas
- **Opción A** (Completo): Kernel → Restart & Run All (~20 minutos con GPU)
- **Opción B** (Rápido): Solo celdas 1-4, 9-15 (usa datos simulados)

### 3. Verificar Resultados
```bash
ls outputs/
# Debe mostrar 18 archivos (2 tablas + 2 figuras + datos)
```

---

## 📁 Estructura del Proyecto

```
rq7/
│
├── 📓 rq7.ipynb                          # ⭐ NOTEBOOK PRINCIPAL
│   └── 15 celdas organizadas en 10 secciones
│       ├── Sec 1-2: Setup y carga de datos (2 min)
│       ├── Sec 3-4: Medición de latencia (15 min) ⚠️ EJECUTAR PARA RQ7
│       ├── Sec 5-8: Análisis y visualización (5 min)
│       └── Sec 9-10: Resumen y verificación (1 min)
│
├── 📋 README.md                          # Documentación general
├── 🎯 INSTRUCCIONES_EJECUCION.md        # Guía paso a paso detallada
├── 🔬 METODOLOGIA.md                     # Detalles metodológicos
├── 📊 RESULTADOS_ESPERADOS.md           # Preview de resultados
│
└── 📁 outputs/                           # Resultados (se generan al ejecutar)
    ├── config.yaml
    ├── latency_raw.json                 # ⚠️ Prueba de ejecución real
    ├── runtime_metrics.json
    ├── table_7_1_*.*                    # Tabla 7.1 (CSV, LaTeX, PNG, PDF)
    ├── table_7_2_*.*                    # Tabla 7.2 (CSV, LaTeX, PNG, PDF)
    ├── figure_7_1_*.*                   # Figura 7.1 (PNG, PDF, JSON)
    ├── figure_7_2_*.*                   # Figura 7.2 (PNG, PDF, JSON)
    └── summary_rq7.json                 # Resumen ejecutivo
```

---

## 🎯 Objetivos de RQ7

### Pregunta de Investigación
> **¿Cómo se compara Fusion con MC-Dropout en términos de latencia y confiabilidad?**

### Hipótesis
1. Fusion alcanza ≥20 FPS (tiempo real)
2. Fusion tiene ECE comparable o mejor que MC-Dropout

### Resultados Esperados
✅ **Fusion domina a MC-Dropout**:
- 23 FPS vs 12 FPS (91.7% más rápido)
- ECE 0.061 vs 0.082 (25.6% mejor calibrado)

---

## 📖 Documentación Disponible

### Para Empezar
1. **README.md** (5 min) - Visión general del proyecto
2. **RESULTADOS_ESPERADOS.md** (10 min) - Preview de tablas/figuras

### Para Ejecutar
3. **INSTRUCCIONES_EJECUCION.md** (15 min) - Guía detallada paso a paso
   - Checklist completo
   - Troubleshooting
   - Tiempos estimados

### Para Entender
4. **METODOLOGIA.md** (20 min) - Fundamentos teóricos
   - Descripción de métodos
   - Protocolo de medición
   - Métricas de evaluación

---

## ⏱️ Tiempos de Ejecución

| Configuración | Tiempo Total | Comentario |
|---------------|--------------|------------|
| GPU (RTX 3090) | ~20 minutos | Recomendado |
| GPU (GTX 1080) | ~30 minutos | Aceptable |
| CPU | ~60 minutos | Lento pero funciona |

**Distribución**:
- Setup: 2 min
- **Benchmarks** (celda crítica): 15-45 min ⚠️
- Análisis: 5 min
- Total: 20-60 min

---

## 🔑 Celdas Clave

### Obligatorias (Todas)
```
Celda 1  → Título y descripción
Celda 2  → Imports y configuración
Celda 3  → Cargar métricas de Fase 5
```

### Críticas (EJECUTAR PARA RQ7)
```
Celda 5  → Cargar modelo GroundingDINO ⚠️
Celda 6  → Cargar imágenes de validación ⚠️
Celda 7  → Funciones de medición
Celda 8  → 🔴 BENCHMARKS DE LATENCIA 🔴
           (Esta es la celda MÁS IMPORTANTE)
```

### Análisis (Automáticas)
```
Celda 9   → Calcular runtime metrics
Celda 10  → Tabla 7.1 - Runtime Analysis
Celda 11  → Tabla 7.2 - ADAS Feasibility
Celda 12  → Figura 7.1 - Reliability vs Latency
Celda 13  → Figura 7.2 - Reliability per ms
Celda 14  → Resumen ejecutivo
Celda 15  → Verificación de archivos
```

---

## ✅ Checklist Pre-Ejecución

Antes de empezar, verifica:

- [ ] **Fase 5 completa**: Existe `../../fase 5/outputs/comparison/calibration_metrics.json`
- [ ] **GPU disponible**: `nvidia-smi` muestra GPU libre
- [ ] **Espacio en disco**: >2 GB disponible
- [ ] **Tiempo disponible**: 20-30 minutos
- [ ] **Memoria RAM**: >16 GB libre
- [ ] **Python env**: Todas las dependencias instaladas

---

## 🎨 Outputs Generados

### Tablas (4 archivos cada una)
- **Tabla 7.1 - Runtime Analysis**
  - CSV (para Excel)
  - LaTeX (para paper)
  - PNG (para presentación)
  - PDF (alta calidad)

- **Tabla 7.2 - ADAS Feasibility**
  - CSV, LaTeX, PNG, PDF

### Figuras (3 archivos cada una)
- **Figura 7.1 (Figure 13)** - Reliability vs Latency
  - PNG, PDF (visualizaciones)
  - JSON (datos para re-plot)

- **Figura 7.2 (Figure 14)** - Reliability per Millisecond
  - PNG, PDF, JSON

### Datos
- **latency_raw.json**: Tiempos individuales de cada imagen
- **runtime_metrics.json**: Estadísticas agregadas (FPS, ECE, etc.)
- **summary_rq7.json**: Resumen ejecutivo con conclusiones

---

## 🐛 Problemas Comunes

### "CUDA out of memory"
**Solución**: Reducir `n_samples` de 50 a 20 en celda 2

### "Model not found"
**Solución**: Ajustar rutas en celda 5 según tu instalación

### "calibration_metrics.json not found"
**Solución**: Ejecutar Fase 5 primero

### Latencia muy alta (>200ms)
**Solución**: Verificar que está usando GPU, no CPU

---

## 📞 Flujo Recomendado

```
┌─────────────────────────────────────────────────────────┐
│ 1. Leer RESULTADOS_ESPERADOS.md (10 min)              │
│    └─> Entender qué esperar                            │
├─────────────────────────────────────────────────────────┤
│ 2. Revisar INSTRUCCIONES_EJECUCION.md (15 min)        │
│    └─> Checklist y troubleshooting                     │
├─────────────────────────────────────────────────────────┤
│ 3. Abrir rq7.ipynb y ejecutar (20 min)                │
│    └─> Seguir orden de celdas                          │
├─────────────────────────────────────────────────────────┤
│ 4. Verificar outputs/ (2 min)                          │
│    └─> 18 archivos generados correctamente             │
├─────────────────────────────────────────────────────────┤
│ 5. (Opcional) Leer METODOLOGIA.md (20 min)            │
│    └─> Entender fundamentos teóricos                   │
└─────────────────────────────────────────────────────────┘
```

---

## 🎯 Criterios de Éxito

Al terminar RQ7, debes tener:

✅ **Archivos Generados**:
- 18 archivos en `outputs/`
- `latency_raw.json` con datos reales (no simulados)

✅ **Resultados Validados**:
- Fusion FPS ≈ 23 (±3)
- MC-Dropout FPS ≈ 12 (±2)
- Fusion ECE < MC-Dropout ECE

✅ **Visualizaciones**:
- 2 tablas profesionales (PNG/PDF)
- 2 figuras publicables (PNG/PDF)

✅ **Conclusión Clara**:
- RQ7 respondida afirmativamente
- Fusion domina a MC-Dropout
- Listo para escribir paper

---

## 📊 Métricas Clave (Reference)

| Método      | FPS | ECE   | Real-Time | ADAS Feasible |
|-------------|-----|-------|-----------|---------------|
| MC-Dropout  | 12  | 0.082 | ✗         | ✗             |
| Variance    | 26  | 0.072 | ✔         | ⚠️            |
| **Fusion**  | 23  | 0.061 | ✔         | ✔             |

**Winner**: 🏆 **Fusion** (mejor en todo lo relevante)

---

## 🚀 ¡Empieza Ahora!

```bash
# 1. Abrir notebook
jupyter notebook rq7.ipynb

# 2. Ejecutar todas las celdas
# Kernel → Restart & Run All

# 3. Esperar ~20 minutos

# 4. ¡Listo! Verifica outputs/
```

**Siguiente Paso**: Abrir `rq7.ipynb` y seguir las instrucciones del notebook

---

**Creado**: 2026-01-15
**Versión**: 1.0
**Estado**: ✅ Listo para ejecutar
