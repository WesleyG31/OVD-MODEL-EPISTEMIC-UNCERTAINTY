# 📚 RQ5 - ÍNDICE DE DOCUMENTACIÓN

## 📁 Estructura de Archivos

```
RQ/rq5/
├── rq5.ipynb                           ← NOTEBOOK PRINCIPAL
├── outputs/                            ← Carpeta de salida (generada al ejecutar)
│
├── README.md                           ← Descripción general
├── INSTRUCCIONES_EJECUCION.md         ← Guía paso a paso
├── RESUMEN_VISUAL.md                   ← Visualización de estructura
├── RESUMEN_EJECUTIVO.md                ← Resumen de resultados
├── ARQUITECTURA_TECNICA.md             ← Detalles técnicos
└── INDICE_DOCUMENTACION.md            ← Este archivo
```

---

## 📖 Guía de Uso por Rol

### 👨‍🔬 Para Investigador / Tesista

**Quiero entender qué hace este notebook:**
1. 📄 Leer `README.md` - Visión general y objetivos
2. 📊 Leer `RESUMEN_VISUAL.md` - Ver estructura con diagramas
3. 🎯 Leer `RESUMEN_EJECUTIVO.md` - Resultados principales

**Quiero ejecutar el notebook:**
1. 📋 Leer `INSTRUCCIONES_EJECUCION.md` - Paso a paso
2. ⚙️ Ejecutar `rq5.ipynb` - Notebook principal
3. ✅ Verificar outputs en carpeta `outputs/`

**Quiero entender la implementación:**
1. 🏗️ Leer `ARQUITECTURA_TECNICA.md` - Algoritmos y flujos
2. 📓 Revisar código en `rq5.ipynb` - Implementación detallada

---

### 👨‍💻 Para Desarrollador

**Quiero modificar el algoritmo de fusión:**
1. Ir a `rq5.ipynb` → Celda 5 "Implementar Decision Fusion"
2. Modificar función `compute_risk_score()`
3. Ajustar pesos α, β según necesidad

**Quiero cambiar niveles de cobertura:**
1. Ir a `rq5.ipynb` → Celda 1 "Configuración"
2. Modificar `CONFIG['coverage_levels']`
3. Ejemplo: `[100, 90, 80, 70, 60, 50]`

**Quiero añadir nuevas métricas:**
1. Ir a `rq5.ipynb` → Celda 7 "FP Reduction"
2. Añadir cálculos de nuevas métricas (Precision, Recall, F1)
3. Actualizar Table 5.2 con nuevas columnas

**Quiero personalizar visualizaciones:**
1. Ir a `rq5.ipynb` → Celdas 8-9 "Visualizaciones"
2. Modificar colores, tamaños, estilos con matplotlib/seaborn
3. Guardar con nuevos nombres de archivo

---

### 👨‍🏫 Para Revisor / Tutor

**Quiero verificar metodología:**
1. 📄 `README.md` - Sección "Metodología"
2. 🏗️ `ARQUITECTURA_TECNICA.md` - Sección "Algoritmos Clave"
3. 📓 `rq5.ipynb` - Revisar celdas con "EJECUTAR PARA RQ5"

**Quiero verificar resultados:**
1. 📊 `RESUMEN_EJECUTIVO.md` - Sección "Resultados Principales"
2. 📁 `outputs/` - Revisar tablas y figuras generadas
3. 📝 `outputs/RQ5_FINAL_REPORT.txt` - Reporte completo

**Quiero verificar reproducibilidad:**
1. ⚙️ `rq5.ipynb` - Verificar seed=42 en configuración
2. 📋 `INSTRUCCIONES_EJECUCION.md` - Verificar prerrequisitos claros
3. 🏗️ `ARQUITECTURA_TECNICA.md` - Sección "Reproducibilidad"

---

## 📄 Descripción de Cada Archivo

### 1. `rq5.ipynb` ⭐ PRINCIPAL

**Propósito**: Notebook ejecutable con toda la implementación

**Secciones**:
1. Configuración e Imports
2. Cargar Resultados de Fase 5
3. Cargar Predicciones Detalladas
4. Implementar Decision Fusion
5. Evaluación de Selective Prediction
6. Análisis de False-Positive Reduction
7. Visualizaciones
8. Resumen y Exportación
9. Verificación de Resultados
10. Conclusiones

**Tiempo de ejecución**: ~15 minutos

**Outputs**: 12 archivos en `outputs/`

---

### 2. `README.md`

**Propósito**: Descripción general del proyecto RQ5

**Contenido**:
- 📋 Descripción y objetivos
- 🎯 Research question
- 📁 Estructura del notebook
- 🚀 Cómo ejecutar
- 📊 Resultados generados
- 🔑 Hallazgos clave
- 🎓 Contribución a la tesis

**Para quién**: Cualquiera que quiera entender qué hace RQ5

**Tiempo de lectura**: 5 minutos

---

### 3. `INSTRUCCIONES_EJECUCION.md`

**Propósito**: Guía detallada paso a paso para ejecutar el notebook

**Contenido**:
- ⚠️ Pre-requisitos
- 🚀 Pasos de ejecución (opción 1: todo, opción 2: paso a paso)
- 📊 Verificación de resultados
- 🔍 Interpretación de resultados
- ⚠️ Errores comunes y soluciones
- 📈 Valores esperados
- 🎯 Criterios de éxito

**Para quién**: Usuario que va a ejecutar el notebook por primera vez

**Tiempo de lectura**: 10 minutos

---

### 4. `RESUMEN_VISUAL.md`

**Propósito**: Visualización gráfica de la estructura y flujo del notebook

**Contenido**:
- 📋 Estructura del notebook (diagrama ASCII)
- 📊 Tablas 5.1 y 5.2 (preview)
- 🖼️ Figuras 5.1 y 5.2 (diagramas ASCII)
- 📁 Archivos generados
- 🔑 Hallazgos clave
- 📚 Contexto en la tesis

**Para quién**: Alguien que prefiere visualizaciones a texto largo

**Tiempo de lectura**: 7 minutos

---

### 5. `RESUMEN_EJECUTIVO.md`

**Propósito**: Consolidación de resultados y conclusiones

**Contenido**:
- 🎯 Objetivo alcanzado
- 📊 Resultados principales (tablas resumidas)
- 📁 Entregables completos
- 🔑 Hallazgos clave
- 🎓 Contribución a la tesis
- 📖 Metodología
- 🚀 Cómo ejecutar
- 🎯 Respuesta a RQ5
- ⚠️ Limitaciones

**Para quién**: Revisor que necesita resumen rápido

**Tiempo de lectura**: 5 minutos

---

### 6. `ARQUITECTURA_TECNICA.md`

**Propósito**: Documentación técnica detallada de implementación

**Contenido**:
- 📐 Diagrama de flujo de datos (completo)
- 🧮 Algoritmos clave (pseudocódigo + Python)
- 📊 Esquemas de datos (input/output)
- 🔧 Configuración técnica
- 🎨 Especificaciones de visualización
- ⚙️ Optimizaciones
- 🧪 Testing y validación
- 📈 Complejidad computacional
- 🔒 Reproducibilidad

**Para quién**: Desarrollador que necesita entender implementación

**Tiempo de lectura**: 15 minutos

---

### 7. `INDICE_DOCUMENTACION.md` (este archivo)

**Propósito**: Índice maestro de toda la documentación RQ5

**Contenido**:
- 📁 Estructura de archivos
- 📖 Guía de uso por rol
- 📄 Descripción de cada archivo
- 🔗 Referencias cruzadas
- 🎯 Mapa de navegación

**Para quién**: Punto de entrada a la documentación

**Tiempo de lectura**: 3 minutos

---

## 🔗 Referencias Cruzadas

### Desde el Notebook a la Documentación:

| Celda en Notebook | Ver Documentación |
|-------------------|-------------------|
| Celda 1 (Config) | `ARQUITECTURA_TECNICA.md` → Configuración |
| Celda 5 (Fusion) | `ARQUITECTURA_TECNICA.md` → Algoritmos |
| Celda 6 (Selective) | `README.md` → Resultados Esperados |
| Celda 7 (FP/FN) | `RESUMEN_VISUAL.md` → Tabla 5.2 |
| Celdas 8-9 (Figs) | `RESUMEN_VISUAL.md` → Figuras |

### Desde la Documentación al Notebook:

| Documentación | Celda en Notebook |
|---------------|-------------------|
| `README.md` → "Cómo Ejecutar" | Ejecutar todo el notebook |
| `INSTRUCCIONES_EJECUCION.md` → "Paso 3" | Celda 5 |
| `ARQUITECTURA_TECNICA.md` → "Algoritmo Risk" | Celda 5 |
| `RESUMEN_VISUAL.md` → "Tabla 5.1" | Celda 6 |

---

## 🗺️ Mapa de Navegación

```
┌─────────────────────────────────────────────────────────┐
│              INICIO: ¿Qué necesitas?                    │
└─────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│ Entender      │   │ Ejecutar      │   │ Modificar     │
│ qué hace      │   │ notebook      │   │ código        │
└───────┬───────┘   └───────┬───────┘   └───────┬───────┘
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│ README.md     │   │INSTRUCCIONES  │   │ARQUITECTURA   │
│ RESUMEN_VISUAL│   │_EJECUCION.md  │   │_TECNICA.md    │
└───────┬───────┘   └───────┬───────┘   └───────┬───────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │  rq5.ipynb    │
                    │  (EJECUTAR)   │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │  outputs/     │
                    │  (VERIFICAR)  │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ RESUMEN       │
                    │ _EJECUTIVO.md │
                    └───────────────┘
```

---

## 📌 Rutas de Aprendizaje Sugeridas

### Ruta 1: Rápida (15 minutos)

1. Leer `RESUMEN_EJECUTIVO.md` (5 min)
2. Ejecutar `rq5.ipynb` completo (15 min esperando)
3. Ver outputs en `outputs/` (2 min)

**Total**: ~20 minutos → Tienes resultados

---

### Ruta 2: Completa (1 hora)

1. Leer `README.md` (5 min)
2. Leer `RESUMEN_VISUAL.md` (7 min)
3. Leer `INSTRUCCIONES_EJECUCION.md` (10 min)
4. Ejecutar `rq5.ipynb` paso a paso (30 min)
5. Leer `RESUMEN_EJECUTIVO.md` (5 min)
6. Revisar outputs (5 min)

**Total**: ~1 hora → Entiendes todo

---

### Ruta 3: Técnica (2 horas)

1. Leer `README.md` (5 min)
2. Leer `ARQUITECTURA_TECNICA.md` completo (20 min)
3. Revisar código en `rq5.ipynb` celda por celda (45 min)
4. Ejecutar y verificar (15 min)
5. Experimentar con parámetros (30 min)

**Total**: ~2 horas → Puedes modificar

---

## 🎯 Checklist de Uso

### ✅ Antes de Ejecutar:

- [ ] Leí `README.md` para entender objetivo
- [ ] Verifiqué prerrequisitos (Fase 3, 4, 5 completadas)
- [ ] Tengo ~15 minutos disponibles
- [ ] Carpeta `outputs/` está lista

### ✅ Durante la Ejecución:

- [ ] Sigo instrucciones de `INSTRUCCIONES_EJECUCION.md`
- [ ] Ejecuto celdas en orden
- [ ] No interrumpo el proceso
- [ ] Verifico outputs de cada celda

### ✅ Después de Ejecutar:

- [ ] Verifico que `outputs/` tiene 12 archivos
- [ ] Reviso tablas 5.1 y 5.2
- [ ] Reviso figuras 5.1 y 5.2
- [ ] Leo `RQ5_FINAL_REPORT.txt`
- [ ] Comparo con resultados esperados

---

## 📞 Soporte y Ayuda

### Problema: No entiendo qué hace RQ5
**Solución**: Leer `README.md` y `RESUMEN_VISUAL.md`

### Problema: No sé cómo ejecutar
**Solución**: Seguir `INSTRUCCIONES_EJECUCION.md` paso a paso

### Problema: Errores al ejecutar
**Solución**: Ver sección "Errores Comunes" en `INSTRUCCIONES_EJECUCION.md`

### Problema: Quiero modificar algo
**Solución**: Leer `ARQUITECTURA_TECNICA.md` para entender implementación

### Problema: Necesito más contexto de tesis
**Solución**: Leer `../../rq_no5.md` y `../../rqq.md`

---

## 🔗 Enlaces Externos

### Documentación del Proyecto:

- `../../rq_no5.md` - Documentación completa de RQ5
- `../../rqq.md` - Índice completo de la tesis
- `../../RESUMEN_INDICE_TESIS.md` - Resumen ejecutivo de tesis

### Fase 5 (Dependencia):

- `../../fase 5/main.ipynb` - Notebook de Fase 5
- `../../fase 5/outputs/comparison/` - Datos de entrada

### Fases Anteriores:

- `../../fase 3/main.ipynb` - MC-Dropout
- `../../fase 4/main.ipynb` - Temperature Scaling

---

## 📊 Métricas de Documentación

| Archivo | Líneas | Palabras | Tiempo Lectura |
|---------|--------|----------|----------------|
| README.md | 250 | 1,800 | 5 min |
| INSTRUCCIONES_EJECUCION.md | 400 | 2,800 | 10 min |
| RESUMEN_VISUAL.md | 500 | 3,500 | 7 min |
| RESUMEN_EJECUTIVO.md | 300 | 2,100 | 5 min |
| ARQUITECTURA_TECNICA.md | 700 | 4,900 | 15 min |
| INDICE_DOCUMENTACION.md | 400 | 2,800 | 3 min |

**Total**: ~2,550 líneas, ~17,900 palabras

---

## ✅ Estado de Documentación

| Componente | Estado | Fecha |
|------------|--------|-------|
| Notebook (`rq5.ipynb`) | ✅ Completo | Ene 2026 |
| README | ✅ Completo | Ene 2026 |
| Instrucciones | ✅ Completo | Ene 2026 |
| Resumen Visual | ✅ Completo | Ene 2026 |
| Resumen Ejecutivo | ✅ Completo | Ene 2026 |
| Arquitectura Técnica | ✅ Completo | Ene 2026 |
| Índice (este doc) | ✅ Completo | Ene 2026 |

---

**✅ Documentación RQ5 — 100% Completa**

**📚 Total: 7 archivos de documentación + 1 notebook ejecutable**

**🎯 Cobertura: Desde usuario principiante hasta desarrollador avanzado**
