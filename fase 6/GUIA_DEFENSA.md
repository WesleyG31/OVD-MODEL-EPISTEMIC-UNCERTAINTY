# 🎓 Guía para la Defensa - Demo Fase 6

## 📋 Preparación Pre-Defensa

### 1. Verificar Sistema (Día antes)
```bash
cd "fase 6"
# Ejecutar todas las celdas del notebook main.ipynb
# Verificar que la celda 7 muestre ✅ SISTEMA LISTO
```

### 2. Seleccionar Casos Clave (30 min antes)

**Caso 1: Escena Fácil**
- Pocos objetos, buena visibilidad
- Objetivo: Mostrar que el sistema funciona bien en condiciones óptimas
- Comparar: Baseline vs Baseline+TS (cambios sutiles en confianza)

**Caso 2: Escena Media**
- Tráfico normal, múltiples objetos
- Objetivo: Mostrar utilidad de la incertidumbre
- Comparar: Baseline+TS vs MC-Dropout+TS (detecciones con alta/baja incertidumbre)

**Caso 3: Escena Difícil**
- Muchos objetos, oclusión, condiciones adversas
- Objetivo: Demostrar cuándo el sistema necesita ayuda
- Mostrar: Filtrado por umbral de incertidumbre

### 3. Tener Listo
- ✅ Demo abierta y funcionando
- ✅ 3 imágenes cargadas (easy, medium, hard)
- ✅ Capturas de pantalla en `outputs/screenshots/`
- ✅ Métricas globales visibles en sidebar

---

## 🎯 Narrativa para la Presentación

### Introducción (1 min)
> "Voy a mostrar una demo interactiva que integra todo el trabajo:
> - Detección Open-Vocabulary en escenas ADAS
> - Calibración para probabilidades honestas
> - Incertidumbre epistémica para decisiones seguras"

### Demostración 1: Calibración (2-3 min)

**Mostrar**: Baseline vs Baseline + TS

**Script**:
1. "Aquí vemos la misma imagen procesada dos veces"
2. "A la izquierda: probabilidades originales"
3. "A la derecha: después de calibración (Temperature Scaling)"
4. **Señalar detección específica**: "Este coche tenía 0.92 de confianza, pero el modelo solo acierta ~75% a ese nivel. Con TS se corrige a 0.76, más honesto"
5. **Mostrar métricas**: "Globalmente, ECE baja de X a Y (mejor calibración)"

**Mensaje clave**: *"La calibración no mejora la detección, pero hace que las probabilidades reflejen la realidad"*

### Demostración 2: Incertidumbre (2-3 min)

**Mostrar**: Baseline + TS vs MC-Dropout + TS

**Script**:
1. "Ahora activamos MC-Dropout para estimar incertidumbre"
2. **Señalar etiquetas**: "Cada detección tiene un nivel: LOW/MED/HIGH"
3. **Mostrar histograma**: "Vemos la distribución de incertidumbre en esta escena"
4. **Ajustar umbral**: "Si filtramos solo las de baja incertidumbre..."
5. "Eliminamos X detecciones dudosas, nos quedamos con las confiables"

**Mensaje clave**: *"La incertidumbre nos dice cuándo el modelo está inseguro, crítico para ADAS"*

### Demostración 3: Caso Difícil (2 min)

**Mostrar**: Escena compleja con filtrado agresivo

**Script**:
1. "En esta escena difícil [noche/lluvia/ciudad], hay Y detecciones"
2. "Z tienen alta incertidumbre"
3. **Filtrar**: "Si el sistema ADAS solo toma decisiones con baja incertidumbre..."
4. "Se queda con N detecciones seguras, evita errores costosos"
5. **Mostrar métrica**: "Esto reduce FP en X%, manteniendo Y% de TP"

**Mensaje clave**: *"Trade-off explícito: cobertura vs riesgo, el sistema decide según contexto"*

---

## 💡 Respuestas a Preguntas Frecuentes

### "¿Cuánto cuesta computacionalmente?"
- Baseline: ~200ms/imagen
- MC-Dropout K=5: ~1s/imagen (5x más lento)
- Varianza decoder: ~250ms (intermedio)
- **Respuesta**: "Para un vehículo a 30 km/h procesando a 10 FPS, MC-Dropout es viable. Para 120 km/h, usaríamos varianza decoder (single-pass)"

### "¿Por qué no usar simplemente un umbral de confianza?"
- Mostrar caso donde score alto pero alta incertidumbre
- "La confianza dice 'qué tan seguro estoy de esta clase', la incertidumbre dice 'qué tan inconsistente es el modelo internamente'. Son ortogonales"

### "¿Qué pasa si la incertidumbre está mal calibrada?"
- "La incertidumbre es relativa, no absoluta. Lo importante es el ranking: las detecciones de alta incertidumbre son estadísticamente más propensas a ser FP"
- Mostrar curvas risk-coverage de Fase 5

### "¿Cómo se integraría en un sistema real?"
1. **Pipeline**: Detección → Calibración → Cálculo de incertidumbre → Decisión
2. **Modos**:
   - **Modo autopista**: Umbral bajo (solo muy confiables)
   - **Modo ciudad**: Umbral medio (balance)
   - **Modo asistido**: Todas las detecciones, UI resalta inciertas
3. **Fallback**: Si toda la escena tiene alta incertidumbre → alerta al conductor

---

## 📊 Métricas a Mencionar

**Sin entrar en detalles técnicos excesivos**:
- "Reducimos ECE de X a Y (mejor calibración)"
- "AUROC TP/FP de Z (incertidumbre discrimina errores)"
- "mAP se mantiene (no perdemos detección)"

**Énfasis**: "No sacrificamos rendimiento, agregamos confiabilidad"

---

## 🚨 Plan B (Si algo falla)

### Demo no carga
- Tener capturas de pantalla pre-generadas
- "Por tiempo, muestro capturas representativas"
- Explicar igual la narrativa

### Modelo es muy lento
- Usar imágenes pequeñas pre-procesadas
- Reducir K de MC-Dropout a 3
- Usar varianza decoder en lugar de MC-Dropout

### Sin conexión a GPU
- Demo funciona en CPU (más lento pero viable)
- Preparar ejemplos pre-calculados

---

## ✅ Checklist Final (10 min antes)

- [ ] Demo corriendo en `localhost:8501`
- [ ] 3 casos cargados (easy/medium/hard)
- [ ] Métricas visibles en sidebar
- [ ] Capturas de respaldo en carpeta
- [ ] Saber responder 3 preguntas clave
- [ ] Tiempo cronometrado: 5-7 min total
- [ ] Mensaje final preparado

---

## 🎬 Cierre de la Demo

> "En resumen:
> 1. La calibración hace que las probabilidades sean honestas
> 2. La incertidumbre identifica cuándo el modelo necesita ayuda
> 3. Juntas, permiten decisiones más seguras en sistemas críticos como ADAS
> 
> Esta demo es una prueba de concepto, pero los principios aplican a cualquier modelo de detección en aplicaciones de seguridad"

**Transición**: "Ahora paso a las conclusiones y trabajo futuro..."

---

## 🎯 Tiempo Sugerido

| Sección | Tiempo |
|---------|--------|
| Introducción | 1 min |
| Demo 1: Calibración | 2 min |
| Demo 2: Incertidumbre | 2 min |
| Demo 3: Caso difícil | 1 min |
| Preguntas/discusión | 2-3 min |
| **TOTAL** | **7-9 min** |

Reservar último 25% del tiempo para preguntas.
