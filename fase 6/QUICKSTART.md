# 🚀 Quick Start - Demo Fase 6

## Ejecución Rápida (5 minutos)

### Paso 1: Preparar Sistema
```bash
cd "fase 6"
jupyter notebook main.ipynb
```

Ejecutar celdas en orden:
1. ✅ Celda 1: Instalar Streamlit
2. ✅ Celda 2: Crear app/demo.py
3. ✅ Celda 3: Preparar imágenes de muestra

### Paso 2: Lanzar Demo

**Windows**:
```powershell
.\launch_demo.ps1
```

**Linux/Mac**:
```bash
./launch_demo.sh
```

**O manualmente**:
```bash
streamlit run app/demo.py
```

### Paso 3: Usar Demo

1. **Abrir navegador**: `http://localhost:8501`
2. **Seleccionar método**: Sidebar → "MC-Dropout K=5 + TS"
3. **Cargar imagen**: Usar "imagen de muestra" → easy_*.jpg
4. **Ejecutar**: Click en "🚀 Ejecutar Detección"
5. **Explorar**: Ajustar umbrales, ver tabla, histograma

---

## Estructura de Archivos

```
fase 6/
├── main.ipynb              # Notebook principal (ejecutar todo)
├── README.md               # Documentación completa
├── GUIA_DEFENSA.md         # Para presentación
├── REPORTE_FINAL_FASE6.md  # Reporte técnico
├── launch_demo.ps1         # Lanzador Windows
├── launch_demo.sh          # Lanzador Linux/Mac
├── app/
│   ├── demo.py            # Aplicación Streamlit
│   └── samples/           # 9 imágenes de prueba
└── outputs/
    └── screenshots/       # Capturas comparativas
```

---

## Verificación Rápida

```python
# En notebook, ejecutar celda 7
# Debe mostrar:
# ✅ SISTEMA LISTO PARA DEMO
```

---

## Troubleshooting

| Problema | Solución |
|----------|----------|
| "app/demo.py not found" | Ejecutar celda 2 del notebook |
| "No samples available" | Ejecutar celda 3 del notebook |
| "Streamlit not installed" | Ejecutar celda 1 del notebook |
| Demo muy lento | Usar "Varianza Decoder" en vez de MC-Dropout |

---

## Demo en 30 Segundos

1. Ejecutar: `streamlit run app/demo.py`
2. Seleccionar: "Baseline + TS"
3. Click: "🚀 Ejecutar Detección"
4. Ver: Cajas coloreadas con confianza calibrada
5. Cambiar a: "MC-Dropout K=5 + TS"
6. Comparar: Ahora con incertidumbre (LOW/MED/HIGH)

---

## Para la Defensa

**Casos recomendados**:
1. **easy_*.jpg** → Mostrar que funciona bien
2. **medium_*.jpg** → Mostrar utilidad de incertidumbre
3. **hard_*.jpg** → Mostrar filtrado por umbral

**Timing**: 7-9 minutos total

Ver `GUIA_DEFENSA.md` para narrativa completa.

---

## Soporte

- 📖 Documentación completa: `README.md`
- 🎓 Guía de defensa: `GUIA_DEFENSA.md`
- 📊 Reporte técnico: `REPORTE_FINAL_FASE6.md`

**Todo listo para ejecutar y presentar** ✅
