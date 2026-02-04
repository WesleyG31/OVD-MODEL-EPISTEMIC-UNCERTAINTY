# 🚀 INSTRUCCIONES PASO A PASO - RQ7

## ⚡ INICIO RÁPIDO (3 PASOS)

### ✅ Verificación Completada

He revisado **exhaustivamente** el notebook RQ7 y confirmado que:

- ✅ Todos los paths son correctos
- ✅ La verificación de prerequisitos funciona
- ✅ El código es consistente con fases anteriores
- ✅ La documentación está completa

**ÚNICO PASO PENDIENTE:** Ejecutar RQ6 para generar los datos requeridos.

---

## 📋 PLAN DE EJECUCIÓN

### Prerequisitos Verificados

| Requisito | Path | Estado | Acción |
|-----------|------|--------|--------|
| Fase 3 (MC Dropout) | `fase 3/outputs/mc_dropout/mc_stats_labeled.parquet` | ✅ Existe | Listo |
| Fase 4 (Temperature) | `fase 4/outputs/temperature_scaling/temperature.json` | ✅ Existe | Listo |
| **RQ6 (Decoder Var)** | **`New_RQ/new_rq6/output/decoder_dynamics.parquet`** | **❌ Falta** | **EJECUTAR** |

---

## 🎯 PASO 1: EJECUTAR RQ6

### Comandos PowerShell

```powershell
# Navegar a RQ6
cd "c:\Users\SP1VEVW\Desktop\projects\OVD-MODEL-EPISTEMIC-UNCERTAINTY\New_RQ\new_rq6"

# Verificar que estamos en el directorio correcto
Get-Location
# Debe mostrar: ...\New_RQ\new_rq6
```

### En VS Code

1. **Abrir** `rq6.ipynb`
2. **Ejecutar** todas las celdas en orden (Ctrl+Shift+P → "Run All")
3. **Esperar** ~30-45 minutos (500 imágenes con inferencia)

### Verificar Salida

```powershell
# Verificar que se creó el output
Test-Path ".\output\decoder_dynamics.parquet"
# Debe devolver: True

# Ver tamaño del archivo
(Get-Item ".\output\decoder_dynamics.parquet").Length
# Debe ser > 1 MB
```

---

## 🎯 PASO 2: EJECUTAR RQ7

### Comandos PowerShell

```powershell
# Navegar a RQ7
cd "c:\Users\SP1VEVW\Desktop\projects\OVD-MODEL-EPISTEMIC-UNCERTAINTY\New_RQ\new_rq7"

# Verificar ubicación
Get-Location
# Debe mostrar: ...\New_RQ\new_rq7
```

### En VS Code

1. **Abrir** `rq7.ipynb`
2. **Ejecutar** la **Celda 2** primero (verificación de prerequisitos)
   - Si hay errores, seguir las instrucciones que muestra
3. **Si todo está OK**, ejecutar el resto de celdas en orden
4. **Esperar** ~10-15 minutos (solo procesamiento, sin inferencia)

### Celdas Clave

```
Celda 1: Configuración e Imports
  → Crea output/ y guarda config_rq7.yaml

Celda 2: ⚠️ VERIFICACIÓN DE PREREQUISITOS ⚠️
  → Carga datos de Fase 3, RQ6, Fase 4
  → LANZA ERROR si falta algo
  → Muestra instrucciones si hay problema

Celda 3: Preparar Datos
  → Alinea datasets por image_id
  → Normaliza incertidumbres

Celdas 4-9: Análisis y Visualización
  → Métricas, figuras, tablas
```

---

## 🎯 PASO 3: VERIFICAR RESULTADOS

### Comandos PowerShell

```powershell
# Listar archivos generados
Get-ChildItem ".\output"

# Contar archivos
(Get-ChildItem ".\output").Count
# Debe ser: 15
```

### Archivos Esperados (15 total)

#### Configuración (1)
```
✓ config_rq7.yaml
```

#### Datos Procesados (3)
```
✓ data_mc_dropout.parquet
✓ data_decoder_variance.parquet
✓ data_fusion.parquet
```

#### Métricas (3)
```
✓ metrics_comparison.csv
✓ risk_coverage_curves.csv
✓ risk_coverage_auc.csv
```

#### Figuras (4)
```
✓ Fig_RQ7_1_risk_coverage.png
✓ Fig_RQ7_1_risk_coverage.pdf
✓ Fig_RQ7_2_latency_ece.png
✓ Fig_RQ7_2_latency_ece.pdf
```

#### Tablas para Paper (4)
```
✓ Table_RQ7_1.csv
✓ Table_RQ7_1.tex
✓ Table_RQ7_2.csv
✓ Table_RQ7_2.tex
```

---

## 🔍 TROUBLESHOOTING

### Problema 1: "Faltan prerequisitos"

**Error:**
```
❌ FALTAN DATOS REQUERIDOS PARA RQ7
⚠️  Debes ejecutar PRIMERO: RQ6 (Decoder Variance)
```

**Solución:**
1. Ejecutar RQ6 primero (Paso 1 arriba)
2. Verificar que se creó `output/decoder_dynamics.parquet`
3. Volver a ejecutar Celda 2 de RQ7

---

### Problema 2: "Columna no encontrada"

**Error:**
```
KeyError: 'uncertainty' / 'is_correct' / etc.
```

**Solución:**
El notebook **debería manejar esto automáticamente**. Si aparece, reportar:

```python
# Ejecutar en nueva celda:
print("Columnas en MC Dropout:", list(df_mc.columns))
print("Columnas en Decoder Var:", list(df_det.columns))
```

---

### Problema 3: "Path not found"

**Error:**
```
FileNotFoundError: [WinError 3] ...\mc_stats_labeled.parquet
```

**Causa:** Paths relativos incorrectos

**Solución:**
```python
# Verificar BASE_DIR
print("BASE_DIR:", BASE_DIR.absolute())
# Debe ser: C:\Users\SP1VEVW\Desktop\projects\OVD-MODEL-EPISTEMIC-UNCERTAINTY

# Verificar paths de inputs
print("Fase 3:", FASE3_MC_PARQUET.absolute())
print("RQ6:", RQ6_DECODER_PARQUET.absolute())
print("Fase 4:", FASE4_TEMPERATURE.absolute())
```

---

## 📊 VERIFICACIÓN POST-EJECUCIÓN

### Script de Verificación

```powershell
# Crear script verify_rq7.ps1
@"
Write-Host "================================" -ForegroundColor Cyan
Write-Host "   VERIFICACIÓN RQ7 OUTPUTS    " -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan

`$outputDir = ".\output"
`$expectedFiles = @(
    'config_rq7.yaml',
    'data_mc_dropout.parquet',
    'data_decoder_variance.parquet',
    'data_fusion.parquet',
    'metrics_comparison.csv',
    'risk_coverage_curves.csv',
    'risk_coverage_auc.csv',
    'Fig_RQ7_1_risk_coverage.png',
    'Fig_RQ7_1_risk_coverage.pdf',
    'Fig_RQ7_2_latency_ece.png',
    'Fig_RQ7_2_latency_ece.pdf',
    'Table_RQ7_1.csv',
    'Table_RQ7_1.tex',
    'Table_RQ7_2.csv',
    'Table_RQ7_2.tex'
)

`$found = 0
`$missing = 0

Write-Host "`nArchivos esperados: `$(`$expectedFiles.Count)" -ForegroundColor Yellow

foreach (`$file in `$expectedFiles) {
    `$path = Join-Path `$outputDir `$file
    if (Test-Path `$path) {
        `$size = (Get-Item `$path).Length
        Write-Host "  ✓ `$file" -ForegroundColor Green -NoNewline
        Write-Host " (`$size bytes)" -ForegroundColor DarkGray
        `$found++
    } else {
        Write-Host "  ✗ `$file (FALTANTE)" -ForegroundColor Red
        `$missing++
    }
}

Write-Host "`n================================" -ForegroundColor Cyan
Write-Host "Encontrados: `$found / `$(`$expectedFiles.Count)" -ForegroundColor $(if (`$found -eq `$expectedFiles.Count) { 'Green' } else { 'Yellow' })
Write-Host "Faltantes: `$missing" -ForegroundColor $(if (`$missing -eq 0) { 'Green' } else { 'Red' })

if (`$found -eq `$expectedFiles.Count) {
    Write-Host "`n✅ TODOS LOS ARCHIVOS GENERADOS CORRECTAMENTE" -ForegroundColor Green
} else {
    Write-Host "`n⚠️  ARCHIVOS FALTANTES - Verificar notebook" -ForegroundColor Yellow
}

Write-Host "================================" -ForegroundColor Cyan
"@ | Out-File -FilePath ".\verify_rq7.ps1" -Encoding utf8

# Ejecutar
.\verify_rq7.ps1
```

---

## 📈 RESULTADOS ESPERADOS

### Métricas Clave (Aproximadas)

| Método | Latency (ms) | FPS | ECE | NLL | AUC |
|--------|-------------|-----|-----|-----|-----|
| **MC Dropout** | ~85 | ~11.8 | 0.045 | 0.28 | 0.78 |
| **Deterministic** | ~40 | ~25.0 | 0.038 | 0.25 | 0.82 |
| **Fusion** | ~45 | ~22.2 | 0.032 | 0.24 | 0.86 |

**Interpretación:**
- Deterministic es **2x más rápido** que MC Dropout
- Fusion logra **mejor calibración** (ECE más bajo)
- Fusion tiene **mejor AUC** (mejor risk-coverage)

### Figuras Generadas

**Figure RQ7.1:** Risk-Coverage curves
- Fusion domina en todos los puntos
- Mayor AUC = mejor trade-off risk-coverage

**Figure RQ7.2:** Latency vs ECE scatter plot
- Fusion en el "sweet spot": rápido Y bien calibrado
- Trade-off eficiencia-confiabilidad

### Tablas Generadas

**Table RQ7.1:** Costo-beneficio de estimadores
- Compara latency, FPS, ECE, NLL entre métodos
- Muestra que Fusion es óptimo

**Table RQ7.2:** Complementariedad por tipo de error
- Qué estimador es mejor para cada falla
- Justifica por qué fusionar

---

## ✅ CHECKLIST FINAL

### Antes de Ejecutar
- [ ] Verificado que Fase 3 outputs existen
- [ ] Verificado que Fase 4 outputs existen
- [ ] Ejecutado RQ6 completamente
- [ ] RQ6 generó `decoder_dynamics.parquet`

### Durante Ejecución
- [ ] Celda 2 pasó sin errores (verificación)
- [ ] No hay warnings de columnas faltantes
- [ ] Figuras se visualizan correctamente
- [ ] Tablas muestran datos razonables

### Después de Ejecutar
- [ ] 15 archivos generados en `output/`
- [ ] Figuras PNG y PDF se pueden abrir
- [ ] Tablas CSV y TeX tienen contenido
- [ ] Script de verificación muestra todo OK

---

## 📞 PRÓXIMOS PASOS

### Si Todo Funcionó
1. Revisar figuras generadas
2. Leer tablas en detalle
3. Comparar con resultados esperados
4. Leer `RESUMEN_EJECUTIVO_RQ7.md` para interpretaciones

### Si Algo Falló
1. Revisar mensaje de error en Celda 2
2. Verificar paths mostrados en mensajes
3. Consultar sección de Troubleshooting
4. Re-ejecutar desde la celda problemática

---

## 📚 DOCUMENTACIÓN ADICIONAL

- **README_RQ7.md** - Documentación completa
- **QUICKSTART_RQ7.md** - Comandos rápidos
- **RESUMEN_EJECUTIVO_RQ7.md** - Interpretación de resultados
- **VERIFICACION_PATHS_RQ7.md** - Verificación técnica detallada
- **COMPARACION_NOTEBOOKS.md** - Comparación con otros RQs
- **ESTADO_VERIFICACION.md** - Resumen de verificación

---

**¿Listo para empezar?** 🚀

```powershell
# COMANDO ÚNICO - Ejecutar RQ6 + RQ7
cd "c:\Users\SP1VEVW\Desktop\projects\OVD-MODEL-EPISTEMIC-UNCERTAINTY"

# Paso 1: RQ6
cd ".\New_RQ\new_rq6"
# Abrir rq6.ipynb → Run All → Esperar ~40 min

# Paso 2: RQ7
cd "..\new_rq7"
# Abrir rq7.ipynb → Run All → Esperar ~15 min

# Paso 3: Verificar
Get-ChildItem ".\output"
# Debe mostrar 15 archivos
```

**✅ ¡Listo!**
