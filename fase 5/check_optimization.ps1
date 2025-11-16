# Script de verificación de optimización - Fase 5
# ================================================

Write-Host "`n" -NoNewline
Write-Host "=" -ForegroundColor Blue -NoNewline
Write-Host ("="*69) -ForegroundColor Blue
Write-Host "VERIFICACIÓN RÁPIDA DE OPTIMIZACIÓN - FASE 5" -ForegroundColor Cyan
Write-Host "=" -ForegroundColor Blue -NoNewline
Write-Host ("="*69) -ForegroundColor Blue
Write-Host ""

# Verificar si Python está disponible
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Python encontrado: " -ForegroundColor Green -NoNewline
    Write-Host $pythonVersion
} catch {
    Write-Host "✗ Python no encontrado en PATH" -ForegroundColor Red
    Write-Host "  Por favor instala Python o agrega al PATH" -ForegroundColor Yellow
    exit 1
}

Write-Host ""

# Verificar si existe el script de verificación
if (Test-Path "verify_optimization.py") {
    Write-Host "✓ Script de verificación encontrado" -ForegroundColor Green
} else {
    Write-Host "✗ verify_optimization.py no encontrado" -ForegroundColor Red
    Write-Host "  Asegúrate de estar en el directorio fase 5/" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "Ejecutando verificación..." -ForegroundColor Cyan
Write-Host ("-"*70) -ForegroundColor Gray
Write-Host ""

# Ejecutar el script de Python
python verify_optimization.py

$exitCode = $LASTEXITCODE

Write-Host ""
Write-Host ("-"*70) -ForegroundColor Gray
Write-Host ""

# Mostrar siguiente paso basado en el resultado
if ($exitCode -eq 0) {
    Write-Host "SIGUIENTE PASO:" -ForegroundColor Green
    Write-Host "  Ejecuta el notebook optimizado:" -ForegroundColor White
    Write-Host "    jupyter notebook main.ipynb" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  Tiempo estimado: ~15-20 minutos ⚡" -ForegroundColor Green
} else {
    Write-Host "SIGUIENTE PASO:" -ForegroundColor Yellow
    Write-Host "  Opción A - Ejecutar fases anteriores primero (recomendado):" -ForegroundColor White
    Write-Host "    cd '../fase 2'" -ForegroundColor Cyan
    Write-Host "    jupyter notebook main.ipynb" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  Opción B - Ejecutar Fase 5 sin optimización:" -ForegroundColor White
    Write-Host "    jupyter notebook main.ipynb" -ForegroundColor Cyan
    Write-Host "    (Tiempo estimado: ~2 horas)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=" -ForegroundColor Blue -NoNewline
Write-Host ("="*69) -ForegroundColor Blue
Write-Host ""

exit $exitCode
