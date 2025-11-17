# Lanzador de Demo - Fase 6
Write-Host "🚀 Iniciando Demo OVD con Calibración e Incertidumbre..." -ForegroundColor Cyan

# Verificar si Streamlit está instalado
$streamlitInstalled = python -m pip show streamlit 2>$null
if (-not $streamlitInstalled) {
    Write-Host "⚠️  Streamlit no instalado, instalando..." -ForegroundColor Yellow
    python -m pip install streamlit streamlit-option-menu plotly -q
}

# Verificar archivos necesarios
if (-not (Test-Path "app/demo.py")) {
    Write-Host "❌ Error: app/demo.py no encontrado" -ForegroundColor Red
    Write-Host "   Ejecuta primero las celdas del notebook main.ipynb" -ForegroundColor Yellow
    exit 1
}

# Lanzar aplicación
Write-Host "✅ Abriendo aplicación en navegador..." -ForegroundColor Green
streamlit run app/demo.py
