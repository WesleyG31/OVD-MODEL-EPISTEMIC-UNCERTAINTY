#!/bin/bash
# Lanzador de Demo - Fase 6

echo "🚀 Iniciando Demo OVD con Calibración e Incertidumbre..."

# Verificar si Streamlit está instalado
if ! python -m pip show streamlit &> /dev/null; then
    echo "⚠️  Streamlit no instalado, instalando..."
    python -m pip install streamlit streamlit-option-menu plotly -q
fi

# Verificar archivos necesarios
if [ ! -f "app/demo.py" ]; then
    echo "❌ Error: app/demo.py no encontrado"
    echo "   Ejecuta primero las celdas del notebook main.ipynb"
    exit 1
fi

# Lanzar aplicación
echo "✅ Abriendo aplicación en navegador..."
streamlit run app/demo.py
