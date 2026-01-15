"""
Script de Verificación Rápida: Datos Reales en Fase 5

Este script verifica si los archivos JSON de la Fase 5 contienen
las incertidumbres reales por capa del decoder.

Uso:
    python verificar_datos_reales.py
"""

import json
import numpy as np
from pathlib import Path


def verificar_datos_reales():
    """Verifica que los datos de Fase 5 contengan layer_uncertainties"""

    print("=" * 80)
    print("VERIFICACIÓN DE DATOS REALES PARA RQ1")
    print("=" * 80)

    # Ruta al archivo
    ruta_base = Path(
        __file__
    ).parent.parent.parent  # Sube 3 niveles: rq1 -> rq -> RQ -> base
    ruta_json = (
        ruta_base / "fase 5" / "outputs" / "comparison" / "eval_decoder_variance.json"
    )

    print(f"\n📂 Ruta del archivo:")
    print(f"   {ruta_json}")

    # Verificar que existe
    if not ruta_json.exists():
        print(f"\n❌ ERROR: Archivo no encontrado")
        print(f"   → Debes ejecutar la Fase 5 primero")
        return False

    # Verificar tamaño
    tamanio_mb = ruta_json.stat().st_size / (1024 * 1024)
    print(f"\n📊 Tamaño del archivo: {tamanio_mb:.2f} MB")

    if tamanio_mb < 0.5:
        print(f"   ⚠️  Archivo parece muy pequeño, puede estar vacío o incompleto")

    # Cargar datos
    print(f"\n⏳ Cargando datos...")
    try:
        with open(ruta_json, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"\n❌ ERROR al cargar JSON: {e}")
        return False

    print(f"✅ Datos cargados: {len(data)} predicciones")

    # Verificar estructura
    print(f"\n🔑 Campos disponibles en la primera predicción:")
    for key in data[0].keys():
        tipo = type(data[0][key]).__name__
        if isinstance(data[0][key], list):
            tipo = f"list (len={len(data[0][key])})"
        print(f"   - {key:25s} : {tipo}")

    # Verificar layer_uncertainties
    print(f"\n🔍 Verificando 'layer_uncertainties'...")

    if "layer_uncertainties" not in data[0]:
        print(f"\n❌ ERROR: Campo 'layer_uncertainties' NO ENCONTRADO")
        print(f"\n💡 Solución:")
        print(f"   1. Abre: fase 5/main.ipynb")
        print(f"   2. Verifica que la función inference_decoder_variance incluya:")
        print(f"      'layer_uncertainties': layer_scores")
        print(f"   3. Re-ejecuta las celdas de decoder_variance")
        print(f"   4. Verifica que el archivo JSON se actualizó (fecha reciente)")
        return False

    # Contar predicciones con layer_uncertainties
    con_layers = [
        p
        for p in data
        if "layer_uncertainties" in p and len(p.get("layer_uncertainties", [])) > 0
    ]
    porcentaje = len(con_layers) / len(data) * 100 if data else 0

    print(f"✅ Campo 'layer_uncertainties' ENCONTRADO")
    print(f"   - Total predicciones: {len(data)}")
    print(f"   - Con layer_uncertainties: {len(con_layers)} ({porcentaje:.1f}%)")

    if len(con_layers) == 0:
        print(f"\n❌ ERROR: Todas las predicciones tienen layer_uncertainties vacío")
        print(f"\n💡 Posibles causas:")
        print(f"   - Los hooks no capturaron las capas del decoder")
        print(f"   - Hay un error en la lógica de captura")
        print(f"   - El modelo no ejecutó las capas como se esperaba")
        print(f"\n💡 Solución:")
        print(f"   1. Agrega prints de debug en inference_decoder_variance:")
        print(f"      print(f'DEBUG: Capturados {{len(layer_logits)}} layers')")
        print(f"   2. Re-ejecuta y verifica que layer_logits no esté vacío")
        return False

    # Analizar layer_uncertainties
    num_layers_list = [len(p["layer_uncertainties"]) for p in con_layers]
    num_layers_promedio = np.mean(num_layers_list)
    num_layers_min = min(num_layers_list)
    num_layers_max = max(num_layers_list)

    print(f"\n📊 Estadísticas de layer_uncertainties:")
    print(f"   - Número promedio de capas: {num_layers_promedio:.1f}")
    print(f"   - Rango de capas: [{num_layers_min}, {num_layers_max}]")

    if num_layers_promedio < 4:
        print(
            f"   ⚠️  Advertencia: Pocas capas capturadas. Se esperaban ~6 capas del decoder"
        )

    # Ejemplos
    print(f"\n📝 Ejemplos de layer_uncertainties:")
    for i in range(min(3, len(con_layers))):
        lu = con_layers[i]["layer_uncertainties"]
        scores_str = [f"{x:.4f}" for x in lu]
        print(f"   {i+1}. {scores_str}")

    # Análisis de valores
    all_layer_unc = [unc for p in con_layers for unc in p["layer_uncertainties"]]

    print(f"\n📈 Análisis de valores:")
    print(f"   - Total de valores: {len(all_layer_unc)}")
    print(f"   - Rango: [{min(all_layer_unc):.6f}, {max(all_layer_unc):.6f}]")
    print(f"   - Promedio: {np.mean(all_layer_unc):.6f}")
    print(f"   - Desviación estándar: {np.std(all_layer_unc):.6f}")

    # Verificar que no son todos iguales
    valores_unicos = len(set(all_layer_unc))
    print(f"   - Valores únicos: {valores_unicos}")

    if valores_unicos < 10:
        print(
            f"   ⚠️  Advertencia: Muy pocos valores únicos. Los datos pueden no ser reales."
        )

    # Verificar patrón de refinamiento (capas tempranas > capas tardías)
    if len(con_layers) > 100:
        # Tomar muestra aleatoria
        muestra = np.random.choice(con_layers, size=100, replace=False)
        primeras_capas = np.mean(
            [
                p["layer_uncertainties"][0]
                for p in muestra
                if len(p["layer_uncertainties"]) >= 2
            ]
        )
        ultimas_capas = np.mean(
            [
                p["layer_uncertainties"][-1]
                for p in muestra
                if len(p["layer_uncertainties"]) >= 2
            ]
        )

        print(f"\n🔬 Patrón de refinamiento (muestra de 100 predicciones):")
        print(f"   - Promedio primera capa: {primeras_capas:.6f}")
        print(f"   - Promedio última capa: {ultimas_capas:.6f}")
        print(f"   - Diferencia: {(primeras_capas - ultimas_capas):.6f}")

        if ultimas_capas < primeras_capas:
            print(
                f"   ✅ Patrón esperado: Las capas tardías tienen menor incertidumbre (más refinadas)"
            )
        else:
            print(
                f"   ⚠️  Patrón inesperado: Las capas tardías tienen mayor incertidumbre"
            )

    # Verificar uncertainty (varianza global)
    uncertainties = [p.get("uncertainty", 0) for p in con_layers]
    uncertainties_no_cero = [u for u in uncertainties if u != 0]

    print(f"\n🔢 Verificación de 'uncertainty' (varianza global):")
    print(
        f"   - Predicciones con uncertainty != 0: {len(uncertainties_no_cero)}/{len(con_layers)}"
    )

    if len(uncertainties_no_cero) > 0:
        print(
            f"   - Rango: [{min(uncertainties_no_cero):.8f}, {max(uncertainties_no_cero):.8f}]"
        )
        print(f"   - Promedio: {np.mean(uncertainties_no_cero):.8f}")
    else:
        print(f"   ⚠️  Todas las uncertainties son 0. Verifica el cálculo de varianza.")

    # Conclusión final
    print(f"\n{'='*80}")
    if porcentaje > 95 and num_layers_promedio >= 4 and valores_unicos > 100:
        print(f"✅✅✅ DATOS REALES VERIFICADOS - LISTOS PARA RQ1")
        print(f"\n🎯 Próximo paso:")
        print(f"   - Ejecuta RQ1 con los datos reales")
        print(f"   - La función simulate_layer_uncertainties() será reemplazada")
        print(f"={'='*80}")
        return True
    else:
        print(f"⚠️  ADVERTENCIA: Los datos parecen tener problemas")
        print(f"\n💡 Recomendación:")
        print(f"   - Revisa la implementación de inference_decoder_variance()")
        print(f"   - Re-ejecuta la Fase 5 con prints de debug")
        print(f"   - Verifica que los hooks capturen correctamente")
        print(f"={'='*80}")
        return False


if __name__ == "__main__":
    exito = verificar_datos_reales()

    if not exito:
        print(
            f"\n❌ Verificación falló. Lee el análisis arriba para resolver problemas."
        )
        exit(1)
    else:
        print(f"\n✅ Verificación exitosa. Puedes proceder con RQ1.")
        exit(0)
