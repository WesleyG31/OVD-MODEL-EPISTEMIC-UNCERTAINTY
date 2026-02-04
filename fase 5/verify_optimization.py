#!/usr/bin/env python3
"""
Script de verificación de optimizaciones de Fase 5
===================================================

Este script verifica que:
1. Los archivos de fases anteriores existen
2. Los formatos de datos son correctos
3. Las predicciones son compatibles
4. Estima el tiempo que se ahorrará
"""

import json
import sys
from pathlib import Path
from datetime import timedelta


# Colores para terminal
class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


def check_file(path, description):
    """Verifica si un archivo existe y retorna su info"""
    path = Path(path)
    if path.exists():
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"{Colors.GREEN}✅ {description}{Colors.END}")
        print(f"   Ubicación: {path}")
        print(f"   Tamaño: {size_mb:.2f} MB")
        return True, size_mb
    else:
        print(f"{Colors.RED}❌ {description}{Colors.END}")
        print(f"   {Colors.YELLOW}No encontrado: {path}{Colors.END}")
        return False, 0


def verify_json_format(path, expected_keys):
    """Verifica que un JSON tenga el formato esperado"""
    try:
        with open(path, "r") as f:
            data = json.load(f)

        if not isinstance(data, list) or len(data) == 0:
            return False, "No es una lista o está vacía"

        sample = data[0]
        missing_keys = [k for k in expected_keys if k not in sample]

        if missing_keys:
            return False, f"Faltan keys: {missing_keys}"

        return True, f"{len(data)} registros"
    except Exception as e:
        return False, str(e)


def main():
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}")
    print("VERIFICACIÓN DE OPTIMIZACIONES - FASE 5")
    print(f"{'='*70}{Colors.END}\n")

    # Paths
    base_dir = Path("..")
    fase2_preds = base_dir / "fase 2" / "outputs" / "baseline" / "preds_raw.json"
    fase3_preds = (
        base_dir / "fase 3" / "outputs" / "mc_dropout" / "preds_mc_aggregated.json"
    )
    fase4_temp = (
        base_dir / "fase 4" / "outputs" / "temperature_scaling" / "temperature.json"
    )

    # Contadores
    files_found = 0
    total_files = 3
    time_saved = 0

    # ========================================================================
    print(f"{Colors.BOLD}1. VERIFICACIÓN DE ARCHIVOS DE FASE 2 (Baseline){Colors.END}")
    print("-" * 70)

    exists, size = check_file(fase2_preds, "Predicciones Baseline")
    if exists:
        files_found += 1
        time_saved += 45  # 45 minutos ahorrados

        # Verificar formato
        valid, info = verify_json_format(
            fase2_preds, ["image_id", "category_id", "bbox", "score"]
        )
        if valid:
            print(f"   {Colors.GREEN}Formato: ✅ Correcto ({info}){Colors.END}")
        else:
            print(f"   {Colors.YELLOW}Formato: ⚠️  {info}{Colors.END}")

    print()

    # ========================================================================
    print(
        f"{Colors.BOLD}2. VERIFICACIÓN DE ARCHIVOS DE FASE 3 (MC-Dropout){Colors.END}"
    )
    print("-" * 70)

    exists, size = check_file(fase3_preds, "Predicciones MC-Dropout")
    if exists:
        files_found += 1
        time_saved += 90  # 90 minutos ahorrados (K=5 es costoso)

        # Verificar formato
        valid, info = verify_json_format(
            fase3_preds, ["image_id", "category_id", "bbox", "score", "uncertainty"]
        )
        if valid:
            print(f"   {Colors.GREEN}Formato: ✅ Correcto ({info}){Colors.END}")
        else:
            print(f"   {Colors.YELLOW}Formato: ⚠️  {info}{Colors.END}")

    print()

    # ========================================================================
    print(
        f"{Colors.BOLD}3. VERIFICACIÓN DE ARCHIVOS DE FASE 4 (Temperature){Colors.END}"
    )
    print("-" * 70)

    exists, size = check_file(fase4_temp, "Temperaturas Optimizadas")
    if exists:
        files_found += 1
        time_saved += 2  # 2 minutos ahorrados

        # Verificar formato
        try:
            with open(fase4_temp, "r") as f:
                temps = json.load(f)

            if "optimal_temperature" in temps:
                T = temps["optimal_temperature"]
                print(f"   {Colors.GREEN}Formato: ✅ Correcto (T={T:.4f}){Colors.END}")
            else:
                print(
                    f"   {Colors.YELLOW}Formato: ⚠️  Falta 'optimal_temperature'{Colors.END}"
                )
        except Exception as e:
            print(f"   {Colors.YELLOW}Formato: ⚠️  Error: {e}{Colors.END}")

    print()

    # ========================================================================
    print(f"{Colors.BOLD}{'='*70}")
    print("RESUMEN")
    print(f"{'='*70}{Colors.END}")

    print(
        f"\n{Colors.BOLD}Archivos encontrados:{Colors.END} {files_found}/{total_files}"
    )

    if files_found == total_files:
        print(f"{Colors.GREEN}✅ TODOS los archivos están disponibles{Colors.END}")
    elif files_found > 0:
        print(f"{Colors.YELLOW}⚠️  Algunos archivos están disponibles{Colors.END}")
    else:
        print(f"{Colors.RED}❌ NO hay archivos disponibles{Colors.END}")

    # Estimación de tiempo
    print(f"\n{Colors.BOLD}Tiempo estimado ahorrado:{Colors.END}")

    if time_saved > 0:
        td = timedelta(minutes=time_saved)
        hours = td.seconds // 3600
        minutes = (td.seconds % 3600) // 60

        print(f"   {Colors.GREEN}⚡ ~{hours}h {minutes}min{Colors.END}")

        if files_found == total_files:
            print(f"\n{Colors.BOLD}Tiempo de ejecución esperado:{Colors.END}")
            print(
                f"   {Colors.GREEN}📊 ~15-20 minutos{Colors.END} (solo Decoder Variance)"
            )
        else:
            missing = total_files - files_found
            est_time = 137 - time_saved  # 137 min total original
            print(f"\n{Colors.BOLD}Tiempo de ejecución esperado:{Colors.END}")
            print(
                f"   {Colors.YELLOW}📊 ~{est_time} minutos{Colors.END} (calcular {missing} método(s) faltante(s))"
            )
    else:
        print(f"   {Colors.RED}❌ 0 minutos{Colors.END}")
        print(f"\n{Colors.BOLD}Tiempo de ejecución esperado:{Colors.END}")
        print(f"   {Colors.RED}📊 ~2 horas{Colors.END} (inferencia completa)")

    # Recomendaciones
    print(f"\n{Colors.BOLD}{'='*70}")
    print("RECOMENDACIONES")
    print(f"{'='*70}{Colors.END}")

    if files_found == total_files:
        print(
            f"{Colors.GREEN}✅ Perfecto! Puedes ejecutar Fase 5 directamente.{Colors.END}"
        )
        print(f"   El notebook usará todos los resultados cacheados.")
    elif files_found == 0:
        print(f"{Colors.YELLOW}⚠️  Ejecuta las siguientes fases primero:{Colors.END}")
        print(f"   1. Fase 2: Genera predicciones baseline")
        print(f"   2. Fase 3: Genera predicciones MC-Dropout")
        print(f"   3. Fase 4: Optimiza temperaturas")
        print(f"\n   O ejecuta Fase 5 directamente (tardará ~2 horas)")
    else:
        print(f"{Colors.YELLOW}⚠️  Tienes optimización parcial.{Colors.END}")

        if not (base_dir / fase2_preds).exists():
            print(f"   • Ejecuta Fase 2 para predicciones baseline")
        if not (base_dir / fase3_preds).exists():
            print(f"   • Ejecuta Fase 3 para predicciones MC-Dropout")
        if not (base_dir / fase4_temp).exists():
            print(f"   • Ejecuta Fase 4 para temperaturas")

        print(f"\n   O ejecuta Fase 5 ahora (ahorrará ~{time_saved} min)")

    print(f"\n{Colors.BOLD}{'='*70}{Colors.END}\n")

    # Exit code
    return 0 if files_found == total_files else 1


if __name__ == "__main__":
    sys.exit(main())
