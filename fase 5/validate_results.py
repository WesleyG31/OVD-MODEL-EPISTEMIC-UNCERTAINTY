#!/usr/bin/env python3
"""
Script de validación post-ejecución - Fase 5
============================================

Verifica que los resultados sean correctos después de ejecutar el notebook.
"""

import pandas as pd
import json
from pathlib import Path
import sys


class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


def test_calibration_data():
    """Verifica que los datos de calibración tengan incertidumbres correctas"""
    print(f"\n{Colors.BOLD}TEST 1: Datos de Calibración{Colors.END}")
    print("-" * 70)

    try:
        baseline = pd.read_csv("outputs/comparison/calib_baseline.csv")
        mc_dropout = pd.read_csv("outputs/comparison/calib_mc_dropout.csv")
        decoder_var = pd.read_csv("outputs/comparison/calib_decoder_variance.csv")

        # Verificar incertidumbres
        b_unc = baseline["uncertainty"].mean()
        m_unc = mc_dropout["uncertainty"].mean()
        d_unc = decoder_var["uncertainty"].mean()

        print(f"Baseline uncertainty:        {b_unc:.6f}", end="")
        if b_unc == 0.0:
            print(f" {Colors.GREEN}✅ CORRECTO{Colors.END}")
        else:
            print(f" {Colors.RED}❌ INCORRECTO (debería ser 0.0){Colors.END}")

        print(f"MC-Dropout uncertainty:      {m_unc:.6f}", end="")
        if m_unc > 0.001:
            print(f" {Colors.GREEN}✅ CORRECTO{Colors.END}")
        else:
            print(f" {Colors.RED}❌ INCORRECTO (debería ser > 0){Colors.END}")

        print(f"Decoder Variance uncertainty: {d_unc:.6f}", end="")
        if d_unc > 0.001:
            print(f" {Colors.GREEN}✅ CORRECTO{Colors.END}")
        else:
            print(f" {Colors.RED}❌ INCORRECTO (debería ser > 0){Colors.END}")

        # Verificar que no sean idénticos
        print(f"\n Verificando que los datos NO sean idénticos...")

        # Comparar primeras 10 filas de scores
        b_scores = baseline["score"].head(10).values
        m_scores = mc_dropout["score"].head(10).values
        d_scores = decoder_var["score"].head(10).values

        identical_bm = all(abs(b - m) < 1e-9 for b, m in zip(b_scores, m_scores))
        identical_bd = all(abs(b - d) < 1e-9 for b, d in zip(b_scores, d_scores))

        if identical_bm:
            print(f"   {Colors.RED}❌ Baseline y MC-Dropout son IDÉNTICOS{Colors.END}")
        else:
            print(
                f"   {Colors.GREEN}✅ Baseline y MC-Dropout son DIFERENTES{Colors.END}"
            )

        if identical_bd:
            print(
                f"   {Colors.RED}❌ Baseline y Decoder Variance son IDÉNTICOS{Colors.END}"
            )
        else:
            print(
                f"   {Colors.GREEN}✅ Baseline y Decoder Variance son DIFERENTES{Colors.END}"
            )

        # Conteo de detecciones
        print(f"\n Conteo de detecciones:")
        print(f"   Baseline:        {len(baseline)} detecciones")
        print(f"   MC-Dropout:      {len(mc_dropout)} detecciones")
        print(f"   Decoder Variance: {len(decoder_var)} detecciones")

        return (
            b_unc == 0.0
            and m_unc > 0.001
            and d_unc > 0.001
            and not identical_bm
            and not identical_bd
        )

    except Exception as e:
        print(f"{Colors.RED}❌ ERROR: {e}{Colors.END}")
        return False


def test_temperatures():
    """Verifica que las temperaturas sean diferentes entre métodos"""
    print(f"\n{Colors.BOLD}TEST 2: Temperaturas Optimizadas{Colors.END}")
    print("-" * 70)

    try:
        with open("outputs/comparison/temperatures.json", "r") as f:
            temps = json.load(f)

        t_baseline = temps["baseline"]["T"]
        t_mc = temps["mc_dropout"]["T"]
        t_dec = temps["decoder_variance"]["T"]

        print(f"Baseline:        T = {t_baseline:.4f}")
        print(f"MC-Dropout:      T = {t_mc:.4f}")
        print(f"Decoder Variance: T = {t_dec:.4f}")

        # Verificar que sean diferentes
        diff_bm = abs(t_baseline - t_mc) > 0.01
        diff_bd = abs(t_baseline - t_dec) > 0.01

        print(f"\n Verificando diferencias:")
        if diff_bm:
            print(
                f"   {Colors.GREEN}✅ Baseline ≠ MC-Dropout (diff: {abs(t_baseline - t_mc):.4f}){Colors.END}"
            )
        else:
            print(
                f"   {Colors.YELLOW}⚠️  Baseline ≈ MC-Dropout (diff: {abs(t_baseline - t_mc):.4f}){Colors.END}"
            )

        if diff_bd:
            print(
                f"   {Colors.GREEN}✅ Baseline ≠ Decoder Variance (diff: {abs(t_baseline - t_dec):.4f}){Colors.END}"
            )
        else:
            print(
                f"   {Colors.YELLOW}⚠️  Baseline ≈ Decoder Variance (diff: {abs(t_baseline - t_dec):.4f}){Colors.END}"
            )

        # Verificar NLLs
        print(f"\n NLL values:")
        for method in ["baseline", "mc_dropout", "decoder_variance"]:
            nll_before = temps[method]["nll_before"]
            nll_after = temps[method]["nll_after"]
            improvement = nll_before - nll_after
            print(
                f"   {method:20s}: {nll_before:.4f} → {nll_after:.4f} (Δ={improvement:.4f})"
            )

        return diff_bm or diff_bd  # Al menos una debe ser diferente

    except Exception as e:
        print(f"{Colors.RED}❌ ERROR: {e}{Colors.END}")
        return False


def test_cache_usage():
    """Verifica mensajes de caché en el output del notebook"""
    print(f"\n{Colors.BOLD}TEST 3: Uso de Caché{Colors.END}")
    print("-" * 70)

    # Esto requeriría leer el output del notebook, lo cual es complejo
    # En su lugar, verificamos archivos esperados

    fase2_baseline = Path("../fase 2/outputs/baseline/preds_raw.json")
    fase3_parquet = Path("../fase 3/outputs/mc_dropout/mc_stats_labeled.parquet")
    fase4_temp = Path("../fase 4/outputs/temperature_scaling/temperature.json")

    print("Verificando archivos de caché:")

    cache_count = 0

    if fase2_baseline.exists():
        print(f"   {Colors.GREEN}✅ Fase 2 Baseline disponible{Colors.END}")
        cache_count += 1
    else:
        print(f"   {Colors.YELLOW}⚠️  Fase 2 Baseline NO disponible{Colors.END}")

    if fase3_parquet.exists():
        print(f"   {Colors.GREEN}✅ Fase 3 MC-Dropout (Parquet) disponible{Colors.END}")
        cache_count += 1
    else:
        print(
            f"   {Colors.YELLOW}⚠️  Fase 3 MC-Dropout (Parquet) NO disponible{Colors.END}"
        )

    if fase4_temp.exists():
        print(f"   {Colors.GREEN}✅ Fase 4 Temperaturas disponible{Colors.END}")
        cache_count += 1
    else:
        print(f"   {Colors.YELLOW}⚠️  Fase 4 Temperaturas NO disponible{Colors.END}")

    print(f"\n Cache disponible: {cache_count}/3")

    return cache_count >= 2


def test_evaluation_files():
    """Verifica que existan archivos de evaluación"""
    print(f"\n{Colors.BOLD}TEST 4: Archivos de Evaluación{Colors.END}")
    print("-" * 70)

    expected_files = [
        "eval_baseline.csv",
        "eval_baseline_ts.csv",
        "eval_mc_dropout.csv",
        "eval_mc_dropout_ts.csv",
        "eval_decoder_variance.csv",
        "eval_decoder_variance_ts.csv",
    ]

    found = 0
    for fname in expected_files:
        path = Path(f"outputs/comparison/{fname}")
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"   {Colors.GREEN}✅{Colors.END} {fname:30s} ({size_mb:.2f} MB)")
            found += 1
        else:
            print(f"   {Colors.RED}❌{Colors.END} {fname:30s} (NO ENCONTRADO)")

    print(f"\n Archivos encontrados: {found}/{len(expected_files)}")

    return found == len(expected_files)


def main():
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}")
    print("VALIDACIÓN DE RESULTADOS - FASE 5")
    print(f"{'='*70}{Colors.END}\n")

    tests = [
        ("Datos de Calibración", test_calibration_data),
        ("Temperaturas", test_temperatures),
        ("Uso de Caché", test_cache_usage),
        ("Archivos de Evaluación", test_evaluation_files),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n{Colors.RED}ERROR en {test_name}: {e}{Colors.END}")
            results.append((test_name, False))

    # Resumen
    print(f"\n{Colors.BOLD}{'='*70}")
    print("RESUMEN")
    print(f"{'='*70}{Colors.END}\n")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = (
            f"{Colors.GREEN}✅ PASSED{Colors.END}"
            if result
            else f"{Colors.RED}❌ FAILED{Colors.END}"
        )
        print(f"   {test_name:30s}: {status}")

    print(f"\n{Colors.BOLD}Total: {passed}/{total} tests passed{Colors.END}")

    if passed == total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 ¡TODO CORRECTO!{Colors.END}")
        print(f"Los resultados parecen válidos. Puedes proceder con el análisis.")
        return 0
    else:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠️  ALGUNOS TESTS FALLARON{Colors.END}")
        print(f"Revisa los detalles arriba y consulta VERIFICACION_COMPLETA.md")
        return 1


if __name__ == "__main__":
    sys.exit(main())
