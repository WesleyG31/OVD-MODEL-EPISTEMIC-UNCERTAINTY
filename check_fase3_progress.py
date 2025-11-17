# Monitoreo en tiempo real de Fase 3

"""
Script para verificar el progreso de la Fase 3 mientras se ejecuta.
Puedes correr este script en una terminal separada mientras Fase 3 corre.

Uso:
    python check_fase3_progress.py
"""

import pandas as pd
from pathlib import Path
import json
import time
import sys


def check_progress():
    """Verifica el progreso de la Fase 3"""

    output_dir = Path("fase 3/outputs/mc_dropout")

    print("=" * 60)
    print("  MONITOREO DE PROGRESO - FASE 3 (MC-DROPOUT)")
    print("=" * 60)
    print()

    # Verificar si los archivos existen
    stats_file = output_dir / "mc_stats.parquet"
    timing_file = output_dir / "timing_data.parquet"
    preds_file = output_dir / "preds_mc_aggregated.json"

    total_images = 2000  # Total esperado de val_eval

    if not stats_file.exists():
        print("⏳ Fase 3 aún no ha comenzado a generar outputs")
        print(f"   Esperando archivo: {stats_file}")
        return False

    try:
        # Leer el parquet de stats
        stats_df = pd.read_parquet(stats_file)
        images_processed = stats_df["image_id"].nunique()
        detections_total = len(stats_df)

        # Leer timing si existe
        if timing_file.exists():
            timing_df = pd.read_parquet(timing_file)
            avg_time = timing_df["time_seconds"].mean()
            total_time = timing_df["time_seconds"].sum()

            # Estimar tiempo restante
            remaining_images = total_images - images_processed
            estimated_remaining = remaining_images * avg_time
            estimated_remaining_hours = estimated_remaining / 3600
            total_time_hours = total_time / 3600
        else:
            avg_time = None
            total_time_hours = None
            estimated_remaining_hours = None

        # Progreso porcentual
        progress_pct = (images_processed / total_images) * 100

        # Mostrar resultados
        print(f"📊 PROGRESO ACTUAL:")
        print(f"   Imágenes procesadas: {images_processed} / {total_images}")
        print(f"   Progreso: {progress_pct:.1f}%")
        print(f"   Detecciones totales: {detections_total}")
        print()

        if avg_time:
            print(f"⏱️  TIEMPOS:")
            print(f"   Tiempo promedio por imagen: {avg_time:.2f}s")
            print(f"   Tiempo total transcurrido: {total_time_hours:.2f} horas")
            if estimated_remaining_hours:
                print(
                    f"   Tiempo estimado restante: {estimated_remaining_hours:.2f} horas"
                )
            print()

        # Barra de progreso visual
        bar_length = 50
        filled = int(bar_length * progress_pct / 100)
        bar = "█" * filled + "░" * (bar_length - filled)
        print(f"   [{bar}] {progress_pct:.1f}%")
        print()

        # Estado
        if images_processed >= total_images:
            print("✅ FASE 3 COMPLETADA!")
            print()
            print("📁 Archivos generados:")
            print(f"   - {stats_file}")
            print(f"   - {timing_file}")
            print(f"   - {preds_file}")
            print()
            print("🚀 Puedes proceder a correr Fase 5")
            return True
        else:
            print(f"⏳ En progreso... ({remaining_images} imágenes restantes)")
            return False

    except Exception as e:
        print(f"❌ Error al leer archivos: {e}")
        print("   El proceso puede estar iniciando o hubo un error.")
        return False


def monitor_continuous(interval=60):
    """Monitorea continuamente el progreso cada 'interval' segundos"""

    print("🔄 Modo de monitoreo continuo activado")
    print(f"   Actualizando cada {interval} segundos")
    print("   Presiona Ctrl+C para salir")
    print()

    try:
        while True:
            completed = check_progress()

            if completed:
                print("\n✅ Monitoreo completado - Fase 3 terminada")
                break

            print(f"\n⏳ Esperando {interval} segundos para próxima verificación...")
            print("-" * 60)
            time.sleep(interval)
            print("\n" * 2)  # Espacio para claridad

    except KeyboardInterrupt:
        print("\n\n⏹️  Monitoreo detenido por el usuario")
        print("   Fase 3 continúa ejecutándose en segundo plano")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Monitorear progreso de Fase 3 (MC-Dropout)"
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Modo continuo: actualiza automáticamente cada 60 segundos",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Intervalo en segundos para modo continuo (default: 60)",
    )

    args = parser.parse_args()

    if args.continuous:
        monitor_continuous(interval=args.interval)
    else:
        check_progress()
