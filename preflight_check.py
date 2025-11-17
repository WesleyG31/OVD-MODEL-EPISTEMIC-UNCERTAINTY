# Pre-vuelo: Verificación antes de correr Fase 3

"""
Script de verificación pre-vuelo para asegurar que todo está listo
antes de correr la Fase 3 (que puede tardar 6-7 horas).

Uso:
    python preflight_check.py
"""

import sys
from pathlib import Path
import json
import torch


def check_item(name, status, details=""):
    """Helper para mostrar checks"""
    icon = "✅" if status else "❌"
    print(f"{icon} {name}")
    if details:
        print(f"   {details}")
    return status


def main():
    print("=" * 70)
    print("  PRE-VUELO: VERIFICACIÓN ANTES DE CORRER FASE 3")
    print("=" * 70)
    print()

    all_checks_passed = True

    # 1. Verificar estructura de directorios
    print("📁 1. ESTRUCTURA DE DIRECTORIOS:")
    print("-" * 70)

    base_dir = Path(".")
    fase3_dir = base_dir / "fase 3"
    data_dir = base_dir / "data"

    checks = [
        ("Directorio Fase 3 existe", fase3_dir.exists(), str(fase3_dir)),
        ("Directorio data existe", data_dir.exists(), str(data_dir)),
        (
            "Notebook main.ipynb existe",
            (fase3_dir / "main.ipynb").exists(),
            "fase 3/main.ipynb",
        ),
    ]

    for name, status, details in checks:
        all_checks_passed &= check_item(name, status, details)

    print()

    # 2. Verificar archivos de datos
    print("📊 2. ARCHIVOS DE DATOS:")
    print("-" * 70)

    val_eval = data_dir / "bdd100k_coco" / "val_eval.json"
    images_dir = (
        data_dir / "bdd100k" / "bdd100k" / "bdd100k" / "images" / "100k" / "val"
    )

    val_eval_exists = val_eval.exists()
    images_dir_exists = images_dir.exists()

    all_checks_passed &= check_item(
        "val_eval.json existe", val_eval_exists, str(val_eval)
    )
    all_checks_passed &= check_item(
        "Directorio imágenes existe", images_dir_exists, str(images_dir)
    )

    # Contar imágenes si existe
    if val_eval_exists:
        try:
            with open(val_eval) as f:
                data = json.load(f)
                num_images = len(data.get("images", []))
                check_item(f"Imágenes en val_eval.json", True, f"{num_images} imágenes")
                if num_images != 2000:
                    print(
                        f"   ⚠️  Advertencia: Se esperaban 2000 imágenes, encontradas {num_images}"
                    )
        except Exception as e:
            check_item("Leer val_eval.json", False, f"Error: {e}")
            all_checks_passed = False

    if images_dir_exists:
        num_image_files = len(list(images_dir.glob("*.jpg")))
        check_item(
            f"Archivos de imagen disponibles", True, f"{num_image_files} archivos .jpg"
        )
        if num_image_files < 2000:
            print(f"   ⚠️  Advertencia: Menos de 2000 imágenes encontradas")

    print()

    # 3. Verificar CUDA/GPU
    print("🖥️  3. CONFIGURACIÓN COMPUTACIONAL:")
    print("-" * 70)

    cuda_available = torch.cuda.is_available()
    all_checks_passed &= check_item(
        "CUDA disponible",
        cuda_available,
        f"GPU: {torch.cuda.get_device_name(0) if cuda_available else 'N/A'}",
    )

    if cuda_available:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        check_item(
            f"Memoria GPU", gpu_memory >= 6, f"{gpu_memory:.1f} GB (recomendado: 6+ GB)"
        )

    print()

    # 4. Verificar modelo GroundingDINO
    print("🤖 4. MODELO GROUNDINGDINO:")
    print("-" * 70)

    model_config = Path(
        "/opt/program/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    )
    model_weights = Path(
        "/opt/program/GroundingDINO/weights/groundingdino_swint_ogc.pth"
    )

    checks = [
        ("Configuración del modelo", model_config.exists(), str(model_config)),
        ("Pesos del modelo", model_weights.exists(), str(model_weights)),
    ]

    for name, status, details in checks:
        all_checks_passed &= check_item(name, status, details)

    print()

    # 5. Verificar modificación del notebook
    print("📝 5. VERIFICACIÓN DEL NOTEBOOK:")
    print("-" * 70)

    notebook_path = fase3_dir / "main.ipynb"
    if notebook_path.exists():
        try:
            with open(notebook_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Verificar que NO tenga [:100]
            has_limitation = "image_ids[:100]" in content
            notebook_ok = not has_limitation

            check_item(
                "Notebook sin limitación [:100]",
                notebook_ok,
                (
                    "Procesará todas las imágenes"
                    if notebook_ok
                    else "⚠️ TODAVÍA TIENE [:100] - necesita corrección"
                ),
            )

            if has_limitation:
                print()
                print("   🔧 ACCIÓN REQUERIDA:")
                print("   El notebook todavía tiene 'image_ids[:100]'")
                print(
                    "   Debe cambiarse a 'image_ids' para procesar todas las imágenes"
                )
                all_checks_passed = False
        except Exception as e:
            check_item("Leer notebook", False, f"Error: {e}")
            all_checks_passed = False

    print()

    # 6. Verificar dependencias Python
    print("📦 6. DEPENDENCIAS PYTHON:")
    print("-" * 70)

    required_packages = [
        "torch",
        "numpy",
        "pandas",
        "PIL",
        "tqdm",
        "yaml",
        "pycocotools",
        "sklearn",
        "matplotlib",
        "seaborn",
    ]

    for package in required_packages:
        try:
            __import__(package)
            check_item(f"Paquete {package}", True, "")
        except ImportError:
            check_item(f"Paquete {package}", False, "⚠️ NO instalado")
            all_checks_passed = False

    print()

    # 7. Espacio en disco
    print("💾 7. ESPACIO EN DISCO:")
    print("-" * 70)

    try:
        import shutil

        total, used, free = shutil.disk_usage(str(base_dir))
        free_gb = free / (1024**3)

        has_space = free_gb >= 5
        check_item(
            "Espacio libre suficiente",
            has_space,
            f"{free_gb:.1f} GB libres (recomendado: 5+ GB)",
        )
    except Exception as e:
        check_item("Verificar espacio", False, f"Error: {e}")

    print()
    print("=" * 70)

    # Resumen final
    if all_checks_passed:
        print("✅ TODOS LOS CHECKS PASARON")
        print()
        print("🚀 Estás listo para correr Fase 3!")
        print()
        print("   Próximos pasos:")
        print("   1. Abrir: fase 3/main.ipynb")
        print("   2. Ejecutar todas las celdas (Run All)")
        print(
            "   3. Monitorear progreso con: python check_fase3_progress.py --continuous"
        )
        print()
        print("   ⏱️  Tiempo estimado: 6-7 horas")
        return 0
    else:
        print("❌ ALGUNOS CHECKS FALLARON")
        print()
        print("   ⚠️  Por favor corrige los problemas antes de continuar")
        print("   Revisa los items marcados con ❌ arriba")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
