from __future__ import annotations

import argparse
import json
from pathlib import Path

from adas_ovd.config import load_config
from adas_ovd.data_preparation import prepare_bdd100k


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Download, convert and audit the pinned BDD100K Kaggle data"
    )
    parser.add_argument(
        "--config",
        default=str(project_root / "configs" / "base.yaml"),
    )
    parser.add_argument(
        "--source-dir",
        help="Use an already downloaded Kaggle dataset root (testing/offline use)",
    )
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument(
        "--fast-audit",
        action="store_true",
        help="Record image sizes but skip per-image SHA256 hashes",
    )
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    audit = prepare_bdd100k(
        config,
        source_dir=arguments.source_dir,
        force_download=arguments.force_download,
        force_prepare=arguments.force_prepare,
        include_image_sha256=not arguments.fast_audit,
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
