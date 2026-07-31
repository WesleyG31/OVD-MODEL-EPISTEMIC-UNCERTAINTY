from __future__ import annotations

import argparse
import json
from pathlib import Path

from adas_ovd.config import load_config, project_path
from adas_ovd.groundingdino_adapter import (
    GroundingDinoAdapter,
    resolve_package_resource,
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Load the pinned RQ1 model and inspect stochastic modules"
    )
    parser.add_argument(
        "--config",
        default=str(project_root / "RQ1" / "configs" / "rq1.yaml"),
    )
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    model_config = str(config["model"]["config"])
    config_path = (
        resolve_package_resource(model_config)
        if model_config.startswith("package://")
        else project_path(config, model_config)
    )
    adapter = GroundingDinoAdapter(
        config_path=config_path,
        checkpoint_path=project_path(
            config, config["model"]["checkpoint"]
        ),
        text_encoder_path=project_path(
            config, config["model"]["text_encoder"]["local_dir"]
        ),
        classes=config["data"]["classes"],
        stochastic_module_types=config["rq1"]["extraction"][
            "stochastic_module_types"
        ],
        device=config["model"]["device"],
        amp=bool(config["model"]["amp"]),
    )
    try:
        with adapter.stochastic_mode() as enabled:
            backbone = [item for item in enabled if "backbone" in item["name"]]
            fusion = [item for item in enabled if "fusion_layers" in item["name"]]
            if not backbone or not fusion:
                raise RuntimeError(
                    "Expected active DropPath modules in both the Swin backbone "
                    "and transformer fusion layers"
                )
            report = {
                "device": str(adapter.device),
                "classes": list(adapter.classes),
                "checkpoint_load_audit": adapter.checkpoint_load_audit,
                "stochastic_module_counts": {
                    "total": len(enabled),
                    "backbone": len(backbone),
                    "fusion": len(fusion),
                },
                "stochastic_modules": enabled,
            }
        print(json.dumps(report, indent=2))
    finally:
        adapter.close()


if __name__ == "__main__":
    main()
