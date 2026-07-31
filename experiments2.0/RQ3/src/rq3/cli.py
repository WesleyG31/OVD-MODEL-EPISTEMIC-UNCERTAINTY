from __future__ import annotations

import argparse
import importlib.metadata
import json

from adas_ovd.config import load_config, project_path

from .evaluation import evaluate_fusions
from .extraction import extract_split
from .fusion import fit_fusions
from .manifest import validate_manifest
from .report import generate_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rq3", description="RQ3 localization-aware confidence fusion pipeline"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--config", required=True)
    extract = commands.add_parser("extract")
    extract.add_argument("--config", required=True)
    extract.add_argument(
        "--split", choices=("train", "validation", "test"), required=True
    )
    extract.add_argument("--limit", type=int)
    for name in ("fit", "evaluate", "report"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--config", required=True)
    smoke.add_argument("--images", type=int, default=2)
    return parser


def _verify_distribution() -> None:
    found = importlib.metadata.version("groundingdino-py")
    if found != "0.4.0":
        raise RuntimeError(f"Expected groundingdino-py==0.4.0, found {found}")


def main() -> None:
    arguments = _parser().parse_args()
    config = load_config(arguments.config)
    manifest_path, manifest = validate_manifest(config)
    if arguments.command == "validate":
        print(
            json.dumps(
                {
                    "manifest": str(manifest_path),
                    "test_partition": manifest["test_partition"],
                    "split_counts": {
                        name: len(values) for name, values in manifest["splits"].items()
                    },
                },
                indent=2,
            )
        )
        return
    _verify_distribution()
    if arguments.command == "extract":
        print(extract_split(config, arguments.split, arguments.limit))
    elif arguments.command == "fit":
        print(json.dumps({"models": sorted(fit_fusions(config))}, indent=2))
    elif arguments.command == "evaluate":
        print(json.dumps(evaluate_fusions(config), indent=2))
    elif arguments.command == "report":
        print(
            json.dumps(
                {name: str(path) for name, path in generate_report(config).items()},
                indent=2,
            )
        )
    elif arguments.command == "smoke":
        destination = project_path(config, "RQ3/outputs/smoke_features.parquet")
        print(
            extract_split(
                config,
                "train",
                arguments.images,
                destination,
                shared_cache_namespace="rq3_cli_smoke",
            )
        )


if __name__ == "__main__":
    main()

