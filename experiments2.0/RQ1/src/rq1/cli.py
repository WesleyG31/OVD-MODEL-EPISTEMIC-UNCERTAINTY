from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

from adas_ovd.config import load_config, project_path

from .evaluation import evaluate_fusions
from .extraction import extract_split
from .fusion import fit_fusions
from .manifest import make_manifest
from .report import generate_report
from .robustness import run_robustness


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rq1",
        description="RQ1 reproducible uncertainty-fusion pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--config", required=True)
    manifest.add_argument("--force", action="store_true")

    extract = subparsers.add_parser("extract")
    extract.add_argument("--config", required=True)
    extract.add_argument(
        "--split", required=True, choices=("train", "validation", "test")
    )
    extract.add_argument("--limit", type=int)

    fit = subparsers.add_parser("fit")
    fit.add_argument("--config", required=True)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--config", required=True)

    report = subparsers.add_parser("report")
    report.add_argument("--config", required=True)

    robustness = subparsers.add_parser("robustness")
    robustness.add_argument("--config", required=True)

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--config", required=True)
    smoke.add_argument("--images", type=int, default=2)
    return parser


def _verify_distribution() -> None:
    version = importlib.metadata.version("groundingdino-py")
    if version != "0.4.0":
        raise RuntimeError(
            f"Expected groundingdino-py==0.4.0, found {version}"
        )


def main() -> None:
    arguments = _parser().parse_args()
    config = load_config(arguments.config)
    if arguments.command == "manifest":
        path = make_manifest(config, force=arguments.force)
        print(path)
        return

    _verify_distribution()
    if arguments.command == "extract":
        make_manifest(config)
        path = extract_split(
            config, split=arguments.split, limit=arguments.limit
        )
        print(path)
    elif arguments.command == "fit":
        models = fit_fusions(config)
        print(json.dumps({"models": sorted(models)}, indent=2))
    elif arguments.command == "evaluate":
        result = evaluate_fusions(config)
        print(json.dumps(result, indent=2))
    elif arguments.command == "report":
        paths = generate_report(config)
        print(
            json.dumps(
                {name: str(path) for name, path in paths.items()}, indent=2
            )
        )
    elif arguments.command == "robustness":
        result = run_robustness(config)
        print(json.dumps(result, indent=2))
    elif arguments.command == "smoke":
        make_manifest(config)
        output = project_path(config, "RQ1/outputs/smoke_features.parquet")
        path = extract_split(
            config,
            split="train",
            limit=arguments.images,
            output_override=output,
            shared_cache_namespace="rq1_cli_smoke",
        )
        print(path)


if __name__ == "__main__":
    main()
