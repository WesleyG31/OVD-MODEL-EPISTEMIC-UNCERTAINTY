from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from adas_ovd.config import load_config, project_path
from adas_ovd.reproducibility import sha256_file
from rq1.evaluation import evaluate_fusions
from rq1.extraction import extract_split
from rq1.fusion import fit_fusions
from rq1.manifest import make_manifest
from rq1.report import generate_report
from rq1.robustness import run_robustness


def default_config_path(project_root: Path, mode: str) -> Path:
    config_name = "rq1_mini.yaml" if mode == "mini" else "rq1.yaml"
    return project_root / "RQ1" / "configs" / config_name


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run the ordered, leakage-safe RQ1 workflow"
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Explicit configuration override. When omitted, mini uses "
            "rq1_mini.yaml and smoke/full use rq1.yaml."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("smoke", "mini", "full"),
        default="smoke",
        help="Smoke checks repeatability; mini is diagnostic; full regenerates paper outputs",
    )
    parser.add_argument("--smoke-images", type=int, default=2)
    parser.add_argument("--mini-train-images", type=int, default=6)
    parser.add_argument("--mini-validation-images", type=int, default=4)
    parser.add_argument("--mini-test-images", type=int, default=8)
    parser.add_argument(
        "--skip-repeatability-check",
        action="store_true",
        help="Run only one smoke extraction instead of two independent runs",
    )
    arguments = parser.parse_args()
    config_path = arguments.config
    if config_path is None:
        config_path = str(default_config_path(project_root, arguments.mode))
    config = load_config(config_path)
    manifest = make_manifest(config)
    with manifest.open("r", encoding="utf-8") as handle:
        manifest_payload = json.load(handle)
    partition = manifest_payload.get("test_partition")
    if arguments.mode == "full" and partition != "confirmatory":
        raise RuntimeError("RQ1 full mode requires the confirmatory manifest")
    if arguments.mode == "mini" and partition != "diagnostic":
        raise RuntimeError("RQ1 mini mode requires the diagnostic manifest")

    if arguments.mode == "smoke":
        output = project_path(config, "RQ1/outputs/smoke_features.parquet")
        result = extract_split(
            config,
            split="train",
            limit=arguments.smoke_images,
            output_override=output,
            shared_cache_namespace="rq1_smoke_a",
        )
        payload = {"manifest": str(manifest), "smoke": str(result)}
        if not arguments.skip_repeatability_check:
            repeat_output = project_path(
                config,
                "RQ1/outputs/repeatability/smoke_features.parquet",
            )
            repeat_result = extract_split(
                config,
                split="train",
                limit=arguments.smoke_images,
                output_override=repeat_output,
                shared_cache_namespace="rq1_smoke_b",
            )
            first = pd.read_parquet(result)
            second = pd.read_parquet(repeat_result)
            pd.testing.assert_frame_equal(
                first, second, check_exact=True, check_like=False
            )
            payload["repeatability"] = {
                "status": "exact_dataframe_match",
                "repeat": str(repeat_result),
                "first_sha256": sha256_file(result),
                "repeat_sha256": sha256_file(repeat_result),
            }
        print(json.dumps(payload, indent=2))
        return

    if arguments.mode == "mini":
        train = extract_split(
            config, split="train", limit=arguments.mini_train_images
        )
        validation = extract_split(
            config,
            split="validation",
            limit=arguments.mini_validation_images,
        )
        models = fit_fusions(config)
        robustness = run_robustness(config)
        test = extract_split(
            config, split="test", limit=arguments.mini_test_images
        )
        metrics = evaluate_fusions(config)
        report = generate_report(config)
        print(
            json.dumps(
                {
                    "status": "mini_e2e_pass",
                    "manifest": str(manifest),
                    "images": {
                        "train": arguments.mini_train_images,
                        "validation": arguments.mini_validation_images,
                        "test": arguments.mini_test_images,
                    },
                    "features": {
                        "train": str(train),
                        "validation": str(validation),
                        "test": str(test),
                    },
                    "models": sorted(models),
                    "robustness": robustness,
                    "metrics": metrics,
                    "report": {
                        name: str(path) for name, path in report.items()
                    },
                },
                indent=2,
            )
        )
        return

    train = extract_split(config, split="train")
    validation = extract_split(config, split="validation")
    models = fit_fusions(config)
    robustness = run_robustness(config)
    test = extract_split(config, split="test")
    metrics = evaluate_fusions(config)
    report = generate_report(config)
    print(
        json.dumps(
            {
                "manifest": str(manifest),
                "features": {
                    "train": str(train),
                    "validation": str(validation),
                    "test": str(test),
                },
            "models": sorted(models),
            "robustness": robustness,
                "metrics": metrics,
                "report": {name: str(path) for name, path in report.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
