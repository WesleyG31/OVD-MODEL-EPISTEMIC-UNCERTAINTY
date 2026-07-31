from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from adas_ovd.config import load_config, project_path
from adas_ovd.reproducibility import sha256_file
from rq2.estimators import fit_estimators
from rq2.evaluation import evaluate_estimators
from rq2.extraction import extract_split
from rq2.manifest import validate_manifest
from rq2.report import generate_report


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run the frozen leakage-safe RQ2 workflow")
    parser.add_argument("--config")
    parser.add_argument("--mode", choices=("smoke", "mini", "full"), default="smoke")
    parser.add_argument("--smoke-images", type=int, default=2)
    parser.add_argument("--mini-train-images", type=int, default=6)
    parser.add_argument("--mini-validation-images", type=int, default=4)
    parser.add_argument("--mini-test-images", type=int, default=8)
    parser.add_argument("--skip-repeatability-check", action="store_true")
    arguments = parser.parse_args()
    config_path = (
        Path(arguments.config)
        if arguments.config
        else project_root
        / "RQ2"
        / "configs"
        / ("rq2_mini.yaml" if arguments.mode == "mini" else "rq2.yaml")
    )
    config = load_config(config_path)
    manifest_path, manifest = validate_manifest(config)
    if arguments.mode == "full" and manifest["test_partition"] != "confirmatory":
        raise RuntimeError("RQ2 full mode requires the confirmatory manifest")
    if arguments.mode == "mini" and manifest["test_partition"] != "diagnostic":
        raise RuntimeError("RQ2 mini mode requires the diagnostic manifest")

    if arguments.mode == "smoke":
        output = project_path(config, "RQ2/outputs/smoke_features.parquet")
        first_path = extract_split(
            config,
            "train",
            arguments.smoke_images,
            output_override=output,
            shared_cache_namespace="rq2_smoke_a",
        )
        payload: dict[str, object] = {
            "status": "smoke_pass",
            "manifest": str(manifest_path),
            "images": arguments.smoke_images,
            "features": str(first_path),
        }
        if not arguments.skip_repeatability_check:
            repeat = project_path(
                config, "RQ2/outputs/repeatability/smoke_features.parquet"
            )
            second_path = extract_split(
                config,
                "train",
                arguments.smoke_images,
                output_override=repeat,
                shared_cache_namespace="rq2_smoke_b",
            )
            first = pd.read_parquet(first_path)
            second = pd.read_parquet(second_path)
            pd.testing.assert_frame_equal(first, second, check_exact=True, check_like=False)
            payload["repeatability"] = {
                "status": "exact_dataframe_match",
                "first_sha256": sha256_file(first_path),
                "repeat_sha256": sha256_file(second_path),
                "parquet_sha256_match": sha256_file(first_path) == sha256_file(second_path),
            }
        print(json.dumps(payload, indent=2))
        return

    limits = (
        {
            "train": arguments.mini_train_images,
            "validation": arguments.mini_validation_images,
            "test": arguments.mini_test_images,
        }
        if arguments.mode == "mini"
        else {"train": None, "validation": None, "test": None}
    )
    train = extract_split(config, "train", limits["train"])
    validation = extract_split(config, "validation", limits["validation"])
    estimators = fit_estimators(config)
    test = extract_split(config, "test", limits["test"])
    metrics = evaluate_estimators(config)
    report = generate_report(config)
    print(
        json.dumps(
            {
                "status": f"{arguments.mode}_e2e_pass",
                "evidence_status": metrics["evidence_status"],
                "manifest": str(manifest_path),
                "images": limits,
                "features": {
                    "train": str(train),
                    "validation": str(validation),
                    "test": str(test),
                },
                "models": sorted(estimators),
                "test_rows": metrics["test_rows"],
                "primary_inference": metrics["primary_inference"],
                "computational_cost": {
                    key: value
                    for key, value in metrics["computational_cost"].items()
                    if key not in {"environment", "stochastic_modules"}
                },
                "artifact_integrity": metrics["artifact_integrity"],
                "report": {name: str(path) for name, path in report.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
