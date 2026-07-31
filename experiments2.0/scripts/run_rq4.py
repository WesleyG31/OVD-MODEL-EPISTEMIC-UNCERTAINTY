from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from adas_ovd.config import load_config, project_path
from adas_ovd.reproducibility import sha256_file
from rq4.calibration import fit_calibrations
from rq4.evaluation import evaluate_calibrations
from rq4.extraction import extract_split
from rq4.manifest import validate_manifest
from rq4.report import generate_report


ALIGNMENT_COLUMNS = [
    "image_id", "detection_index", "query_index", "category_index", "category_id",
    "score", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "is_error",
    "matched_iou", "mc_matches",
]


def _metadata(path: Path) -> dict[str, Any]:
    with path.with_suffix(".metadata.json").open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    receipt = Path(payload["shared_request_metadata_path"])
    payload["receipt_hash_valid"] = bool(
        receipt.is_file() and payload["shared_request_metadata_sha256"] == sha256_file(receipt)
    )
    return payload


def _alignment_audit(project_root: Path, rq4_paths: dict[str, Path]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for consumer in ("RQ1", "RQ2", "RQ3"):
        consumer_result: dict[str, Any] = {}
        for split, rq4_path in rq4_paths.items():
            other = project_root / consumer / "outputs" / "mini_e2e" / f"features_{split}.parquet"
            if not other.is_file():
                consumer_result[split] = {"status": "not_available"}
                continue
            rq4_frame = pd.read_parquet(rq4_path, columns=ALIGNMENT_COLUMNS)
            other_frame = pd.read_parquet(other, columns=ALIGNMENT_COLUMNS)
            rq4_images = set(int(value) for value in rq4_frame["image_id"].unique())
            other_images = set(int(value) for value in other_frame["image_id"].unique())
            if rq4_images != other_images:
                consumer_result[split] = {
                    "status": "different_prespecified_image_universe",
                    "rq4_images": len(rq4_images),
                    "consumer_images": len(other_images),
                    "overlap_images": len(rq4_images & other_images),
                    "reason": "RQ4 train/validation are source-domain-only by frozen design",
                }
                continue
            pd.testing.assert_frame_equal(rq4_frame, other_frame, check_exact=True, check_like=False)
            consumer_result[split] = {"status": "exact_common_universe_match", "rows": len(rq4_frame)}
        result[consumer] = consumer_result
    return result


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run the frozen leakage-safe RQ4 workflow")
    parser.add_argument("--config")
    parser.add_argument("--mode", choices=("smoke", "mini", "full"), default="smoke")
    parser.add_argument("--smoke-images", type=int, default=2)
    parser.add_argument("--mini-train-images", type=int, default=6)
    parser.add_argument("--mini-validation-images", type=int, default=4)
    parser.add_argument("--mini-test-images", type=int, default=8)
    parser.add_argument("--skip-repeatability-check", action="store_true")
    arguments = parser.parse_args()
    config_path = (
        Path(arguments.config) if arguments.config else project_root / "RQ4" / "configs" / ("rq4_mini.yaml" if arguments.mode == "mini" else "rq4.yaml")
    )
    config = load_config(config_path)
    manifest_path, manifest = validate_manifest(config)
    if arguments.mode == "full" and manifest["test_partition"] != "confirmatory":
        raise RuntimeError("RQ4 full mode requires the confirmatory manifest")
    if arguments.mode == "mini" and manifest["test_partition"] != "diagnostic":
        raise RuntimeError("RQ4 mini mode requires the diagnostic manifest")

    if arguments.mode == "smoke":
        first_path = extract_split(
            config, "train", arguments.smoke_images,
            output_override=project_path(config, "RQ4/outputs/smoke_features.parquet"),
            shared_cache_namespace="rq4_smoke_a",
        )
        payload: dict[str, Any] = {
            "status": "smoke_pass", "evidence_status": "diagnostic_not_scientific_evidence",
            "manifest": str(manifest_path), "images": arguments.smoke_images, "features": str(first_path),
        }
        if not arguments.skip_repeatability_check:
            second_path = extract_split(
                config, "train", arguments.smoke_images,
                output_override=project_path(config, "RQ4/outputs/repeatability/smoke_features.parquet"),
                shared_cache_namespace="rq4_smoke_b",
            )
            first = pd.read_parquet(first_path)
            second = pd.read_parquet(second_path)
            pd.testing.assert_frame_equal(first, second, check_exact=True, check_like=False)
            first_hash = sha256_file(first_path)
            second_hash = sha256_file(second_path)
            if first_hash != second_hash:
                raise RuntimeError("RQ4 smoke Parquet hashes differ despite equal frames")
            payload["repeatability"] = {
                "status": "exact_dataframe_and_parquet_match", "first_sha256": first_hash,
                "repeat_sha256": second_hash, "namespaces": ["rq4_smoke_a", "rq4_smoke_b"],
            }
        print(json.dumps(payload, indent=2))
        return

    limits = (
        {"train": arguments.mini_train_images, "validation": arguments.mini_validation_images, "test": arguments.mini_test_images}
        if arguments.mode == "mini" else {"train": None, "validation": None, "test": None}
    )
    feature_paths = {
        split: extract_split(config, split, limit)
        for split, limit in limits.items() if split != "test"
    }
    models = fit_calibrations(config)
    feature_paths["test"] = extract_split(config, "test", limits["test"])
    metrics = evaluate_calibrations(config)
    report = generate_report(config)
    split_metadata = {split: _metadata(path) for split, path in feature_paths.items()}
    fingerprints = {metadata["shared_fingerprint"] for metadata in split_metadata.values()}
    if len(fingerprints) != 1:
        raise RuntimeError("RQ4 splits consumed different shared fingerprints")
    if not all(metadata["receipt_hash_valid"] for metadata in split_metadata.values()):
        raise RuntimeError("RQ4 shared request receipt hash audit failed")
    alignment = _alignment_audit(project_root, feature_paths) if arguments.mode == "mini" else {"status": "not_run_for_full"}
    print(
        json.dumps(
            {
                "status": f"{arguments.mode}_e2e_pass", "evidence_status": metrics["evidence_status"],
                "manifest": str(manifest_path),
                "images": {
                    split: int(metadata["images_requested"])
                    for split, metadata in split_metadata.items()
                },
                "features": {name: str(path) for name, path in feature_paths.items()},
                "models": sorted(models), "test_rows": metrics["test_rows"],
                "primary_inference": metrics["primary_inference"],
                "shared_cache_audit": {
                    split: {
                        "shared_fingerprint": metadata["shared_fingerprint"],
                        "shared_shards_computed": metadata["shared_shards_computed"],
                        "shared_shards_reused": metadata["shared_shards_reused"],
                        "feature_shards_recomputed": metadata["feature_shards_recomputed"],
                        "feature_shards_reused": metadata["feature_shards_reused"],
                        "receipt_hash_valid": metadata["receipt_hash_valid"],
                    }
                    for split, metadata in split_metadata.items()
                },
                "zero_duplicate_canonical_gpu_inference": all(metadata["shared_shards_computed"] == 0 for metadata in split_metadata.values()),
                "common_universe_alignment": alignment,
                "computational_cost": {key: value for key, value in metrics["computational_cost"].items() if key not in {"environment", "stochastic_modules"}},
                "artifact_integrity": metrics["artifact_integrity"],
                "report": {name: str(path) for name, path in report.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
