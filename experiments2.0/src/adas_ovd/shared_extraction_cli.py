from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import load_config, project_path
from .reproducibility import sha256_file
from .shared_extraction import (
    ARRAY_SCHEMA,
    ensure_shared_split,
    shared_identity,
    validate_consumer_compatibility,
)


def _manifest_path(config: dict[str, Any], override: str | None) -> Path:
    if override:
        return Path(override).resolve()
    if "rq1" in config and "outputs" in config["rq1"]:
        return project_path(config, config["rq1"]["outputs"]["manifest"])
    if "rq2" in config and "manifest" in config["rq2"]:
        return project_path(config, config["rq2"]["manifest"]["path"])
    if "rq3" in config and "manifest" in config["rq3"]:
        return project_path(config, config["rq3"]["manifest"]["path"])
    raise RuntimeError(
        "Pass --manifest when the consumer config does not declare an RQ1/RQ2/RQ3 manifest"
    )


def _request_payload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {
        "request_metadata": str(path),
        "request_metadata_sha256": sha256_file(path),
        "images": payload["images"],
        "shards_computed": payload["shards_computed"],
        "shards_reused": payload["shards_reused"],
        "total_shard_bytes": payload["total_shard_bytes"],
        "elapsed_seconds": payload["elapsed_seconds"],
        "shard_inventory_sha256": payload["shard_inventory_sha256"],
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Create or validate neutral GroundingDINO inference shards for all "
            "compatible research questions"
        )
    )
    parser.add_argument("--config")
    parser.add_argument("--manifest")
    parser.add_argument("--consumer", help="Optional compatibility check, e.g. rq1")
    parser.add_argument(
        "--mode", choices=("smoke", "mini", "full"), default="smoke"
    )
    parser.add_argument(
        "--splits", nargs="+", choices=("train", "validation", "test")
    )
    parser.add_argument("--smoke-images", type=int, default=2)
    parser.add_argument("--mini-train-images", type=int, default=6)
    parser.add_argument("--mini-validation-images", type=int, default=4)
    parser.add_argument("--mini-test-images", type=int, default=8)
    parser.add_argument("--cache-namespace")
    arguments = parser.parse_args()

    if arguments.config:
        config_path = Path(arguments.config)
    else:
        config_name = "rq1_mini.yaml" if arguments.mode == "mini" else "rq1.yaml"
        config_path = project_root / "RQ1" / "configs" / config_name
    config = load_config(config_path)
    if arguments.consumer:
        validate_consumer_compatibility(config, arguments.consumer.lower())
    manifest = _manifest_path(config, arguments.manifest)
    with manifest.open("r", encoding="utf-8") as handle:
        manifest_payload = json.load(handle)
    if arguments.mode == "full" and manifest_payload.get("test_partition") != "confirmatory":
        raise RuntimeError("Full shared extraction requires a confirmatory manifest")
    if arguments.mode == "mini" and manifest_payload.get("test_partition") != "diagnostic":
        raise RuntimeError("Mini shared extraction requires a diagnostic manifest")

    if arguments.splits:
        splits = arguments.splits
    elif arguments.mode == "smoke":
        splits = ["train"]
    else:
        splits = ["train", "validation", "test"]
    limits = {
        "smoke": {"train": arguments.smoke_images, "validation": arguments.smoke_images, "test": arguments.smoke_images},
        "mini": {
            "train": arguments.mini_train_images,
            "validation": arguments.mini_validation_images,
            "test": arguments.mini_test_images,
        },
        "full": {"train": None, "validation": None, "test": None},
    }[arguments.mode]
    namespace = arguments.cache_namespace
    if namespace is None and arguments.mode == "smoke":
        namespace = "shared_cli_smoke"

    split_results: dict[str, Any] = {}
    for split in splits:
        shared = ensure_shared_split(
            config,
            manifest_path=manifest,
            split=split,
            limit=limits[split],
            cache_namespace=namespace,
        )
        split_results[split] = _request_payload(shared.request_metadata_path)
    identity = shared_identity(config)
    print(
        json.dumps(
            {
                "status": f"shared_{arguments.mode}_pass",
                "schema_version": identity["schema_version"],
                "shared_fingerprint": identity["configuration_fingerprint"],
                "namespace": namespace or config["shared_extraction"]["cache_namespace"],
                "manifest": str(manifest),
                "manifest_sha256": sha256_file(manifest),
                "passes_per_image": 1 + int(config["shared_extraction"]["mc_passes"]),
                "array_schema": ARRAY_SCHEMA,
                "splits": split_results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
