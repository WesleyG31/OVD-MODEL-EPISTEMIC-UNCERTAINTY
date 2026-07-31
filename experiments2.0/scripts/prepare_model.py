from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from pathlib import Path
from typing import Any

from adas_ovd.config import load_config, project_path
from adas_ovd.reproducibility import portable_path, sha256_file, write_json


def _prepare_text_encoder(
    config: dict[str, Any], force: bool
) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    text_config = config["model"]["text_encoder"]
    destination = project_path(config, text_config["local_dir"])
    required = [str(name) for name in text_config["required_files"]]
    missing = [
        name for name in required if not (destination / name).is_file()
    ]
    acquisition = "verified_existing"
    if force or missing:
        destination.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=str(text_config["repository"]),
            revision=str(text_config["revision"]),
            allow_patterns=required,
            local_dir=str(destination),
            force_download=force,
        )
        acquisition = "downloaded"
    missing = [
        name for name in required if not (destination / name).is_file()
    ]
    if missing:
        raise SystemExit(f"Text encoder files are missing: {missing}")
    return {
        "repository": str(text_config["repository"]),
        "revision": str(text_config["revision"]),
        "path": portable_path(
            destination, config["_meta"]["project_root"]
        ),
        "acquisition": acquisition,
        "files": {
            name: {
                "bytes": (destination / name).stat().st_size,
                "sha256": sha256_file(destination / name),
            }
            for name in required
        },
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Download and verify the pinned GroundingDINO checkpoint"
    )
    parser.add_argument(
        "--config",
        default=str(project_root / "configs" / "base.yaml"),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--source",
        help="Import an existing checkpoint after verifying the pinned SHA256",
    )
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    destination = project_path(config, config["model"]["checkpoint"])
    expected = str(config["model"]["checkpoint_sha256"]).lower()
    acquisition = "verified_existing"

    if destination.exists() and not arguments.force:
        actual = sha256_file(destination)
        if actual != expected:
            raise SystemExit(
                f"Checkpoint hash mismatch: expected {expected}, found {actual}"
            )
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".download")
        if arguments.source:
            shutil.copyfile(Path(arguments.source).resolve(), temporary)
            acquisition = "imported_verified_copy"
        else:
            urllib.request.urlretrieve(
                str(config["model"]["checkpoint_url"]), temporary
            )
            acquisition = "downloaded"
        actual = sha256_file(temporary)
        if actual != expected:
            temporary.unlink(missing_ok=True)
            raise SystemExit(
                f"Downloaded checkpoint hash mismatch: {actual}"
            )
        temporary.replace(destination)

    text_encoder = _prepare_text_encoder(config, arguments.force)
    provenance = {
        "schema_version": 1,
        "path": portable_path(destination, config["_meta"]["project_root"]),
        "url": str(config["model"]["checkpoint_url"]),
        "acquisition": acquisition,
        "source_filename": (
            Path(arguments.source).name if arguments.source else None
        ),
        "sha256": expected,
        "bytes": destination.stat().st_size,
        "text_encoder": text_encoder,
    }
    write_json(project_path(config, "artifacts/model_provenance.json"), provenance)
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
