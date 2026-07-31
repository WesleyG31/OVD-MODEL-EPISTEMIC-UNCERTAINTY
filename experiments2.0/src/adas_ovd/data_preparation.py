from __future__ import annotations

import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from PIL import Image
from tqdm import tqdm

from .config import project_path
from .reproducibility import portable_path, sha256_file, write_json


CATEGORY_NAMES = (
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "traffic light",
    "traffic sign",
)

CATEGORY_ALIASES = {
    "pedestrian": "person",
    "person": "person",
    "rider": "rider",
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "train": "train",
    "motor": "motorcycle",
    "motorcycle": "motorcycle",
    "bike": "bicycle",
    "bicycle": "bicycle",
    "traffic light": "traffic light",
    "traffic sign": "traffic sign",
}


@dataclass(frozen=True)
class RawBddPaths:
    root: Path
    images_dir: Path
    labels_file: Path


def _matching_directories(root: Path, name: str) -> list[Path]:
    return sorted(
        path
        for path in root.rglob(name)
        if path.is_dir() and "100k" in {part.lower() for part in path.parts}
    )


def discover_bdd100k_validation(root: str | Path) -> RawBddPaths:
    root = Path(root).resolve()
    label_names = (
        "bdd100k_labels_images_val.json",
        "det_val.json",
    )
    label_candidates = sorted(
        {
            path.resolve()
            for label_name in label_names
            for path in root.rglob(label_name)
            if path.is_file()
        }
    )
    image_candidates = [
        path.resolve()
        for path in _matching_directories(root, "val")
        if any(path.glob("*.jpg"))
    ]

    if len(label_candidates) != 1:
        raise RuntimeError(
            "Expected exactly one BDD100K validation label file below "
            f"{root}, found {len(label_candidates)}: {label_candidates}"
        )
    if not image_candidates:
        raise RuntimeError(
            f"No BDD100K 100k/val image directory was found below {root}"
        )
    counts = {
        candidate: sum(1 for _ in candidate.glob("*.jpg"))
        for candidate in image_candidates
    }
    images_dir = max(counts, key=counts.get)
    return RawBddPaths(
        root=root,
        images_dir=images_dir,
        labels_file=label_candidates[0],
    )


def download_kaggle_source(
    handle: str,
    output_dir: str | Path,
    resource_paths: Iterable[str] | None = None,
    force: bool = False,
) -> Path:
    import kagglehub

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    requested_paths = tuple(resource_paths or ())
    if requested_paths:
        for resource_path in requested_paths:
            kagglehub.dataset_download(
                handle,
                path=resource_path,
                output_dir=str(destination),
                force_download=force,
            )
    else:
        kagglehub.dataset_download(
            handle,
            output_dir=str(destination),
            force_download=force,
        )
    return destination


def _categories() -> list[dict[str, Any]]:
    return [
        {"id": index, "name": name, "supercategory": "object"}
        for index, name in enumerate(CATEGORY_NAMES, start=1)
    ]


def _finite_box(box: Mapping[str, Any]) -> tuple[float, float, float, float]:
    coordinates = tuple(
        float(box[key]) for key in ("x1", "y1", "x2", "y2")
    )
    if not all(math.isfinite(value) for value in coordinates):
        raise ValueError(f"Non-finite BDD100K box: {box}")
    return coordinates


def convert_bdd_detection_to_coco(
    labels_file: str | Path,
    images_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    labels_file = Path(labels_file).resolve()
    images_dir = Path(images_dir).resolve()
    with labels_file.open("r", encoding="utf-8") as handle:
        frames = json.load(handle)
    if not isinstance(frames, list):
        raise ValueError("BDD100K detection labels must be a JSON list")

    categories = _categories()
    category_ids = {item["name"]: item["id"] for item in categories}
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    raw_box_categories: Counter[str] = Counter()
    normalized_categories: Counter[str] = Counter()
    missing_images: list[str] = []
    invalid_boxes: list[dict[str, Any]] = []
    annotation_id = 1

    for image_id, frame in enumerate(tqdm(frames, desc="BDD100K -> COCO")):
        file_name = str(frame["name"])
        image_path = images_dir / file_name
        if not image_path.is_file():
            missing_images.append(file_name)
            continue
        with Image.open(image_path) as image:
            width, height = image.size
            image.verify()

        video_name = frame.get("videoName")
        group_id = str(video_name) if video_name else Path(file_name).stem
        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": int(width),
                "height": int(height),
                "group_id": group_id,
                "bdd_attributes": frame.get("attributes", {}),
            }
        )

        for label in frame.get("labels", []):
            box = label.get("box2d")
            if box is None:
                continue
            raw_category = str(label.get("category", "")).strip().lower()
            raw_box_categories[raw_category] += 1
            category = CATEGORY_ALIASES.get(raw_category)
            if category is None:
                continue
            x1, y1, x2, y2 = _finite_box(box)
            box_width = x2 - x1
            box_height = y2 - y1
            if (
                box_width <= 0
                or box_height <= 0
                or x1 < 0
                or y1 < 0
                or x2 > width + 1e-3
                or y2 > height + 1e-3
            ):
                invalid_boxes.append(
                    {
                        "file_name": file_name,
                        "category": raw_category,
                        "box": [x1, y1, x2, y2],
                        "image_size": [width, height],
                    }
                )
                continue
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_ids[category],
                    "bbox": [x1, y1, box_width, box_height],
                    "area": box_width * box_height,
                    "iscrowd": 0,
                    "source_label_id": label.get("id"),
                    "bdd_attributes": label.get("attributes", {}),
                }
            )
            normalized_categories[category] += 1
            annotation_id += 1

    unmapped_box_categories = {
        name: count
        for name, count in raw_box_categories.items()
        if name not in CATEGORY_ALIASES
    }
    if missing_images:
        raise RuntimeError(
            f"{len(missing_images)} labeled images are missing; examples: "
            f"{missing_images[:10]}"
        )
    if invalid_boxes:
        raise RuntimeError(
            f"{len(invalid_boxes)} invalid boxes found; examples: "
            f"{invalid_boxes[:3]}"
        )
    if unmapped_box_categories:
        raise RuntimeError(
            "Unmapped categories with box2d annotations: "
            f"{unmapped_box_categories}"
        )
    absent = [
        category for category in CATEGORY_NAMES if normalized_categories[category] == 0
    ]
    if absent:
        raise RuntimeError(
            f"Canonical categories without annotations after conversion: {absent}"
        )

    payload = {
        "info": {
            "description": "BDD100K validation detection labels in COCO format",
            "source_labels_sha256": sha256_file(labels_file),
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    conversion_report = {
        "raw_frames": len(frames),
        "converted_images": len(images),
        "converted_annotations": len(annotations),
        "raw_box_categories": dict(sorted(raw_box_categories.items())),
        "normalized_categories": dict(sorted(normalized_categories.items())),
        "aliases": dict(sorted(CATEGORY_ALIASES.items())),
    }
    return payload, conversion_report


def _coco_subset(
    payload: Mapping[str, Any],
    images: Iterable[Mapping[str, Any]],
    description: str,
) -> dict[str, Any]:
    selected_images = [dict(image) for image in images]
    selected_ids = {int(image["id"]) for image in selected_images}
    info = dict(payload["info"])
    info["description"] = description
    return {
        "info": info,
        "licenses": list(payload.get("licenses", [])),
        "images": selected_images,
        "annotations": [
            annotation
            for annotation in payload["annotations"]
            if int(annotation["image_id"]) in selected_ids
        ],
        "categories": list(payload["categories"]),
    }


def deterministic_development_test_split(
    payload: Mapping[str, Any],
    development_fraction: float,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not 0.0 < development_fraction < 1.0:
        raise ValueError("development_fraction must be between zero and one")
    images = [dict(image) for image in payload["images"]]
    random.Random(seed).shuffle(images)
    split_index = int(len(images) * development_fraction)
    development = _coco_subset(
        payload,
        images[:split_index],
        "BDD100K validation development partition",
    )
    test = _coco_subset(
        payload,
        images[split_index:],
        "BDD100K validation held-out test partition",
    )
    return development, test


def _image_inventory(
    images_dir: Path,
    names: Iterable[str],
    include_sha256: bool,
) -> list[dict[str, Any]]:
    inventory = []
    for name in tqdm(sorted(names), desc="Image provenance"):
        path = images_dir / name
        item = {
            "file_name": name,
            "bytes": path.stat().st_size,
        }
        if include_sha256:
            item["sha256"] = sha256_file(path)
        inventory.append(item)
    return inventory


def audit_prepared_data(
    full: Mapping[str, Any],
    development: Mapping[str, Any],
    test: Mapping[str, Any],
    raw: RawBddPaths,
    expected_images: int,
    include_image_sha256: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    errors: list[str] = []
    full_names = [str(item["file_name"]) for item in full["images"]]
    disk_names = {
        path.name for path in raw.images_dir.glob("*.jpg") if path.is_file()
    }
    development_names = {
        str(item["file_name"]) for item in development["images"]
    }
    test_names = {str(item["file_name"]) for item in test["images"]}
    full_ids = [int(item["id"]) for item in full["images"]]
    annotation_ids = [int(item["id"]) for item in full["annotations"]]
    image_id_set = set(full_ids)
    category_id_set = {int(item["id"]) for item in full["categories"]}

    if len(full_names) != expected_images:
        errors.append(
            f"expected {expected_images} images, found {len(full_names)}"
        )
    if len(full_names) != len(set(full_names)):
        errors.append("duplicate image file names")
    if len(disk_names) != expected_images:
        errors.append(
            f"expected {expected_images} validation JPEGs on disk, "
            f"found {len(disk_names)}"
        )
    missing_on_disk = set(full_names) - disk_names
    unexpected_on_disk = disk_names - set(full_names)
    if missing_on_disk:
        errors.append(f"{len(missing_on_disk)} labeled images missing on disk")
    if unexpected_on_disk:
        errors.append(
            f"{len(unexpected_on_disk)} validation JPEGs have no label frame"
        )
    if len(full_ids) != len(set(full_ids)):
        errors.append("duplicate image ids")
    if len(annotation_ids) != len(set(annotation_ids)):
        errors.append("duplicate annotation ids")
    if development_names & test_names:
        errors.append("development/test image leakage")
    if development_names | test_names != set(full_names):
        errors.append("development/test union differs from full validation set")

    orphan_annotations = sum(
        int(annotation["image_id"]) not in image_id_set
        or int(annotation["category_id"]) not in category_id_set
        for annotation in full["annotations"]
    )
    if orphan_annotations:
        errors.append(f"{orphan_annotations} orphan annotations")

    category_by_id = {
        int(item["id"]): str(item["name"]) for item in full["categories"]
    }
    category_counts = Counter(
        category_by_id[int(item["category_id"])]
        for item in full["annotations"]
    )
    absent = [name for name in CATEGORY_NAMES if category_counts[name] == 0]
    if absent:
        errors.append(f"categories without annotations: {absent}")

    group_sets = {
        "development": {
            str(item["group_id"]) for item in development["images"]
        },
        "test": {str(item["group_id"]) for item in test["images"]},
    }
    group_overlap = group_sets["development"] & group_sets["test"]
    if group_overlap:
        errors.append(
            f"development/test source-group leakage ({len(group_overlap)} groups)"
        )

    inventory = _image_inventory(
        raw.images_dir,
        full_names,
        include_sha256=include_image_sha256,
    )
    dimensions = Counter(
        (int(item["width"]), int(item["height"])) for item in full["images"]
    )
    audit = {
        "schema_version": 1,
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "counts": {
            "full_images": len(full["images"]),
            "full_annotations": len(full["annotations"]),
            "development_images": len(development["images"]),
            "development_annotations": len(development["annotations"]),
            "test_images": len(test["images"]),
            "test_annotations": len(test["annotations"]),
        },
        "category_counts": dict(sorted(category_counts.items())),
        "image_dimensions": {
            f"{width}x{height}": count
            for (width, height), count in sorted(dimensions.items())
        },
        "split_overlap": {
            "image_files": len(development_names & test_names),
            "source_groups": len(group_overlap),
        },
        "hash_policy": (
            "sha256-every-used-image"
            if include_image_sha256
            else "file-name-and-byte-size"
        ),
    }
    return audit, inventory


def prepare_bdd100k(
    config: dict[str, Any],
    *,
    source_dir: str | Path | None = None,
    force_download: bool = False,
    force_prepare: bool = False,
    include_image_sha256: bool = True,
) -> dict[str, Any]:
    data_config = config["data"]
    project_root = Path(config["_meta"]["project_root"])
    raw_dir = project_path(config, data_config["raw_dir"])
    processed_dir = project_path(config, data_config["processed_dir"])
    annotation_dir = processed_dir / "annotations"
    configured_images_dir = project_path(config, data_config["images_dir"])

    if source_dir is None:
        existing_raw: RawBddPaths | None = None
        if raw_dir.exists() and not force_download:
            try:
                existing_raw = discover_bdd100k_validation(raw_dir)
            except RuntimeError:
                if any(path.is_file() for path in raw_dir.rglob("*")):
                    raise RuntimeError(
                        "The Kaggle destination is incomplete. Re-run with "
                        "--force-download to replace only data.raw_dir."
                    )
        if existing_raw is None:
            source_root = download_kaggle_source(
                str(data_config["source"]["handle"]),
                raw_dir,
                resource_paths=data_config["source"].get("download_paths"),
                force=force_download,
            )
        else:
            source_root = raw_dir
    else:
        source_root = Path(source_dir).resolve()
    raw = discover_bdd100k_validation(source_root)

    full_path = annotation_dir / "full.json"
    development_path = project_path(
        config, data_config["calibration_annotations"]
    )
    test_path = project_path(config, data_config["evaluation_annotations"])
    if (
        not force_prepare
        and full_path.exists()
        and development_path.exists()
        and test_path.exists()
        and configured_images_dir.exists()
    ):
        with full_path.open("r", encoding="utf-8") as handle:
            full = json.load(handle)
        with development_path.open("r", encoding="utf-8") as handle:
            development = json.load(handle)
        with test_path.open("r", encoding="utf-8") as handle:
            test = json.load(handle)
        current_labels_sha256 = sha256_file(raw.labels_file)
        recorded_labels_sha256 = full.get("info", {}).get(
            "source_labels_sha256"
        )
        if recorded_labels_sha256 != current_labels_sha256:
            raise RuntimeError(
                "Prepared annotations came from different raw labels. "
                "Re-run with --force-prepare."
            )
        expected_development, expected_test = (
            deterministic_development_test_split(
                full,
                development_fraction=float(
                    data_config["original_development_fraction"]
                ),
                seed=int(data_config["original_split_seed"]),
            )
        )
        expected_development_ids = [
            int(item["id"]) for item in expected_development["images"]
        ]
        actual_development_ids = [
            int(item["id"]) for item in development["images"]
        ]
        expected_test_ids = [
            int(item["id"]) for item in expected_test["images"]
        ]
        actual_test_ids = [int(item["id"]) for item in test["images"]]
        if (
            actual_development_ids != expected_development_ids
            or actual_test_ids != expected_test_ids
        ):
            raise RuntimeError(
                "Prepared split does not match the configured seed/fraction. "
                "Re-run with --force-prepare."
            )
        conversion_report = {"reused": True}
    else:
        full, conversion_report = convert_bdd_detection_to_coco(
            raw.labels_file, raw.images_dir
        )
        development, test = deterministic_development_test_split(
            full,
            development_fraction=float(
                data_config["original_development_fraction"]
            ),
            seed=int(data_config["original_split_seed"]),
        )
        annotation_dir.mkdir(parents=True, exist_ok=True)
        write_json(full_path, full)
        write_json(development_path, development)
        write_json(test_path, test)
    if configured_images_dir.resolve() != raw.images_dir.resolve():
        raise RuntimeError(
            "The pinned Kaggle layout does not match data.images_dir. "
            f"Configured: {configured_images_dir}; discovered: {raw.images_dir}. "
            "Update the pinned path instead of silently copying data."
        )

    audit, inventory = audit_prepared_data(
        full,
        development,
        test,
        raw,
        expected_images=int(data_config["expected_validation_images"]),
        include_image_sha256=include_image_sha256,
    )
    provenance = {
        "schema_version": 1,
        "source": dict(data_config["source"]),
        "resolved_download_root": portable_path(source_root, project_root),
        "raw_images_dir": portable_path(raw.images_dir, project_root),
        "raw_labels_file": portable_path(raw.labels_file, project_root),
        "raw_labels_sha256": sha256_file(raw.labels_file),
        "processed": {
            "full_annotations": portable_path(full_path, project_root),
            "full_annotations_sha256": sha256_file(full_path),
            "development_annotations": portable_path(
                development_path, project_root
            ),
            "development_annotations_sha256": sha256_file(development_path),
            "test_annotations": portable_path(test_path, project_root),
            "test_annotations_sha256": sha256_file(test_path),
        },
        "conversion": conversion_report,
        "images": inventory,
    }
    write_json(project_path(config, data_config["audit_report"]), audit)
    write_json(
        project_path(config, data_config["provenance_manifest"]),
        provenance,
    )
    if audit["status"] != "pass":
        raise RuntimeError(f"BDD100K data audit failed: {audit['errors']}")
    return audit


def require_passing_data_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    data_config = config["data"]
    audit_path = project_path(config, data_config["audit_report"])
    provenance_path = project_path(
        config, data_config["provenance_manifest"]
    )
    if not audit_path.is_file() or not provenance_path.is_file():
        raise RuntimeError(
            "Prepared-data audit/provenance is missing. Run "
            "experiments2.0/scripts/prepare_data.py first."
        )
    with audit_path.open("r", encoding="utf-8") as handle:
        audit = json.load(handle)
    with provenance_path.open("r", encoding="utf-8") as handle:
        provenance = json.load(handle)
    if audit.get("status") != "pass":
        raise RuntimeError(
            f"Prepared-data audit did not pass: {audit.get('errors', [])}"
        )

    expected_handle = str(data_config["source"]["handle"])
    actual_handle = str(provenance.get("source", {}).get("handle"))
    if actual_handle != expected_handle:
        raise RuntimeError(
            f"Dataset provenance mismatch: {actual_handle} != {expected_handle}"
        )
    checks = (
        ("calibration_annotations", "development_annotations"),
        ("evaluation_annotations", "test_annotations"),
    )
    for config_key, provenance_key in checks:
        path = project_path(config, data_config[config_key])
        recorded = provenance["processed"][f"{provenance_key}_sha256"]
        if not path.is_file() or sha256_file(path) != recorded:
            raise RuntimeError(
                f"Prepared annotation integrity check failed for {path}"
            )
    return audit
