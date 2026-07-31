from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from .reproducibility import portable_path, sha256_file


@dataclass(frozen=True)
class ImageRecord:
    image_id: int
    file_name: str
    width: int
    height: int
    source_group_id: str
    attributes: dict[str, str] = field(default_factory=dict)

    @property
    def sequence_id(self) -> str:
        return self.source_group_id


@dataclass(frozen=True)
class GroundTruth:
    annotation_id: int
    image_id: int
    category_id: int
    bbox_xyxy: tuple[float, float, float, float]
    iscrowd: bool


class CocoDataset:
    def __init__(self, annotation_path: str | Path, images_dir: str | Path):
        self.annotation_path = Path(annotation_path).resolve()
        self.images_dir = Path(images_dir).resolve()
        with self.annotation_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        self.images = {
            int(item["id"]): ImageRecord(
                image_id=int(item["id"]),
                file_name=item["file_name"],
                width=int(item["width"]),
                height=int(item["height"]),
                source_group_id=str(
                    item.get("group_id") or Path(item["file_name"]).stem
                ),
                attributes={
                    str(key): str(value)
                    for key, value in item.get("bdd_attributes", {}).items()
                },
            )
            for item in payload["images"]
        }
        self.categories = {
            int(item["id"]): str(item["name"]) for item in payload["categories"]
        }
        self.category_ids_by_name = {
            name: category_id for category_id, name in self.categories.items()
        }
        annotations: dict[int, list[GroundTruth]] = defaultdict(list)
        for item in payload["annotations"]:
            x, y, width, height = (float(v) for v in item["bbox"])
            annotations[int(item["image_id"])].append(
                GroundTruth(
                    annotation_id=int(item["id"]),
                    image_id=int(item["image_id"]),
                    category_id=int(item["category_id"]),
                    bbox_xyxy=(x, y, x + width, y + height),
                    iscrowd=bool(item.get("iscrowd", 0)),
                )
            )
        self.annotations = dict(annotations)

    @property
    def sha256(self) -> str:
        return sha256_file(self.annotation_path)

    def image_path(self, image_id: int) -> Path:
        path = self.images[image_id]
        return self.images_dir / path.file_name

    def ground_truth(self, image_id: int, include_crowd: bool = False) -> list[GroundTruth]:
        records = self.annotations.get(image_id, [])
        if include_crowd:
            return records
        return [record for record in records if not record.iscrowd]


def create_sequence_manifest(
    calibration: CocoDataset,
    evaluation: CocoDataset,
    train_fraction: float,
    seed: int,
    project_root: str | Path | None = None,
) -> dict:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between zero and one")

    image_ids = np.array(sorted(calibration.images), dtype=np.int64)
    groups = np.array(
        [calibration.images[int(image_id)].sequence_id for image_id in image_ids]
    )
    splitter = GroupShuffleSplit(
        n_splits=1, train_size=train_fraction, random_state=seed
    )
    train_indices, validation_indices = next(
        splitter.split(image_ids, groups=groups)
    )
    train_ids = image_ids[train_indices]
    validation_ids = image_ids[validation_indices]

    train_groups = set(groups[train_indices])
    validation_groups = set(groups[validation_indices])
    overlap = train_groups & validation_groups
    if overlap:
        raise RuntimeError(f"Sequence leakage detected: {sorted(overlap)[:5]}")

    test_ids = np.array(sorted(evaluation.images), dtype=np.int64)
    test_groups = {
        evaluation.images[int(image_id)].sequence_id
        for image_id in test_ids
    }
    development_groups = train_groups | validation_groups
    test_overlap = development_groups & test_groups
    if test_overlap:
        raise RuntimeError(
            "Development/test source-group leakage detected: "
            f"{sorted(test_overlap)[:5]}"
        )
    calibration_path = (
        portable_path(calibration.annotation_path, project_root)
        if project_root is not None
        else str(calibration.annotation_path)
    )
    evaluation_path = (
        portable_path(evaluation.annotation_path, project_root)
        if project_root is not None
        else str(evaluation.annotation_path)
    )
    return {
        "schema_version": 3,
        "seed": seed,
        "train_fraction": train_fraction,
        "group_definition": "BDD videoName, otherwise the unique image stem",
        "calibration_annotations": calibration_path,
        "calibration_sha256": calibration.sha256,
        "evaluation_annotations": evaluation_path,
        "evaluation_sha256": evaluation.sha256,
        "splits": {
            "train": [int(value) for value in sorted(train_ids.tolist())],
            "validation": [
                int(value) for value in sorted(validation_ids.tolist())
            ],
            "test": [int(value) for value in test_ids.tolist()],
        },
        "sequence_counts": {
            "train": len(train_groups),
            "validation": len(validation_groups),
            "test": len(test_groups),
        },
        "group_overlap_counts": {
            "train_validation": len(train_groups & validation_groups),
            "development_test": len(test_overlap),
        },
    }


def load_manifest_ids(manifest_path: str | Path, split: str) -> list[int]:
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    try:
        values = manifest["splits"][split]
    except KeyError as error:
        raise KeyError(f"Unknown manifest split {split!r}") from error
    return [int(value) for value in values]
