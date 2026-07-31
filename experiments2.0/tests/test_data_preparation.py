import json
from pathlib import Path

from PIL import Image

from adas_ovd.data_preparation import (
    CATEGORY_NAMES,
    convert_bdd_detection_to_coco,
    deterministic_development_test_split,
)


def test_conversion_normalizes_legacy_bdd_categories(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    labels = []
    raw_categories = [
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motor",
        "bike",
        "traffic light",
        "traffic sign",
    ]
    for index, category in enumerate(raw_categories):
        name = f"frame-{index}.jpg"
        Image.new("RGB", (32, 24)).save(images_dir / name)
        labels.append(
            {
                "name": name,
                "labels": [
                    {
                        "id": str(index),
                        "category": category,
                        "box2d": {"x1": 1, "y1": 2, "x2": 12, "y2": 14},
                    }
                ],
            }
        )
    labels_file = tmp_path / "bdd100k_labels_images_val.json"
    labels_file.write_text(json.dumps(labels), encoding="utf-8")

    coco, report = convert_bdd_detection_to_coco(labels_file, images_dir)

    assert len(coco["images"]) == len(CATEGORY_NAMES)
    assert len(coco["annotations"]) == len(CATEGORY_NAMES)
    assert report["normalized_categories"]["motorcycle"] == 1
    assert report["normalized_categories"]["bicycle"] == 1


def test_split_is_deterministic_and_disjoint(tmp_path: Path) -> None:
    payload = {
        "info": {},
        "licenses": [],
        "categories": [],
        "images": [
            {
                "id": index,
                "file_name": f"{index}.jpg",
                "group_id": str(index),
            }
            for index in range(10)
        ],
        "annotations": [],
    }
    first = deterministic_development_test_split(payload, 0.8, 42)
    second = deterministic_development_test_split(payload, 0.8, 42)

    assert first == second
    development, test = first
    development_ids = {item["id"] for item in development["images"]}
    test_ids = {item["id"] for item in test["images"]}
    assert len(development_ids) == 8
    assert len(test_ids) == 2
    assert not development_ids & test_ids
