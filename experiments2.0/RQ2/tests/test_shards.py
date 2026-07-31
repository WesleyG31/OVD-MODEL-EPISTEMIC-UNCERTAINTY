from __future__ import annotations

from pathlib import Path

import pandas as pd

from rq2.extraction import _valid_shard, _write_shard


def test_shard_hash_sidecar_detects_corruption(tmp_path: Path) -> None:
    detection = tmp_path / "1.parquet"
    image = tmp_path / "1_image.parquet"
    metadata = tmp_path / "1.json"
    _write_shard(
        detection,
        image,
        metadata,
        pd.DataFrame({"image_id": [1], "score": [0.5]}),
        {"image_id": 1, "reference_detections": 1},
        "fingerprint",
    )
    assert _valid_shard(detection, image, metadata, 1, "fingerprint")
    with detection.open("ab") as handle:
        handle.write(b"corrupt")
    assert not _valid_shard(detection, image, metadata, 1, "fingerprint")

