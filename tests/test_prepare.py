"""DatasetPreparer / dataset_info 单元测试（不依赖 torch）。"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from safeyolo.dataset.info import dataset_info
from safeyolo.dataset.prepare import DatasetPreparer


@pytest.fixture()
def tiny_raw(tmp_path: Path) -> Path:
    """构造 6 张图的 YOLO 原生数据：raw/images + raw/original_annotations。"""
    raw = tmp_path / "raw"
    (raw / "images").mkdir(parents=True)
    (raw / "original_annotations").mkdir(parents=True)
    for i in range(6):
        (raw / "images" / f"img{i}.jpg").write_bytes(b"fake-image")
        (raw / "original_annotations" / f"img{i}.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n1 0.1 0.1 0.1 0.1\n", encoding="utf-8"
        )
    return raw


class TestDatasetPreparer:
    def test_split_and_yaml(self, tiny_raw: Path, tmp_path: Path) -> None:
        out = tmp_path / "dataset"
        preparer = DatasetPreparer(
            raw_root=tiny_raw,
            output_root=out,
            annotation_format="yolo",
            classes=["person", "helmet"],
            seed=42,
        )
        yaml_path = preparer.prepare()

        config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        assert config["names"] == ["person", "helmet"]
        assert config["nc"] == 2
        assert Path(config["path"]).resolve() == out.resolve()

        counts = {
            split: len(list((out / "images" / split).glob("*.jpg")))
            for split in ("train", "val", "test")
        }
        assert sum(counts.values()) == 6
        assert counts["train"] == 4  # train_rate=0.8 -> 4.8 -> 4
        # 图片与标签一一对应
        for split in ("train", "val", "test"):
            imgs = {p.stem for p in (out / "images" / split).iterdir()}
            labels = {p.stem for p in (out / "labels" / split).iterdir()}
            assert imgs == labels

    def test_auto_format_detection(self, tiny_raw: Path, tmp_path: Path) -> None:
        preparer = DatasetPreparer(
            raw_root=tiny_raw,
            output_root=tmp_path / "dataset",
            annotation_format="auto",
            classes=["person", "helmet"],
        )
        # raw/original_annotations 下是 txt -> 应识别为 yolo
        assert preparer._resolve_format() == "yolo"

    def test_invalid_rates_raise(self, tiny_raw: Path, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            DatasetPreparer(raw_root=tiny_raw, output_root=tmp_path, train_rate=1.5)
        with pytest.raises(ValueError):
            DatasetPreparer(raw_root=tiny_raw, output_root=tmp_path, train_rate=0.9, valid_rate=0.2)

    def test_no_pairs_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "raw"
        (empty / "images").mkdir(parents=True)
        (empty / "original_annotations").mkdir(parents=True)
        preparer = DatasetPreparer(
            raw_root=empty,
            output_root=tmp_path / "dataset",
            annotation_format="yolo",
            classes=["a"],
        )
        with pytest.raises(RuntimeError, match="未找到任何可配对"):
            preparer.prepare()


class TestDatasetInfo:
    def test_counts_and_names(self, tiny_raw: Path, tmp_path: Path) -> None:
        out = tmp_path / "dataset"
        yaml_path = DatasetPreparer(
            raw_root=tiny_raw, output_root=out, annotation_format="yolo", classes=["person", "helmet"]
        ).prepare()

        info = dataset_info(yaml_path)
        assert info["names"] == ["person", "helmet"]
        assert info["nc"] == 2
        assert info["total"] == 6
        assert info["splits"]["train"] == 4
