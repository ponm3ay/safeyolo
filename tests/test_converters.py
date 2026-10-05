"""COCO / Pascal VOC -> YOLO 转换器单元测试（不依赖 torch）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from safeyolo.dataset.converters import coco_to_yolo, convert_to_yolo, voc_to_yolo


@pytest.fixture()
def voc_dir(tmp_path: Path) -> Path:
    """构造两个 VOC XML：一张含 person(0)/helmet(1)，一张仅 helmet。"""
    annotations = tmp_path / "original_annotations"
    annotations.mkdir()
    (annotations / "img1.xml").write_text(
        """<annotation><size><width>100</width><height>50</height></size>
        <object><name>person</name><bndbox><xmin>0</xmin><ymin>0</ymin><xmax>50</xmax><ymax>50</ymax></bndbox></object>
        <object><name>helmet</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>30</xmax><ymax>30</ymax></bndbox></object>
        </annotation>""",
        encoding="utf-8",
    )
    (annotations / "img2.xml").write_text(
        """<annotation><size><width>200</width><height>100</height></size>
        <object><name>helmet</name><bndbox><xmin>100</xmin><ymin>0</ymin><xmax>200</xmax><ymax>100</ymax></bndbox></object>
        </annotation>""",
        encoding="utf-8",
    )
    return annotations


class TestVocConverter:
    def test_auto_class_order_is_sorted(self, voc_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "yolo"
        classes = voc_to_yolo(voc_dir, out)
        assert classes == ["helmet", "person"]

    def test_txt_content_normalized(self, voc_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "yolo"
        classes = voc_to_yolo(voc_dir, out, class_order=["person", "helmet"])
        assert classes == ["person", "helmet"]
        lines = (out / "img1.txt").read_text(encoding="utf-8").splitlines()
        # person: center (25,25), w=0.5, h=1.0 -> id 0
        assert lines[0] == "0 0.250000 0.500000 0.500000 1.000000"
        # helmet: center (20,20), w=0.2, h=0.4 -> id 1
        assert lines[1] == "1 0.200000 0.400000 0.200000 0.400000"

    def test_invalid_box_skipped(self, voc_dir: Path, tmp_path: Path) -> None:
        (voc_dir / "bad.xml").write_text(
            """<annotation><size><width>10</width><height>10</height></size>
            <object><name>person</name><bndbox><xmin>5</xmin><ymin>0</ymin><xmax>5</xmax><ymax>5</ymax></bndbox></object>
            </annotation>""",
            encoding="utf-8",
        )
        out = tmp_path / "yolo"
        voc_to_yolo(voc_dir, out)
        assert not (out / "bad.txt").exists()

    def test_missing_dir_returns_empty(self, tmp_path: Path) -> None:
        assert voc_to_yolo(tmp_path / "nope", tmp_path / "out") == []


@pytest.fixture()
def coco_dir(tmp_path: Path) -> Path:
    """构造一个 COCO JSON：2 张图、2 个类别、3 个标注。"""
    annotations = tmp_path / "original_annotations"
    annotations.mkdir()
    data = {
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 100, "height": 100},
            {"id": 2, "file_name": "b.jpg", "width": 50, "height": 50},
        ],
        "categories": [
            {"id": 7, "name": "helmet"},
            {"id": 3, "name": "person"},
        ],
        "annotations": [
            {"image_id": 1, "category_id": 3, "bbox": [0, 0, 50, 50]},
            {"image_id": 1, "category_id": 7, "bbox": [25, 25, 50, 50]},
            {"image_id": 2, "category_id": 7, "bbox": [0, 0, 50, 50]},
        ],
    }
    (annotations / "instances.json").write_text(json.dumps(data), encoding="utf-8")
    return annotations


class TestCocoConverter:
    def test_conversion(self, coco_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "yolo"
        classes = coco_to_yolo(coco_dir / "instances.json", out)
        assert classes == ["helmet", "person"]  # 按名称排序
        lines = (out / "a.txt").read_text(encoding="utf-8").splitlines()
        # person bbox [0,0,50,50] @100x100 -> center(0.25,0.25) w=0.5 h=0.5
        assert lines[0] == "1 0.250000 0.250000 0.500000 0.500000"
        assert len(lines) == 2
        assert (out / "b.txt").exists()

    def test_class_order_override(self, coco_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "yolo"
        classes = coco_to_yolo(coco_dir / "instances.json", out, class_order=["person", "helmet"])
        assert classes == ["person", "helmet"]
        first = (out / "a.txt").read_text(encoding="utf-8").splitlines()[0]
        assert first.startswith("0 ")  # person 现在是 id 0

    def test_broken_json_returns_empty(self, tmp_path: Path) -> None:
        broken = tmp_path / "broken.json"
        broken.write_text("{not json", encoding="utf-8")
        assert coco_to_yolo(broken, tmp_path / "out") == []


class TestDispatcher:
    def test_dispatch_voc(self, voc_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "yolo"
        classes = convert_to_yolo(voc_dir, "pascal_voc", out)
        assert classes == ["helmet", "person"]
        assert (out / "img1.txt").exists()

    def test_dispatch_yolo_copies_and_infers(self, tmp_path: Path) -> None:
        src = tmp_path / "original_annotations"
        src.mkdir()
        (src / "x.txt").write_text("1 0.5 0.5 0.2 0.2\n2 0.1 0.1 0.1 0.1\n", encoding="utf-8")
        out = tmp_path / "staged"
        classes = convert_to_yolo(src, "yolo", out)
        assert (out / "x.txt").exists()
        assert classes == ["class_1", "class_2"]

    def test_unsupported_format_raises(self, tmp_path: Path) -> None:
        import pytest

        with pytest.raises(ValueError, match="不支持的标注格式"):
            convert_to_yolo(tmp_path, "csv", tmp_path / "out")
