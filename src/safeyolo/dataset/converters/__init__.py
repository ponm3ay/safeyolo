"""标注格式转换统一入口。"""

from __future__ import annotations

import logging
from pathlib import Path

from safeyolo.dataset.converters.coco import coco_to_yolo
from safeyolo.dataset.converters.pascal_voc import voc_to_yolo

__all__ = ["convert_to_yolo", "detect_annotation_format"]

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = ("yolo", "coco", "pascal_voc")


def detect_annotation_format(raw_dir: str | Path) -> str:
    """根据 raw 目录结构自动推断标注格式，优先级 COCO > Pascal VOC > YOLO。

    Raises:
        ValueError: 无法识别时抛出，提示手动指定格式。
    """
    raw_dir = Path(raw_dir)
    if (raw_dir / "annotations").exists() or list(raw_dir.glob("*.json")):
        return "coco"
    if (raw_dir / "Annotations").exists() or list((raw_dir / "original_annotations").glob("*.xml")):
        return "pascal_voc"
    if (raw_dir / "images").exists() and (
        list((raw_dir / "original_annotations").glob("*.txt")) or (raw_dir / "yolo_staged_labels").exists()
    ):
        return "yolo"
    raise ValueError(f"无法自动识别标注格式，请手动指定 --format。已检查目录: {raw_dir}")


def convert_to_yolo(
    annotations_dir: str | Path,
    annotation_format: str,
    yolo_output_dir: str | Path,
    class_order: list[str] | None = None,
) -> list[str]:
    """将原始标注统一转换为 YOLO txt 格式。

    Args:
        annotations_dir: 原始标注目录（COCO 传入包含 json 的目录）。
        annotation_format: ``yolo`` / ``coco`` / ``pascal_voc``。
        yolo_output_dir: YOLO txt 输出目录。
        class_order: 指定类别顺序，None 时自动确定。

    Returns:
        类别名列表；转换失败返回空列表。
    """
    annotations_dir = Path(annotations_dir)
    yolo_output_dir = Path(yolo_output_dir)

    if annotation_format == "yolo":
        txt_files = list(annotations_dir.glob("*.txt"))
        if not txt_files:
            logger.error("YOLO 标注目录下无 txt 文件: %s", annotations_dir)
            return []
        import shutil

        yolo_output_dir.mkdir(parents=True, exist_ok=True)
        for txt in txt_files:
            shutil.copy2(txt, yolo_output_dir / txt.name)
        if class_order:
            return list(class_order)
        ids: set[int] = set()
        for txt in yolo_output_dir.glob("*.txt"):
            with open(txt, encoding="utf-8") as f:
                for line in f:
                    first = line.strip().split(" ")[0]
                    if first.isdigit():
                        ids.add(int(first))
        return [f"class_{i}" for i in sorted(ids)] if ids else []

    if annotation_format == "coco":
        json_files = list(annotations_dir.glob("*.json"))
        if not json_files:
            logger.error("COCO 标注目录下无 json 文件: %s", annotations_dir)
            return []
        classes: list[str] = []
        for json_file in json_files:
            classes = coco_to_yolo(json_file, yolo_output_dir, class_order)
        return classes

    if annotation_format == "pascal_voc":
        return voc_to_yolo(annotations_dir, yolo_output_dir, class_order)

    raise ValueError(f"不支持的标注格式: {annotation_format}，可选 {SUPPORTED_FORMATS}")
