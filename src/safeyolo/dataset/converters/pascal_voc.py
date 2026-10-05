"""Pascal VOC XML 标注转 YOLO txt 格式。

纯函数实现：所有输入输出路径均由调用方显式传入，模块内不依赖任何全局目录常量。
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

__all__ = ["voc_to_yolo"]

logger = logging.getLogger(__name__)


def parse_voc_xml(xml_path: Path) -> dict[str, Any] | None:
    """解析单个 VOC XML，返回 ``{"width", "height", "objects": [{"name", "bbox"}]}``。

    无效文件（缺 size、尺寸非法、XML 损坏）返回 None。
    """
    try:
        root = ET.parse(xml_path).getroot()
    except (ET.ParseError, OSError) as exc:
        logger.error("解析 XML 失败: %s | %s", xml_path, exc)
        return None

    size = root.find("size")
    if size is None:
        logger.warning("XML 缺少 <size> 标签: %s", xml_path)
        return None
    width = int(size.findtext("width", default="0"))
    height = int(size.findtext("height", default="0"))
    if width <= 0 or height <= 0:
        logger.warning("XML 图片尺寸无效: %s", xml_path)
        return None

    objects: list[dict[str, Any]] = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        bndbox = obj.find("bndbox")
        if name is None or bndbox is None:
            continue
        try:
            xmin = float(bndbox.findtext("xmin", default="0"))
            ymin = float(bndbox.findtext("ymin", default="0"))
            xmax = float(bndbox.findtext("xmax", default="0"))
            ymax = float(bndbox.findtext("ymax", default="0"))
        except (TypeError, ValueError):
            logger.warning("边界框坐标解析失败: %s", xml_path)
            continue
        if xmin >= xmax or ymin >= ymax:
            logger.warning("无效边界框，跳过: %s [%s, %s, %s, %s]", xml_path, xmin, ymin, xmax, ymax)
            continue
        objects.append({"name": name, "bbox": [xmin, ymin, xmax, ymax]})
    return {"width": width, "height": height, "objects": objects}


def _voc_bbox_to_yolo(bbox: list[float], img_w: int, img_h: int) -> list[float]:
    """VOC bbox [xmin, ymin, xmax, ymax]（像素） -> YOLO [x_center, y_center, w, h]（归一化）。"""
    xmin, ymin, xmax, ymax = bbox
    return [
        (xmin + xmax) / 2 / img_w,
        (ymin + ymax) / 2 / img_h,
        (xmax - xmin) / img_w,
        (ymax - ymin) / img_h,
    ]


def voc_to_yolo(
    annotations_dir: str | Path,
    yolo_output_dir: str | Path,
    class_order: list[str] | None = None,
) -> list[str]:
    """将目录下所有 Pascal VOC XML 标注转换为 YOLO txt 标签集合。

    Args:
        annotations_dir: 存放 VOC XML 的目录。
        yolo_output_dir: YOLO txt 输出目录（不存在时自动创建）。
        class_order: 指定类别顺序；为 None 时按发现的类别名排序自动确定。

    Returns:
        最终类别名列表（与 YOLO txt 中的类别 id 顺序一致）。
        目录不存在或无有效标注时返回空列表。
    """
    annotations_dir = Path(annotations_dir)
    yolo_output_dir = Path(yolo_output_dir)
    if not annotations_dir.is_dir():
        logger.error("标注目录不存在: %s", annotations_dir)
        return []

    parsed: list[tuple[Path, dict[str, Any]]] = []
    discovered_names: set[str] = set()
    for xml_path in sorted(annotations_dir.glob("*.xml")):
        result = parse_voc_xml(xml_path)
        if result is None:
            continue
        parsed.append((xml_path, result))
        discovered_names.update(obj["name"] for obj in result["objects"])

    if not parsed:
        logger.error("目录下无有效 VOC XML 标注: %s", annotations_dir)
        return []

    # 显式传入的类别顺序必须原样保留（决定 txt 中的类别 id），仅自动模式才排序
    class_order = list(dict.fromkeys(class_order)) if class_order else sorted(discovered_names)
    class_to_id = {name: idx for idx, name in enumerate(class_order)}

    yolo_output_dir.mkdir(parents=True, exist_ok=True)
    for xml_path, result in parsed:
        lines: list[str] = []
        for obj in result["objects"]:
            if obj["name"] not in class_to_id:
                logger.warning("类别 %s 不在类别列表中，跳过 (%s)", obj["name"], xml_path.name)
                continue
            box = _voc_bbox_to_yolo(obj["bbox"], result["width"], result["height"])
            lines.append(f"{class_to_id[obj['name']]} " + " ".join(f"{v:.6f}" for v in box))
        if not lines:
            # 无有效标注的图片不生成空标签，避免引入未标注的背景样本
            logger.warning("标注无有效目标，跳过: %s", xml_path.name)
            continue
        txt_path = yolo_output_dir / (xml_path.stem + ".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    logger.info("VOC -> YOLO 完成: %d 个标注，类别 %s", len(parsed), class_order)
    return class_order
