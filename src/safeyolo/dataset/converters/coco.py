"""COCO JSON 标注转 YOLO txt 格式。

纯函数实现：所有输入输出路径均由调用方显式传入，模块内不依赖任何全局目录常量。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

__all__ = ["coco_to_yolo"]

logger = logging.getLogger(__name__)


def _coco_bbox_to_yolo(bbox: list[float], img_w: int, img_h: int) -> list[float]:
    """COCO bbox [x_min, y_min, w, h]（像素） -> YOLO [x_center, y_center, w, h]（归一化）。"""
    x_min, y_min, w, h = bbox
    return [
        (x_min + w / 2) / img_w,
        (y_min + h / 2) / img_h,
        w / img_w,
        h / img_h,
    ]


def coco_to_yolo(
    json_path: str | Path,
    yolo_output_dir: str | Path,
    class_order: list[str] | None = None,
) -> list[str]:
    """将单个 COCO JSON 标注文件转换为 YOLO txt 标签集合。

    Args:
        json_path: COCO JSON 文件路径。
        yolo_output_dir: YOLO txt 输出目录（不存在时自动创建）。
        class_order: 指定类别顺序；为 None 时按类别名排序自动确定。

    Returns:
        最终类别名列表（与 YOLO txt 中的类别 id 顺序一致）。
        JSON 无法解析时返回空列表。
    """
    json_path = Path(json_path)
    yolo_output_dir = Path(yolo_output_dir)

    try:
        with open(json_path, encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("解析 COCO JSON 失败: %s | %s", json_path, exc)
        return []

    categories = data.get("categories", [])
    id_to_name = {cat["id"]: cat["name"] for cat in categories}
    # 显式传入的类别顺序必须原样保留（决定 txt 中的类别 id），仅自动模式才排序
    class_order = list(dict.fromkeys(class_order)) if class_order else sorted(set(id_to_name.values()))
    class_to_id = {name: idx for idx, name in enumerate(class_order)}

    images = {img["id"]: img for img in data.get("images", [])}
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for anno in data.get("annotations", []):
        annotations_by_image.setdefault(anno["image_id"], []).append(anno)

    yolo_output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for img_id, img in images.items():
        width, height, file_name = img.get("width"), img.get("height"), img.get("file_name")
        if not width or not height or not file_name:
            logger.warning("图片信息缺失，跳过: %s", img)
            continue

        lines: list[str] = []
        for anno in annotations_by_image.get(img_id, []):
            name = id_to_name.get(anno.get("category_id"))
            if name not in class_to_id:
                logger.warning("类别 %s 不在类别列表中，跳过 (%s)", name, file_name)
                continue
            bbox = anno.get("bbox")
            if not bbox or len(bbox) != 4:
                logger.warning("无效 bbox，跳过: %s (%s)", anno, file_name)
                continue
            box = _coco_bbox_to_yolo(bbox, width, height)
            lines.append(f"{class_to_id[name]} " + " ".join(f"{v:.6f}" for v in box))

        txt_path = yolo_output_dir / (Path(file_name).stem + ".txt")
        if not lines:
            # 无有效标注的图片不生成空标签，避免引入未标注的背景样本
            logger.warning("图片无有效标注，跳过: %s", file_name)
            continue
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        written += 1

    logger.info("COCO -> YOLO 完成: %d/%d 张图生成标签，类别 %s", written, len(images), class_order)
    return class_order
