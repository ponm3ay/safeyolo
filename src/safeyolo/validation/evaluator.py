"""模型评估：实现旧版本缺失的 val/test 功能，并保留数据集健康检查。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from safeyolo.config import load_yaml
from safeyolo.dataset.info import IMAGE_EXTENSIONS

__all__ = ["evaluate", "check_dataset"]

logger = logging.getLogger(__name__)


def evaluate(
    weights: str | Path,
    data_yaml: str | Path,
    split: str = "val",
    device: Any = None,
    imgsz: int = 640,
) -> Any:
    """在指定数据划分上评估模型，返回 Ultralytics 验证结果。

    Args:
        weights: 模型权重路径。
        data_yaml: 数据集配置文件。
        split: 评估划分，``val`` 或 ``test``。
        device: 计算设备，None 时使用 Ultralytics 默认。
        imgsz: 评估图片尺寸。

    Returns:
        Ultralytics 的验证结果对象（含 box 指标）。
    """
    from ultralytics import YOLO

    if split not in ("val", "test"):
        raise ValueError(f"split 必须是 val 或 test: {split}")
    logger.info("开始评估 | weights=%s split=%s data=%s", weights, split, data_yaml)
    model = YOLO(str(weights))
    kwargs: dict[str, Any] = {"data": str(data_yaml), "split": split, "imgsz": imgsz}
    if device is not None:
        kwargs["device"] = device
    results = model.val(**kwargs)

    box = getattr(results, "box", None)
    logger.info("=" * 60)
    logger.info("评估结果 (%s)", split)
    logger.info("-" * 60)
    logger.info("Precision     : %.4f", getattr(box, "mp", 0))
    logger.info("Recall        : %.4f", getattr(box, "mr", 0))
    logger.info("mAP50         : %.4f", getattr(box, "map50", 0))
    logger.info("mAP50-95      : %.4f", getattr(box, "map", 0))
    if box is not None and hasattr(box, "maps") and hasattr(results, "names"):
        for i, value in enumerate(box.maps):
            logger.info("  %-20s: %.4f", results.names[i], value)
    logger.info("=" * 60)
    return results


def check_dataset(data_yaml: str | Path) -> tuple[bool, list[dict[str, str]]]:
    """数据集健康检查：配置一致性、图片-标签配对、类别 id 与 bbox 有效性。

    Returns:
        (是否通过, 问题列表)。问题项形如 {"image", "message"}。
    """
    data_yaml = Path(data_yaml)
    config = load_yaml(data_yaml)

    names = config.get("names", [])
    nc = config.get("nc")
    issues: list[dict[str, str]] = []
    if not isinstance(names, list) or not names:
        issues.append({"image": "-", "message": "names 缺失或不是列表"})
        names = []
    elif nc != len(names):
        issues.append({"image": "-", "message": f"nc={nc} 与 names 数量 {len(names)} 不一致"})

    root = Path(config.get("path", "."))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()

    total_images = 0
    for split in ("train", "val", "test"):
        split_dir = config.get(split)
        if not split_dir:
            continue
        img_dir = root / split_dir
        label_dir = img_dir.parent.parent / "labels" / split
        if not img_dir.is_dir():
            issues.append({"image": str(img_dir), "message": f"{split} 图片目录不存在"})
            continue
        images = [p for p in sorted(img_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS]
        total_images += len(images)
        for img_path in images:
            label_path = label_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                issues.append({"image": str(img_path), "message": "缺少标签文件"})
                continue
            try:
                lines = label_path.read_text(encoding="utf-8").splitlines()
            except OSError as exc:
                issues.append({"image": str(img_path), "message": f"标签读取失败: {exc}"})
                continue
            for line_no, line in enumerate(lines, 1):
                parts = line.strip().split()
                if not parts:
                    continue
                if not parts[0].isdigit() or (names and int(parts[0]) >= len(names)):
                    issues.append({"image": str(img_path), "message": f"第 {line_no} 行类别 id 无效: {line!r}"})
                    continue
                if len(parts) != 5:
                    issues.append({"image": str(img_path), "message": f"第 {line_no} 行字段数异常: {len(parts)} != 5"})
                    continue
                _, _, _, w, h = map(float, parts)
                if not (0 < w <= 1 and 0 < h <= 1):
                    issues.append({"image": str(img_path), "message": f"第 {line_no} 行 bbox 尺寸非法: w={w}, h={h}"})

    if issues:
        for issue in issues[:50]:
            logger.warning("[数据集检查] %s: %s", issue["message"], issue["image"])
        if len(issues) > 50:
            logger.warning("[数据集检查] ... 共 %d 个问题（仅显示前 50 个）", len(issues))
    logger.info(
        "数据集检查完成: 共 %d 张图，%d 个问题 -> %s", total_images, len(issues), "不通过" if issues else "通过"
    )
    return not issues, issues
