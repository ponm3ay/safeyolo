"""数据集信息统计。"""

from __future__ import annotations

import logging
from pathlib import Path

from safeyolo.config import load_yaml

__all__ = ["dataset_info", "IMAGE_EXTENSIONS"]

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _count_images(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def dataset_info(data_yaml: str | Path) -> dict:
    """统计 data.yaml 描述的数据集信息。

    data.yaml 中的 ``path`` 为数据集根目录（绝对路径或相对该 yaml 所在目录），
    ``train`` / ``val`` / ``test`` 为相对根目录的图片目录。

    Returns:
        包含 splits 图片数、类别数与类别名的字典。
    """
    data_yaml = Path(data_yaml)
    config = load_yaml(data_yaml)

    root = Path(config.get("path", "."))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()

    names = config.get("names", [])
    splits = {}
    for split in ("train", "val", "test"):
        split_dir = Path(config.get(split, "")) if config.get(split) else None
        splits[split] = _count_images(root / split_dir) if split_dir else 0

    info = {
        "yaml": str(data_yaml),
        "root": str(root),
        "nc": config.get("nc", len(names)),
        "names": names,
        "splits": splits,
        "total": sum(splits.values()),
    }
    logger.info(
        "数据集: train=%d val=%d test=%d 共 %d 张，%d 类 %s",
        splits["train"], splits["val"], splits["test"], info["total"], info["nc"], names,
    )
    return info
