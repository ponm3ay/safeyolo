"""数据集准备：标注转换 -> 图片标签配对 -> 训练集划分 -> 生成 data.yaml。"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split

from safeyolo.config import dump_yaml
from safeyolo.dataset.converters import SUPPORTED_FORMATS, convert_to_yolo, detect_annotation_format
from safeyolo.dataset.info import IMAGE_EXTENSIONS

__all__ = ["DatasetPreparer"]

logger = logging.getLogger(__name__)


class DatasetPreparer:
    """从原始数据目录构建 YOLO 训练数据集。

    流程：
        1. 将原始标注（yolo / coco / pascal_voc）统一转换为 YOLO txt（暂存区）；
        2. 将暂存区标签与原始图片按文件名配对；
        3. 按 train/val/test 比例划分并复制到输出目录；
        4. 生成 ``dataset/data.yaml``（含类别信息与绝对路径，训练可直接使用）。

    所有输入输出目录均显式传入，便于测试与多项目复用。
    """

    def __init__(
        self,
        raw_root: str | Path,
        output_root: str | Path,
        annotation_format: str = "auto",
        train_rate: float = 0.8,
        valid_rate: float = 0.1,
        classes: list[str] | None = None,
        seed: int = 42,
        data_yaml_name: str = "data.yaml",
        logger: logging.Logger | None = None,
    ) -> None:
        if not 0 < train_rate < 1:
            raise ValueError(f"train_rate 必须在 (0, 1) 区间: {train_rate}")
        if not 0 <= valid_rate < 1 or train_rate + valid_rate >= 1:
            raise ValueError(f"train_rate + valid_rate 必须小于 1: {train_rate} + {valid_rate}")

        self.raw_root = Path(raw_root)
        self.output_root = Path(output_root)
        self.annotation_format = annotation_format
        self.train_rate = train_rate
        self.valid_rate = valid_rate
        self.classes = list(classes) if classes else None
        self.seed = seed
        self.data_yaml_path = self.output_root / data_yaml_name
        self.logger = logger or logging.getLogger(__name__)

        self.raw_images_dir = self.raw_root / "images"
        self.raw_annotations_dir = self.raw_root / "original_annotations"
        self.staged_labels_dir = self.raw_root / "yolo_staged_labels"

    # ------------------------------------------------------------------
    # 对外主流程
    # ------------------------------------------------------------------
    def prepare(self) -> Path:
        """执行完整数据准备流程，返回生成的 data.yaml 路径。"""
        fmt = self._resolve_format()
        self.logger.info("标注格式: %s", fmt)

        self._reset_output_dirs()

        class_names = self._stage_labels(fmt)
        pairs = self._pair_images_and_labels()
        if not pairs:
            raise RuntimeError("未找到任何可配对的图片-标签对，请检查原始数据目录")
        self._split_and_copy(pairs)
        self._write_data_yaml(class_names)
        self.logger.info("数据集准备完成: %s", self.data_yaml_path)
        return self.data_yaml_path

    # ------------------------------------------------------------------
    # 内部步骤
    # ------------------------------------------------------------------
    def _resolve_format(self) -> str:
        if self.annotation_format == "auto":
            fmt = detect_annotation_format(self.raw_root)
            self.logger.info("自动识别标注格式: %s", fmt)
            return fmt
        if self.annotation_format not in SUPPORTED_FORMATS:
            raise ValueError(f"不支持的标注格式: {self.annotation_format}，可选 {SUPPORTED_FORMATS} 或 auto")
        return self.annotation_format

    def _reset_output_dirs(self) -> None:
        for split in ("train", "val", "test"):
            for kind in ("images", "labels"):
                target = self.output_root / kind / split
                if target.exists():
                    shutil.rmtree(target)
                target.mkdir(parents=True, exist_ok=True)
        self.logger.info("输出目录已重置: %s", self.output_root)

    def _stage_labels(self, fmt: str) -> list[str]:
        """将原始标注转换为 YOLO txt 并放入暂存区，返回类别名列表。"""
        source_dir = self.raw_annotations_dir
        if not source_dir.is_dir():
            raise FileNotFoundError(f"原始标注目录不存在: {source_dir}")

        # 每次准备都重建暂存区，避免上一次运行残留标签污染配对
        if self.staged_labels_dir.exists():
            shutil.rmtree(self.staged_labels_dir)
        self.staged_labels_dir.mkdir(parents=True, exist_ok=True)

        class_names = convert_to_yolo(
            source_dir, fmt, self.staged_labels_dir, class_order=self.classes
        )
        if not class_names:
            raise RuntimeError("标注转换失败：未获取到类别信息")
        self.logger.info("标签暂存完成，类别: %s", class_names)
        return class_names

    def _pair_images_and_labels(self) -> list[tuple[Path, Path]]:
        pairs: list[tuple[Path, Path]] = []
        for label_path in sorted(self.staged_labels_dir.glob("*.txt")):
            for ext in IMAGE_EXTENSIONS:
                img_path = self.raw_images_dir / f"{label_path.stem}{ext}"
                if img_path.exists():
                    pairs.append((img_path, label_path))
                    break
        self.logger.info("图片-标签配对: %d 对", len(pairs))
        return pairs

    def _split_and_copy(self, pairs: list[tuple[Path, Path]]) -> None:
        train_pairs, temp_pairs = train_test_split(
            pairs, test_size=1 - self.train_rate, random_state=self.seed
        )
        test_rate = 1 - self.train_rate - self.valid_rate
        if self.valid_rate > 0 and temp_pairs:
            val_share = self.valid_rate / (self.valid_rate + test_rate)
            val_pairs, test_pairs = train_test_split(
                temp_pairs, test_size=1 - val_share, random_state=self.seed
            )
        else:
            val_pairs, test_pairs = [], temp_pairs
        if len(pairs) < 3:  # 极小数据集：全部作为训练集
            train_pairs, val_pairs, test_pairs = pairs, [], []

        for split, split_pairs in (("train", train_pairs), ("val", val_pairs), ("test", test_pairs)):
            img_dir = self.output_root / "images" / split
            label_dir = self.output_root / "labels" / split
            for img_path, label_path in split_pairs:
                shutil.copy2(img_path, img_dir / img_path.name)
                shutil.copy2(label_path, label_dir / label_path.name)
            self.logger.info("%s: %d 对", split, len(split_pairs))

    def _write_data_yaml(self, class_names: list[str]) -> None:
        config = {
            # 绝对路径保证任意工作目录下均可训练；脚本每次准备时重新生成
            "path": str(self.output_root.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(class_names),
            "names": class_names,
        }
        dump_yaml(config, self.data_yaml_path)
        self.logger.info("已生成 %s", self.data_yaml_path)
