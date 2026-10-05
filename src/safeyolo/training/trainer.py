"""模型训练：封装 Ultralytics YOLO 统一训练流程。

yolov8 与 yolo11 等模型在 Ultralytics 中共用同一套 API，
通过权重文件名即可切换，不再需要旧版本中按 model_type 分支的写法。
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from safeyolo.config import dump_yaml
from safeyolo.dataset.info import dataset_info
from safeyolo.paths import CHECKPOINTS_DIR, RUNS_DIR

__all__ = ["Trainer", "resolve_weights"]

logger = logging.getLogger(__name__)

# 常用模型别名 -> 默认预训练权重（Ultralytics 缺失时会自动下载）
MODEL_DEFAULT_WEIGHTS = {
    "yolov8": "yolov8n.pt",
    "yolov8n": "yolov8n.pt",
    "yolov8s": "yolov8s.pt",
    "yolov8m": "yolov8m.pt",
    "yolo11": "yolo11n.pt",
    "yolo11n": "yolo11n.pt",
    "yolo11s": "yolo11s.pt",
}


def resolve_weights(model: str | None, weights: str | None) -> str:
    """解析预训练权重路径。

    优先级：显式 weights > models/pretrained/<weights> > models/pretrained/<model 别名默认权重>
    均不存在时返回别名默认权重名，交由 Ultralytics 自动下载。
    """
    from safeyolo.paths import PRETRAINED_DIR

    if weights:
        candidates = [Path(weights), PRETRAINED_DIR / weights]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return weights  # 交由 Ultralytics 自动下载
    if model and model in MODEL_DEFAULT_WEIGHTS:
        default = MODEL_DEFAULT_WEIGHTS[model]
        local = PRETRAINED_DIR / default
        if local.exists():
            return str(local)
        return default
    raise ValueError("必须提供 --weights 或可识别的 --model（如 yolov8 / yolo11）")


class Trainer:
    """YOLO 检测模型训练器。"""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger(__name__)

    def prepare_data_yaml(self, data_yaml: str | Path, class_names: list[str] | None) -> Path:
        """生成训练用 data 配置：补充类别信息并写入绝对路径，避免污染源 data.yaml。"""
        config = dataset_info(data_yaml)
        names = class_names or config["names"]
        if not names:
            raise ValueError(f"data.yaml 中缺少类别信息且未通过 --classes 指定: {data_yaml}")

        active = {
            "path": str(Path(config["root"]).resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(names),
            "names": names,
        }
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        active_path = RUNS_DIR / f"active_data_{timestamp}.yaml"
        dump_yaml(active, active_path)
        self.logger.info("训练用数据配置: %s", active_path)
        return active_path

    def train(self, config: dict[str, Any]) -> Any:
        """执行训练并返回 Ultralytics 训练结果对象。

        Args:
            config: 合并后的训练配置（见 configs/train.yaml 的键）。
        """
        from ultralytics import YOLO

        weights = resolve_weights(config.get("model"), config.get("weights"))
        data_yaml = config.get("data", "dataset/data.yaml")
        active_yaml = self.prepare_data_yaml(data_yaml, config.get("classes"))

        train_params: dict[str, Any] = {
            "data": str(active_yaml),
            "epochs": config.get("epochs", 100),
            "batch": config.get("batch", 16),
            "imgsz": config.get("imgsz", 640),
            "device": config.get("device", 0),
            "workers": config.get("workers", 8),
            "seed": config.get("seed", 42),
            "project": str(RUNS_DIR / "detect"),
            "name": config.get("name") or datetime.now().strftime("train_%Y%m%d_%H%M%S"),
            "exist_ok": True,
        }
        self.logger.info("=" * 60)
        self.logger.info("开始训练 | weights=%s", weights)
        for key, value in train_params.items():
            self.logger.info("  %-10s: %s", key, value)
        self.logger.info("=" * 60)

        model = YOLO(weights)
        results = model.train(**train_params)

        self._log_metrics(results)
        self._archive_checkpoints(results, train_params["name"])
        return results

    # ------------------------------------------------------------------
    def _log_metrics(self, results: Any) -> None:
        box = getattr(results, "box", None)
        self.logger.info("=" * 60)
        self.logger.info("训练结果")
        self.logger.info("-" * 60)
        self.logger.info("保存目录      : %s", getattr(results, "save_dir", "未知"))
        self.logger.info("训练耗时      : %s s", getattr(results, "elapsed", "未知"))
        self.logger.info("Precision     : %.4f", getattr(box, "mp", 0))
        self.logger.info("Recall        : %.4f", getattr(box, "mr", 0))
        self.logger.info("mAP50         : %.4f", getattr(box, "map50", 0))
        self.logger.info("mAP50-95      : %.4f", getattr(box, "map", 0))
        if box is not None and hasattr(box, "maps") and hasattr(results, "names"):
            for i, value in enumerate(box.maps):
                self.logger.info("  %-20s: %.4f", results.names[i], value)
        self.logger.info("=" * 60)

    def _archive_checkpoints(self, results: Any, run_name: str) -> None:
        """将 best/last 权重归档到 models/checkpoints，便于统一管理。"""
        save_dir = Path(getattr(results, "save_dir", ""))
        weights_dir = save_dir / "weights" if save_dir else None
        if not weights_dir or not weights_dir.is_dir():
            self.logger.warning("未找到训练输出权重目录，跳过归档")
            return
        CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        for pt in ("best.pt", "last.pt"):
            source = weights_dir / pt
            if source.exists():
                target = CHECKPOINTS_DIR / f"{run_name}-{timestamp}-{pt}"
                shutil.copy2(source, target)
                self.logger.info("已归档 %s -> %s", pt, target)
