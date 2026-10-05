"""YAML 配置加载与合并。

合并优先级（高到低）：命令行覆盖项 > 训练配置文件 > 内置默认值。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from safeyolo.paths import CONFIGS_DIR

__all__ = ["load_yaml", "dump_yaml", "deep_merge", "load_train_config"]

logger = logging.getLogger(__name__)

DEFAULT_TRAIN_CONFIG: dict[str, Any] = {
    "model": "yolov8",
    "weights": "yolov8n.pt",
    "data": "dataset/data.yaml",
    "epochs": 100,
    "batch": 16,
    "imgsz": 640,
    "device": 0,
    "workers": 8,
    "seed": 42,
}


def load_yaml(path: str | Path) -> dict[str, Any]:
    """读取 YAML 文件，空文件返回空字典。"""
    path = Path(path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"配置文件不存在: {path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML 解析失败: {path}\n{exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"配置文件顶层必须是键值映射: {path}")
    logger.info("已加载配置: %s", path)
    return data


def dump_yaml(data: dict[str, Any], path: str | Path) -> Path:
    """将字典写入 YAML 文件（UTF-8、Unicode 原样）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    return path


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并字典，override 中的值优先；返回新字典，不修改入参。"""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_train_config(
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """加载训练配置：内置默认值 <- 训练 YAML <- overrides（命令行覆盖项）。

    Args:
        config_path: 训练配置文件路径，默认 ``configs/train.yaml``，不存在时仅用默认值。
        overrides: 命令行等来源的覆盖项，值为 None 的键会被忽略。

    Returns:
        合并后的配置字典。
    """
    config = dict(DEFAULT_TRAIN_CONFIG)
    if config_path is None:
        config_path = CONFIGS_DIR / "train.yaml"
    config_path = Path(config_path)
    if config_path.exists():
        config = deep_merge(config, load_yaml(config_path))
    elif str(config_path) != str(CONFIGS_DIR / "train.yaml"):
        raise FileNotFoundError(f"训练配置文件不存在: {config_path}")
    if overrides:
        config = deep_merge(config, {k: v for k, v in overrides.items() if v is not None})
    return config
