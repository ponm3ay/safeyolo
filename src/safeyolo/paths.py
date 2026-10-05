"""项目路径解析。

所有目录均相对于项目根目录（包含 pyproject.toml 或 configs/ 的目录）解析，
避免旧版本中硬编码绝对路径与多处重复定义路径常量的问题。
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "find_project_root",
    "PROJECT_ROOT",
    "CONFIGS_DIR",
    "DATASET_DIR",
    "RAW_DATA_DIR",
    "RAW_IMAGES_DIR",
    "RAW_ANNOTATIONS_DIR",
    "RUNS_DIR",
    "LOGS_DIR",
    "MODELS_DIR",
    "PRETRAINED_DIR",
    "CHECKPOINTS_DIR",
]


def find_project_root(start: Path | None = None) -> Path:
    """从 start（默认当前工作目录）逐级向上查找项目根目录。

    判定依据：目录下存在 pyproject.toml 或 configs/ 子目录。
    找不到时返回 start 本身，保证脚本在任意位置仍可运行。
    """
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() or (candidate / "configs").is_dir():
            return candidate
    return current


PROJECT_ROOT: Path = find_project_root()
CONFIGS_DIR: Path = PROJECT_ROOT / "configs"

# 数据目录
DATASET_DIR: Path = PROJECT_ROOT / "dataset"
RAW_DATA_DIR: Path = PROJECT_ROOT / "raw"
RAW_IMAGES_DIR: Path = RAW_DATA_DIR / "images"
RAW_ANNOTATIONS_DIR: Path = RAW_DATA_DIR / "original_annotations"

# 训练产物目录
RUNS_DIR: Path = PROJECT_ROOT / "runs"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
MODELS_DIR: Path = PROJECT_ROOT / "models"
PRETRAINED_DIR: Path = MODELS_DIR / "pretrained"
CHECKPOINTS_DIR: Path = MODELS_DIR / "checkpoints"
