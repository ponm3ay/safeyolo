"""统一日志配置：彩色控制台输出 + 滚动文件输出。

旧版本在 yolo_validate.py 中复制了一份完全相同的实现，此处收敛为唯一来源。
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import colorlog

__all__ = ["setup_logging", "get_logger"]

_LOG_COLORS = {
    "DEBUG": "cyan",
    "INFO": "blue",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "bold_red",
}

_CONSOLE_FORMAT = "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_FILE_FORMAT = "%(asctime)s - %(name)s - %(levelname)s : %(message)s"


def get_logger(
    base_path: Path,
    log_type: str = "general",
    *,
    logger_name: str = "safeyolo",
    model_name: str | None = None,
    log_level: int = logging.INFO,
    temp_log: bool = False,
    encoding: str = "utf-8",
) -> logging.Logger:
    """创建（或复用并重置）一个同时输出到控制台与文件的 logger。

    Args:
        base_path: 日志根目录，实际文件写入 ``base_path / log_type /``。
        log_type: 日志类别，用作子目录名（如 train / validate）。
        logger_name: logger 实例名称。
        model_name: 可选，附加到日志文件名中。
        log_level: 最低记录级别。
        temp_log: 为 True 时文件名使用 ``temp_`` 前缀（由调用方随后重命名）。
        encoding: 日志文件编码。

    Returns:
        配置完成的 logging.Logger。
    """
    log_dir = Path(base_path) / log_type
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    prefix = "temp" if temp_log else log_type
    name_parts = [prefix, timestamp]
    if model_name:
        name_parts.append(model_name.replace(" ", "_"))
    log_file = log_dir / ("_".join(name_parts) + ".log")

    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_file, encoding=encoding)
    file_handler.setFormatter(logging.Formatter(_FILE_FORMAT))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        colorlog.ColoredFormatter(_CONSOLE_FORMAT, reset=True, log_colors=_LOG_COLORS)
    )
    logger.addHandler(console_handler)

    logger.info("日志已启用，文件: %s", log_file)
    return logger


def setup_logging(base_path: Path, log_type: str = "general", **kwargs) -> logging.Logger:
    """兼容旧调用的别名。"""
    return get_logger(base_path, log_type, **kwargs)
