"""训练设备信息采集（CPU / 内存 / GPU）。

torch 与 psutil 均为可选依赖，缺失时优雅降级。
"""

from __future__ import annotations

import logging
import platform
from typing import Any

__all__ = ["get_device_info", "log_device_info"]

logger = logging.getLogger(__name__)


def get_device_info() -> dict[str, Any]:
    """采集当前设备信息，返回结构化字典。任何单项失败不影响整体。"""
    info: dict[str, Any] = {
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
    }

    try:
        import psutil

        memory = psutil.virtual_memory()
        info["cpu"] = {
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "usage_percent": psutil.cpu_percent(),
            "memory_total_gb": round(memory.total / 1e9, 2),
            "memory_available_gb": round(memory.available / 1e9, 2),
        }
    except ImportError:
        info["cpu"] = "psutil 未安装，跳过"

    try:
        import torch

        cuda_available = torch.cuda.is_available()
        info["cuda"] = {
            "available": cuda_available,
            "torch_version": torch.__version__,
        }
        if cuda_available:
            info["cuda"]["version"] = torch.version.cuda
            info["cuda"]["device_count"] = torch.cuda.device_count()
            info["cuda"]["devices"] = [
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "total_memory_gb": round(torch.cuda.get_device_properties(i).total_memory / 1e9, 2),
                }
                for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        info["cuda"] = "torch 未安装，跳过"

    return info


def log_device_info(log: logging.Logger) -> dict[str, Any]:
    """以分块日志形式打印设备信息概览。"""
    info = get_device_info()
    log.info("=" * 40)
    log.info("设备信息概览")
    log.info("=" * 40)
    log.info("操作系统       : %s", info["os"])
    log.info("Python 版本    : %s", info["python"])
    if isinstance(info.get("cpu"), dict):
        cpu = info["cpu"]
        log.info(
            "CPU            : %s 物理核 / %s 逻辑核",
            cpu.get("physical_cores"),
            cpu.get("logical_cores"),
        )
        log.info(
            "内存           : 共 %.2f GB，可用 %.2f GB",
            cpu.get("memory_total_gb", 0),
            cpu.get("memory_available_gb", 0),
        )
    if isinstance(info.get("cuda"), dict):
        cuda = info["cuda"]
        log.info("CUDA 可用      : %s (torch %s)", cuda.get("available"), cuda.get("torch_version"))
        for gpu in cuda.get("devices", []):
            log.info(
                "  GPU %s        : %s (%.2f GB)",
                gpu["index"],
                gpu["name"],
                gpu["total_memory_gb"],
            )
    else:
        log.info("CUDA           : %s", info.get("cuda"))
    log.info("=" * 40)
    return info
