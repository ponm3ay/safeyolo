"""safeyolo —— 基于 Ultralytics YOLO 的安全帽检测训练工具包。

模块结构：
    paths       项目路径解析
    config      YAML 配置加载与合并
    logging_setup  统一日志（控制台彩色 + 文件）
    device      训练设备信息采集
    dataset     数据集准备 / 校验 / 标注格式转换
    training    模型训练
    validation  模型评估
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__"]
