"""safeyolo 命令行入口。

用法（或安装后直接使用 ``safeyolo`` 命令）::

    python -m safeyolo prepare --format auto --raw raw
    python -m safeyolo train --prepare --model yolov8 --epochs 100
    python -m safeyolo val --weights runs/detect/<run>/weights/best.pt
    python -m safeyolo test --weights models/checkpoints/<...>-best.pt
    python -m safeyolo predict --weights best.pt --source image.jpg
    python -m safeyolo check
    python -m safeyolo info
    python -m safeyolo devices
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from safeyolo import __version__
from safeyolo.config import load_train_config
from safeyolo.dataset.info import dataset_info
from safeyolo.logging_setup import get_logger
from safeyolo.paths import (
    CHECKPOINTS_DIR,
    DATASET_DIR,
    LOGS_DIR,
    PRETRAINED_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    RUNS_DIR,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="safeyolo",
        description="基于 Ultralytics YOLO 的安全帽检测训练工具",
    )
    parser.add_argument("--version", action="version", version=f"safeyolo {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ------------------------------------------------------------------
    p_prepare = subparsers.add_parser("prepare", help="准备数据集（转换标注 / 划分 / 生成 data.yaml）")
    p_prepare.add_argument("--format", default="auto", choices=["auto", "yolo", "coco", "pascal_voc"])
    p_prepare.add_argument("--raw-dir", default=str(RAW_DATA_DIR), help="原始数据根目录（默认 raw/）")
    p_prepare.add_argument("--train-rate", type=float, default=0.8)
    p_prepare.add_argument("--valid-rate", type=float, default=0.1)
    p_prepare.add_argument("--classes", nargs="*", help="类别名称列表（YOLO 原生格式必须指定）")
    p_prepare.add_argument("--seed", type=int, default=42)

    # ------------------------------------------------------------------
    p_train = subparsers.add_parser("train", help="训练模型")
    p_train.add_argument("--config", default=None, help="训练配置文件（默认 configs/train.yaml）")
    p_train.add_argument("--model", default=None, help="模型别名，如 yolov8 / yolo11")
    p_train.add_argument("--weights", default=None, help="预训练权重路径或名称")
    p_train.add_argument("--data", default=None, help="数据集配置文件（默认 dataset/data.yaml）")
    p_train.add_argument("--epochs", type=int, default=None)
    p_train.add_argument("--batch", type=int, default=None)
    p_train.add_argument("--imgsz", type=int, default=None)
    p_train.add_argument("--device", default=None, help="计算设备，如 0 / 0,1 / cpu")
    p_train.add_argument("--workers", type=int, default=None)
    p_train.add_argument("--classes", nargs="*", help="覆盖类别名称列表")
    p_train.add_argument("--prepare", action="store_true", help="训练前先执行数据准备")
    p_train.add_argument("--prepare-format", default="auto", choices=["auto", "yolo", "coco", "pascal_voc"])

    # ------------------------------------------------------------------
    for name, help_text in (("val", "在验证集上评估"), ("test", "在测试集上评估")):
        p_eval = subparsers.add_parser(name, help=help_text)
        p_eval.add_argument("--weights", required=True, help="模型权重路径")
        p_eval.add_argument("--data", default=str(DATASET_DIR / "data.yaml"), help="数据集配置文件")
        p_eval.add_argument("--device", default=None)
        p_eval.add_argument("--imgsz", type=int, default=640)

    # ------------------------------------------------------------------
    p_predict = subparsers.add_parser("predict", help="使用训练好的模型推理")
    p_predict.add_argument("--weights", required=True, help="模型权重路径")
    p_predict.add_argument("--source", required=True, help="图片 / 目录 / 视频 / 摄像头编号")
    p_predict.add_argument("--save-dir", default=None, help="结果保存目录（默认 runs/predict）")
    p_predict.add_argument("--conf", type=float, default=0.25, help="置信度阈值")

    # ------------------------------------------------------------------
    p_check = subparsers.add_parser("check", help="数据集健康检查（配对 / 类别 id / bbox）")
    p_check.add_argument("--data", default=str(DATASET_DIR / "data.yaml"))

    p_info = subparsers.add_parser("info", help="查看数据集统计信息")
    p_info.add_argument("--data", default=str(DATASET_DIR / "data.yaml"))

    subparsers.add_parser("devices", help="查看 CPU / 内存 / GPU 设备信息")
    return parser


def _resolve_device(value: str | None) -> object:
    """将命令行设备参数转为 Ultralytics 可接受的类型。"""
    if value is None:
        return None
    if value.lower() in ("cpu", "mps"):
        return value.lower()
    return value  # 形如 "0" / "0,1" 由 Ultralytics 解析


# ----------------------------------------------------------------------
# 子命令实现
# ----------------------------------------------------------------------
def cmd_prepare(args: argparse.Namespace) -> int:
    from safeyolo.dataset.prepare import DatasetPreparer

    logger = get_logger(LOGS_DIR, "prepare", logger_name="safeyolo.prepare")
    preparer = DatasetPreparer(
        raw_root=args.raw_dir,
        output_root=DATASET_DIR,
        annotation_format=args.format,
        train_rate=args.train_rate,
        valid_rate=args.valid_rate,
        classes=args.classes,
        seed=args.seed,
        logger=logger,
    )
    preparer.prepare()
    dataset_info(preparer.data_yaml_path)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from safeyolo.training.trainer import Trainer

    logger = get_logger(LOGS_DIR, "train", logger_name="safeyolo.train", model_name=args.model, temp_log=True)
    from safeyolo.device import log_device_info

    log_device_info(logger)

    overrides = {
        "model": args.model,
        "weights": args.weights,
        "data": args.data,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": _resolve_device(args.device),
        "workers": args.workers,
        "classes": args.classes,
    }
    config = load_train_config(args.config, overrides)
    logger.info("训练配置: %s", {k: v for k, v in config.items() if k != "classes"})

    if args.prepare:
        from safeyolo.dataset.prepare import DatasetPreparer

        logger.info("先执行数据准备（--prepare）...")
        preparer = DatasetPreparer(
            raw_root=RAW_DATA_DIR,
            output_root=DATASET_DIR,
            annotation_format=args.prepare_format,
            classes=config.get("classes"),
            logger=logger,
        )
        preparer.prepare()
        config["data"] = str(preparer.data_yaml_path)

    Trainer(logger).train(config)
    logger.info("训练流程完成。")
    return 0


def _cmd_eval(args: argparse.Namespace, split: str) -> int:
    from safeyolo.validation.evaluator import evaluate

    logger = get_logger(LOGS_DIR, "validate", logger_name="safeyolo.validate")
    evaluate(
        weights=args.weights,
        data_yaml=args.data,
        split=split,
        device=_resolve_device(args.device),
        imgsz=args.imgsz,
    )
    return 0


def cmd_val(args: argparse.Namespace) -> int:
    return _cmd_eval(args, "val")


def cmd_test(args: argparse.Namespace) -> int:
    return _cmd_eval(args, "test")


def cmd_predict(args: argparse.Namespace) -> int:
    from ultralytics import YOLO

    logger = get_logger(LOGS_DIR, "predict", logger_name="safeyolo.predict")
    save_dir = args.save_dir or str(RUNS_DIR / "predict")
    logger.info("推理 | weights=%s source=%s", args.weights, args.source)
    model = YOLO(args.weights)
    model.predict(source=args.source, save=True, project=save_dir, name="exp", exist_ok=True, conf=args.conf)
    logger.info("结果已保存到: %s", save_dir)
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    from safeyolo.validation.evaluator import check_dataset

    get_logger(LOGS_DIR, "check", logger_name="safeyolo.check")
    ok, issues = check_dataset(args.data)
    print(f"数据集检查: {'通过' if ok else f'不通过（{len(issues)} 个问题）'}")
    return 0 if ok else 1


def cmd_info(args: argparse.Namespace) -> int:
    info = dataset_info(args.data)
    print(f"数据集根目录 : {info['root']}")
    print(f"类别数       : {info['nc']}")
    print(f"类别         : {', '.join(info['names'])}")
    for split, count in info["splits"].items():
        print(f"{split:<13}: {count} 张")
    print(f"总计         : {info['total']} 张")
    return 0


def cmd_devices(args: argparse.Namespace) -> int:
    from safeyolo.device import log_device_info

    log_device_info(get_logger(LOGS_DIR, "device", logger_name="safeyolo.device"))
    return 0


_COMMANDS = {
    "prepare": cmd_prepare,
    "train": cmd_train,
    "val": cmd_val,
    "test": cmd_test,
    "predict": cmd_predict,
    "check": cmd_check,
    "info": cmd_info,
    "devices": cmd_devices,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _COMMANDS[args.command](args)
    except KeyboardInterrupt:
        print("\n已中断")
        return 130
    except Exception as exc:  # noqa: BLE001 —— 顶层统一兜底，给出清晰错误
        print(f"执行失败: {exc}", file=sys.stderr)
        print(f"项目根目录: {PROJECT_ROOT}", file=sys.stderr)
        print(f"提示: 日志见 {LOGS_DIR}；权重存放见 {PRETRAINED_DIR} / {CHECKPOINTS_DIR}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
