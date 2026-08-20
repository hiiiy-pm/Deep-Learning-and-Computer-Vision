from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from src.data import build_external_labeled_dataset, build_loader, build_transforms
from src.engine import evaluate, matrix_to_list
from src.metrics import class_report, save_paper_evaluation_plots, scalar_metrics
from src.models import build_model
from src.utils import class_names_zh, describe_device, ensure_dir, get_device, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估已训练好的玉米叶部病害分类模型。")
    parser.add_argument("--checkpoint", default="outputs/cnn_mamba_internal_val/best.pt")
    parser.add_argument("--data-dir", default="test_data")
    parser.add_argument("--output-dir", default="outputs/eval")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--img-size", type=int, default=0, help="覆盖 checkpoint 中的评估尺寸；0 表示使用 checkpoint 尺寸")
    parser.add_argument("--eval-crops", type=int, default=1, choices=[1, 5, 10], help="多裁剪评估数量")
    return parser.parse_args()


def dataset_title(data_dir: str | Path) -> str:
    name = Path(data_dir).name.lower()
    if "test" in name:
        return "测试集"
    if "val" in name or "valid" in name:
        return "外部验证集"
    if "train" in name:
        return "训练集"
    return "评估集"


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    output_dir = ensure_dir(args.output_dir)
    print(f"评估设备：{describe_device(device)}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    class_names: list[str] = checkpoint["class_names"]
    img_size = args.img_size if args.img_size > 0 else int(checkpoint.get("img_size", 224))
    model_config = checkpoint["model_config"]

    dataset, source = build_external_labeled_dataset(
        args.data_dir,
        build_transforms(img_size, train=False, eval_crops=args.eval_crops),
        class_names,
    )
    if dataset is None:
        raise FileNotFoundError(source)
    print(f"数据来源：{source}")
    loader = build_loader(dataset, args.batch_size, False, args.num_workers, device)

    model = build_model(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    criterion = nn.CrossEntropyLoss()
    metrics = evaluate(model, loader, criterion, device, len(class_names), amp=args.amp)

    data_name = Path(args.data_dir).name
    title_prefix = dataset_title(args.data_dir)
    save_paper_evaluation_plots(
        metrics,
        class_names,
        output_dir,
        prefix=data_name,
        title_prefix=title_prefix,
    )
    save_json(
        {
            "checkpoint": str(args.checkpoint),
            "data_dir": str(args.data_dir),
            "class_names": class_names_zh(class_names),
            "class_names_en": class_names,
            "img_size": img_size,
            "eval_crops": args.eval_crops,
            "metrics": scalar_metrics(metrics),
            "confusion_matrix": matrix_to_list(metrics["confusion_matrix"]),
            "class_report": class_report(metrics["confusion_matrix"], class_names),
        },
        output_dir / f"{data_name}_metrics.json",
    )
    print(
        f"{data_name} | 损失={metrics['loss']:.4f} 准确率={metrics['accuracy']:.4f} "
        f"宏平均F1={metrics['macro_f1']:.4f}"
    )
    print(f"评估结果已保存到：{output_dir}")


if __name__ == "__main__":
    main()
