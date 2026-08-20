from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.inference import CornDiseasePredictor, iter_image_paths
from src.settings import DEFAULT_DEVICE, DEFAULT_TOPK
from src.utils import describe_device, ensure_dir, get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="预测单张图片或文件夹中的玉米叶部病害类别。")
    parser.add_argument("--checkpoint", default="", help="模型权重路径；留空时自动查找默认 best.pt")
    parser.add_argument("--input", required=True, help="image path or folder path")
    parser.add_argument("--output", default="outputs/predictions.csv")
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--eval-crops", type=int, default=1, choices=[1, 5, 10], help="预测时使用的多裁剪数量")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    print(f"预测设备：{describe_device(device)}")
    predictor = CornDiseasePredictor(
        checkpoint=args.checkpoint or None,
        device=device,
        eval_crops=args.eval_crops,
    )
    print(f"模型权重：{predictor.checkpoint_path}")

    paths = iter_image_paths(args.input)
    results = predictor.predict_many(paths, topk=args.topk)
    rows = [result.to_csv_row() for result in results]

    for result in results:
        print(f"{result.image_path} -> {result.predicted_class_zh}（置信度 {result.confidence:.4f}）")

    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"预测结果已保存到：{output_path}")


if __name__ == "__main__":
    main()
