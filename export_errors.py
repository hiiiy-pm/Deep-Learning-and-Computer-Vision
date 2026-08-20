from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data import build_external_labeled_dataset, build_transforms
from src.engine import _forward_eval
from src.models import build_model
from src.utils import class_name_zh, ensure_dir, get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出模型在数据集上的错误样本，便于论文误差分析。")
    parser.add_argument("--checkpoint", default="outputs/cnn_mamba_internal_val/best.pt")
    parser.add_argument("--data-dir", default="val_data")
    parser.add_argument("--output", default="outputs/error_analysis/external_val_errors.csv")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--eval-crops", type=int, default=5, choices=[1, 5, 10])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--summary-output", default="", help="错误类型汇总 CSV；留空时自动使用 *_summary.csv")
    parser.add_argument("--top-k-per-pair", type=int, default=0, help="每种真实-预测错误类型最多保留多少个高置信样本；0 表示全部保留")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    class_names: list[str] = checkpoint["class_names"]

    dataset, source = build_external_labeled_dataset(
        args.data_dir,
        build_transforms(args.img_size, train=False, eval_crops=args.eval_crops),
        class_names,
    )
    if dataset is None:
        raise FileNotFoundError(source)

    model = build_model(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    rows: list[dict[str, str | float]] = []
    pair_counts: Counter[tuple[str, str]] = Counter()
    total_error_count = 0
    sample_offset = 0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        logits = _forward_eval(model, images)
        probs = torch.softmax(logits, dim=1)
        confs, preds = probs.max(dim=1)

        for i, (target, pred, conf) in enumerate(zip(targets.cpu(), preds.cpu(), confs.cpu())):
            if int(target) == int(pred):
                continue
            image_path = dataset.samples[sample_offset + i][0] if hasattr(dataset, "samples") else ""
            true_name = class_name_zh(class_names[int(target)])
            pred_name = class_name_zh(class_names[int(pred)])
            pair_counts[(true_name, pred_name)] += 1
            total_error_count += 1
            rows.append(
                {
                    "图片路径": str(image_path),
                    "真实类别": true_name,
                    "预测类别": pred_name,
                    "预测置信度": float(conf),
                }
            )
        sample_offset += int(targets.numel())

    if args.top_k_per_pair > 0:
        grouped_rows: dict[tuple[str, str], list[dict[str, str | float]]] = {}
        for row in rows:
            key = (str(row["真实类别"]), str(row["预测类别"]))
            grouped_rows.setdefault(key, []).append(row)
        rows = [
            row
            for key in sorted(grouped_rows)
            for row in sorted(grouped_rows[key], key=lambda item: float(item["预测置信度"]), reverse=True)[
                : args.top_k_per_pair
            ]
        ]

    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = ["图片路径", "真实类别", "预测类别", "预测置信度"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_output = Path(args.summary_output) if args.summary_output else output_path.with_name(f"{output_path.stem}_summary.csv")
    with summary_output.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = ["真实类别", "预测类别", "错误数量"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (true_name, pred_name), count in pair_counts.most_common():
            writer.writerow({"真实类别": true_name, "预测类别": pred_name, "错误数量": count})
    print(f"总错误样本数：{total_error_count}")
    if args.top_k_per_pair > 0:
        print(f"导出错误样本数：{len(rows)}（每种错误类型最多 {args.top_k_per_pair} 个高置信样本）")
    else:
        print(f"导出错误样本数：{len(rows)}")
    print(f"错误样本已保存到：{output_path}")
    print(f"错误类型汇总已保存到：{summary_output}")


if __name__ == "__main__":
    main()
