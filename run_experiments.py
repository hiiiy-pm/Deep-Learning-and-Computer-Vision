from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

from src.metrics import save_model_comparison_plot
from src.models import MODEL_CHOICES


METRIC_COLUMNS = [
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "expected_calibration_error",
    "brier_score",
    "negative_log_likelihood",
    "loss",
]
MODEL_DISPLAY_NAMES = {
    "cnn": "轻量 CNN",
    "cnn_mamba": "CNN-Mamba S6 门控融合",
    "cnn_mamba_unidirectional": "CNN-Mamba S6 单向扫描",
    "cnn_mamba_no_fusion": "CNN-Mamba S6 无局部融合",
    "cnn_mamba_simple_fusion": "CNN-Mamba S6 简单融合",
    "mamba_only": "纯 Mamba（无 CNN stem）",
    "mamba_only_unidirectional": "纯 Mamba 单向扫描",
    "resnet18": "ResNet18",
    "resnet50": "ResNet50",
    "mobilenet_v3_small": "MobileNetV3-Small",
    "mobilenet_v3_large": "MobileNetV3-Large",
    "efficientnet_b0": "EfficientNet-B0",
}
METRIC_DISPLAY_NAMES = {
    "accuracy": "准确率",
    "macro_f1": "宏平均F1",
    "weighted_f1": "加权F1",
    "balanced_accuracy": "平衡准确率",
    "macro_precision": "宏平均精确率",
    "macro_recall": "宏平均召回率",
    "expected_calibration_error": "期望校准误差",
    "brier_score": "Brier分数",
    "negative_log_likelihood": "负对数似然",
    "loss": "损失",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行论文对比实验并汇总结果。")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--models", nargs="+", default=["cnn", "cnn_mamba"], choices=MODEL_CHOICES)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42], help="重复实验随机种子；建议论文实验使用 3 个以上")
    parser.add_argument(
        "--augment",
        default="",
        choices=["", "standard", "strong", "randaugment", "domain"],
        help="覆盖 train.py 的增强策略",
    )
    parser.add_argument("--mixup-alpha", type=float, default=-1.0, help="覆盖 train.py 的 MixUp alpha；负数表示不覆盖")
    parser.add_argument("--cutmix-alpha", type=float, default=-1.0, help="覆盖 train.py 的 CutMix alpha；负数表示不覆盖")
    parser.add_argument("--pretrained", action="store_true", help="torchvision 基线使用 ImageNet 预训练")
    parser.add_argument("--freeze-backbone", action="store_true", help="冻结 torchvision 预训练骨干")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="从训练集分层划分独立测试集的比例")
    parser.add_argument("--exclude-duplicates", default="", help="重复文件路径列表 JSON，传给 train.py")
    parser.add_argument("--external-val-dir", default="val_data", help="外部验证集目录")
    parser.add_argument("--external-eval-crops", type=int, default=5, choices=[1, 5, 10])
    parser.add_argument("--external-img-size", type=int, default=384)
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--output-root", default="outputs/paper_experiments")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="传给 train.py 的额外参数")
    return parser.parse_args()


def load_metrics(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {key: float(value) for key, value in data.get("metrics", {}).items()}


def write_csv(rows: list[dict[str, str | int | float]], path: Path) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, str | int | float]]) -> list[dict[str, str | int | float]]:
    groups: dict[tuple[str, str], list[dict[str, str | int | float]]] = {}
    for row in rows:
        model = str(row["model"])
        split = str(row["split"])
        groups.setdefault((model, split), []).append(row)

    summary_rows: list[dict[str, str | int | float]] = []
    for (model, split), group_rows in sorted(groups.items()):
        summary: dict[str, str | int | float] = {
            "model": model,
            "model_zh": MODEL_DISPLAY_NAMES.get(model, model),
            "split": split,
            "runs": len(group_rows),
        }
        for metric in METRIC_COLUMNS:
            values = [float(row[metric]) for row in group_rows if metric in row]
            if not values:
                continue
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
            std = variance**0.5 if len(values) > 1 else 0.0
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std
            summary[f"{metric}_mean_std"] = f"{mean:.4f}±{std:.4f}"
        summary_rows.append(summary)
    return summary_rows


def compute_statistical_tests(
    rows: list[dict[str, str | int | float]],
    split: str,
    metric: str = "macro_f1",
) -> list[dict[str, str]]:
    """Pairwise paired t-test between models for a given split and metric."""
    model_values: dict[str, list[float]] = {}
    for row in rows:
        if row.get("split") != split or metric not in row:
            continue
        model = str(row["model"])
        model_values.setdefault(model, []).append(float(row[metric]))

    results: list[dict[str, str]] = []
    model_names = sorted(model_values)
    for i, model_a in enumerate(model_names):
        for model_b in model_names[i + 1:]:
            vals_a = model_values[model_a]
            vals_b = model_values[model_b]
            if len(vals_a) < 2 or len(vals_b) < 2:
                continue
            if len(vals_a) != len(vals_b):
                continue
            t_stat, p_val = scipy_stats.ttest_rel(vals_a, vals_b)
            significant = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            results.append(
                {
                    "数据集": split,
                    "指标": METRIC_DISPLAY_NAMES.get(metric, metric),
                    "模型A": MODEL_DISPLAY_NAMES.get(model_a, model_a),
                    "模型B": MODEL_DISPLAY_NAMES.get(model_b, model_b),
                    "A均值": f"{np.mean(vals_a):.4f}",
                    "B均值": f"{np.mean(vals_b):.4f}",
                    "t统计量": f"{t_stat:.4f}",
                    "p值": f"{p_val:.4f}",
                    "显著性": significant,
                }
            )
    return results


def make_paper_table(summary_rows: list[dict[str, str | int | float]]) -> list[dict[str, str | int | float]]:
    paper_rows: list[dict[str, str | int | float]] = []
    preferred_metrics = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "balanced_accuracy",
        "expected_calibration_error",
        "brier_score",
    ]
    for row in summary_rows:
        paper_row: dict[str, str | int | float] = {
            "模型": row["model_zh"],
            "数据集": row["split"],
            "重复次数": row["runs"],
        }
        for metric in preferred_metrics:
            key = f"{metric}_mean_std"
            if key in row:
                paper_row[METRIC_DISPLAY_NAMES[metric]] = row[key]
        paper_rows.append(paper_row)
    return paper_rows


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str | int | float]] = []
    for model in args.models:
        for seed in args.seeds:
            out_dir = output_root / f"{model}_seed{seed}"
            cmd = [
                sys.executable,
                "train.py",
                "--model",
                model,
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--seed",
                str(seed),
                "--output-dir",
                str(out_dir),
            ]
            if args.augment:
                cmd.extend(["--augment", args.augment])
            if args.mixup_alpha >= 0:
                cmd.extend(["--mixup-alpha", str(args.mixup_alpha)])
            if args.cutmix_alpha >= 0:
                cmd.extend(["--cutmix-alpha", str(args.cutmix_alpha)])
            if args.pretrained:
                cmd.append("--pretrained")
            if args.freeze_backbone:
                cmd.append("--freeze-backbone")
            cmd.extend(["--test-ratio", str(args.test_ratio)])
            cmd.extend(["--external-val-dir", args.external_val_dir])
            cmd.extend(["--external-eval-crops", str(args.external_eval_crops)])
            cmd.extend(["--external-img-size", str(args.external_img_size)])
            cmd.extend(["--img-size", str(args.img_size)])
            if args.exclude_duplicates:
                cmd.extend(["--exclude-duplicates", args.exclude_duplicates])
            cmd.extend(args.extra)
            print("运行实验：" + " ".join(cmd))
            subprocess.run(cmd, check=True)

            split_files = {
                "内部验证集": out_dir / "best_internal_val_metrics.json",
                "外部验证集": out_dir / "external_val_final_metrics.json",
                "测试集": out_dir / "test_metrics.json",
            }
            for split, metric_path in split_files.items():
                metrics = load_metrics(metric_path)
                if not metrics:
                    continue
                row: dict[str, str | int | float] = {
                    "model": model,
                    "model_zh": MODEL_DISPLAY_NAMES.get(model, model),
                    "seed": seed,
                    "split": split,
                    "output_dir": str(out_dir),
                }
                for key in METRIC_COLUMNS:
                    if key in metrics:
                        row[key] = metrics[key]
                rows.append(row)

    write_csv(rows, output_root / "summary_runs.csv")
    summary_rows = summarize(rows)
    write_csv(summary_rows, output_root / "summary_mean_std.csv")
    write_csv(make_paper_table(summary_rows), output_root / "paper_table.csv")

    for split in ["内部验证集", "外部验证集", "测试集"]:
        split_rows = [row for row in rows if row.get("split") == split]
        save_model_comparison_plot(
            split_rows,
            metric_key="macro_f1",
            metric_name="宏平均 F1",
            path=output_root / f"{split}_macro_f1_comparison.png",
            title=f"{split}模型宏平均 F1 对比",
        )
        save_model_comparison_plot(
            split_rows,
            metric_key="accuracy",
            metric_name="准确率",
            path=output_root / f"{split}_accuracy_comparison.png",
            title=f"{split}模型准确率对比",
        )

    # Statistical significance tests
    stat_rows: list[dict[str, str]] = []
    for split in ["内部验证集", "外部验证集", "测试集"]:
        for metric in ["macro_f1", "accuracy"]:
            stat_rows.extend(compute_statistical_tests(rows, split, metric))
    if stat_rows:
        write_csv(stat_rows, output_root / "statistical_tests.csv")
        print(f"统计显著性检验已保存到：{output_root / 'statistical_tests.csv'}")

    print(f"单次实验结果已保存到：{output_root / 'summary_runs.csv'}")
    print(f"均值和标准差汇总已保存到：{output_root / 'summary_mean_std.csv'}")
    print(f"论文表格已保存到：{output_root / 'paper_table.csv'}")


if __name__ == "__main__":
    main()
