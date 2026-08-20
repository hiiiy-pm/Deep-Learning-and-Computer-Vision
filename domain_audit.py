from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, median, pstdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter

from src.data import IMG_EXTENSIONS, build_external_labeled_dataset, build_transforms, make_image_folder
from src.metrics import PAPER_COLORS
from src.utils import class_name_zh, configure_chinese_matplotlib, ensure_dir


FEATURES = [
    ("brightness", "亮度"),
    ("contrast", "对比度"),
    ("saturation", "饱和度"),
    ("sharpness", "清晰度"),
    ("width", "宽度"),
    ("height", "高度"),
    ("aspect_ratio", "宽高比"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分析训练集与外部验证集之间的图像域差异。")
    parser.add_argument("--train-dir", default="data", help="数据集目录")
    parser.add_argument("--external-dir", default="val_data", help="外部验证集目录")
    parser.add_argument("--output-dir", default="outputs/domain_audit", help="输出目录")
    parser.add_argument("--max-samples-per-class", type=int, default=0, help="每类最多抽样数量；0 表示使用全部样本")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _laplacian_variance(gray: np.ndarray) -> float:
    padded = np.pad(gray, 1, mode="edge")
    lap = (
        -4.0 * padded[1:-1, 1:-1]
        + padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    )
    return float(lap.var())


def image_features(path: Path) -> dict[str, float]:
    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        small = image.resize((128, 128))
        rgb = np.asarray(small, dtype=np.float32) / 255.0
        gray = np.asarray(small.convert("L"), dtype=np.float32) / 255.0
        hsv = np.asarray(small.convert("HSV"), dtype=np.float32) / 255.0
        # PIL conversion can smooth very noisy images; a tiny edge enhancement makes
        # the sharpness statistic less dominated by resizing artifacts.
        edge_gray = np.asarray(small.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0
    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "saturation": float(hsv[:, :, 1].mean()),
        "sharpness": float(0.5 * _laplacian_variance(gray) + 0.5 * edge_gray.var()),
        "width": float(width),
        "height": float(height),
        "aspect_ratio": float(width / max(height, 1)),
        "red_mean": float(rgb[:, :, 0].mean()),
        "green_mean": float(rgb[:, :, 1].mean()),
        "blue_mean": float(rgb[:, :, 2].mean()),
    }


def collect_rows(
    samples: list[tuple[Path, int]],
    class_names: list[str],
    split_name: str,
    max_samples_per_class: int,
    seed: int,
) -> list[dict[str, str | float]]:
    rng = np.random.default_rng(seed)
    by_class: dict[int, list[Path]] = {idx: [] for idx in range(len(class_names))}
    for path, target in samples:
        by_class[int(target)].append(path)

    selected: list[tuple[Path, int]] = []
    for target, paths in by_class.items():
        paths = list(paths)
        if max_samples_per_class > 0 and len(paths) > max_samples_per_class:
            indices = rng.choice(len(paths), size=max_samples_per_class, replace=False)
            paths = [paths[int(index)] for index in sorted(indices)]
        selected.extend((path, target) for path in paths)

    rows: list[dict[str, str | float]] = []
    for path, target in selected:
        try:
            values = image_features(path)
        except Exception as exc:
            print(f"警告：跳过无法读取的图片：{path}；原因：{exc}")
            continue
        rows.append(
            {
                "数据集": split_name,
                "类别": class_name_zh(class_names[target]),
                "类别英文": class_names[target],
                "图片路径": str(path),
                **values,
            }
        )
    return rows


def summarize(rows: list[dict[str, str | float]]) -> list[dict[str, str | float]]:
    groups: dict[tuple[str, str], list[dict[str, str | float]]] = {}
    for row in rows:
        groups.setdefault((str(row["数据集"]), str(row["类别"])), []).append(row)

    summary_rows: list[dict[str, str | float]] = []
    for (split_name, class_name), group in sorted(groups.items()):
        summary: dict[str, str | float] = {
            "数据集": split_name,
            "类别": class_name,
            "样本数": len(group),
        }
        for key, display_name in FEATURES:
            values = [float(row[key]) for row in group]
            summary[f"{display_name}_均值"] = mean(values)
            summary[f"{display_name}_中位数"] = median(values)
            summary[f"{display_name}_标准差"] = pstdev(values) if len(values) > 1 else 0.0
        summary_rows.append(summary)
    return summary_rows


def _write_csv(rows: list[dict[str, str | float]], path: Path) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _effect_size(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    var_a = np.var(a, ddof=1) if len(a) > 1 else 0.0
    var_b = np.var(b, ddof=1) if len(b) > 1 else 0.0
    pooled = np.sqrt(((len(a) - 1) * var_a + (len(b) - 1) * var_b) / max(len(a) + len(b) - 2, 1))
    if pooled <= 1e-12:
        return 0.0
    return float((np.mean(b) - np.mean(a)) / pooled)


def save_domain_shift_plot(rows: list[dict[str, str | float]], output_path: Path) -> None:
    splits = list(dict.fromkeys(str(row["数据集"]) for row in rows))
    if len(splits) < 2:
        return
    reference_split, external_split = splits[0], splits[1]

    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.8))
    fig.suptitle("训练集与外部验证集图像域差异分析", y=0.98, fontsize=14)
    colors = [PAPER_COLORS["blue"], PAPER_COLORS["orange"]]
    selected_features = FEATURES[:6]
    for ax, (feature_key, feature_name) in zip(axes.ravel(), selected_features):
        data = [
            [float(row[feature_key]) for row in rows if str(row["数据集"]) == split_name]
            for split_name in splits[:2]
        ]
        parts = ax.violinplot(data, showmeans=False, showmedians=True, widths=0.78)
        for idx, body in enumerate(parts["bodies"]):
            body.set_facecolor(colors[idx % len(colors)])
            body.set_edgecolor("#222222")
            body.set_alpha(0.52)
        for key in ["cbars", "cmins", "cmaxes", "cmedians"]:
            if key in parts:
                parts[key].set_color("#222222")
                parts[key].set_linewidth(0.9)
        x_positions = [1, 2]
        rng = np.random.default_rng(42)
        for idx, values in enumerate(data):
            values_array = np.asarray(values, dtype=np.float64)
            if values_array.size > 500:
                values_array = rng.choice(values_array, size=500, replace=False)
            jitter = rng.normal(0.0, 0.035, size=values_array.size)
            ax.scatter(
                np.full(values_array.size, x_positions[idx]) + jitter,
                values_array,
                s=4.5,
                color=colors[idx % len(colors)],
                alpha=0.18,
                linewidths=0,
            )
        d_value = _effect_size(data[0], data[1])
        ax.set_title(f"{feature_name}（Cohen's d={d_value:.2f}）", loc="left", pad=4)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(splits[:2], rotation=12, ha="right")
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    effect_rows = []
    for feature_key, feature_name in FEATURES:
        reference_values = [float(row[feature_key]) for row in rows if str(row["数据集"]) == reference_split]
        external_values = [float(row[feature_key]) for row in rows if str(row["数据集"]) == external_split]
        effect_rows.append(
            {
                "特征": feature_name,
                "训练集均值": float(np.mean(reference_values)) if reference_values else 0.0,
                "外部验证集均值": float(np.mean(external_values)) if external_values else 0.0,
                "Cohen_d": _effect_size(reference_values, external_values),
            }
        )
    _write_csv(effect_rows, output_path.with_name("domain_shift_effect_size.csv"))


def main() -> None:
    args = parse_args()
    configure_chinese_matplotlib()
    output_dir = ensure_dir(args.output_dir)

    train_dataset = make_image_folder(args.train_dir, build_transforms(224, train=False))
    external_dataset, source = build_external_labeled_dataset(
        args.external_dir,
        build_transforms(224, train=False),
        train_dataset.classes,
    )
    if external_dataset is None:
        raise FileNotFoundError(source)

    train_samples = [(Path(path), int(target)) for path, target in train_dataset.samples]
    external_samples = [(Path(path), int(target)) for path, target in external_dataset.samples]
    rows = []
    rows.extend(
        collect_rows(
            train_samples,
            train_dataset.classes,
            "训练集",
            args.max_samples_per_class,
            args.seed,
        )
    )
    rows.extend(
        collect_rows(
            external_samples,
            train_dataset.classes,
            "外部验证集",
            args.max_samples_per_class,
            args.seed + 1,
        )
    )

    _write_csv(rows, output_dir / "domain_image_features.csv")
    _write_csv(summarize(rows), output_dir / "domain_feature_summary.csv")
    save_domain_shift_plot(rows, output_dir / "domain_shift_analysis.png")

    print(f"训练集样本数：{len(train_samples)}")
    print(f"外部验证集样本数：{len(external_samples)} | 来源：{source}")
    print(f"图像域差异分析已保存到：{output_dir}")


if __name__ == "__main__":
    main()
