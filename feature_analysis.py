from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from src.data import build_external_labeled_dataset, build_loader, build_transforms
from src.metrics import CLASS_COLORS, PAPER_COLORS
from src.models import build_model
from src.utils import class_names_zh, configure_chinese_matplotlib, describe_device, ensure_dir, get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="提取模型特征并生成论文用特征空间可视化图。")
    parser.add_argument("--checkpoint", default="outputs/cnn_mamba_internal_val/best.pt")
    parser.add_argument("--data-dir", default="test_data")
    parser.add_argument("--output-dir", default="outputs/feature_analysis")
    parser.add_argument("--method", default="pca", choices=["pca", "tsne"], help="降维方法")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def pca_2d(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = features - features.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T
    eigenvalues = (singular_values**2) / max(1, len(features) - 1)
    explained = eigenvalues[:2] / max(eigenvalues.sum(), 1e-12)
    return coords, explained


def tsne_2d(features: np.ndarray, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=seed, perplexity=min(30, len(features) - 1))
    coords = tsne.fit_transform(features)
    explained = np.array([0.0, 0.0])  # t-SNE does not provide variance explained
    return coords, explained


@torch.no_grad()
def extract_features(
    model: nn.Module,
    loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_features: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        features = model.forward_features(images)
        logits = model.head(features)
        preds = logits.argmax(dim=1)
        all_features.append(features.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
    return np.concatenate(all_features), np.concatenate(all_targets), np.concatenate(all_preds)


def save_feature_plot(
    coords: np.ndarray,
    targets: np.ndarray,
    preds: np.ndarray,
    explained: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    display_names = class_names_zh(class_names)
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    for idx, display_name in enumerate(display_names):
        mask = targets == idx
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=18,
            alpha=0.72,
            color=CLASS_COLORS[idx % len(CLASS_COLORS)],
            edgecolors="white",
            linewidths=0.25,
            label=display_name,
        )
    wrong = targets != preds
    if wrong.any():
        ax.scatter(
            coords[wrong, 0],
            coords[wrong, 1],
            s=46,
            facecolors="none",
            edgecolors=PAPER_COLORS["red"],
            linewidths=1.0,
            label="错分样本",
        )
    method = "t-SNE" if explained.sum() < 1e-9 else "PCA"
    if method == "PCA":
        xlabel = f"主成分 1（解释方差 {explained[0] * 100:.1f}%）"
        ylabel = f"主成分 2（解释方差 {explained[1] * 100:.1f}%）"
    else:
        xlabel = "t-SNE 维度 1"
        ylabel = "t-SNE 维度 2"
    ax.set_title(f"模型深层特征 {method} 分布")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", linewidth=0.55, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", frameon=True, fancybox=False, edgecolor="#D0D0D0")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_feature_csv(
    coords: np.ndarray,
    targets: np.ndarray,
    preds: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    display_names = class_names_zh(class_names)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["样本序号", "真实类别", "预测类别", "主成分1", "主成分2", "是否正确"])
        writer.writeheader()
        for idx, (coord, target, pred) in enumerate(zip(coords, targets, preds), start=1):
            writer.writerow(
                {
                    "样本序号": idx,
                    "真实类别": display_names[int(target)],
                    "预测类别": display_names[int(pred)],
                    "主成分1": float(coord[0]),
                    "主成分2": float(coord[1]),
                    "是否正确": int(target == pred),
                }
            )


def main() -> None:
    args = parse_args()
    configure_chinese_matplotlib()
    device = get_device(args.device)
    output_dir = ensure_dir(args.output_dir)
    print(f"特征分析设备：{describe_device(device)}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    class_names: list[str] = checkpoint["class_names"]
    img_size = args.img_size if args.img_size > 0 else int(checkpoint.get("img_size", 224))
    dataset, source = build_external_labeled_dataset(
        args.data_dir,
        build_transforms(img_size, train=False),
        class_names,
    )
    if dataset is None:
        raise FileNotFoundError(source)
    print(f"数据来源：{source}")
    loader = build_loader(dataset, args.batch_size, False, args.num_workers, device)

    model = build_model(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    features, targets, preds = extract_features(model, loader, device)

    if args.method == "tsne":
        print("正在计算 t-SNE 降维（可能需要几秒到几十秒）...")
        coords, explained = tsne_2d(features)
    else:
        coords, explained = pca_2d(features)

    suffix = "tsne" if args.method == "tsne" else "pca"
    save_feature_plot(coords, targets, preds, explained, class_names, output_dir / f"feature_{suffix}.png")
    save_feature_csv(coords, targets, preds, class_names, output_dir / f"feature_{suffix}.csv")
    print(f"特征空间图已保存到：{output_dir / f'feature_{suffix}.png'}")


if __name__ == "__main__":
    main()
