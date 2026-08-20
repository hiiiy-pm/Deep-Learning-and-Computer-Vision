from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.data import read_labeled_image_samples
from src.models import build_model
from src.utils import class_name_zh, configure_chinese_matplotlib, describe_device, ensure_dir, get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 Grad-CAM 可解释性热力图。")
    parser.add_argument("--checkpoint", default="outputs/cnn_mamba_internal_val/best.pt")
    parser.add_argument("--data-dir", default="test_data")
    parser.add_argument("--output-dir", default="outputs/grad_cam")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--samples-per-class", type=int, default=1)
    parser.add_argument("--target", default="fusion", choices=["fusion", "cnn_stem", "cnn_mamba"],
                        help="目标层：fusion=融合后精炼层, cnn_stem=CNN骨干末层, cnn_mamba=两分支同时生成")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def find_image_samples(root: str | Path, class_names: list[str], samples_per_class: int, seed: int) -> list[tuple[Path, int]]:
    root = Path(root)
    rng = random.Random(seed)
    samples_with_labels, source = read_labeled_image_samples(root, class_names)
    if not samples_with_labels:
        raise FileNotFoundError(source)
    print(f"Grad-CAM 数据来源：{source}")

    buckets: dict[int, list[Path]] = {idx: [] for idx in range(len(class_names))}
    for image_path, target in samples_with_labels:
        buckets[int(target)].append(Path(image_path))

    samples: list[tuple[Path, int]] = []
    for target, paths in buckets.items():
        if not paths:
            continue
        rng.shuffle(paths)
        samples.extend((path, target) for path in paths[:samples_per_class])
    if not samples:
        raise FileNotFoundError(f"没有在 {root} 中找到可用于 Grad-CAM 的图片。")
    return samples


def build_eval_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.handles = [
            target_layer.register_forward_hook(self._forward_hook),
            target_layer.register_full_backward_hook(self._backward_hook),
        ]

    def _forward_hook(self, _module, _inputs, output) -> None:
        self.activations = output.detach()

    def _backward_hook(self, _module, _grad_inputs, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def __call__(self, image_tensor: torch.Tensor, class_idx: int | None = None) -> tuple[np.ndarray, int, float]:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = int(probs.argmax(dim=1).item()) if class_idx is None else int(class_idx)
        score = logits[:, pred_idx].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("没有捕获到目标层的激活或梯度。")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / max(cam.max(), 1e-8)
        return cam, pred_idx, float(probs[0, pred_idx].item())


def overlay_heatmap(image: Image.Image, cam: np.ndarray, alpha: float = 0.42) -> np.ndarray:
    image_array = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    cmap = plt.get_cmap("turbo")
    heatmap = cmap(cam)[..., :3]
    blended = (1 - alpha) * image_array + alpha * heatmap
    return np.clip(blended, 0.0, 1.0)


def save_grad_cam_grid(
    rows: list[dict[str, object]],
    output_path: Path,
    target_label: str = "",
) -> None:
    n_cols = len(rows[0]) - 3 if rows else 1  # image/true_name/pred_name/confidence are metadata
    # Determine layout: we expect rows of [image, overlay, ...] or [image, overlay_a, overlay_b, ...]
    overlay_keys = [k for k in rows[0] if k.startswith("overlay")]
    n_cols = 1 + len(overlay_keys)
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(3.5 * n_cols, 2.0 * len(rows)))
    if len(rows) == 1:
        axes = np.asarray([axes])
    subtitle = f"（目标层：{target_label}）" if target_label else ""
    fig.suptitle(f"Grad-CAM 可解释性分析{subtitle}", y=0.992, fontsize=13)

    for row_idx, row in enumerate(rows):
        image = row["image"]
        true_name = row["true_name"]

        axes[row_idx, 0].imshow(image)
        axes[row_idx, 0].set_title(f"{true_name} 原图", pad=3, fontsize=10)
        axes[row_idx, 0].axis("off")

        for col_idx, key in enumerate(overlay_keys, start=1):
            label = key.replace("overlay_", "").replace("_", " ")
            conf = row.get(f"confidence_{label}", row["confidence"])
            axes[row_idx, col_idx].imshow(row[key])
            axes[row_idx, col_idx].set_title(f"{label}（{conf:.3f}）", pad=3, fontsize=10)
            axes[row_idx, col_idx].axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=0.55, w_pad=0.20)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _resolve_target_layer(model: torch.nn.Module, target_name: str) -> tuple[torch.nn.Module, str]:
    """Resolve a human-readable target name to a concrete layer and label."""
    has_fusion = hasattr(model, "fusion") and model.fusion is not None
    fusion = model.fusion if has_fusion else None

    if target_name == "fusion":
        if fusion is not None and hasattr(fusion, "refine"):
            return fusion.refine.block[0], "融合层"
        elif fusion is not None and hasattr(fusion, "net"):
            return fusion.net[-1].block[0], "融合层"
        else:
            return model.stem.net[-1].block[0], "CNN末层(无融合)"

    if target_name == "cnn_stem":
        return model.stem.net[-1].block[0], "CNN骨干"

    raise ValueError(f"未知目标层：{target_name}")


def main() -> None:
    args = parse_args()
    configure_chinese_matplotlib()
    device = get_device(args.device)
    output_dir = ensure_dir(args.output_dir)
    print(f"Grad-CAM 设备：{describe_device(device)}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    class_names: list[str] = checkpoint["class_names"]
    model = build_model(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    transform = build_eval_transform(args.img_size)
    samples = find_image_samples(args.data_dir, class_names, args.samples_per_class, args.seed)

    if args.target == "cnn_mamba":
        targets_to_run = ["cnn_stem", "fusion"]
    else:
        targets_to_run = [args.target]

    rows: list[dict[str, object]] = []
    for image_path, target_idx in samples:
        image = Image.open(image_path).convert("RGB").resize((args.img_size, args.img_size))
        image_tensor = transform(image).unsqueeze(0).to(device)
        row: dict[str, object] = {
            "image": image,
            "true_name": class_name_zh(class_names[target_idx]),
            "pred_name": "",
            "confidence": 0.0,
        }

        for target_name in targets_to_run:
            layer, label = _resolve_target_layer(model, target_name)
            grad_cam = GradCAM(model, layer)
            cam, pred_idx, confidence = grad_cam(image_tensor)
            grad_cam.close()
            row[f"overlay_{label}"] = overlay_heatmap(image, cam)
            if not row["pred_name"]:
                row["pred_name"] = class_name_zh(class_names[pred_idx])
                row["confidence"] = confidence
            else:
                row[f"confidence_{label}"] = confidence

        rows.append(row)

    output_path = output_dir / "grad_cam_examples.png"
    save_grad_cam_grid(rows, output_path)
    print(f"Grad-CAM 图已保存到：{output_path}")


if __name__ == "__main__":
    main()
