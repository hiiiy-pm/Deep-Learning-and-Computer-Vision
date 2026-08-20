from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from torch import nn

from src.models import MODEL_CHOICES, build_model
from src.utils import count_parameters, describe_device, ensure_dir, get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统计模型参数量、近似 FLOPs 和推理速度。")
    parser.add_argument("--models", nargs="+", default=["cnn", "cnn_mamba"], choices=MODEL_CHOICES)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--output-dir", default="outputs/model_profile")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def module_flops(module: nn.Module, inputs, output) -> int:
    if isinstance(module, nn.Conv2d):
        batch_size = output.shape[0]
        out_channels, out_h, out_w = output.shape[1:]
        kernel_h, kernel_w = module.kernel_size
        in_channels = module.in_channels // module.groups
        return int(batch_size * out_channels * out_h * out_w * in_channels * kernel_h * kernel_w * 2)
    if isinstance(module, nn.Linear):
        batch_size = output.shape[0] if output.ndim > 1 else 1
        return int(batch_size * module.in_features * module.out_features * 2)
    return 0


@torch.no_grad()
def estimate_flops(model: nn.Module, sample: torch.Tensor) -> int:
    flops = 0
    handles = []

    def hook(module, inputs, output):
        nonlocal flops
        flops += module_flops(module, inputs, output)

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handles.append(module.register_forward_hook(hook))
    model(sample)
    for handle in handles:
        handle.remove()
    return flops


@torch.no_grad()
def benchmark_latency(model: nn.Module, sample: torch.Tensor, warmup: int, iters: int, device: torch.device) -> float:
    model.eval()
    for _ in range(warmup):
        model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed * 1000.0 / iters


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    output_dir = ensure_dir(args.output_dir)
    print(f"统计设备：{describe_device(device)}")

    rows: list[dict[str, str | float]] = []
    for model_name in args.models:
        model = build_model(
            model_name=model_name,
            num_classes=args.num_classes,
            pretrained=args.pretrained,
            freeze_backbone=args.freeze_backbone,
        ).to(device)
        sample = torch.randn(args.batch_size, 3, args.img_size, args.img_size, device=device)
        params = count_parameters(model)
        flops = estimate_flops(model, sample)
        latency_ms = benchmark_latency(model, sample, args.warmup, args.iters, device)
        row = {
            "模型": model_name,
            "参数量(M)": params / 1e6,
            "FLOPs(G)": flops / 1e9,
            "单批推理时间(ms)": latency_ms,
            "batch_size": args.batch_size,
            "输入尺寸": f"{args.img_size}x{args.img_size}",
        }
        rows.append(row)
        print(
            f"{model_name}: 参数量={row['参数量(M)']:.3f}M | "
            f"FLOPs={row['FLOPs(G)']:.3f}G | 推理时间={latency_ms:.2f}ms"
        )

    output_path = Path(output_dir) / "model_profile.csv"
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"模型复杂度表已保存到：{output_path}")


if __name__ == "__main__":
    main()
