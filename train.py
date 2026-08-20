from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn

from src.data import (
    build_datasets,
    build_external_labeled_dataset,
    build_loader,
    build_transforms,
    class_weights_from_targets,
    targets_from_dataset,
)
from src.engine import evaluate, matrix_to_list, train_one_epoch
from src.metrics import (
    PAPER_COLORS,
    class_report,
    save_dataset_distribution_plot,
    save_paper_evaluation_plots,
    scalar_metrics,
)
from src.models import MODEL_CHOICES, build_model
from src.utils import (
    class_names_zh,
    configure_chinese_matplotlib,
    count_parameters,
    describe_device,
    ensure_dir,
    format_class_pairs,
    get_device,
    require_cuda,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 CNN + Vision Mamba 玉米叶部病害分类模型。")
    parser.add_argument("--train-dir", default="data", help="数据集目录")
    parser.add_argument(
        "--val-dir",
        default="",
        help="同分布验证集目录；留空时从 data 内部分层划分验证集",
    )
    parser.add_argument(
        "--external-val-dir",
        default="val_data",
        help="外部田间验证集目录，只用于泛化评估，不用于早停和最佳模型选择",
    )
    parser.add_argument("--test-dir", default="", help="独立测试集目录；留空时跳过最终测试，避免误用同源重复数据")
    parser.add_argument("--allow-duplicate-test", action="store_true", help="允许使用可能与训练集重复的测试集，仅建议调试使用")
    parser.add_argument("--exclude-duplicates", default="", help="JSON 文件路径，包含需从评估集排除的重复文件路径（由 check_duplicates.py --export-duplicate-list 生成）")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="从训练集分层划分独立测试集的比例（推荐 0.15-0.20），避免使用可能重复的 --test-dir")
    parser.add_argument("--output-dir", default="outputs/cnn_mamba_internal_val", help="模型、日志和图片输出目录")
    parser.add_argument("--model", default="cnn_mamba", choices=MODEL_CHOICES, help="模型结构")
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--external-img-size", type=int, default=384, help="外部田间验证集评估尺寸")
    parser.add_argument("--eval-crops", type=int, default=1, choices=[1, 5, 10], help="内部验证和测试的多裁剪评估数量")
    parser.add_argument("--external-eval-crops", type=int, default=5, choices=[1, 5, 10], help="外部田间验证的多裁剪评估数量")
    parser.add_argument(
        "--augment",
        default="domain",
        choices=["standard", "strong", "randaugment", "domain"],
        help="训练数据增强强度；domain 针对跨数据集颜色、清晰度、宽高比偏移",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--external-batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2, help="内部验证集划分比例")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--d-state", type=int, default=16, help="S6 SSM 状态维度 N，越大选择性越强但计算量增加")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--mlp-ratio", type=float, default=2.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--mixup-alpha", type=float, default=0.0, help="MixUp alpha；0 表示关闭")
    parser.add_argument("--cutmix-alpha", type=float, default=0.0, help="CutMix alpha；0 表示关闭")
    parser.add_argument("--pretrained", action="store_true", help="torchvision 基线模型使用 ImageNet 预训练权重")
    parser.add_argument("--freeze-backbone", action="store_true", help="冻结 torchvision 预训练骨干，只训练分类头")
    parser.add_argument("--patience", type=int, default=8, help="内部验证集 macro F1 连续不提升时的早停轮数")
    parser.add_argument("--device", default="auto", help="auto、cuda 或 cpu；默认优先使用 GPU")
    parser.add_argument("--allow-cpu", action="store_true", help="没有 GPU 时允许退回 CPU")
    parser.add_argument("--amp", dest="amp", action="store_true", default=True, help="启用 CUDA 自动混合精度")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="关闭 CUDA 自动混合精度")
    parser.add_argument("--no-class-weights", dest="class_weights", action="store_false")
    parser.set_defaults(class_weights=True)
    return parser.parse_args()


def save_curves(history: list[dict[str, float]], output_dir: Path) -> None:
    if not history:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="训练损失", color=PAPER_COLORS["blue"])
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="内部验证损失", color=PAPER_COLORS["green"])
    if "external_val_loss" in history[0]:
        axes[0].plot(
            epochs,
            [row["external_val_loss"] for row in history],
            label="外部验证损失",
            color=PAPER_COLORS["orange"],
        )
    axes[0].set_title("损失曲线")
    axes[0].set_xlabel("训练轮次")
    axes[0].set_ylabel("损失")
    axes[0].legend(frameon=False)

    axes[1].plot(epochs, [row["train_acc"] for row in history], label="训练准确率", color=PAPER_COLORS["blue"])
    axes[1].plot(epochs, [row["val_acc"] for row in history], label="内部验证准确率", color=PAPER_COLORS["green"])
    axes[1].plot(
        epochs,
        [row["val_macro_f1"] for row in history],
        label="内部验证宏平均 F1",
        color=PAPER_COLORS["purple"],
    )
    if "external_val_acc" in history[0]:
        axes[1].plot(
            epochs,
            [row["external_val_acc"] for row in history],
            label="外部验证准确率",
            color=PAPER_COLORS["orange"],
        )
        axes[1].plot(
            epochs,
            [row["external_val_macro_f1"] for row in history],
            label="外部验证宏平均 F1",
            color=PAPER_COLORS["red"],
        )
    axes[1].set_title("评估指标")
    axes[1].set_xlabel("训练轮次")
    axes[1].set_ylabel("指标值")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].legend(frameon=False)

    best_internal_idx = max(range(len(history)), key=lambda idx: history[idx]["val_macro_f1"])
    best_epoch = int(history[best_internal_idx]["epoch"])
    best_f1 = float(history[best_internal_idx]["val_macro_f1"])
    axes[1].scatter([best_epoch], [best_f1], color=PAPER_COLORS["black"], s=34, zorder=5)
    axes[1].annotate(
        f"最佳内部F1={best_f1:.3f}",
        xy=(best_epoch, best_f1),
        xytext=(6, -16),
        textcoords="offset points",
        fontsize=8.5,
        color=PAPER_COLORS["black"],
    )

    for ax in axes:
        ax.grid(True, linestyle="--", linewidth=0.55, alpha=0.28, color="#9A9A9A")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "training_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def write_history_csv(history: list[dict[str, float]], path: Path) -> None:
    if not history:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def dataset_class_counts(dataset, num_classes: int) -> list[int]:
    targets = targets_from_dataset(dataset)
    return torch.bincount(torch.tensor(targets, dtype=torch.long), minlength=num_classes).int().tolist()


def save_eval_artifacts(
    metrics: dict,
    class_names: list[str],
    output_dir: Path,
    prefix: str,
    title: str,
    epoch: int | None = None,
) -> None:
    payload = {
        "metrics": scalar_metrics(metrics),
        "confusion_matrix": matrix_to_list(metrics["confusion_matrix"]),
        "class_report": class_report(metrics["confusion_matrix"], class_names),
    }
    if epoch is not None:
        payload["epoch"] = epoch
    save_json(payload, output_dir / f"{prefix}_metrics.json")
    save_paper_evaluation_plots(metrics, class_names, output_dir, prefix, title_prefix=title.replace("混淆矩阵", ""))


def main() -> None:
    args = parse_args()
    chosen_font = configure_chinese_matplotlib()
    set_seed(args.seed)
    device = get_device(args.device)
    require_cuda(device, allow_cpu=args.allow_cpu)
    output_dir = ensure_dir(args.output_dir)

    exclude_test_paths: set[str] = set()
    exclude_val_paths: set[str] = set()
    if args.exclude_duplicates:
        import json
        dup_json = Path(args.exclude_duplicates)
        if dup_json.exists():
            dup_map = json.loads(dup_json.read_text(encoding="utf-8"))
            test_dir_str = str(Path(args.test_dir).resolve()) if args.test_dir else ""
            val_dir_str = str(Path(args.val_dir).resolve()) if args.val_dir else str(Path(args.external_val_dir).resolve())
            for qdir, paths in dup_map.items():
                path_set = set(paths)
                if test_dir_str and Path(qdir).resolve() == Path(test_dir_str).resolve():
                    exclude_test_paths |= path_set
                if val_dir_str and Path(qdir).resolve() == Path(val_dir_str).resolve():
                    exclude_val_paths |= path_set
            print(f"从去重列表加载：排除测试集 {len(exclude_test_paths)} 文件，排除验证集 {len(exclude_val_paths)} 文件")

    val_dir = args.val_dir if args.val_dir else None
    train_ds, val_ds, test_ds, class_names, val_source = build_datasets(
        train_dir=args.train_dir,
        val_dir=val_dir,
        test_dir=args.test_dir if args.allow_duplicate_test else "",
        img_size=args.img_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        augment=args.augment,
        eval_crops=args.eval_crops,
        exclude_test_paths=exclude_test_paths,
        exclude_val_paths=exclude_val_paths,
        test_ratio=args.test_ratio,
    )
    train_loader = build_loader(train_ds, args.batch_size, True, args.num_workers, device)
    val_loader = build_loader(val_ds, args.eval_batch_size, False, args.num_workers, device)
    test_loader = build_loader(test_ds, args.eval_batch_size, False, args.num_workers, device) if test_ds else None

    external_val_ds = None
    external_val_source = ""
    if args.external_val_dir:
        external_val_ds, external_val_source = build_external_labeled_dataset(
            args.external_val_dir,
            build_transforms(args.external_img_size, train=False, eval_crops=args.external_eval_crops),
            class_names,
        )
    external_val_loader = (
        build_loader(external_val_ds, args.external_batch_size, False, args.num_workers, device)
        if external_val_ds is not None
        else None
    )
    split_counts = {
        "训练集": dataset_class_counts(train_ds, len(class_names)),
        "内部验证集": dataset_class_counts(val_ds, len(class_names)),
    }
    if external_val_ds is not None:
        split_counts["外部验证集"] = dataset_class_counts(external_val_ds, len(class_names))
    if test_ds is not None:
        split_counts["测试集"] = dataset_class_counts(test_ds, len(class_names))
    save_dataset_distribution_plot(
        split_counts,
        class_names,
        output_dir / "dataset_distribution.png",
        title="玉米叶部病害数据集类别分布",
    )

    model_config = {
        "model_name": args.model,
        "num_classes": len(class_names),
        "embed_dim": args.embed_dim,
        "d_state": args.d_state,
        "depth": args.depth,
        "mlp_ratio": args.mlp_ratio,
        "dropout": args.dropout,
        "pretrained": args.pretrained,
        "freeze_backbone": args.freeze_backbone,
    }
    model = build_model(**model_config).to(device)

    weights = None
    if args.class_weights:
        weights = class_weights_from_targets(targets_from_dataset(train_ds), len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    save_json(
        {
            "class_names": class_names_zh(class_names),
            "class_names_en": class_names,
            "class_to_idx": {class_names_zh(class_names)[idx]: idx for idx in range(len(class_names))},
            "class_to_idx_en": {name: idx for idx, name in enumerate(class_names)},
            "model_config": model_config,
            "train_dir": str(args.train_dir),
            "val_source": val_source,
            "external_val_source": external_val_source,
            "test_dir": str(args.test_dir),
            "allow_duplicate_test": args.allow_duplicate_test,
            "img_size": args.img_size,
            "external_img_size": args.external_img_size,
            "eval_crops": args.eval_crops,
            "external_eval_crops": args.external_eval_crops,
            "augment": args.augment,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "external_batch_size": args.external_batch_size,
            "class_weights": args.class_weights,
            "label_smoothing": args.label_smoothing,
            "mixup_alpha": args.mixup_alpha,
            "cutmix_alpha": args.cutmix_alpha,
            "pretrained": args.pretrained,
            "freeze_backbone": args.freeze_backbone,
            "best_model_selection": "内部验证集 macro F1",
        },
        output_dir / "run_config.json",
    )

    print(f"训练设备：{describe_device(device)}")
    print(f"中文字体：{chosen_font or '未检测到常见中文字体，图片中中文可能显示为方框'}")
    print(f"类别显示：{format_class_pairs(class_names)}")
    print(f"训练样本数：{len(train_ds)} | 内部验证样本数：{len(val_ds)} | 内部验证来源：{val_source}")
    if external_val_ds is not None:
        print(f"外部验证样本数：{len(external_val_ds)} | 外部验证来源：{external_val_source}")
    if test_ds is not None:
        print(f"测试样本数：{len(test_ds)}")
    elif args.test_dir and not args.allow_duplicate_test:
        print("测试集评估：已跳过。若确认测试集独立，可加入 --allow-duplicate-test 启用。")
    print(f"模型：{args.model} | 可训练参数量：{count_parameters(model):,}")
    print(f"混合精度训练：{'开启' if args.amp and device.type == 'cuda' else '关闭'}")
    print(f"训练增强：{args.augment} | 标签平滑：{args.label_smoothing}")
    print(f"混合增强：MixUp alpha={args.mixup_alpha} | CutMix alpha={args.cutmix_alpha}")
    if args.pretrained:
        print(f"预训练骨干：开启 | 冻结骨干：{'是' if args.freeze_backbone else '否'}")
    print(f"外部验证尺寸：{args.external_img_size} | 外部验证多裁剪：{args.external_eval_crops}")
    print("最佳模型选择依据：内部验证集宏平均 F1；外部验证集只用于泛化评估。")

    best_f1 = -1.0
    best_epoch = 0
    best_external_f1 = -1.0
    best_external_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    for epoch in range(1, args.epochs + 1):
        train_metrics, scaler = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            amp=args.amp,
            num_classes=len(class_names),
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            scaler=scaler,
        )
        val_metrics = evaluate(model, val_loader, criterion, device, len(class_names), amp=args.amp)
        external_val_metrics = (
            evaluate(model, external_val_loader, criterion, device, len(class_names), amp=args.amp)
            if external_val_loader is not None
            else None
        )
        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_macro_precision": val_metrics["macro_precision"],
            "val_macro_recall": val_metrics["macro_recall"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        if external_val_metrics is not None:
            row.update(
                {
                    "external_val_loss": external_val_metrics["loss"],
                    "external_val_acc": external_val_metrics["accuracy"],
                    "external_val_macro_precision": external_val_metrics["macro_precision"],
                    "external_val_macro_recall": external_val_metrics["macro_recall"],
                    "external_val_macro_f1": external_val_metrics["macro_f1"],
                }
            )
        history.append(row)
        write_history_csv(history, output_dir / "history.csv")
        try:
            save_curves(history, output_dir)
        except Exception as exc:
            print(f"警告：训练曲线保存失败：{exc}")

        print(
            f"第 {epoch:03d} 轮汇总 | "
            f"训练损失={row['train_loss']:.4f} 训练准确率={row['train_acc']:.4f} "
            f"内部验证损失={row['val_loss']:.4f} 内部验证准确率={row['val_acc']:.4f} "
            f"内部验证宏平均F1={row['val_macro_f1']:.4f}"
        )
        if external_val_metrics is not None:
            print(
                f"第 {epoch:03d} 轮外部验证 | "
                f"损失={external_val_metrics['loss']:.4f} 准确率={external_val_metrics['accuracy']:.4f} "
                f"宏平均F1={external_val_metrics['macro_f1']:.4f}"
            )

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "class_names": class_names,
            "model_config": model_config,
            "img_size": args.img_size,
            "best_f1": best_f1,
            "args": vars(args),
        }
        torch.save(checkpoint, output_dir / "last.pt")

        if external_val_metrics is not None and external_val_metrics["macro_f1"] > best_external_f1:
            best_external_f1 = float(external_val_metrics["macro_f1"])
            best_external_epoch = epoch
            diagnostic_checkpoint = dict(checkpoint)
            diagnostic_checkpoint["best_external_f1"] = best_external_f1
            diagnostic_checkpoint["selection_note"] = (
                "Diagnostic only: selected by external validation macro F1. "
                "Do not use as the primary final model unless the study protocol explicitly allows external validation model selection."
            )
            torch.save(diagnostic_checkpoint, output_dir / "best_external_diagnostic.pt")
            save_eval_artifacts(
                external_val_metrics,
                class_names,
                output_dir,
                prefix="external_val_best_diagnostic",
                title=f"外部验证集诊断最优混淆矩阵 - 第 {epoch} 轮",
                epoch=epoch,
            )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = float(val_metrics["macro_f1"])
            best_epoch = epoch
            epochs_without_improvement = 0
            checkpoint["best_f1"] = best_f1
            torch.save(checkpoint, output_dir / "best.pt")
            save_eval_artifacts(
                val_metrics,
                class_names,
                output_dir,
                prefix="best_internal_val",
                title=f"内部验证集混淆矩阵 - 第 {epoch} 轮",
                epoch=epoch,
            )
            if external_val_metrics is not None:
                save_eval_artifacts(
                    external_val_metrics,
                    class_names,
                    output_dir,
                    prefix="external_val_at_best_internal",
                    title=f"外部验证集混淆矩阵 - 第 {epoch} 轮",
                    epoch=epoch,
                )
            print(f"已保存新的最佳模型：内部验证宏平均F1={best_f1:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"内部验证宏平均F1暂未提升：{epochs_without_improvement}/{args.patience} 轮")

        if epochs_without_improvement >= args.patience:
            print(f"触发早停：当前第 {epoch} 轮；最佳轮次第 {best_epoch} 轮，最佳内部验证宏平均F1={best_f1:.4f}")
            break

    if (output_dir / "best.pt").exists():
        best_ckpt = torch.load(output_dir / "best.pt", map_location=device)
        model.load_state_dict(best_ckpt["model_state"])

        if external_val_loader is not None:
            external_val_metrics = evaluate(model, external_val_loader, criterion, device, len(class_names), amp=args.amp)
            save_eval_artifacts(
                external_val_metrics,
                class_names,
                output_dir,
                prefix="external_val_final",
                title="外部验证集混淆矩阵",
            )
            print(
                "外部验证集最终汇总 | "
                f"损失={external_val_metrics['loss']:.4f} 准确率={external_val_metrics['accuracy']:.4f} "
                f"宏平均F1={external_val_metrics['macro_f1']:.4f}"
            )
            if best_external_epoch:
                print(
                    "外部验证集诊断最优 | "
                    f"第 {best_external_epoch} 轮 宏平均F1={best_external_f1:.4f}；"
                    "该结果仅用于分析内部模型选择与外部泛化不一致，不作为默认最终模型。"
                )

        if test_loader is not None:
            test_metrics = evaluate(model, test_loader, criterion, device, len(class_names), amp=args.amp)
            save_eval_artifacts(test_metrics, class_names, output_dir, prefix="test", title="测试集混淆矩阵")
            print(
                "测试集汇总 | "
                f"损失={test_metrics['loss']:.4f} 准确率={test_metrics['accuracy']:.4f} "
                f"宏平均F1={test_metrics['macro_f1']:.4f}"
            )

    print(f"训练完成。输出文件已保存到：{output_dir}")


if __name__ == "__main__":
    main()
