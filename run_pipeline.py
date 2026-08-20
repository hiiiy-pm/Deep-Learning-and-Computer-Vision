from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="一键运行玉米叶部病害分类论文实验流水线。")
    parser.add_argument("--epochs", type=int, default=30, help="主模型训练轮数")
    parser.add_argument("--batch-size", type=int, default=32, help="训练 batch size")
    parser.add_argument("--output-root", default="outputs/paper_pipeline", help="流水线输出根目录")
    parser.add_argument("--model", default="cnn_mamba", help="主训练模型")
    parser.add_argument("--img-size", type=int, default=384, help="训练与内部验证图像尺寸")
    parser.add_argument("--augment", default="domain", choices=["standard", "strong", "randaugment", "domain"])
    parser.add_argument("--mixup-alpha", type=float, default=0.1)
    parser.add_argument("--cutmix-alpha", type=float, default=0.5)
    parser.add_argument("--external-img-size", type=int, default=384)
    parser.add_argument("--external-eval-crops", type=int, default=5, choices=[1, 5, 10])
    parser.add_argument("--train-dir", default="data")
    parser.add_argument("--external-dir", default="val_data")
    parser.add_argument("--test-dir", default="", help="外部测试集目录；留空时使用 --test-ratio 从训练集划分")
    parser.add_argument("--samples-per-class", type=int, default=3, help="Grad-CAM 每类样本数")
    parser.add_argument("--skip-train", action="store_true", help="跳过训练，直接使用 --checkpoint")
    parser.add_argument("--checkpoint", default="", help="跳过训练或复用模型时指定 best.pt")
    parser.add_argument("--skip-explain", action="store_true", help="跳过 Grad-CAM 和 PCA")
    parser.add_argument("--skip-profile", action="store_true", help="跳过复杂度统计")
    parser.add_argument("--skip-error-analysis", action="store_true", help="跳过外部验证错误样本导出")
    parser.add_argument("--run-ablation", action="store_true", help="额外运行多模型/多种子消融对比，耗时较长")
    parser.add_argument("--ablation-epochs", type=int, default=30)
    parser.add_argument("--ablation-seeds", nargs="+", type=int, default=[42, 2024, 3407])
    parser.add_argument("--exact-only", action="store_true", default=True, help="重复检查只做精确重复，不做感知哈希近重复")
    parser.add_argument("--near-duplicate", action="store_true", help="重复检查启用感知哈希近重复搜索，耗时更长")
    parser.add_argument("--pixel-exact", action="store_true", help="重复检查额外启用解码像素级精确重复，较慢")
    parser.add_argument("--within-dataset-audit", action="store_true", default=True, help="检查待评估数据集内部文件级重复")
    parser.add_argument("--skip-pixel-exact", action="store_true", help="兼容旧参数：重复检查跳过解码像素级精确重复")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="从训练集分层划分独立测试集的比例")
    parser.add_argument("--dry-run", action="store_true", help="只打印将要执行的命令，不真正运行")
    return parser.parse_args()


def run_step(name: str, cmd: list[str], dry_run: bool = False) -> None:
    printable = " ".join(f'"{part}"' if " " in part else part for part in cmd)
    print(f"\n========== {name} ==========")
    print(printable)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    python = sys.executable
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    train_output = output_root / args.model
    checkpoint = Path(args.checkpoint) if args.checkpoint else train_output / "best.pt"

    duplicate_cmd = [
        python,
        "check_duplicates.py",
        "--reference-dir",
        args.train_dir,
        "--query-dirs",
        args.test_dir,
        args.external_dir,
        "--output-dir",
        str(output_root / "data_audit"),
    ]
    if args.exact_only:
        duplicate_cmd.append("--exact-only")
    if args.near_duplicate and "--exact-only" in duplicate_cmd:
        duplicate_cmd.remove("--exact-only")
    if args.pixel_exact:
        duplicate_cmd.append("--pixel-exact")
    if args.within_dataset_audit:
        duplicate_cmd.append("--within-dataset")
    if args.skip_pixel_exact:
        duplicate_cmd.append("--skip-pixel-exact")
    duplicate_cmd.append("--export-duplicate-list")

    run_step(
        "1. 数据重复审计 + 导出重复路径",
        duplicate_cmd,
        args.dry_run,
    )

    run_step(
        "2. 训练集与外部验证集域差异分析",
        [
            python,
            "domain_audit.py",
            "--train-dir",
            args.train_dir,
            "--external-dir",
            args.external_dir,
            "--output-dir",
            str(output_root / "domain_audit"),
        ],
        args.dry_run,
    )

    if not args.skip_train:
        run_step(
            "3. 训练主模型",
            [
                python,
                "train.py",
                "--train-dir",
                args.train_dir,
                "--external-val-dir",
                args.external_dir,
                "--model",
                args.model,
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--img-size",
                str(args.img_size),
                "--augment",
                args.augment,
                "--mixup-alpha",
                str(args.mixup_alpha),
                "--cutmix-alpha",
                str(args.cutmix_alpha),
                "--external-img-size",
                str(args.external_img_size),
                "--external-eval-crops",
                str(args.external_eval_crops),
                "--test-ratio",
                str(args.test_ratio),
                "--output-dir",
                str(train_output),
                "--exclude-duplicates",
                str(output_root / "data_audit" / "duplicate_paths.json"),
            ],
            args.dry_run,
        )

    run_step(
        "4. 外部验证集独立评估",
        [
            python,
            "evaluate.py",
            "--checkpoint",
            str(checkpoint),
            "--data-dir",
            args.external_dir,
            "--output-dir",
            str(output_root / "eval_external"),
            "--batch-size",
            "8",
            "--img-size",
            str(args.external_img_size),
            "--eval-crops",
            str(args.external_eval_crops),
            "--amp",
        ],
        args.dry_run,
    )

    if not args.skip_error_analysis:
        run_step(
            "4.1 外部验证错误样本分析",
            [
                python,
                "export_errors.py",
                "--checkpoint",
                str(checkpoint),
                "--data-dir",
                args.external_dir,
                "--output",
                str(output_root / "error_analysis" / "external_val_errors.csv"),
                "--batch-size",
                "8",
                "--img-size",
                str(args.external_img_size),
                "--eval-crops",
                str(args.external_eval_crops),
                "--top-k-per-pair",
                "20",
            ],
            args.dry_run,
        )

    if not args.skip_explain:
        run_step(
            "5. Grad-CAM 可解释性图",
            [
                python,
                "grad_cam.py",
                "--checkpoint",
                str(checkpoint),
                "--data-dir",
                args.external_dir,
                "--output-dir",
                str(output_root / "grad_cam_val"),
                "--samples-per-class",
                str(args.samples_per_class),
                "--img-size",
                "224",
                "--target",
                "cnn_mamba",
            ],
            args.dry_run,
        )

        run_step(
            "6. 深层特征 PCA 图",
            [
                python,
                "feature_analysis.py",
                "--checkpoint",
                str(checkpoint),
                "--data-dir",
                args.external_dir,
                "--output-dir",
                str(output_root / "feature_analysis_val"),
                "--img-size",
                str(args.external_img_size),
                "--method",
                "pca",
            ],
            args.dry_run,
        )

        run_step(
            "6.1 深层特征 t-SNE 图",
            [
                python,
                "feature_analysis.py",
                "--checkpoint",
                str(checkpoint),
                "--data-dir",
                args.external_dir,
                "--output-dir",
                str(output_root / "feature_analysis_val"),
                "--img-size",
                str(args.external_img_size),
                "--method",
                "tsne",
            ],
            args.dry_run,
        )

    run_step(
        "7. 模型结构图",
        [python, "plot_model_diagram.py", "--output-dir", str(output_root / "model_diagram")],
        args.dry_run,
    )

    if not args.skip_profile:
        run_step(
            "8. 模型复杂度统计",
            [
                python,
                "model_profile.py",
                "--models",
                "cnn",
                "cnn_mamba",
                "cnn_mamba_unidirectional",
                "cnn_mamba_no_fusion",
                "cnn_mamba_simple_fusion",
                "mamba_only",
                "resnet18",
                "mobilenet_v3_small",
                "efficientnet_b0",
                "--output-dir",
                str(output_root / "model_profile"),
            ],
            args.dry_run,
        )

    if args.run_ablation:
        run_step(
            "9. 多模型多种子消融实验",
            [
                python,
                "run_experiments.py",
                "--epochs",
                str(args.ablation_epochs),
                "--batch-size",
                str(args.batch_size),
                "--seeds",
                *[str(seed) for seed in args.ablation_seeds],
                "--models",
                "cnn",
                "cnn_mamba",
                "cnn_mamba_unidirectional",
                "cnn_mamba_no_fusion",
                "cnn_mamba_simple_fusion",
                "mamba_only",
                "resnet18",
                "mobilenet_v3_small",
                "efficientnet_b0",
                "--augment",
                args.augment,
                "--mixup-alpha",
                str(args.mixup_alpha),
                "--cutmix-alpha",
                str(args.cutmix_alpha),
                "--output-root",
                str(output_root / "paper_experiments"),
            ],
            args.dry_run,
        )

    print(f"\n流水线完成。主要输出目录：{output_root}")


if __name__ == "__main__":
    main()
