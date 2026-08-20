from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from src.data import IMG_EXTENSIONS, read_labeled_image_samples
from src.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查训练集与验证/测试集之间的近重复图片。")
    parser.add_argument("--reference-dir", default="data", help="参考数据集，通常为训练集")
    parser.add_argument("--query-dirs", nargs="+", default=["test_data", "val_data"], help="待检查数据集")
    parser.add_argument("--classes", nargs="+", default=["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"])
    parser.add_argument("--hash-size", type=int, default=8)
    parser.add_argument("--threshold", type=int, default=3, help="感知哈希汉明距离阈值，越小越严格")
    parser.add_argument(
        "--exact-only",
        action="store_true",
        help="只检查精确重复，不做感知哈希近重复检查；默认只做文件 SHA-256，配合 --pixel-exact 可加做像素级精确重复",
    )
    parser.add_argument(
        "--pixel-exact",
        action="store_true",
        help="额外计算解码后 RGB 像素 SHA-256，可发现重编码后的同图像；高分辨率图片会明显变慢",
    )
    parser.add_argument(
        "--skip-pixel-exact",
        action="store_true",
        help="兼容旧参数：跳过解码像素级精确重复",
    )
    parser.add_argument("--within-dataset", action="store_true", help="同时检查每个待检查数据集内部的文件级重复")
    parser.add_argument("--export-duplicate-list", action="store_true", help="额外导出 JSON 格式的重复文件路径列表，供去重排除使用")
    parser.add_argument("--output-dir", default="outputs/data_audit")
    return parser.parse_args()


def iter_images(root: str | Path, class_names: list[str] | None = None) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    if class_names is None:
        return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS)

    samples, source = read_labeled_image_samples(root, class_names)
    if samples:
        print(f"数据读取：{source}")
        return [Path(path) for path, _ in samples]

    print(f"警告：{source}")
    return []


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decoded_pixel_signature(path: Path) -> tuple[str, tuple[int, int]]:
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"{image.width}x{image.height}|RGB|".encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest(), image.size


def average_hash(path: Path, hash_size: int) -> int:
    image = ImageOps.exif_transpose(Image.open(path)).convert("L").resize((hash_size, hash_size))
    pixels = np.asarray(image, dtype=np.float32).reshape(-1)
    avg = sum(pixels) / len(pixels)
    value = 0
    for pixel in pixels:
        value = (value << 1) | int(pixel >= avg)
    return value


def difference_hash(path: Path, hash_size: int) -> int:
    image = ImageOps.exif_transpose(Image.open(path)).convert("L").resize((hash_size + 1, hash_size))
    pixels = np.asarray(image, dtype=np.int16)
    diff = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def hamming_array(values: np.ndarray, query: int) -> np.ndarray:
    xor = np.bitwise_xor(values, np.uint64(query))
    return np.unpackbits(xor.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)


def make_row(
    query_dir: str | Path,
    query_path: Path,
    reference_path: Path,
    duplicate_type: str,
    evidence: str,
    threshold: int,
    file_sha_same: bool = False,
    pixel_exact_enabled: bool = False,
    pixel_sha_same: bool | str = "",
    average_hash_distance: int | str = "",
    difference_hash_distance: int | str = "",
) -> dict[str, str | int]:
    return {
        "待检查目录": str(query_dir),
        "待检查图片": str(query_path),
        "训练集相似图片": str(reference_path),
        "文件SHA256相同": int(file_sha_same),
        "启用像素级检查": int(pixel_exact_enabled),
        "解码像素SHA256相同": int(pixel_sha_same) if isinstance(pixel_sha_same, bool) else pixel_sha_same,
        "平均哈希距离": average_hash_distance,
        "差异哈希距离": difference_hash_distance,
        "重复类型": duplicate_type,
        "证据说明": evidence,
        "阈值": threshold,
    }


def main() -> None:
    args = parse_args()
    if args.hash_size > 8:
        raise ValueError("当前快速近重复检查要求 --hash-size 不超过 8。")
    output_dir = ensure_dir(args.output_dir)
    class_names = list(args.classes)
    reference_paths = iter_images(args.reference_dir, class_names)
    if not reference_paths:
        raise FileNotFoundError(f"参考目录没有图片：{args.reference_dir}")

    print(f"参考图片数：{len(reference_paths)}")
    reference_sha_map: dict[str, Path] = {}
    for path in reference_paths:
        reference_sha_map.setdefault(file_sha256(path), path)

    use_pixel_exact = args.pixel_exact and not args.skip_pixel_exact
    reference_pixel_map: dict[str, tuple[Path, tuple[int, int]]] = {}
    if use_pixel_exact:
        for path in reference_paths:
            try:
                pixel_sha, size = decoded_pixel_signature(path)
            except Exception as exc:
                print(f"警告：参考图片解码失败，已跳过像素级指纹：{path} | {exc}")
                continue
            reference_pixel_map.setdefault(pixel_sha, (path, size))

    reference_paths_array: list[Path] = []
    reference_ahashes = np.asarray([], dtype=np.uint64)
    reference_dhashes = np.asarray([], dtype=np.uint64)
    if not args.exact_only:
        reference_hashes = [
            (path, average_hash(path, args.hash_size), difference_hash(path, args.hash_size))
            for path in reference_paths
        ]
        reference_paths_array = [path for path, _, _ in reference_hashes]
        reference_ahashes = np.asarray([ahash for _, ahash, _ in reference_hashes], dtype=np.uint64)
        reference_dhashes = np.asarray([dhash for _, _, dhash in reference_hashes], dtype=np.uint64)
    rows: list[dict[str, str | int]] = []
    within_rows: list[dict[str, str | int]] = []
    summary: dict[str, object] = {
        "reference_dir": str(args.reference_dir),
        "reference_count": len(reference_paths),
        "query_counts": {},
        "duplicate_counts_by_type": {},
        "pixel_exact_enabled": use_pixel_exact,
        "perceptual_hash_enabled": not args.exact_only,
        "notes": [
            "文件级精确重复表示两个文件字节 SHA-256 完全一致。",
            "像素级精确重复需要启用 --pixel-exact，表示文件字节可以不同，但按 EXIF 方向解码为 RGB 后像素完全一致。",
            "高置信近重复需要不启用 --exact-only，表示文件和像素不完全一致，但平均哈希与差异哈希均不超过阈值。",
        ],
    }
    duplicate_counter: Counter[str] = Counter()

    for query_dir in args.query_dirs:
        query_paths = iter_images(query_dir, class_names)
        print(f"检查目录：{query_dir}，图片数：{len(query_paths)}")
        query_duplicate_counter: Counter[str] = Counter()
        query_sha_seen: dict[str, Path] = {}
        query_internal_duplicates = 0
        for query_path in query_paths:
            query_sha = file_sha256(query_path)
            if args.within_dataset:
                existing_query_path = query_sha_seen.get(query_sha)
                if existing_query_path is not None:
                    query_internal_duplicates += 1
                    within_rows.append(
                        {
                            "数据集目录": str(query_dir),
                            "重复图片": str(query_path),
                            "同数据集已出现图片": str(existing_query_path),
                            "重复类型": "文件级精确重复",
                            "证据说明": "同一数据集内两个文件的字节 SHA-256 完全相同。",
                        }
                    )
                else:
                    query_sha_seen[query_sha] = query_path
            exact_match = reference_sha_map.get(query_sha)
            if exact_match is not None:
                duplicate_type = "文件级精确重复"
                rows.append(
                    make_row(
                        query_dir=query_dir,
                        query_path=query_path,
                        reference_path=exact_match,
                        duplicate_type=duplicate_type,
                        evidence="文件字节 SHA-256 完全相同；不同下载源仍可能来自同一上游图片。",
                        threshold=args.threshold,
                        file_sha_same=True,
                        pixel_exact_enabled=use_pixel_exact,
                        pixel_sha_same=True if use_pixel_exact else "未检查；文件SHA256相同已足以判定精确重复",
                        average_hash_distance=0,
                        difference_hash_distance=0,
                    )
                )
                duplicate_counter[duplicate_type] += 1
                query_duplicate_counter[duplicate_type] += 1
                continue

            if use_pixel_exact:
                try:
                    query_pixel_sha, query_size = decoded_pixel_signature(query_path)
                except Exception as exc:
                    print(f"警告：待检查图片解码失败，已跳过像素级指纹：{query_path} | {exc}")
                    query_pixel_sha, query_size = "", (0, 0)
                pixel_match = reference_pixel_map.get(query_pixel_sha) if query_pixel_sha else None
                if pixel_match is not None:
                    pixel_match_path, reference_size = pixel_match
                    duplicate_type = "像素级精确重复"
                    rows.append(
                        make_row(
                            query_dir=query_dir,
                            query_path=query_path,
                            reference_path=pixel_match_path,
                            duplicate_type=duplicate_type,
                            evidence=(
                                "文件字节不同，但 EXIF 转正后 RGB 像素 SHA-256 完全相同；"
                                f"待检尺寸={query_size[0]}x{query_size[1]}，参考尺寸={reference_size[0]}x{reference_size[1]}。"
                            ),
                            threshold=args.threshold,
                            file_sha_same=False,
                            pixel_exact_enabled=True,
                            pixel_sha_same=True,
                            average_hash_distance=0,
                            difference_hash_distance=0,
                        )
                    )
                    duplicate_counter[duplicate_type] += 1
                    query_duplicate_counter[duplicate_type] += 1
                    continue

            if args.exact_only:
                continue

            query_ahash = average_hash(query_path, args.hash_size)
            query_dhash = difference_hash(query_path, args.hash_size)

            a_distances = hamming_array(reference_ahashes, query_ahash)
            d_distances = hamming_array(reference_dhashes, query_dhash)
            near_mask = (a_distances <= args.threshold) & (d_distances <= args.threshold)
            if np.any(near_mask):
                scores = a_distances + d_distances
                scores = np.where(near_mask, scores, np.iinfo(np.int64).max)
                best_idx = int(np.argmin(scores))
                best_path = reference_paths_array[best_idx]
                best_a_distance = int(a_distances[best_idx])
                best_d_distance = int(d_distances[best_idx])
                duplicate_type = "高置信近重复"
                rows.append(
                    make_row(
                        query_dir=query_dir,
                        query_path=query_path,
                        reference_path=best_path,
                        duplicate_type=duplicate_type,
                        evidence="文件和像素不完全一致，但两种感知哈希距离均低于阈值。",
                        threshold=args.threshold,
                        file_sha_same=False,
                        pixel_exact_enabled=use_pixel_exact,
                        pixel_sha_same=False if use_pixel_exact else "未检查",
                        average_hash_distance=best_a_distance,
                        difference_hash_distance=best_d_distance,
                    )
                )
                duplicate_counter[duplicate_type] += 1
                query_duplicate_counter[duplicate_type] += 1

        summary["query_counts"][str(query_dir)] = {
            "images": len(query_paths),
            "duplicates": sum(query_duplicate_counter.values()),
            "by_type": dict(query_duplicate_counter),
            "within_dataset_file_duplicates": query_internal_duplicates if args.within_dataset else "未检查",
        }

    output_path = Path(output_dir) / "possible_duplicates.csv"
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "待检查目录",
                "待检查图片",
                "训练集相似图片",
                "文件SHA256相同",
                "启用像素级检查",
                "解码像素SHA256相同",
                "平均哈希距离",
                "差异哈希距离",
                "重复类型",
                "证据说明",
                "阈值",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary["duplicate_counts_by_type"] = dict(duplicate_counter)
    summary["duplicates_total"] = len(rows)
    if args.within_dataset:
        within_path = Path(output_dir) / "within_dataset_duplicates.csv"
        with within_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["数据集目录", "重复图片", "同数据集已出现图片", "重复类型", "证据说明"],
            )
            writer.writeheader()
            writer.writerows(within_rows)
        summary["within_dataset_duplicates_total"] = len(within_rows)
        summary["within_dataset_duplicates_csv"] = str(within_path)
    summary_path = Path(output_dir) / "duplicates_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if args.export_duplicate_list:
        dup_list: dict[str, list[str]] = {}
        for row in rows:
            qdir = str(row["待检查目录"])
            qpath = str(row["待检查图片"])
            dup_list.setdefault(qdir, []).append(qpath)
        dup_list_path = Path(output_dir) / "duplicate_paths.json"
        with dup_list_path.open("w", encoding="utf-8") as f:
            json.dump(dup_list, f, ensure_ascii=False, indent=2)
        print(f"重复文件路径列表已导出到：{dup_list_path}")

    print(f"发现重复/近重复数量：{len(rows)}")
    if args.within_dataset:
        print(f"数据集内部文件级重复数量：{len(within_rows)}")
    print(f"检查结果已保存到：{output_path}")
    print(f"审计摘要已保存到：{summary_path}")


if __name__ == "__main__":
    main()
