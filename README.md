# 玉米叶部病害分类：CNN + Vision Mamba S6 融合模型

## 项目简介

本项目实现了一个 CNN 与 Vision Mamba S6 融合的深度学习模型，用于玉米叶部病害四分类：北方叶枯病（Blight）、普通锈病（Common Rust）、灰叶斑病（Gray Leaf Spot）和健康（Healthy）。

模型用 CNN 提取局部病斑纹理特征，用 Vision Mamba S6 选择性状态空间模型进行全局序列建模，最后通过可学习门控融合两路特征完成分类。

---

## 环境准备

### 1. 创建 Python 虚拟环境

```bash
python -m venv mamba_cv
```

### 2. 激活虚拟环境

**Windows:**
```bash
mamba_cv\Scripts\activate
```

**macOS / Linux:**
```bash
source mamba_cv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

> 本项目使用 Python 3.10，依赖包括：PyTorch、torchvision、matplotlib、numpy、Pillow、scipy、scikit-learn、FastAPI、uvicorn、python-multipart。

### 4. 验证安装

```bash
python -c "import torch; print(f'PyTorch {torch.__version__} — CUDA: {torch.cuda.is_available()}')"
```

如果显示 `CUDA: True`，训练将使用 GPU 加速。没有 GPU 也可以运行，但训练速度会慢很多。

---

## 数据说明

### 数据集结构

```
data/                        # 数据集（PlantVillage 风格，实验室叶片图像）
├── Blight/                  # 北方叶枯病
├── Common_Rust/             # 普通锈病
├── Gray_Leaf_Spot/          # 灰叶斑病
└── Healthy/                 # 健康

val_data/                    # 外部验证集（田间实拍照片，与训练集分布不同）
├── Database.csv             # 标签文件
└── leaf_images/             # 2355 张田间叶片照片
```

### 类别对照

| 英文目录名 | 中文名称 |
|-----------|---------|
| Blight | 北方叶枯病 |
| Common_Rust | 普通锈病 |
| Gray_Leaf_Spot | 灰叶斑病 |
| Healthy | 健康 |

> **重要**：外部验证集（`val_data`）来自田间实拍，与训练集的 PlantVillage 实验室图像存在明显域差异。验证集**只用于泛化评估，不参与早停和模型选择**。模型选择依据内部验证集的宏平均 F1。

---

## 快速开始（一键运行）

如果只想快速得到完整实验结果，运行一条命令：

```bash
python run_pipeline.py
```

这会按顺序自动执行以下全部步骤：

| 步骤 | 内容 | 对应脚本 |
|------|------|---------|
| 1 | 数据重复审计 | `check_duplicates.py` |
| 2 | 训练集与验证集域差异分析 | `domain_audit.py` |
| 3 | 训练 CNN-Mamba S6 主模型 | `train.py` |
| 4 | 外部验证集独立评估 | `evaluate.py` |
| 5 | 错误样本导出 | `export_errors.py` |
| 6 | Grad-CAM 可解释性热力图 | `grad_cam.py` |
| 7 | 深层特征 PCA 可视化 | `feature_analysis.py` |
| 8 | 深层特征 t-SNE 可视化 | `feature_analysis.py` |
| 9 | 模型复杂度统计 | `model_profile.py` |

所有输出保存在 `outputs/paper_pipeline/`。

### 常用参数

```bash
# 显存不够时减小 batch size
python run_pipeline.py --batch-size 16

# 已完成训练，只重新生成分析图
python run_pipeline.py --skip-train --checkpoint outputs/cnn_mamba_internal_val/best.pt

# 同时运行消融实验（耗时较长）
python run_pipeline.py --run-ablation

# 预览会执行哪些命令，不实际运行
python run_pipeline.py --dry-run
```

---

## 分步操作

如果想逐步了解每个环节，可以按以下步骤手动执行。

### 步骤 1：数据审计

检查训练集与验证集之间是否有重复图片，避免数据泄漏。

```bash
# 精确重复检查（SHA-256），同时导出重复路径供后续训练排除
python check_duplicates.py \
    --reference-dir data \
    --query-dirs val_data \
    --output-dir outputs/data_audit \
    --exact-only \
    --within-dataset \
    --export-duplicate-list
```

输出文件：
- `outputs/data_audit/duplicates_summary.json` — 重复概况
- `outputs/data_audit/duplicate_paths.json` — 重复文件路径列表
- `outputs/data_audit/within_dataset_duplicates.csv` — 数据内部重复详情

### 步骤 2：域差异分析

分析训练集和外部验证集在图像特征上的分布差异，解释泛化难度的来源。

```bash
python domain_audit.py \
    --train-dir data \
    --external-dir val_data \
    --output-dir outputs/domain_audit
```

输出：亮度、对比度、饱和度、清晰度、尺寸、宽高比 7 个特征的分布对比图。

### 步骤 3：训练主模型

```bash
python train.py \
    --epochs 30 \
    --batch-size 32 \
    --model cnn_mamba \
    --output-dir outputs/cnn_mamba_s6 \
    --exclude-duplicates outputs/data_audit/duplicate_paths.json
```

训练过程说明：
- 自动从 `data` 分层划分 80% 训练 / 20% 内部验证
- 内部验证集用于保存最佳模型和早停（patience=8）
- `val_data` 作为外部验证集，每轮输出指标但不参与模型选择
- 默认启用：domain 跨域增强、标签平滑（0.05）、AMP 混合精度
- 外部验证使用 384×384 尺寸 + 5-crop 多裁剪评估

**常用可选参数：**

| 参数 | 作用 |
|------|------|
| `--batch-size 16` | 显存不足时减小 |
| `--augment standard` | 改用标准增强（默认 domain） |
| `--mixup-alpha 0.1 --cutmix-alpha 0.5` | 开启混合增强 |
| `--d-state 32` | 增大 S6 状态维度 |
| `--model resnet18 --pretrained` | 使用预训练 ResNet18 基线 |
| `--no-amp` | 关闭混合精度 |

### 步骤 4：外部验证集评估

```bash
python evaluate.py \
    --checkpoint outputs/cnn_mamba_s6/best.pt \
    --data-dir val_data \
    --output-dir outputs/eval_external \
    --batch-size 8 \
    --img-size 384 \
    --eval-crops 5 \
    --amp
```

### 步骤 5：错误样本分析

导出模型预测错误的样本，用于分析模型弱点。

```bash
python export_errors.py \
    --checkpoint outputs/cnn_mamba_s6/best.pt \
    --data-dir val_data \
    --output outputs/error_analysis/external_val_errors.csv \
    --batch-size 8 \
    --img-size 384 \
    --eval-crops 5 \
    --top-k-per-pair 20
```

输出：
- `external_val_errors.csv` — 逐样本错误详情
- `external_val_errors_summary.csv` — 错误类型汇总统计

### 步骤 6：可解释性分析

**Grad-CAM 热力图**（查看模型关注区域）：

```bash
python grad_cam.py \
    --checkpoint outputs/cnn_mamba_s6/best.pt \
    --data-dir val_data \
    --output-dir outputs/grad_cam_val \
    --samples-per-class 3 \
    --img-size 224 \
    --target cnn_mamba
```

> `--target` 可选值：`cnn_mamba`（双分支同时生成）、`fusion`（融合后精炼层）、`cnn_stem`（CNN 骨干末层）

**深层特征可视化**：

```bash
# PCA
python feature_analysis.py \
    --checkpoint outputs/cnn_mamba_s6/best.pt \
    --data-dir val_data \
    --output-dir outputs/feature_analysis \
    --method pca

# t-SNE
python feature_analysis.py \
    --checkpoint outputs/cnn_mamba_s6/best.pt \
    --data-dir val_data \
    --output-dir outputs/feature_analysis \
    --method tsne
```

### 步骤 7：模型复杂度统计

```bash
python model_profile.py \
    --models cnn cnn_mamba cnn_mamba_unidirectional cnn_mamba_no_fusion cnn_mamba_simple_fusion mamba_only resnet18 mobilenet_v3_small efficientnet_b0 \
    --output-dir outputs/model_profile
```

输出：各模型的参数量、FLOPs、推理时延对比表。

---

## 消融实验

消融实验通过多模型 × 多随机种子的批量训练，生成论文所需的对比表格和显著性检验。

### 完整消融实验

```bash
python run_experiments.py \
    --epochs 30 \
    --batch-size 32 \
    --seeds 42 2024 3407 \
    --models cnn cnn_mamba cnn_mamba_unidirectional cnn_mamba_no_fusion cnn_mamba_simple_fusion mamba_only \
    --augment domain \
    --mixup-alpha 0.1 \
    --cutmix-alpha 0.5
```

### 加入迁移学习基线

```bash
python run_experiments.py \
    --epochs 30 \
    --batch-size 32 \
    --seeds 42 2024 3407 \
    --models cnn cnn_mamba resnet18 mobilenet_v3_small efficientnet_b0 \
    --pretrained
```

### 输出文件

保存在 `outputs/paper_experiments/`：

| 文件 | 内容 |
|------|------|
| `summary_runs.csv` | 每次运行的完整指标 |
| `summary_mean_std.csv` | 各模型均值 ± 标准差 |
| `paper_table.csv` | 论文格式表格（中文） |
| `statistical_tests.csv` | 配对 t 检验结果 |
| `内部验证集_*_comparison.png/pdf` | 内部验证集对比图 |
| `外部验证集_*_comparison.png/pdf` | 外部验证集对比图 |
| `测试集_*_comparison.png/pdf` | 测试集对比图 |

### 模型变体说明

| 模型名 | 说明 | 用途 |
|--------|------|------|
| `cnn_mamba` | S6 双向扫描 + 门控融合 | **主模型** |
| `cnn_mamba_unidirectional` | S6 单向扫描 | 消融：双向扫描有效性 |
| `cnn_mamba_no_fusion` | 仅 Mamba 特征，无 CNN 融合 | 消融：CNN 局部特征贡献 |
| `cnn_mamba_simple_fusion` | concat + 1×1 卷积融合 | 消融：门控融合贡献 |
| `mamba_only` | 纯 Mamba，无 CNN stem | 消融：CNN 前端必要性 |
| `cnn` | 纯 CNN 基线 | 基准对比 |
| `resnet18` / `resnet50` | torchvision 迁移学习 | 经典基线 |
| `mobilenet_v3_small` / `mobilenet_v3_large` | 轻量级 CNN | 轻量基线 |
| `efficientnet_b0` | 高效 CNN | 高效基线 |

---

## 预测

### 命令行预测

```bash
# 单张图片
python predict.py --checkpoint outputs/cnn_mamba_s6/best.pt --input path/to/leaf.jpg

# 整个文件夹
python predict.py --checkpoint outputs/cnn_mamba_s6/best.pt --input path/to/images/

# 指定输出路径
python predict.py --checkpoint outputs/cnn_mamba_s6/best.pt --input path/to/leaf.jpg --output results.csv
```

> 不指定 `--checkpoint` 时，会自动查找 `outputs/cnn_mamba_internal_val/best.pt` 或 `outputs/paper_pipeline/cnn_mamba/best.pt`。

---

## 启动 Web 预测系统

### 1. 启动后端 API

```bash
python -m uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

指定模型权重：

```bash
# Windows
set CORN_CHECKPOINT=outputs/paper_pipeline/cnn_mamba/best.pt
python -m uvicorn api:app --reload

# macOS / Linux
CORN_CHECKPOINT=outputs/paper_pipeline/cnn_mamba/best.pt python -m uvicorn api:app --reload
```

### 2. 启动前端

```bash
cd frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

### 3. 使用

浏览器访问 `http://127.0.0.1:5173/`，上传玉米叶片图片即可获得预测结果。

接口列表：

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 服务状态和模型路径 |
| GET | `/metadata` | 模型配置、类别、设备信息 |
| POST | `/predict` | 上传图片（表单字段 `file`），返回预测结果和 Top-K 候选 |

> 如果后端地址不是默认的 `http://127.0.0.1:8000`，复制 `frontend/.env.example` 为 `frontend/.env`，修改 `VITE_API_BASE_URL`。

---

## 重要输出文件说明

训练完成后，模型输出目录（如 `outputs/cnn_mamba_s6/`）包含：

```
best.pt                                     # 最佳模型权重（按内部验证集 macro F1 选择）
last.pt                                     # 最后一轮权重
history.csv                                 # 每轮训练指标
training_curves.png / .pdf                  # 损失和指标曲线
dataset_distribution.png / .pdf             # 数据集类别分布
run_config.json                             # 运行配置记录

# 内部验证集（最佳模型时）
best_internal_val_metrics.json              # 完整指标
best_internal_val_confusion_matrix.png/pdf   # 混淆矩阵
best_internal_val_roc_curve.png/pdf         # ROC 曲线
best_internal_val_pr_curve.png/pdf          # PR 曲线
best_internal_val_calibration_curve.png/pdf # 校准曲线

# 外部验证集（最终评估）
external_val_final_metrics.json
external_val_final_confusion_matrix.png/pdf
external_val_final_roc_curve.png/pdf
external_val_final_pr_curve.png/pdf
external_val_final_calibration_curve.png/pdf
```

> 所有图表同时保存 PNG（300 dpi）和 PDF 矢量格式。标题、坐标轴、图例和类别名称均为中文。

---

## 常见问题

**Q: 训练时显存不足？**
```bash
python train.py --batch-size 16          # 减小 batch size
python train.py --batch-size 8 --no-amp  # 进一步减小 + 关闭混合精度
```

**Q: 图片中文显示为方框？**
程序会自动检测系统字体（微软雅黑、宋体、黑体等）。如果仍显示方框，需安装中文字体或指定字体路径。

**Q: 没有 GPU 可以运行吗？**
可以，但训练会很慢。所有脚本默认优先使用 GPU，没有 GPU 时自动退回 CPU。

**Q: 如何只评估已有模型，不重新训练？**
```bash
python run_pipeline.py --skip-train --checkpoint outputs/cnn_mamba_s6/best.pt
```

**Q: 测试集和训练集有重复怎么办？**
本项目已通过 `check_duplicates.py` 审计重复数据。训练时使用 `--exclude-duplicates` 参数可自动排除重复文件。建议从训练集分层划分独立测试集（`--test-ratio 0.15`），而非使用可能有重复的外部测试集。

---

## 论文图表建议顺序

```
图1  数据集类别分布与样本示例
图2  训练集与外部验证集图像域差异分析
图3  模型结构示意图
图4  内部验证集综合评价结果
图5  外部验证集综合评价结果与置信度校准曲线
图6  消融实验模型对比
图7  深层特征 PCA / t-SNE 分布
图8  Grad-CAM 可解释性分析
```

---

## 项目结构

```
├── train.py                  # 核心训练脚本
├── run_pipeline.py           # 一键运行完整流水线
├── run_experiments.py        # 消融实验批处理
├── check_duplicates.py       # 数据重复审计
├── domain_audit.py           # 域差异分析
├── evaluate.py               # 独立评估
├── export_errors.py          # 错误样本导出
├── grad_cam.py               # Grad-CAM 可解释性
├── feature_analysis.py       # 特征空间可视化（PCA/t-SNE）
├── model_profile.py          # 模型复杂度统计
├── predict.py                # 命令行预测工具
├── api.py                    # FastAPI 推理服务
├── requirements.txt          # Python 依赖
├── src/                      # 核心库
│   ├── models.py             #   模型定义
│   ├── data.py               #   数据加载与增强
│   ├── engine.py             #   训练与评估引擎
│   ├── metrics.py            #   指标计算与论文图表
│   ├── inference.py          #   推理封装
│   ├── checkpoint.py         #   权重加载
│   ├── utils.py              #   通用工具
│   └── settings.py           #   全局配置
├── frontend/                 # React 前端工作台
├── data/                     # 数据集
├── val_data/                 # 外部验证集
└── outputs/                  # 实验输出
    ├── paper_pipeline/       #   流水线产出
    └── paper_experiments/    #   消融实验产出
```
