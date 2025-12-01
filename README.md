# NORA-ECoT

基于 NORA 预训练模型的 LoRA 微调实验框架，在 LIBERO 机器人操作任务上进行三阶段实验。


## 项目概述

本项目实现了基于 NORA（Qwen2.5-VL-3B + FAST+）模型的 LoRA 微调方法，在 LIBERO 四个任务子集上进行三阶段实验：

- **Phase A (Baseline)**: 纯行为克隆，只预测动作
- **Phase B (Text CoT)**: 文本 ECoT + Reasoning Dropout
- **Phase C (Text + Flow CoT)**: 文本 CoT + RAFT 光流视觉 CoT + Reasoning Dropout

### 模型架构

共训练 12 个独立的 LoRA 模型：
- 4 个 LoRA_B（每个子集一个 Baseline）
- 4 个 LoRA_T（每个子集一个 Text CoT）
- 4 个 LoRA_TF（每个子集一个 Text+Flow CoT）

## 项目结构

```
NORA-ECoT/
├── configs/                    # 训练配置文件
│   ├── baseline.yaml          # Phase A 配置
│   ├── text_cot.yaml          # Phase B 配置
│   └── text_flow_cot.yaml     # Phase C 配置
├── data_pipeline/             # 数据处理模块
│   ├── raw_loader.py          # LIBERO HDF5 数据加载
│   ├── data_splitter.py       # 数据划分 (95% train / 5% test)
│   ├── ecot_annotator.py      # ECoT 标注生成
│   ├── primitive_movements.py # 运动原语提取
│   ├── gripper_positions.py   # 夹爪位置检测
│   ├── libero_utils.py        # LIBERO 环境工具函数
│   └── libero_regenerator.py  # LIBERO 数据重生成器
├── training/                  # 训练模块
│   ├── config.py              # 训练配置定义
│   ├── lora_model.py          # LoRA 模型加载
│   ├── lora_trainer.py        # LoRA 训练器
│   ├── action_tokenizer.py    # FAST+ 动作编码
│   └── datasets/              # 数据集定义
│       ├── libero_dataset.py  # LIBERO 数据集
│       ├── baseline_collator.py
│       └── ecot_collator.py
├── flow_pipeline/             # 光流处理模块
│   ├── raft_processor.py      # RAFT 光流计算
│   └── vq_encoder.py          # VQ 编码器
├── evaluation/                # 评估模块
│   └── libero_eval.py         # LIBERO 评估器
├── scripts/                   # 运行脚本
│   ├── run_experiment.py      # 主实验脚本（端到端）
│   ├── generate_ecot.py       # ECoT 标注生成
│   ├── preprocess_flow.py     # 光流预处理
│   ├── regenerate_libero.py   # LIBERO 数据重生成
│   ├── train_baseline.py      # Phase A 训练
│   ├── train_text_cot.py      # Phase B 训练
│   ├── train_text_flow_cot.py # Phase C 训练
│   └── evaluate.py            # 模型评估
├── data/                      # 数据目录
│   └── libero_object/         # LIBERO 数据集
└── outputs/                   # 输出目录
    ├── baseline/              # Baseline 模型
    ├── text_cot/              # Text CoT 模型
    ├── text_flow_cot/         # Text+Flow CoT 模型
    └── eval_results/          # 评估结果
```

## 环境配置

### 依赖安装

```bash
# 创建 conda 环境
conda create -n nora-ecot python=3.10
conda activate nora-ecot

# 安装 PyTorch (根据 CUDA 版本选择)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装核心依赖
pip install transformers accelerate peft
pip install qwen-vl-utils

# 安装 FAST+ tokenizer
pip install fast-tokenizer  # 或从 physical-intelligence/fast 安装

# 安装 LIBERO 环境
cd LIBERO
pip install -e .
cd ..

# 安装其他依赖
pip install h5py pyyaml tqdm wandb
pip install opencv-python pillow
pip install openai  # 用于 ECoT 生成
```

### 数据准备

1. 下载 LIBERO 数据集：
```bash
# 下载 LIBERO 数据到 data/ 目录
# 参考 LIBERO/README.md 获取下载链接
```

2. 数据目录结构：
```
data/
├── libero_spatial/
│   ├── task1.hdf5
│   └── ...
├── libero_object/
│   ├── task1.hdf5
│   └── ...
├── libero_goal/
│   └── ...
└── libero_long/
    └── ...
```

### 数据重生成（可选）

通过在 LIBERO 仿真环境中重放原始演示轨迹，可以生成更高质量的数据集：

- 提升图像分辨率（256x256 替代 128x128）
- 过滤 no-op（无操作）动作，减少冗余数据
- 过滤失败的演示轨迹，确保数据质量
- 生成元信息 JSON 文件，记录每个 episode 的成功状态

```bash
# 重生成 libero_object 数据集
python scripts/regenerate_libero.py \
    --libero_task_suite libero_object \
    --libero_raw_data_dir ./data/libero_object \
    --libero_target_dir ./data/libero_object_clean \
    --image_resolution 256

# 重生成 libero_spatial 数据集
python scripts/regenerate_libero.py \
    --libero_task_suite libero_spatial \
    --libero_raw_data_dir ./data/libero_spatial \
    --libero_target_dir ./data/libero_spatial_clean

# 重生成 libero_goal 数据集
python scripts/regenerate_libero.py \
    --libero_task_suite libero_goal \
    --libero_raw_data_dir ./data/libero_goal \
    --libero_target_dir ./data/libero_goal_clean

# 重生成 libero_10 (libero_long) 数据集
python scripts/regenerate_libero.py \
    --libero_task_suite libero_10 \
    --libero_raw_data_dir ./data/libero_10 \
    --libero_target_dir ./data/libero_10_clean
```

命令行参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--libero_task_suite` | 任务套件名称 (libero_spatial/libero_object/libero_goal/libero_10) | 必填 |
| `--libero_raw_data_dir` | 原始 HDF5 数据集目录 | 必填 |
| `--libero_target_dir` | 重生成数据输出目录 | 必填 |
| `--image_resolution` | 图像分辨率 | 256 |

重生成后会生成：
- 新的 HDF5 文件（与原始文件同名，保存在目标目录）
- `{task_suite}_metainfo.json` 元信息文件（记录每个 episode 的成功状态和初始状态）

## 快速开始

### 端到端实验

使用主实验脚本运行完整流程：

```bash
# 运行 Baseline 实验（object 子集）
python scripts/run_experiment.py --phase baseline --subset object

# 运行 Text CoT 实验（需要 LLM API）
python scripts/run_experiment.py --phase text_cot --subset object \
    --llm-api-key YOUR_API_KEY

# 运行 Text+Flow CoT 实验
python scripts/run_experiment.py --phase text_flow_cot --subset object \
    --llm-api-key YOUR_API_KEY

# 运行所有实验（12 个模型）
python scripts/run_experiment.py --all-phases --all-subsets \
    --llm-api-key YOUR_API_KEY
```

### 分步执行

#### 1. 生成 ECoT 标注（Phase B/C 需要）

```bash
python scripts/generate_ecot.py \
    --subset object \
    --api-key YOUR_OPENAI_API_KEY \
    --model gpt-4 \
    --output annotations/object_ecot.json
```

#### 2. 预处理光流（Phase C 需要）

```bash
python scripts/preprocess_flow.py \
    --subset object \
    --output flow_tokens/object_flow.json
```

#### 3. 训练模型

```bash
# Phase A: Baseline
python scripts/train_baseline.py \
    --subset object \
    --data_dir ./data \
    --output_dir ./outputs

# Phase B: Text CoT
python scripts/train_text_cot.py \
    --subset object \
    --ecot_annotations annotations/object_ecot.json \
    --reasoning_dropout_prob 0.5

# Phase C: Text + Flow CoT
python scripts/train_text_flow_cot.py \
    --subset object \
    --ecot_annotations annotations/object_ecot.json \
    --flow_tokens flow_tokens/object_flow.json
```

#### 4. 评估模型

```bash
python scripts/evaluate.py \
    --model_type baseline \
    --libero_subset object \
    --lora_path outputs/baseline/object/checkpoint-final \
    --num_trials 50
```

## 配置说明

### 训练配置 (configs/*.yaml)

```yaml
# LoRA 配置
lora:
  rank: 16          # LoRA 秩
  alpha: 32         # LoRA alpha
  dropout: 0.05     # LoRA dropout

# 优化器配置
optimizer:
  learning_rate: 5.0e-5
  warmup_steps: 1000

# Reasoning Dropout 配置 (Phase B/C)
reasoning_dropout:
  prob: 0.5              # No CoT 概率
  cot_loss_weight: 0.5   # CoT 辅助损失权重

# 训练配置
per_device_batch_size: 16
max_train_steps: 100000
mixed_precision: "bf16"
```

### 命令行参数

主要参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--phase` | 训练阶段 (baseline/text_cot/text_flow_cot) | - |
| `--subset` | LIBERO 子集 (spatial/object/goal/long) | - |
| `--data_dir` | 数据目录 | ./data |
| `--output_dir` | 输出目录 | ./outputs |
| `--lora_rank` | LoRA 秩 | 16 |
| `--learning_rate` | 学习率 | 5e-5 |
| `--max_train_steps` | 最大训练步数 | 100000 |
| `--reasoning_dropout_prob` | Reasoning Dropout 概率 | 0.5 |

## 实验结果

评估指标：任务成功率（50 trials/task）

| 模型 | Spatial | Object | Goal | Long |
|------|---------|--------|------|------|
| LoRA_B (Baseline) | - | - | - | - |
| LoRA_T (Text CoT) | - | - | - | - |
| LoRA_TF (Text+Flow CoT) | - | - | - | - |

## 参考项目

- [NORA](https://github.com/declare-lab/nora): 基础 VLA 模型
- [embodied-CoT](https://github.com/embodied-cot/embodied-cot): ECoT 标注方法
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO): 机器人操作基准
- [FAST+](https://github.com/physical-intelligence/fast): 动作离散化编码器

## License

MIT License
