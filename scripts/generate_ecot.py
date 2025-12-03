#!/usr/bin/env python
# Filename: scripts/generate_ecot.py
"""
ECoT 标注生成脚本

支持命令行指定 LIBERO 子集和 LLM API 配置。

Requirements: 4.2

Usage:
    python scripts/generate_ecot.py --subset object --api-url https://api.openai.com/v1 --api-key YOUR_KEY
    python scripts/generate_ecot.py --subset spatial --model gpt-4 --output annotations/spatial_ecot.json
"""

import argparse
import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_pipeline.ecot_annotator import (
    ECoTAnnotator,
    LLMConfig,
    AnnotationConfig
)
from data_pipeline.data_splitter import LiberoDataSplitter
from data_pipeline.raw_loader import LiberoBatchLoader


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="为 LIBERO 数据集生成 ECoT 标注",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 使用 OpenAI API
    python scripts/generate_ecot.py --subset object --api-key sk-xxx
    
    # 使用自定义 API 端点
    python scripts/generate_ecot.py --subset spatial --api-url http://localhost:8000/v1 --api-key local
    
    # 指定输出路径和模型
    python scripts/generate_ecot.py --subset goal --model gpt-4-turbo --output ./my_annotations.json
"""
    )
    
    # 数据配置
    parser.add_argument(
        "--subset",
        type=str,
        required=True,
        choices=["spatial", "object", "goal", "10"],
        help="LIBERO 子集名称"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="LIBERO 数据目录 (默认: ./data)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test", "all"],
        help="数据划分 (默认: train)"
    )
    
    # LLM API 配置
    parser.add_argument(
        "--api-url",
        type=str,
        default="",
        help="LLM API URL (默认: OpenAI 官方 API)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="",
        help="LLM API Key (也可通过 OPENAI_API_KEY 环境变量设置)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4",
        help="LLM 模型名称 (默认: gpt-4)"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=8,
        help="API 调用最大重试次数 (默认: 8)"
    )
    
    # 输出配置
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出文件路径 (默认: annotations/{subset}_ecot.json)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="从已有结果继续 (默认: True)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="不从已有结果继续，重新生成"
    )
    
    # 标注配置
    parser.add_argument(
        "--use-gripper-detection",
        action="store_true",
        default=False,
        help="使用视觉检测 gripper 位置 (需要 GPU，默认: False)"
    )
    parser.add_argument(
        "--captions-path",
        type=str,
        default=None,
        help="场景描述文件路径 (可选)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="显示详细输出 (默认: True)"
    )
    parser.add_argument(
        "--quiet",
        action="store_false",
        dest="verbose",
        help="静默模式"
    )
    
    # 限制处理数量（用于测试）
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=None,
        help="最大处理轨迹数 (用于测试)"
    )
    
    return parser.parse_args()


def get_api_key(args) -> str:
    """获取 API Key"""
    if args.api_key:
        return args.api_key
    
    # 从环境变量获取
    env_key = os.environ.get("OPENAI_API_KEY", "")
    if env_key:
        return env_key
    
    raise ValueError(
        "请提供 API Key: 使用 --api-key 参数或设置 OPENAI_API_KEY 环境变量"
    )


def get_output_path(args) -> str:
    """获取输出路径"""
    if args.output:
        return args.output
    
    # 默认路径
    output_dir = Path("annotations")
    output_dir.mkdir(exist_ok=True)
    return str(output_dir / f"{args.subset}_ecot.json")


def load_trajectories(args) -> list:
    """加载轨迹数据"""
    print(f"加载 LIBERO-{args.subset} 数据...")
    
    # 使用数据划分器
    splitter = LiberoDataSplitter(
        data_dir=args.data_dir,
        subset=args.subset,
        seed=42
    )
    
    if args.split == "all":
        # 加载所有数据
        split_info = splitter.split()
        all_files = list(set(split_info.train_files + split_info.test_files))
        loader = LiberoBatchLoader(file_paths=all_files, verbose=args.verbose)
    else:
        # 加载指定划分
        loader = LiberoBatchLoader.from_splitter(
            splitter,
            split=args.split,
            verbose=args.verbose
        )
    
    print(f"已加载 {len(loader)} 条轨迹")
    
    # 转换为列表
    trajectories = []
    max_traj = args.max_trajectories or len(loader)
    
    for i in range(min(max_traj, len(loader))):
        trajectories.append(loader.get_trajectory(i))
    
    return trajectories


def main():
    """主函数"""
    args = parse_args()
    
    # 获取 API Key
    try:
        api_key = get_api_key(args)
    except ValueError as e:
        print(f"错误: {e}")
        sys.exit(1)
    
    # 配置 LLM
    llm_config = LLMConfig(
        api_url=args.api_url,
        api_key=api_key,
        model_name=args.model,
        max_retries=args.max_retries
    )
    
    # 配置标注
    annotation_config = AnnotationConfig(
        use_gripper_detection=args.use_gripper_detection,
        use_captions=args.captions_path is not None,
        captions_path=args.captions_path,
        save_intermediate=True,
        verbose=args.verbose
    )
    
    # 创建标注器
    annotator = ECoTAnnotator(
        llm_config=llm_config,
        annotation_config=annotation_config
    )
    
    # 加载轨迹
    trajectories = load_trajectories(args)
    
    if not trajectories:
        print("错误: 未找到轨迹数据")
        sys.exit(1)
    
    # 获取输出路径
    output_path = get_output_path(args)
    print(f"输出路径: {output_path}")
    
    # 生成标注
    print(f"\n开始生成 ECoT 标注...")
    print(f"  子集: {args.subset}")
    print(f"  模型: {args.model}")
    print(f"  轨迹数: {len(trajectories)}")
    print(f"  使用 gripper 检测: {args.use_gripper_detection}")
    print()
    
    annotations = annotator.annotate_batch(
        trajectories=trajectories,
        save_path=output_path,
        resume=args.resume
    )
    
    # 统计
    total_annotations = sum(len(v) for v in annotations.values())
    total_steps = sum(
        len(entry.get("reasoning", {}))
        for file_annotations in annotations.values()
        for entry in file_annotations.values()
    )
    
    print(f"\n标注完成!")
    print(f"  总轨迹数: {total_annotations}")
    print(f"  总步骤数: {total_steps}")
    print(f"  保存路径: {output_path}")


if __name__ == "__main__":
    main()
