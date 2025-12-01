#!/usr/bin/env python
# Filename: scripts/preprocess_flow.py
"""
光流预处理脚本

批量处理训练轨迹的光流，使用 RAFT 计算光流并用 VQ 编码为离散 token。
保存 flow_tokens 到文件供 Phase C 训练使用。

Requirements: 6.1, 6.5

Usage:
    python scripts/preprocess_flow.py --subset object --data-dir ./data
    python scripts/preprocess_flow.py --subset spatial --output ./flow_tokens/spatial.json
    python scripts/preprocess_flow.py --subset goal --vq-model ./models/vq_encoder.pt
"""

import argparse
import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="为 LIBERO 数据集预处理光流并编码为 token",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 处理 object 子集
    python scripts/preprocess_flow.py --subset object
    
    # 指定输出路径
    python scripts/preprocess_flow.py --subset spatial --output ./flow_tokens/spatial.json
    
    # 使用预训练的 VQ 编码器
    python scripts/preprocess_flow.py --subset goal --vq-model ./models/vq_encoder.pt
    
    # 只处理训练集
    python scripts/preprocess_flow.py --subset long --split train
"""
    )
    
    # 数据配置
    parser.add_argument(
        "--subset",
        type=str,
        required=True,
        choices=["spatial", "object", "goal", "long"],
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
    
    # RAFT 配置
    parser.add_argument(
        "--raft-model",
        type=str,
        default="raft_large",
        choices=["raft_large", "raft_small"],
        help="RAFT 模型 (默认: raft_large)"
    )
    parser.add_argument(
        "--raft-iters",
        type=int,
        default=12,
        help="RAFT 迭代次数 (默认: 12)"
    )
    
    # VQ 编码器配置
    parser.add_argument(
        "--vq-model",
        type=str,
        default=None,
        help="VQ 编码器权重路径 (可选，默认使用随机初始化)"
    )
    parser.add_argument(
        "--codebook-size",
        type=int,
        default=512,
        help="VQ 码本大小 (默认: 512)"
    )
    parser.add_argument(
        "--flow-size",
        type=int,
        default=64,
        help="光流图输出尺寸 (默认: 64)"
    )
    
    # 输出配置
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出文件路径 (默认: flow_tokens/{subset}_flow.json)"
    )
    parser.add_argument(
        "--save-flow-rgb",
        action="store_true",
        default=False,
        help="保存光流 RGB 可视化图像 (默认: False)"
    )
    parser.add_argument(
        "--flow-rgb-dir",
        type=str,
        default=None,
        help="光流 RGB 保存目录 (默认: flow_visualizations/{subset})"
    )
    
    # 处理配置
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="计算设备 (默认: 自动选择)"
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
        help="不从已有结果继续，重新处理"
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=None,
        help="最大处理轨迹数 (用于测试)"
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
    
    return parser.parse_args()


def get_output_path(args) -> str:
    """获取输出路径"""
    if args.output:
        return args.output
    
    # 默认路径
    output_dir = Path("flow_tokens")
    output_dir.mkdir(exist_ok=True)
    return str(output_dir / f"{args.subset}_flow.json")


def get_flow_rgb_dir(args) -> Optional[str]:
    """获取光流 RGB 保存目录"""
    if not args.save_flow_rgb:
        return None
    
    if args.flow_rgb_dir:
        return args.flow_rgb_dir
    
    # 默认路径
    flow_dir = Path("flow_visualizations") / args.subset
    flow_dir.mkdir(parents=True, exist_ok=True)
    return str(flow_dir)


def load_existing_results(output_path: str) -> Dict[str, Any]:
    """加载已有结果"""
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            return json.load(f)
    return {}


def save_results(results: Dict[str, Any], output_path: str):
    """保存结果"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)


def get_trajectory_key(trajectory: Dict[str, Any]) -> str:
    """生成轨迹的唯一标识"""
    file_path = trajectory.get('file_path', 'unknown')
    demo_key = trajectory.get('demo_key', 'unknown')
    return f"{os.path.basename(file_path)}:{demo_key}"



def process_trajectory(
    trajectory: Dict[str, Any],
    raft_processor,
    vq_encoder,
    raft_iters: int = 12,
    save_flow_rgb: bool = False,
    flow_rgb_dir: Optional[str] = None,
    traj_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    处理单条轨迹的光流。
    
    Args:
        trajectory: 轨迹数据
        raft_processor: RAFT 处理器
        vq_encoder: VQ 编码器
        raft_iters: RAFT 迭代次数
        save_flow_rgb: 是否保存光流 RGB
        flow_rgb_dir: 光流 RGB 保存目录
        traj_key: 轨迹标识
    
    Returns:
        Dict containing:
            - tokens: List[List[int]] 每帧的 flow tokens
            - num_frames: int 帧数
            - token_grid_size: Tuple[int, int]
            - num_tokens_per_frame: int
    """
    import numpy as np
    import cv2
    
    # 获取图像序列 (使用 agentview)
    images = trajectory['images']['agentview']  # (N, H, W, 3)
    
    all_tokens = []
    
    for i in range(len(images) - 1):
        # 计算光流
        flow, flow_rgb = raft_processor.compute_flow_rgb(
            images[i], images[i + 1],
            num_iters=raft_iters,
            resize=True
        )
        
        # VQ 编码
        tokens = vq_encoder.encode(flow_rgb)
        all_tokens.append(tokens)
        
        # 保存光流 RGB 可视化
        if save_flow_rgb and flow_rgb_dir and traj_key:
            safe_key = traj_key.replace(':', '_').replace('/', '_')
            rgb_path = os.path.join(flow_rgb_dir, f"{safe_key}_frame{i:04d}.png")
            cv2.imwrite(rgb_path, cv2.cvtColor(flow_rgb, cv2.COLOR_RGB2BGR))
    
    return {
        'tokens': all_tokens,
        'num_frames': len(images),
        'num_flow_frames': len(all_tokens),
        'token_grid_size': list(vq_encoder.token_grid_size),
        'num_tokens_per_frame': vq_encoder.num_tokens
    }


def load_trajectories(args):
    """加载轨迹数据"""
    from data_pipeline.data_splitter import LiberoDataSplitter
    from data_pipeline.raw_loader import LiberoBatchLoader
    
    if args.verbose:
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
        loader = LiberoBatchLoader(file_paths=all_files, verbose=False)
    else:
        # 加载指定划分
        loader = LiberoBatchLoader.from_splitter(
            splitter,
            split=args.split,
            verbose=False
        )
    
    if args.verbose:
        print(f"已加载 {len(loader)} 条轨迹")
    
    return loader


def main():
    """主函数"""
    args = parse_args()
    
    # 延迟导入（避免在帮助信息时加载 torch）
    from flow_pipeline.raft_processor import RAFTProcessor
    from flow_pipeline.vq_encoder import VQEncoder
    
    # 获取输出路径
    output_path = get_output_path(args)
    flow_rgb_dir = get_flow_rgb_dir(args)
    
    if args.verbose:
        print(f"\n{'='*60}")
        print(f"LIBERO 光流预处理")
        print(f"{'='*60}")
        print(f"子集: {args.subset}")
        print(f"数据划分: {args.split}")
        print(f"RAFT 模型: {args.raft_model}")
        print(f"VQ 码本大小: {args.codebook_size}")
        print(f"光流尺寸: {args.flow_size}x{args.flow_size}")
        print(f"输出路径: {output_path}")
        if args.save_flow_rgb:
            print(f"光流 RGB 目录: {flow_rgb_dir}")
        print(f"{'='*60}\n")
    
    # 加载已有结果
    results = {}
    if args.resume:
        results = load_existing_results(output_path)
        if results and args.verbose:
            print(f"已加载 {len(results)} 条已处理轨迹")
    
    # 初始化 RAFT 处理器
    if args.verbose:
        print("初始化 RAFT 处理器...")
    raft_processor = RAFTProcessor(
        model_name=args.raft_model,
        device=args.device,
        output_size=(args.flow_size, args.flow_size)
    )
    
    # 初始化 VQ 编码器
    if args.verbose:
        print("初始化 VQ 编码器...")
    vq_encoder = VQEncoder(
        codebook_size=args.codebook_size,
        input_size=(args.flow_size, args.flow_size),
        device=args.device,
        model_path=args.vq_model
    )
    
    # 加载轨迹
    loader = load_trajectories(args)
    
    # 确定处理数量
    max_traj = args.max_trajectories or len(loader)
    num_to_process = min(max_traj, len(loader))
    
    # 处理轨迹
    if args.verbose:
        print(f"\n开始处理 {num_to_process} 条轨迹...")
    
    processed_count = 0
    skipped_count = 0
    
    pbar = tqdm(range(num_to_process), desc="处理光流", disable=not args.verbose)
    
    for i in pbar:
        trajectory = loader.get_trajectory(i)
        traj_key = get_trajectory_key(trajectory)
        
        # 检查是否已处理
        if args.resume and traj_key in results:
            skipped_count += 1
            pbar.set_postfix({'processed': processed_count, 'skipped': skipped_count})
            continue
        
        try:
            # 处理轨迹
            flow_data = process_trajectory(
                trajectory=trajectory,
                raft_processor=raft_processor,
                vq_encoder=vq_encoder,
                raft_iters=args.raft_iters,
                save_flow_rgb=args.save_flow_rgb,
                flow_rgb_dir=flow_rgb_dir,
                traj_key=traj_key
            )
            
            # 添加元数据
            flow_data['file_path'] = trajectory['file_path']
            flow_data['demo_key'] = trajectory['demo_key']
            flow_data['instruction'] = trajectory['instruction']
            
            results[traj_key] = flow_data
            processed_count += 1
            
            # 定期保存
            if processed_count % 10 == 0:
                save_results(results, output_path)
            
            pbar.set_postfix({'processed': processed_count, 'skipped': skipped_count})
            
        except Exception as e:
            if args.verbose:
                print(f"\n警告: 处理轨迹 {traj_key} 失败: {e}")
            continue
    
    # 最终保存
    save_results(results, output_path)
    
    # 统计
    total_tokens = sum(
        len(r['tokens']) * r['num_tokens_per_frame']
        for r in results.values()
    )
    
    if args.verbose:
        print(f"\n{'='*60}")
        print(f"处理完成!")
        print(f"{'='*60}")
        print(f"总轨迹数: {len(results)}")
        print(f"新处理: {processed_count}")
        print(f"跳过: {skipped_count}")
        print(f"总 token 数: {total_tokens}")
        print(f"保存路径: {output_path}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
