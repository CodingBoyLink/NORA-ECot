#!/usr/bin/env python
"""
测试轨迹回放脚本 - 验证 libero_10_clean 中的轨迹能否正常达到目标

用法:
    # 测试第一个任务的第一条轨迹
    python scripts/test_trajectory_replay.py --data_dir ./data/libero_10_clean
    
    # 测试指定任务文件的指定轨迹
    python scripts/test_trajectory_replay.py --data_file ./data/libero_10_clean/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5 --demo_idx 0
    
    # 保存回放视频
    python scripts/test_trajectory_replay.py --data_dir ./data/libero_10_clean --save_video
"""

import argparse
import os
import sys
import glob

# 设置环境变量 - 必须在导入 mujoco/robosuite 之前
os.environ["MUJOCO_GL"] = "egl"  # 使用 EGL 渲染（无需显示器）
os.environ["PYOPENGL_PLATFORM"] = "egl"
# 如果 EGL 不工作，可以尝试 osmesa:
# os.environ["MUJOCO_GL"] = "osmesa"
# os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import h5py
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import robosuite.utils.transform_utils as T
    LIBERO_AVAILABLE = True
except ImportError:
    LIBERO_AVAILABLE = False
    print("警告: LIBERO 未安装，请先安装 LIBERO")


def get_libero_dummy_action():
    """获取空动作用于环境稳定"""
    return [0.0] * 7


def create_env_from_hdf5(hdf5_path: str, resolution: int = 256):
    """
    从 HDF5 文件创建对应的 LIBERO 环境
    
    Args:
        hdf5_path: HDF5 文件路径
        resolution: 图像分辨率
    
    Returns:
        env: LIBERO 环境实例
        task_description: 任务描述
    """
    # 从文件名解析任务信息
    basename = os.path.basename(hdf5_path)
    task_name = basename.replace('_demo.hdf5', '').replace('.hdf5', '')
    
    # 尝试匹配 LIBERO benchmark 中的任务
    benchmark_dict = benchmark.get_benchmark_dict()
    
    # 尝试不同的 task suite
    for suite_name in ['libero_10', 'libero_spatial', 'libero_object', 'libero_goal']:
        task_suite = benchmark_dict[suite_name]()
        
        for task_id in range(task_suite.n_tasks):
            task = task_suite.get_task(task_id)
            # 检查任务名是否匹配
            task_file_name = f"{task.problem_folder}_{task.bddl_file}".replace('.bddl', '')
            
            if task_name in task_file_name or task_file_name in task_name:
                # 找到匹配的任务
                task_bddl_file = os.path.join(
                    get_libero_path("bddl_files"),
                    task.problem_folder,
                    task.bddl_file
                )
                
                env_args = {
                    "bddl_file_name": task_bddl_file,
                    "camera_heights": resolution,
                    "camera_widths": resolution,
                }
                
                env = OffScreenRenderEnv(**env_args)
                env.seed(0)
                
                return env, task.language
    
    raise ValueError(f"无法找到匹配的任务: {task_name}")


def replay_trajectory(
    hdf5_path: str,
    demo_idx: int = 0,
    settle_steps: int = 10,
    save_video: bool = False,
    output_dir: str = "./replay_videos"
):
    """
    回放单条轨迹并检查是否达到目标
    
    Args:
        hdf5_path: HDF5 文件路径
        demo_idx: 要回放的 demo 索引
        settle_steps: 环境稳定步数
        save_video: 是否保存视频
        output_dir: 视频输出目录
    
    Returns:
        success: 是否成功达到目标
        info: 回放信息字典
    """
    print(f"\n{'='*60}")
    print(f"回放轨迹: {os.path.basename(hdf5_path)}")
    print(f"Demo 索引: {demo_idx}")
    print(f"{'='*60}")
    
    # 创建环境
    env, task_description = create_env_from_hdf5(hdf5_path)
    print(f"任务描述: {task_description}")
    
    # 加载轨迹数据
    with h5py.File(hdf5_path, 'r') as f:
        demo_keys = list(f['data'].keys())
        
        if demo_idx >= len(demo_keys):
            raise IndexError(f"Demo 索引 {demo_idx} 超出范围 (共 {len(demo_keys)} 条轨迹)")
        
        demo_key = demo_keys[demo_idx]
        demo_group = f['data'][demo_key]
        
        # 加载动作和状态
        actions = demo_group['actions'][:]
        
        # 尝试加载初始状态
        if 'states' in demo_group:
            states = demo_group['states'][:]
            initial_state = states[0]
        else:
            # 如果没有 states，尝试从 obs 构建
            print("警告: 未找到 states 数据，将使用默认初始状态")
            initial_state = None
    
    print(f"轨迹长度: {len(actions)} 步")
    
    # 重置环境
    env.reset()
    
    # 设置初始状态
    if initial_state is not None:
        env.set_init_state(initial_state)
    
    # 等待环境稳定
    dummy_action = get_libero_dummy_action()
    for _ in range(settle_steps):
        obs, reward, done, info = env.step(dummy_action)
    
    print(f"环境稳定完成 ({settle_steps} 步)")
    
    # 收集回放图像（用于视频）
    replay_images = []
    
    # 回放动作
    success = False
    for step_idx, action in enumerate(actions):
        # 收集图像
        if save_video:
            img = obs.get("agentview_image", obs.get("agentview_rgb"))
            if img is not None:
                replay_images.append(img)
        
        # 执行动作
        obs, reward, done, info = env.step(action.tolist())
        
        # 检查是否成功
        if done:
            success = True
            print(f"✓ 在第 {step_idx + 1} 步达到目标!")
            break
        
        # 打印进度
        if (step_idx + 1) % 50 == 0:
            print(f"  已执行 {step_idx + 1}/{len(actions)} 步...")
    
    # 最终状态
    if not success:
        print(f"✗ 执行完所有 {len(actions)} 步后未达到目标")
    
    # 保存视频
    if save_video and replay_images:
        try:
            import imageio
            os.makedirs(output_dir, exist_ok=True)
            
            status = "success" if success else "fail"
            video_name = f"{os.path.basename(hdf5_path).replace('.hdf5', '')}_{demo_idx}_{status}.mp4"
            video_path = os.path.join(output_dir, video_name)
            
            imageio.mimsave(video_path, replay_images, fps=30)
            print(f"视频已保存: {video_path}")
        except ImportError:
            print("警告: imageio 未安装，无法保存视频")
    
    # 清理
    env.close()
    
    return success, {
        "task_description": task_description,
        "demo_idx": demo_idx,
        "trajectory_length": len(actions),
        "success": success
    }


def test_all_trajectories(data_dir: str, max_demos_per_file: int = 1, save_video: bool = False):
    """
    测试目录中所有 HDF5 文件的轨迹
    
    Args:
        data_dir: 数据目录
        max_demos_per_file: 每个文件测试的最大 demo 数量
        save_video: 是否保存视频
    """
    hdf5_files = sorted(glob.glob(os.path.join(data_dir, "*.hdf5")))
    
    if not hdf5_files:
        print(f"未找到 HDF5 文件: {data_dir}")
        return
    
    print(f"\n找到 {len(hdf5_files)} 个 HDF5 文件")
    
    results = []
    total_success = 0
    total_tested = 0
    
    for hdf5_path in hdf5_files:
        # 获取文件中的 demo 数量
        with h5py.File(hdf5_path, 'r') as f:
            num_demos = len(f['data'].keys())
        
        demos_to_test = min(max_demos_per_file, num_demos)
        
        for demo_idx in range(demos_to_test):
            try:
                success, info = replay_trajectory(
                    hdf5_path, 
                    demo_idx=demo_idx,
                    save_video=save_video
                )
                results.append(info)
                total_tested += 1
                if success:
                    total_success += 1
            except Exception as e:
                print(f"错误: {e}")
                results.append({
                    "file": hdf5_path,
                    "demo_idx": demo_idx,
                    "error": str(e)
                })
    
    # 打印总结
    print(f"\n{'='*60}")
    print("测试总结")
    print(f"{'='*60}")
    print(f"测试轨迹数: {total_tested}")
    print(f"成功数: {total_success}")
    print(f"成功率: {total_success/total_tested*100:.1f}%" if total_tested > 0 else "N/A")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="测试 LIBERO 轨迹回放",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/libero_10_clean",
        help="数据目录路径"
    )
    parser.add_argument(
        "--data_file",
        type=str,
        help="单个 HDF5 文件路径（优先于 data_dir）"
    )
    parser.add_argument(
        "--demo_idx",
        type=int,
        default=0,
        help="要测试的 demo 索引"
    )
    parser.add_argument(
        "--max_demos",
        type=int,
        default=1,
        help="每个文件测试的最大 demo 数量"
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="保存回放视频"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./replay_videos",
        help="视频输出目录"
    )
    
    args = parser.parse_args()
    
    if not LIBERO_AVAILABLE:
        print("错误: LIBERO 未安装，请先安装 LIBERO")
        sys.exit(1)
    
    if args.data_file:
        # 测试单个文件
        success, info = replay_trajectory(
            args.data_file,
            demo_idx=args.demo_idx,
            save_video=args.save_video,
            output_dir=args.output_dir
        )
        sys.exit(0 if success else 1)
    else:
        # 测试目录中的所有文件
        results = test_all_trajectories(
            args.data_dir,
            max_demos_per_file=args.max_demos,
            save_video=args.save_video
        )


if __name__ == "__main__":
    main()
