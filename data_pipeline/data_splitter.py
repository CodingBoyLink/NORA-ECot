# Filename: data_pipeline/data_splitter.py
"""
LIBERO 数据划分器

实现 95/5 轨迹划分，确保每个任务都有覆盖。
使用固定随机种子保证可复现性。

Requirements: 1.2, 1.3, 1.6
"""

import os
import glob
import h5py
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SplitInfo:
    """数据划分信息"""
    train_files: List[str]
    test_files: List[str]
    train_trajectories: Dict[str, List[int]]  # {file_path: [traj_indices]}
    test_trajectories: Dict[str, List[int]]   # {file_path: [traj_indices]}
    task_distribution: Dict[str, Dict[str, int]]  # {task: {train: n, test: n}}


class LiberoDataSplitter:
    """
    按轨迹划分 95% train / 5% test，确保每个任务都有覆盖。
    
    支持 LIBERO 四个子集: spatial, object, goal, long
    """
    
    VALID_SUBSETS = ['spatial', 'object', 'goal', 'long']
    
    def __init__(
        self,
        data_dir: str,
        subset: str,
        train_ratio: float = 0.95,
        seed: int = 42
    ):
        """
        Args:
            data_dir: LIBERO 数据根目录
            subset: 'spatial' | 'object' | 'goal' | 'long'
            train_ratio: 训练集比例 (default: 0.95)
            seed: 随机种子 (default: 42)
        """
        if subset.lower() not in self.VALID_SUBSETS:
            raise ValueError(
                f"Invalid subset '{subset}'. Must be one of {self.VALID_SUBSETS}"
            )
        
        self.data_dir = data_dir
        self.subset = subset.lower()
        self.train_ratio = train_ratio
        self.seed = seed
        
        # 构建子集目录路径
        self.subset_dir = os.path.join(data_dir, f"libero_{self.subset}")
        if not os.path.exists(self.subset_dir):
            raise FileNotFoundError(
                f"Subset directory not found: {self.subset_dir}"
            )
        
        # 加载所有 HDF5 文件
        self.hdf5_files = self._discover_hdf5_files()
        if not self.hdf5_files:
            raise FileNotFoundError(
                f"No HDF5 files found in {self.subset_dir}"
            )
        
        # 缓存每个文件的轨迹数量
        self._trajectory_counts: Dict[str, int] = {}
        self._task_names: Dict[str, str] = {}
        self._load_file_metadata()
    
    def _discover_hdf5_files(self) -> List[str]:
        """发现所有 HDF5 文件"""
        pattern = os.path.join(self.subset_dir, "*.hdf5")
        files = glob.glob(pattern)
        return sorted(files)
    
    def _load_file_metadata(self):
        """加载每个文件的元数据（轨迹数量、任务名称）"""
        for file_path in self.hdf5_files:
            with h5py.File(file_path, 'r') as f:
                demo_keys = list(f['data'].keys())
                self._trajectory_counts[file_path] = len(demo_keys)
            
            # 从文件名解析任务名称
            filename = os.path.basename(file_path)
            task_name = filename.replace('_demo.hdf5', '').replace('.hdf5', '')
            self._task_names[file_path] = task_name

    def split(self) -> SplitInfo:
        """
        执行数据划分。
        
        策略：
        1. 对每个任务（HDF5文件）独立划分轨迹
        2. 确保每个任务在 train 和 test 中都有至少 1 条轨迹
        3. 使用固定随机种子保证可复现性
        
        Returns:
            SplitInfo: 包含划分结果的数据结构
        """
        np.random.seed(self.seed)
        
        train_trajectories: Dict[str, List[int]] = {}
        test_trajectories: Dict[str, List[int]] = {}
        task_distribution: Dict[str, Dict[str, int]] = {}
        
        for file_path in self.hdf5_files:
            task_name = self._task_names[file_path]
            num_trajectories = self._trajectory_counts[file_path]
            
            # 生成轨迹索引并打乱
            indices = np.arange(num_trajectories)
            np.random.shuffle(indices)
            
            # 计算划分点，确保 test 至少有 1 条
            num_train = max(1, int(num_trajectories * self.train_ratio))
            # 确保 test 至少有 1 条（如果总数 > 1）
            if num_trajectories > 1:
                num_train = min(num_train, num_trajectories - 1)
            
            train_indices = indices[:num_train].tolist()
            test_indices = indices[num_train:].tolist()
            
            # 如果只有 1 条轨迹，train 和 test 都使用它
            if num_trajectories == 1:
                train_indices = [0]
                test_indices = [0]
            
            train_trajectories[file_path] = sorted(train_indices)
            test_trajectories[file_path] = sorted(test_indices)
            
            task_distribution[task_name] = {
                'train': len(train_indices),
                'test': len(test_indices),
                'total': num_trajectories
            }
        
        # 所有文件都参与 train 和 test
        train_files = list(train_trajectories.keys())
        test_files = list(test_trajectories.keys())
        
        return SplitInfo(
            train_files=train_files,
            test_files=test_files,
            train_trajectories=train_trajectories,
            test_trajectories=test_trajectories,
            task_distribution=task_distribution
        )
    
    def get_task_distribution(self) -> Dict[str, Dict[str, int]]:
        """
        获取每个任务在 train/test 中的轨迹数量。
        
        Returns:
            Dict[str, Dict[str, int]]: {task_name: {train: n, test: n, total: n}}
        """
        split_info = self.split()
        return split_info.task_distribution
    
    def get_total_trajectories(self) -> Tuple[int, int]:
        """
        获取总轨迹数量。
        
        Returns:
            Tuple[int, int]: (train_count, test_count)
        """
        split_info = self.split()
        train_count = sum(len(v) for v in split_info.train_trajectories.values())
        test_count = sum(len(v) for v in split_info.test_trajectories.values())
        return train_count, test_count
    
    def print_summary(self):
        """打印划分摘要"""
        split_info = self.split()
        train_count, test_count = self.get_total_trajectories()
        
        print(f"\n{'='*60}")
        print(f"LIBERO-{self.subset.capitalize()} Data Split Summary")
        print(f"{'='*60}")
        print(f"Subset directory: {self.subset_dir}")
        print(f"Number of tasks: {len(self.hdf5_files)}")
        print(f"Train ratio: {self.train_ratio:.0%}")
        print(f"Random seed: {self.seed}")
        print(f"\nTotal trajectories: {train_count + test_count}")
        print(f"  - Train: {train_count} ({train_count/(train_count+test_count)*100:.1f}%)")
        print(f"  - Test: {test_count} ({test_count/(train_count+test_count)*100:.1f}%)")
        print(f"\nPer-task distribution:")
        print(f"{'-'*60}")
        
        for task_name, dist in split_info.task_distribution.items():
            short_name = task_name[:40] + "..." if len(task_name) > 40 else task_name
            print(f"  {short_name}")
            print(f"    Train: {dist['train']:3d} | Test: {dist['test']:3d} | Total: {dist['total']:3d}")
        
        print(f"{'='*60}\n")
    
    @property
    def num_tasks(self) -> int:
        """任务数量"""
        return len(self.hdf5_files)
    
    @property
    def total_trajectories(self) -> int:
        """总轨迹数量"""
        return sum(self._trajectory_counts.values())


def create_splits_for_all_subsets(
    data_dir: str,
    train_ratio: float = 0.95,
    seed: int = 42
) -> Dict[str, SplitInfo]:
    """
    为所有 LIBERO 子集创建数据划分。
    
    Args:
        data_dir: LIBERO 数据根目录
        train_ratio: 训练集比例
        seed: 随机种子
    
    Returns:
        Dict[str, SplitInfo]: {subset_name: SplitInfo}
    """
    splits = {}
    for subset in LiberoDataSplitter.VALID_SUBSETS:
        subset_dir = os.path.join(data_dir, f"libero_{subset}")
        if os.path.exists(subset_dir):
            splitter = LiberoDataSplitter(data_dir, subset, train_ratio, seed)
            splits[subset] = splitter.split()
    return splits


if __name__ == "__main__":
    # 示例用法
    import argparse
    
    parser = argparse.ArgumentParser(description="LIBERO Data Splitter")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="LIBERO data directory")
    parser.add_argument("--subset", type=str, default="object",
                        choices=LiberoDataSplitter.VALID_SUBSETS,
                        help="LIBERO subset to split")
    parser.add_argument("--train_ratio", type=float, default=0.95,
                        help="Training set ratio")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    splitter = LiberoDataSplitter(
        data_dir=args.data_dir,
        subset=args.subset,
        train_ratio=args.train_ratio,
        seed=args.seed
    )
    splitter.print_summary()
