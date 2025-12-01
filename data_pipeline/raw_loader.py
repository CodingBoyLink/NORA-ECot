# Filename: data_pipeline/raw_loader.py
"""
LIBERO 原始数据读取器

支持单文件和批量多文件加载。

Requirements: 1.1, 1.4, 1.5
"""

import os
import re
import h5py
import numpy as np
from typing import Dict, Any, List, Optional, Iterator, Tuple


def parse_task_instruction(filename: str) -> str:
    """
    从 LIBERO HDF5 文件名解析任务指令。
    
    LIBERO 文件命名规则:
    - 格式: {task_description}_demo.hdf5
    - 下划线分隔的单词组成任务描述
    
    Args:
        filename: HDF5 文件名（可以是完整路径或仅文件名）
    
    Returns:
        str: 解析后的任务指令文本
    
    Examples:
        >>> parse_task_instruction("pick_up_the_black_bowl_demo.hdf5")
        'pick up the black bowl'
        >>> parse_task_instruction("KITCHEN_SCENE1_open_the_microwave_demo.hdf5")
        'open the microwave'
    """
    # 提取文件名（去除路径）
    basename = os.path.basename(filename)
    
    # 移除 _demo.hdf5 或 .hdf5 后缀
    task_name = basename.replace('_demo.hdf5', '').replace('.hdf5', '')
    
    # 处理 LIBERO 特殊格式：SCENE_NAME_task_description
    # 例如: KITCHEN_SCENE1_open_the_microwave -> open the microwave
    # 检测是否有场景前缀（全大写单词）
    parts = task_name.split('_')
    
    # 找到第一个非全大写的单词作为任务描述的开始
    task_start_idx = 0
    for i, part in enumerate(parts):
        # 跳过全大写的场景标识符（如 KITCHEN, SCENE1, LIVING_ROOM 等）
        if part.isupper() or (part[:-1].isupper() and part[-1].isdigit() if part else False):
            task_start_idx = i + 1
        else:
            break
    
    # 提取任务描述部分
    task_parts = parts[task_start_idx:]
    task_description = ' '.join(task_parts)
    
    # 如果解析失败，回退到简单替换
    if not task_description:
        print(f"[Warning] 无法解析任务指令: {basename}")
        task_description = task_name.replace('_', ' ')
    
    return task_description


class LiberoRawLoader:
    """
    LIBERO 原始数据读取器（单文件）。
    
    Structure based on LIBERO HDF5 format:
    - obs/agentview_rgb: (N, H, W, 3) RGB images from agent view
    - obs/eye_in_hand_rgb: (N, H, W, 3) RGB images from gripper camera
    - obs/ee_pos: (N, 3) end-effector position
    - obs/ee_ori: (N, 3) end-effector orientation (euler angles)
    - obs/gripper_states: (N, 2) gripper state
    - actions: (N, 7) continuous actions
    """

    def __init__(self, dataset_path: str, verbose: bool = True):
        """
        初始化单文件加载器。
        
        Args:
            dataset_path: HDF5 文件路径
            verbose: 是否打印加载信息
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"数据集文件未找到: {dataset_path}")
        
        self.dataset_path = dataset_path
        self.verbose = verbose
        
        # 从文件名解析任务指令
        self.task_description = parse_task_instruction(dataset_path)
        
        with h5py.File(self.dataset_path, 'r') as f:
            self.demo_keys = list(f['data'].keys())
        
        if self.verbose:
            filename = os.path.basename(dataset_path)
            print(f"[LiberoRawLoader] 已加载: {filename}")
            print(f"[LiberoRawLoader] 解析任务指令: {self.task_description}")
            print(f"[LiberoRawLoader] 轨迹数量: {len(self.demo_keys)}")

    def get_trajectory(self, index: int) -> Dict[str, Any]:
        """
        读取单条轨迹数据。
        
        Args:
            index: 轨迹索引
        
        Returns:
            Dict containing:
                - actions: (N, 7) continuous actions
                - agentview: (N, H, W, 3) agent view images
                - eye_in_hand: (N, H, W, 3) gripper camera images
                - states: (N, 8) proprioceptive states [ee_pos, ee_ori, gripper]
                - images: dict with 'agentview' and 'eye_in_hand' keys
                - instruction: task instruction text
                - file_path: source file path
                - demo_key: demo key in HDF5 file
        """
        if index >= len(self.demo_keys):
            raise IndexError(f"Index {index} out of bounds for {len(self.demo_keys)} demos")

        demo_key = self.demo_keys[index]
        trajectory_data = {}
        
        with h5py.File(self.dataset_path, 'r') as f:
            demo_group = f['data'][demo_key]
            
            # 1. Action (N, 7)
            trajectory_data['actions'] = demo_group['actions'][:]
            
            # 2. Observations
            obs_group = demo_group['obs']
            
            # 根据 LIBERO 格式直接指定键名
            trajectory_data['agentview'] = obs_group['agentview_rgb'][:]
            trajectory_data['eye_in_hand'] = obs_group['eye_in_hand_rgb'][:]
            
            # 3. Proprioception (States)
            # 拼接 ee_pos (3), ee_ori (3), gripper_states (2) -> (N, 8)
            ee_pos = obs_group['ee_pos'][:]
            ee_ori = obs_group['ee_ori'][:]
            gripper = obs_group['gripper_states'][:]
            
            trajectory_data['states'] = np.concatenate([ee_pos, ee_ori, gripper], axis=-1)

            # 结构化输出
            trajectory_data['images'] = {
                'agentview': trajectory_data['agentview'],
                'eye_in_hand': trajectory_data['eye_in_hand']
            }
            trajectory_data['instruction'] = self.task_description
            trajectory_data['file_path'] = self.dataset_path
            trajectory_data['demo_key'] = demo_key
            
        return trajectory_data

    def __len__(self) -> int:
        """返回轨迹数量"""
        return len(self.demo_keys)
    
    @property
    def num_trajectories(self) -> int:
        """轨迹数量"""
        return len(self.demo_keys)


class LiberoBatchLoader:
    """
    LIBERO 批量数据加载器，支持加载多个 HDF5 文件。
    
    可以与 LiberoDataSplitter 配合使用，加载指定的轨迹子集。
    
    Requirements: 1.1, 1.4, 1.5
    """
    
    def __init__(
        self,
        file_paths: List[str],
        trajectory_indices: Optional[Dict[str, List[int]]] = None,
        verbose: bool = False
    ):
        """
        初始化批量加载器。
        
        Args:
            file_paths: HDF5 文件路径列表
            trajectory_indices: 可选，指定每个文件要加载的轨迹索引
                               格式: {file_path: [traj_idx1, traj_idx2, ...]}
                               如果为 None，则加载所有轨迹
            verbose: 是否打印加载信息
        """
        self.file_paths = file_paths
        self.trajectory_indices = trajectory_indices
        self.verbose = verbose
        
        # 验证文件存在
        for fp in file_paths:
            if not os.path.exists(fp):
                raise FileNotFoundError(f"文件未找到: {fp}")
        
        # 创建单文件加载器缓存
        self._loaders: Dict[str, LiberoRawLoader] = {}
        
        # 构建全局索引映射: global_idx -> (file_path, local_idx)
        self._index_map: List[Tuple[str, int]] = []
        self._build_index_map()
        
        if self.verbose:
            print(f"[LiberoBatchLoader] 已加载 {len(self.file_paths)} 个文件")
            print(f"[LiberoBatchLoader] 总轨迹数: {len(self._index_map)}")
    
    def _build_index_map(self):
        """构建全局索引到 (文件, 局部索引) 的映射"""
        self._index_map = []
        
        for file_path in self.file_paths:
            loader = self._get_loader(file_path)
            
            if self.trajectory_indices and file_path in self.trajectory_indices:
                # 使用指定的轨迹索引
                indices = self.trajectory_indices[file_path]
            else:
                # 使用所有轨迹
                indices = list(range(len(loader)))
            
            for local_idx in indices:
                self._index_map.append((file_path, local_idx))
    
    def _get_loader(self, file_path: str) -> LiberoRawLoader:
        """获取或创建单文件加载器"""
        if file_path not in self._loaders:
            self._loaders[file_path] = LiberoRawLoader(
                file_path, verbose=False
            )
        return self._loaders[file_path]
    
    def get_trajectory(self, index: int) -> Dict[str, Any]:
        """
        获取指定索引的轨迹。
        
        Args:
            index: 全局轨迹索引
        
        Returns:
            轨迹数据字典
        """
        if index < 0 or index >= len(self._index_map):
            raise IndexError(
                f"Index {index} out of bounds for {len(self._index_map)} trajectories"
            )
        
        file_path, local_idx = self._index_map[index]
        loader = self._get_loader(file_path)
        return loader.get_trajectory(local_idx)
    
    def __len__(self) -> int:
        """返回总轨迹数量"""
        return len(self._index_map)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """支持索引访问"""
        return self.get_trajectory(index)
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """支持迭代访问"""
        for i in range(len(self)):
            yield self.get_trajectory(i)
    
    def get_task_instructions(self) -> Dict[str, str]:
        """
        获取所有任务的指令文本。
        
        Returns:
            Dict[str, str]: {file_path: instruction}
        """
        instructions = {}
        for file_path in self.file_paths:
            loader = self._get_loader(file_path)
            instructions[file_path] = loader.task_description
        return instructions
    
    def get_trajectories_by_task(self, task_instruction: str) -> List[Dict[str, Any]]:
        """
        获取指定任务的所有轨迹。
        
        Args:
            task_instruction: 任务指令文本
        
        Returns:
            该任务的所有轨迹列表
        """
        trajectories = []
        for i, (file_path, local_idx) in enumerate(self._index_map):
            loader = self._get_loader(file_path)
            if loader.task_description == task_instruction:
                trajectories.append(self.get_trajectory(i))
        return trajectories
    
    @property
    def num_files(self) -> int:
        """文件数量"""
        return len(self.file_paths)
    
    @property
    def num_trajectories(self) -> int:
        """总轨迹数量"""
        return len(self._index_map)
    
    @property
    def num_tasks(self) -> int:
        """任务数量（等于文件数量）"""
        return len(self.file_paths)
    
    def get_file_trajectory_counts(self) -> Dict[str, int]:
        """
        获取每个文件的轨迹数量。
        
        Returns:
            Dict[str, int]: {file_path: num_trajectories}
        """
        counts = {}
        for file_path in self.file_paths:
            if self.trajectory_indices and file_path in self.trajectory_indices:
                counts[file_path] = len(self.trajectory_indices[file_path])
            else:
                loader = self._get_loader(file_path)
                counts[file_path] = len(loader)
        return counts
    
    @classmethod
    def from_splitter(
        cls,
        splitter: 'LiberoDataSplitter',
        split: str = 'train',
        verbose: bool = False
    ) -> 'LiberoBatchLoader':
        """
        从 LiberoDataSplitter 创建批量加载器。
        
        Args:
            splitter: 数据划分器实例
            split: 'train' 或 'test'
            verbose: 是否打印加载信息
        
        Returns:
            LiberoBatchLoader 实例
        """
        # 需要导入 data_splitter 模块
        from data_pipeline.data_splitter import LiberoDataSplitter, SplitInfo
        
        split_info = splitter.split()
        
        if split == 'train':
            file_paths = split_info.train_files
            trajectory_indices = split_info.train_trajectories
        elif split == 'test':
            file_paths = split_info.test_files
            trajectory_indices = split_info.test_trajectories
        else:
            raise ValueError(f"Invalid split '{split}'. Must be 'train' or 'test'")
        
        return cls(
            file_paths=file_paths,
            trajectory_indices=trajectory_indices,
            verbose=verbose
        )