# Filename: data_pipeline/raw_loader.py
import os
import h5py
import numpy as np
from typing import Dict, Any

class LiberoRawLoader:
    """
    LIBERO 原始数据读取器。
    
    Structure based on user log:
    - obs/agentview_rgb
    - obs/eye_in_hand_rgb
    - obs/ee_pos, obs/ee_ori, obs/gripper_states
    """

    def __init__(self, dataset_path: str):
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"数据集文件未找到: {dataset_path}")
        
        self.dataset_path = dataset_path
        
        # 从文件名解析任务指令
        # 例如: "pick_up_the_soup_..._demo.hdf5" -> "pick up the soup..."
        filename = os.path.basename(dataset_path)
        task_name = filename.replace('_demo.hdf5', '').replace('.hdf5', '')
        self.task_description = task_name.replace('_', ' ')
        
        with h5py.File(self.dataset_path, 'r') as f:
            self.demo_keys = list(f['data'].keys())
            
        print(f"[LiberoRawLoader] 已加载: {filename}")
        print(f"[LiberoRawLoader] 解析任务指令: {self.task_description}")
        print(f"[LiberoRawLoader] 轨迹数量: {len(self.demo_keys)}")

    def get_trajectory(self, index: int) -> Dict[str, Any]:
        """
        读取单条轨迹数据。
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
            
            # 根据你的 Log 直接指定键名
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
            
        return trajectory_data

    def __len__(self):
        return len(self.demo_keys)