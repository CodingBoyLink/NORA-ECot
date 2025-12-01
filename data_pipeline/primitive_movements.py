# Filename: data_pipeline/primitive_movements.py
"""
LIBERO 适配的 Primitive Movements 模块

基于 embodied-CoT-main/scripts/generate_embodied_data/primitive_movements.py 适配。
适配 LIBERO 的状态格式 (ee_pos, ee_ori, gripper_states)。

Requirements: 4.3, 4.4
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional


def describe_move(move_vec: np.ndarray) -> str:
    """
    将移动向量转换为自然语言描述。
    
    Args:
        move_vec: 7维移动向量 [x, y, z, roll, pitch, yaw, gripper]
                  每个维度的值为 -1, 0, 或 1
    
    Returns:
        str: 移动的自然语言描述
    """
    names = [
        {-1: "backward", 0: None, 1: "forward"},
        {-1: "right", 0: None, 1: "left"},
        {-1: "down", 0: None, 1: "up"},
        {-1: "tilt down", 0: None, 1: "tilt up"},      # roll
        {-1: "tilt right", 0: None, 1: "tilt left"},   # pitch
        {-1: "rotate clockwise", 0: None, 1: "rotate counterclockwise"},  # yaw
        {-1: "close gripper", 0: None, 1: "open gripper"},
    ]

    # 处理 xyz 移动
    xyz_move = [names[i].get(move_vec[i]) for i in range(3)]
    xyz_move = [m for m in xyz_move if m is not None]

    if len(xyz_move) != 0:
        description = "move " + " ".join(xyz_move)
    else:
        description = ""

    # 处理旋转 (roll/pitch 合并)
    # 如果 roll 为 0，使用 pitch 的值
    roll_pitch = move_vec[3] if move_vec[3] != 0 else move_vec[4]
    if roll_pitch != 0:
        if len(description) > 0:
            description = description + ", "
        description = description + names[3].get(roll_pitch, "")

    # 处理 yaw
    if move_vec[5] != 0:
        if len(description) > 0:
            description = description + ", "
        description = description + names[5].get(move_vec[5], "")

    # 处理 gripper
    if move_vec[6] != 0:
        if len(description) > 0:
            description = description + ", "
        description = description + names[6].get(move_vec[6], "")

    if len(description) == 0:
        description = "stop"

    return description


def classify_movement(move: np.ndarray, threshold: float = 0.03) -> Tuple[str, np.ndarray]:
    """
    根据状态变化分类移动类型。
    
    Args:
        move: 状态序列，形状为 (T, state_dim)，通常 T=4
        threshold: 判断移动方向的阈值
    
    Returns:
        Tuple[str, np.ndarray]: (移动描述, 移动向量)
    """
    diff = move[-1] - move[0]
    
    # 对 xyz 进行归一化，防止某个方向主导
    if np.sum(np.abs(diff[:3])) > 3 * threshold:
        diff[:3] *= 3 * threshold / np.sum(np.abs(diff[:3]))
    
    # 旋转角度的变化通常较小，需要缩放
    diff[3:6] /= 10
    
    # 将连续差值转换为离散方向 (-1, 0, 1)
    move_vec = 1 * (diff > threshold) - 1 * (diff < -threshold)
    
    return describe_move(move_vec), move_vec


def get_move_primitives_trajectory(trajectory: Dict[str, Any]) -> List[Tuple[str, np.ndarray]]:
    """
    为 LIBERO 轨迹生成移动原语序列。
    
    适配 LIBERO 的状态格式:
    - states: (N, 8) = [ee_pos(3), ee_ori(3), gripper_states(2)]
    - 或者从 actions 推断移动
    
    Args:
        trajectory: LIBERO 轨迹数据，包含 'states' 或 'actions' 键
    
    Returns:
        List[Tuple[str, np.ndarray]]: 每个时间步的 (移动描述, 移动向量)
    """
    # 优先使用 states（包含 ee_pos, ee_ori, gripper）
    if 'states' in trajectory:
        states = np.array(trajectory['states'])
        # states 格式: (N, 8) = [ee_pos(3), ee_ori(3), gripper_states(2)]
        # 我们需要 7 维: [ee_pos(3), ee_ori(3), gripper(1)]
        # 取 gripper_states 的第一个值作为 gripper 状态
        states_7d = np.concatenate([
            states[:, :6],  # ee_pos + ee_ori
            states[:, 6:7]  # gripper (取第一个)
        ], axis=-1)
    elif 'actions' in trajectory:
        # 如果没有 states，使用 actions 累积计算
        actions = np.array(trajectory['actions'])
        # 从初始状态累积 actions
        states_7d = np.cumsum(actions, axis=0)
    else:
        raise ValueError("Trajectory must contain 'states' or 'actions' key")
    
    # 使用滑动窗口计算移动原语
    # 每个时间步使用前后 4 帧来判断移动方向
    primitives = []
    window_size = 4
    
    for i in range(len(states_7d)):
        # 获取窗口范围
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(states_7d), i + window_size // 2 + 1)
        
        if end_idx - start_idx < 2:
            # 窗口太小，使用 stop
            primitives.append(("stop", np.zeros(7)))
        else:
            window = states_7d[start_idx:end_idx]
            move_desc, move_vec = classify_movement(window)
            primitives.append((move_desc, move_vec))
    
    return primitives


def get_move_primitives_from_states(
    states: np.ndarray,
    window_size: int = 4,
    threshold: float = 0.03
) -> List[Tuple[str, np.ndarray]]:
    """
    从状态序列生成移动原语。
    
    Args:
        states: 状态序列，形状为 (N, state_dim)
                LIBERO 格式: (N, 8) = [ee_pos(3), ee_ori(3), gripper_states(2)]
        window_size: 滑动窗口大小
        threshold: 移动判断阈值
    
    Returns:
        List[Tuple[str, np.ndarray]]: 每个时间步的 (移动描述, 移动向量)
    """
    # 转换为 7 维格式
    if states.shape[-1] == 8:
        # LIBERO 格式: [ee_pos(3), ee_ori(3), gripper_states(2)]
        states_7d = np.concatenate([
            states[:, :6],  # ee_pos + ee_ori
            states[:, 6:7]  # gripper (取第一个)
        ], axis=-1)
    elif states.shape[-1] == 7:
        states_7d = states
    else:
        raise ValueError(f"Unexpected state dimension: {states.shape[-1]}, expected 7 or 8")
    
    primitives = []
    
    for i in range(len(states_7d)):
        start_idx = max(0, i)
        end_idx = min(len(states_7d), i + window_size)
        
        if end_idx - start_idx < 2:
            primitives.append(("stop", np.zeros(7)))
        else:
            window = states_7d[start_idx:end_idx]
            move_desc, move_vec = classify_movement(window, threshold)
            primitives.append((move_desc, move_vec))
    
    return primitives


def get_state_3d_positions(trajectory: Dict[str, Any]) -> List[List[float]]:
    """
    从轨迹中提取 3D 末端执行器位置。
    
    Args:
        trajectory: LIBERO 轨迹数据
    
    Returns:
        List[List[float]]: 每个时间步的 [x, y, z] 位置
    """
    if 'states' in trajectory:
        states = np.array(trajectory['states'])
        # ee_pos 是前 3 维
        return states[:, :3].tolist()
    else:
        raise ValueError("Trajectory must contain 'states' key")


# 用于统计移动动作的全局字典
move_actions: Dict[str, List[np.ndarray]] = {}


def accumulate_move_actions(primitives: List[Tuple[str, np.ndarray]], actions: np.ndarray):
    """
    累积移动动作统计（用于分析）。
    
    Args:
        primitives: 移动原语列表
        actions: 对应的动作序列
    """
    global move_actions
    
    for (move, _), action in zip(primitives, actions):
        if move in move_actions:
            move_actions[move].append(action[:3])
        else:
            move_actions[move] = [action[:3]]
