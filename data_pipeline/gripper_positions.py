# Filename: data_pipeline/gripper_positions.py
"""
LIBERO 适配的 Gripper Positions 模块

基于 embodied-CoT-main/scripts/generate_embodied_data/gripper_positions.py 适配。
适配 LIBERO 的图像键名 (agentview_rgb)。

Requirements: 4.4, 4.5
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from PIL import Image

# 延迟导入，避免在没有 GPU 时报错
_detector = None
_sam_model = None
_sam_processor = None

# LIBERO 图像配置
LIBERO_IMAGE_DIMS = (256, 256)  # LIBERO 重生成后图像尺寸 (原始为 128x128)
LIBERO_IMAGE_KEY = "agentview_rgb"  # LIBERO 主视角图像键名


def _load_models():
    """延迟加载检测和分割模型"""
    global _detector, _sam_model, _sam_processor
    
    if _detector is None:
        from transformers import pipeline
        checkpoint = "google/owlvit-base-patch16"
        _detector = pipeline(model=checkpoint, task="zero-shot-object-detection")
    
    if _sam_model is None:
        from transformers import SamModel, SamProcessor
        _sam_model = SamModel.from_pretrained("facebook/sam-vit-base")
        _sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    
    return _detector, _sam_model, _sam_processor


def get_bounding_boxes(
    img: Image.Image,
    prompt: str = "the robotic gripper"
) -> List[Dict]:
    """
    使用 OWL-ViT 检测图像中的目标边界框。
    
    Args:
        img: PIL Image
        prompt: 检测提示词
    
    Returns:
        List[Dict]: 检测结果列表，每个包含 'box' 和 'score'
    """
    detector, _, _ = _load_models()
    predictions = detector(img, candidate_labels=[prompt], threshold=0.01)
    return predictions


def get_gripper_mask(img: Image.Image, pred: Dict) -> np.ndarray:
    """
    使用 SAM 获取 gripper 的分割掩码。
    
    Args:
        img: PIL Image
        pred: 检测结果，包含 'box' 键
    
    Returns:
        np.ndarray: 二值掩码
    """
    import torch
    
    _, sam_model, sam_processor = _load_models()
    
    box = [
        round(pred["box"]["xmin"], 2),
        round(pred["box"]["ymin"], 2),
        round(pred["box"]["xmax"], 2),
        round(pred["box"]["ymax"], 2),
    ]
    
    inputs = sam_processor(img, input_boxes=[[[box]]], return_tensors="pt")
    
    with torch.no_grad():
        outputs = sam_model(**inputs)
    
    mask = sam_processor.image_processor.post_process_masks(
        outputs.pred_masks,
        inputs["original_sizes"],
        inputs["reshaped_input_sizes"]
    )[0][0][0].numpy()
    
    return mask


def sq(w: int, h: int) -> np.ndarray:
    """生成坐标网格"""
    return np.concatenate([
        (np.arange(w * h).reshape(h, w) % w)[:, :, None],
        (np.arange(w * h).reshape(h, w) // w)[:, :, None]
    ], axis=-1)


def mask_to_pos_naive(mask: np.ndarray, image_dims: Tuple[int, int] = None) -> Tuple[int, int]:
    """
    从掩码中提取 gripper 位置（简单方法）。
    
    使用掩码中最靠近右下角的点作为 gripper 位置。
    
    Args:
        mask: 二值掩码
        image_dims: 图像尺寸 (width, height)
    
    Returns:
        Tuple[int, int]: (x, y) 位置
    """
    if image_dims is None:
        image_dims = mask.shape[:2][::-1]  # (width, height)
    
    pos = sq(*image_dims)
    weight = pos[:, :, 0] + pos[:, :, 1]
    min_pos = np.argmax((weight * mask).flatten())
    
    x = min_pos % image_dims[0] - (image_dims[0] / 16)
    y = min_pos // image_dims[0] - (image_dims[0] / 24)
    
    return int(x), int(y)


def get_gripper_pos_raw(img: np.ndarray) -> Tuple[Tuple[int, int], np.ndarray, Optional[Dict]]:
    """
    从单帧图像获取 gripper 位置。
    
    Args:
        img: RGB 图像数组
    
    Returns:
        Tuple: ((x, y), mask, prediction)
    """
    pil_img = Image.fromarray(img)
    predictions = get_bounding_boxes(pil_img)
    
    if len(predictions) > 0:
        mask = get_gripper_mask(pil_img, predictions[0])
        pos = mask_to_pos_naive(mask, (img.shape[1], img.shape[0]))
    else:
        mask = np.zeros((img.shape[0], img.shape[1]))
        pos = (-1, -1)
        predictions = [None]
    
    return (int(pos[0]), int(pos[1])), mask, predictions[0]


def process_trajectory_images(
    images: List[np.ndarray],
    states: Optional[np.ndarray] = None
) -> Optional[List[Tuple[Tuple[int, int], np.ndarray, Optional[Dict], Optional[np.ndarray]]]]:
    """
    处理轨迹中的所有图像，提取 gripper 位置。
    
    Args:
        images: 图像列表
        states: 可选的状态序列（用于后处理）
    
    Returns:
        处理后的轨迹数据，如果 gripper 从未被检测到则返回 None
    """
    raw_trajectory = []
    
    for i, img in enumerate(images):
        pos, mask, pred = get_gripper_pos_raw(img)
        state = states[i] if states is not None else None
        raw_trajectory.append((pos, mask, pred, state))
    
    # 处理未检测到的帧：使用最近的有效检测结果填充
    prev_found = list(range(len(raw_trajectory)))
    next_found = list(range(len(raw_trajectory)))
    
    prev_found[0] = -int(1e6)
    next_found[-1] = int(1e6)
    
    for i in range(1, len(raw_trajectory)):
        if raw_trajectory[i][2] is None:
            prev_found[i] = prev_found[i - 1]
    
    for i in reversed(range(0, len(raw_trajectory) - 1)):
        if raw_trajectory[i][2] is None:
            next_found[i] = next_found[i + 1]
    
    if next_found[0] == next_found[-1]:
        # gripper 从未被检测到
        return None
    
    # 用最近的有效检测替换未检测到的位置
    corrected_trajectory = []
    for i in range(len(raw_trajectory)):
        if raw_trajectory[i][2] is None:
            # 使用最近的有效检测
            nearest_idx = prev_found[i] if i - prev_found[i] < next_found[i] - i else next_found[i]
            if 0 <= nearest_idx < len(raw_trajectory):
                corrected_trajectory.append(raw_trajectory[nearest_idx])
            else:
                corrected_trajectory.append(raw_trajectory[i])
        else:
            corrected_trajectory.append(raw_trajectory[i])
    
    return corrected_trajectory


def get_gripper_positions_trajectory(trajectory: Dict[str, Any]) -> List[Tuple[int, int]]:
    """
    从 LIBERO 轨迹中提取 gripper 位置序列。
    
    Args:
        trajectory: LIBERO 轨迹数据，包含 'images' 或 'agentview' 键
    
    Returns:
        List[Tuple[int, int]]: 每帧的 gripper (x, y) 位置
    """
    # 获取图像序列
    if 'images' in trajectory and 'agentview' in trajectory['images']:
        images = trajectory['images']['agentview']
    elif 'agentview' in trajectory:
        images = trajectory['agentview']
    else:
        raise ValueError("Trajectory must contain 'images/agentview' or 'agentview' key")
    
    # 获取状态（可选）
    states = trajectory.get('states', None)
    
    # 处理轨迹
    processed = process_trajectory_images(images, states)
    
    if processed is None:
        # 如果 gripper 从未被检测到，返回默认位置
        return [(-1, -1)] * len(images)
    
    return [item[0] for item in processed]


def get_corrected_positions_with_regression(
    trajectory: Dict[str, Any],
    use_ransac: bool = True
) -> np.ndarray:
    """
    使用回归校正 gripper 位置（基于 3D 状态）。
    
    Args:
        trajectory: LIBERO 轨迹数据
        use_ransac: 是否使用 RANSAC 回归
    
    Returns:
        np.ndarray: 校正后的 2D 位置，形状为 (N, 2)
    """
    from sklearn.linear_model import RANSACRegressor, LinearRegression
    
    # 获取原始 gripper 位置
    gripper_positions = get_gripper_positions_trajectory(trajectory)
    
    # 获取 3D 状态
    if 'states' not in trajectory:
        return np.array(gripper_positions)
    
    states = np.array(trajectory['states'])
    points_3d = states[:, :3]  # ee_pos
    points_2d = np.array(gripper_positions, dtype=np.float32)
    
    # 过滤无效位置
    valid_mask = (points_2d[:, 0] >= 0) & (points_2d[:, 1] >= 0)
    
    if np.sum(valid_mask) < 4:
        # 有效点太少，无法回归
        return points_2d
    
    # 准备回归数据
    points_3d_pr = np.concatenate([points_3d, np.ones_like(points_3d[:, :1])], axis=-1)
    points_2d_pr = np.concatenate([points_2d, np.ones_like(points_2d[:, :1])], axis=-1)
    
    # 只使用有效点进行回归
    if use_ransac:
        reg = RANSACRegressor(random_state=0).fit(
            points_3d_pr[valid_mask],
            points_2d_pr[valid_mask]
        )
    else:
        reg = LinearRegression().fit(
            points_3d_pr[valid_mask],
            points_2d_pr[valid_mask]
        )
    
    # 预测所有位置
    predicted = reg.predict(points_3d_pr)[:, :-1].astype(int)
    
    return predicted


def format_gripper_position(pos: Tuple[int, int]) -> str:
    """
    将 gripper 位置格式化为字符串。
    
    Args:
        pos: (x, y) 位置
    
    Returns:
        str: 格式化的位置字符串
    """
    return f"[{pos[0]}, {pos[1]}]"


def format_gripper_positions_sequence(positions: List[Tuple[int, int]]) -> str:
    """
    将 gripper 位置序列格式化为字符串。
    
    Args:
        positions: 位置列表
    
    Returns:
        str: 格式化的位置序列字符串
    """
    return ", ".join([format_gripper_position(p) for p in positions])
