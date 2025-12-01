# Filename: flow_pipeline/raft_processor.py
"""
RAFT 光流处理器

使用 RAFT (Recurrent All-Pairs Field Transforms) 计算连续帧之间的光流，
并提供 RGB 可视化功能。

Requirements: 6.1, 6.2
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from pathlib import Path


def flow_to_rgb(
    flow: np.ndarray,
    max_flow: Optional[float] = None,
    clip_flow: Optional[float] = None
) -> np.ndarray:
    """
    将光流场转换为 RGB 可视化图像。
    
    使用 HSV 色彩空间编码：
    - Hue: 光流方向
    - Saturation: 固定为 1
    - Value: 光流幅度（归一化）
    
    Args:
        flow: (H, W, 2) 光流场，包含 (u, v) 分量
        max_flow: 用于归一化的最大光流幅度，如果为 None 则自动计算
        clip_flow: 裁剪光流幅度的阈值，如果为 None 则不裁剪
    
    Returns:
        (H, W, 3) RGB 图像，值范围 [0, 255]，dtype=uint8
    """
    import cv2
    
    u = flow[:, :, 0]
    v = flow[:, :, 1]
    
    # 计算光流幅度和方向
    magnitude = np.sqrt(u ** 2 + v ** 2)
    angle = np.arctan2(v, u)
    
    # 可选：裁剪光流幅度
    if clip_flow is not None:
        magnitude = np.clip(magnitude, 0, clip_flow)
    
    # 归一化幅度
    if max_flow is None:
        max_flow = np.max(magnitude) + 1e-8
    magnitude_normalized = np.clip(magnitude / max_flow, 0, 1)
    
    # 创建 HSV 图像
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.float32)
    hsv[:, :, 0] = (angle + np.pi) / (2 * np.pi) * 180  # Hue: 0-180 (OpenCV)
    hsv[:, :, 1] = 255  # Saturation: 固定为最大值
    hsv[:, :, 2] = magnitude_normalized * 255  # Value: 归一化幅度
    
    # 转换为 RGB
    hsv = hsv.astype(np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return rgb


def make_color_wheel() -> np.ndarray:
    """
    创建光流可视化的色轮。
    
    Returns:
        (ncols, 3) 色轮数组
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6
    
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    
    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col = col + RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col:col+YG, 1] = 255
    col = col + YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col = col + GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    colorwheel[col:col+CB, 2] = 255
    col = col + CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col = col + BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    colorwheel[col:col+MR, 0] = 255
    
    return colorwheel


def flow_to_rgb_middlebury(flow: np.ndarray, max_flow: Optional[float] = None) -> np.ndarray:
    """
    使用 Middlebury 色彩编码将光流转换为 RGB 图像。
    
    这是光流可视化的标准方法，与 RAFT 论文中使用的方法一致。
    
    Args:
        flow: (H, W, 2) 光流场
        max_flow: 用于归一化的最大光流幅度
    
    Returns:
        (H, W, 3) RGB 图像，值范围 [0, 255]，dtype=uint8
    """
    colorwheel = make_color_wheel()
    ncols = colorwheel.shape[0]
    
    u = flow[:, :, 0]
    v = flow[:, :, 1]
    
    # 计算幅度
    rad = np.sqrt(u ** 2 + v ** 2)
    
    if max_flow is None:
        max_flow = np.max(rad)
    
    # 归一化
    eps = np.finfo(float).eps
    u = u / (max_flow + eps)
    v = v / (max_flow + eps)
    
    rad = np.sqrt(u ** 2 + v ** 2)
    
    # 计算角度
    a = np.arctan2(-v, -u) / np.pi
    
    fk = (a + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    
    img = np.zeros((u.shape[0], u.shape[1], 3), dtype=np.uint8)
    
    for i in range(3):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1
        
        # 增加饱和度
        col = 1 - rad[:, :, np.newaxis] * (1 - col[:, :, np.newaxis])
        col = col.squeeze()
        
        img[:, :, i] = np.floor(255 * col).astype(np.uint8)
    
    return img


class RAFTProcessor:
    """
    RAFT 光流处理器。
    
    使用 torchvision 提供的 RAFT 预训练模型计算光流。
    支持多种预训练权重：raft_large, raft_small。
    
    Requirements: 6.1, 6.2
    """
    
    def __init__(
        self,
        model_name: str = "raft_large",
        device: Optional[str] = None,
        output_size: Tuple[int, int] = (64, 64)
    ):
        """
        初始化 RAFT 处理器。
        
        Args:
            model_name: 模型名称，支持 'raft_large' 或 'raft_small'
            device: 计算设备，如果为 None 则自动选择
            output_size: 输出光流图的尺寸 (H, W)
        """
        self.model_name = model_name
        self.output_size = output_size
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model = self._load_model()
        self.model.eval()
        
    def _load_model(self) -> torch.nn.Module:
        """加载 RAFT 预训练模型"""
        from torchvision.models.optical_flow import raft_large, raft_small
        from torchvision.models.optical_flow import Raft_Large_Weights, Raft_Small_Weights
        
        if self.model_name == "raft_large":
            weights = Raft_Large_Weights.DEFAULT
            model = raft_large(weights=weights)
        elif self.model_name == "raft_small":
            weights = Raft_Small_Weights.DEFAULT
            model = raft_small(weights=weights)
        else:
            raise ValueError(f"Unknown model name: {self.model_name}. "
                           f"Supported: 'raft_large', 'raft_small'")
        
        model = model.to(self.device)
        return model

    
    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        预处理图像用于 RAFT 模型。
        
        Args:
            image: (H, W, 3) RGB 图像，值范围 [0, 255]，dtype=uint8
        
        Returns:
            (1, 3, H, W) 预处理后的张量
        """
        # 转换为 float32 并归一化到 [0, 1]
        img = image.astype(np.float32) / 255.0
        
        # 转换为 (C, H, W) 格式
        img = np.transpose(img, (2, 0, 1))
        
        # 转换为 tensor 并添加 batch 维度
        img_tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)
        
        # RAFT 期望输入范围为 [0, 255]
        img_tensor = img_tensor * 255.0
        
        return img_tensor
    
    def compute_flow(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        num_iters: int = 12
    ) -> np.ndarray:
        """
        计算两帧之间的光流。
        
        Args:
            frame1: (H, W, 3) 第一帧 RGB 图像，值范围 [0, 255]
            frame2: (H, W, 3) 第二帧 RGB 图像，值范围 [0, 255]
            num_iters: RAFT 迭代次数，更多迭代通常更准确
        
        Returns:
            (H, W, 2) 光流场，包含 (u, v) 分量
        """
        # 预处理
        img1 = self._preprocess(frame1)
        img2 = self._preprocess(frame2)
        
        # 计算光流
        with torch.no_grad():
            # RAFT 返回一个光流预测列表，最后一个是最终结果
            flow_predictions = self.model(img1, img2)
            flow = flow_predictions[-1]  # (1, 2, H, W)
        
        # 转换为 numpy
        flow = flow.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, 2)
        
        return flow
    
    def compute_flow_rgb(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        num_iters: int = 12,
        resize: bool = True,
        use_middlebury: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算光流并转换为 RGB 可视化。
        
        Args:
            frame1: (H, W, 3) 第一帧 RGB 图像
            frame2: (H, W, 3) 第二帧 RGB 图像
            num_iters: RAFT 迭代次数
            resize: 是否将输出调整为 output_size
            use_middlebury: 是否使用 Middlebury 色彩编码
        
        Returns:
            Tuple of:
                - (H, W, 2) 或 (output_size[0], output_size[1], 2) 光流场
                - (H, W, 3) 或 (output_size[0], output_size[1], 3) RGB 可视化
        """
        import cv2
        
        # 计算光流
        flow = self.compute_flow(frame1, frame2, num_iters)
        
        # 转换为 RGB
        if use_middlebury:
            flow_rgb = flow_to_rgb_middlebury(flow)
        else:
            flow_rgb = flow_to_rgb(flow)
        
        # 调整尺寸
        if resize and self.output_size is not None:
            h, w = self.output_size
            
            # 调整光流场尺寸（需要缩放光流值）
            scale_h = h / flow.shape[0]
            scale_w = w / flow.shape[1]
            
            flow_resized = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
            flow_resized[:, :, 0] *= scale_w
            flow_resized[:, :, 1] *= scale_h
            
            # 调整 RGB 可视化尺寸
            flow_rgb_resized = cv2.resize(flow_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
            
            return flow_resized, flow_rgb_resized
        
        return flow, flow_rgb
    
    def process_trajectory(
        self,
        images: np.ndarray,
        num_iters: int = 12,
        resize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        处理整条轨迹的光流。
        
        Args:
            images: (N, H, W, 3) 轨迹图像序列
            num_iters: RAFT 迭代次数
            resize: 是否调整输出尺寸
        
        Returns:
            Tuple of:
                - (N-1, H', W', 2) 光流场序列
                - (N-1, H', W', 3) RGB 可视化序列
        """
        n_frames = len(images)
        flows = []
        flow_rgbs = []
        
        for i in range(n_frames - 1):
            flow, flow_rgb = self.compute_flow_rgb(
                images[i], images[i + 1],
                num_iters=num_iters,
                resize=resize
            )
            flows.append(flow)
            flow_rgbs.append(flow_rgb)
        
        return np.stack(flows), np.stack(flow_rgbs)
    
    def save_flow_visualization(
        self,
        flow_rgb: np.ndarray,
        output_path: str
    ):
        """
        保存光流 RGB 可视化图像。
        
        Args:
            flow_rgb: (H, W, 3) RGB 图像
            output_path: 输出文件路径
        """
        import cv2
        
        # OpenCV 使用 BGR 格式
        bgr = cv2.cvtColor(flow_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, bgr)


def compute_flow_batch(
    processor: RAFTProcessor,
    images: np.ndarray,
    batch_size: int = 8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    批量计算光流（用于大规模处理）。
    
    Args:
        processor: RAFTProcessor 实例
        images: (N, H, W, 3) 图像序列
        batch_size: 批处理大小
    
    Returns:
        Tuple of:
            - (N-1, H', W', 2) 光流场序列
            - (N-1, H', W', 3) RGB 可视化序列
    """
    n_frames = len(images)
    all_flows = []
    all_flow_rgbs = []
    
    for start_idx in range(0, n_frames - 1, batch_size):
        end_idx = min(start_idx + batch_size, n_frames - 1)
        
        for i in range(start_idx, end_idx):
            flow, flow_rgb = processor.compute_flow_rgb(
                images[i], images[i + 1]
            )
            all_flows.append(flow)
            all_flow_rgbs.append(flow_rgb)
    
    return np.stack(all_flows), np.stack(all_flow_rgbs)
