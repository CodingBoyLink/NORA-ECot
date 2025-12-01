# Filename: flow_pipeline/vq_encoder.py
"""
VQ-VAE 编码器

将光流 RGB 图像编码为离散 token 序列，用于 Phase C 训练。
使用向量量化 (Vector Quantization) 将连续特征映射到离散码本。

Requirements: 6.3, 6.4, 6.5
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path


class VectorQuantizer(nn.Module):
    """
    向量量化层。
    
    将连续的特征向量映射到最近的码本向量。
    使用 EMA (Exponential Moving Average) 更新码本。
    """
    
    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5
    ):
        """
        初始化向量量化器。
        
        Args:
            num_embeddings: 码本大小（离散 token 数量）
            embedding_dim: 每个码本向量的维度
            commitment_cost: commitment loss 权重
            decay: EMA 衰减率
            epsilon: 数值稳定性常数
        """
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        
        # 码本嵌入
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
        
        # EMA 更新参数
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', self.embedding.weight.data.clone())
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播：量化输入特征。
        
        Args:
            z: (B, C, H, W) 输入特征图
        
        Returns:
            Tuple of:
                - quantized: (B, C, H, W) 量化后的特征
                - indices: (B, H, W) 码本索引
                - loss: 量化损失
        """
        # 重排为 (B, H, W, C)
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.embedding_dim)
        
        # 计算到所有码本向量的距离
        # d(z, e) = ||z||^2 + ||e||^2 - 2 * z @ e^T
        d = (
            torch.sum(z_flattened ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )
        
        # 找到最近的码本向量
        indices = torch.argmin(d, dim=1)
        
        # 获取量化后的向量
        quantized = self.embedding(indices).view(z.shape)
        
        # 计算损失
        if self.training:
            # EMA 更新码本
            self._ema_update(z_flattened, indices)
            
            # Commitment loss
            e_latent_loss = F.mse_loss(quantized.detach(), z)
            loss = self.commitment_cost * e_latent_loss
        else:
            loss = torch.tensor(0.0, device=z.device)
        
        # Straight-through estimator
        quantized = z + (quantized - z).detach()
        
        # 重排回 (B, C, H, W)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        indices = indices.view(z.shape[0], z.shape[1], z.shape[2])
        
        return quantized, indices, loss
    
    def _ema_update(self, z_flattened: torch.Tensor, indices: torch.Tensor):
        """EMA 更新码本"""
        # 计算每个码本向量被使用的次数
        encodings = F.one_hot(indices, self.num_embeddings).float()
        
        # 更新 cluster size
        self.ema_cluster_size.data.mul_(self.decay).add_(
            encodings.sum(0), alpha=1 - self.decay
        )
        
        # 更新码本向量
        dw = torch.matmul(encodings.t(), z_flattened)
        self.ema_w.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)
        
        # 归一化
        n = self.ema_cluster_size.sum()
        cluster_size = (
            (self.ema_cluster_size + self.epsilon)
            / (n + self.num_embeddings * self.epsilon) * n
        )
        self.embedding.weight.data.copy_(self.ema_w / cluster_size.unsqueeze(1))
    
    def encode(self, z: torch.Tensor) -> torch.Tensor:
        """
        仅编码，返回码本索引。
        
        Args:
            z: (B, C, H, W) 输入特征图
        
        Returns:
            (B, H, W) 码本索引
        """
        _, indices, _ = self.forward(z)
        return indices
    
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """
        从码本索引解码。
        
        Args:
            indices: (B, H, W) 码本索引
        
        Returns:
            (B, C, H, W) 解码后的特征
        """
        # 获取码本向量
        quantized = self.embedding(indices)  # (B, H, W, C)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        return quantized


class Encoder(nn.Module):
    """VQ-VAE 编码器网络"""
    
    def __init__(
        self,
        in_channels: int = 3,
        hidden_dims: List[int] = [32, 64, 128],
        latent_dim: int = 64
    ):
        super().__init__()
        
        layers = []
        prev_channels = in_channels
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Conv2d(prev_channels, h_dim, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.ReLU(inplace=True)
            ])
            prev_channels = h_dim
        
        # 最终投影到 latent_dim
        layers.append(nn.Conv2d(prev_channels, latent_dim, kernel_size=1))
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class Decoder(nn.Module):
    """VQ-VAE 解码器网络"""
    
    def __init__(
        self,
        out_channels: int = 3,
        hidden_dims: List[int] = [128, 64, 32],
        latent_dim: int = 64
    ):
        super().__init__()
        
        layers = []
        
        # 从 latent_dim 投影
        layers.append(nn.Conv2d(latent_dim, hidden_dims[0], kernel_size=1))
        
        for i in range(len(hidden_dims) - 1):
            layers.extend([
                nn.ConvTranspose2d(
                    hidden_dims[i], hidden_dims[i + 1],
                    kernel_size=4, stride=2, padding=1
                ),
                nn.BatchNorm2d(hidden_dims[i + 1]),
                nn.ReLU(inplace=True)
            ])
        
        # 最终输出层
        layers.append(
            nn.ConvTranspose2d(
                hidden_dims[-1], out_channels,
                kernel_size=4, stride=2, padding=1
            )
        )
        layers.append(nn.Sigmoid())  # 输出范围 [0, 1]
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)



class VQVAE(nn.Module):
    """
    VQ-VAE 模型。
    
    将输入图像编码为离散 token 序列。
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        hidden_dims: List[int] = [32, 64, 128],
        latent_dim: int = 64,
        num_embeddings: int = 512,
        commitment_cost: float = 0.25
    ):
        """
        初始化 VQ-VAE。
        
        Args:
            in_channels: 输入通道数
            hidden_dims: 隐藏层维度列表
            latent_dim: 潜在空间维度
            num_embeddings: 码本大小
            commitment_cost: commitment loss 权重
        """
        super().__init__()
        
        self.encoder = Encoder(in_channels, hidden_dims, latent_dim)
        self.vq = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)
        self.decoder = Decoder(in_channels, hidden_dims[::-1], latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播。
        
        Args:
            x: (B, C, H, W) 输入图像
        
        Returns:
            Tuple of:
                - recon: (B, C, H, W) 重建图像
                - indices: (B, H', W') 码本索引
                - vq_loss: 量化损失
        """
        z = self.encoder(x)
        quantized, indices, vq_loss = self.vq(z)
        recon = self.decoder(quantized)
        return recon, indices, vq_loss
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        编码为离散 token。
        
        Args:
            x: (B, C, H, W) 输入图像
        
        Returns:
            (B, H', W') 码本索引
        """
        z = self.encoder(x)
        indices = self.vq.encode(z)
        return indices
    
    def decode_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        从码本索引解码。
        
        Args:
            indices: (B, H', W') 码本索引
        
        Returns:
            (B, C, H, W) 重建图像
        """
        quantized = self.vq.decode(indices)
        recon = self.decoder(quantized)
        return recon


class VQEncoder:
    """
    VQ 编码器封装类。
    
    将 64x64 光流 RGB 图像编码为约 64 个离散 token (8x8 grid)。
    
    Requirements: 6.3, 6.4, 6.5
    """
    
    def __init__(
        self,
        codebook_size: int = 512,
        latent_dim: int = 64,
        input_size: Tuple[int, int] = (64, 64),
        device: Optional[str] = None,
        model_path: Optional[str] = None
    ):
        """
        初始化 VQ 编码器。
        
        Args:
            codebook_size: 码本大小（离散 token 数量）
            latent_dim: 潜在空间维度
            input_size: 输入图像尺寸 (H, W)
            device: 计算设备
            model_path: 预训练模型路径，如果为 None 则使用随机初始化
        """
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim
        self.input_size = input_size
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # 计算输出 token grid 尺寸
        # 3 层下采样，每层 stride=2，所以 64 -> 32 -> 16 -> 8
        self.token_grid_size = (input_size[0] // 8, input_size[1] // 8)
        self.num_tokens = self.token_grid_size[0] * self.token_grid_size[1]
        
        # 创建模型
        self.model = VQVAE(
            in_channels=3,
            hidden_dims=[32, 64, 128],
            latent_dim=latent_dim,
            num_embeddings=codebook_size
        ).to(self.device)
        
        # 加载预训练权重
        if model_path is not None:
            self.load_weights(model_path)
        
        self.model.eval()
    
    def load_weights(self, model_path: str):
        """加载预训练权重"""
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        print(f"[VQEncoder] 已加载权重: {model_path}")
    
    def save_weights(self, model_path: str):
        """保存模型权重"""
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), model_path)
        print(f"[VQEncoder] 已保存权重: {model_path}")
    
    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        预处理图像。
        
        Args:
            image: (H, W, 3) RGB 图像，值范围 [0, 255]
        
        Returns:
            (1, 3, H, W) 预处理后的张量
        """
        import cv2
        
        # 调整尺寸
        if image.shape[:2] != self.input_size:
            image = cv2.resize(image, (self.input_size[1], self.input_size[0]))
        
        # 归一化到 [0, 1]
        img = image.astype(np.float32) / 255.0
        
        # 转换为 (C, H, W)
        img = np.transpose(img, (2, 0, 1))
        
        # 转换为 tensor
        img_tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)
        
        return img_tensor
    
    def encode(self, flow_rgb: np.ndarray) -> List[int]:
        """
        将光流 RGB 图像编码为离散 token 序列。
        
        Args:
            flow_rgb: (H, W, 3) 光流 RGB 图像，值范围 [0, 255]
        
        Returns:
            List[int]: 约 64 个 token (8x8 grid 展平)
        """
        # 预处理
        x = self._preprocess(flow_rgb)
        
        # 编码
        with torch.no_grad():
            indices = self.model.encode(x)  # (1, H', W')
        
        # 展平为 token 列表
        tokens = indices.squeeze(0).flatten().cpu().numpy().tolist()
        
        return tokens
    
    def decode(self, tokens: List[int]) -> np.ndarray:
        """
        将 token 序列解码为光流 RGB 图像。
        
        Args:
            tokens: List[int] token 序列
        
        Returns:
            (H, W, 3) 重建的光流 RGB 图像，值范围 [0, 255]
        """
        # 重塑为 grid
        indices = torch.tensor(tokens, device=self.device)
        indices = indices.view(1, *self.token_grid_size)
        
        # 解码
        with torch.no_grad():
            recon = self.model.decode_from_indices(indices)  # (1, 3, H, W)
        
        # 转换为 numpy
        recon = recon.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        recon = (recon * 255).clip(0, 255).astype(np.uint8)
        
        return recon
    
    def encode_batch(self, flow_rgbs: np.ndarray) -> List[List[int]]:
        """
        批量编码光流 RGB 图像。
        
        Args:
            flow_rgbs: (N, H, W, 3) 光流 RGB 图像序列
        
        Returns:
            List[List[int]]: N 个 token 序列
        """
        all_tokens = []
        for flow_rgb in flow_rgbs:
            tokens = self.encode(flow_rgb)
            all_tokens.append(tokens)
        return all_tokens
    
    def encode_trajectory(
        self,
        flow_rgbs: np.ndarray
    ) -> Dict[str, Any]:
        """
        编码整条轨迹的光流。
        
        Args:
            flow_rgbs: (N, H, W, 3) 光流 RGB 图像序列
        
        Returns:
            Dict containing:
                - tokens: List[List[int]] token 序列
                - num_frames: int 帧数
                - token_grid_size: Tuple[int, int] token grid 尺寸
                - num_tokens_per_frame: int 每帧 token 数量
        """
        tokens = self.encode_batch(flow_rgbs)
        
        return {
            'tokens': tokens,
            'num_frames': len(flow_rgbs),
            'token_grid_size': self.token_grid_size,
            'num_tokens_per_frame': self.num_tokens
        }
    
    def train_step(
        self,
        images: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        训练一步。
        
        Args:
            images: (B, 3, H, W) 输入图像批次
            optimizer: 优化器
        
        Returns:
            Dict containing loss values
        """
        self.model.train()
        
        optimizer.zero_grad()
        
        recon, indices, vq_loss = self.model(images)
        
        # 重建损失
        recon_loss = F.mse_loss(recon, images)
        
        # 总损失
        loss = recon_loss + vq_loss
        
        loss.backward()
        optimizer.step()
        
        self.model.eval()
        
        return {
            'loss': loss.item(),
            'recon_loss': recon_loss.item(),
            'vq_loss': vq_loss.item()
        }


def tokens_to_string(tokens: List[int], prefix: str = "flow") -> str:
    """
    将 token 列表转换为字符串格式。
    
    Args:
        tokens: token 列表
        prefix: token 前缀
    
    Returns:
        格式化的 token 字符串，如 "<flow_0><flow_1>..."
    """
    return ''.join([f"<{prefix}_{t}>" for t in tokens])


def string_to_tokens(token_string: str, prefix: str = "flow") -> List[int]:
    """
    将 token 字符串解析为 token 列表。
    
    Args:
        token_string: 格式化的 token 字符串
        prefix: token 前缀
    
    Returns:
        token 列表
    """
    import re
    pattern = f"<{prefix}_(\d+)>"
    matches = re.findall(pattern, token_string)
    return [int(m) for m in matches]
