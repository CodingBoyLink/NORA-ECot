# Flow pipeline module for RAFT optical flow processing and VQ encoding
"""
光流处理模块

包含:
- RAFTProcessor: RAFT 光流计算
- VQEncoder: VQ-VAE 光流编码
- 辅助函数: 光流可视化、token 转换等

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5
"""

from flow_pipeline.raft_processor import (
    RAFTProcessor,
    flow_to_rgb,
    flow_to_rgb_middlebury,
    compute_flow_batch
)

from flow_pipeline.vq_encoder import (
    VQEncoder,
    VQVAE,
    VectorQuantizer,
    tokens_to_string,
    string_to_tokens
)

__all__ = [
    # RAFT
    'RAFTProcessor',
    'flow_to_rgb',
    'flow_to_rgb_middlebury',
    'compute_flow_batch',
    # VQ
    'VQEncoder',
    'VQVAE',
    'VectorQuantizer',
    'tokens_to_string',
    'string_to_tokens',
]
