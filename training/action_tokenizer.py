"""
Action Tokenizer module for NORA LoRA training.
Wraps the FAST+ tokenizer from physical-intelligence/fast for action encoding/decoding.

Requirements covered:
- 2.1: Use FAST+ tokenizer from physical-intelligence/fast consistent with NORA
- 2.2: Convert 7-DoF continuous actions to discrete action_tokens using FAST+ encoding
- 2.3: Map FAST+ token indices to VLM action format as <robot_action_N> strings
- 2.5: Handle gripper action normalization: invert from [-1,1] to [0,1] for LIBERO
- 2.6: Use action token ID range 151665-153712 consistent with NORA
"""
from typing import List, Optional, Tuple, Union
import numpy as np
from transformers import AutoProcessor


# Action token ID range consistent with NORA (Req 2.6)
ACTION_TOKEN_MIN = 151665
ACTION_TOKEN_MAX = 153712
ACTION_TOKEN_VOCAB_SIZE = ACTION_TOKEN_MAX - ACTION_TOKEN_MIN + 1  # 2048 tokens


class ActionTokenizer:
    """
    Wrapper for FAST+ tokenizer that handles action encoding/decoding.
    
    This class provides:
    - Encoding continuous 7-DoF actions to discrete tokens
    - Decoding tokens back to continuous actions
    - Formatting tokens as <robot_action_N> strings for VLM
    - Gripper action normalization for LIBERO compatibility
    
    Attributes:
        fast_tokenizer: The underlying FAST+ tokenizer from physical-intelligence/fast
        action_dim: Number of action dimensions (default 7 for LIBERO)
        time_horizon: Number of timesteps to predict (default 1)
    """
    
    def __init__(
        self,
        tokenizer_path: str = "physical-intelligence/fast",
        action_dim: int = 7,
        time_horizon: int = 1,
    ):
        """
        Initialize the ActionTokenizer.
        
        Args:
            tokenizer_path: HuggingFace path to FAST+ tokenizer
            action_dim: Number of action dimensions (default 7 for 6-DoF + gripper)
            time_horizon: Number of timesteps to predict per inference
        """
        # Load FAST+ tokenizer (Req 2.1)
        self.fast_tokenizer = AutoProcessor.from_pretrained(
            tokenizer_path, trust_remote_code=True
        )
        
        # Configure tokenizer
        self.action_dim = action_dim
        self.time_horizon = time_horizon
        self.fast_tokenizer.action_dim = action_dim
        self.fast_tokenizer.time_horizon = time_horizon
    
    def encode(
        self,
        action: Union[np.ndarray, List[float]],
        normalize_gripper: bool = False,
    ) -> List[int]:
        """
        Encode continuous action to discrete FAST+ tokens.
        
        Args:
            action: Continuous action array of shape (action_dim,) or (time_horizon, action_dim)
            normalize_gripper: If True, normalize gripper from [-1,1] to [0,1] before encoding
            
        Returns:
            List of FAST+ token indices
            
        Requirements:
            - 2.2: Convert 7-DoF continuous actions to discrete action_tokens
        """
        action = np.asarray(action, dtype=np.float32)
        
        # Ensure correct shape
        if action.ndim == 1:
            action = action.reshape(1, -1)  # (1, action_dim)
        
        # Normalize gripper if needed (Req 2.5)
        if normalize_gripper:
            action = self._normalize_gripper_for_encoding(action.copy())
        
        # Encode using FAST+ tokenizer
        # FAST+ expects shape (batch, time_horizon, action_dim)
        action_batch = action.reshape(1, -1, self.action_dim)
        tokens = self.fast_tokenizer(action_batch)
        
        # Return flat list of token indices
        return list(tokens[0])
    
    def decode(
        self,
        tokens: Union[List[int], np.ndarray],
        denormalize_gripper: bool = False,
    ) -> np.ndarray:
        """
        Decode FAST+ tokens back to continuous actions.
        
        Args:
            tokens: List of FAST+ token indices
            denormalize_gripper: If True, denormalize gripper from [0,1] to [-1,1] after decoding
            
        Returns:
            Continuous action array of shape (time_horizon, action_dim)
            
        Requirements:
            - 2.4: Provide decode function to convert action_tokens back to continuous actions
        """
        tokens = np.asarray(tokens, dtype=np.int64)
        
        # Decode using FAST+ tokenizer
        action = self.fast_tokenizer.decode(tokens)
        
        # Denormalize gripper if needed
        if denormalize_gripper:
            action = self._denormalize_gripper_after_decoding(action.copy())
        
        return action
    
    def decode_from_vlm_ids(
        self,
        vlm_token_ids: Union[List[int], np.ndarray],
        denormalize_gripper: bool = False,
    ) -> np.ndarray:
        """
        Decode VLM token IDs (in range 151665-153712) back to continuous actions.
        
        This handles the offset between VLM vocabulary IDs and FAST+ token indices.
        
        Args:
            vlm_token_ids: List of VLM token IDs in range [ACTION_TOKEN_MIN, ACTION_TOKEN_MAX]
            denormalize_gripper: If True, denormalize gripper from [0,1] to [-1,1]
            
        Returns:
            Continuous action array of shape (time_horizon, action_dim)
            
        Requirements:
            - 2.6: Use action token ID range 151665-153712 consistent with NORA
        """
        vlm_token_ids = np.asarray(vlm_token_ids, dtype=np.int64)
        
        # Convert VLM IDs to FAST+ indices by subtracting offset
        fast_tokens = vlm_token_ids - ACTION_TOKEN_MIN
        
        return self.decode(fast_tokens, denormalize_gripper=denormalize_gripper)
    
    def tokens_to_vlm_string(self, tokens: List[int]) -> str:
        """
        Convert FAST+ token indices to VLM action string format.
        
        Args:
            tokens: List of FAST+ token indices
            
        Returns:
            String in format "<robot_action_0><robot_action_1>..."
            
        Requirements:
            - 2.3: Map FAST+ token indices to VLM action format as <robot_action_N> strings
        """
        return ''.join([f"<robot_action_{token}>" for token in tokens])
    
    def encode_to_vlm_string(
        self,
        action: Union[np.ndarray, List[float]],
        normalize_gripper: bool = False,
    ) -> str:
        """
        Encode continuous action directly to VLM action string.
        
        Convenience method that combines encode() and tokens_to_vlm_string().
        
        Args:
            action: Continuous action array
            normalize_gripper: If True, normalize gripper before encoding
            
        Returns:
            String in format "<robot_action_0><robot_action_1>..."
        """
        tokens = self.encode(action, normalize_gripper=normalize_gripper)
        return self.tokens_to_vlm_string(tokens)
    
    def tokens_to_vlm_ids(self, tokens: List[int]) -> List[int]:
        """
        Convert FAST+ token indices to VLM vocabulary IDs.
        
        Args:
            tokens: List of FAST+ token indices
            
        Returns:
            List of VLM token IDs in range [ACTION_TOKEN_MIN, ACTION_TOKEN_MAX]
            
        Requirements:
            - 2.6: Use action token ID range 151665-153712 consistent with NORA
        """
        return [token + ACTION_TOKEN_MIN for token in tokens]
    
    def vlm_ids_to_tokens(self, vlm_ids: List[int]) -> List[int]:
        """
        Convert VLM vocabulary IDs back to FAST+ token indices.
        
        Args:
            vlm_ids: List of VLM token IDs
            
        Returns:
            List of FAST+ token indices
        """
        return [vlm_id - ACTION_TOKEN_MIN for vlm_id in vlm_ids]
    
    def is_action_token_id(self, token_id: int) -> bool:
        """
        Check if a token ID is within the action token range.
        
        Args:
            token_id: VLM token ID to check
            
        Returns:
            True if token_id is in range [ACTION_TOKEN_MIN, ACTION_TOKEN_MAX]
        """
        return ACTION_TOKEN_MIN <= token_id <= ACTION_TOKEN_MAX
    
    def _normalize_gripper_for_encoding(self, action: np.ndarray) -> np.ndarray:
        """
        Normalize gripper action from [-1, 1] to [0, 1] for LIBERO.
        
        LIBERO uses: -1 = open, +1 = close
        NORA/FAST+ expects: 0 = close, 1 = open
        
        Transformation: gripper_normalized = (1 - gripper_libero) / 2
        - LIBERO -1 (open) -> NORA 1 (open)
        - LIBERO +1 (close) -> NORA 0 (close)
        
        Args:
            action: Action array with gripper in last dimension
            
        Returns:
            Action array with normalized gripper
            
        Requirements:
            - 2.5: Handle gripper action normalization: invert from [-1,1] to [0,1]
        """
        # Invert and scale: [-1, 1] -> [1, 0] -> [0, 1] with inversion
        # LIBERO: -1=open, +1=close
        # NORA: 0=close, 1=open
        # So we need: gripper_nora = (1 - gripper_libero) / 2
        action[..., -1] = (1.0 - action[..., -1]) / 2.0
        return action
    
    def _denormalize_gripper_after_decoding(self, action: np.ndarray) -> np.ndarray:
        """
        Denormalize gripper action from [0, 1] back to [-1, 1] for LIBERO.
        
        Inverse of _normalize_gripper_for_encoding.
        
        Transformation: gripper_libero = 1 - 2 * gripper_nora
        - NORA 1 (open) -> LIBERO -1 (open)
        - NORA 0 (close) -> LIBERO +1 (close)
        
        Args:
            action: Action array with gripper in last dimension
            
        Returns:
            Action array with denormalized gripper
        """
        # Inverse: gripper_libero = 1 - 2 * gripper_nora
        action[..., -1] = 1.0 - 2.0 * action[..., -1]
        return action


def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    """
    Normalize gripper action from [0, 1] to [-1, +1].
    
    This is used during evaluation to convert model output to environment format.
    Consistent with nora/experiments/libero/nora_utils.py
    
    Args:
        action: Action array with gripper in last dimension
        binarize: If True, binarize gripper to -1 or +1
        
    Returns:
        Action array with normalized gripper
    """
    action = action.copy()
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1
    
    if binarize:
        action[..., -1] = np.sign(action[..., -1])
    
    return action


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    """
    Flip the sign of the gripper action.
    
    This is used during evaluation for environments where -1 = open, +1 = close,
    since the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.
    Consistent with nora/experiments/libero/nora_utils.py
    
    Args:
        action: Action array with gripper in last dimension
        
    Returns:
        Action array with inverted gripper
    """
    action = action.copy()
    action[..., -1] = action[..., -1] * -1.0
    return action
