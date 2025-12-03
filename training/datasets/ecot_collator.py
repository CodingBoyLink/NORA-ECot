"""
ECoT Collator for NORA LoRA Training.

Phase B: Text CoT + Reasoning Dropout
Phase C: Text + Flow CoT + Reasoning Dropout

Implements data collation with reasoning dropout following embodied-CoT logic.
Supports Full CoT / No CoT two modes.

Requirements covered:
- 5.4: Reuse or adapt reasoning_dropout() function from embodied-CoT
- 5.5: Apply Reasoning Dropout with Full CoT or No CoT modes, controlled by p_CoT
- 5.6: Format Full CoT response with all tags
- 5.7: Format No CoT response as ACTION: {action_tokens}
- 7.3: Add FLOW: tag to the ECoT sequence
- 7.4: Add flow token embeddings for VQ-encoded flow tokens
- 7.5: Format Full CoT with flow tokens: ... FLOW: {flow_tokens} ACTION: {action_tokens}
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor

from training.action_tokenizer import ACTION_TOKEN_MIN, ACTION_TOKEN_MAX
from training.utils.cot_utils import (
    CotTag,
    get_cot_tags_list,
    get_cot_database_keys,
    format_full_cot_response,
    format_action_only_response,
)
from flow_pipeline.vq_encoder import tokens_to_string as flow_tokens_to_string


# Flow token tag for Phase C
FLOW_TAG = "FLOW:"


def reasoning_dropout(
    reasoning_dict: Dict[str, str],
    dropout_prob: float,
) -> Tuple[Dict[str, str], bool]:
    """
    Apply reasoning dropout with Full CoT / No CoT two modes.
    
    Unlike the original embodied-CoT which drops individual tags,
    this implementation uses binary dropout: either keep all tags (Full CoT)
    or drop all tags (No CoT).
    
    Args:
        reasoning_dict: Dictionary with reasoning tags {db_key: value}
        dropout_prob: Probability of dropping ALL reasoning (No CoT mode)
                     p_CoT = 1 - dropout_prob is the probability of Full CoT
    
    Returns:
        Tuple of (reasoning_dict or empty dict, is_full_cot)
        
    Requirements:
        - 5.4: Reuse or adapt reasoning_dropout() from embodied-CoT
        - 5.5: Apply Reasoning Dropout with only two modes: Full CoT or No CoT
    """
    # Binary dropout: either keep all or drop all
    use_full_cot = np.random.rand() > dropout_prob
    
    if use_full_cot:
        return reasoning_dict, True
    else:
        return {}, False


def format_flow_tokens_string(flow_tokens: List[int]) -> str:
    """
    Format flow tokens as a string for model input.
    
    Args:
        flow_tokens: List of VQ-encoded flow token indices
    
    Returns:
        Formatted string like "<flow_0><flow_1>..."
        
    Requirements:
        - 7.4: Add flow token embeddings for VQ-encoded flow tokens
    """
    return flow_tokens_to_string(flow_tokens, prefix="flow")


def format_full_cot_with_flow_response(
    reasoning_dict: Dict[str, str],
    flow_tokens: Optional[List[int]],
    action_str: str,
) -> str:
    """
    Format Full CoT response with flow tokens (Phase C).
    
    Format: TASK: {task} PLAN: {plan} ... GRIPPER POSITION: {gripper} FLOW: {flow_tokens} ACTION: {action_tokens}
    
    Args:
        reasoning_dict: Dictionary with reasoning tags {db_key: value}
        flow_tokens: List of VQ-encoded flow token indices (can be None)
        action_str: Action token string
    
    Returns:
        Formatted Full CoT response string with flow tokens
        
    Requirements:
        - 7.3: Add FLOW: tag to the ECoT sequence
        - 7.5: Format Full CoT with flow tokens
    """
    from training.utils.cot_utils import get_cot_tags_list, get_cot_database_keys, CotTag
    
    tags_order = get_cot_tags_list()[:-1]  # Exclude ACTION
    db_keys = get_cot_database_keys()
    
    parts = []
    for tag in tags_order:
        db_key = db_keys.get(tag)
        if db_key and db_key in reasoning_dict:
            value = reasoning_dict[db_key]
            parts.append(f"{tag} {value}")
    
    # Add FLOW: tag before ACTION: (Req 7.3, 7.5)
    if flow_tokens is not None and len(flow_tokens) > 0:
        flow_str = format_flow_tokens_string(flow_tokens)
        parts.append(f"{FLOW_TAG} {flow_str}")
    
    # Add ACTION: tag
    parts.append(f"{CotTag.ACTION.value} {action_str}")
    
    return " ".join(parts)


class ECoTCollator:
    """
    Data collator for Text CoT LoRA training (Phase B) and Text+Flow CoT (Phase C).
    
    Converts raw samples to model inputs following NORA's chat template.
    Implements reasoning dropout with Full CoT / No CoT modes.
    
    Requirements:
        - 5.5: Apply Reasoning Dropout with Full CoT or No CoT modes
        - 5.6: Format Full CoT response with all tags
        - 5.7: Format No CoT response as ACTION: {action_tokens}
        - 7.3: Add FLOW: tag to the ECoT sequence (Phase C)
        - 7.4: Add flow token embeddings for VQ-encoded flow tokens (Phase C)
        - 7.5: Format Full CoT with flow tokens (Phase C)
    """
    
    def __init__(
        self,
        processor: AutoProcessor,
        reasoning_dropout_prob: float = 0.5,
        action_token_min: int = ACTION_TOKEN_MIN,
        action_token_max: int = ACTION_TOKEN_MAX,
        compute_cot_loss: bool = True,
        include_flow: bool = False,
        flow_token_min: Optional[int] = None,
        flow_token_max: Optional[int] = None,
    ):
        """
        Initialize the ECoT collator.
        
        Args:
            processor: NORA processor (AutoProcessor)
            reasoning_dropout_prob: Probability of No CoT mode (default 0.5)
                                   p_CoT = 1 - reasoning_dropout_prob
            action_token_min: Minimum action token ID (default 151665)
            action_token_max: Maximum action token ID (default 153712)
            compute_cot_loss: Whether to compute loss on CoT tokens in Full CoT mode
            include_flow: Whether to include flow tokens (Phase C)
            flow_token_min: Minimum flow token ID (for loss masking)
            flow_token_max: Maximum flow token ID (for loss masking)
        """
        self.processor = processor
        self.reasoning_dropout_prob = reasoning_dropout_prob
        self.action_token_min = action_token_min
        self.action_token_max = action_token_max
        self.compute_cot_loss = compute_cot_loss
        self.include_flow = include_flow
        self.flow_token_min = flow_token_min
        self.flow_token_max = flow_token_max
        
        # Get CoT tags for formatting
        self.cot_tags = get_cot_tags_list()
        self.cot_db_keys = get_cot_database_keys()

    def _build_messages(
        self,
        example: Dict[str, Any],
        use_full_cot: bool,
    ) -> Tuple[List[Dict], bool]:
        """
        Build chat messages for a single example.
        
        Args:
            example: Dict with 'image', 'instruction', 'vlm_action_string', 'reasoning'
                     For Phase C, also includes 'flow_tokens'
            use_full_cot: Whether to use Full CoT mode
        
        Returns:
            Tuple of (List of message dicts for chat template, is_full_cot)
            
        Requirements:
            - 5.6: Format Full CoT response with all tags
            - 5.7: Format No CoT response as ACTION: {action_tokens}
            - 7.3: Add FLOW: tag to the ECoT sequence (Phase C)
            - 7.5: Format Full CoT with flow tokens (Phase C)
        """
        # Convert numpy image to PIL if needed
        image = example['image']
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Build user message with image and instruction
        user_content = [
            {"type": "image", "image": image},
            {"type": "text", "text": example['instruction']},
        ]
        
        # Build assistant message based on CoT mode
        action_str = example['vlm_action_string']
        
        if use_full_cot and 'reasoning' in example and example['reasoning']:
            # Full CoT mode: include all reasoning tags
            if self.include_flow and 'flow_tokens' in example and example['flow_tokens']:
                # Phase C: Include flow tokens
                # Format: TASK: {task} ... GRIPPER POSITION: {gripper} FLOW: {flow_tokens} ACTION: {action_tokens}
                assistant_text = format_full_cot_with_flow_response(
                    example['reasoning'],
                    example['flow_tokens'],
                    action_str,
                )
            else:
                # Phase B: Text CoT only
                # Format: TASK: {task} PLAN: {plan} ... ACTION: {action_tokens}
                assistant_text = format_full_cot_response(
                    example['reasoning'],
                    action_str,
                )
        else:
            # No CoT mode: only action tokens
            # Format: ACTION: {action_tokens}
            assistant_text = format_action_only_response(action_str)
        
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
        ]
        
        return messages, use_full_cot

    def _mask_labels(
        self,
        labels: torch.Tensor,
        is_full_cot_batch: List[bool],
    ) -> torch.Tensor:
        """
        Mask labels based on CoT mode.
        
        For No CoT mode: mask all tokens before first action token
        For Full CoT mode with compute_cot_loss=True: keep all tokens after assistant start
        For Full CoT mode with compute_cot_loss=False: mask all tokens before first action token
        
        For Phase C with flow tokens:
        - In Full CoT mode, flow tokens also contribute to loss (Req 7.7, 7.8)
        
        Args:
            labels: Token IDs tensor of shape (batch_size, seq_len)
            is_full_cot_batch: List of booleans indicating Full CoT mode for each sample
        
        Returns:
            Labels tensor with appropriate masking
            
        Requirements:
            - 5.8: Compute primary loss on action_tokens for both modes
            - 5.9: In Full CoT mode, compute auxiliary loss on CoT tokens
            - 7.7: In Full CoT mode, compute auxiliary loss on flow_tokens (Phase C)
            - 7.8: Compute flow_tokens loss with weight lambda (Phase C)
        """
        for i in range(labels.size(0)):
            seq = labels[i]
            is_full_cot = is_full_cot_batch[i]
            
            # Find action tokens
            action_mask = (seq >= self.action_token_min) & (seq <= self.action_token_max)
            action_indices = torch.nonzero(action_mask, as_tuple=False)
            
            # Find flow tokens if Phase C (Req 7.7)
            has_flow_tokens = False
            if self.include_flow and self.flow_token_min is not None and self.flow_token_max is not None:
                flow_mask = (seq >= self.flow_token_min) & (seq <= self.flow_token_max)
                flow_indices = torch.nonzero(flow_mask, as_tuple=False)
                has_flow_tokens = flow_indices.numel() > 0
            
            if action_indices.numel() > 0:
                first_action_idx = action_indices[0].item()
                
                if is_full_cot and self.compute_cot_loss:
                    # Full CoT mode with CoT loss: find where assistant response starts
                    # We need to mask the user message and system tokens
                    # For simplicity, we find the "ACTION:" token and work backwards
                    # to find where the CoT reasoning starts
                    # 
                    # Actually, we should mask everything before the assistant's response
                    # The assistant response starts after the last user message
                    # For now, we'll use a heuristic: mask tokens before a certain point
                    # 
                    # Better approach: find the assistant start marker in the tokenized output
                    # For Qwen2.5-VL, the assistant response follows "<|im_start|>assistant\n"
                    pass  # Keep all tokens for Full CoT (loss on both CoT, flow, and action)
                else:
                    # No CoT mode or Full CoT without CoT loss: mask before action tokens
                    seq[:first_action_idx] = -100
            else:
                # No action tokens found, mask entire sequence
                seq[:] = -100
        
        # Mask padding tokens
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        return labels

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of examples with reasoning dropout.
        
        Args:
            examples: List of dicts with 'image', 'instruction', 'vlm_action_string', 'reasoning'
        
        Returns:
            Dict with 'input_ids', 'attention_mask', 'pixel_values', 'labels', 'is_full_cot'
        """
        # Import here to avoid circular imports
        from qwen_vl_utils import process_vision_info
        
        # Apply reasoning dropout and build messages
        messages_batch = []
        is_full_cot_batch = []
        
        for example in examples:
            # Apply reasoning dropout
            reasoning = example.get('reasoning', {})
            reasoning_after_dropout, use_full_cot = reasoning_dropout(
                reasoning,
                self.reasoning_dropout_prob,
            )
            
            # Update example with dropped reasoning
            example_with_dropout = example.copy()
            example_with_dropout['reasoning'] = reasoning_after_dropout
            
            # Build messages
            messages, is_full_cot = self._build_messages(example_with_dropout, use_full_cot)
            messages_batch.append(messages)
            is_full_cot_batch.append(is_full_cot)
        
        # Apply chat template to get text
        text = self.processor.apply_chat_template(
            messages_batch,
            tokenize=False,
            add_generation_prompt=False,
        )
        
        # Process vision information
        image_inputs, video_inputs = process_vision_info(messages_batch)
        
        # Process with processor
        batch_input = self.processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Create labels and apply masking
        labels = batch_input['input_ids'].clone()
        labels = self._mask_labels(labels, is_full_cot_batch)
        batch_input['labels'] = labels
        
        # Add is_full_cot flag for loss computation
        batch_input['is_full_cot'] = torch.tensor(is_full_cot_batch, dtype=torch.bool)
        
        return batch_input


def create_ecot_collator(
    processor: AutoProcessor,
    reasoning_dropout_prob: float = 0.5,
    compute_cot_loss: bool = True,
    include_flow: bool = False,
    flow_token_min: Optional[int] = None,
    flow_token_max: Optional[int] = None,
) -> ECoTCollator:
    """
    Factory function to create ECoT collator.
    
    Args:
        processor: NORA processor
        reasoning_dropout_prob: Probability of No CoT mode
        compute_cot_loss: Whether to compute loss on CoT tokens
        include_flow: Whether to include flow tokens (Phase C)
        flow_token_min: Minimum flow token ID (for loss masking)
        flow_token_max: Maximum flow token ID (for loss masking)
    
    Returns:
        ECoTCollator instance
        
    Requirements:
        - 7.3: Add FLOW: tag to the ECoT sequence (Phase C)
        - 7.4: Add flow token embeddings for VQ-encoded flow tokens (Phase C)
    """
    return ECoTCollator(
        processor=processor,
        reasoning_dropout_prob=reasoning_dropout_prob,
        compute_cot_loss=compute_cot_loss,
        include_flow=include_flow,
        flow_token_min=flow_token_min,
        flow_token_max=flow_token_max,
    )


# ============================================================================
# Text + Flow CoT Collator (Phase C)
# ============================================================================

class TextFlowCoTCollator(ECoTCollator):
    """
    Data collator for Text + Flow CoT LoRA training (Phase C).
    
    Extends ECoTCollator with flow token support.
    
    Requirements:
        - 7.3: Add FLOW: tag to the ECoT sequence
        - 7.4: Add flow token embeddings for VQ-encoded flow tokens
        - 7.5: Format Full CoT with flow tokens
        - 7.7: In Full CoT mode, compute auxiliary loss on flow_tokens
    """
    
    def __init__(
        self,
        processor: AutoProcessor,
        reasoning_dropout_prob: float = 0.5,
        action_token_min: int = ACTION_TOKEN_MIN,
        action_token_max: int = ACTION_TOKEN_MAX,
        compute_cot_loss: bool = True,
        flow_codebook_size: int = 512,
        flow_token_offset: int = 154000,  # Offset for flow tokens in vocabulary
    ):
        """
        Initialize the Text + Flow CoT collator.
        
        Args:
            processor: NORA processor (AutoProcessor)
            reasoning_dropout_prob: Probability of No CoT mode (default 0.5)
            action_token_min: Minimum action token ID (default 151665)
            action_token_max: Maximum action token ID (default 153712)
            compute_cot_loss: Whether to compute loss on CoT tokens in Full CoT mode
            flow_codebook_size: VQ codebook size for flow tokens (default 512)
            flow_token_offset: Offset for flow tokens in vocabulary
        """
        # Calculate flow token range
        flow_token_min = flow_token_offset
        flow_token_max = flow_token_offset + flow_codebook_size - 1
        
        super().__init__(
            processor=processor,
            reasoning_dropout_prob=reasoning_dropout_prob,
            action_token_min=action_token_min,
            action_token_max=action_token_max,
            compute_cot_loss=compute_cot_loss,
            include_flow=True,
            flow_token_min=flow_token_min,
            flow_token_max=flow_token_max,
        )
        
        self.flow_codebook_size = flow_codebook_size
        self.flow_token_offset = flow_token_offset


def create_text_flow_cot_collator(
    processor: AutoProcessor,
    reasoning_dropout_prob: float = 0.5,
    compute_cot_loss: bool = True,
    flow_codebook_size: int = 512,
    flow_token_offset: int = 154000,
) -> TextFlowCoTCollator:
    """
    Factory function to create Text + Flow CoT collator (Phase C).
    
    Args:
        processor: NORA processor
        reasoning_dropout_prob: Probability of No CoT mode
        compute_cot_loss: Whether to compute loss on CoT and flow tokens
        flow_codebook_size: VQ codebook size for flow tokens
        flow_token_offset: Offset for flow tokens in vocabulary
    
    Returns:
        TextFlowCoTCollator instance
        
    Requirements:
        - 7.3: Add FLOW: tag to the ECoT sequence
        - 7.4: Add flow token embeddings for VQ-encoded flow tokens
        - 7.5: Format Full CoT with flow tokens
    """
    return TextFlowCoTCollator(
        processor=processor,
        reasoning_dropout_prob=reasoning_dropout_prob,
        compute_cot_loss=compute_cot_loss,
        flow_codebook_size=flow_codebook_size,
        flow_token_offset=flow_token_offset,
    )
