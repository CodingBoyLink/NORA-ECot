"""
Baseline Collator for NORA LoRA Training.

Implements data collation following nora/training/train.py collate_fn.
Handles action token loss masking.

Requirements covered:
- 3.6: Compute language modeling loss only on action_tokens (mask all tokens before first action token with -100)
"""
from typing import Any, Dict, List, Optional
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor

from training.action_tokenizer import ACTION_TOKEN_MIN, ACTION_TOKEN_MAX


class BaselineCollator:
    """
    Data collator for Baseline LoRA training.
    
    Converts raw samples to model inputs following NORA's chat template.
    Implements action token loss masking where only action tokens contribute to loss.
    
    Requirements:
        - 3.6: Compute loss only on action_tokens (mask others with -100)
    """
    
    def __init__(
        self,
        processor: AutoProcessor,
        action_token_min: int = ACTION_TOKEN_MIN,
        action_token_max: int = ACTION_TOKEN_MAX,
    ):
        """
        Initialize the collator.
        
        Args:
            processor: NORA processor (AutoProcessor)
            action_token_min: Minimum action token ID (default 151665)
            action_token_max: Maximum action token ID (default 153712)
        """
        self.processor = processor
        self.action_token_min = action_token_min
        self.action_token_max = action_token_max
    
    def _build_messages(self, example: Dict[str, Any]) -> List[Dict]:
        """
        Build chat messages for a single example.
        
        Format: [VIS, L, ACTION:, action_tokens]
        
        Args:
            example: Dict with 'image', 'instruction', 'vlm_action_string'
        
        Returns:
            List of message dicts for chat template
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
        
        # Build assistant message with action tokens
        # Format: just the action tokens (e.g., "<robot_action_0><robot_action_1>...")
        assistant_text = example['vlm_action_string']
        
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
        ]
        
        return messages

    def _mask_non_action_tokens(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Mask all tokens before the first action token with -100.
        
        This ensures loss is only computed on action tokens.
        
        Args:
            labels: Token IDs tensor of shape (batch_size, seq_len)
        
        Returns:
            Labels tensor with non-action tokens masked
            
        Requirements:
            - 3.6: Mask all tokens before first action token with -100
        """
        for i in range(labels.size(0)):
            seq = labels[i]
            
            # Find tokens within action token range
            mask_seq = (seq >= self.action_token_min) & (seq <= self.action_token_max)
            nonzero_indices = torch.nonzero(mask_seq, as_tuple=False)
            
            if nonzero_indices.numel() > 0:
                # Mask all tokens before the first action token
                first_action_index = nonzero_indices[0].item()
                seq[:first_action_index] = -100
            else:
                # If no action token found, mask entire sequence
                seq[:] = -100
        
        # Also mask padding tokens
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        return labels
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of examples.
        
        Args:
            examples: List of dicts with 'image', 'instruction', 'vlm_action_string'
        
        Returns:
            Dict with 'input_ids', 'attention_mask', 'pixel_values', 'labels'
        """
        # Import here to avoid circular imports
        from qwen_vl_utils import process_vision_info
        
        # Build messages for all examples
        messages_batch = [self._build_messages(ex) for ex in examples]
        
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
        labels = self._mask_non_action_tokens(labels)
        batch_input['labels'] = labels
        
        return batch_input


def create_baseline_collator(
    processor: AutoProcessor,
) -> BaselineCollator:
    """
    Factory function to create baseline collator.
    
    Args:
        processor: NORA processor
    
    Returns:
        BaselineCollator instance
    """
    return BaselineCollator(processor=processor)
