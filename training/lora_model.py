"""
LoRA Model Loading Module for NORA LoRA Training.

Implements LoRA adapter initialization on top of frozen NORA base model.

Requirements covered:
- 3.1: Load pretrained NORA weights from declare-lab/nora with all parameters frozen
- 3.2: Initialize LoRA adapters on attention projections (W_q, W_k, W_v, W_o)
- 3.3: Use LoRA rank r=16 or 32, alpha=32-64, and dropout=0-0.05
"""
from typing import Optional, Tuple, Dict, Any
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

from training.config import TrainingConfig, LoRAConfig


def load_nora_base_model(
    model_path: str = "declare-lab/nora",
    torch_dtype: torch.dtype = torch.bfloat16,
    use_flash_attention: bool = True,
    device_map: Optional[str] = None,
) -> Qwen2_5_VLForConditionalGeneration:
    """
    Load the pretrained NORA base model.
    
    Args:
        model_path: HuggingFace model path or local path
        torch_dtype: Model dtype (default bf16)
        use_flash_attention: Whether to use flash attention 2
        device_map: Device map for model parallelism
    
    Returns:
        Loaded Qwen2.5-VL model
        
    Requirements:
        - 3.1: Load pretrained NORA weights from declare-lab/nora
    """
    attn_impl = "flash_attention_2" if use_flash_attention else "eager"
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        attn_implementation=attn_impl,
        device_map=device_map,
    )
    
    return model


def load_processor(model_path: str = "declare-lab/nora") -> AutoProcessor:
    """
    Load the processor for NORA model.
    
    Args:
        model_path: HuggingFace model path
    
    Returns:
        AutoProcessor instance with left padding
    """
    processor = AutoProcessor.from_pretrained(model_path)
    processor.tokenizer.padding_side = 'left'
    return processor


def create_lora_config(lora_config: LoRAConfig) -> LoraConfig:
    """
    Create PEFT LoraConfig from our LoRAConfig.
    
    Args:
        lora_config: Our LoRAConfig dataclass
    
    Returns:
        PEFT LoraConfig
        
    Requirements:
        - 3.2: Initialize LoRA adapters on attention projections
        - 3.3: Use LoRA rank r=16 or 32, alpha=32-64, dropout=0-0.05
    """
    return LoraConfig(
        r=lora_config.rank,
        lora_alpha=lora_config.alpha,
        lora_dropout=lora_config.dropout,
        target_modules=lora_config.target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def freeze_base_model(model: Qwen2_5_VLForConditionalGeneration) -> None:
    """
    Freeze all parameters of the base model.
    
    Args:
        model: The model to freeze
        
    Requirements:
        - 3.1: All parameters frozen
    """
    for param in model.parameters():
        param.requires_grad = False


def add_lora_adapter(
    model: Qwen2_5_VLForConditionalGeneration,
    lora_config: LoRAConfig,
) -> PeftModel:
    """
    Add LoRA adapter to the model.
    
    This freezes the base model and adds trainable LoRA parameters.
    
    Args:
        model: Base NORA model
        lora_config: LoRA configuration
    
    Returns:
        PeftModel with LoRA adapter
        
    Requirements:
        - 3.1: All base parameters frozen
        - 3.2: Initialize LoRA adapters on attention projections
        - 3.3: Use configured LoRA hyperparameters
    """
    # Freeze base model first
    freeze_base_model(model)
    
    # Create PEFT config
    peft_config = create_lora_config(lora_config)
    
    # Add LoRA adapter
    peft_model = get_peft_model(model, peft_config)
    
    return peft_model


def load_model_with_lora(
    config: TrainingConfig,
    use_flash_attention: bool = True,
    device_map: Optional[str] = None,
) -> Tuple[PeftModel, AutoProcessor]:
    """
    Load NORA base model with LoRA adapter.
    
    This is the main entry point for loading a model ready for LoRA training.
    
    Args:
        config: Training configuration
        use_flash_attention: Whether to use flash attention 2
        device_map: Device map for model parallelism
    
    Returns:
        Tuple of (PeftModel with LoRA, processor)
        
    Requirements:
        - 3.1: Load pretrained NORA weights with all parameters frozen
        - 3.2: Initialize LoRA adapters on attention projections
        - 3.3: Use configured LoRA hyperparameters
    """
    # Determine dtype from config
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "no": torch.float32,
    }
    torch_dtype = dtype_map.get(config.mixed_precision, torch.bfloat16)
    
    # Load base model
    base_model = load_nora_base_model(
        model_path=config.base_model,
        torch_dtype=torch_dtype,
        use_flash_attention=use_flash_attention,
        device_map=device_map,
    )
    
    # Add LoRA adapter
    model = add_lora_adapter(base_model, config.lora)
    
    # Load processor
    processor = load_processor(config.base_model)
    
    return model, processor


def load_lora_weights(
    model: PeftModel,
    lora_path: str,
) -> PeftModel:
    """
    Load saved LoRA weights into a model.
    
    Args:
        model: PeftModel to load weights into
        lora_path: Path to saved LoRA weights
    
    Returns:
        Model with loaded LoRA weights
    """
    model.load_adapter(lora_path, adapter_name="default")
    return model


def save_lora_weights(
    model: PeftModel,
    output_path: str,
) -> None:
    """
    Save only the LoRA weights (not the base model).
    
    Args:
        model: PeftModel with LoRA adapter
        output_path: Path to save LoRA weights
        
    Requirements:
        - 3.10: Save LoRA weights separately from base model
    """
    model.save_pretrained(output_path)


def load_model_for_inference(
    base_model_path: str,
    lora_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    use_flash_attention: bool = True,
    device: Optional[str] = None,
) -> Tuple[PeftModel, AutoProcessor]:
    """
    Load model with LoRA weights for inference.
    
    Args:
        base_model_path: Path to base NORA model
        lora_path: Path to saved LoRA weights
        torch_dtype: Model dtype
        use_flash_attention: Whether to use flash attention
        device: Device to load model on
    
    Returns:
        Tuple of (model with LoRA, processor)
    """
    # Load base model
    base_model = load_nora_base_model(
        model_path=base_model_path,
        torch_dtype=torch_dtype,
        use_flash_attention=use_flash_attention,
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    if device:
        model = model.to(device)
    
    model.eval()
    
    # Load processor
    processor = load_processor(base_model_path)
    
    return model, processor


def print_trainable_parameters(model: PeftModel) -> Dict[str, Any]:
    """
    Print and return trainable parameter statistics.
    
    Args:
        model: PeftModel with LoRA adapter
    
    Returns:
        Dict with parameter statistics
    """
    trainable_params = 0
    all_params = 0
    
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    trainable_percent = 100 * trainable_params / all_params
    
    stats = {
        "trainable_params": trainable_params,
        "all_params": all_params,
        "trainable_percent": trainable_percent,
    }
    
    print(f"Trainable parameters: {trainable_params:,} / {all_params:,} ({trainable_percent:.2f}%)")
    
    return stats
