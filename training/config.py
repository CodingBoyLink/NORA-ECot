"""
Training configuration module for NORA LoRA training.
Defines TrainingConfig dataclass with LoRA, training, and Reasoning Dropout configurations.

Requirements covered: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8
"""
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
import yaml


@dataclass
class LoRAConfig:
    """LoRA adapter configuration."""
    # Req 9.1: Configurable LoRA rank (default 16 or 32)
    rank: int = 16
    # Req 9.2: Configurable LoRA alpha (default 32-64)
    alpha: int = 32
    # Req 9.3: Configurable LoRA dropout (default 0-0.05)
    dropout: float = 0.05
    # Target modules for LoRA adaptation
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"  # attention projections
    ])


@dataclass
class ReasoningDropoutConfig:
    """Reasoning Dropout configuration for CoT training."""
    # Req 9.5: Configurable Reasoning Dropout probability p_CoT (default 0.5)
    prob: float = 0.5
    # Req 9.6: Configurable CoT loss weight lambda (default 0.5)
    cot_loss_weight: float = 0.5


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    # Req 9.4: Configurable learning rate (default 5e-5 to 8e-5)
    learning_rate: float = 5e-5
    # Req 9.8: AdamW optimizer configuration
    betas: tuple = (0.9, 0.95)
    weight_decay: float = 1e-8
    # Warmup steps for cosine scheduler
    warmup_steps: int = 1000


@dataclass
class TrainingConfig:
    """
    Main training configuration for NORA LoRA training.
    
    Supports three training phases:
    - baseline: Pure behavior cloning (LoRA_B)
    - text_cot: Text ECoT + Reasoning Dropout (LoRA_T)
    - text_flow_cot: Text + Flow CoT + Reasoning Dropout (LoRA_TF)
    """
    # Model configuration
    base_model: str = "declare-lab/nora"
    fast_tokenizer: str = "physical-intelligence/fast"
    
    # Training phase: 'baseline' | 'text_cot' | 'text_flow_cot'
    phase: str = "baseline"
    
    # LIBERO subset: 'spatial' | 'object' | 'goal' | 'long'
    libero_subset: str = "spatial"
    
    # LoRA configuration
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    
    # Optimizer configuration
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    
    # Reasoning Dropout configuration (for text_cot and text_flow_cot phases)
    reasoning_dropout: ReasoningDropoutConfig = field(default_factory=ReasoningDropoutConfig)
    
    # Training hyperparameters
    per_device_batch_size: int = 16
    gradient_accumulation_steps: int = 2
    max_train_steps: int = 100000
    
    # Req 9.7: Mixed precision training (bf16/fp16)
    mixed_precision: str = "bf16"  # 'no' | 'fp16' | 'bf16'
    
    # Data configuration
    data_dir: str = "./data"
    train_split_ratio: float = 0.95
    data_seed: int = 42
    
    # Output configuration
    output_dir: str = "./outputs"
    checkpoint_save_frequency: int = 20000
    logging_steps: int = 100
    
    # Evaluation configuration
    eval_steps: int = 5000
    num_eval_trials: int = 50
    max_eval_steps: int = 500
    num_steps_wait: int = 10
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TrainingConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "TrainingConfig":
        """Create configuration from dictionary."""
        # Handle nested configs
        if 'lora' in config_dict:
            config_dict['lora'] = LoRAConfig(**config_dict['lora'])
        if 'optimizer' in config_dict:
            config_dict['optimizer'] = OptimizerConfig(**config_dict['optimizer'])
        if 'reasoning_dropout' in config_dict:
            config_dict['reasoning_dropout'] = ReasoningDropoutConfig(**config_dict['reasoning_dropout'])
        return cls(**config_dict)
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'base_model': self.base_model,
            'fast_tokenizer': self.fast_tokenizer,
            'phase': self.phase,
            'libero_subset': self.libero_subset,
            'lora': {
                'rank': self.lora.rank,
                'alpha': self.lora.alpha,
                'dropout': self.lora.dropout,
                'target_modules': self.lora.target_modules,
            },
            'optimizer': {
                'learning_rate': self.optimizer.learning_rate,
                'betas': self.optimizer.betas,
                'weight_decay': self.optimizer.weight_decay,
                'warmup_steps': self.optimizer.warmup_steps,
            },
            'reasoning_dropout': {
                'prob': self.reasoning_dropout.prob,
                'cot_loss_weight': self.reasoning_dropout.cot_loss_weight,
            },
            'per_device_batch_size': self.per_device_batch_size,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'max_train_steps': self.max_train_steps,
            'mixed_precision': self.mixed_precision,
            'data_dir': self.data_dir,
            'train_split_ratio': self.train_split_ratio,
            'data_seed': self.data_seed,
            'output_dir': self.output_dir,
            'checkpoint_save_frequency': self.checkpoint_save_frequency,
            'logging_steps': self.logging_steps,
            'eval_steps': self.eval_steps,
            'num_eval_trials': self.num_eval_trials,
            'max_eval_steps': self.max_eval_steps,
            'num_steps_wait': self.num_steps_wait,
        }
    
    def save_yaml(self, yaml_path: str):
        """Save configuration to YAML file."""
        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    def get_output_path(self) -> str:
        """Get output path for this configuration."""
        return f"{self.output_dir}/{self.phase}/{self.libero_subset}"
    
    def get_lora_name(self) -> str:
        """Get LoRA model name based on phase and subset."""
        phase_prefix = {
            'baseline': 'LoRA_B',
            'text_cot': 'LoRA_T',
            'text_flow_cot': 'LoRA_TF',
        }
        prefix = phase_prefix.get(self.phase, 'LoRA')
        return f"{prefix}_{self.libero_subset}"
