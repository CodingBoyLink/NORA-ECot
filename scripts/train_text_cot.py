#!/usr/bin/env python
"""
Text CoT LoRA Training Script for NORA (Phase B).

Train Text CoT LoRA models (LoRA_T) on LIBERO subsets with reasoning dropout.
Supports specifying LIBERO subset, ECoT annotations, and reasoning dropout probability.

Requirements covered:
- 5.10: Train separate LoRA_T weights for each LIBERO subset
- 5.11: Save each LoRA_T weights separately from base model with subset identifier

Usage:
    # Train on LIBERO-Object subset
    python scripts/train_text_cot.py --subset object --data_dir ./data \
        --ecot_annotations ./data/ecot_annotations/object.json
    
    # Train with custom config
    python scripts/train_text_cot.py --config configs/text_cot.yaml \
        --ecot_annotations ./data/ecot_annotations/object.json
    
    # Train with custom reasoning dropout probability
    python scripts/train_text_cot.py --subset object --reasoning_dropout_prob 0.3
    
    # Resume from checkpoint
    python scripts/train_text_cot.py --subset object --resume outputs/text_cot/object/step_20000
"""
import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from training.config import TrainingConfig
from training.lora_trainer import TextCoTTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Text CoT LoRA model on LIBERO subset (Phase B)"
    )
    
    # Data arguments
    parser.add_argument(
        "--subset",
        type=str,
        default="object",
        choices=["spatial", "object", "goal", "long"],
        help="LIBERO subset to train on",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="LIBERO data directory",
    )
    parser.add_argument(
        "--ecot_annotations",
        type=str,
        required=True,
        help="Path to ECoT annotations JSON file",
    )
    
    # Config arguments
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (overrides other arguments)",
    )
    
    # Model arguments
    parser.add_argument(
        "--base_model",
        type=str,
        default="declare-lab/nora",
        help="Base model path",
    )

    # LoRA arguments
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout",
    )

    # Reasoning Dropout arguments (Req 5.5)
    parser.add_argument(
        "--reasoning_dropout_prob",
        type=float,
        default=0.5,
        help="Probability of No CoT mode (p_CoT = 1 - this value)",
    )
    parser.add_argument(
        "--cot_loss_weight",
        type=float,
        default=0.5,
        help="Weight for CoT auxiliary loss (lambda)",
    )

    # Training arguments
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Per-device batch size",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=100000,
        help="Maximum training steps",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=1000,
        help="Warmup steps",
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory",
    )
    parser.add_argument(
        "--checkpoint_frequency",
        type=int,
        default=20000,
        help="Checkpoint save frequency (steps)",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Logging frequency (steps)",
    )
    
    # Resume arguments
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    
    # Misc arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable wandb logging",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision training",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    
    # Validate ECoT annotations path
    ecot_path = Path(args.ecot_annotations)
    if not ecot_path.exists():
        logger.error(f"ECoT annotations file not found: {ecot_path}")
        sys.exit(1)
    
    # Load or create config
    if args.config:
        logger.info(f"Loading config from {args.config}")
        config = TrainingConfig.from_yaml(args.config)
        # Override subset if specified
        if args.subset:
            config.libero_subset = args.subset
    else:
        # Create config from arguments
        config = TrainingConfig(
            # Model
            base_model=args.base_model,
            phase="text_cot",
            libero_subset=args.subset,
            # LoRA
            lora=TrainingConfig.__dataclass_fields__['lora'].default_factory(),
            # Training
            per_device_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_train_steps=args.max_train_steps,
            mixed_precision=args.mixed_precision,
            # Data
            data_dir=args.data_dir,
            data_seed=args.seed,
            # Output
            output_dir=args.output_dir,
            checkpoint_save_frequency=args.checkpoint_frequency,
            logging_steps=args.logging_steps,
        )
        
        # Update LoRA config
        config.lora.rank = args.lora_rank
        config.lora.alpha = args.lora_alpha
        config.lora.dropout = args.lora_dropout
        
        # Update optimizer config
        config.optimizer.learning_rate = args.learning_rate
        config.optimizer.warmup_steps = args.warmup_steps
        
        # Update reasoning dropout config (Req 5.5)
        config.reasoning_dropout.prob = args.reasoning_dropout_prob
        config.reasoning_dropout.cot_loss_weight = args.cot_loss_weight

    # Ensure phase is text_cot
    config.phase = "text_cot"

    # Log configuration
    logger.info("=" * 60)
    logger.info("Text CoT LoRA Training Configuration (Phase B)")
    logger.info("=" * 60)
    logger.info(f"  Phase: {config.phase}")
    logger.info(f"  LIBERO subset: {config.libero_subset}")
    logger.info(f"  Base model: {config.base_model}")
    logger.info(f"  LoRA rank: {config.lora.rank}")
    logger.info(f"  LoRA alpha: {config.lora.alpha}")
    logger.info(f"  Learning rate: {config.optimizer.learning_rate}")
    logger.info(f"  Batch size: {config.per_device_batch_size}")
    logger.info(f"  Max train steps: {config.max_train_steps}")
    logger.info(f"  Reasoning dropout prob: {config.reasoning_dropout.prob}")
    logger.info(f"  CoT loss weight (lambda): {config.reasoning_dropout.cot_loss_weight}")
    logger.info(f"  ECoT annotations: {args.ecot_annotations}")
    logger.info(f"  Output dir: {config.get_output_path()}")
    logger.info(f"  LoRA name: {config.get_lora_name()}")
    logger.info("=" * 60)
    
    # Save config
    output_path = Path(config.get_output_path())
    output_path.mkdir(parents=True, exist_ok=True)
    config.save_yaml(str(output_path / "config.yaml"))
    logger.info(f"Config saved to {output_path / 'config.yaml'}")
    
    # Create trainer
    trainer = TextCoTTrainer(
        config=config,
        ecot_annotations_path=str(args.ecot_annotations),
        use_wandb=not args.no_wandb,
        wandb_project=f"nora-lora-text-cot-{config.libero_subset}",
    )
    
    # Setup trainer
    logger.info("Setting up trainer...")
    trainer.setup()
    
    # Train
    logger.info("Starting Text CoT training...")
    trainer.train(resume_from_checkpoint=args.resume)
    
    logger.info("Training complete!")
    logger.info(f"LoRA weights saved to: {config.get_output_path()}")


if __name__ == "__main__":
    main()
