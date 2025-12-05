#!/usr/bin/env python
"""
Baseline LoRA Training Script for NORA.

Train baseline LoRA models (LoRA_B) on LIBERO subsets.
All configurations are read from YAML config file.

Requirements covered:
- 3.9: Train separate LoRA_B weights for each LIBERO subset
- 3.10: Save each LoRA_B weights separately from base model with subset identifier

Usage:
    python scripts/train_baseline.py --config configs/baseline.yaml
    
    # Resume from checkpoint
    python scripts/train_baseline.py --config configs/baseline.yaml --resume outputs/baseline/spatial/step_2000
"""
import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from training.config import TrainingConfig
from training.lora_trainer import LoRATrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train baseline LoRA model on LIBERO subset"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable wandb logging",
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
    
    # Load config from YAML
    logger.info(f"Loading config from {args.config}")
    config = TrainingConfig.from_yaml(args.config)

    # Log configuration
    logger.info("=" * 60)
    logger.info("Baseline LoRA Training Configuration")
    logger.info("=" * 60)
    logger.info(f"  Phase: {config.phase}")
    logger.info(f"  LIBERO subset: {config.libero_subset}")
    logger.info(f"  Base model: {config.base_model}")
    logger.info(f"  LoRA rank: {config.lora.rank}")
    logger.info(f"  LoRA alpha: {config.lora.alpha}")
    logger.info(f"  Learning rate: {config.optimizer.learning_rate}")
    logger.info(f"  Batch size: {config.per_device_batch_size}")
    logger.info(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    logger.info(f"  Max train steps: {config.max_train_steps}")
    logger.info(f"  Output dir: {config.get_output_path()}")
    logger.info(f"  LoRA name: {config.get_lora_name()}")
    logger.info("=" * 60)
    
    # Save config to output directory
    output_path = Path(config.get_output_path())
    output_path.mkdir(parents=True, exist_ok=True)
    config.save_yaml(str(output_path / "config.yaml"))
    logger.info(f"Config saved to {output_path / 'config.yaml'}")
    
    # Create trainer
    trainer = LoRATrainer(
        config=config,
        use_wandb=not args.no_wandb,
        wandb_project=f"nora-lora-baseline-{config.libero_subset}",
    )
    
    # Setup trainer
    logger.info("Setting up trainer...")
    trainer.setup()
    
    # Train
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume)
    
    logger.info("Training complete!")
    logger.info(f"LoRA weights saved to: {config.get_output_path()}")


if __name__ == "__main__":
    main()
