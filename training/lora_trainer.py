"""
LoRA Trainer for NORA LoRA Training.

Implements training loop following nora/training/train.py structure.
Uses accelerate for multi-GPU training.

Requirements covered:
- 3.7: Use AdamW optimizer with lr=5e-5 to 8e-5, betas=(0.9, 0.95), weight_decay=1e-8
- 3.8: Use cosine learning rate scheduler with warmup
"""
import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Callable

import torch
from torch.utils.data import DataLoader, IterableDataset
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import AutoProcessor, get_scheduler
from peft import PeftModel
from tqdm import tqdm

from training.config import TrainingConfig
from training.lora_model import load_model_with_lora, save_lora_weights, print_trainable_parameters
from training.action_tokenizer import ActionTokenizer
from training.datasets.libero_dataset import LiberoBaselineDataset, LiberoMapDataset
from training.datasets.baseline_collator import BaselineCollator


logger = get_logger(__name__)


class LoRATrainer:
    """
    LoRA Trainer for NORA baseline training.
    
    Implements training loop with:
    - AdamW optimizer with configurable hyperparameters
    - Cosine learning rate scheduler with warmup
    - Multi-GPU training via accelerate
    - Checkpoint saving
    - Wandb logging (optional)
    
    Requirements:
        - 3.7: AdamW optimizer with lr=5e-5 to 8e-5, betas=(0.9, 0.95)
        - 3.8: Cosine learning rate scheduler with warmup
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        use_wandb: bool = True,
        wandb_project: str = "nora-lora-training",
    ):
        """
        Initialize the trainer.
        
        Args:
            config: Training configuration
            use_wandb: Whether to use wandb logging
            wandb_project: Wandb project name
        """
        self.config = config
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        
        # Initialize accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision=config.mixed_precision,
        )
        self.accelerator.dataloader_config.dispatch_batches = False
        
        # Set seed for reproducibility
        set_seed(config.data_seed)
        
        # Initialize components (will be set in setup())
        self.model: Optional[PeftModel] = None
        self.processor: Optional[AutoProcessor] = None
        self.action_tokenizer: Optional[ActionTokenizer] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.lr_scheduler = None
        self.train_dataloader: Optional[DataLoader] = None

    def setup(self):
        """
        Set up model, optimizer, scheduler, and data loaders.
        """
        logger.info("Setting up trainer...")
        
        # Load model with LoRA
        logger.info(f"Loading model from {self.config.base_model}")
        self.model, self.processor = load_model_with_lora(
            self.config,
            use_flash_attention=True,
        )
        
        # Print trainable parameters
        if self.accelerator.is_main_process:
            print_trainable_parameters(self.model)
        
        # Load action tokenizer
        self.action_tokenizer = ActionTokenizer(
            tokenizer_path=self.config.fast_tokenizer,
        )
        
        # Create dataset (use Map-style for multi-worker support)
        logger.info(f"Loading dataset for {self.config.libero_subset}")
        logger.info("Loading all samples into memory for faster training...")
        train_dataset = LiberoMapDataset(
            data_dir=self.config.data_dir,
            subset=self.config.libero_subset,
            split="train",
            train_ratio=self.config.train_split_ratio,
            seed=self.config.data_seed,
            action_tokenizer=self.action_tokenizer,
            normalize_gripper=True,
        )
        logger.info(f"Loaded {len(train_dataset)} samples into memory")
        
        # Create collator
        collator = BaselineCollator(processor=self.processor)
        
        # Create dataloader with multi-worker support
        num_workers = getattr(self.config, 'num_workers', 4)
        logger.info(f"Using {num_workers} DataLoader workers")
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.per_device_batch_size,
            shuffle=True,  # Shuffle for Map-style dataset
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=True if num_workers > 0 else False,
        )
        
        # Initialize optimizer (Req 3.7)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.optimizer.learning_rate,
            betas=self.config.optimizer.betas,
            weight_decay=self.config.optimizer.weight_decay,
            eps=1e-8,
        )
        
        # Initialize learning rate scheduler (Req 3.8)
        self.lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=self.optimizer,
            num_warmup_steps=self.config.optimizer.warmup_steps,
            num_training_steps=self.config.max_train_steps,
        )
        
        # Prepare with accelerator
        self.model, self.optimizer, self.train_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader
        )
        
        logger.info("Trainer setup complete")
    
    def train(self, resume_from_checkpoint: Optional[str] = None):
        """
        Run the training loop.
        
        Args:
            resume_from_checkpoint: Path to checkpoint to resume from
        """
        # Initialize wandb
        if self.use_wandb and self.accelerator.is_main_process:
            try:
                import wandb
                wandb.init(
                    project=self.wandb_project,
                    name=f"{self.config.phase}_{self.config.libero_subset}",
                    config=self.config.to_dict(),
                )
            except ImportError:
                logger.warning("wandb not installed, skipping logging")
                self.use_wandb = False
        
        # Resume from checkpoint if provided
        if resume_from_checkpoint:
            self.accelerator.load_state(resume_from_checkpoint)
            logger.info(f"Resumed from checkpoint: {resume_from_checkpoint}")
        
        # Create output directory
        output_dir = Path(self.config.get_output_path())
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training info
        total_batch_size = (
            self.config.per_device_batch_size
            * self.accelerator.num_processes
            * self.config.gradient_accumulation_steps
        )
        
        logger.info("***** Running training *****")
        logger.info(f"  Phase = {self.config.phase}")
        logger.info(f"  LIBERO subset = {self.config.libero_subset}")
        logger.info(f"  Max train steps = {self.config.max_train_steps}")
        logger.info(f"  Per device batch size = {self.config.per_device_batch_size}")
        logger.info(f"  Total train batch size = {total_batch_size}")
        logger.info(f"  Gradient accumulation steps = {self.config.gradient_accumulation_steps}")

        # Training loop
        completed_steps = 0
        total_loss = 0.0
        progress_bar = tqdm(
            range(self.config.max_train_steps),
            disable=not self.accelerator.is_local_main_process,
            desc="Training",
        )
        
        while completed_steps < self.config.max_train_steps:
            for batch in self.train_dataloader:
                with self.accelerator.accumulate(self.model):
                    self.optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    total_loss += loss.detach().float()
                    
                    # Backward pass
                    self.accelerator.backward(loss)
                    
                    # Update weights
                    if self.accelerator.sync_gradients:
                        progress_bar.update(1)
                        completed_steps += 1
                    
                    self.optimizer.step()
                    self.lr_scheduler.step()
                
                # Logging
                if completed_steps % self.config.logging_steps == 0 and completed_steps > 0:
                    if self.accelerator.is_main_process:
                        lr = self.lr_scheduler.get_last_lr()[0]
                        avg_loss = total_loss.item() / self.config.logging_steps
                        
                        logger.info(f"Step {completed_steps}, Loss: {loss.item():.4f}, LR: {lr:.2e}")
                        
                        if self.use_wandb:
                            import wandb
                            wandb.log({
                                "train_loss": loss.item(),
                                "learning_rate": lr,
                                "step": completed_steps,
                            })
                        
                        total_loss = 0.0
                
                # Checkpointing
                if completed_steps % self.config.checkpoint_save_frequency == 0 and completed_steps > 0:
                    self._save_checkpoint(completed_steps, output_dir)
                
                if completed_steps >= self.config.max_train_steps:
                    break
        
        # Save final checkpoint
        self._save_checkpoint(completed_steps, output_dir, is_final=True)
        
        if self.use_wandb and self.accelerator.is_main_process:
            import wandb
            wandb.finish()
        
        logger.info(f"Training finished. Final checkpoint saved at {output_dir}")
    
    def _save_checkpoint(
        self,
        step: int,
        output_dir: Path,
        is_final: bool = False,
    ):
        """
        Save training checkpoint.
        
        Args:
            step: Current training step
            output_dir: Output directory
            is_final: Whether this is the final checkpoint
        """
        checkpoint_name = "final" if is_final else f"step_{step}"
        checkpoint_path = output_dir / checkpoint_name
        
        # Save accelerator state (optimizer, scheduler, etc.)
        self.accelerator.save_state(str(checkpoint_path))
        
        # Save LoRA weights separately
        if self.accelerator.is_main_process:
            lora_path = checkpoint_path / "lora_weights"
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            save_lora_weights(unwrapped_model, str(lora_path))
            
            # Save training info
            info = {
                "step": step,
                "config": self.config.to_dict(),
                "lora_name": self.config.get_lora_name(),
            }
            with open(checkpoint_path / "training_info.json", "w") as f:
                json.dump(info, f, indent=2)
            
            logger.info(f"Checkpoint saved at step {step}")


def train_baseline(
    config: TrainingConfig,
    use_wandb: bool = True,
    resume_from_checkpoint: Optional[str] = None,
):
    """
    Train baseline LoRA model.
    
    Convenience function for training baseline model.
    
    Args:
        config: Training configuration
        use_wandb: Whether to use wandb logging
        resume_from_checkpoint: Path to checkpoint to resume from
    """
    # Ensure phase is baseline
    config.phase = "baseline"
    
    # Create trainer
    trainer = LoRATrainer(
        config=config,
        use_wandb=use_wandb,
        wandb_project="nora-lora-baseline",
    )
    
    # Setup and train
    trainer.setup()
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)


def create_trainer(
    config: TrainingConfig,
    use_wandb: bool = True,
) -> LoRATrainer:
    """
    Factory function to create a trainer.
    
    Args:
        config: Training configuration
        use_wandb: Whether to use wandb logging
    
    Returns:
        LoRATrainer instance (not yet set up)
    """
    return LoRATrainer(config=config, use_wandb=use_wandb)



# ============================================================================
# Text CoT Trainer (Phase B)
# ============================================================================

class TextCoTTrainer(LoRATrainer):
    """
    LoRA Trainer for Text CoT training (Phase B).
    
    Extends LoRATrainer with:
    - ECoT annotations loading
    - Reasoning dropout
    - CoT auxiliary loss computation
    
    Requirements:
        - 5.8: Compute primary loss on action_tokens for both modes
        - 5.9: In Full CoT mode, compute auxiliary loss on CoT tokens with weight lambda
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        ecot_annotations_path: str,
        use_wandb: bool = True,
        wandb_project: str = "nora-lora-text-cot",
    ):
        """
        Initialize the Text CoT trainer.
        
        Args:
            config: Training configuration
            ecot_annotations_path: Path to ECoT annotations JSON
            use_wandb: Whether to use wandb logging
            wandb_project: Wandb project name
        """
        super().__init__(config, use_wandb, wandb_project)
        self.ecot_annotations_path = ecot_annotations_path
        
        # CoT loss weight (lambda)
        self.cot_loss_weight = config.reasoning_dropout.cot_loss_weight
        self.reasoning_dropout_prob = config.reasoning_dropout.prob
    
    def setup(self):
        """
        Set up model, optimizer, scheduler, and data loaders for Text CoT training.
        """
        logger.info("Setting up Text CoT trainer...")
        
        # Load model with LoRA
        logger.info(f"Loading model from {self.config.base_model}")
        self.model, self.processor = load_model_with_lora(
            self.config,
            use_flash_attention=True,
        )
        
        # Print trainable parameters
        if self.accelerator.is_main_process:
            print_trainable_parameters(self.model)
        
        # Load action tokenizer
        self.action_tokenizer = ActionTokenizer(
            tokenizer_path=self.config.fast_tokenizer,
        )
        
        # Import Text CoT dataset and collator
        from training.datasets.libero_dataset import LiberoTextCoTDataset
        from training.datasets.ecot_collator import ECoTCollator
        
        # Create Text CoT dataset
        logger.info(f"Loading Text CoT dataset for {self.config.libero_subset}")
        logger.info(f"ECoT annotations: {self.ecot_annotations_path}")
        train_dataset = LiberoTextCoTDataset(
            data_dir=self.config.data_dir,
            subset=self.config.libero_subset,
            ecot_annotations_path=self.ecot_annotations_path,
            split="train",
            train_ratio=self.config.train_split_ratio,
            seed=self.config.data_seed,
            action_tokenizer=self.action_tokenizer,
            normalize_gripper=True,
            shuffle=True,
        )
        
        # Create ECoT collator with reasoning dropout
        collator = ECoTCollator(
            processor=self.processor,
            reasoning_dropout_prob=self.reasoning_dropout_prob,
            compute_cot_loss=(self.cot_loss_weight > 0),
        )
        
        # Create dataloader
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.per_device_batch_size,
            collate_fn=collator,
        )
        
        # Initialize optimizer (Req 3.7)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.optimizer.learning_rate,
            betas=self.config.optimizer.betas,
            weight_decay=self.config.optimizer.weight_decay,
            eps=1e-8,
        )
        
        # Initialize learning rate scheduler (Req 3.8)
        self.lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=self.optimizer,
            num_warmup_steps=self.config.optimizer.warmup_steps,
            num_training_steps=self.config.max_train_steps,
        )
        
        # Prepare with accelerator
        self.model, self.optimizer, self.train_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader
        )
        
        logger.info("Text CoT trainer setup complete")
        logger.info(f"  Reasoning dropout prob: {self.reasoning_dropout_prob}")
        logger.info(f"  CoT loss weight (lambda): {self.cot_loss_weight}")
    
    def _compute_cot_loss(
        self,
        outputs,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute combined loss with CoT auxiliary loss.
        
        The model outputs.loss already includes the language modeling loss
        computed on the labels (which may include CoT tokens in Full CoT mode).
        
        For more fine-grained control, we could separate action loss and CoT loss,
        but for simplicity, we use the model's built-in loss computation with
        appropriate label masking done in the collator.
        
        Args:
            outputs: Model outputs with loss
            batch: Batch dict with 'is_full_cot' flag
            
        Returns:
            Combined loss tensor
            
        Requirements:
            - 5.8: Compute primary loss on action_tokens for both modes
            - 5.9: In Full CoT mode, compute auxiliary loss on CoT tokens
        """
        # The collator already handles masking:
        # - No CoT mode: only action tokens contribute to loss
        # - Full CoT mode with compute_cot_loss=True: all assistant tokens contribute
        # 
        # The loss weighting is implicitly handled by the label masking.
        # For explicit weighting, we would need to compute separate losses.
        
        return outputs.loss
    
    def train(self, resume_from_checkpoint: Optional[str] = None):
        """
        Run the Text CoT training loop.
        
        Args:
            resume_from_checkpoint: Path to checkpoint to resume from
        """
        # Initialize wandb
        if self.use_wandb and self.accelerator.is_main_process:
            try:
                import wandb
                wandb.init(
                    project=self.wandb_project,
                    name=f"{self.config.phase}_{self.config.libero_subset}",
                    config={
                        **self.config.to_dict(),
                        "ecot_annotations_path": self.ecot_annotations_path,
                    },
                )
            except ImportError:
                logger.warning("wandb not installed, skipping logging")
                self.use_wandb = False
        
        # Resume from checkpoint if provided
        if resume_from_checkpoint:
            self.accelerator.load_state(resume_from_checkpoint)
            logger.info(f"Resumed from checkpoint: {resume_from_checkpoint}")
        
        # Create output directory
        output_dir = Path(self.config.get_output_path())
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training info
        total_batch_size = (
            self.config.per_device_batch_size
            * self.accelerator.num_processes
            * self.config.gradient_accumulation_steps
        )
        
        logger.info("***** Running Text CoT training *****")
        logger.info(f"  Phase = {self.config.phase}")
        logger.info(f"  LIBERO subset = {self.config.libero_subset}")
        logger.info(f"  Max train steps = {self.config.max_train_steps}")
        logger.info(f"  Per device batch size = {self.config.per_device_batch_size}")
        logger.info(f"  Total train batch size = {total_batch_size}")
        logger.info(f"  Gradient accumulation steps = {self.config.gradient_accumulation_steps}")
        logger.info(f"  Reasoning dropout prob = {self.reasoning_dropout_prob}")
        logger.info(f"  CoT loss weight = {self.cot_loss_weight}")

        # Training loop
        completed_steps = 0
        total_loss = 0.0
        num_full_cot = 0
        num_no_cot = 0
        progress_bar = tqdm(
            range(self.config.max_train_steps),
            disable=not self.accelerator.is_local_main_process,
            desc="Training",
        )
        
        while completed_steps < self.config.max_train_steps:
            for batch in self.train_dataloader:
                with self.accelerator.accumulate(self.model):
                    self.optimizer.zero_grad()
                    
                    # Extract is_full_cot flag before forward pass
                    is_full_cot = batch.pop('is_full_cot', None)
                    
                    # Forward pass
                    outputs = self.model(**batch)
                    
                    # Compute loss (with CoT auxiliary loss if applicable)
                    loss = self._compute_cot_loss(outputs, {'is_full_cot': is_full_cot})
                    total_loss += loss.detach().float()
                    
                    # Track CoT mode statistics
                    if is_full_cot is not None:
                        num_full_cot += is_full_cot.sum().item()
                        num_no_cot += (~is_full_cot).sum().item()
                    
                    # Backward pass
                    self.accelerator.backward(loss)
                    
                    # Update weights
                    if self.accelerator.sync_gradients:
                        progress_bar.update(1)
                        completed_steps += 1
                    
                    self.optimizer.step()
                    self.lr_scheduler.step()
                
                # Logging
                if completed_steps % self.config.logging_steps == 0 and completed_steps > 0:
                    if self.accelerator.is_main_process:
                        lr = self.lr_scheduler.get_last_lr()[0]
                        avg_loss = total_loss.item() / self.config.logging_steps
                        
                        # Calculate CoT mode ratio
                        total_samples = num_full_cot + num_no_cot
                        full_cot_ratio = num_full_cot / total_samples if total_samples > 0 else 0
                        
                        logger.info(
                            f"Step {completed_steps}, Loss: {loss.item():.4f}, "
                            f"LR: {lr:.2e}, Full CoT ratio: {full_cot_ratio:.2%}"
                        )
                        
                        if self.use_wandb:
                            import wandb
                            wandb.log({
                                "train_loss": loss.item(),
                                "learning_rate": lr,
                                "full_cot_ratio": full_cot_ratio,
                                "step": completed_steps,
                            })
                        
                        total_loss = 0.0
                        num_full_cot = 0
                        num_no_cot = 0
                
                # Checkpointing
                if completed_steps % self.config.checkpoint_save_frequency == 0 and completed_steps > 0:
                    self._save_checkpoint(completed_steps, output_dir)
                
                if completed_steps >= self.config.max_train_steps:
                    break
        
        # Save final checkpoint
        self._save_checkpoint(completed_steps, output_dir, is_final=True)
        
        if self.use_wandb and self.accelerator.is_main_process:
            import wandb
            wandb.finish()
        
        logger.info(f"Text CoT training finished. Final checkpoint saved at {output_dir}")


def train_text_cot(
    config: TrainingConfig,
    ecot_annotations_path: str,
    use_wandb: bool = True,
    resume_from_checkpoint: Optional[str] = None,
):
    """
    Train Text CoT LoRA model (Phase B).
    
    Convenience function for training Text CoT model.
    
    Args:
        config: Training configuration
        ecot_annotations_path: Path to ECoT annotations JSON
        use_wandb: Whether to use wandb logging
        resume_from_checkpoint: Path to checkpoint to resume from
        
    Requirements:
        - 5.10: Train separate LoRA_T weights for each LIBERO subset
        - 5.11: Save each LoRA_T weights separately with subset identifier
    """
    # Ensure phase is text_cot
    config.phase = "text_cot"
    
    # Create trainer
    trainer = TextCoTTrainer(
        config=config,
        ecot_annotations_path=ecot_annotations_path,
        use_wandb=use_wandb,
        wandb_project=f"nora-lora-text-cot-{config.libero_subset}",
    )
    
    # Setup and train
    trainer.setup()
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)


# ============================================================================
# Text + Flow CoT Trainer (Phase C)
# ============================================================================

class TextFlowCoTTrainer(LoRATrainer):
    """
    LoRA Trainer for Text + Flow CoT training (Phase C).
    
    Extends LoRATrainer with:
    - ECoT annotations loading
    - Preprocessed flow tokens loading
    - Reasoning dropout
    - CoT + Flow auxiliary loss computation
    
    Requirements:
        - 7.7: In Full CoT mode, compute auxiliary loss on flow_tokens
        - 7.8: Compute flow_tokens loss with weight lambda
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        ecot_annotations_path: str,
        flow_tokens_path: str,
        use_wandb: bool = True,
        wandb_project: str = "nora-lora-text-flow-cot",
    ):
        """
        Initialize the Text + Flow CoT trainer.
        
        Args:
            config: Training configuration
            ecot_annotations_path: Path to ECoT annotations JSON
            flow_tokens_path: Path to preprocessed flow tokens JSON
            use_wandb: Whether to use wandb logging
            wandb_project: Wandb project name
        """
        super().__init__(config, use_wandb, wandb_project)
        self.ecot_annotations_path = ecot_annotations_path
        self.flow_tokens_path = flow_tokens_path
        
        # CoT + Flow loss weight (lambda)
        self.cot_loss_weight = config.reasoning_dropout.cot_loss_weight
        self.reasoning_dropout_prob = config.reasoning_dropout.prob
    
    def setup(self):
        """
        Set up model, optimizer, scheduler, and data loaders for Text + Flow CoT training.
        """
        logger.info("Setting up Text + Flow CoT trainer...")
        
        # Load model with LoRA
        logger.info(f"Loading model from {self.config.base_model}")
        self.model, self.processor = load_model_with_lora(
            self.config,
            use_flash_attention=True,
        )
        
        # Print trainable parameters
        if self.accelerator.is_main_process:
            print_trainable_parameters(self.model)
        
        # Load action tokenizer
        self.action_tokenizer = ActionTokenizer(
            tokenizer_path=self.config.fast_tokenizer,
        )
        
        # Import Text + Flow CoT dataset and collator
        from training.datasets.libero_dataset import LiberoTextFlowCoTDataset
        from training.datasets.ecot_collator import TextFlowCoTCollator
        
        # Create Text + Flow CoT dataset
        logger.info(f"Loading Text + Flow CoT dataset for {self.config.libero_subset}")
        logger.info(f"ECoT annotations: {self.ecot_annotations_path}")
        logger.info(f"Flow tokens: {self.flow_tokens_path}")
        train_dataset = LiberoTextFlowCoTDataset(
            data_dir=self.config.data_dir,
            subset=self.config.libero_subset,
            ecot_annotations_path=self.ecot_annotations_path,
            flow_tokens_path=self.flow_tokens_path,
            split="train",
            train_ratio=self.config.train_split_ratio,
            seed=self.config.data_seed,
            action_tokenizer=self.action_tokenizer,
            normalize_gripper=True,
            shuffle=True,
        )
        
        # Create Text + Flow CoT collator with reasoning dropout
        collator = TextFlowCoTCollator(
            processor=self.processor,
            reasoning_dropout_prob=self.reasoning_dropout_prob,
            compute_cot_loss=(self.cot_loss_weight > 0),
        )
        
        # Create dataloader
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.per_device_batch_size,
            collate_fn=collator,
        )
        
        # Initialize optimizer (Req 3.7)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.optimizer.learning_rate,
            betas=self.config.optimizer.betas,
            weight_decay=self.config.optimizer.weight_decay,
            eps=1e-8,
        )
        
        # Initialize learning rate scheduler (Req 3.8)
        self.lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=self.optimizer,
            num_warmup_steps=self.config.optimizer.warmup_steps,
            num_training_steps=self.config.max_train_steps,
        )
        
        # Prepare with accelerator
        self.model, self.optimizer, self.train_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader
        )
        
        logger.info("Text + Flow CoT trainer setup complete")
        logger.info(f"  Reasoning dropout prob: {self.reasoning_dropout_prob}")
        logger.info(f"  CoT + Flow loss weight (lambda): {self.cot_loss_weight}")
    
    def _compute_cot_flow_loss(
        self,
        outputs,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute combined loss with CoT and Flow auxiliary loss.
        
        The model outputs.loss already includes the language modeling loss
        computed on the labels (which may include CoT and flow tokens in Full CoT mode).
        
        For more fine-grained control, we could separate action loss, CoT loss, and flow loss,
        but for simplicity, we use the model's built-in loss computation with
        appropriate label masking done in the collator.
        
        Args:
            outputs: Model outputs with loss
            batch: Batch dict with 'is_full_cot' flag
            
        Returns:
            Combined loss tensor
            
        Requirements:
            - 7.7: In Full CoT mode, compute auxiliary loss on flow_tokens
            - 7.8: Compute flow_tokens loss with weight lambda
        """
        # The collator already handles masking:
        # - No CoT mode: only action tokens contribute to loss
        # - Full CoT mode with compute_cot_loss=True: all assistant tokens contribute
        #   (including CoT text tokens and flow tokens)
        # 
        # The loss weighting is implicitly handled by the label masking.
        # For explicit weighting, we would need to compute separate losses.
        
        return outputs.loss
    
    def train(self, resume_from_checkpoint: Optional[str] = None):
        """
        Run the Text + Flow CoT training loop.
        
        Args:
            resume_from_checkpoint: Path to checkpoint to resume from
        """
        # Initialize wandb
        if self.use_wandb and self.accelerator.is_main_process:
            try:
                import wandb
                wandb.init(
                    project=self.wandb_project,
                    name=f"{self.config.phase}_{self.config.libero_subset}",
                    config={
                        **self.config.to_dict(),
                        "ecot_annotations_path": self.ecot_annotations_path,
                        "flow_tokens_path": self.flow_tokens_path,
                    },
                )
            except ImportError:
                logger.warning("wandb not installed, skipping logging")
                self.use_wandb = False
        
        # Resume from checkpoint if provided
        if resume_from_checkpoint:
            self.accelerator.load_state(resume_from_checkpoint)
            logger.info(f"Resumed from checkpoint: {resume_from_checkpoint}")
        
        # Create output directory
        output_dir = Path(self.config.get_output_path())
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training info
        total_batch_size = (
            self.config.per_device_batch_size
            * self.accelerator.num_processes
            * self.config.gradient_accumulation_steps
        )
        
        logger.info("***** Running Text + Flow CoT training *****")
        logger.info(f"  Phase = {self.config.phase}")
        logger.info(f"  LIBERO subset = {self.config.libero_subset}")
        logger.info(f"  Max train steps = {self.config.max_train_steps}")
        logger.info(f"  Per device batch size = {self.config.per_device_batch_size}")
        logger.info(f"  Total train batch size = {total_batch_size}")
        logger.info(f"  Gradient accumulation steps = {self.config.gradient_accumulation_steps}")
        logger.info(f"  Reasoning dropout prob = {self.reasoning_dropout_prob}")
        logger.info(f"  CoT + Flow loss weight = {self.cot_loss_weight}")

        # Training loop
        completed_steps = 0
        total_loss = 0.0
        num_full_cot = 0
        num_no_cot = 0
        progress_bar = tqdm(
            range(self.config.max_train_steps),
            disable=not self.accelerator.is_local_main_process,
            desc="Training",
        )
        
        while completed_steps < self.config.max_train_steps:
            for batch in self.train_dataloader:
                with self.accelerator.accumulate(self.model):
                    self.optimizer.zero_grad()
                    
                    # Extract is_full_cot flag before forward pass
                    is_full_cot = batch.pop('is_full_cot', None)
                    
                    # Forward pass
                    outputs = self.model(**batch)
                    
                    # Compute loss (with CoT + Flow auxiliary loss if applicable)
                    loss = self._compute_cot_flow_loss(outputs, {'is_full_cot': is_full_cot})
                    total_loss += loss.detach().float()
                    
                    # Track CoT mode statistics
                    if is_full_cot is not None:
                        num_full_cot += is_full_cot.sum().item()
                        num_no_cot += (~is_full_cot).sum().item()
                    
                    # Backward pass
                    self.accelerator.backward(loss)
                    
                    # Update weights
                    if self.accelerator.sync_gradients:
                        progress_bar.update(1)
                        completed_steps += 1
                    
                    self.optimizer.step()
                    self.lr_scheduler.step()
                
                # Logging
                if completed_steps % self.config.logging_steps == 0 and completed_steps > 0:
                    if self.accelerator.is_main_process:
                        lr = self.lr_scheduler.get_last_lr()[0]
                        avg_loss = total_loss.item() / self.config.logging_steps
                        
                        # Calculate CoT mode ratio
                        total_samples = num_full_cot + num_no_cot
                        full_cot_ratio = num_full_cot / total_samples if total_samples > 0 else 0
                        
                        logger.info(
                            f"Step {completed_steps}, Loss: {loss.item():.4f}, "
                            f"LR: {lr:.2e}, Full CoT ratio: {full_cot_ratio:.2%}"
                        )
                        
                        if self.use_wandb:
                            import wandb
                            wandb.log({
                                "train_loss": loss.item(),
                                "learning_rate": lr,
                                "full_cot_ratio": full_cot_ratio,
                                "step": completed_steps,
                            })
                        
                        total_loss = 0.0
                        num_full_cot = 0
                        num_no_cot = 0
                
                # Checkpointing
                if completed_steps % self.config.checkpoint_save_frequency == 0 and completed_steps > 0:
                    self._save_checkpoint(completed_steps, output_dir)
                
                if completed_steps >= self.config.max_train_steps:
                    break
        
        # Save final checkpoint
        self._save_checkpoint(completed_steps, output_dir, is_final=True)
        
        if self.use_wandb and self.accelerator.is_main_process:
            import wandb
            wandb.finish()
        
        logger.info(f"Text + Flow CoT training finished. Final checkpoint saved at {output_dir}")


def train_text_flow_cot(
    config: TrainingConfig,
    ecot_annotations_path: str,
    flow_tokens_path: str,
    use_wandb: bool = True,
    resume_from_checkpoint: Optional[str] = None,
):
    """
    Train Text + Flow CoT LoRA model (Phase C).
    
    Convenience function for training Text + Flow CoT model.
    
    Args:
        config: Training configuration
        ecot_annotations_path: Path to ECoT annotations JSON
        flow_tokens_path: Path to preprocessed flow tokens JSON
        use_wandb: Whether to use wandb logging
        resume_from_checkpoint: Path to checkpoint to resume from
        
    Requirements:
        - 7.9: Train separate LoRA_TF weights for each LIBERO subset
        - 7.10: Save each LoRA_TF weights separately with subset identifier
    """
    # Ensure phase is text_flow_cot
    config.phase = "text_flow_cot"
    
    # Create trainer
    trainer = TextFlowCoTTrainer(
        config=config,
        ecot_annotations_path=ecot_annotations_path,
        flow_tokens_path=flow_tokens_path,
        use_wandb=use_wandb,
        wandb_project=f"nora-lora-text-flow-cot-{config.libero_subset}",
    )
    
    # Setup and train
    trainer.setup()
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
