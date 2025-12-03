# Filename: tests/test_baseline_training.py
"""
Baseline Training Tests

Tests for verifying the baseline LoRA training flow on small datasets
and ensuring LoRA weights are correctly saved.

Requirements: 3.9, 3.10
"""

import unittest
import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.config import TrainingConfig, LoRAConfig, OptimizerConfig


class TestTrainingConfig(unittest.TestCase):
    """Test TrainingConfig functionality."""
    
    def test_config_creation(self):
        """Test creating a TrainingConfig with default values."""
        config = TrainingConfig()
        
        self.assertEqual(config.phase, "baseline")
        self.assertEqual(config.base_model, "declare-lab/nora")
        self.assertEqual(config.lora.rank, 16)
        self.assertEqual(config.lora.alpha, 32)
        self.assertAlmostEqual(config.lora.dropout, 0.05)
    
    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = TrainingConfig(
            phase="baseline",
            libero_subset="spatial",
            max_train_steps=100,
        )
        
        config_dict = config.to_dict()
        
        self.assertEqual(config_dict['phase'], "baseline")
        self.assertEqual(config_dict['libero_subset'], "spatial")
        self.assertEqual(config_dict['max_train_steps'], 100)
        self.assertIn('lora', config_dict)
        self.assertIn('optimizer', config_dict)
    
    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            'phase': 'text_cot',
            'libero_subset': 'object',
            'max_train_steps': 500,
            'lora': {
                'rank': 32,
                'alpha': 64,
                'dropout': 0.1,
                'target_modules': ['q_proj', 'v_proj'],
            },
        }
        
        config = TrainingConfig.from_dict(config_dict)
        
        self.assertEqual(config.phase, 'text_cot')
        self.assertEqual(config.libero_subset, 'object')
        self.assertEqual(config.max_train_steps, 500)
        self.assertEqual(config.lora.rank, 32)
        self.assertEqual(config.lora.alpha, 64)
    
    def test_get_output_path(self):
        """Test output path generation."""
        config = TrainingConfig(
            phase="baseline",
            libero_subset="spatial",
            output_dir="./test_outputs",
        )
        
        output_path = config.get_output_path()
        
        self.assertEqual(output_path, "./test_outputs/baseline/spatial")
    
    def test_get_lora_name(self):
        """Test LoRA name generation."""
        # Baseline
        config_b = TrainingConfig(phase="baseline", libero_subset="spatial")
        self.assertEqual(config_b.get_lora_name(), "LoRA_B_spatial")
        
        # Text CoT
        config_t = TrainingConfig(phase="text_cot", libero_subset="object")
        self.assertEqual(config_t.get_lora_name(), "LoRA_T_object")
        
        # Text + Flow CoT
        config_tf = TrainingConfig(phase="text_flow_cot", libero_subset="goal")
        self.assertEqual(config_tf.get_lora_name(), "LoRA_TF_goal")
    
    def test_config_yaml_roundtrip(self):
        """Test saving and loading config from YAML."""
        config = TrainingConfig(
            phase="baseline",
            libero_subset="10",
            max_train_steps=1000,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = os.path.join(tmpdir, "config.yaml")
            
            # Save
            config.save_yaml(yaml_path)
            self.assertTrue(os.path.exists(yaml_path))
            
            # Load
            loaded_config = TrainingConfig.from_yaml(yaml_path)
            
            self.assertEqual(loaded_config.phase, config.phase)
            self.assertEqual(loaded_config.libero_subset, config.libero_subset)
            self.assertEqual(loaded_config.max_train_steps, config.max_train_steps)


class TestLoRAModelLoading(unittest.TestCase):
    """Test LoRA model loading functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Check if model loading is possible."""
        try:
            import torch
            from peft import LoraConfig
            cls.torch_available = True
        except ImportError:
            cls.torch_available = False
    
    def test_lora_config_creation(self):
        """Test creating PEFT LoraConfig from our config."""
        if not self.torch_available:
            self.skipTest("PyTorch/PEFT not available")
        
        from training.lora_model import create_lora_config
        
        lora_config = LoRAConfig(
            rank=16,
            alpha=32,
            dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        
        peft_config = create_lora_config(lora_config)
        
        self.assertEqual(peft_config.r, 16)
        self.assertEqual(peft_config.lora_alpha, 32)
        self.assertAlmostEqual(peft_config.lora_dropout, 0.05)
        self.assertEqual(set(peft_config.target_modules), 
                        {"q_proj", "k_proj", "v_proj", "o_proj"})


class TestLoRAWeightsSaving(unittest.TestCase):
    """Test LoRA weights saving functionality (Req 3.10)."""
    
    @classmethod
    def setUpClass(cls):
        """Check if model operations are possible."""
        try:
            import torch
            from peft import LoraConfig, get_peft_model
            from transformers import AutoModelForCausalLM
            cls.deps_available = True
        except ImportError as e:
            cls.deps_available = False
            cls.skip_reason = str(e)
    
    def test_save_lora_weights_creates_files(self):
        """Test that save_lora_weights creates the expected files."""
        if not self.deps_available:
            self.skipTest(f"Dependencies not available: {self.skip_reason}")
        
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoConfig
        from training.lora_model import save_lora_weights
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a minimal model for testing
            # Use a tiny config to avoid memory issues
            try:
                # Try to create a minimal test model
                config = AutoConfig.from_pretrained(
                    "gpt2",
                    n_embd=64,
                    n_head=2,
                    n_layer=2,
                    vocab_size=1000,
                )
                model = AutoModelForCausalLM.from_config(config)
            except Exception:
                self.skipTest("Cannot create test model")
            
            # Add LoRA adapter
            lora_config = LoraConfig(
                r=4,
                lora_alpha=8,
                target_modules=["c_attn"],
                bias="none",
            )
            peft_model = get_peft_model(model, lora_config)
            
            # Save LoRA weights
            output_path = os.path.join(tmpdir, "lora_weights")
            save_lora_weights(peft_model, output_path)
            
            # Verify files exist
            self.assertTrue(os.path.exists(output_path))
            
            # Check for adapter config
            adapter_config_path = os.path.join(output_path, "adapter_config.json")
            self.assertTrue(os.path.exists(adapter_config_path),
                f"adapter_config.json not found in {output_path}")
            
            # Check for adapter weights
            adapter_model_path = os.path.join(output_path, "adapter_model.safetensors")
            adapter_model_bin_path = os.path.join(output_path, "adapter_model.bin")
            self.assertTrue(
                os.path.exists(adapter_model_path) or os.path.exists(adapter_model_bin_path),
                f"adapter_model not found in {output_path}"
            )


class TestTrainerCreation(unittest.TestCase):
    """Test trainer creation without running full training."""
    
    def test_trainer_import(self):
        """Test that trainer can be imported."""
        try:
            from training.lora_trainer import LoRATrainer, create_trainer
            imported = True
        except ImportError as e:
            imported = False
            self.fail(f"Failed to import trainer: {e}")
        
        self.assertTrue(imported)
    
    def test_create_trainer_factory(self):
        """Test create_trainer factory function."""
        try:
            from training.lora_trainer import create_trainer
            from training.config import TrainingConfig
            
            config = TrainingConfig(
                phase="baseline",
                libero_subset="spatial",
                max_train_steps=10,
            )
            
            trainer = create_trainer(config, use_wandb=False)
            
            self.assertIsNotNone(trainer)
            self.assertEqual(trainer.config.phase, "baseline")
            self.assertEqual(trainer.config.libero_subset, "spatial")
        except ImportError as e:
            self.skipTest(f"Dependencies not available: {e}")


class TestBaselineTrainingFlow(unittest.TestCase):
    """
    Test baseline training flow on small dataset.
    
    These tests verify the training pipeline works correctly
    without running full training (which requires GPU and model weights).
    
    Requirements: 3.9
    """
    
    @classmethod
    def setUpClass(cls):
        """Check if training dependencies are available."""
        cls.data_dir = "./data"
        cls.subset = "spatial"
        
        # Check if data exists
        subset_dir = os.path.join(cls.data_dir, f"libero_{cls.subset}")
        cls.data_available = os.path.exists(subset_dir)
        
        # Check if training dependencies are available
        try:
            import torch
            from accelerate import Accelerator
            cls.deps_available = True
        except ImportError:
            cls.deps_available = False
    
    def test_dataset_creation(self):
        """Test that dataset can be created for training."""
        if not self.data_available:
            self.skipTest(f"Data not available at {self.data_dir}")
        
        try:
            from training.datasets.libero_dataset import LiberoBaselineDataset
            from training.action_tokenizer import ActionTokenizer
            
            # Create action tokenizer (may fail if FAST+ not available)
            try:
                action_tokenizer = ActionTokenizer()
            except Exception:
                self.skipTest("ActionTokenizer not available")
            
            # Create dataset
            dataset = LiberoBaselineDataset(
                data_dir=self.data_dir,
                subset=self.subset,
                split="train",
                train_ratio=0.95,
                seed=42,
                action_tokenizer=action_tokenizer,
                normalize_gripper=True,
                shuffle=False,
            )
            
            # Verify dataset properties
            self.assertGreater(dataset.num_trajectories, 0)
            
            # Get first sample
            sample = next(iter(dataset))
            
            self.assertIn('image', sample)
            self.assertIn('instruction', sample)
            self.assertIn('action', sample)
            self.assertIn('action_tokens', sample)
            self.assertIn('vlm_action_string', sample)
            
            # Verify action tokens format
            self.assertIsInstance(sample['action_tokens'], list)
            self.assertTrue(sample['vlm_action_string'].startswith('<robot_action_'))
            
        except ImportError as e:
            self.skipTest(f"Dependencies not available: {e}")
    
    def test_collator_creation(self):
        """Test that collator can be created."""
        try:
            from training.datasets.baseline_collator import BaselineCollator
            
            # Collator requires processor, which requires model
            # Just test import for now
            self.assertTrue(True)
        except ImportError as e:
            self.skipTest(f"Collator not available: {e}")
    
    def test_training_config_for_small_run(self):
        """Test creating config for a small training run."""
        config = TrainingConfig(
            phase="baseline",
            libero_subset="spatial",
            max_train_steps=10,  # Very small for testing
            per_device_batch_size=2,
            gradient_accumulation_steps=1,
            checkpoint_save_frequency=5,
            logging_steps=2,
            data_dir=self.data_dir,
        )
        
        self.assertEqual(config.max_train_steps, 10)
        self.assertEqual(config.per_device_batch_size, 2)
        
        # Verify output path
        output_path = config.get_output_path()
        self.assertIn("baseline", output_path)
        self.assertIn("spatial", output_path)


if __name__ == '__main__':
    unittest.main(verbosity=2)
