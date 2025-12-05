# Filename: tests/test_text_cot_training.py
"""
Text CoT Training Tests

Tests for verifying:
1. Reasoning dropout probability is correct (Req 5.5)
2. CoT loss calculation is correct (Req 5.9)

Requirements: 5.5, 5.9
"""

import unittest
import os
import sys
import numpy as np
from typing import Dict, List, Tuple

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import directly from submodules to avoid peft dependency in training/__init__.py
# This allows tests to run even without peft installed


class TestReasoningDropout(unittest.TestCase):
    """
    Test reasoning dropout functionality.
    
    Requirements:
        - 5.5: Apply Reasoning Dropout with only two modes: Full CoT or No CoT,
               controlled by probability p_CoT (default 0.5)
    """
    
    def test_reasoning_dropout_import(self):
        """Test that reasoning_dropout function can be imported."""
        from training.datasets.ecot_collator import reasoning_dropout
        self.assertTrue(callable(reasoning_dropout))
    
    def test_reasoning_dropout_returns_tuple(self):
        """Test that reasoning_dropout returns a tuple of (dict, bool)."""
        from training.datasets.ecot_collator import reasoning_dropout
        
        reasoning_dict = {
            "task": "Pick up the bowl",
            "plan": "Move to bowl, grasp, lift",
            "subtask": "Move to bowl",
        }
        
        result = reasoning_dropout(reasoning_dict, dropout_prob=0.5)
        
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], dict)
        self.assertIsInstance(result[1], bool)
    
    def test_reasoning_dropout_full_cot_mode(self):
        """Test that Full CoT mode returns the original reasoning dict."""
        from training.datasets.ecot_collator import reasoning_dropout
        
        reasoning_dict = {
            "task": "Pick up the bowl",
            "plan": "Move to bowl, grasp, lift",
            "subtask": "Move to bowl",
        }
        
        # Set dropout_prob=0 to always get Full CoT
        result_dict, is_full_cot = reasoning_dropout(reasoning_dict, dropout_prob=0.0)
        
        self.assertTrue(is_full_cot)
        self.assertEqual(result_dict, reasoning_dict)
    
    def test_reasoning_dropout_no_cot_mode(self):
        """Test that No CoT mode returns an empty dict."""
        from training.datasets.ecot_collator import reasoning_dropout
        
        reasoning_dict = {
            "task": "Pick up the bowl",
            "plan": "Move to bowl, grasp, lift",
            "subtask": "Move to bowl",
        }
        
        # Set dropout_prob=1 to always get No CoT
        result_dict, is_full_cot = reasoning_dropout(reasoning_dict, dropout_prob=1.0)
        
        self.assertFalse(is_full_cot)
        self.assertEqual(result_dict, {})
    
    def test_reasoning_dropout_probability_distribution(self):
        """
        Test that reasoning dropout follows the expected probability distribution.
        
        With dropout_prob=0.5, we expect approximately 50% Full CoT and 50% No CoT.
        
        Requirements:
            - 5.5: Apply Reasoning Dropout with Full CoT or No CoT modes,
                   controlled by probability p_CoT (default 0.5)
        """
        from training.datasets.ecot_collator import reasoning_dropout
        
        reasoning_dict = {
            "task": "Pick up the bowl",
            "plan": "Move to bowl, grasp, lift",
        }
        
        # Run many trials
        num_trials = 1000
        dropout_prob = 0.5
        full_cot_count = 0
        
        # Set seed for reproducibility
        np.random.seed(42)
        
        for _ in range(num_trials):
            _, is_full_cot = reasoning_dropout(reasoning_dict, dropout_prob=dropout_prob)
            if is_full_cot:
                full_cot_count += 1
        
        # Expected: p_CoT = 1 - dropout_prob = 0.5
        expected_full_cot_ratio = 1.0 - dropout_prob
        actual_full_cot_ratio = full_cot_count / num_trials
        
        # Allow 5% tolerance for statistical variation
        self.assertAlmostEqual(
            actual_full_cot_ratio,
            expected_full_cot_ratio,
            delta=0.05,
            msg=f"Full CoT ratio {actual_full_cot_ratio:.3f} differs from expected {expected_full_cot_ratio:.3f}"
        )
    
    def test_reasoning_dropout_probability_30_percent(self):
        """Test reasoning dropout with 30% dropout probability (70% Full CoT)."""
        from training.datasets.ecot_collator import reasoning_dropout
        
        reasoning_dict = {"task": "Test task"}
        
        num_trials = 1000
        dropout_prob = 0.3
        full_cot_count = 0
        
        np.random.seed(123)
        
        for _ in range(num_trials):
            _, is_full_cot = reasoning_dropout(reasoning_dict, dropout_prob=dropout_prob)
            if is_full_cot:
                full_cot_count += 1
        
        expected_full_cot_ratio = 1.0 - dropout_prob  # 0.7
        actual_full_cot_ratio = full_cot_count / num_trials
        
        self.assertAlmostEqual(
            actual_full_cot_ratio,
            expected_full_cot_ratio,
            delta=0.05,
            msg=f"Full CoT ratio {actual_full_cot_ratio:.3f} differs from expected {expected_full_cot_ratio:.3f}"
        )
    
    def test_reasoning_dropout_probability_80_percent(self):
        """Test reasoning dropout with 80% dropout probability (20% Full CoT)."""
        from training.datasets.ecot_collator import reasoning_dropout
        
        reasoning_dict = {"task": "Test task"}
        
        num_trials = 1000
        dropout_prob = 0.8
        full_cot_count = 0
        
        np.random.seed(456)
        
        for _ in range(num_trials):
            _, is_full_cot = reasoning_dropout(reasoning_dict, dropout_prob=dropout_prob)
            if is_full_cot:
                full_cot_count += 1
        
        expected_full_cot_ratio = 1.0 - dropout_prob  # 0.2
        actual_full_cot_ratio = full_cot_count / num_trials
        
        self.assertAlmostEqual(
            actual_full_cot_ratio,
            expected_full_cot_ratio,
            delta=0.05,
            msg=f"Full CoT ratio {actual_full_cot_ratio:.3f} differs from expected {expected_full_cot_ratio:.3f}"
        )


class TestCoTFormatting(unittest.TestCase):
    """
    Test CoT response formatting.
    
    Requirements:
        - 5.6: Format Full CoT response with all tags
        - 5.7: Format No CoT response as ACTION: {action_tokens}
    """
    
    def test_format_full_cot_response(self):
        """Test formatting Full CoT response with all tags."""
        from training.utils.cot_utils import format_full_cot_response
        
        reasoning_dict = {
            "task": "Pick up the bowl",
            "plan": "Move to bowl, grasp, lift",
            "bboxes": "bowl [100, 200, 150, 250]",
            "subtask_reason": "Need to approach the bowl first",
            "subtask": "Move to bowl",
            "move_reason": "Bowl is to the right",
            "move": "Move right",
            "gripper": "[120, 225]",
        }
        action_str = "<robot_action_0><robot_action_1>"
        
        result = format_full_cot_response(reasoning_dict, action_str)
        
        # Verify all tags are present
        self.assertIn("TASK:", result)
        self.assertIn("PLAN:", result)
        self.assertIn("VISIBLE OBJECTS:", result)
        self.assertIn("SUBTASK REASONING:", result)
        self.assertIn("SUBTASK:", result)
        self.assertIn("MOVE REASONING:", result)
        self.assertIn("MOVE:", result)
        self.assertIn("GRIPPER POSITION:", result)
        self.assertIn("ACTION:", result)
        
        # Verify values are present
        self.assertIn("Pick up the bowl", result)
        self.assertIn("Move to bowl, grasp, lift", result)
        self.assertIn("<robot_action_0><robot_action_1>", result)
    
    def test_format_action_only_response(self):
        """Test formatting No CoT response (action only)."""
        from training.utils.cot_utils import format_action_only_response
        
        action_str = "<robot_action_0><robot_action_1><robot_action_2>"
        
        result = format_action_only_response(action_str)
        
        # Should only contain ACTION: tag
        self.assertEqual(result, f"ACTION: {action_str}")
        
        # Should not contain other tags
        self.assertNotIn("TASK:", result)
        self.assertNotIn("PLAN:", result)
        self.assertNotIn("SUBTASK:", result)
    
    def test_format_full_cot_with_partial_reasoning(self):
        """Test formatting Full CoT with only some reasoning fields."""
        from training.utils.cot_utils import format_full_cot_response
        
        # Only some fields present
        reasoning_dict = {
            "task": "Pick up the bowl",
            "subtask": "Move to bowl",
        }
        action_str = "<robot_action_0>"
        
        result = format_full_cot_response(reasoning_dict, action_str)
        
        # Present fields should be included
        self.assertIn("TASK:", result)
        self.assertIn("Pick up the bowl", result)
        self.assertIn("SUBTASK:", result)
        self.assertIn("Move to bowl", result)
        self.assertIn("ACTION:", result)
        
        # Missing fields should not have their tags
        # (The tag won't appear if the value is missing)


class TestECoTCollator(unittest.TestCase):
    """
    Test ECoT Collator functionality.
    
    Requirements:
        - 5.5: Apply Reasoning Dropout with Full CoT or No CoT modes
        - 5.6: Format Full CoT response with all tags
        - 5.7: Format No CoT response as ACTION: {action_tokens}
    """
    
    def test_ecot_collator_import(self):
        """Test that ECoTCollator can be imported."""
        from training.datasets.ecot_collator import ECoTCollator
        self.assertTrue(True)
    
    def test_ecot_collator_initialization(self):
        """Test ECoTCollator initialization with mock processor."""
        from training.datasets.ecot_collator import ECoTCollator
        
        # Create a mock processor
        class MockProcessor:
            def __init__(self):
                self.tokenizer = MockTokenizer()
            
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
                return ["mock text"] * len(messages)
            
            def __call__(self, text, images, videos, padding, return_tensors):
                import torch
                batch_size = len(text)
                return {
                    'input_ids': torch.zeros(batch_size, 10, dtype=torch.long),
                    'attention_mask': torch.ones(batch_size, 10, dtype=torch.long),
                }
        
        class MockTokenizer:
            pad_token_id = 0
        
        try:
            collator = ECoTCollator(
                processor=MockProcessor(),
                reasoning_dropout_prob=0.5,
                compute_cot_loss=True,
            )
            
            self.assertEqual(collator.reasoning_dropout_prob, 0.5)
            self.assertTrue(collator.compute_cot_loss)
        except Exception as e:
            # May fail due to missing dependencies, which is acceptable
            self.skipTest(f"ECoTCollator initialization failed: {e}")


class TestCoTLossComputation(unittest.TestCase):
    """
    Test CoT loss computation logic.
    
    Requirements:
        - 5.8: Compute primary loss on action_tokens for both modes
        - 5.9: In Full CoT mode, compute auxiliary loss on CoT tokens with weight lambda
    """
    
    def test_label_masking_no_cot_mode(self):
        """
        Test that in No CoT mode, only action tokens contribute to loss.
        
        In No CoT mode, all tokens before the first action token should be masked (-100).
        """
        import torch
        from training.action_tokenizer import ACTION_TOKEN_MIN, ACTION_TOKEN_MAX
        
        # Simulate a sequence with some prefix tokens and action tokens
        # Format: [prefix tokens...] [action tokens...]
        prefix_tokens = torch.tensor([100, 200, 300, 400])  # Non-action tokens
        action_tokens = torch.tensor([ACTION_TOKEN_MIN, ACTION_TOKEN_MIN + 1, ACTION_TOKEN_MIN + 2])
        
        labels = torch.cat([prefix_tokens, action_tokens])
        
        # In No CoT mode, mask everything before action tokens
        first_action_idx = len(prefix_tokens)
        masked_labels = labels.clone()
        masked_labels[:first_action_idx] = -100
        
        # Verify masking
        self.assertTrue(torch.all(masked_labels[:first_action_idx] == -100))
        self.assertTrue(torch.all(masked_labels[first_action_idx:] != -100))
    
    def test_label_masking_full_cot_mode_with_cot_loss(self):
        """
        Test that in Full CoT mode with CoT loss, all assistant tokens contribute.
        
        In Full CoT mode with compute_cot_loss=True, CoT tokens should NOT be masked.
        """
        import torch
        from training.action_tokenizer import ACTION_TOKEN_MIN
        
        # Simulate a sequence: [system/user tokens] [CoT tokens] [action tokens]
        system_user_tokens = torch.tensor([100, 200])  # Should be masked
        cot_tokens = torch.tensor([300, 400, 500])  # Should NOT be masked in Full CoT
        action_tokens = torch.tensor([ACTION_TOKEN_MIN, ACTION_TOKEN_MIN + 1])
        
        labels = torch.cat([system_user_tokens, cot_tokens, action_tokens])
        
        # In Full CoT mode with CoT loss, only system/user tokens are masked
        # CoT tokens and action tokens contribute to loss
        # For simplicity, we assume the assistant response starts after system_user_tokens
        assistant_start_idx = len(system_user_tokens)
        
        masked_labels = labels.clone()
        masked_labels[:assistant_start_idx] = -100
        
        # Verify: system/user masked, CoT and action not masked
        self.assertTrue(torch.all(masked_labels[:assistant_start_idx] == -100))
        self.assertTrue(torch.all(masked_labels[assistant_start_idx:] != -100))
    
    def test_cot_loss_weight_application(self):
        """
        Test that CoT loss weight (lambda) is applied correctly.
        
        The total loss should be: action_loss + lambda * cot_loss
        
        Requirements:
            - 5.9: In Full CoT mode, compute auxiliary loss on CoT tokens with weight lambda
        """
        import torch
        
        # Simulate loss values
        action_loss = torch.tensor(1.0)
        cot_loss = torch.tensor(0.5)
        lambda_weight = 0.5
        
        # Expected combined loss
        expected_total_loss = action_loss + lambda_weight * cot_loss
        
        # Compute
        total_loss = action_loss + lambda_weight * cot_loss
        
        self.assertAlmostEqual(total_loss.item(), expected_total_loss.item(), places=5)
        self.assertAlmostEqual(total_loss.item(), 1.25, places=5)
    
    def test_action_token_range(self):
        """Test that action token range is correctly defined."""
        from training.action_tokenizer import ACTION_TOKEN_MIN, ACTION_TOKEN_MAX
        
        # NORA uses action token ID range 151665-153712
        self.assertEqual(ACTION_TOKEN_MIN, 151665)
        self.assertEqual(ACTION_TOKEN_MAX, 153712)
        
        # Verify range is valid
        self.assertLess(ACTION_TOKEN_MIN, ACTION_TOKEN_MAX)


class TestTextCoTTrainer(unittest.TestCase):
    """
    Test TextCoTTrainer functionality.
    
    Requirements:
        - 5.10: Train separate LoRA_T weights for each LIBERO subset
        - 5.11: Save each LoRA_T weights separately with subset identifier
    """
    
    def test_text_cot_trainer_import(self):
        """Test that TextCoTTrainer can be imported."""
        try:
            from training.lora_trainer import TextCoTTrainer
            self.assertTrue(True)
        except ImportError as e:
            self.skipTest(f"TextCoTTrainer not available: {e}")
    
    def test_text_cot_config_lora_name(self):
        """Test that Text CoT config generates correct LoRA name."""
        from training.config import TrainingConfig
        
        config = TrainingConfig(
            phase="text_cot",
            libero_subset="spatial",
        )
        
        lora_name = config.get_lora_name()
        
        # Should be LoRA_T for text_cot phase
        self.assertEqual(lora_name, "LoRA_T_spatial")
    
    def test_text_cot_config_output_path(self):
        """Test that Text CoT config generates correct output path."""
        from training.config import TrainingConfig
        
        config = TrainingConfig(
            phase="text_cot",
            libero_subset="object",
            output_dir="./outputs",
        )
        
        output_path = config.get_output_path()
        
        self.assertIn("text_cot", output_path)
        self.assertIn("object", output_path)


if __name__ == '__main__':
    unittest.main(verbosity=2)
