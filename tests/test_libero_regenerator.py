#!/usr/bin/env python
"""
Unit tests for is_noop() function logic used in LiberoDataRegenerator.

Requirements covered:
- 2.1: Identify no-op using norm threshold for dimensions 0-5
- 2.2: Check gripper action equals previous timestep
"""

import os
import sys
import unittest
from typing import Optional

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def is_noop(
    action: np.ndarray,
    prev_action: Optional[np.ndarray] = None,
    noop_threshold: float = 1e-4
) -> bool:
    """
    Standalone is_noop function for testing without robosuite dependency.
    
    This mirrors the logic in LiberoDataRegenerator.is_noop().
    """
    # Criterion 1: Check if movement dimensions (0-5) are near zero
    movement_norm = np.linalg.norm(action[:-1])
    movement_is_zero = movement_norm < noop_threshold
    
    # Special case: First action in episode (prev_action is None)
    if prev_action is None:
        return movement_is_zero
    
    # Criterion 2: Check if gripper action equals previous gripper action
    gripper_action = action[-1]
    prev_gripper_action = prev_action[-1]
    gripper_unchanged = gripper_action == prev_gripper_action
    
    return movement_is_zero and gripper_unchanged


class TestIsNoop(unittest.TestCase):
    """Test is_noop() function for no-op action detection."""
    
    def setUp(self):
        """Set up test parameters."""
        self.noop_threshold = 1e-4
    
    def test_zero_action_first_step_is_noop(self):
        """Test that zero action on first step (no prev_action) is detected as no-op."""
        # Requirement 2.1: All dimensions 0-5 have norm < threshold
        action = np.zeros(7, dtype=np.float32)
        result = is_noop(action, prev_action=None, noop_threshold=self.noop_threshold)
        self.assertTrue(result, "Zero action on first step should be no-op")
    
    def test_significant_movement_is_not_noop(self):
        """Test that significant movement is not detected as no-op."""
        action = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        result = is_noop(action, prev_action=None, noop_threshold=self.noop_threshold)
        self.assertFalse(result, "Significant movement should not be no-op")
    
    def test_zero_movement_with_gripper_change_is_not_noop(self):
        """Test that zero movement with gripper change is not no-op."""
        # Requirement 2.2: Gripper action must equal previous for no-op
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        prev_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        result = is_noop(action, prev_action, noop_threshold=self.noop_threshold)
        self.assertFalse(result, "Zero movement with gripper change should not be no-op")
    
    def test_zero_movement_with_same_gripper_is_noop(self):
        """Test that zero movement with same gripper state is no-op."""
        # Requirements 2.1 + 2.2: Both criteria satisfied
        action = np.zeros(7, dtype=np.float32)
        action[-1] = 1.0
        prev_action = np.zeros(7, dtype=np.float32)
        prev_action[-1] = 1.0
        result = is_noop(action, prev_action, noop_threshold=self.noop_threshold)
        self.assertTrue(result, "Zero movement with same gripper should be no-op")
    
    def test_tiny_movement_below_threshold_is_noop(self):
        """Test that tiny movement below threshold is detected as no-op."""
        action = np.array([1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 0.0], dtype=np.float32)
        prev_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        result = is_noop(action, prev_action, noop_threshold=self.noop_threshold)
        self.assertTrue(result, "Tiny movement below threshold should be no-op")
    
    def test_movement_at_threshold_boundary(self):
        """Test behavior at threshold boundary."""
        # Just above threshold - should NOT be no-op
        action = np.array([1e-4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        prev_action = np.zeros(7, dtype=np.float32)
        result = is_noop(action, prev_action, noop_threshold=self.noop_threshold)
        self.assertFalse(result, "Movement at threshold should not be no-op")


if __name__ == "__main__":
    unittest.main(verbosity=2)
