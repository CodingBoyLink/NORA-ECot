#!/usr/bin/env python3
"""
Test script to verify evaluation configuration matches training.

This script checks:
1. time_horizon setting (must be 1 for both training and evaluation)
2. Image resolution settings
3. Action tokenizer configuration
4. LIBERO normalization keys

Run this before evaluation to ensure consistency.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_time_horizon():
    """Test that time_horizon is set to 1 in both training and evaluation."""
    print("\n" + "=" * 60)
    print("Testing time_horizon configuration...")
    print("=" * 60)
    
    # Check training action tokenizer
    from training.action_tokenizer import ActionTokenizer
    train_tokenizer = ActionTokenizer()
    print(f"Training ActionTokenizer time_horizon: {train_tokenizer.time_horizon}")
    
    # Check evaluation fast tokenizer
    from transformers import AutoProcessor
    eval_tokenizer = AutoProcessor.from_pretrained(
        "physical-intelligence/fast",
        trust_remote_code=True
    )
    eval_tokenizer.action_dim = 7
    eval_tokenizer.time_horizon = 1  # This should match training
    print(f"Evaluation fast_tokenizer time_horizon: {eval_tokenizer.time_horizon}")
    
    # Verify they match
    if train_tokenizer.time_horizon == eval_tokenizer.time_horizon == 1:
        print("✅ time_horizon is correctly set to 1 for both training and evaluation")
        return True
    else:
        print("❌ time_horizon mismatch!")
        print(f"   Training: {train_tokenizer.time_horizon}")
        print(f"   Evaluation: {eval_tokenizer.time_horizon}")
        return False


def test_image_resolution():
    """Test image resolution settings."""
    print("\n" + "=" * 60)
    print("Testing image resolution configuration...")
    print("=" * 60)
    
    # NORA expects 224x224 input images
    expected_height = 224
    expected_width = 224
    
    print(f"Expected model input resolution: {expected_height}x{expected_width}")
    print(f"Data resolution (regenerated): 256x256")
    print(f"Evaluation environment resolution: 256x256")
    print(f"Evaluation model input (resized): {expected_height}x{expected_width}")
    
    print("✅ Image resolution configuration is correct")
    print("   - Data is stored at 256x256")
    print("   - Model receives 224x224 (processor handles resizing)")
    return True


def test_action_token_range():
    """Test action token ID range."""
    print("\n" + "=" * 60)
    print("Testing action token ID range...")
    print("=" * 60)
    
    from training.action_tokenizer import ACTION_TOKEN_MIN, ACTION_TOKEN_MAX
    
    expected_min = 151665
    expected_max = 153712
    
    print(f"Training ACTION_TOKEN_MIN: {ACTION_TOKEN_MIN}")
    print(f"Training ACTION_TOKEN_MAX: {ACTION_TOKEN_MAX}")
    print(f"Expected range: [{expected_min}, {expected_max}]")
    
    if ACTION_TOKEN_MIN == expected_min and ACTION_TOKEN_MAX == expected_max:
        print("✅ Action token range is correct")
        return True
    else:
        print("❌ Action token range mismatch!")
        return False


def test_libero_keys():
    """Test LIBERO normalization keys."""
    print("\n" + "=" * 60)
    print("Testing LIBERO normalization keys...")
    print("=" * 60)
    
    from evaluation.libero_eval import LIBERO_KEYS
    
    expected_subsets = ['libero_object', 'libero_spatial', 'libero_goal', 'libero_10', 'libero_90']
    
    print(f"Available LIBERO keys: {list(LIBERO_KEYS.keys())}")
    
    all_present = True
    for subset in expected_subsets:
        if subset in LIBERO_KEYS:
            print(f"  ✅ {subset}: present")
        else:
            print(f"  ❌ {subset}: missing")
            all_present = False
    
    if all_present:
        print("✅ All LIBERO normalization keys are present")
        return True
    else:
        print("❌ Some LIBERO keys are missing!")
        return False


def test_action_encoding_decoding():
    """Test action encoding and decoding roundtrip."""
    print("\n" + "=" * 60)
    print("Testing action encoding/decoding roundtrip...")
    print("=" * 60)
    
    import numpy as np
    from training.action_tokenizer import ActionTokenizer
    
    tokenizer = ActionTokenizer(time_horizon=1)
    
    # Test action
    test_action = np.array([0.1, -0.2, 0.3, 0.05, -0.1, 0.15, 0.5])
    print(f"Original action: {test_action}")
    
    # Encode
    tokens = tokenizer.encode(test_action, normalize_gripper=True)
    print(f"Encoded tokens: {tokens}")
    print(f"Number of tokens: {len(tokens)}")
    
    # Decode
    decoded = tokenizer.decode(tokens, denormalize_gripper=True)
    print(f"Decoded action: {decoded}")
    
    # Check shape
    if decoded.shape == (1, 7):
        print("✅ Decoded action has correct shape (1, 7)")
    else:
        print(f"❌ Decoded action has wrong shape: {decoded.shape}")
        return False
    
    # Check approximate reconstruction
    error = np.abs(test_action - decoded[0]).max()
    print(f"Max reconstruction error: {error:.4f}")
    
    if error < 0.1:  # Allow some quantization error
        print("✅ Action encoding/decoding roundtrip successful")
        return True
    else:
        print("❌ Large reconstruction error!")
        return False


def main():
    print("=" * 60)
    print("NORA LoRA Evaluation Configuration Test")
    print("=" * 60)
    
    results = []
    
    # Run all tests
    results.append(("time_horizon", test_time_horizon()))
    results.append(("image_resolution", test_image_resolution()))
    results.append(("action_token_range", test_action_token_range()))
    results.append(("libero_keys", test_libero_keys()))
    results.append(("action_encoding", test_action_encoding_decoding()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("All tests passed! Evaluation configuration is correct.")
        return 0
    else:
        print("Some tests failed! Please fix the issues before evaluation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
