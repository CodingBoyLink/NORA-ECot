#!/usr/bin/env python3
"""
LIBERO Evaluation Script for NORA LoRA Training.

Usage:
    # Evaluate baseline model on libero_spatial
    python scripts/evaluate.py \
        --lora_path ./outputs/baseline/spatial/final/lora_weights \
        --libero_subset spatial \
        --model_type baseline

    # Evaluate with debug mode (generates Full CoT)
    python scripts/evaluate.py \
        --lora_path ./outputs/baseline/spatial/final/lora_weights \
        --libero_subset spatial \
        --debug_mode

    # Quick test with fewer trials
    python scripts/evaluate.py \
        --lora_path ./outputs/baseline/spatial/final/lora_weights \
        --libero_subset spatial \
        --num_trials 5

Requirements:
    - LIBERO environment installed
    - NORA base model (declare-lab/nora)
    - Trained LoRA weights
"""
import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.libero_eval import EvalConfig, LiberoEvaluator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate NORA LoRA model on LIBERO benchmark"
    )
    
    # Model configuration
    parser.add_argument(
        "--base_model",
        type=str,
        default="declare-lab/nora",
        help="Base NORA model path (default: declare-lab/nora)"
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="Path to trained LoRA weights"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="baseline",
        choices=["baseline", "text_cot", "text_flow_cot"],
        help="Model type (default: baseline)"
    )
    
    # LIBERO configuration
    parser.add_argument(
        "--libero_subset",
        type=str,
        default="spatial",
        choices=["spatial", "object", "goal", "10", "90"],
        help="LIBERO subset to evaluate (default: spatial)"
    )
    
    # Evaluation parameters
    parser.add_argument(
        "--num_trials",
        type=int,
        default=50,
        help="Number of trials per task (default: 50)"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=500,
        help="Maximum steps per episode (default: 500)"
    )
    parser.add_argument(
        "--num_steps_wait",
        type=int,
        default=10,
        help="Steps to wait for environment stabilization (default: 10)"
    )
    
    # Debug options
    parser.add_argument(
        "--debug_mode",
        action="store_true",
        help="Enable debug mode with Full CoT generation"
    )
    parser.add_argument(
        "--save_videos",
        action="store_true",
        help="Save rollout videos"
    )
    parser.add_argument(
        "--single_task",
        type=int,
        default=None,
        help="Evaluate only a single task by ID (for debugging)"
    )
    
    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_results",
        help="Output directory for results (default: ./eval_results)"
    )
    
    # Device configuration
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype (default: bfloat16)"
    )
    
    # Random seed
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed (default: 7)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create evaluation config
    config = EvalConfig(
        base_model_path=args.base_model,
        lora_path=args.lora_path,
        model_type=args.model_type,
        libero_subset=args.libero_subset,
        num_trials_per_task=args.num_trials,
        max_steps=args.max_steps,
        num_steps_wait=args.num_steps_wait,
        debug_mode=args.debug_mode,
        save_videos=args.save_videos,
        generate_full_cot=args.debug_mode,
        output_dir=args.output_dir,
        device=args.device,
        torch_dtype=args.dtype,
        seed=args.seed,
    )
    
    print("=" * 60)
    print("NORA LoRA Evaluation")
    print("=" * 60)
    print(f"Base model: {config.base_model_path}")
    print(f"LoRA path: {config.lora_path}")
    print(f"Model type: {config.model_type}")
    print(f"LIBERO subset: {config.libero_subset}")
    print(f"Trials per task: {config.num_trials_per_task}")
    print(f"Max steps: {config.max_steps}")
    print(f"Debug mode: {config.debug_mode}")
    print(f"Save videos: {config.save_videos}")
    print(f"Output dir: {config.output_dir}")
    print("=" * 60)
    
    # Create evaluator
    print("\nInitializing evaluator...")
    evaluator = LiberoEvaluator(config)
    
    # Run evaluation
    if args.single_task is not None:
        print(f"\nEvaluating single task: {args.single_task}")
        if args.debug_mode:
            results = evaluator.evaluate_with_debug(
                task_id=args.single_task,
                episode_idx=0,
            )
        else:
            results = evaluator.evaluate_single_task(args.single_task)
    else:
        print("\nStarting full evaluation...")
        results = evaluator.evaluate()
    
    # Print summary
    print("\n" + "=" * 60)
    print("Evaluation Complete")
    print("=" * 60)
    
    if "overall_success_rate" in results:
        print(f"Overall Success Rate: {results['overall_success_rate']:.2%}")
        print(f"Total Episodes: {results['total_episodes']}")
        print(f"Total Successes: {results['total_successes']}")
    elif "success_rate" in results:
        print(f"Task Success Rate: {results['success_rate']:.2%}")
        print(f"Episodes: {results['episodes']}")
        print(f"Successes: {results['successes']}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
