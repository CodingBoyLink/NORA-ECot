#!/usr/bin/env python3
"""
Evaluation Script for NORA LoRA Models on LIBERO Benchmark.

Supports evaluation of:
- Baseline (LoRA_B): Pure behavior cloning
- Text CoT (LoRA_T): Text ECoT + Reasoning Dropout
- Text+Flow CoT (LoRA_TF): Text + Flow CoT + Reasoning Dropout

Usage:
    # Evaluate baseline model on LIBERO-Object
    python scripts/evaluate.py \
        --model_type baseline \
        --libero_subset object \
        --lora_path outputs/baseline/object/checkpoint-final

    # Evaluate text_cot model on LIBERO-Spatial
    python scripts/evaluate.py \
        --model_type text_cot \
        --libero_subset spatial \
        --lora_path outputs/text_cot/spatial/checkpoint-final

    # Evaluate with debug mode (save videos, generate CoT)
    python scripts/evaluate.py \
        --model_type text_flow_cot \
        --libero_subset goal \
        --lora_path outputs/text_flow_cot/goal/checkpoint-final \
        --debug_mode \
        --save_videos

Requirements covered:
- 8.6: Compute success rate with 50 trials per task
- 8.7: Use max_steps=500 and num_steps_wait=10
- 8.8: Evaluate each model on corresponding LIBERO subset
- 8.9: Support comparison of 12 models (3 types × 4 subsets)
"""
import argparse
import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluation.libero_eval import LiberoEvaluator, EvalConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate NORA LoRA models on LIBERO benchmark"
    )
    
    # Model configuration
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["baseline", "text_cot", "text_flow_cot"],
        help="Model type: baseline (LoRA_B), text_cot (LoRA_T), or text_flow_cot (LoRA_TF)"
    )
    parser.add_argument(
        "--libero_subset",
        type=str,
        required=True,
        choices=["spatial", "object", "goal", "10", "10", "90"],
        help="LIBERO subset to evaluate on"
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to LoRA weights (if None, uses base NORA model)"
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="declare-lab/nora",
        help="Path to base NORA model"
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
        help="Steps to wait for object stabilization (default: 10)"
    )
    
    # Debug options
    parser.add_argument(
        "--debug_mode",
        action="store_true",
        help="Enable debug mode"
    )
    parser.add_argument(
        "--save_videos",
        action="store_true",
        help="Save rollout videos"
    )
    parser.add_argument(
        "--generate_cot",
        action="store_true",
        help="Generate full CoT for visualization (debug mode)"
    )
    
    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_results",
        help="Output directory for results"
    )
    
    # Logging
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Log results to Weights & Biases"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="nora-lora-eval",
        help="W&B project name"
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity name"
    )
    
    # Device configuration
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype"
    )
    
    # Random seed
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for reproducibility"
    )
    
    # Single task evaluation (for debugging)
    parser.add_argument(
        "--task_id",
        type=int,
        default=None,
        help="Evaluate only a single task (for debugging)"
    )
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Print configuration
    print("=" * 60)
    print("NORA LoRA Evaluation")
    print("=" * 60)
    print(f"Model Type: {args.model_type}")
    print(f"LIBERO Subset: {args.libero_subset}")
    print(f"LoRA Path: {args.lora_path}")
    print(f"Base Model: {args.base_model_path}")
    print(f"Num Trials: {args.num_trials}")
    print(f"Max Steps: {args.max_steps}")
    print(f"Device: {args.device}")
    print("=" * 60)
    
    # Create output directory with model info
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_subdir = f"{args.model_type}_{args.libero_subset}_{timestamp}"
    output_dir = os.path.join(args.output_dir, output_subdir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create evaluation config
    config = EvalConfig(
        base_model_path=args.base_model_path,
        lora_path=args.lora_path,
        model_type=args.model_type,
        libero_subset=args.libero_subset,
        num_trials_per_task=args.num_trials,
        max_steps=args.max_steps,
        num_steps_wait=args.num_steps_wait,
        debug_mode=args.debug_mode,
        save_videos=args.save_videos,
        generate_full_cot=args.generate_cot,
        output_dir=output_dir,
        log_to_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        device=args.device,
        torch_dtype=args.dtype,
        seed=args.seed,
    )
    
    # Save config
    config_path = os.path.join(output_dir, "eval_config.json")
    with open(config_path, "w") as f:
        json.dump({
            "model_type": config.model_type,
            "libero_subset": config.libero_subset,
            "lora_path": config.lora_path,
            "base_model_path": config.base_model_path,
            "num_trials_per_task": config.num_trials_per_task,
            "max_steps": config.max_steps,
            "num_steps_wait": config.num_steps_wait,
            "debug_mode": config.debug_mode,
            "save_videos": config.save_videos,
            "seed": config.seed,
        }, f, indent=2)
    
    # Initialize evaluator
    print("\nInitializing evaluator...")
    evaluator = LiberoEvaluator(config)
    
    # Run evaluation
    if args.task_id is not None:
        # Single task evaluation
        print(f"\nEvaluating single task: {args.task_id}")
        results = evaluator.evaluate_single_task(args.task_id)
        print(f"\nTask Results:")
        print(f"  Task: {results['task_description']}")
        print(f"  Success Rate: {results['success_rate']:.2%}")
        print(f"  Successes: {results['successes']}/{results['episodes']}")
    else:
        # Full evaluation
        print("\nStarting full evaluation...")
        results = evaluator.evaluate()
        
        # Print summary
        print("\n" + "=" * 60)
        print("Evaluation Summary")
        print("=" * 60)
        print(f"Model Type: {args.model_type}")
        print(f"LIBERO Subset: {args.libero_subset}")
        print(f"Overall Success Rate: {results['overall_success_rate']:.2%}")
        print(f"Total Episodes: {results['total_episodes']}")
        print(f"Total Successes: {results['total_successes']}")
        print("\nPer-Task Results:")
        for task_name, task_result in results['task_results'].items():
            print(f"  {task_name}: {task_result['success_rate']:.2%}")
        print("=" * 60)
    
    print(f"\nResults saved to: {output_dir}")
    
    return results


def evaluate_all_models(
    base_model_path: str = "declare-lab/nora",
    lora_base_dir: str = "./outputs",
    output_dir: str = "./eval_results",
    num_trials: int = 50,
    device: str = "cuda",
):
    """
    Evaluate all 12 models (3 types × 4 subsets).
    
    This function is useful for running a complete comparison experiment.
    
    Args:
        base_model_path: Path to base NORA model
        lora_base_dir: Base directory containing LoRA weights
        output_dir: Output directory for results
        num_trials: Number of trials per task
        device: Device to use
        
    Returns:
        Dictionary with all results
        
    Requirements:
        - 8.9: Support comparison of 12 models
    """
    model_types = ["baseline", "text_cot", "text_flow_cot"]
    subsets = ["spatial", "object", "goal", "10"]
    
    all_results = {}
    
    for model_type in model_types:
        all_results[model_type] = {}
        
        for subset in subsets:
            print(f"\n{'='*60}")
            print(f"Evaluating {model_type} on {subset}")
            print(f"{'='*60}")
            
            # Construct LoRA path
            lora_path = os.path.join(lora_base_dir, model_type, subset, "checkpoint-final")
            
            if not os.path.exists(lora_path):
                print(f"Warning: LoRA path not found: {lora_path}")
                print("Skipping this model...")
                continue
            
            # Create config
            config = EvalConfig(
                base_model_path=base_model_path,
                lora_path=lora_path,
                model_type=model_type,
                libero_subset=subset,
                num_trials_per_task=num_trials,
                output_dir=os.path.join(output_dir, f"{model_type}_{subset}"),
                device=device,
            )
            
            # Run evaluation
            evaluator = LiberoEvaluator(config)
            results = evaluator.evaluate()
            
            all_results[model_type][subset] = results
    
    # Save combined results
    combined_path = os.path.join(output_dir, "all_results.json")
    
    # Convert to serializable format
    serializable_results = {}
    for model_type, subset_results in all_results.items():
        serializable_results[model_type] = {}
        for subset, results in subset_results.items():
            serializable_results[model_type][subset] = {
                "overall_success_rate": results.get("overall_success_rate", 0),
                "total_episodes": results.get("total_episodes", 0),
                "total_successes": results.get("total_successes", 0),
            }
    
    with open(combined_path, "w") as f:
        json.dump(serializable_results, f, indent=2)
    
    # Print summary table
    print("\n" + "=" * 80)
    print("Complete Evaluation Summary")
    print("=" * 80)
    print(f"{'Model Type':<20} {'Spatial':<12} {'Object':<12} {'Goal':<12} {'10':<12}")
    print("-" * 80)
    
    for model_type in model_types:
        row = f"{model_type:<20}"
        for subset in subsets:
            if subset in all_results.get(model_type, {}):
                sr = all_results[model_type][subset].get("overall_success_rate", 0)
                row += f"{sr:.1%}".ljust(12)
            else:
                row += "N/A".ljust(12)
        print(row)
    
    print("=" * 80)
    print(f"\nCombined results saved to: {combined_path}")
    
    return all_results


if __name__ == "__main__":
    main()
