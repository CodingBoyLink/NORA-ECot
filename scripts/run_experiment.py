#!/usr/bin/env python
"""
Main Experiment Runner for NORA LoRA Training.

Supports end-to-end execution: Data Preparation → Training → Evaluation.
Supports specifying experiment configuration via YAML or command line.

Requirements covered:
- 0.1: Prioritize reusing existing code
- 0.6: Support end-to-end experiment execution

Usage:
    # Run full experiment for baseline on object subset
    python scripts/run_experiment.py --phase baseline --subset object
    
    # Run only training (skip data prep)
    python scripts/run_experiment.py --phase text_cot --subset spatial --skip-data-prep
    
    # Run only evaluation
    python scripts/run_experiment.py --phase text_flow_cot --subset goal --eval-only
    
    # Run with custom config
    python scripts/run_experiment.py --config configs/baseline.yaml --subset object
    
    # Run all phases for a subset
    python scripts/run_experiment.py --all-phases --subset object
    
    # Run all subsets for a phase
    python scripts/run_experiment.py --phase baseline --all-subsets
"""
import argparse
import logging
import os
import sys
import subprocess
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# Constants
PHASES = ["baseline", "text_cot", "text_flow_cot"]
SUBSETS = ["spatial", "object", "goal", "long"]


def setup_logging(log_dir: str, experiment_name: str) -> logging.Logger:
    """Setup logging for the experiment."""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{experiment_name}_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to: {log_file}")
    return logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run NORA LoRA training experiments end-to-end",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full pipeline for baseline on object subset
    python scripts/run_experiment.py --phase baseline --subset object
    
    # Text CoT with custom ECoT annotations
    python scripts/run_experiment.py --phase text_cot --subset spatial \\
        --ecot-annotations ./annotations/spatial_ecot.json
    
    # Evaluate only (model already trained)
    python scripts/run_experiment.py --phase baseline --subset object --eval-only
    
    # Run all 12 experiments (3 phases × 4 subsets)
    python scripts/run_experiment.py --all-phases --all-subsets
"""
    )
    
    # Experiment selection
    parser.add_argument(
        "--phase",
        type=str,
        choices=PHASES,
        help="Training phase: baseline, text_cot, or text_flow_cot"
    )
    parser.add_argument(
        "--subset",
        type=str,
        choices=SUBSETS,
        help="LIBERO subset: spatial, object, goal, or long"
    )
    parser.add_argument(
        "--all-phases",
        action="store_true",
        help="Run all three phases"
    )
    parser.add_argument(
        "--all-subsets",
        action="store_true",
        help="Run all four subsets"
    )
    
    # Config
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file"
    )
    
    # Pipeline control
    parser.add_argument(
        "--skip-data-prep",
        action="store_true",
        help="Skip data preparation step"
    )
    parser.add_argument(
        "--skip-ecot-generation",
        action="store_true",
        help="Skip ECoT annotation generation (for text_cot and text_flow_cot)"
    )
    parser.add_argument(
        "--skip-flow-preprocessing",
        action="store_true",
        help="Skip flow preprocessing (for text_flow_cot)"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training step"
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip evaluation step"
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run evaluation only (skip data prep and training)"
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Run training only (skip data prep and evaluation)"
    )
    
    # Data paths
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="LIBERO data directory"
    )
    parser.add_argument(
        "--ecot-annotations",
        type=str,
        default=None,
        help="Path to ECoT annotations (auto-generated if not provided)"
    )
    parser.add_argument(
        "--flow-tokens",
        type=str,
        default=None,
        help="Path to flow tokens (auto-generated if not provided)"
    )
    
    # LLM API for ECoT generation
    parser.add_argument(
        "--llm-api-url",
        type=str,
        default="",
        help="LLM API URL for ECoT generation"
    )
    parser.add_argument(
        "--llm-api-key",
        type=str,
        default="",
        help="LLM API key (or set OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-4",
        help="LLM model name for ECoT generation"
    )
    
    # Training parameters
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=None,
        help="Override max training steps"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override learning rate"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size"
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=None,
        help="Override LoRA rank"
    )
    
    # Evaluation parameters
    parser.add_argument(
        "--num-eval-trials",
        type=int,
        default=50,
        help="Number of evaluation trials per task"
    )
    parser.add_argument(
        "--eval-debug",
        action="store_true",
        help="Enable evaluation debug mode"
    )
    parser.add_argument(
        "--save-eval-videos",
        action="store_true",
        help="Save evaluation rollout videos"
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Output directory for models and results"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs",
        help="Directory for experiment logs"
    )
    
    # Misc
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume training from checkpoint"
    )
    
    return parser.parse_args()


def run_command(cmd: List[str], logger: logging.Logger, dry_run: bool = False) -> int:
    """Run a command and log output."""
    cmd_str = " ".join(cmd)
    logger.info(f"Running: {cmd_str}")
    
    if dry_run:
        logger.info("[DRY RUN] Command not executed")
        return 0
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            text=True,
            capture_output=False
        )
        return result.returncode
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with return code {e.returncode}")
        return e.returncode


def get_ecot_path(subset: str, output_dir: str) -> str:
    """Get default ECoT annotations path."""
    return os.path.join(output_dir, "annotations", f"{subset}_ecot.json")


def get_flow_tokens_path(subset: str, output_dir: str) -> str:
    """Get default flow tokens path."""
    return os.path.join(output_dir, "flow_tokens", f"{subset}_flow.json")


def get_lora_path(phase: str, subset: str, output_dir: str) -> str:
    """Get LoRA weights path."""
    return os.path.join(output_dir, phase, subset, "checkpoint-final")


def run_ecot_generation(
    subset: str,
    args,
    logger: logging.Logger
) -> str:
    """Run ECoT annotation generation."""
    logger.info(f"=" * 60)
    logger.info(f"Generating ECoT annotations for {subset}")
    logger.info(f"=" * 60)
    
    output_path = get_ecot_path(subset, args.output_dir)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    cmd = [
        sys.executable, "scripts/generate_ecot.py",
        "--subset", subset,
        "--data-dir", args.data_dir,
        "--output", output_path,
        "--model", args.llm_model,
    ]
    
    if args.llm_api_url:
        cmd.extend(["--api-url", args.llm_api_url])
    if args.llm_api_key:
        cmd.extend(["--api-key", args.llm_api_key])
    
    ret = run_command(cmd, logger, args.dry_run)
    if ret != 0:
        raise RuntimeError(f"ECoT generation failed for {subset}")
    
    return output_path


def run_flow_preprocessing(
    subset: str,
    args,
    logger: logging.Logger
) -> str:
    """Run flow preprocessing."""
    logger.info(f"=" * 60)
    logger.info(f"Preprocessing optical flow for {subset}")
    logger.info(f"=" * 60)
    
    output_path = get_flow_tokens_path(subset, args.output_dir)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    cmd = [
        sys.executable, "scripts/preprocess_flow.py",
        "--subset", subset,
        "--data-dir", args.data_dir,
        "--output", output_path,
    ]
    
    ret = run_command(cmd, logger, args.dry_run)
    if ret != 0:
        raise RuntimeError(f"Flow preprocessing failed for {subset}")
    
    return output_path


def run_training(
    phase: str,
    subset: str,
    args,
    logger: logging.Logger,
    ecot_path: Optional[str] = None,
    flow_tokens_path: Optional[str] = None
) -> str:
    """Run training for a specific phase and subset."""
    logger.info(f"=" * 60)
    logger.info(f"Training {phase} on {subset}")
    logger.info(f"=" * 60)
    
    # Select training script
    script_map = {
        "baseline": "scripts/train_baseline.py",
        "text_cot": "scripts/train_text_cot.py",
        "text_flow_cot": "scripts/train_text_flow_cot.py",
    }
    script = script_map[phase]
    
    # Build command
    cmd = [
        sys.executable, script,
        "--subset", subset,
        "--data_dir", args.data_dir,
        "--output_dir", args.output_dir,
        "--seed", str(args.seed),
    ]
    
    # Add config if provided
    if args.config:
        cmd.extend(["--config", args.config])
    
    # Add ECoT annotations for text_cot and text_flow_cot
    if phase in ["text_cot", "text_flow_cot"] and ecot_path:
        cmd.extend(["--ecot_annotations", ecot_path])
    
    # Add flow tokens for text_flow_cot
    if phase == "text_flow_cot" and flow_tokens_path:
        cmd.extend(["--flow_tokens", flow_tokens_path])
    
    # Override parameters if specified
    if args.max_train_steps:
        cmd.extend(["--max_train_steps", str(args.max_train_steps)])
    if args.learning_rate:
        cmd.extend(["--learning_rate", str(args.learning_rate)])
    if args.batch_size:
        cmd.extend(["--batch_size", str(args.batch_size)])
    if args.lora_rank:
        cmd.extend(["--lora_rank", str(args.lora_rank)])
    
    # Resume from checkpoint
    if args.resume:
        cmd.extend(["--resume", args.resume])
    
    # Wandb
    if args.no_wandb:
        cmd.append("--no_wandb")
    
    ret = run_command(cmd, logger, args.dry_run)
    if ret != 0:
        raise RuntimeError(f"Training failed for {phase}/{subset}")
    
    return get_lora_path(phase, subset, args.output_dir)


def run_evaluation(
    phase: str,
    subset: str,
    lora_path: str,
    args,
    logger: logging.Logger
) -> Dict[str, Any]:
    """Run evaluation for a trained model."""
    logger.info(f"=" * 60)
    logger.info(f"Evaluating {phase} on {subset}")
    logger.info(f"=" * 60)
    
    cmd = [
        sys.executable, "scripts/evaluate.py",
        "--model_type", phase,
        "--libero_subset", subset,
        "--lora_path", lora_path,
        "--num_trials", str(args.num_eval_trials),
        "--output_dir", os.path.join(args.output_dir, "eval_results"),
        "--seed", str(args.seed),
    ]
    
    if args.eval_debug:
        cmd.append("--debug_mode")
    if args.save_eval_videos:
        cmd.append("--save_videos")
    if not args.no_wandb:
        cmd.append("--use_wandb")
    
    ret = run_command(cmd, logger, args.dry_run)
    if ret != 0:
        raise RuntimeError(f"Evaluation failed for {phase}/{subset}")
    
    return {"phase": phase, "subset": subset, "lora_path": lora_path}



def run_single_experiment(
    phase: str,
    subset: str,
    args,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Run a single experiment (one phase, one subset).
    
    Pipeline:
    1. Data preparation (ECoT generation, flow preprocessing)
    2. Training
    3. Evaluation
    """
    results = {
        "phase": phase,
        "subset": subset,
        "status": "started",
        "ecot_path": None,
        "flow_tokens_path": None,
        "lora_path": None,
        "eval_results": None,
    }
    
    try:
        # Determine what data prep is needed
        needs_ecot = phase in ["text_cot", "text_flow_cot"]
        needs_flow = phase == "text_flow_cot"
        
        # Get paths (use provided or default)
        ecot_path = args.ecot_annotations or get_ecot_path(subset, args.output_dir)
        flow_tokens_path = args.flow_tokens or get_flow_tokens_path(subset, args.output_dir)
        
        # Step 1: Data Preparation
        if not args.skip_data_prep and not args.eval_only and not args.train_only:
            # Generate ECoT annotations if needed
            if needs_ecot and not args.skip_ecot_generation:
                if not os.path.exists(ecot_path) or not args.ecot_annotations:
                    ecot_path = run_ecot_generation(subset, args, logger)
                else:
                    logger.info(f"Using existing ECoT annotations: {ecot_path}")
            
            # Preprocess flow if needed
            if needs_flow and not args.skip_flow_preprocessing:
                if not os.path.exists(flow_tokens_path) or not args.flow_tokens:
                    flow_tokens_path = run_flow_preprocessing(subset, args, logger)
                else:
                    logger.info(f"Using existing flow tokens: {flow_tokens_path}")
        
        results["ecot_path"] = ecot_path if needs_ecot else None
        results["flow_tokens_path"] = flow_tokens_path if needs_flow else None
        
        # Step 2: Training
        lora_path = get_lora_path(phase, subset, args.output_dir)
        
        if not args.skip_training and not args.eval_only:
            lora_path = run_training(
                phase=phase,
                subset=subset,
                args=args,
                logger=logger,
                ecot_path=ecot_path if needs_ecot else None,
                flow_tokens_path=flow_tokens_path if needs_flow else None
            )
        else:
            logger.info(f"Skipping training, using existing model: {lora_path}")
        
        results["lora_path"] = lora_path
        
        # Step 3: Evaluation
        if not args.skip_evaluation and not args.train_only:
            if os.path.exists(lora_path) or args.dry_run:
                eval_results = run_evaluation(
                    phase=phase,
                    subset=subset,
                    lora_path=lora_path,
                    args=args,
                    logger=logger
                )
                results["eval_results"] = eval_results
            else:
                logger.warning(f"LoRA path not found, skipping evaluation: {lora_path}")
        
        results["status"] = "completed"
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        results["status"] = "failed"
        results["error"] = str(e)
        raise
    
    return results


def run_experiments(args, logger: logging.Logger) -> List[Dict[str, Any]]:
    """Run all requested experiments."""
    # Determine phases and subsets to run
    phases = PHASES if args.all_phases else ([args.phase] if args.phase else [])
    subsets = SUBSETS if args.all_subsets else ([args.subset] if args.subset else [])
    
    if not phases:
        raise ValueError("No phase specified. Use --phase or --all-phases")
    if not subsets:
        raise ValueError("No subset specified. Use --subset or --all-subsets")
    
    total_experiments = len(phases) * len(subsets)
    logger.info(f"Running {total_experiments} experiment(s): {len(phases)} phase(s) × {len(subsets)} subset(s)")
    
    all_results = []
    
    for phase in phases:
        for subset in subsets:
            logger.info(f"\n{'#' * 60}")
            logger.info(f"# Experiment: {phase} / {subset}")
            logger.info(f"{'#' * 60}\n")
            
            try:
                results = run_single_experiment(phase, subset, args, logger)
                all_results.append(results)
            except Exception as e:
                logger.error(f"Experiment {phase}/{subset} failed: {e}")
                all_results.append({
                    "phase": phase,
                    "subset": subset,
                    "status": "failed",
                    "error": str(e)
                })
                # Continue with other experiments
                continue
    
    return all_results


def print_summary(results: List[Dict[str, Any]], logger: logging.Logger):
    """Print experiment summary."""
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 60)
    
    completed = [r for r in results if r["status"] == "completed"]
    failed = [r for r in results if r["status"] == "failed"]
    
    logger.info(f"Total: {len(results)}")
    logger.info(f"Completed: {len(completed)}")
    logger.info(f"Failed: {len(failed)}")
    
    if completed:
        logger.info("\nCompleted experiments:")
        for r in completed:
            logger.info(f"  ✓ {r['phase']}/{r['subset']}")
            if r.get("lora_path"):
                logger.info(f"    LoRA: {r['lora_path']}")
    
    if failed:
        logger.info("\nFailed experiments:")
        for r in failed:
            logger.info(f"  ✗ {r['phase']}/{r['subset']}: {r.get('error', 'Unknown error')}")
    
    logger.info("=" * 60)


def save_results(results: List[Dict[str, Any]], output_dir: str, logger: logging.Logger):
    """Save experiment results to JSON."""
    results_dir = os.path.join(output_dir, "experiment_results")
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"results_{timestamp}.json")
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to: {results_file}")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Validate arguments
    if not args.phase and not args.all_phases:
        print("Error: Must specify --phase or --all-phases")
        sys.exit(1)
    if not args.subset and not args.all_subsets:
        print("Error: Must specify --subset or --all-subsets")
        sys.exit(1)
    
    # Setup logging
    experiment_name = f"{args.phase or 'all'}_{args.subset or 'all'}"
    logger = setup_logging(args.log_dir, experiment_name)
    
    # Log configuration
    logger.info("=" * 60)
    logger.info("NORA LoRA Experiment Runner")
    logger.info("=" * 60)
    logger.info(f"Phase(s): {PHASES if args.all_phases else args.phase}")
    logger.info(f"Subset(s): {SUBSETS if args.all_subsets else args.subset}")
    logger.info(f"Data dir: {args.data_dir}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info(f"Skip data prep: {args.skip_data_prep}")
    logger.info(f"Skip training: {args.skip_training}")
    logger.info(f"Skip evaluation: {args.skip_evaluation}")
    logger.info("=" * 60)
    
    try:
        # Run experiments
        results = run_experiments(args, logger)
        
        # Print and save summary
        print_summary(results, logger)
        save_results(results, args.output_dir, logger)
        
        # Exit with error if any experiment failed
        failed = [r for r in results if r["status"] == "failed"]
        if failed:
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
