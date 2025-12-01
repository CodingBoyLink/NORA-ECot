#!/usr/bin/env python
"""
Regenerates a LIBERO dataset (HDF5 files) by replaying demonstrations in the environments.

Key improvements over original data:
    - Higher image resolution (256x256 instead of 128x128)
    - Filters out no-op (zero) actions that do not change the robot's state
    - Filters out unsuccessful demonstrations
    - Generates metainfo JSON for tracking episode success and initial states

Usage:
    python scripts/regenerate_libero.py \
        --libero_task_suite [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
        --libero_raw_data_dir <PATH TO RAW HDF5 DATASET DIR> \
        --libero_target_dir <PATH TO TARGET DIR> \
        [--image_resolution 256]

    Example (LIBERO-Spatial):
        python scripts/regenerate_libero.py \
            --libero_task_suite libero_spatial \
            --libero_raw_data_dir ./data/libero_spatial \
            --libero_target_dir ./data/libero_spatial_clean \
            --image_resolution 256

Requirements covered:
    - 6.1: Command-line arguments (--libero_task_suite, --libero_raw_data_dir, 
           --libero_target_dir, --image_resolution)
    - 6.2: Target directory check and overwrite confirmation
    - 6.3: Progress information and statistics display
"""

import argparse
import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.libero_regenerator import LiberoDataRegenerator


def main():
    """Command-line entry point for LIBERO data regeneration."""
    parser = argparse.ArgumentParser(
        description="Regenerate LIBERO dataset by replaying demonstrations in simulation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Regenerate libero_spatial dataset
  python scripts/regenerate_libero.py \\
      --libero_task_suite libero_spatial \\
      --libero_raw_data_dir ./data/libero_spatial \\
      --libero_target_dir ./data/libero_spatial_clean

  # Regenerate libero_object dataset with custom resolution
  python scripts/regenerate_libero.py \\
      --libero_task_suite libero_object \\
      --libero_raw_data_dir ./data/libero_object \\
      --libero_target_dir ./data/libero_object_clean \\
      --image_resolution 256
        """
    )
    
    parser.add_argument(
        "--libero_task_suite",
        type=str,
        required=True,
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
        help="LIBERO task suite name (libero_spatial, libero_object, libero_goal, libero_10, libero_90)"
    )
    
    parser.add_argument(
        "--libero_raw_data_dir",
        type=str,
        required=True,
        help="Path to directory containing original HDF5 dataset files"
    )
    
    parser.add_argument(
        "--libero_target_dir",
        type=str,
        required=True,
        help="Path to output directory for regenerated data"
    )
    
    parser.add_argument(
        "--image_resolution",
        type=int,
        default=256,
        help="Image resolution for camera observations (default: 256)"
    )
    
    args = parser.parse_args()
    
    # Create regenerator and execute
    regenerator = LiberoDataRegenerator(
        task_suite=args.libero_task_suite,
        raw_data_dir=args.libero_raw_data_dir,
        target_dir=args.libero_target_dir,
        image_resolution=args.image_resolution
    )
    
    result = regenerator.regenerate()
    
    # Check if cancelled
    if result.get("cancelled"):
        sys.exit(0)
    
    # Print final summary
    print("\n" + "=" * 50)
    print("Regeneration Summary")
    print("=" * 50)
    print(f"Total episodes replayed: {result['num_replays']}")
    print(f"Successful episodes: {result['num_success']}")
    print(f"Success rate: {result['success_rate']:.1f}%")
    print(f"No-op actions filtered: {result['num_noops']}")


if __name__ == "__main__":
    main()
