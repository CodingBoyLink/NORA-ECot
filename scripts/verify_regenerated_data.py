#!/usr/bin/env python
"""
Verification script for regenerated LIBERO data compatibility with LiberoRawLoader.

This script verifies that regenerated HDF5 files are compatible with the
existing data loading pipeline (LiberoRawLoader).

Requirements covered:
- 7.1: Regenerated HDF5 files compatible with LiberoRawLoader
- 7.2: Regenerated data maintains expected structure
- 7.3: Task instruction parsing preserved

Usage:
    python scripts/verify_regenerated_data.py --data_dir ./data/libero_object_clean
    python scripts/verify_regenerated_data.py --data_file ./data/libero_object_clean/task_demo.hdf5
"""

import argparse
import os
import sys
import glob

import h5py
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.raw_loader import LiberoRawLoader, parse_task_instruction


# Required fields for LiberoRawLoader compatibility
REQUIRED_OBS_FIELDS = [
    "agentview_rgb",
    "eye_in_hand_rgb", 
    "ee_pos",
    "ee_ori",
    "gripper_states",
]

REQUIRED_DATA_FIELDS = [
    "actions",
]

# Expected data shapes and types
EXPECTED_SHAPES = {
    "agentview_rgb": (None, None, None, 3),  # (N, H, W, 3)
    "eye_in_hand_rgb": (None, None, None, 3),  # (N, H, W, 3)
    "ee_pos": (None, 3),  # (N, 3)
    "ee_ori": (None, 3),  # (N, 3)
    "gripper_states": (None, 2),  # (N, 2)
    "actions": (None, 7),  # (N, 7)
}


class DataVerificationError(Exception):
    """Custom exception for data verification errors."""
    pass


def verify_hdf5_structure(file_path: str, verbose: bool = True) -> dict:
    """
    Verify HDF5 file structure matches expected format.
    
    Args:
        file_path: Path to HDF5 file
        verbose: Print detailed information
    
    Returns:
        dict: Verification results
    
    Raises:
        DataVerificationError: If structure is invalid
    """
    results = {
        "file": file_path,
        "valid": True,
        "num_demos": 0,
        "errors": [],
        "warnings": [],
    }
    
    if not os.path.exists(file_path):
        results["valid"] = False
        results["errors"].append(f"File not found: {file_path}")
        return results
    
    try:
        with h5py.File(file_path, "r") as f:
            # Check for 'data' group
            if "data" not in f:
                results["valid"] = False
                results["errors"].append("Missing 'data' group in HDF5 file")
                return results
            
            data_group = f["data"]
            demo_keys = list(data_group.keys())
            results["num_demos"] = len(demo_keys)
            
            if len(demo_keys) == 0:
                results["warnings"].append("No demos found in file")
                return results
            
            # Verify each demo
            for demo_key in demo_keys:
                demo_group = data_group[demo_key]
                
                # Check required data fields
                for field in REQUIRED_DATA_FIELDS:
                    if field not in demo_group:
                        results["valid"] = False
                        results["errors"].append(
                            f"{demo_key}: Missing required field '{field}'"
                        )
                
                # Check obs group
                if "obs" not in demo_group:
                    results["valid"] = False
                    results["errors"].append(f"{demo_key}: Missing 'obs' group")
                    continue
                
                obs_group = demo_group["obs"]
                
                # Check required obs fields
                for field in REQUIRED_OBS_FIELDS:
                    if field not in obs_group:
                        results["valid"] = False
                        results["errors"].append(
                            f"{demo_key}/obs: Missing required field '{field}'"
                        )
                
                # Verify data shapes
                if results["valid"]:
                    _verify_data_shapes(demo_group, demo_key, results)
    
    except Exception as e:
        results["valid"] = False
        results["errors"].append(f"Error reading HDF5 file: {str(e)}")
    
    if verbose:
        _print_verification_results(results)
    
    return results


def _verify_data_shapes(demo_group, demo_key: str, results: dict):
    """Verify data shapes match expected format."""
    obs_group = demo_group["obs"]
    
    # Get number of timesteps from actions
    num_steps = demo_group["actions"].shape[0]
    
    # Verify obs field shapes
    for field, expected_shape in EXPECTED_SHAPES.items():
        if field == "actions":
            data = demo_group[field]
        else:
            if field not in obs_group:
                continue
            data = obs_group[field]
        
        actual_shape = data.shape
        
        # Check dimensions match (None means any value)
        if len(actual_shape) != len(expected_shape):
            results["errors"].append(
                f"{demo_key}: {field} has wrong number of dimensions. "
                f"Expected {len(expected_shape)}, got {len(actual_shape)}"
            )
            results["valid"] = False
            continue
        
        for i, (actual, expected) in enumerate(zip(actual_shape, expected_shape)):
            if expected is not None and actual != expected:
                results["errors"].append(
                    f"{demo_key}: {field} dimension {i} mismatch. "
                    f"Expected {expected}, got {actual}"
                )
                results["valid"] = False


def _print_verification_results(results: dict):
    """Print verification results."""
    status = "✓ VALID" if results["valid"] else "✗ INVALID"
    print(f"\n[{status}] {results['file']}")
    print(f"  Demos: {results['num_demos']}")
    
    if results["errors"]:
        print("  Errors:")
        for error in results["errors"]:
            print(f"    - {error}")
    
    if results["warnings"]:
        print("  Warnings:")
        for warning in results["warnings"]:
            print(f"    - {warning}")


def verify_loader_compatibility(file_path: str, verbose: bool = True) -> dict:
    """
    Verify file can be loaded by LiberoRawLoader.
    
    Args:
        file_path: Path to HDF5 file
        verbose: Print detailed information
    
    Returns:
        dict: Verification results
    """
    results = {
        "file": file_path,
        "loader_compatible": True,
        "task_instruction": None,
        "trajectories_loaded": 0,
        "errors": [],
    }
    
    try:
        # Test task instruction parsing
        results["task_instruction"] = parse_task_instruction(file_path)
        
        # Test loader initialization
        loader = LiberoRawLoader(file_path, verbose=False)
        results["trajectories_loaded"] = len(loader)
        
        # Test loading first trajectory
        if len(loader) > 0:
            traj = loader.get_trajectory(0)
            
            # Verify trajectory data structure
            required_keys = ["actions", "agentview", "eye_in_hand", "states", 
                          "images", "instruction"]
            for key in required_keys:
                if key not in traj:
                    results["loader_compatible"] = False
                    results["errors"].append(f"Missing key in trajectory: {key}")
            
            # Verify data types and shapes
            if "actions" in traj:
                if traj["actions"].ndim != 2 or traj["actions"].shape[1] != 7:
                    results["errors"].append(
                        f"Actions shape mismatch: {traj['actions'].shape}"
                    )
                    results["loader_compatible"] = False
            
            if "states" in traj:
                if traj["states"].ndim != 2 or traj["states"].shape[1] != 8:
                    results["errors"].append(
                        f"States shape mismatch: {traj['states'].shape}"
                    )
                    results["loader_compatible"] = False
            
            if "agentview" in traj:
                if traj["agentview"].ndim != 4 or traj["agentview"].shape[-1] != 3:
                    results["errors"].append(
                        f"Agentview shape mismatch: {traj['agentview'].shape}"
                    )
                    results["loader_compatible"] = False
    
    except Exception as e:
        results["loader_compatible"] = False
        results["errors"].append(f"Loader error: {str(e)}")
    
    if verbose:
        status = "✓ COMPATIBLE" if results["loader_compatible"] else "✗ INCOMPATIBLE"
        print(f"\n[{status}] LiberoRawLoader compatibility")
        print(f"  Task instruction: {results['task_instruction']}")
        print(f"  Trajectories: {results['trajectories_loaded']}")
        if results["errors"]:
            print("  Errors:")
            for error in results["errors"]:
                print(f"    - {error}")
    
    return results


def verify_directory(data_dir: str, verbose: bool = True) -> dict:
    """
    Verify all HDF5 files in a directory.
    
    Args:
        data_dir: Directory containing HDF5 files
        verbose: Print detailed information
    
    Returns:
        dict: Summary of verification results
    """
    hdf5_files = glob.glob(os.path.join(data_dir, "*.hdf5"))
    
    if not hdf5_files:
        print(f"No HDF5 files found in: {data_dir}")
        return {"valid": False, "files_checked": 0}
    
    summary = {
        "directory": data_dir,
        "files_checked": len(hdf5_files),
        "files_valid": 0,
        "files_compatible": 0,
        "total_demos": 0,
        "errors": [],
    }
    
    print(f"\nVerifying {len(hdf5_files)} HDF5 files in: {data_dir}")
    print("=" * 60)
    
    for file_path in hdf5_files:
        # Verify HDF5 structure
        struct_results = verify_hdf5_structure(file_path, verbose=verbose)
        if struct_results["valid"]:
            summary["files_valid"] += 1
            summary["total_demos"] += struct_results["num_demos"]
        else:
            summary["errors"].extend(struct_results["errors"])
        
        # Verify loader compatibility
        loader_results = verify_loader_compatibility(file_path, verbose=verbose)
        if loader_results["loader_compatible"]:
            summary["files_compatible"] += 1
        else:
            summary["errors"].extend(loader_results["errors"])
    
    # Print summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"Directory: {data_dir}")
    print(f"Files checked: {summary['files_checked']}")
    print(f"Files with valid structure: {summary['files_valid']}")
    print(f"Files compatible with LiberoRawLoader: {summary['files_compatible']}")
    print(f"Total demos: {summary['total_demos']}")
    
    if summary["files_valid"] == summary["files_checked"] and \
       summary["files_compatible"] == summary["files_checked"]:
        print("\n✓ All files passed verification!")
        summary["valid"] = True
    else:
        print(f"\n✗ {summary['files_checked'] - summary['files_valid']} files failed verification")
        summary["valid"] = False
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Verify regenerated LIBERO data compatibility"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Directory containing regenerated HDF5 files"
    )
    parser.add_argument(
        "--data_file",
        type=str,
        help="Single HDF5 file to verify"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print summary"
    )
    
    args = parser.parse_args()
    
    if not args.data_dir and not args.data_file:
        parser.error("Either --data_dir or --data_file must be specified")
    
    verbose = not args.quiet
    
    if args.data_file:
        # Verify single file
        struct_results = verify_hdf5_structure(args.data_file, verbose=verbose)
        loader_results = verify_loader_compatibility(args.data_file, verbose=verbose)
        
        if struct_results["valid"] and loader_results["loader_compatible"]:
            print("\n✓ File passed all verification checks!")
            sys.exit(0)
        else:
            print("\n✗ File failed verification")
            sys.exit(1)
    
    if args.data_dir:
        # Verify directory
        summary = verify_directory(args.data_dir, verbose=verbose)
        sys.exit(0 if summary.get("valid", False) else 1)


if __name__ == "__main__":
    main()
