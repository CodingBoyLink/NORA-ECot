"""
LIBERO Environment Utilities.

Provides utility functions for creating and interacting with LIBERO environments.
Used by the data regeneration pipeline to replay demonstrations.

Requirements covered:
- 1.1: Support loading LIBERO benchmark tasks using libero.libero.benchmark
- 1.2: Create LIBERO environments with configurable image resolution (default 256x256)
- 1.3: Support all four LIBERO task suites: libero_spatial, libero_object, libero_goal, libero_10
- 1.4: Provide utility functions for environment creation consistent with NORA's implementation
"""

import os
from typing import Any, List, Tuple

try:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    LIBERO_AVAILABLE = True
except ImportError:
    LIBERO_AVAILABLE = False
    benchmark = None
    get_libero_path = None
    OffScreenRenderEnv = None


def get_libero_env(
    task: Any,
    model_family: str = "llava",
    resolution: int = 256
) -> Tuple[Any, str]:
    """
    Create LIBERO environment instance.
    
    Args:
        task: LIBERO task object from benchmark
        model_family: Model type (used for determining action format, default "llava")
        resolution: Image resolution for camera observations (default 256)
    
    Returns:
        Tuple[env, task_description]: Environment instance and task description string
    
    Raises:
        ImportError: If LIBERO is not installed
        FileNotFoundError: If BDDL file cannot be found
    
    Requirements:
        - 1.1: Support loading LIBERO benchmark tasks
        - 1.2: Create environments with configurable image resolution
        - 1.3: Support all four LIBERO task suites
    """
    if not LIBERO_AVAILABLE:
        raise ImportError(
            "LIBERO is not installed. Please install it with: "
            "pip install libero"
        )
    
    # Get task description
    task_description = task.language
    
    # Find BDDL file path
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file
    )
    
    if not os.path.exists(task_bddl_file):
        raise FileNotFoundError(
            f"BDDL file not found: {task_bddl_file}. "
            f"Please ensure LIBERO is properly installed."
        )
    
    # Create environment with specified resolution
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    
    return env, task_description


def get_libero_dummy_action(model_family: str = "llava") -> List[float]:
    """
    Get dummy action for LIBERO environment stabilization.
    
    Used to let the environment settle after setting initial state.
    Returns a 7-DoF action with all zeros (no movement).
    
    Args:
        model_family: Model type (for future compatibility, currently unused)
    
    Returns:
        List[float]: 7-DoF dummy action [dx, dy, dz, dax, day, daz, gripper]
    
    Requirements:
        - 1.4: Provide utility functions consistent with NORA's implementation
    """
    return [0.0] * 7


def get_benchmark_tasks(task_suite: str) -> Tuple[Any, int]:
    """
    Get LIBERO benchmark task suite and number of tasks.
    
    Args:
        task_suite: Name of task suite. One of:
            - "libero_spatial"
            - "libero_object" 
            - "libero_goal"
            - "libero_10" (also known as libero_10)
            - "libero_90"
    
    Returns:
        Tuple[task_suite_obj, num_tasks]: Task suite object and number of tasks
    
    Raises:
        ImportError: If LIBERO is not installed
        ValueError: If task_suite name is invalid
    
    Requirements:
        - 1.1: Support loading LIBERO benchmark tasks
        - 1.3: Support all four LIBERO task suites
    """
    if not LIBERO_AVAILABLE:
        raise ImportError(
            "LIBERO is not installed. Please install it with: "
            "pip install libero"
        )
    
    valid_suites = [
        "libero_spatial", 
        "libero_object", 
        "libero_goal", 
        "libero_10",
        "libero_90"
    ]
    
    if task_suite not in valid_suites:
        raise ValueError(
            f"Invalid task suite: {task_suite}. "
            f"Must be one of: {valid_suites}"
        )
    
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite_obj = benchmark_dict[task_suite]()
    num_tasks = task_suite_obj.n_tasks
    
    return task_suite_obj, num_tasks
