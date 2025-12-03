"""
LIBERO Data Regenerator.

Regenerates LIBERO dataset by replaying demonstrations in simulation environments.
Key improvements over original data:
1. Higher image resolution (256x256 instead of 128x128)
2. Filters out no-op actions to reduce redundant data
3. Filters out failed demonstrations to ensure data quality
4. Generates metainfo JSON for tracking episode success and initial states

Requirements covered:
- 2.1-2.4: No-op action filtering
- 3.1-3.5: Trajectory replay and data collection
- 4.1-4.5: Success filtering and data saving
- 5.1-5.4: Metainfo recording
- 6.1-6.4: CLI interface and integration
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np

try:
    import robosuite.utils.transform_utils as T
    ROBOSUITE_AVAILABLE = True
except ImportError:
    ROBOSUITE_AVAILABLE = False
    T = None

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None

from data_pipeline.libero_utils import (
    get_libero_env,
    get_libero_dummy_action,
    get_benchmark_tasks,
)


class LiberoDataRegenerator:
    """
    LIBERO dataset regenerator.
    
    Replays original demonstrations in LIBERO environments to generate
    cleaner, higher-quality training data.
    
    Requirements:
        - 6.1, 6.2: Configuration and directory management
    """
    
    def __init__(
        self,
        task_suite: str,
        raw_data_dir: str,
        target_dir: str,
        image_resolution: int = 256,
        noop_threshold: float = 1e-4,
        settle_steps: int = 10
    ):
        """
        Initialize the regenerator.
        
        Args:
            task_suite: Task suite name (libero_spatial, libero_object, 
                       libero_goal, libero_10, libero_90)
            raw_data_dir: Path to original HDF5 dataset directory
            target_dir: Path to output directory for regenerated data
            image_resolution: Image resolution for camera observations (default 256)
            noop_threshold: Threshold for no-op detection (default 1e-4)
            settle_steps: Number of steps to let environment settle (default 10)
        
        Requirements:
            - 6.1: Support command-line arguments
            - 6.2: Check target directory and handle overwriting
        """
        self.task_suite = task_suite
        self.raw_data_dir = raw_data_dir
        self.target_dir = target_dir
        self.image_resolution = image_resolution
        self.noop_threshold = noop_threshold
        self.settle_steps = settle_steps
        
        # Statistics
        self.num_replays = 0
        self.num_success = 0
        self.num_noops = 0
        
        # Metainfo dictionary
        self.metainfo: Dict[str, Dict[str, Any]] = {}
        
        # Validate dependencies
        if not ROBOSUITE_AVAILABLE:
            raise ImportError(
                "robosuite is not installed. Please install it with: "
                "pip install robosuite"
            )

    def _check_target_directory(self) -> bool:
        """
        Check if target directory exists and handle accordingly.
        
        Returns:
            bool: True if should proceed, False if user cancelled
        
        Requirements:
            - 6.2: Check if target directory exists and prompt for confirmation
        """
        if os.path.isdir(self.target_dir):
            user_input = input(
                f"Target directory already exists at path: {self.target_dir}\n"
                "Enter 'y' to overwrite the directory, or anything else to exit: "
            )
            if user_input.lower() != 'y':
                return False
        
        os.makedirs(self.target_dir, exist_ok=True)
        return True
    
    def is_noop(
        self,
        action: np.ndarray,
        prev_action: Optional[np.ndarray] = None
    ) -> bool:
        """
        Determine if an action is a no-op (no operation).
        
        A no-op action satisfies two criteria:
        1. All action dimensions except gripper (indices 0-5) have norm < threshold
        2. The gripper action equals the previous timestep's gripper action
        
        Args:
            action: Current action (7-DoF: [dx, dy, dz, dax, day, daz, gripper])
            prev_action: Previous action (None for first action in episode)
        
        Returns:
            bool: True if action is a no-op
        
        Requirements:
            - 2.1: Identify no-op using norm threshold for dimensions 0-5
            - 2.2: Check gripper action equals previous timestep
            - 2.3: Handle first action (only check criterion 1)
            - 2.4: Log filtered no-op actions
        """
        # Criterion 1: Check if movement dimensions (0-5) are near zero
        movement_norm = np.linalg.norm(action[:-1])
        movement_is_zero = movement_norm < self.noop_threshold
        
        # Special case: First action in episode (prev_action is None)
        # Only check criterion 1
        if prev_action is None:
            return movement_is_zero
        
        # Criterion 2: Check if gripper action equals previous gripper action
        gripper_action = action[-1]
        prev_gripper_action = prev_action[-1]
        gripper_unchanged = gripper_action == prev_gripper_action
        
        # Both criteria must be satisfied for no-op
        return movement_is_zero and gripper_unchanged
    
    def replay_episode(
        self,
        env: Any,
        orig_actions: np.ndarray,
        orig_states: np.ndarray,
        demo_data: Any
    ) -> Tuple[Dict[str, Any], bool, int]:
        """
        Replay a single episode in the environment.
        
        Args:
            env: LIBERO environment instance
            orig_actions: Original action sequence from demo
            orig_states: Original state sequence from demo
            demo_data: Original demo data (HDF5 group)
        
        Returns:
            Tuple containing:
                - data_dict: Collected observation data
                - success: Whether episode completed successfully
                - num_noops: Number of no-op actions filtered
        
        Requirements:
            - 3.1: Reset environment and set initial state
            - 3.2: Wait for environment to settle (default 10 steps)
            - 3.3: Replay original demonstration actions
            - 3.4: Collect observations at each timestep
            - 3.5: Record original actions for non-no-op transitions
        """
        # Reset environment and set initial state
        env.reset()
        env.set_init_state(orig_states[0])
        
        # Wait for environment to settle
        dummy_action = get_libero_dummy_action()
        for _ in range(self.settle_steps):
            obs, reward, done, info = env.step(dummy_action)
        
        # Initialize data collection lists
        states = []
        actions = []
        ee_states = []
        gripper_states = []
        joint_states = []
        robot_states = []
        agentview_images = []
        eye_in_hand_images = []
        
        num_noops = 0
        success = False  # Track if task was completed successfully
        
        # Replay original demo actions
        for action in orig_actions:
            # Check for no-op action
            prev_action = actions[-1] if len(actions) > 0 else None
            if self.is_noop(action, prev_action):
                num_noops += 1
                continue
            
            # Collect state data
            if len(states) == 0:
                # First timestep: copy initial state from original demo
                states.append(orig_states[0])
                robot_states.append(demo_data["robot_states"][0])
            else:
                # Other timesteps: get state from environment
                states.append(env.sim.get_state().flatten())
                robot_states.append(
                    np.concatenate([
                        obs["robot0_gripper_qpos"],
                        obs["robot0_eef_pos"],
                        obs["robot0_eef_quat"]
                    ])
                )
            
            # Record original action
            actions.append(action)
            
            # Record observation data
            if "robot0_gripper_qpos" in obs:
                gripper_states.append(obs["robot0_gripper_qpos"])
            joint_states.append(obs["robot0_joint_pos"])
            
            # Compute ee_states: position + axis-angle orientation
            ee_pos = obs["robot0_eef_pos"]
            ee_ori = T.quat2axisangle(obs["robot0_eef_quat"])
            ee_states.append(np.hstack((ee_pos, ee_ori)))
            
            # Record images
            agentview_images.append(obs["agentview_image"])
            eye_in_hand_images.append(obs["robot0_eye_in_hand_image"])
            
            # Execute action in environment
            obs, reward, done, info = env.step(action.tolist())
            
            # Check if task completed - stop early if successful
            if done:
                success = True
                break
        
        # Package collected data
        data_dict = {
            "states": states,
            "actions": actions,
            "ee_states": ee_states,
            "gripper_states": gripper_states,
            "joint_states": joint_states,
            "robot_states": robot_states,
            "agentview_images": agentview_images,
            "eye_in_hand_images": eye_in_hand_images,
        }
        
        return data_dict, success, num_noops

    def save_episode_to_hdf5(
        self,
        hdf5_group: Any,
        episode_id: int,
        data_dict: Dict[str, Any]
    ) -> None:
        """
        Save episode data to HDF5 file.
        
        Args:
            hdf5_group: HDF5 group to save data to
            episode_id: Episode identifier
            data_dict: Collected episode data
        
        Requirements:
            - 4.3: Save regenerated data to HDF5 files
            - 4.4: Save all required datasets
            - 4.5: Set rewards and dones arrays appropriately
        """
        actions = data_dict["actions"]
        num_steps = len(actions)
        
        # Create rewards and dones arrays
        # reward=1 and done=1 only at final timestep
        dones = np.zeros(num_steps, dtype=np.uint8)
        dones[-1] = 1
        rewards = np.zeros(num_steps, dtype=np.uint8)
        rewards[-1] = 1
        
        # Create episode group
        ep_data_grp = hdf5_group.create_group(f"demo_{episode_id}")
        obs_grp = ep_data_grp.create_group("obs")
        
        # Stack arrays
        ee_states_arr = np.stack(data_dict["ee_states"], axis=0)
        
        # Save observation data
        obs_grp.create_dataset(
            "gripper_states", 
            data=np.stack(data_dict["gripper_states"], axis=0)
        )
        obs_grp.create_dataset(
            "joint_states", 
            data=np.stack(data_dict["joint_states"], axis=0)
        )
        obs_grp.create_dataset("ee_states", data=ee_states_arr)
        obs_grp.create_dataset("ee_pos", data=ee_states_arr[:, :3])
        obs_grp.create_dataset("ee_ori", data=ee_states_arr[:, 3:])
        obs_grp.create_dataset(
            "agentview_rgb", 
            data=np.stack(data_dict["agentview_images"], axis=0)
        )
        obs_grp.create_dataset(
            "eye_in_hand_rgb", 
            data=np.stack(data_dict["eye_in_hand_images"], axis=0)
        )
        
        # Save action, state, and metadata
        ep_data_grp.create_dataset("actions", data=actions)
        ep_data_grp.create_dataset("states", data=np.stack(data_dict["states"]))
        ep_data_grp.create_dataset(
            "robot_states", 
            data=np.stack(data_dict["robot_states"], axis=0)
        )
        ep_data_grp.create_dataset("rewards", data=rewards)
        ep_data_grp.create_dataset("dones", data=dones)
    
    def update_metainfo(
        self,
        task_description: str,
        episode_id: int,
        success: bool,
        initial_state: np.ndarray
    ) -> None:
        """
        Update metainfo dictionary and save to JSON file.
        
        Args:
            task_description: Task description string
            episode_id: Episode identifier
            success: Whether replay was successful
            initial_state: Initial simulation state
        
        Requirements:
            - 5.1: Create metainfo JSON file for each task suite
            - 5.2: Record success and initial_state for each episode
            - 5.3: Organize metainfo by task description and episode ID
            - 5.4: Update JSON file after each episode (crash recovery)
        """
        # Convert task description to key format (replace spaces with underscores)
        task_key = task_description.replace(" ", "_")
        episode_key = f"demo_{episode_id}"
        
        # Initialize task entry if needed
        if task_key not in self.metainfo:
            self.metainfo[task_key] = {}
        
        # Record episode info
        self.metainfo[task_key][episode_key] = {
            "success": success,
            "initial_state": initial_state.tolist()
        }
        
        # Write to JSON file (overwrite for crash recovery)
        metainfo_path = self._get_metainfo_path()
        with open(metainfo_path, "w") as f:
            json.dump(self.metainfo, f, indent=2)
    
    def _get_metainfo_path(self) -> str:
        """Get path to metainfo JSON file."""
        return f"./{self.task_suite}_metainfo.json"
    
    def regenerate(self) -> Dict[str, Any]:
        """
        Execute the complete data regeneration process.
        
        Returns:
            Dict containing statistics:
                - num_replays: Total episodes replayed
                - num_success: Total successful episodes
                - num_noops: Total no-op actions filtered
                - success_rate: Success rate percentage
        
        Requirements:
            - 4.1: Check episode success status at end of replay
            - 4.2: Only save trajectories that completed successfully
            - 6.3: Display progress information and statistics
        """
        print(f"Regenerating {self.task_suite} dataset!")
        
        # Check and create target directory
        if not self._check_target_directory():
            print("Regeneration cancelled by user.")
            return {"cancelled": True}
        
        # Initialize metainfo JSON file
        metainfo_path = self._get_metainfo_path()
        with open(metainfo_path, "w") as f:
            json.dump({}, f)
        
        # Get task suite
        task_suite_obj, num_tasks = get_benchmark_tasks(self.task_suite)
        
        # Reset statistics
        self.num_replays = 0
        self.num_success = 0
        self.num_noops = 0
        
        # Create progress iterator
        task_iterator = range(num_tasks)
        if TQDM_AVAILABLE:
            task_iterator = tqdm(task_iterator, desc="Tasks")
        
        for task_id in task_iterator:
            # Get task
            task = task_suite_obj.get_task(task_id)
            
            try:
                env, task_description = get_libero_env(
                    task, 
                    resolution=self.image_resolution
                )
            except Exception as e:
                print(f"[ERROR] Failed to create environment for task: {task.name}")
                print(f"[ERROR] {e}")
                continue
            
            # Get original dataset
            orig_data_path = os.path.join(
                self.raw_data_dir, 
                f"{task.name}_demo.hdf5"
            )
            
            if not os.path.exists(orig_data_path):
                print(f"[WARNING] Raw data file not found: {orig_data_path}")
                continue
            
            orig_data_file = h5py.File(orig_data_path, "r")
            orig_data = orig_data_file["data"]
            
            # Create new HDF5 file for regenerated demos
            new_data_path = os.path.join(
                self.target_dir, 
                f"{task.name}_demo.hdf5"
            )
            new_data_file = h5py.File(new_data_path, "w")
            grp = new_data_file.create_group("data")
            
            # Process each episode
            num_episodes = len(orig_data.keys())
            saved_episode_count = 0  # Track number of successfully saved episodes
            
            for i in range(num_episodes):
                # Get demo data
                demo_data = orig_data[f"demo_{i}"]
                orig_actions = demo_data["actions"][()]
                orig_states = demo_data["states"][()]
                
                # Replay episode
                data_dict, success, episode_noops = self.replay_episode(
                    env, orig_actions, orig_states, demo_data
                )
                
                self.num_noops += episode_noops
                self.num_replays += 1
                
                # Save only successful trajectories (with consecutive numbering)
                if success:
                    self.save_episode_to_hdf5(grp, saved_episode_count, data_dict)
                    saved_episode_count += 1
                    self.num_success += 1
                
                # Update metainfo (for both success and failure)
                self.update_metainfo(
                    task_description, i, success, orig_states[0]
                )
                
                # Print progress
                success_rate = self.num_success / self.num_replays * 100
                print(
                    f"Total # episodes replayed: {self.num_replays}, "
                    f"Total # successes: {self.num_success} ({success_rate:.1f}%)"
                )
                print(f"  Total # no-op actions filtered out: {self.num_noops}")
            
            # Close HDF5 files
            orig_data_file.close()
            new_data_file.close()
            print(f"Saved regenerated demos for task '{task_description}' at: {new_data_path}")
        
        # Final summary
        print(f"\nDataset regeneration complete!")
        print(f"Saved new dataset at: {self.target_dir}")
        print(f"Saved metainfo JSON at: {metainfo_path}")
        
        return {
            "num_replays": self.num_replays,
            "num_success": self.num_success,
            "num_noops": self.num_noops,
            "success_rate": self.num_success / max(self.num_replays, 1) * 100
        }
