#!/usr/bin/env python
"""
Unit tests for regenerated LIBERO data compatibility with LiberoRawLoader.

Tests that regenerated HDF5 files maintain the expected data structure
and can be loaded by the existing data pipeline.

Requirements covered:
- 7.1: Regenerated HDF5 files compatible with LiberoRawLoader
- 7.2: Regenerated data maintains expected structure
- 7.3: Task instruction parsing preserved
"""

import os
import sys
import tempfile
import unittest

import h5py
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.raw_loader import LiberoRawLoader, parse_task_instruction


class TestRegeneratedDataCompatibility(unittest.TestCase):
    """Test regenerated data compatibility with LiberoRawLoader."""
    
    @classmethod
    def setUpClass(cls):
        """Create a mock regenerated HDF5 file for testing."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_file = os.path.join(
            cls.temp_dir, 
            "pick_up_the_butter_and_place_it_in_the_basket_demo.hdf5"
        )
        
        # Create mock regenerated data with expected format
        cls._create_mock_regenerated_file(cls.test_file)
    
    @classmethod
    def _create_mock_regenerated_file(cls, file_path: str):
        """
        Create a mock HDF5 file with regenerated data format.
        
        This mimics the output of LiberoDataRegenerator.save_episode_to_hdf5()
        """
        num_demos = 3
        num_steps = 50
        image_resolution = 256
        
        with h5py.File(file_path, "w") as f:
            data_grp = f.create_group("data")
            
            for demo_id in range(num_demos):
                demo_grp = data_grp.create_group(f"demo_{demo_id}")
                obs_grp = demo_grp.create_group("obs")
                
                # Create observation data (matching regenerator output format)
                obs_grp.create_dataset(
                    "agentview_rgb",
                    data=np.random.randint(
                        0, 255, 
                        (num_steps, image_resolution, image_resolution, 3),
                        dtype=np.uint8
                    )
                )
                obs_grp.create_dataset(
                    "eye_in_hand_rgb",
                    data=np.random.randint(
                        0, 255,
                        (num_steps, image_resolution, image_resolution, 3),
                        dtype=np.uint8
                    )
                )
                obs_grp.create_dataset(
                    "ee_pos",
                    data=np.random.randn(num_steps, 3).astype(np.float32)
                )
                obs_grp.create_dataset(
                    "ee_ori",
                    data=np.random.randn(num_steps, 3).astype(np.float32)
                )
                obs_grp.create_dataset(
                    "gripper_states",
                    data=np.random.randn(num_steps, 2).astype(np.float32)
                )
                obs_grp.create_dataset(
                    "joint_states",
                    data=np.random.randn(num_steps, 7).astype(np.float32)
                )
                obs_grp.create_dataset(
                    "ee_states",
                    data=np.random.randn(num_steps, 6).astype(np.float32)
                )
                
                # Create action and state data
                demo_grp.create_dataset(
                    "actions",
                    data=np.random.randn(num_steps, 7).astype(np.float32)
                )
                demo_grp.create_dataset(
                    "states",
                    data=np.random.randn(num_steps, 100).astype(np.float32)
                )
                demo_grp.create_dataset(
                    "robot_states",
                    data=np.random.randn(num_steps, 9).astype(np.float32)
                )
                
                # Create rewards and dones
                rewards = np.zeros(num_steps, dtype=np.uint8)
                rewards[-1] = 1
                dones = np.zeros(num_steps, dtype=np.uint8)
                dones[-1] = 1
                demo_grp.create_dataset("rewards", data=rewards)
                demo_grp.create_dataset("dones", data=dones)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(cls.temp_dir, ignore_errors=True)
    
    def test_task_instruction_parsing(self):
        """Test that task instruction is correctly parsed from filename."""
        instruction = parse_task_instruction(self.test_file)
        expected = "pick up the butter and place it in the basket"
        self.assertEqual(instruction, expected)
    
    def test_loader_initialization(self):
        """Test that LiberoRawLoader can initialize with regenerated file."""
        loader = LiberoRawLoader(self.test_file, verbose=False)
        self.assertEqual(len(loader), 3)  # 3 demos
        self.assertEqual(loader.task_description, 
                        "pick up the butter and place it in the basket")
    
    def test_trajectory_loading(self):
        """Test that trajectories can be loaded correctly."""
        loader = LiberoRawLoader(self.test_file, verbose=False)
        
        for i in range(len(loader)):
            traj = loader.get_trajectory(i)
            
            # Check required keys exist
            self.assertIn("actions", traj)
            self.assertIn("agentview", traj)
            self.assertIn("eye_in_hand", traj)
            self.assertIn("states", traj)
            self.assertIn("images", traj)
            self.assertIn("instruction", traj)
    
    def test_actions_shape(self):
        """Test that actions have correct shape (N, 7)."""
        loader = LiberoRawLoader(self.test_file, verbose=False)
        traj = loader.get_trajectory(0)
        
        self.assertEqual(traj["actions"].ndim, 2)
        self.assertEqual(traj["actions"].shape[1], 7)
    
    def test_states_shape(self):
        """Test that states have correct shape (N, 8) = ee_pos + ee_ori + gripper."""
        loader = LiberoRawLoader(self.test_file, verbose=False)
        traj = loader.get_trajectory(0)
        
        self.assertEqual(traj["states"].ndim, 2)
        self.assertEqual(traj["states"].shape[1], 8)  # 3 + 3 + 2
    
    def test_image_shape(self):
        """Test that images have correct shape (N, H, W, 3)."""
        loader = LiberoRawLoader(self.test_file, verbose=False)
        traj = loader.get_trajectory(0)
        
        # Agentview
        self.assertEqual(traj["agentview"].ndim, 4)
        self.assertEqual(traj["agentview"].shape[-1], 3)
        
        # Eye in hand
        self.assertEqual(traj["eye_in_hand"].ndim, 4)
        self.assertEqual(traj["eye_in_hand"].shape[-1], 3)
    
    def test_image_resolution(self):
        """Test that images have expected resolution (256x256)."""
        loader = LiberoRawLoader(self.test_file, verbose=False)
        traj = loader.get_trajectory(0)
        
        self.assertEqual(traj["agentview"].shape[1], 256)
        self.assertEqual(traj["agentview"].shape[2], 256)
        self.assertEqual(traj["eye_in_hand"].shape[1], 256)
        self.assertEqual(traj["eye_in_hand"].shape[2], 256)
    
    def test_images_dict_structure(self):
        """Test that images dict contains expected keys."""
        loader = LiberoRawLoader(self.test_file, verbose=False)
        traj = loader.get_trajectory(0)
        
        self.assertIn("agentview", traj["images"])
        self.assertIn("eye_in_hand", traj["images"])
    
    def test_instruction_preserved(self):
        """Test that instruction is preserved in trajectory data."""
        loader = LiberoRawLoader(self.test_file, verbose=False)
        traj = loader.get_trajectory(0)
        
        self.assertEqual(
            traj["instruction"],
            "pick up the butter and place it in the basket"
        )


class TestHDF5StructureValidation(unittest.TestCase):
    """Test HDF5 structure validation for regenerated data."""
    
    def setUp(self):
        """Create temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_missing_obs_group_fails(self):
        """Test that missing obs group causes loader to fail."""
        file_path = os.path.join(self.temp_dir, "test_demo.hdf5")
        
        with h5py.File(file_path, "w") as f:
            data_grp = f.create_group("data")
            demo_grp = data_grp.create_group("demo_0")
            demo_grp.create_dataset("actions", data=np.zeros((10, 7)))
            # Missing obs group
        
        with self.assertRaises(KeyError):
            loader = LiberoRawLoader(file_path, verbose=False)
            loader.get_trajectory(0)
    
    def test_missing_required_field_fails(self):
        """Test that missing required obs field causes loader to fail."""
        file_path = os.path.join(self.temp_dir, "test_demo.hdf5")
        
        with h5py.File(file_path, "w") as f:
            data_grp = f.create_group("data")
            demo_grp = data_grp.create_group("demo_0")
            obs_grp = demo_grp.create_group("obs")
            
            demo_grp.create_dataset("actions", data=np.zeros((10, 7)))
            obs_grp.create_dataset("agentview_rgb", data=np.zeros((10, 256, 256, 3)))
            # Missing other required fields
        
        with self.assertRaises(KeyError):
            loader = LiberoRawLoader(file_path, verbose=False)
            loader.get_trajectory(0)


class TestRealDataCompatibility(unittest.TestCase):
    """Test compatibility with real regenerated data if available."""
    
    @classmethod
    def setUpClass(cls):
        """Check if real regenerated data exists."""
        # Look for regenerated data directories
        cls.regenerated_dirs = []
        possible_dirs = [
            "./data/libero_object_clean",
            "./data/libero_spatial_clean",
            "./data/libero_goal_clean",
            "./data/libero_10_clean",
        ]
        
        for dir_path in possible_dirs:
            if os.path.isdir(dir_path):
                cls.regenerated_dirs.append(dir_path)
        
        if not cls.regenerated_dirs:
            raise unittest.SkipTest("No regenerated data directories found")
    
    def test_real_regenerated_data_loads(self):
        """Test that real regenerated data can be loaded."""
        import glob
        
        for data_dir in self.regenerated_dirs:
            hdf5_files = glob.glob(os.path.join(data_dir, "*.hdf5"))
            
            for file_path in hdf5_files[:2]:  # Test first 2 files per directory
                with self.subTest(file=file_path):
                    loader = LiberoRawLoader(file_path, verbose=False)
                    self.assertGreater(len(loader), 0)
                    
                    # Load first trajectory
                    traj = loader.get_trajectory(0)
                    self.assertIn("actions", traj)
                    self.assertIn("agentview", traj)
                    self.assertIn("states", traj)


if __name__ == "__main__":
    unittest.main(verbosity=2)
