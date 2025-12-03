"""
LIBERO Dataset for NORA LoRA Training.

Implements dataset loading for baseline and CoT training phases.
Follows the structure from nora/training/datasets/datasets.py.

Requirements covered:
- 3.4: Format each training sample following NORA's chat template
- 3.5: Format action tokens as <robot_action_N> strings
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
from torch.utils.data import IterableDataset, Dataset

from data_pipeline.raw_loader import LiberoRawLoader, LiberoBatchLoader
from data_pipeline.data_splitter import LiberoDataSplitter, SplitInfo
from training.action_tokenizer import ActionTokenizer


# HuggingFace Default IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


@dataclass
class LiberoSample:
    """Single training sample from LIBERO dataset."""
    image: np.ndarray           # (H, W, 3) RGB image
    instruction: str            # Task instruction text
    action: np.ndarray          # (7,) continuous action
    file_path: str              # Source file path
    demo_key: str               # Demo key in HDF5
    step_idx: int               # Step index in trajectory
    # Optional fields for CoT training
    reasoning: Optional[Dict[str, str]] = None
    flow_tokens: Optional[List[int]] = None


class LiberoBaselineDataset(IterableDataset):
    """
    LIBERO Dataset for Baseline LoRA training.
    
    Yields individual timesteps from trajectories with:
    - Image observation (agentview)
    - Task instruction
    - Action (7-DoF continuous)
    
    Format: [VIS, L, ACTION:, action_tokens]
    
    Requirements:
        - 3.4: Format each training sample following NORA's chat template
        - 3.5: Format action tokens as <robot_action_N> strings
    """
    
    def __init__(
        self,
        data_dir: str,
        subset: str,
        split: str = "train",
        train_ratio: float = 0.95,
        seed: int = 42,
        action_tokenizer: Optional[ActionTokenizer] = None,
        normalize_gripper: bool = True,
        shuffle: bool = True,
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: LIBERO data root directory
            subset: LIBERO subset ('spatial', 'object', 'goal', 'long')
            split: 'train' or 'test'
            train_ratio: Train/test split ratio
            seed: Random seed for splitting and shuffling
            action_tokenizer: ActionTokenizer instance (created if None)
            normalize_gripper: Whether to normalize gripper actions
            shuffle: Whether to shuffle data
        """
        self.data_dir = data_dir
        self.subset = subset
        self.split = split
        self.seed = seed
        self.normalize_gripper = normalize_gripper
        self.shuffle = shuffle
        
        # Create data splitter
        self.splitter = LiberoDataSplitter(
            data_dir=data_dir,
            subset=subset,
            train_ratio=train_ratio,
            seed=seed,
        )
        self.split_info = self.splitter.split()
        
        # Create batch loader for the specified split
        self.loader = LiberoBatchLoader.from_splitter(
            self.splitter,
            split=split,
            verbose=False,
        )
        
        # Create action tokenizer if not provided
        self.action_tokenizer = action_tokenizer or ActionTokenizer()

    def _iterate_samples(self) -> Iterator[LiberoSample]:
        """
        Iterate over all samples in the dataset.
        
        Yields individual timesteps from all trajectories.
        """
        # Get trajectory indices
        indices = list(range(len(self.loader)))
        
        # Shuffle trajectory order if requested
        if self.shuffle:
            rng = np.random.RandomState(self.seed)
            rng.shuffle(indices)
        
        for traj_idx in indices:
            trajectory = self.loader.get_trajectory(traj_idx)
            
            # Iterate over timesteps in trajectory
            num_steps = len(trajectory['actions'])
            for step_idx in range(num_steps):
                yield LiberoSample(
                    image=trajectory['agentview'][step_idx],
                    instruction=trajectory['instruction'],
                    action=trajectory['actions'][step_idx],
                    file_path=trajectory['file_path'],
                    demo_key=trajectory['demo_key'],
                    step_idx=step_idx,
                )
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Iterate over processed samples ready for collation.
        
        Yields:
            Dict with 'image', 'instruction', 'action', 'action_tokens', 'vlm_action_string'
        """
        for sample in self._iterate_samples():
            # Encode action to tokens
            action_tokens = self.action_tokenizer.encode(
                sample.action,
                normalize_gripper=self.normalize_gripper,
            )
            vlm_action_string = self.action_tokenizer.tokens_to_vlm_string(action_tokens)
            
            yield {
                'image': sample.image,
                'instruction': sample.instruction,
                'action': sample.action,
                'action_tokens': action_tokens,
                'vlm_action_string': vlm_action_string,
                'file_path': sample.file_path,
                'demo_key': sample.demo_key,
                'step_idx': sample.step_idx,
            }
    
    def __len__(self) -> int:
        """
        Return estimated total number of samples.
        
        Note: This is an estimate based on trajectory count.
        Actual count requires iterating through all trajectories.
        """
        # Estimate: average ~100 steps per trajectory
        return len(self.loader) * 100
    
    @property
    def num_trajectories(self) -> int:
        """Number of trajectories in the dataset."""
        return len(self.loader)


class LiberoMapDataset(Dataset):
    """
    Map-style LIBERO Dataset for Baseline LoRA training.
    
    Loads all samples into memory for random access.
    Use this for smaller datasets or when random access is needed.
    """
    
    def __init__(
        self,
        data_dir: str,
        subset: str,
        split: str = "train",
        train_ratio: float = 0.95,
        seed: int = 42,
        action_tokenizer: Optional[ActionTokenizer] = None,
        normalize_gripper: bool = True,
        max_samples: Optional[int] = None,
    ):
        """
        Initialize the map-style dataset.
        
        Args:
            data_dir: LIBERO data root directory
            subset: LIBERO subset
            split: 'train' or 'test'
            train_ratio: Train/test split ratio
            seed: Random seed
            action_tokenizer: ActionTokenizer instance
            normalize_gripper: Whether to normalize gripper actions
            max_samples: Maximum number of samples to load (for debugging)
        """
        self.action_tokenizer = action_tokenizer or ActionTokenizer()
        self.normalize_gripper = normalize_gripper
        
        # Create iterable dataset to load samples
        iterable_dataset = LiberoBaselineDataset(
            data_dir=data_dir,
            subset=subset,
            split=split,
            train_ratio=train_ratio,
            seed=seed,
            action_tokenizer=self.action_tokenizer,
            normalize_gripper=normalize_gripper,
            shuffle=False,  # Don't shuffle during loading
        )
        
        # Load all samples into memory
        self.samples: List[Dict[str, Any]] = []
        for i, sample in enumerate(iterable_dataset):
            if max_samples and i >= max_samples:
                break
            self.samples.append(sample)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def create_libero_dataset(
    config: 'TrainingConfig',
    split: str = "train",
    action_tokenizer: Optional[ActionTokenizer] = None,
    use_map_style: bool = False,
    max_samples: Optional[int] = None,
) -> Union[LiberoBaselineDataset, LiberoMapDataset]:
    """
    Factory function to create LIBERO dataset from config.
    
    Args:
        config: TrainingConfig instance
        split: 'train' or 'test'
        action_tokenizer: ActionTokenizer instance
        use_map_style: If True, return map-style dataset
        max_samples: Maximum samples for map-style dataset
    
    Returns:
        LiberoBaselineDataset or LiberoMapDataset
    """
    from training.config import TrainingConfig
    
    if use_map_style:
        return LiberoMapDataset(
            data_dir=config.data_dir,
            subset=config.libero_subset,
            split=split,
            train_ratio=config.train_split_ratio,
            seed=config.data_seed,
            action_tokenizer=action_tokenizer,
            normalize_gripper=True,
            max_samples=max_samples,
        )
    else:
        return LiberoBaselineDataset(
            data_dir=config.data_dir,
            subset=config.libero_subset,
            split=split,
            train_ratio=config.train_split_ratio,
            seed=config.data_seed,
            action_tokenizer=action_tokenizer,
            normalize_gripper=True,
            shuffle=(split == "train"),
        )



# ============================================================================
# Text CoT Dataset (Phase B)
# ============================================================================

class LiberoTextCoTDataset(IterableDataset):
    """
    LIBERO Dataset for Text CoT LoRA training (Phase B).
    
    Extends LiberoBaselineDataset to include ECoT annotations.
    Supports reasoning dropout during training.
    
    Requirements:
        - 5.3: Use text-based ECoT tags (TASK:, PLAN:, etc.)
        - 5.4: Support reasoning dropout
    """
    
    def __init__(
        self,
        data_dir: str,
        subset: str,
        ecot_annotations_path: str,
        split: str = "train",
        train_ratio: float = 0.95,
        seed: int = 42,
        action_tokenizer: Optional[ActionTokenizer] = None,
        normalize_gripper: bool = True,
        shuffle: bool = True,
    ):
        """
        Initialize the Text CoT dataset.
        
        Args:
            data_dir: LIBERO data root directory
            subset: LIBERO subset ('spatial', 'object', 'goal', 'long')
            ecot_annotations_path: Path to ECoT annotations JSON file
            split: 'train' or 'test'
            train_ratio: Train/test split ratio
            seed: Random seed for splitting and shuffling
            action_tokenizer: ActionTokenizer instance (created if None)
            normalize_gripper: Whether to normalize gripper actions
            shuffle: Whether to shuffle data
        """
        self.data_dir = data_dir
        self.subset = subset
        self.split = split
        self.seed = seed
        self.normalize_gripper = normalize_gripper
        self.shuffle = shuffle
        
        # Create data splitter
        self.splitter = LiberoDataSplitter(
            data_dir=data_dir,
            subset=subset,
            train_ratio=train_ratio,
            seed=seed,
        )
        self.split_info = self.splitter.split()
        
        # Create batch loader for the specified split
        self.loader = LiberoBatchLoader.from_splitter(
            self.splitter,
            split=split,
            verbose=False,
        )
        
        # Create action tokenizer if not provided
        self.action_tokenizer = action_tokenizer or ActionTokenizer()
        
        # Load ECoT annotations
        self.ecot_annotations = self._load_ecot_annotations(ecot_annotations_path)
    
    def _load_ecot_annotations(self, annotations_path: str) -> Dict[str, Any]:
        """
        Load ECoT annotations from JSON file.
        
        Expected format:
        {
            "file_path": {
                "episode_id": {
                    "reasoning": {
                        "step_id": {
                            "task": "...",
                            "plan": "...",
                            "bboxes": "...",
                            "subtask_reason": "...",
                            "subtask": "...",
                            "move_reason": "...",
                            "move": "...",
                            "gripper": "..."
                        }
                    }
                }
            }
        }
        
        Args:
            annotations_path: Path to JSON file
            
        Returns:
            Dict with annotations
        """
        import json
        
        annotations_path = Path(annotations_path)
        if not annotations_path.exists():
            raise FileNotFoundError(f"ECoT annotations not found: {annotations_path}")
        
        with open(annotations_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        return annotations
    
    def _get_reasoning_for_step(
        self,
        file_path: str,
        demo_key: str,
        step_idx: int,
    ) -> Optional[Dict[str, str]]:
        """
        Get reasoning annotations for a specific step.
        
        Args:
            file_path: Source HDF5 file path
            demo_key: Demo key in HDF5 (e.g., "demo_0")
            step_idx: Step index in trajectory
            
        Returns:
            Dict with reasoning tags or None if not found
        """
        # Normalize file path for lookup
        file_key = str(Path(file_path).name)  # Use filename as key
        
        # Try different key formats
        for key in [file_path, file_key, str(Path(file_path).absolute())]:
            if key in self.ecot_annotations:
                file_annotations = self.ecot_annotations[key]
                
                # Extract episode ID from demo_key (e.g., "demo_0" -> "0")
                episode_id = demo_key.replace("demo_", "")
                
                if episode_id in file_annotations:
                    episode_data = file_annotations[episode_id]
                    
                    if 'reasoning' in episode_data:
                        step_key = str(step_idx)
                        if step_key in episode_data['reasoning']:
                            return episode_data['reasoning'][step_key]
        
        return None

    def _iterate_samples(self) -> Iterator[LiberoSample]:
        """
        Iterate over all samples in the dataset with reasoning.
        
        Yields individual timesteps from all trajectories with ECoT annotations.
        """
        # Get trajectory indices
        indices = list(range(len(self.loader)))
        
        # Shuffle trajectory order if requested
        if self.shuffle:
            rng = np.random.RandomState(self.seed)
            rng.shuffle(indices)
        
        for traj_idx in indices:
            trajectory = self.loader.get_trajectory(traj_idx)
            
            # Iterate over timesteps in trajectory
            num_steps = len(trajectory['actions'])
            for step_idx in range(num_steps):
                # Get reasoning for this step
                reasoning = self._get_reasoning_for_step(
                    trajectory['file_path'],
                    trajectory['demo_key'],
                    step_idx,
                )
                
                yield LiberoSample(
                    image=trajectory['agentview'][step_idx],
                    instruction=trajectory['instruction'],
                    action=trajectory['actions'][step_idx],
                    file_path=trajectory['file_path'],
                    demo_key=trajectory['demo_key'],
                    step_idx=step_idx,
                    reasoning=reasoning,
                )
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Iterate over processed samples ready for collation.
        
        Yields:
            Dict with 'image', 'instruction', 'action', 'action_tokens', 
            'vlm_action_string', 'reasoning'
        """
        for sample in self._iterate_samples():
            # Encode action to tokens
            action_tokens = self.action_tokenizer.encode(
                sample.action,
                normalize_gripper=self.normalize_gripper,
            )
            vlm_action_string = self.action_tokenizer.tokens_to_vlm_string(action_tokens)
            
            yield {
                'image': sample.image,
                'instruction': sample.instruction,
                'action': sample.action,
                'action_tokens': action_tokens,
                'vlm_action_string': vlm_action_string,
                'file_path': sample.file_path,
                'demo_key': sample.demo_key,
                'step_idx': sample.step_idx,
                'reasoning': sample.reasoning or {},
            }
    
    def __len__(self) -> int:
        """Return estimated total number of samples."""
        return len(self.loader) * 100
    
    @property
    def num_trajectories(self) -> int:
        """Number of trajectories in the dataset."""
        return len(self.loader)


class LiberoTextCoTMapDataset(Dataset):
    """
    Map-style LIBERO Dataset for Text CoT LoRA training.
    
    Loads all samples into memory for random access.
    """
    
    def __init__(
        self,
        data_dir: str,
        subset: str,
        ecot_annotations_path: str,
        split: str = "train",
        train_ratio: float = 0.95,
        seed: int = 42,
        action_tokenizer: Optional[ActionTokenizer] = None,
        normalize_gripper: bool = True,
        max_samples: Optional[int] = None,
    ):
        """
        Initialize the map-style Text CoT dataset.
        
        Args:
            data_dir: LIBERO data root directory
            subset: LIBERO subset
            ecot_annotations_path: Path to ECoT annotations JSON
            split: 'train' or 'test'
            train_ratio: Train/test split ratio
            seed: Random seed
            action_tokenizer: ActionTokenizer instance
            normalize_gripper: Whether to normalize gripper actions
            max_samples: Maximum number of samples to load
        """
        self.action_tokenizer = action_tokenizer or ActionTokenizer()
        self.normalize_gripper = normalize_gripper
        
        # Create iterable dataset to load samples
        iterable_dataset = LiberoTextCoTDataset(
            data_dir=data_dir,
            subset=subset,
            ecot_annotations_path=ecot_annotations_path,
            split=split,
            train_ratio=train_ratio,
            seed=seed,
            action_tokenizer=self.action_tokenizer,
            normalize_gripper=normalize_gripper,
            shuffle=False,
        )
        
        # Load all samples into memory
        self.samples: List[Dict[str, Any]] = []
        for i, sample in enumerate(iterable_dataset):
            if max_samples and i >= max_samples:
                break
            self.samples.append(sample)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def create_text_cot_dataset(
    config: 'TrainingConfig',
    ecot_annotations_path: str,
    split: str = "train",
    action_tokenizer: Optional[ActionTokenizer] = None,
    use_map_style: bool = False,
    max_samples: Optional[int] = None,
) -> Union[LiberoTextCoTDataset, LiberoTextCoTMapDataset]:
    """
    Factory function to create Text CoT LIBERO dataset from config.
    
    Args:
        config: TrainingConfig instance
        ecot_annotations_path: Path to ECoT annotations JSON
        split: 'train' or 'test'
        action_tokenizer: ActionTokenizer instance
        use_map_style: If True, return map-style dataset
        max_samples: Maximum samples for map-style dataset
    
    Returns:
        LiberoTextCoTDataset or LiberoTextCoTMapDataset
    """
    from training.config import TrainingConfig
    
    if use_map_style:
        return LiberoTextCoTMapDataset(
            data_dir=config.data_dir,
            subset=config.libero_subset,
            ecot_annotations_path=ecot_annotations_path,
            split=split,
            train_ratio=config.train_split_ratio,
            seed=config.data_seed,
            action_tokenizer=action_tokenizer,
            normalize_gripper=True,
            max_samples=max_samples,
        )
    else:
        return LiberoTextCoTDataset(
            data_dir=config.data_dir,
            subset=config.libero_subset,
            ecot_annotations_path=ecot_annotations_path,
            split=split,
            train_ratio=config.train_split_ratio,
            seed=config.data_seed,
            action_tokenizer=action_tokenizer,
            normalize_gripper=True,
            shuffle=(split == "train"),
        )


# ============================================================================
# Text + Flow CoT Dataset (Phase C)
# ============================================================================

class LiberoTextFlowCoTDataset(IterableDataset):
    """
    LIBERO Dataset for Text + Flow CoT LoRA training (Phase C).
    
    Extends LiberoTextCoTDataset to include preprocessed flow tokens.
    Supports reasoning dropout during training.
    
    Requirements:
        - 7.4: Load preprocessed flow_tokens
        - 7.5: Include flow tokens in training samples
    """
    
    def __init__(
        self,
        data_dir: str,
        subset: str,
        ecot_annotations_path: str,
        flow_tokens_path: str,
        split: str = "train",
        train_ratio: float = 0.95,
        seed: int = 42,
        action_tokenizer: Optional[ActionTokenizer] = None,
        normalize_gripper: bool = True,
        shuffle: bool = True,
    ):
        """
        Initialize the Text + Flow CoT dataset.
        
        Args:
            data_dir: LIBERO data root directory
            subset: LIBERO subset ('spatial', 'object', 'goal', 'long')
            ecot_annotations_path: Path to ECoT annotations JSON file
            flow_tokens_path: Path to preprocessed flow tokens JSON file
            split: 'train' or 'test'
            train_ratio: Train/test split ratio
            seed: Random seed for splitting and shuffling
            action_tokenizer: ActionTokenizer instance (created if None)
            normalize_gripper: Whether to normalize gripper actions
            shuffle: Whether to shuffle data
        """
        self.data_dir = data_dir
        self.subset = subset
        self.split = split
        self.seed = seed
        self.normalize_gripper = normalize_gripper
        self.shuffle = shuffle
        
        # Create data splitter
        self.splitter = LiberoDataSplitter(
            data_dir=data_dir,
            subset=subset,
            train_ratio=train_ratio,
            seed=seed,
        )
        self.split_info = self.splitter.split()
        
        # Create batch loader for the specified split
        self.loader = LiberoBatchLoader.from_splitter(
            self.splitter,
            split=split,
            verbose=False,
        )
        
        # Create action tokenizer if not provided
        self.action_tokenizer = action_tokenizer or ActionTokenizer()
        
        # Load ECoT annotations
        self.ecot_annotations = self._load_json(ecot_annotations_path, "ECoT annotations")
        
        # Load preprocessed flow tokens (Req 7.4)
        self.flow_tokens = self._load_json(flow_tokens_path, "flow tokens")
    
    def _load_json(self, json_path: str, name: str) -> Dict[str, Any]:
        """
        Load JSON file.
        
        Args:
            json_path: Path to JSON file
            name: Name for error messages
            
        Returns:
            Dict with loaded data
        """
        import json
        
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"{name} not found: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    
    def _get_reasoning_for_step(
        self,
        file_path: str,
        demo_key: str,
        step_idx: int,
    ) -> Optional[Dict[str, str]]:
        """
        Get reasoning annotations for a specific step.
        
        Args:
            file_path: Source HDF5 file path
            demo_key: Demo key in HDF5 (e.g., "demo_0")
            step_idx: Step index in trajectory
            
        Returns:
            Dict with reasoning tags or None if not found
        """
        # Normalize file path for lookup
        file_key = str(Path(file_path).name)  # Use filename as key
        
        # Try different key formats
        for key in [file_path, file_key, str(Path(file_path).absolute())]:
            if key in self.ecot_annotations:
                file_annotations = self.ecot_annotations[key]
                
                # Extract episode ID from demo_key (e.g., "demo_0" -> "0")
                episode_id = demo_key.replace("demo_", "")
                
                if episode_id in file_annotations:
                    episode_data = file_annotations[episode_id]
                    
                    if 'reasoning' in episode_data:
                        step_key = str(step_idx)
                        if step_key in episode_data['reasoning']:
                            return episode_data['reasoning'][step_key]
        
        return None
    
    def _get_flow_tokens_for_step(
        self,
        file_path: str,
        demo_key: str,
        step_idx: int,
    ) -> Optional[List[int]]:
        """
        Get preprocessed flow tokens for a specific step.
        
        Flow tokens represent the optical flow between frame t and t+1.
        
        Args:
            file_path: Source HDF5 file path
            demo_key: Demo key in HDF5 (e.g., "demo_0")
            step_idx: Step index in trajectory
            
        Returns:
            List of flow token indices or None if not found
            
        Requirements:
            - 7.4: Load preprocessed flow_tokens
        """
        # Normalize file path for lookup
        file_key = str(Path(file_path).name)  # Use filename as key
        
        # Try different key formats
        for key in [file_path, file_key, str(Path(file_path).absolute())]:
            if key in self.flow_tokens:
                file_flow_data = self.flow_tokens[key]
                
                # Extract episode ID from demo_key (e.g., "demo_0" -> "0")
                episode_id = demo_key.replace("demo_", "")
                
                if episode_id in file_flow_data:
                    episode_flow = file_flow_data[episode_id]
                    
                    # Flow tokens are stored as list of lists (one per step)
                    if 'tokens' in episode_flow:
                        tokens_list = episode_flow['tokens']
                        if step_idx < len(tokens_list):
                            return tokens_list[step_idx]
                    # Alternative format: direct list
                    elif isinstance(episode_flow, list):
                        if step_idx < len(episode_flow):
                            return episode_flow[step_idx]
        
        return None

    def _iterate_samples(self) -> Iterator[LiberoSample]:
        """
        Iterate over all samples in the dataset with reasoning and flow tokens.
        
        Yields individual timesteps from all trajectories with ECoT annotations
        and preprocessed flow tokens.
        """
        # Get trajectory indices
        indices = list(range(len(self.loader)))
        
        # Shuffle trajectory order if requested
        if self.shuffle:
            rng = np.random.RandomState(self.seed)
            rng.shuffle(indices)
        
        for traj_idx in indices:
            trajectory = self.loader.get_trajectory(traj_idx)
            
            # Iterate over timesteps in trajectory
            num_steps = len(trajectory['actions'])
            for step_idx in range(num_steps):
                # Get reasoning for this step
                reasoning = self._get_reasoning_for_step(
                    trajectory['file_path'],
                    trajectory['demo_key'],
                    step_idx,
                )
                
                # Get flow tokens for this step (Req 7.4)
                flow_tokens = self._get_flow_tokens_for_step(
                    trajectory['file_path'],
                    trajectory['demo_key'],
                    step_idx,
                )
                
                yield LiberoSample(
                    image=trajectory['agentview'][step_idx],
                    instruction=trajectory['instruction'],
                    action=trajectory['actions'][step_idx],
                    file_path=trajectory['file_path'],
                    demo_key=trajectory['demo_key'],
                    step_idx=step_idx,
                    reasoning=reasoning,
                    flow_tokens=flow_tokens,
                )
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Iterate over processed samples ready for collation.
        
        Yields:
            Dict with 'image', 'instruction', 'action', 'action_tokens', 
            'vlm_action_string', 'reasoning', 'flow_tokens'
            
        Requirements:
            - 7.5: Include flow tokens in training samples
        """
        for sample in self._iterate_samples():
            # Encode action to tokens
            action_tokens = self.action_tokenizer.encode(
                sample.action,
                normalize_gripper=self.normalize_gripper,
            )
            vlm_action_string = self.action_tokenizer.tokens_to_vlm_string(action_tokens)
            
            yield {
                'image': sample.image,
                'instruction': sample.instruction,
                'action': sample.action,
                'action_tokens': action_tokens,
                'vlm_action_string': vlm_action_string,
                'file_path': sample.file_path,
                'demo_key': sample.demo_key,
                'step_idx': sample.step_idx,
                'reasoning': sample.reasoning or {},
                'flow_tokens': sample.flow_tokens or [],
            }
    
    def __len__(self) -> int:
        """Return estimated total number of samples."""
        return len(self.loader) * 100
    
    @property
    def num_trajectories(self) -> int:
        """Number of trajectories in the dataset."""
        return len(self.loader)


class LiberoTextFlowCoTMapDataset(Dataset):
    """
    Map-style LIBERO Dataset for Text + Flow CoT LoRA training.
    
    Loads all samples into memory for random access.
    """
    
    def __init__(
        self,
        data_dir: str,
        subset: str,
        ecot_annotations_path: str,
        flow_tokens_path: str,
        split: str = "train",
        train_ratio: float = 0.95,
        seed: int = 42,
        action_tokenizer: Optional[ActionTokenizer] = None,
        normalize_gripper: bool = True,
        max_samples: Optional[int] = None,
    ):
        """
        Initialize the map-style Text + Flow CoT dataset.
        
        Args:
            data_dir: LIBERO data root directory
            subset: LIBERO subset
            ecot_annotations_path: Path to ECoT annotations JSON
            flow_tokens_path: Path to preprocessed flow tokens JSON
            split: 'train' or 'test'
            train_ratio: Train/test split ratio
            seed: Random seed
            action_tokenizer: ActionTokenizer instance
            normalize_gripper: Whether to normalize gripper actions
            max_samples: Maximum number of samples to load
        """
        self.action_tokenizer = action_tokenizer or ActionTokenizer()
        self.normalize_gripper = normalize_gripper
        
        # Create iterable dataset to load samples
        iterable_dataset = LiberoTextFlowCoTDataset(
            data_dir=data_dir,
            subset=subset,
            ecot_annotations_path=ecot_annotations_path,
            flow_tokens_path=flow_tokens_path,
            split=split,
            train_ratio=train_ratio,
            seed=seed,
            action_tokenizer=self.action_tokenizer,
            normalize_gripper=normalize_gripper,
            shuffle=False,
        )
        
        # Load all samples into memory
        self.samples: List[Dict[str, Any]] = []
        for i, sample in enumerate(iterable_dataset):
            if max_samples and i >= max_samples:
                break
            self.samples.append(sample)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def create_text_flow_cot_dataset(
    config: 'TrainingConfig',
    ecot_annotations_path: str,
    flow_tokens_path: str,
    split: str = "train",
    action_tokenizer: Optional[ActionTokenizer] = None,
    use_map_style: bool = False,
    max_samples: Optional[int] = None,
) -> Union[LiberoTextFlowCoTDataset, LiberoTextFlowCoTMapDataset]:
    """
    Factory function to create Text + Flow CoT LIBERO dataset from config.
    
    Args:
        config: TrainingConfig instance
        ecot_annotations_path: Path to ECoT annotations JSON
        flow_tokens_path: Path to preprocessed flow tokens JSON
        split: 'train' or 'test'
        action_tokenizer: ActionTokenizer instance
        use_map_style: If True, return map-style dataset
        max_samples: Maximum samples for map-style dataset
    
    Returns:
        LiberoTextFlowCoTDataset or LiberoTextFlowCoTMapDataset
        
    Requirements:
        - 7.4: Load preprocessed flow_tokens
        - 7.5: Include flow tokens in training samples
    """
    from training.config import TrainingConfig
    
    if use_map_style:
        return LiberoTextFlowCoTMapDataset(
            data_dir=config.data_dir,
            subset=config.libero_subset,
            ecot_annotations_path=ecot_annotations_path,
            flow_tokens_path=flow_tokens_path,
            split=split,
            train_ratio=config.train_split_ratio,
            seed=config.data_seed,
            action_tokenizer=action_tokenizer,
            normalize_gripper=True,
            max_samples=max_samples,
        )
    else:
        return LiberoTextFlowCoTDataset(
            data_dir=config.data_dir,
            subset=config.libero_subset,
            ecot_annotations_path=ecot_annotations_path,
            flow_tokens_path=flow_tokens_path,
            split=split,
            train_ratio=config.train_split_ratio,
            seed=config.data_seed,
            action_tokenizer=action_tokenizer,
            normalize_gripper=True,
            shuffle=(split == "train"),
        )
