"""
LIBERO Evaluation Module for NORA LoRA Training.

Implements evaluation system for LIBERO benchmark with support for:
- Loading NORA base + LoRA weights (LoRA_B, LoRA_T, LoRA_TF)
- Running rollout evaluation in LIBERO simulation
- Computing success rate metrics

Requirements covered:
- 8.1: Load NORA base with specified LoRA weights
- 8.2: Evaluate using No CoT input format
- 8.3: Run rollout evaluation using NORA protocol
- 8.4: Use LIBERO-specific action unnormalization (libero_keys)
- 8.5: Apply gripper action normalization and inversion
- 8.6: Compute success rate with 50 trials per task
- 8.7: Use max_steps=500 and num_steps_wait=10
- 8.8: Evaluate each model on corresponding LIBERO subset
- 8.9: Support comparison of 12 models (3 types × 4 subsets)
- 8.10: Support debug mode with Full CoT generation
"""
import os
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime

import numpy as np
import torch
import PIL.Image
from tqdm import tqdm

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, GenerationConfig
from peft import PeftModel
from huggingface_hub import hf_hub_download

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None

# Try to import LIBERO - handle different installation scenarios
LIBERO_AVAILABLE = False
benchmark = None
OffScreenRenderEnv = None

try:
    # Standard installation: pip install -e . from LIBERO directory
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    LIBERO_AVAILABLE = True
except ImportError:
    # Try adding local LIBERO directory to path
    import sys
    _libero_path = os.path.join(os.path.dirname(__file__), "..", "LIBERO")
    if os.path.exists(_libero_path) and _libero_path not in sys.path:
        sys.path.insert(0, _libero_path)
        try:
            from libero.libero import benchmark
            from libero.libero.envs import OffScreenRenderEnv
            LIBERO_AVAILABLE = True
            print(f"Loaded LIBERO from local path: {_libero_path}")
        except ImportError as e:
            print(f"Warning: Could not import LIBERO: {e}")
            print("Please install LIBERO: cd LIBERO && pip install -e .")


# Action token ID range consistent with NORA
ACTION_TOKEN_MIN = 151665
ACTION_TOKEN_MAX = 153712


@dataclass
class EvalConfig:
    """Configuration for LIBERO evaluation."""
    # Model paths
    base_model_path: str = "declare-lab/nora"
    lora_path: Optional[str] = None
    
    # Model type: 'baseline' | 'text_cot' | 'text_flow_cot'
    model_type: str = "baseline"
    
    # LIBERO subset: 'spatial' | 'object' | 'goal' | '10' | '10' | '90'
    libero_subset: str = "object"
    
    # Evaluation parameters (Req 8.6, 8.7)
    num_trials_per_task: int = 50
    max_steps: int = 500
    num_steps_wait: int = 10
    
    # Debug mode (Req 8.10)
    debug_mode: bool = False
    save_videos: bool = False
    generate_full_cot: bool = False
    
    # Output
    output_dir: str = "./eval_results"
    log_to_wandb: bool = False
    wandb_project: str = "nora-lora-eval"
    wandb_entity: Optional[str] = None
    
    # Device
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    
    # Random seed
    seed: int = 7


# LIBERO action normalization statistics (from nora_utils.py)
# Format: {subset_name: [action_low, action_high]}
LIBERO_KEYS = {
    'libero_object': [
        np.array([
            -0.5383928418159485, -0.8758928775787354, -0.9375,
            -0.06964285671710968, -0.11678571254014969, -0.15964286029338837, 0.0
        ]),
        np.array([
            0.8464285731315613, 0.84375, 0.9375,
            0.08142857253551483, 0.14892856776714325, 0.0867857113480568, 1.0
        ])
    ],
    'libero_spatial': [
        np.array([
            -0.7454732114076613, -0.6616071462631226, -0.9375,
            -0.1071428582072258, -0.20678570866584778, -0.1842857152223587, 0.0
        ]),
        np.array([
            0.9375, 0.8758928775787354, 0.9321428537368774,
            0.1039285734295845, 0.17678570747375488, 0.14571428298950195, 1.0
        ])
    ],
    'libero_goal': [
        np.array([
            -0.8785714507102966, -0.7553571462631226, -0.9375,
            -0.1510714292526245, -0.1639285683631897, -0.13777500048279764, 0.0
        ]),
        np.array([
            0.9375, 0.9107142686843872, 0.9375,
            0.20357142388820648, 0.26357144117355347, 0.375, 1.0
        ])
    ],
    'libero_10': [
        np.array([
            -0.6348214149475098, -0.7741071581840515, -0.7633928656578064,
            -0.09749999642372131, -0.14819999992847435, -0.2742857038974762, 0.0
        ]),
        np.array([
            0.7714285850524902, 0.8464285731315613, 0.9375,
            0.13928571343421936, 0.15964286029338837, 0.3246428668498993, 1.0
        ])
    ],
    'libero_90': [
        np.array([
            -0.6348214149475098, -0.7741071581840515, -0.7633928656578064,
            -0.09749999642372131, -0.14819999992847435, -0.2742857038974762, 0.0
        ]),
        np.array([
            0.7714285850524902, 0.8464285731315613, 0.9375,
            0.13928571343421936, 0.15964286029338837, 0.3246428668498993, 1.0
        ])
    ],
    # Alias for '10' subset (maps to libero_10)
    'libero_10': [
        np.array([
            -0.6348214149475098, -0.7741071581840515, -0.7633928656578064,
            -0.09749999642372131, -0.14819999992847435, -0.2742857038974762, 0.0
        ]),
        np.array([
            0.7714285850524902, 0.8464285731315613, 0.9375,
            0.13928571343421936, 0.15964286029338837, 0.3246428668498993, 1.0
        ])
    ],
}


def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    """
    Normalize gripper action from [0,1] to [-1,+1].
    
    Consistent with nora/experiments/libero/nora_utils.py
    
    Args:
        action: Action array with gripper in last dimension
        binarize: If True, binarize gripper to -1 or +1
        
    Returns:
        Action array with normalized gripper
        
    Requirements:
        - 8.5: Apply gripper action normalization
    """
    action = action.copy()
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1
    
    if binarize:
        action[..., -1] = np.sign(action[..., -1])
    
    return action


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    """
    Flip the sign of the gripper action.
    
    Consistent with nora/experiments/libero/nora_utils.py
    
    Args:
        action: Action array with gripper in last dimension
        
    Returns:
        Action array with inverted gripper
        
    Requirements:
        - 8.5: Apply gripper action inversion
    """
    action = action.copy()
    action[..., -1] = action[..., -1] * -1.0
    return action


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to axis-angle representation.
    
    Args:
        quat: Quaternion array [x, y, z, w]
        
    Returns:
        Axis-angle array [ax, ay, az]
    """
    # Normalize quaternion
    quat = quat / np.linalg.norm(quat)
    
    # Extract angle
    angle = 2 * np.arccos(np.clip(quat[3], -1, 1))
    
    # Extract axis
    sin_half = np.sin(angle / 2)
    if sin_half < 1e-6:
        return np.zeros(3)
    
    axis = quat[:3] / sin_half
    return axis * angle


def get_libero_dummy_action(model_family: str = "openvla") -> List[float]:
    """Get dummy action for LIBERO environment stabilization."""
    return [0.0] * 7


def get_libero_image(obs: Dict, resize_size: Tuple[int, int]) -> np.ndarray:
    """
    Extract and resize image from LIBERO observation.
    
    Args:
        obs: LIBERO observation dictionary
        resize_size: Target size (height, width)
        
    Returns:
        Resized RGB image as numpy array
    """
    # Get agentview image
    img = obs.get("agentview_image", obs.get("agentview_rgb"))
    
    if img is None:
        raise ValueError("No image found in observation")
    
    # Convert to PIL for resizing
    pil_img = PIL.Image.fromarray(img)
    pil_img = pil_img.resize((resize_size[1], resize_size[0]), PIL.Image.BILINEAR)
    
    return np.array(pil_img)


def get_libero_env(task, model_family: str = "openvla", resolution: int = 256):
    """
    Create LIBERO environment for a task.
    
    Args:
        task: LIBERO task object
        model_family: Model family string
        resolution: Image resolution
        
    Returns:
        Tuple of (environment, task_description)
    """
    if not LIBERO_AVAILABLE:
        raise ImportError("LIBERO is not installed. Please install it first.")
    
    task_description = task.language
    task_bddl_file = os.path.join(
        os.path.dirname(__file__),
        "..", "LIBERO", "libero", "libero", "bddl_files",
        task.problem_folder, task.bddl_file
    )
    
    # Try to find the bddl file
    if not os.path.exists(task_bddl_file):
        # Try alternative path
        from libero.libero import get_libero_path
        task_bddl_file = os.path.join(
            get_libero_path("bddl_files"),
            task.problem_folder,
            task.bddl_file
        )
    
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    
    return env, task_description


def save_rollout_video(
    images: List[np.ndarray],
    episode_idx: int,
    success: bool,
    task_description: str,
    output_dir: str,
    log_file=None
) -> None:
    """
    Save rollout video from images.
    
    Args:
        images: List of RGB images
        episode_idx: Episode index
        success: Whether episode was successful
        task_description: Task description string
        output_dir: Output directory
        log_file: Optional log file handle
    """
    try:
        import imageio
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename
        status = "success" if success else "fail"
        safe_task = task_description.replace(" ", "_")[:50]
        filename = f"episode_{episode_idx}_{status}_{safe_task}.mp4"
        filepath = os.path.join(output_dir, filename)
        
        # Save video
        imageio.mimsave(filepath, images, fps=30)
        
        if log_file:
            log_file.write(f"Saved video: {filepath}\n")
            
    except ImportError:
        print("imageio not installed, skipping video save")
    except Exception as e:
        print(f"Error saving video: {e}")


class LiberoEvaluator:
    """
    LIBERO Evaluator for NORA + LoRA models.
    
    Supports evaluation of:
    - LoRA_B (Baseline)
    - LoRA_T (Text CoT)
    - LoRA_TF (Text + Flow CoT)
    
    Requirements:
        - 8.1: Load NORA base with specified LoRA weights
        - 8.2: Evaluate using No CoT input format
        - 8.3: Run rollout evaluation using NORA protocol
    """
    
    def __init__(self, config: EvalConfig):
        """
        Initialize the evaluator.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.device = config.device
        
        # Set dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self.torch_dtype = dtype_map.get(config.torch_dtype, torch.bfloat16)
        
        # Load model and processor
        self.model, self.processor = self._load_model()
        
        # Load FAST+ tokenizer
        self.fast_tokenizer = self._load_fast_tokenizer()
        
        # Get LIBERO normalization keys (Req 8.4)
        self.libero_keys = self._get_libero_keys()
        
        # Setup output directory
        os.makedirs(config.output_dir, exist_ok=True)
    
    def _load_model(self) -> Tuple[Any, AutoProcessor]:
        """
        Load NORA base model with optional LoRA weights.
        
        Returns:
            Tuple of (model, processor)
            
        Requirements:
            - 8.1: Load NORA base with specified LoRA weights
        """
        print(f"Loading base model from: {self.config.base_model_path}")
        
        # Load base model
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.config.base_model_path,
            torch_dtype=self.torch_dtype,
            # attn_implementation="flash_attention_2",  # Uncomment if flash attention available
        )
        
        # Load LoRA weights if provided
        if self.config.lora_path:
            print(f"Loading LoRA weights from: {self.config.lora_path}")
            model = PeftModel.from_pretrained(model, self.config.lora_path)
            print(f"LoRA config: {model.peft_config}")
        
        model.to(self.device)
        model.eval()
        
        # Setup generation config
        model.generation_config = GenerationConfig.from_pretrained(self.config.base_model_path)
        model.generation_config.do_sample = False
        
        # Load processor
        processor = AutoProcessor.from_pretrained(self.config.base_model_path)
        
        return model, processor
    
    def _load_fast_tokenizer(self) -> AutoProcessor:
        """Load FAST+ tokenizer for action decoding."""
        fast_tokenizer = AutoProcessor.from_pretrained(
            "physical-intelligence/fast",
            trust_remote_code=True
        )
        fast_tokenizer.action_dim = 7
        # IMPORTANT: time_horizon must match training setting
        # Training uses time_horizon=1, so evaluation must also use 1
        fast_tokenizer.time_horizon = 1
        return fast_tokenizer
    
    def _get_libero_keys(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get LIBERO normalization statistics for the current subset.
        
        Returns:
            Tuple of (action_low, action_high)
            
        Requirements:
            - 8.4: Use LIBERO-specific action unnormalization
        """
        subset_key = f"libero_{self.config.libero_subset}"
        if subset_key not in LIBERO_KEYS:
            # Try without prefix
            subset_key = self.config.libero_subset
        
        if subset_key not in LIBERO_KEYS:
            raise ValueError(f"Unknown LIBERO subset: {self.config.libero_subset}")
        
        return LIBERO_KEYS[subset_key]
    
    def _get_task_suite_name(self) -> str:
        """Get LIBERO task suite name from config subset."""
        subset = self.config.libero_subset.lower()
        
        # Map subset names to LIBERO benchmark names
        mapping = {
            'spatial': 'libero_spatial',
            'object': 'libero_object',
            'goal': 'libero_goal',
            '10': 'libero_10',  # LIBERO-10 is libero_10
            '10': 'libero_10',
            '90': 'libero_90',
        }
        
        return mapping.get(subset, f"libero_{subset}")
    
    @torch.inference_mode()
    def inference(
        self,
        image: np.ndarray,
        instruction: str,
        generate_cot: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Run inference to get action from image and instruction.
        
        Args:
            image: RGB image as numpy array
            instruction: Task instruction string
            generate_cot: If True, generate full CoT and return it (for debug mode)
            
        Returns:
            If generate_cot is False: Unnormalized action array
            If generate_cot is True: Tuple of (action array, cot_info dict)
            
        Requirements:
            - 8.2: Evaluate using No CoT input format
            - 8.10: Support debug mode with Full CoT generation
        """
        # Convert to PIL Image
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)
        
        # Build message (No CoT format - Req 8.2)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                        "resized_height": 224,
                        "resized_width": 224,
                    },
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Process vision info
        if process_vision_info:
            image_inputs, video_inputs = process_vision_info(messages)
        else:
            image_inputs = [image]
            video_inputs = None
        
        # Prepare inputs
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate with 10er max_new_tokens for CoT if needed
        max_new_tokens = 512 if generate_cot else 256
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        
        # Extract action tokens
        action_mask = (ACTION_TOKEN_MIN <= generated_ids[0]) & (generated_ids[0] <= ACTION_TOKEN_MAX)
        action_indices = torch.where(action_mask)[0]
        
        # Debug: print token info
        print(f"[DEBUG] Total generated tokens: {len(generated_ids[0])}")
        print(f"[DEBUG] Action tokens found: {len(action_indices)}")
        if len(action_indices) > 0:
            print(f"[DEBUG] First 10 action token IDs: {generated_ids[0][action_indices][:10].tolist()}")
        
        if len(action_indices) == 0:
            print("Warning: No action tokens generated, returning zero action")
            action = np.zeros((1, 7))
            if generate_cot:
                return action, {"raw_output": "", "cot_text": "", "action_tokens": []}
            return action
        
        # Decode action tokens
        action_token_ids = generated_ids[0][action_indices] - ACTION_TOKEN_MIN
        output_action = self.fast_tokenizer.decode([action_token_ids])
        
        # Unnormalize action (Req 8.4)
        action_low, action_high = self.libero_keys
        unnorm_action = (
            0.5 * (output_action + 1) * (action_high - action_low) + action_low
        )
        
        action = np.array(unnorm_action[0])
        
        # If debug mode, also return CoT information
        if generate_cot:
            # Decode full output for CoT analysis
            full_output = self.processor.decode(
                generated_ids[0], 
                skip_special_tokens=False
            )
            
            # Extract CoT text (everything before ACTION:)
            cot_text = ""
            if "ACTION:" in full_output:
                cot_text = full_output.split("ACTION:")[0]
            
            # Parse CoT tags
            cot_info = self._parse_cot_output(full_output)
            cot_info["raw_output"] = full_output
            cot_info["cot_text"] = cot_text
            cot_info["action_tokens"] = action_token_ids.cpu().numpy().tolist()
            
            return action, cot_info
        
        return action
    
    def _parse_cot_output(self, output: str) -> Dict[str, Any]:
        """
        Parse CoT output to extract individual tags.
        
        Args:
            output: Raw model output string
            
        Returns:
            Dictionary with parsed CoT components
            
        Requirements:
            - 8.10: Support debug mode with Full CoT generation
        """
        cot_tags = [
            "TASK:", "PLAN:", "VISIBLE OBJECTS:", 
            "SUBTASK REASONING:", "SUBTASK:", 
            "MOVE REASONING:", "MOVE:", 
            "GRIPPER POSITION:", "FLOW:", "ACTION:"
        ]
        
        parsed = {}
        
        for i, tag in enumerate(cot_tags):
            if tag in output:
                # Find start of this tag's content
                start = output.find(tag) + len(tag)
                
                # Find end (start of next tag or end of string)
                end = len(output)
                for next_tag in cot_tags[i+1:]:
                    if next_tag in output:
                        next_pos = output.find(next_tag)
                        if next_pos > start:
                            end = min(end, next_pos)
                            break
                
                # Extract and clean content
                content = output[start:end].strip()
                tag_key = tag.rstrip(":").lower().replace(" ", "_")
                parsed[tag_key] = content
        
        return parsed
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Run full evaluation on LIBERO benchmark.
        
        Returns:
            Dictionary with evaluation results
            
        Requirements:
            - 8.3: Run rollout evaluation using NORA protocol
            - 8.6: Compute success rate with 50 trials per task
            - 8.7: Use max_steps=500 and num_steps_wait=10
            - 8.8: Evaluate on corresponding LIBERO subset
        """
        if not LIBERO_AVAILABLE:
            raise ImportError("LIBERO is not installed. Please install it first.")
        
        # Set random seed
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        
        # Get task suite
        task_suite_name = self._get_task_suite_name()
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[task_suite_name]()
        num_tasks = task_suite.n_tasks
        
        print(f"Evaluating on {task_suite_name} with {num_tasks} tasks")
        
        # Setup logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"EVAL-{task_suite_name}-{self.config.model_type}-{timestamp}"
        log_filepath = os.path.join(self.config.output_dir, f"{run_id}.txt")
        log_file = open(log_filepath, "w")
        
        # Results storage
        results = {
            "task_suite": task_suite_name,
            "model_type": self.config.model_type,
            "lora_path": self.config.lora_path,
            "task_results": {},
            "total_episodes": 0,
            "total_successes": 0,
        }
        
        total_episodes, total_successes = 0, 0
        
        # Evaluate each task
        for task_id in tqdm(range(num_tasks), desc="Tasks"):
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            
            # Create environment
            print(f"\n[Task {task_id}] Creating environment...")
            env, task_description = get_libero_env(task, resolution=256)
            print(f"[Task {task_id}] Task: {task_description}")
            
            task_episodes, task_successes = 0, 0
            
            # Run trials (Req 8.6)
            for episode_idx in tqdm(range(self.config.num_trials_per_task), 
                                   desc=f"Task {task_id}", leave=False):
                
                print(f"  Episode {episode_idx + 1}/{self.config.num_trials_per_task}", end=" ", flush=True)
                log_file.write(f"\nTask: {task_description}\n")
                log_file.write(f"Episode: {episode_idx + 1}\n")
                
                # Reset environment
                env.reset()
                obs = env.set_init_state(initial_states[episode_idx])
                
                t = 0
                replay_images = []
                done = False
                
                # Run episode (Req 8.7)
                while t < self.config.max_steps + self.config.num_steps_wait:
                    try:
                        # Wait for objects to stabilize
                        if t < self.config.num_steps_wait:
                            obs, reward, done, info = env.step(get_libero_dummy_action())
                            t += 1
                            continue
                        
                        # Get image
                        img = get_libero_image(obs, (224, 224))
                        
                        if self.config.save_videos:
                            replay_images.append(img)
                        
                        # Get action from model
                        action = self.inference(
                            img, 
                            task_description,
                            generate_cot=self.config.generate_full_cot
                        )
                        
                        # Debug: print raw action
                        print(f"[DEBUG] Raw action from model: {action}")
                        
                        # Post-process action (Req 8.5)
                        action = normalize_gripper_action(action, binarize=True)
                        action = invert_gripper_action(action)
                        
                        # Debug: print processed action
                        print(f"[DEBUG] After gripper processing: {action}")
                        
                        # Binarize gripper for execution
                        action[..., -1] = np.where(action[..., -1] >= 0.0, 1.0, action[..., -1])
                        
                        # Execute action(s)
                        for i in range(len(action)):
                            obs, reward, done, info = env.step(action[i].tolist())
                            
                            if done:
                                task_successes += 1
                                total_successes += 1
                                break
                        
                        if done:
                            break
                        
                        t += 1
                        
                    except Exception as e:
                        log_file.write(f"Exception: {e}\n")
                        print(f"Exception during episode: {e}")
                        break
                
                task_episodes += 1
                total_episodes += 1
                
                # Save video if enabled
                if self.config.save_videos and replay_images:
                    video_dir = os.path.join(self.config.output_dir, "videos")
                    save_rollout_video(
                        replay_images, total_episodes, done,
                        task_description, video_dir, log_file
                    )
                
                # Log progress
                log_file.write(f"Success: {done}\n")
                log_file.write(f"Total: {total_successes}/{total_episodes} "
                             f"({100*total_successes/total_episodes:.1f}%)\n")
                log_file.flush()
            
            # Store task results
            task_success_rate = task_successes / task_episodes if task_episodes > 0 else 0
            results["task_results"][task_description] = {
                "success_rate": task_success_rate,
                "successes": task_successes,
                "episodes": task_episodes,
            }
            
            print(f"Task '{task_description}': {task_success_rate:.2%}")
            
            # Cleanup
            env.close()
        
        # Compute overall results
        results["total_episodes"] = total_episodes
        results["total_successes"] = total_successes
        results["overall_success_rate"] = total_successes / total_episodes if total_episodes > 0 else 0
        
        # Save results
        results_path = os.path.join(self.config.output_dir, f"{run_id}_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        log_file.write(f"\n\nFinal Results:\n")
        log_file.write(f"Overall Success Rate: {results['overall_success_rate']:.2%}\n")
        log_file.close()
        
        print(f"\nEvaluation complete!")
        print(f"Overall Success Rate: {results['overall_success_rate']:.2%}")
        print(f"Results saved to: {results_path}")
        
        return results
    
    def evaluate_single_task(self, task_id: int) -> Dict[str, Any]:
        """
        Evaluate a single task (useful for debugging).
        
        Args:
            task_id: Task index in the benchmark
            
        Returns:
            Dictionary with task evaluation results
        """
        if not LIBERO_AVAILABLE:
            raise ImportError("LIBERO is not installed.")
        
        task_suite_name = self._get_task_suite_name()
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[task_suite_name]()
        
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        
        env, task_description = get_libero_env(task, resolution=256)
        
        successes = 0
        for episode_idx in tqdm(range(self.config.num_trials_per_task)):
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])
            
            t = 0
            done = False
            
            while t < self.config.max_steps + self.config.num_steps_wait:
                if t < self.config.num_steps_wait:
                    obs, _, done, _ = env.step(get_libero_dummy_action())
                    t += 1
                    continue
                
                img = get_libero_image(obs, (224, 224))
                action = self.inference(img, task_description)
                action = normalize_gripper_action(action, binarize=True)
                action = invert_gripper_action(action)
                action[..., -1] = np.where(action[..., -1] >= 0.0, 1.0, action[..., -1])
                
                for i in range(len(action)):
                    obs, _, done, _ = env.step(action[i].tolist())
                    if done:
                        successes += 1
                        break
                
                if done:
                    break
                t += 1
        
        env.close()
        
        return {
            "task_description": task_description,
            "success_rate": successes / self.config.num_trials_per_task,
            "successes": successes,
            "episodes": self.config.num_trials_per_task,
        }
    
    def evaluate_with_debug(
        self,
        task_id: int = 0,
        episode_idx: int = 0,
        save_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run a single episode with full debug information.
        
        This method generates Full CoT at each step and saves detailed
        information for visualization and analysis.
        
        Args:
            task_id: Task index in the benchmark
            episode_idx: Episode index for initial state
            save_dir: Directory to save debug outputs
            
        Returns:
            Dictionary with detailed debug information
            
        Requirements:
            - 8.10: Support debug mode with Full CoT generation
        """
        if not LIBERO_AVAILABLE:
            raise ImportError("LIBERO is not installed.")
        
        if save_dir is None:
            save_dir = os.path.join(self.config.output_dir, "debug")
        os.makedirs(save_dir, exist_ok=True)
        
        task_suite_name = self._get_task_suite_name()
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[task_suite_name]()
        
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        
        env, task_description = get_libero_env(task, resolution=256)
        
        # Reset environment
        env.reset()
        obs = env.set_init_state(initial_states[episode_idx])
        
        # Storage for debug info
        debug_info = {
            "task_description": task_description,
            "task_id": task_id,
            "episode_idx": episode_idx,
            "steps": [],
            "images": [],
            "success": False,
        }
        
        t = 0
        done = False
        
        print(f"Running debug episode for task: {task_description}")
        
        while t < self.config.max_steps + self.config.num_steps_wait:
            if t < self.config.num_steps_wait:
                obs, _, done, _ = env.step(get_libero_dummy_action())
                t += 1
                continue
            
            # Get image
            img = get_libero_image(obs, (224, 224))
            debug_info["images"].append(img)
            
            # Run inference with CoT generation
            action, cot_info = self.inference(
                img, 
                task_description, 
                generate_cot=True
            )
            
            # Store step info
            step_info = {
                "step": t - self.config.num_steps_wait,
                "cot": cot_info,
                "raw_action": action.tolist(),
            }
            
            # Post-process action
            action = normalize_gripper_action(action, binarize=True)
            action = invert_gripper_action(action)
            action[..., -1] = np.where(action[..., -1] >= 0.0, 1.0, action[..., -1])
            
            step_info["processed_action"] = action.tolist()
            debug_info["steps"].append(step_info)
            
            # Print CoT info
            if cot_info.get("cot_text"):
                print(f"\nStep {step_info['step']}:")
                print(f"CoT: {cot_info['cot_text'][:200]}...")
            
            # Execute action
            for i in range(len(action)):
                obs, _, done, _ = env.step(action[i].tolist())
                if done:
                    break
            
            if done:
                debug_info["success"] = True
                print(f"\nSuccess at step {t}!")
                break
            
            t += 1
        
        env.close()
        
        # Save debug info
        debug_path = os.path.join(
            save_dir, 
            f"debug_task{task_id}_ep{episode_idx}.json"
        )
        
        # Convert numpy arrays for JSON serialization
        serializable_info = {
            k: v for k, v in debug_info.items() 
            if k != "images"
        }
        
        with open(debug_path, "w") as f:
            json.dump(serializable_info, f, indent=2)
        
        # Save video if images were collected
        if debug_info["images"]:
            video_path = os.path.join(
                save_dir,
                f"debug_task{task_id}_ep{episode_idx}.mp4"
            )
            save_rollout_video(
                debug_info["images"],
                episode_idx,
                debug_info["success"],
                task_description,
                save_dir
            )
        
        print(f"\nDebug info saved to: {debug_path}")
        
        return debug_info
