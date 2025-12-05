# Filename: data_pipeline/ecot_annotator.py
"""
ECoT (Embodied Chain-of-Thought) 标注生成器

基于 embodied-CoT-main/scripts/generate_embodied_data/full_reasonings.py 适配。
支持用户配置的 LLM API（替换 Gemini API）。

Requirements: 4.1, 4.2, 4.6, 4.7, 4.8
"""

import json
import os
import re
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

from data_pipeline.primitive_movements import (
    get_move_primitives_trajectory,
    get_state_3d_positions
)


@dataclass
class LLMConfig:
    """LLM API 配置"""
    api_url: str = ""  # 通过命令行参数传入
    api_key: str = ""  # 通过命令行参数或环境变量传入
    model_name: str = "gpt-4"  # 默认模型
    max_retries: int = 8
    retry_delay: float = 5.0
    timeout: float = 60.0


@dataclass
class AnnotationConfig:
    """标注配置"""
    use_gripper_detection: bool = True  # 是否使用视觉检测 gripper 位置
    use_captions: bool = False  # 是否使用场景描述
    captions_path: Optional[str] = None  # 场景描述文件路径
    save_intermediate: bool = True  # 是否保存中间结果
    verbose: bool = True


class LLMClient:
    """通用 LLM API 客户端"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = None
    
    def _init_client(self):
        """初始化 API 客户端"""
        if self._client is not None:
            return
        
        # 支持 OpenAI 兼容的 API
        try:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.api_url if self.config.api_url else None,
                timeout=self.config.timeout
            )
        except ImportError:
            raise ImportError("请安装 openai 包: pip install openai")
    
    def generate(self, prompt: str) -> Optional[str]:
        """
        生成 LLM 响应。
        
        Args:
            prompt: 输入提示
        
        Returns:
            str: LLM 响应，如果失败返回 None
        """
        self._init_client()
        
        messages = [{"role": "user", "content": prompt}]
        full_response = ""
        
        for retry in range(self.config.max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.config.model_name,
                    messages=messages,
                    max_tokens=4096,
                    temperature=0.7
                )
                
                content = response.choices[0].message.content
                full_response += content
                
                if "FINISHED" in full_response:
                    return full_response
                
                # 如果响应被截断，继续请求
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": "Truncated, please continue."})
                
            except Exception as e:
                print(f"LLM API 错误 (重试 {retry + 1}/{self.config.max_retries}): {e}")
                time.sleep(self.config.retry_delay)
        
        return full_response if full_response else None


def build_prompt(
    features: Dict[str, List],
    language_instruction: str,
    caption: Optional[str] = None,
    list_only_moves: bool = False
) -> str:
    """
    构建 LLM 提示词。
    
    复用自 embodied-CoT full_reasonings.py 的 build_prompt 函数。
    
    Args:
        features: 轨迹特征字典
        language_instruction: 任务指令
        caption: 可选的场景描述
        list_only_moves: 是否只列出移动原语
    
    Returns:
        str: 构建的提示词
    """
    structured_features = "{\n"
    keys = list(features.keys())
    
    for i in range(len(features[keys[0]])):
        if list_only_moves:
            structured_features += f'    {i}: "{features["move_primitive"][i]}"\n'
        else:
            structured_features += f'    {i}: {{\n'
            for key in keys:
                feature_value = features[key][i]
                if isinstance(feature_value, str):
                    feature_value = f'"{feature_value}"'
                elif isinstance(feature_value, (list, tuple)):
                    feature_value = str(list(feature_value))
                structured_features += f'        "{key}": {feature_value},\n'
            structured_features += "    },\n"
    
    structured_features += "}"
    
    if list_only_moves:
        features_desc = (
            "Each entry in that dictionary corresponds to a single step on the "
            "trajectory and describes the move that is about to be executed."
        )
    else:
        features_desc = (
            "Each entry in that dictionary corresponds to a single step on "
            "the trajectory. The provided features are the following:\n"
            "\n"
            '- "state_3d" are the current 3d coordinates of the robotic arm end effector; '
            "moving forward increases the first coordinate; moving left increases the second "
            "coordinate; moving up increases the third coordinate,\n"
            '- "move_primitive" describes the move that is about to be executed,\n'
            '- "gripper_position" denotes the location of the gripper in the image observation'
        )
    
    caption_section = ""
    if caption:
        caption_section = f"""## Scene description

The robot is operating in the following environment. {caption}

"""

    return f"""# Annotate the training trajectory with reasoning

## Specification of the experimental setup

You're an expert reinforcement learning researcher. You've trained an optimal policy for controlling a robotic arm. The
robot successfully completed a task specified by the instruction: "{language_instruction}". For that purpose, the
robotic arm executed a sequence of actions. Consecutive moves that were executed are the following:


```python
trajectory_features = {structured_features}
```

{features_desc}

{caption_section}## Your objective

I want you to annotate the given trajectory with reasoning. That is, for each step, I need to know not only which action should be chosen, but importantly what reasoning justifies that action choice. I want you to be descriptive and include all the relevant information available. The reasoning should include the task to complete, the remaining high-level steps, the high-level movements that should be executed and why they are required, the premises that allow inferring the direction of each move, including the locations of relevant objects, possible obstacles or difficulties to avoid, and any other relevant justification.

### Begin by describing the task

Start by giving an overview of the task. Make it more comprehensive than the simple instruction. Include the activity, the objects the robotic arm interacts with, and their relative locations in the environment. Then, describe the high-level movements that were most likely executed, based on the task that was completed and the primitive movements that were executed. Then, for each high-level movement write the interval of steps that movement consists of. Also, for each high-level movement write a justification for why it should be executed. Write an answer for this part using markdown and natural language. Be descriptive and highlight all the relevant details, but ensure that your description is consistent with the trajectory that was executed, specified by the features listed above in the `trajectory_features` dictionary.

### List the reasonings for each step

Finally, for each step describe the reasoning that allows to determine the correct action. For each step describe the remaining part of the objective, the current progress, the objects that are still relevant for determining the plan, and the plan for the next steps, based on the available features. Start the reasoning from a high level and gradually add finer features. I need you to be descriptive and very precise. Ensure that the reasoning is consistent with the task and the executed trajectory. Write the answer for this part as a Python-executable dictionary. For every step in the initial trajectory there should be exactly one separate item of the form <step id>:<reasoning>. Do not group the answers. The final dictionary should have exactly the same set of integer keys as the dictionary of features provided in the `trajectory_features` dictionary above. The reasoning should be a single string that describes the reasoning in natural language and includes all the required features.

Each reasoning string should have the following form:
- Describe the full task that remains to be completed (but only describe what remains), and place it inside a tag <task>.
- Describe the complete high-level plan for completing the remaining task (the list of remaining high-level steps), and place it inside a tag <plan>.
- Describe the high-level step that should be executed now (chosen from the list of high-level steps), and place it inside a tag <subtask>.
- Describe why the chosen high-level step should be executed now, which features of the current environment influence that decision, and how it should be done. Place it within a tag <subtask_reason>.
- Describe the current primitive movement of the arm that needs to be executed, and place it inside a tag <move>.
- Describe why the chosen movement should be executed now and which features of the current environment influence that decision. Place it inside a tag <move_reason>.

## Task summary

Here is a breakdown of what needs to be done:

- Describe the task.
- Describe the high-level movements that were executed, based on the completed task and the listed features.
- Describe the plan for the solution that allowed the robot to complete the task successfully.
- For each step on the trajectory, describe the reasoning that leads to determining the correct action. The reasoning should be descriptive and precise. You should provide exactly one reasoning string for each step on the trajectory specified by `trajectory_features`.
- At the very end of the response, write a single label FINISHED to indicate that the answer is complete."""


def find_task_occurrences(
    input_string: str,
    tags: Tuple[str, ...] = ("task", "plan", "subtask", "subtask_reason", "move", "move_reason")
) -> List[Tuple]:
    """
    从 LLM 输出中提取标签内容。
    
    Args:
        input_string: LLM 输出字符串
        tags: 要提取的标签列表
    
    Returns:
        List[Tuple]: 匹配结果列表
    """
    pattern = r"(\d+):"
    for tag in tags:
        pattern += r"\s*<" + tag + r">([^<]*)<\/" + tag + ">"
    
    matches = re.findall(pattern, input_string, re.DOTALL)
    return matches


def extract_single_tag(text: str, tag: str) -> Optional[str]:
    """
    从文本中提取单个标签的内容。
    
    Args:
        text: 输入文本
        tag: 标签名
    
    Returns:
        str: 标签内容，如果未找到返回 None
    """
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_reasoning_dict(
    reasoning_output: Optional[str],
    tags: Tuple[str, ...] = ("task", "plan", "subtask", "subtask_reason", "move", "move_reason")
) -> Dict[int, Dict[str, str]]:
    """
    从 LLM 输出中提取推理字典。
    
    支持两种格式：
    1. 严格格式: 所有标签在同一行
    2. 宽松格式: 按 step_id 分块，逐个提取标签
    
    Args:
        reasoning_output: LLM 输出字符串
        tags: 要提取的标签列表
    
    Returns:
        Dict[int, Dict[str, str]]: {step_id: {tag: value}}
    """
    if reasoning_output is None:
        return {}
    
    trajectory = {}
    
    # 方法1: 尝试严格匹配
    matches = find_task_occurrences(reasoning_output, tags)
    if matches:
        for match in matches:
            step_id = int(match[0])
            trajectory[step_id] = dict(zip(tags, match[1:]))
        return trajectory
    
    # 方法2: 宽松匹配 - 按 step_id 分块解析
    # 匹配格式: "0:" 或 "**0:**" 或 "### 0:" 等
    step_pattern = r'(?:^|\n)\s*(?:\*\*)?(\d+)(?:\*\*)?[:\s]'
    step_matches = list(re.finditer(step_pattern, reasoning_output))
    
    for i, match in enumerate(step_matches):
        step_id = int(match.group(1))
        start_pos = match.end()
        end_pos = step_matches[i + 1].start() if i + 1 < len(step_matches) else len(reasoning_output)
        
        step_text = reasoning_output[start_pos:end_pos]
        step_reasoning = {}
        
        for tag in tags:
            content = extract_single_tag(step_text, tag)
            if content:
                step_reasoning[tag] = content
        
        if step_reasoning:
            trajectory[step_id] = step_reasoning
    
    return trajectory


def format_reasoning_as_tags(reasoning_dict: Dict[str, str]) -> str:
    """
    将推理字典格式化为 @tag@value 格式。
    
    Args:
        reasoning_dict: {tag: value} 格式的推理字典
                       键为小写标签名: task, plan, subtask, subtask_reason, move, move_reason
    
    Returns:
        str: @tag@value@tag@value... 格式的字符串
    """
    # 直接定义映射，避免导入 training 模块（可能有 peft 等依赖）
    # 小写标签名 -> 输出格式标签
    tag_mapping = {
        "task": "TASK:",
        "plan": "PLAN:",
        "subtask": "SUBTASK:",
        "subtask_reason": "SUBTASK REASONING:",
        "move": "MOVE:",
        "move_reason": "MOVE REASONING:",
    }
    
    # 输出顺序
    tags_order = ["task", "plan", "subtask", "subtask_reason", "move", "move_reason"]
    
    parts = []
    for tag in tags_order:
        if tag in reasoning_dict:
            value = reasoning_dict[tag]
            output_tag = tag_mapping.get(tag, tag.upper() + ":")
            parts.append(f"@{output_tag}@{value}")
    
    return "".join(parts)


class ECoTAnnotator:
    """
    ECoT 标注生成器。
    
    复用 embodied-CoT 代码结构，适配 LIBERO 数据格式。
    """
    
    def __init__(
        self,
        llm_config: LLMConfig,
        annotation_config: Optional[AnnotationConfig] = None
    ):
        """
        初始化标注生成器。
        
        Args:
            llm_config: LLM API 配置
            annotation_config: 标注配置
        """
        self.llm_config = llm_config
        self.annotation_config = annotation_config or AnnotationConfig()
        self.llm_client = LLMClient(llm_config)
        
        # 加载场景描述（如果配置了）
        self.captions = {}
        if self.annotation_config.use_captions and self.annotation_config.captions_path:
            if os.path.exists(self.annotation_config.captions_path):
                with open(self.annotation_config.captions_path, 'r') as f:
                    self.captions = json.load(f)
    
    def extract_features(self, trajectory: Dict[str, Any]) -> Dict[str, List]:
        """
        从轨迹中提取特征。
        
        Args:
            trajectory: LIBERO 轨迹数据
        
        Returns:
            Dict[str, List]: 特征字典
        """
        features = {}
        
        # 1. 提取 3D 状态
        features["state_3d"] = get_state_3d_positions(trajectory)
        
        # 2. 提取移动原语
        primitives = get_move_primitives_trajectory(trajectory)
        features["move_primitive"] = [p[0] for p in primitives]
        
        # 3. 提取 gripper 位置（可选）
        if self.annotation_config.use_gripper_detection:
            try:
                from data_pipeline.gripper_positions import get_gripper_positions_trajectory
                gripper_positions = get_gripper_positions_trajectory(trajectory)
                features["gripper_positions"] = gripper_positions
            except Exception as e:
                print(f"警告: 无法提取 gripper 位置: {e}")
                # 使用默认值
                n_steps = len(features["state_3d"])
                features["gripper_positions"] = [(-1, -1)] * n_steps
        
        return features
    
    def extract_metadata(self, trajectory: Dict[str, Any], traj_idx: int = 0) -> Dict[str, Any]:
        """
        从轨迹中提取元数据。
        
        Args:
            trajectory: LIBERO 轨迹数据
            traj_idx: 轨迹索引
        
        Returns:
            Dict[str, Any]: 元数据字典
        """
        metadata = {
            "episode_id": str(traj_idx),
            "file_path": trajectory.get("file_path", "unknown"),
            "n_steps": len(trajectory.get("actions", [])),
            "language_instruction": trajectory.get("instruction", ""),
        }
        
        # 添加场景描述（如果有）
        if self.captions:
            file_path = metadata["file_path"]
            episode_id = metadata["episode_id"]
            if file_path in self.captions and episode_id in self.captions[file_path]:
                metadata["caption"] = self.captions[file_path][episode_id].get("caption", "")
        
        return metadata
    
    def get_reasoning_dict(
        self,
        features: Dict[str, List],
        metadata: Dict[str, Any]
    ) -> Dict[int, Dict[str, str]]:
        """
        调用 LLM 生成推理。
        
        Args:
            features: 轨迹特征
            metadata: 轨迹元数据
        
        Returns:
            Dict[int, Dict[str, str]]: {step_id: {tag: value}}
        """
        language_instruction = metadata.get("language_instruction", "")
        caption = metadata.get("caption")
        
        prompt = build_prompt(
            features,
            language_instruction,
            caption=caption,
            list_only_moves=True  # 简化版本，只使用移动原语
        )
        
        if self.annotation_config.verbose:
            print(f"生成推理: {language_instruction}")
        
        reasoning_output = self.llm_client.generate(prompt)
        
        if self.annotation_config.verbose and reasoning_output:
            print(f"LLM 响应长度: {len(reasoning_output)}")
            # 保存原始响应用于调试
            debug_path = "annotations/debug_llm_response.txt"
            os.makedirs(os.path.dirname(debug_path), exist_ok=True)
            with open(debug_path, 'w', encoding='utf-8') as f:
                f.write(reasoning_output)
            print(f"LLM 原始响应已保存到: {debug_path}")
        
        result = extract_reasoning_dict(reasoning_output)
        if self.annotation_config.verbose:
            print(f"解析到 {len(result)} 个步骤的推理")
        
        return result
    
    def annotate_trajectory(
        self,
        trajectory: Dict[str, Any],
        traj_idx: int = 0
    ) -> Dict[str, Any]:
        """
        为单条轨迹生成 ECoT 标注。
        
        Args:
            trajectory: LIBERO 轨迹数据
            traj_idx: 轨迹索引
        
        Returns:
            Dict[str, Any]: 标注结果
        """
        # 提取特征和元数据
        features = self.extract_features(trajectory)
        metadata = self.extract_metadata(trajectory, traj_idx)
        
        # 调用 LLM 生成推理
        reasoning = self.get_reasoning_dict(features, metadata)
        
        # 转换为 @tag@value 格式
        reasoning_formatted = {}
        for step_id, step_reasoning in reasoning.items():
            reasoning_formatted[step_id] = format_reasoning_as_tags(step_reasoning)
        
        return {
            "reasoning": reasoning,
            "reasoning_formatted": reasoning_formatted,
            "features": {
                "state_3d": features["state_3d"],
                "move_primitive": features["move_primitive"],
                "gripper_positions": features.get("gripper_positions", [])
            },
            "metadata": metadata
        }
    
    def annotate_batch(
        self,
        trajectories: List[Dict[str, Any]],
        save_path: Optional[str] = None,
        resume: bool = True
    ) -> Dict[str, Dict]:
        """
        批量标注轨迹。
        
        Args:
            trajectories: 轨迹列表
            save_path: 保存路径
            resume: 是否从已有结果继续
        
        Returns:
            Dict[str, Dict]: {file_path: {episode_id: annotation}}
        """
        annotations = {}
        
        # 加载已有结果
        if resume and save_path and os.path.exists(save_path):
            print(f"加载已有标注: {save_path}")
            with open(save_path, 'r') as f:
                annotations = json.load(f)
            print(f"已加载 {sum(len(v) for v in annotations.values())} 条标注")
        
        for i, trajectory in enumerate(trajectories):
            file_path = trajectory.get("file_path", f"unknown_{i}")
            episode_id = str(i)
            
            # 检查是否已标注
            if file_path in annotations and episode_id in annotations[file_path]:
                if self.annotation_config.verbose:
                    print(f"跳过已标注: {file_path} / {episode_id}")
                continue
            
            try:
                entry = self.annotate_trajectory(trajectory, i)
                
                if file_path not in annotations:
                    annotations[file_path] = {}
                annotations[file_path][episode_id] = entry
                
                if self.annotation_config.verbose:
                    print(f"完成标注 [{i + 1}/{len(trajectories)}]: {file_path}")
                
                # 保存中间结果
                if self.annotation_config.save_intermediate and save_path:
                    with open(save_path, 'w') as f:
                        json.dump(annotations, f, indent=2)
                        
            except Exception as e:
                print(f"标注失败 [{i}]: {e}")
                continue
        
        # 最终保存
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(annotations, f, indent=2)
            print(f"标注已保存: {save_path}")
        
        return annotations
    
    @staticmethod
    def load_annotations(path: str) -> Dict[str, Dict]:
        """
        加载标注文件。
        
        Args:
            path: 标注文件路径
        
        Returns:
            Dict[str, Dict]: 标注数据
        """
        with open(path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def save_annotations(annotations: Dict[str, Dict], path: str):
        """
        保存标注文件。
        
        Args:
            annotations: 标注数据
            path: 保存路径
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(annotations, f, indent=2)
