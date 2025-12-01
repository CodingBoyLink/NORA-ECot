# Filename: training/utils/cot_utils.py
"""
CoT (Chain-of-Thought) 工具模块

复用自 embodied-CoT-main/prismatic/util/cot_utils.py
"""

import enum
from typing import List, Dict


class CotTag(enum.Enum):
    """ECoT 标签枚举"""
    TASK = "TASK:"
    PLAN = "PLAN:"
    VISIBLE_OBJECTS = "VISIBLE OBJECTS:"
    SUBTASK_REASONING = "SUBTASK REASONING:"
    SUBTASK = "SUBTASK:"
    MOVE_REASONING = "MOVE REASONING:"
    MOVE = "MOVE:"
    GRIPPER_POSITION = "GRIPPER POSITION:"
    ACTION = "ACTION:"


def abbreviate_tag(tag: str) -> str:
    """
    缩写标签名称。
    
    Args:
        tag: 完整标签名
    
    Returns:
        str: 缩写后的标签
    """
    return tag[0] + tag[-2]


def get_cot_tags_list() -> List[str]:
    """
    获取 CoT 标签列表（按顺序）。
    
    Returns:
        List[str]: 标签值列表
    """
    return [
        CotTag.TASK.value,
        CotTag.PLAN.value,
        CotTag.VISIBLE_OBJECTS.value,
        CotTag.SUBTASK_REASONING.value,
        CotTag.SUBTASK.value,
        CotTag.MOVE_REASONING.value,
        CotTag.MOVE.value,
        CotTag.GRIPPER_POSITION.value,
        CotTag.ACTION.value,
    ]


def get_cot_database_keys() -> Dict[str, str]:
    """
    获取 CoT 标签到数据库键的映射。
    
    Returns:
        Dict[str, str]: {tag_value: database_key}
    """
    return {
        CotTag.TASK.value: "task",
        CotTag.PLAN.value: "plan",
        CotTag.VISIBLE_OBJECTS.value: "bboxes",
        CotTag.SUBTASK_REASONING.value: "subtask_reason",
        CotTag.SUBTASK.value: "subtask",
        CotTag.MOVE_REASONING.value: "move_reason",
        CotTag.MOVE.value: "move",
        CotTag.GRIPPER_POSITION.value: "gripper",
        CotTag.ACTION.value: "action",
    }


def get_database_to_cot_keys() -> Dict[str, str]:
    """
    获取数据库键到 CoT 标签的映射。
    
    Returns:
        Dict[str, str]: {database_key: tag_value}
    """
    return {v: k for k, v in get_cot_database_keys().items()}


def format_reasoning_string(reasoning_dict: Dict[str, str]) -> str:
    """
    将推理字典格式化为 @tag@value 格式的字符串。
    
    Args:
        reasoning_dict: {database_key: value} 格式的推理字典
    
    Returns:
        str: @tag@value@tag@value... 格式的字符串
    """
    db_to_cot = get_database_to_cot_keys()
    tags_order = get_cot_tags_list()[:-1]  # 排除 ACTION
    
    parts = []
    for tag in tags_order:
        db_key = get_cot_database_keys().get(tag)
        if db_key and db_key in reasoning_dict:
            value = reasoning_dict[db_key]
            parts.append(f"@{tag}@{value}")
    
    return "".join(parts)


def parse_reasoning_string(reasoning_str: str) -> Dict[str, str]:
    """
    解析 @tag@value 格式的推理字符串。
    
    Args:
        reasoning_str: @tag@value@tag@value... 格式的字符串
    
    Returns:
        Dict[str, str]: {database_key: value} 格式的推理字典
    """
    import re
    
    cot_to_db = get_cot_database_keys()
    tags = get_cot_tags_list()
    
    result = {}
    
    # 构建正则表达式匹配所有标签
    for i, tag in enumerate(tags[:-1]):  # 排除 ACTION
        # 找到当前标签和下一个标签之间的内容
        pattern = f"@{re.escape(tag)}@([^@]*)"
        match = re.search(pattern, reasoning_str)
        if match:
            value = match.group(1).strip()
            db_key = cot_to_db.get(tag)
            if db_key:
                result[db_key] = value
    
    return result


def format_full_cot_response(reasoning_dict: Dict[str, str], action_str: str) -> str:
    """
    格式化完整的 CoT 响应（用于训练）。
    
    Args:
        reasoning_dict: {database_key: value} 格式的推理字典
        action_str: 动作 token 字符串
    
    Returns:
        str: 完整的 CoT 响应字符串
    """
    db_to_cot = get_database_to_cot_keys()
    tags_order = get_cot_tags_list()[:-1]  # 排除 ACTION
    
    parts = []
    for tag in tags_order:
        db_key = get_cot_database_keys().get(tag)
        if db_key and db_key in reasoning_dict:
            value = reasoning_dict[db_key]
            parts.append(f"{tag} {value}")
    
    parts.append(f"{CotTag.ACTION.value} {action_str}")
    
    return " ".join(parts)


def format_action_only_response(action_str: str) -> str:
    """
    格式化仅动作的响应（No CoT 模式）。
    
    Args:
        action_str: 动作 token 字符串
    
    Returns:
        str: 仅动作的响应字符串
    """
    return f"{CotTag.ACTION.value} {action_str}"
