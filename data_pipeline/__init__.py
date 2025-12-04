# Filename: data_pipeline/__init__.py
"""
数据处理流水线模块

包含 LIBERO 数据加载、划分、ECoT 标注生成和数据重生成功能。
"""

from data_pipeline.raw_loader import (
    LiberoRawLoader,
    LiberoBatchLoader,
    parse_task_instruction
)
from data_pipeline.data_splitter import (
    LiberoDataSplitter,
    SplitInfo
)
from data_pipeline.primitive_movements import (
    describe_move,
    classify_movement,
    get_move_primitives_trajectory,
    get_move_primitives_from_states,
    get_state_3d_positions
)

# Lazy imports for modules that require optional dependencies (robosuite, libero)
# These are only needed for data regeneration, not for training
def _get_ecot_annotator():
    from data_pipeline.ecot_annotator import (
        ECoTAnnotator,
        LLMConfig,
        AnnotationConfig,
        build_prompt,
        extract_reasoning_dict
    )
    return ECoTAnnotator, LLMConfig, AnnotationConfig, build_prompt, extract_reasoning_dict

def _get_libero_utils():
    from data_pipeline.libero_utils import (
        get_libero_env,
        get_libero_dummy_action,
        get_benchmark_tasks,
        LIBERO_AVAILABLE
    )
    return get_libero_env, get_libero_dummy_action, get_benchmark_tasks, LIBERO_AVAILABLE

def _get_libero_regenerator():
    from data_pipeline.libero_regenerator import LiberoDataRegenerator
    return LiberoDataRegenerator

__all__ = [
    # raw_loader
    "LiberoRawLoader",
    "LiberoBatchLoader",
    "parse_task_instruction",
    # data_splitter
    "LiberoDataSplitter",
    "SplitInfo",
    # primitive_movements
    "describe_move",
    "classify_movement",
    "get_move_primitives_trajectory",
    "get_move_primitives_from_states",
    "get_state_3d_positions",
    # lazy imports (access via functions)
    "_get_ecot_annotator",
    "_get_libero_utils",
    "_get_libero_regenerator",
]
