"""
LeRobot Modal - Common infrastructure for LeRobot models.

This module provides shared functionality for training and serving
LeRobot models on Modal infrastructure.
"""

from .helper import (
    InferenceRequest,
    ParquetEpisodesDataset,
    compute_stats,
    tensor_to_list,
    decode_video_frames_torchvision,
    get_stats_einops_patterns,
    validate_inputs,
    prepare_base_batch,
)

from .act import (
    process_act_inference,
    compute_bboxes,
    read_first_frame_with_pyav,
    NotEnoughBBoxesError,
    InvalidInputError,
)

from .smolvla import (
    process_smolvla_inference,
)

from .app import (
    serve_act,
    train_act,
    serve_smolvla,
    train_smolvla,
    act_app,
    smolvla_app,
)

__all__ = [
    # Helper functions
    "InferenceRequest",
    "ParquetEpisodesDataset",
    "compute_stats",
    "tensor_to_list",
    "decode_video_frames_torchvision",
    "get_stats_einops_patterns",
    "validate_inputs",
    "prepare_base_batch",

    # ACT specific
    "process_act_inference",
    "compute_bboxes",
    "read_first_frame_with_pyav",
    "NotEnoughBBoxesError",
    "InvalidInputError",

    # SmolVLA specific
    "process_smolvla_inference",

    # Modal apps and functions
    "serve_act",
    "train_act",
    "serve_smolvla",
    "train_smolvla",
    "act_app",
    "smolvla_app",
]