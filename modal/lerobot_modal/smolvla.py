from typing import Any

import numpy as np
import torch
from loguru import logger

from phosphobot.am.smolvla import SmolVLASpawnConfig
from .helper import validate_inputs, prepare_base_batch


def process_smolvla_inference(
    policy: Any,
    model_specifics: SmolVLASpawnConfig,
    current_qpos: list[float],
    images: list[np.ndarray],
    image_names: list[str],
    target_size: tuple[int, int],
    prompt: str,
    prompt_key: str = "task",
) -> np.ndarray:
    """
    Process images and perform inference using the SmolVLA policy.

    Args:
        policy: lerobot.policies.smolvla.modeling_smolvla.SmolVLAPolicy object
        model_specifics: SmolVLASpawnConfig object
        current_qpos: Current robot state
        images: List of images
        image_names: List of image names corresponding to the images
        target_size: Target size for resizing images (height, width)
        prompt: Text instruction for the model
        prompt_key: prompt key used for policy inference

    Returns:
        np.ndarray: predicted action chunk

    Raises:
        AssertionError: If input validations fail.
        ValueError: If prompt is None.
    """
    # Validate inputs using common function
    validate_inputs(current_qpos, images, model_specifics, target_size)

    if prompt is None:
        raise ValueError("Prompt cannot be None for SmolVLA inference")

    with torch.no_grad(), torch.autocast(device_type="cuda"):
        # Prepare base observation using common function
        batch = prepare_base_batch(current_qpos, images, image_names, model_specifics, target_size)
        batch[prompt_key] = prompt    # prompt is required for SmolVLA

        # Run inference
        actions = policy.predict_action_chunk(batch)

        return actions.cpu().numpy()