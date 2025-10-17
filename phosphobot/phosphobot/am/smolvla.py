from typing import Dict, Literal, Optional

import numpy as np

from phosphobot.am.lerobot import (
    HuggingFaceAugmentedValidator,
    HuggingFaceModelValidator,
    LeRobot,
    LeRobotSpawnConfig,
)
from phosphobot.camera import AllCameras


class SmolVLAHuggingFaceModelValidator(HuggingFaceModelValidator):
    """SmolVLA-specific HuggingFace model validator"""
    type: Literal["smolvla"]


class SmolVLAHuggingFaceAugmentedValidator(HuggingFaceAugmentedValidator):
    """
    SmolVLA-specific augmented model validator that includes additional fields
    for augmented models, such as available checkpoints.
    """
    type: Literal["smolvla"]


class SmolVLASpawnConfig(LeRobotSpawnConfig):
    """SmolVLA-specific spawn configuration"""
    hf_model_config: SmolVLAHuggingFaceAugmentedValidator  # type: ignore[assignment]


class SmolVLA(LeRobot):
    """Action model implementation for SmolVLA. Inherits from LeRobot."""

    @classmethod
    def _get_model_validator_class(cls) -> type:
        """Return SmolVLA-specific model validator class"""
        return SmolVLAHuggingFaceModelValidator

    @classmethod
    def _get_augmented_validator_class(cls) -> type:
        """Return SmolVLA-specific augmented validator class"""
        return SmolVLAHuggingFaceAugmentedValidator

    @classmethod
    def _get_spawn_config_class(cls) -> type:
        """Return SmolVLA-specific spawn config class"""
        return SmolVLASpawnConfig

    def _prepare_model_inputs(
        self,
        config: HuggingFaceAugmentedValidator,
        state: np.ndarray,
        image_inputs: Dict[str, np.ndarray],
        prompt: Optional[str] = None,
        selected_camera_id: Optional[int] = None,
        all_cameras: Optional[AllCameras] = None,
    ) -> Dict[str, np.ndarray | str]:
        """Prepare model inputs for SmolVLA"""
        if prompt is None:
            raise ValueError("Prompt must be provided for SmolVLA models.")

        inputs: Dict[str, np.ndarray | str] = {
            config.input_features.state_key: state,
            "prompt": prompt,
            **image_inputs,
        }
        return inputs
