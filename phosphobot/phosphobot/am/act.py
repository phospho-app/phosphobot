from typing import Dict, Literal, Optional

import numpy as np

from phosphobot.am.lerobot import (
    HuggingFaceAugmentedValidator,
    HuggingFaceModelValidator,
    LeRobot,
    LeRobotSpawnConfig,
)
from phosphobot.camera import AllCameras


class ACTHuggingFaceModelValidator(HuggingFaceModelValidator):
    """ACT-specific HuggingFace model validator"""
    type: Literal["act"]


class ACTHuggingFaceAugmentedValidator(HuggingFaceAugmentedValidator):
    """
    ACT-specific augmented model validator that includes additional fields
    for augmented models, such as available checkpoints.
    """
    type: Literal["act"]


class ACTSpawnConfig(LeRobotSpawnConfig):
    """ACT-specific spawn configuration"""
    hf_model_config: ACTHuggingFaceAugmentedValidator  # type: ignore[assignment]


class ACT(LeRobot):
    """Action model implementation for Action Chunking Transformer (ACT). Inherits from LeRobot."""

    @classmethod
    def _get_model_validator_class(cls) -> type:
        """Return ACT-specific model validator class"""
        return ACTHuggingFaceModelValidator

    @classmethod
    def _get_augmented_validator_class(cls) -> type:
        """Return ACT-specific augmented validator class"""
        return ACTHuggingFaceAugmentedValidator

    @classmethod
    def _get_spawn_config_class(cls) -> type:
        """Return ACT-specific spawn config class"""
        return ACTSpawnConfig

    def _prepare_model_inputs(
        self,
        config: HuggingFaceAugmentedValidator,
        state: np.ndarray,
        image_inputs: Dict[str, np.ndarray],
        prompt: Optional[str] = None,
        selected_camera_id: Optional[int] = None,
        all_cameras: Optional[AllCameras] = None,
    ) -> Dict[str, np.ndarray | str]:
        """Prepare model inputs for ACT"""
        inputs: Dict[str, np.ndarray | str] = {
            config.input_features.state_key: state,
            **image_inputs,
        }

        # ACT: Handle env_key with object detection
        if config.input_features.env_key is not None:
            if prompt is None or selected_camera_id is None or all_cameras is None:
                raise ValueError(
                    f"detect_instruction, camera_id_to_use, and all_cameras must be provided when env_key is set.\n"
                    f"Got prompt: {prompt}, selected_camera_id: {selected_camera_id}, all_cameras: {all_cameras}"
                )
            inputs["detect_instruction"] = prompt

            frame_array = ACT.fetch_frame(
                all_cameras=all_cameras,
                camera_id=selected_camera_id,
                resolution=[3, 224, 224],
            )
            inputs["image_for_bboxes"] = frame_array

        return inputs
