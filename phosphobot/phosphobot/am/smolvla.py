from loguru import logger
from typing import Dict, List, Optional, Literal
from fastapi import HTTPException
from huggingface_hub import HfApi
from pydantic import BaseModel, Field

from phosphobot.am.act import (
    InputFeatures,
    ACT
)
from phosphobot.camera import AllCameras
from phosphobot.hardware.base import BaseManipulator
from phosphobot.utils import get_hf_token

"""
NOTE: SmolVLA config on Hugging Face follows the same schema as ACT models
(see https://huggingface.co/lerobot/smolvla_base). Therefore we can reuse
the dataclasses from phosphobot.am.act
"""

class HuggingFaceModelValidator(BaseModel):
    type: Literal["smolvla"]
    input_features: InputFeatures

    class Config:
        extra = "allow"


class HuggingFaceAugmentedValidator(HuggingFaceModelValidator):
    """
    This model extends HuggingFaceModelValidator to include additional fields
    for augmented models, such as available checkpoints.
    """

    checkpoints: List[str] = Field(
        default_factory=list,
        description="List of available checkpoints for the model.",
    )


class SmolVLASpawnConfig(BaseModel):
    state_key: str
    state_size: list[int]
    env_key: Optional[str] = None
    env_size: Optional[list[int]] = None
    video_keys: list[str]
    video_size: list[int]
    hf_model_config: HuggingFaceAugmentedValidator


class SmolVLA(ACT):
    """Client wrapper for SmolVLA model. Reuses ACT implementation."""

    @classmethod
    def fetch_config(cls, model_id: str) -> HuggingFaceAugmentedValidator:  # type: ignore
        try:
            api = HfApi(token=get_hf_token())
            model_info = api.model_info(model_id)
            if model_info is None:
                raise Exception(f"Model {model_id} not found on HuggingFace.")
            # Fetch the available revisions
            branches = []
            refs = api.list_repo_refs(model_id)
            for branch in refs.branches:
                branches.append(branch.name)
            config_path = api.hf_hub_download(
                repo_id=model_id,
                filename="config.json",
                force_download=True,
            )
            with open(config_path, "r") as f:
                config_content = f.read()
            hf_model_config = HuggingFaceModelValidator.model_validate_json(
                config_content
            )
            hf_augmented_config = HuggingFaceAugmentedValidator(
                **hf_model_config.model_dump(),
                checkpoints=branches,
            )
        except Exception as e:
            raise Exception(f"Error loading model {model_id} from HuggingFace: {e}")
        return hf_augmented_config

    @classmethod
    def fetch_spawn_config(cls, model_id: str) -> SmolVLASpawnConfig:  # type: ignore
        hf_model_config = cls.fetch_config(model_id=model_id)

        state_key: str = hf_model_config.input_features.state_key
        state_size: List[int] = hf_model_config.input_features.features[state_key].shape
        env_key: Optional[str] = hf_model_config.input_features.env_key
        env_size: Optional[List[int]] = (
            hf_model_config.input_features.features[env_key].shape  # type: ignore
            if env_key is not None
            else None
        )
        video_keys: List[str] = hf_model_config.input_features.video_keys
        video_size: List[int] = (
            hf_model_config.input_features.features[video_keys[0]].shape  # type: ignore
            if len(video_keys) > 0
            else [3, 224, 224]
        )

        return SmolVLASpawnConfig(
            state_key=state_key,
            state_size=state_size,
            env_key=env_key,
            env_size=env_size,
            video_keys=video_keys,
            video_size=video_size,
            hf_model_config=hf_model_config,
        )

    @classmethod
    def fetch_and_verify_config(
        cls,
        model_id: str,
        all_cameras: AllCameras,
        robots: List[BaseManipulator],
        cameras_keys_mapping: Dict[str, int] | None = None,
        verify_cameras: bool = True,
    ) -> SmolVLASpawnConfig:
        """
        Verify if the HuggingFace model is compatible with the current setup.
        """

        hf_model_config = cls.fetch_config(model_id=model_id)

        state_key: str = hf_model_config.input_features.state_key
        state_size: list[int] = hf_model_config.input_features.features[state_key].shape
        env_key: Optional[str] = hf_model_config.input_features.env_key
        env_size: Optional[list[int]] = (
            hf_model_config.input_features.features[env_key].shape
            if env_key is not None
            else None
        )
        video_keys: list[str] = hf_model_config.input_features.video_keys
        video_size: list[int] = (
            hf_model_config.input_features.features[video_keys[0]].shape
            if len(video_keys) > 0
            else [3, 224, 224]
        )

        if cameras_keys_mapping is None:
            nb_connected_cams = len(all_cameras.video_cameras)
        else:
            # Check if all keys are in the model config
            keys_in_common = set(
                [
                    k.replace("video.", "") if k.startswith("video.") else k
                    for k in cameras_keys_mapping.keys()
                ]
            ).intersection(hf_model_config.input_features.video_keys)
            nb_connected_cams = len(keys_in_common)

        if nb_connected_cams < len(video_keys) and verify_cameras:
            logger.warning(
                f"Model has {len(video_keys)} cameras but {nb_connected_cams} camera streams are detected."
            )
            raise HTTPException(
                status_code=400,
                detail=f"Model has {len(video_keys)} cameras but {nb_connected_cams} camera streams are detected.",
            )

        number_of_robots = hf_model_config.input_features.number_of_arms
        if number_of_robots != len(robots):
            raise HTTPException(
                status_code=400,
                detail=f"Model has {number_of_robots} robots but {len(robots)} robots are connected.",
            )

        return SmolVLASpawnConfig(
            state_key=state_key,
            state_size=state_size,
            env_key=env_key,
            env_size=env_size,
            video_keys=video_keys,
            video_size=video_size,
            hf_model_config=hf_model_config,
        )
