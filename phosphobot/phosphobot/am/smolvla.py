import httpx
import json_numpy  # type: ignore
import numpy as np
from loguru import logger
from typing import Dict, List, Optional, Literal
from fastapi import HTTPException
from huggingface_hub import HfApi
from pydantic import BaseModel

from phosphobot.am.act import (
    ACT,
    ACTSpawnConfig,
    InputFeatures,
    InputFeature,
    HuggingFaceModelValidator,
)
from phosphobot.camera import AllCameras
from phosphobot.hardware.base import BaseManipulator
from phosphobot.utils import get_hf_token

# NOTE: SmolVLA config on Hugging Face follows the same schema as ACT models
# (see https://huggingface.co/lerobot/smolvla_base). Therefore we can reuse
# the InputFeatures / ACTSpawnConfig dataclasses. Only the type literal differs.

class SmolVLAConfigValidator(HuggingFaceModelValidator):
    type: Literal["smolvla"]


class SmolVLASpawnConfig(ACTSpawnConfig):  # identical fields, different HF validator
    hf_model_config: SmolVLAConfigValidator  # type: ignore


class SmolVLA(ACT):
    """Client wrapper for SmolVLA model. Reuses ACT implementation."""

    @classmethod
    def fetch_config(cls, model_id: str) -> SmolVLAConfigValidator:  # type: ignore
        try:
            api = HfApi(token=get_hf_token())
            api.model_info(model_id)  # will raise if missing
            config_path = api.hf_hub_download(
                repo_id=model_id,
                filename="config.json",
                force_download=True,
            )
            with open(config_path, "r") as f:
                config_content = f.read()
            hf_model_config = SmolVLAConfigValidator.model_validate_json(config_content)  # type: ignore
        except Exception as e:
            raise Exception(f"Error loading model {model_id} from HuggingFace: {e}")
        return hf_model_config  # type: ignore

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
    ) -> SmolVLASpawnConfig:  # type: ignore
        # Reuse ACT verification logic (which mainly checks camera counts etc.)
        # It returns an ACTSpawnConfig; we'll cast to our subclass but it's same fields.
        from phosphobot.am.act import ACT as _ACT

        act_cfg: ACTSpawnConfig = _ACT.fetch_and_verify_config(
            model_id=model_id,
            all_cameras=all_cameras,
            robots=robots,
            cameras_keys_mapping=cameras_keys_mapping,
            verify_cameras=verify_cameras,
        )
        return SmolVLASpawnConfig(**act_cfg.model_dump())  # type: ignore
