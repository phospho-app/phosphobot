import asyncio
import time
from collections import deque
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

if TYPE_CHECKING:
    # We only need BaseManipulator for type checking
    # This prevents loading pybullet in modal
    from phosphobot.hardware.base import BaseManipulator

import httpx
import numpy as np
from fastapi import HTTPException
from huggingface_hub import HfApi
from loguru import logger
from pydantic import BaseModel, Field

from phosphobot.am.act import (
    InputFeatures,
    ACT
)
from phosphobot.camera import AllCameras
from phosphobot.control_signal import AIControlSignal
from phosphobot.hardware.base import BaseManipulator
from phosphobot.utils import background_task_log_exceptions, get_hf_token

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
    def __init__(
        self,
        server_url: str = "http://localhost",
        server_port: int = 8080,
        **kwargs: Any,
    ) -> None:
        super().__init__(server_url, server_port)
        # Modal tunnels expose ports via a URL, so we use that directly
        # Appending the port sometimes results in `[SSL: WRONG_VERSION_NUMBER]` errors
        # Ref: https://modal.com/docs/guide/tunnels
        self.async_client = httpx.AsyncClient(
            base_url=server_url,
            timeout=10,
            headers={"Content-Type": "application/json"},
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=100),
            http2=True,  # Enables HTTP/2 for better performance if supported
        )
        self.sync_client = httpx.Client(
            base_url=server_url,
            timeout=10,
            headers={"Content-Type": "application/json"},
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=100),
            http2=True,  # Enables HTTP/2 if supported by the server
        )

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

    @background_task_log_exceptions
    async def control_loop(
        self,
        control_signal: AIControlSignal,
        robots: list["BaseManipulator"],
        model_spawn_config: SmolVLASpawnConfig,
        all_cameras: AllCameras,
        fps: int = 30,
        speed: float = 1.0,
        cameras_keys_mapping: Optional[Dict[str, int]] = None,
        prompt: Optional[str] = None,
        selected_camera_id: Optional[int] = None,
        angle_format: Literal["degrees", "radians", "other"] = "radians",
        min_angle: Optional[float] = None,
        max_angle: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """
        AI control loop that runs in the background and sends actions to the robot.
        It uses the model to get the actions based on the current state of the robot and the cameras.
        The loop runs until the control signal is stopped or the model is not available anymore.
        The loop runs at the specified fps and speed.
        """

        nb_iter = 0
        config = model_spawn_config.hf_model_config

        signal_marked_as_started = False
        actions_queue: deque = deque([])

        while control_signal.is_in_loop():
            logger.debug(
                f"AI control loop iteration {nb_iter}, status: {control_signal.status}, with id {control_signal.id}"
            )
            if control_signal.status == "paused":
                logger.debug("AI control loop paused")
                await asyncio.sleep(0.1)
                continue

            start_time = time.perf_counter()

            # Get the images from the cameras based on the config
            # For now, just put as many cameras as the model config
            image_inputs: Dict[str, np.ndarray] = {}
            for i, camera_name in enumerate(config.input_features.video_keys):
                if cameras_keys_mapping is None:
                    camera_id = i
                else:
                    camera_id = cameras_keys_mapping.get(camera_name, i)

                video_resolution = config.input_features.features[camera_name].shape
                frame_array = SmolVLA.fetch_frame(
                    all_cameras=all_cameras,
                    camera_id=camera_id,
                    resolution=video_resolution,
                )
                image_inputs[camera_name] = frame_array

            # Number of cameras
            if len(image_inputs) != len(config.input_features.video_keys):
                logger.warning(
                    f"Model has {len(config.input_features.video_keys)} cameras but {len(image_inputs)} cameras are plugged."
                )
                control_signal.stop()
                raise Exception(
                    f"Model has {config.input_features.video_keys} cameras but {len(image_inputs)} cameras are plugged."
                )

            # Number of robots
            number_of_robots = len(robots)
            number_of_robots_in_config = config.input_features.number_of_arms
            if number_of_robots != number_of_robots_in_config:
                logger.warning("No robot connected. Exiting AI control loop.")
                control_signal.stop()
                raise Exception("No robot connected. Exiting AI control loop.")

            # Concatenate all robot states
            state = robots[0].read_joints_position(unit="rad")
            for robot in robots[1:]:
                state = np.concatenate(
                    (state, robot.read_joints_position(unit="rad")), axis=0
                )

            inputs: dict[str, np.ndarray | str] = {
                config.input_features.state_key: state,
                "prompt": prompt if prompt is not None else "",
                **image_inputs,
            }

            try:
                if len(actions_queue) == 0:
                    actions = await self.async_sample_actions(inputs)
                    actions_queue.extend(actions)
                actions = actions_queue.popleft()
            except Exception as e:
                logger.warning(
                    f"Failed to get actions from model: {e}. Exiting AI control loop."
                )
                control_signal.stop()
                break

            if not signal_marked_as_started:
                control_signal.set_running()
                signal_marked_as_started = True

            for action in actions:
                # Early stop
                if not control_signal.is_in_loop():
                    break

                # Send the new joint position to the robot
                action_list = action.tolist()

                unit: Literal["rad", "motor_units", "degrees", "other"]
                if angle_format == "radians":
                    unit = "rad"
                else:
                    unit = angle_format

                for robot_index in range(len(robots)):
                    robots[robot_index].write_joint_positions(
                        angles=action_list[robot_index * 6 : robot_index * 6 + 6],
                        unit=unit,
                        min_value=min_angle,
                        max_value=max_angle,
                    )

                # Wait fps time
                elapsed_time = time.perf_counter() - start_time
                sleep_time = max(0, 1.0 / (fps * speed) - elapsed_time)
                await asyncio.sleep(sleep_time)
                start_time = time.perf_counter()

            nb_iter += 1
