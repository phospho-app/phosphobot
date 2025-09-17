import time
import dill
import asyncio
from collections import deque
from typing import TYPE_CHECKING, Any, Dict, List, Literal

if TYPE_CHECKING:
    # We only need BaseManipulator for type checking
    # This prevents loading pybullet in modal
    from phosphobot.hardware.base import BaseManipulator

import cv2
import httpx
import json_numpy  # type: ignore
import numpy as np
from fastapi import HTTPException
from huggingface_hub import HfApi
from loguru import logger
from pydantic import BaseModel, Field

from phosphobot.am.base import (
    ActionModel,
)
from phosphobot.camera import AllCameras
from phosphobot.control_signal import AIControlSignal
from phosphobot.models import ModelConfigurationResponse
from phosphobot.utils import background_task_log_exceptions, get_hf_token


class Statistics(BaseModel):
    mean: List[float]
    std: List[float]
    q01: List[float]
    q99: List[float]


class InputFeature(BaseModel):
    state: Statistics
    actions: Statistics


class NormFile(BaseModel):
    class Config:
        extra = "allow"

    norm_stats: InputFeature

    @property
    def action_dim(self) -> int:
        """
        Count the number of values that are not zero in the std of actions.
        """
        return sum(1 for x in self.norm_stats.actions.std if x != 0.0)


class Pi05SpawnConfig(BaseModel):
    action_dim: int = Field(
        ...,
        description="Dimension of the action space (number of joints)",
    )
    image_keys: List[str] = Field(
        default_factory=list,
        description="List of image keys expected by the model",
    )


class HuggingFaceAugmentedValidator(BaseModel):
    class Config:
        extra = "allow"

    config: Pi05SpawnConfig
    checkpoints: List[str] = Field(
        default_factory=list,
        description="List of available checkpoints/branches for the model",
    )


def fetch_camera_images(
    config: Pi05SpawnConfig,
    all_cameras: AllCameras,
    cameras_keys_mapping: Dict[str, int] | None = None,
) -> Dict[str, np.ndarray]:
    """
    Fetch images from cameras based on the model configuration.

    Args:
        config: The model configuration containing video keys and resolutions
        all_cameras: Camera manager instance
        cameras_keys_mapping: [Optional] mapping of camera names to camera IDs

    Returns:
        Dictionary mapping camera names to captured image arrays
    """
    image_inputs: Dict[str, np.ndarray] = {}
    for i, camera_name in enumerate(config.image_keys):
        if cameras_keys_mapping is None:
            camera_id = i
        else:
            camera_id = cameras_keys_mapping.get(camera_name, i)

        video_resolution = [3, 224, 224]  # Default resolution (C, H, W)
        frame_array = Pi05.fetch_frame(
            all_cameras=all_cameras,
            camera_id=camera_id,
            resolution=video_resolution,
        )
        image_inputs[camera_name] = frame_array

    return image_inputs


class RetryError(Exception):
    """Custom exception to retry the inference call."""

    pass


class Pi05(ActionModel):
    """Client for Pi0.5 model inference server."""

    REQUIRED_CAMERA_KEYS = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]

    def __init__(
        self,
        server_url: str = "http://localhost",
        server_port: int = 8080,
        **kwargs: Any,
    ):
        super().__init__(server_url, server_port)
        self.async_client = httpx.AsyncClient(
            base_url=server_url + f":{server_port}",
            timeout=10,
            headers={"Content-Type": "application/json"},
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=100),
            http2=True,  # Enables HTTP/2 for better performance if supported
        )
        self.sync_client = httpx.Client(
            base_url=server_url + f":{server_port}",
            timeout=10,
            headers={"Content-Type": "application/json"},
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=100),
            http2=True,  # Enables HTTP/2 if supported by the server
        )

    def sample_actions(self, inputs: dict) -> np.ndarray:
        # Double-encoded version (to send numpy arrays as JSON)
        encoded_payload = {"encoded": json_numpy.dumps(inputs)}

        try:
            response = self.sync_client.post("/get_action", json=encoded_payload)

            if response.status_code == 202:
                raise RetryError(response.content)

            if response.status_code != 200:
                raise RuntimeError(response.text)
            actions = json_numpy.loads(response.json())
        except RetryError as e:
            raise RetryError(e)
        except Exception as e:
            logger.error(f"Error in sampling actions: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error in sampling actions: {e}",
            )
        return actions

    async def async_sample_actions(self, inputs: dict) -> np.ndarray:
        # Clean up the input to avoid JSON serialization issues
        encoded_payload = {"encoded": json_numpy.dumps(inputs)}

        try:
            response = await self.async_client.post(
                f"{self.server_url}/get_action", json=encoded_payload, timeout=30
            )

            if response.status_code == 202:
                raise RetryError(response.content)

            if response.status_code != 200:
                raise RuntimeError(response.text)
            actions = json_numpy.loads(response.json())
        except RetryError as e:
            raise RetryError(e)
        except Exception as e:
            logger.error(f"Error in sampling actions: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error in sampling actions: {e}",
            )
        return actions

    @classmethod
    def fetch_config(cls, model_id: str) -> HuggingFaceAugmentedValidator:
        """
        Fetch the model configuration from HuggingFace.
        """
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

            config = api.hf_hub_download(
                repo_id=model_id,
                filename="config.pkl",
                force_download=True,
            )
            with open(config, "rb") as f:
                config_content = f.read()
            config_dict = dill.load(config_content)

            norm_stats = api.hf_hub_download(
                repo_id=model_id,
                filename="norm_stats.json",
                force_download=True,
            )
            with open(norm_stats, "r") as f:
                norm_stats_content = f.read()
            norm_parsed = NormFile.model_validate(norm_stats_content)

            logger.debug(f"Fetched model config: {config_dict}")

            return HuggingFaceAugmentedValidator(
                config=Pi05SpawnConfig(
                    action_dim=norm_parsed.action_dim,
                    image_keys=config_dict.get("image_keys", []),
                ),
                checkpoints=branches,
            )
        except Exception as e:
            logger.warning(f"Could not fetch model config from HuggingFace: {e}")
            return HuggingFaceAugmentedValidator(
                config=Pi05SpawnConfig(action_dim=0, image_keys=[]),
                checkpoints=[],
            )

    @classmethod
    def fetch_and_get_configuration(cls, model_id: str) -> ModelConfigurationResponse:
        """
        Fetch the model configuration from HuggingFace and return the video keys.
        """
        hf_model_config = cls.fetch_config(model_id=model_id)
        configuration = ModelConfigurationResponse(
            video_keys=hf_model_config.config.image_keys,
            checkpoints=hf_model_config.checkpoints,
        )
        return configuration

    @classmethod
    def fetch_spawn_config(cls, model_id: str) -> Pi05SpawnConfig:
        """Fetch spawn configuration for Pi0 model."""
        hf_model_config = cls.fetch_config(model_id=model_id)

        return Pi05SpawnConfig(
            action_dim=hf_model_config.config.action_dim,
            image_keys=hf_model_config.config.image_keys,
        )

    @classmethod
    def fetch_and_verify_config(
        cls,
        model_id: str,
        all_cameras: AllCameras,
        robots: list["BaseManipulator"],
        cameras_keys_mapping: Dict[str, int] | None = None,
        verify_cameras: bool = True,
    ) -> Pi05SpawnConfig:
        """
        Verify if the HuggingFace model is compatible with the current setup.
        """
        hf_model_config = cls.fetch_config(model_id=model_id)

        action_dim = hf_model_config.config.action_dim
        if action_dim != sum(robot.num_actuated_joints for robot in robots):
            raise HTTPException(
                status_code=400,
                detail=f"Model has {action_dim} action dimensions but we found {sum(robot.num_actuated_joints for robot in robots)} connected joints on {len(robots)} robots.",
            )

        return Pi05SpawnConfig(
            action_dim=action_dim,
        )

    @classmethod
    def fetch_frame(
        cls, all_cameras: AllCameras, camera_id: int, resolution: list[int]
    ) -> np.ndarray:
        rgb_frame = all_cameras.get_rgb_frame(
            camera_id=camera_id,
            resize=(resolution[2], resolution[1]),
        )
        if rgb_frame is not None:
            # Convert to BGR
            image = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            # Ensure dtype is uint8 (if it isn’t already)
            converted_array = image.astype(np.uint8)
            return converted_array

        else:
            logger.warning(f"Camera {camera_id} not available. Sending all black.")
            return np.zeros(
                (
                    resolution[2],
                    resolution[1],
                    resolution[0],
                ),
                dtype=np.uint8,
            )

    @background_task_log_exceptions
    async def control_loop(
        self,
        control_signal: AIControlSignal,
        robots: list["BaseManipulator"],
        model_spawn_config: Pi05SpawnConfig,
        all_cameras: AllCameras,
        prompt: str,
        fps: int = 30,
        speed: float = 1.0,
        cameras_keys_mapping: Dict[str, int] | None = None,
        angle_format: Literal["degrees", "radians", "other"] = "radians",
        min_angle: float | None = None,
        max_angle: float | None = None,
        **kwargs: Any,
    ) -> None:
        """
        AI control loop that runs in the background and sends actions to the robot.
        It uses the model to get the actions based on the current state of the robot and the cameras.
        The loop runs until the control signal is stopped or the model is not available anymore.
        The loop runs at the specified fps and speed.
        """
        nb_iter = 0
        action_dim = model_spawn_config.action_dim

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
            image_inputs = fetch_camera_images(
                config=model_spawn_config,
                all_cameras=all_cameras,
                cameras_keys_mapping=cameras_keys_mapping,
            )

            # Verify number of cameras
            if len(image_inputs) != len(model_spawn_config.image_keys):
                logger.warning(
                    f"Model has {len(model_spawn_config.image_keys)} cameras but "
                    f"{len(image_inputs)} cameras are plugged."
                )
                control_signal.stop()
                raise Exception(
                    f"Model has {len(model_spawn_config.image_keys)} cameras but "
                    f"{len(image_inputs)} cameras are plugged."
                )

            # Verify number of robots
            number_of_joints = sum(robot.num_actuated_joints for robot in robots)
            number_of_joints_in_config = model_spawn_config.action_dim
            if number_of_joints != number_of_joints_in_config:
                logger.warning(
                    f"Model has {number_of_joints_in_config} joints but {number_of_joints} joints are connected with {len(robots)} robots."
                )
                control_signal.stop()
                raise Exception(
                    f"Model has {number_of_joints_in_config} joints but {number_of_joints} joints are connected with {len(robots)} robots."
                )

            # Concatenate all robot states
            state = robots[0].read_joints_position(unit="rad")
            for robot in robots[1:]:
                state = np.concatenate(
                    (state, robot.read_joints_position(unit="rad")), axis=0
                )

            # Prepare model input
            inputs: dict[str, np.ndarray | str] = {
                "state": state,
                "prompt": prompt,
                **image_inputs,
            }

            try:
                if len(actions_queue) == 0:
                    actions = await self.async_sample_actions(inputs)
                    actions_queue.extend(actions)
                actions = actions_queue.popleft()
            except Exception as e:
                logger.warning(
                    f"Failed to get actions from model, exiting AI control loop.\nError: {e}"
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
                        angles=action_list[
                            robot_index * action_dim : robot_index * action_dim
                            + action_dim
                        ],
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
