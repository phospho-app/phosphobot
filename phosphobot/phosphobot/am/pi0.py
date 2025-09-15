import asyncio
import time
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
from pydantic import BaseModel, Field, field_validator, model_validator

from phosphobot.am.base import (
    ActionModel,
    BaseTrainer,
    BaseTrainerConfig,
    TrainingParamsPi0,
)
from phosphobot.camera import AllCameras
from phosphobot.control_signal import AIControlSignal
from phosphobot.models import ModelConfigurationResponse
from phosphobot.utils import background_task_log_exceptions, get_hf_token


class InputFeature(BaseModel):
    """Individual input feature for the pi0 model"""

    type: Literal["STATE", "VISUAL"]
    shape: List[int]


class InputFeatures(BaseModel):
    """Collection of input features for the pi0 model."""

    state_key: str
    video_keys: List[str] = []
    # Currently all supported robots have 6 joints (5 joints + 1 gripper).
    action_dim: int = 6
    features: Dict[str, InputFeature]

    @property
    def number_of_arms(self) -> int:
        """
        Currently all supported robots have 6 joints.
        To be changed when this is no longer true.
        """
        return self.features[self.state_key].shape[0] // self.action_dim

    @model_validator(mode="before")
    def infer_keys(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess input to infer state_key and video_keys if input is a flat features dict.
        Runs before field validation.
        """
        if isinstance(values, dict) and "features" not in values:
            features = values
            state_keys = [k for k in features if ".state" in k.lower()]
            video_keys = [k for k in features if "image" in k.lower()]
            if len(state_keys) != 1:
                raise ValueError(
                    "Exactly one state key must be present in the features"
                )
            state_key = state_keys[0]
            return {
                "state_key": state_key,
                "video_keys": video_keys,
                "features": features,
            }
        return values

    @field_validator("features", mode="before")
    def validate_features(cls, value: Dict[str, Any]) -> Dict[str, InputFeature]:
        """
        Validate and transform the features dictionary into InputFeature instances.
        """
        if not isinstance(value, dict):
            raise ValueError("Features must be a dictionary")
        result = {}
        for key, item in value.items():
            if ".state" in key.lower():
                if item.get("type") != "STATE":
                    raise ValueError(f"Key {key} with 'state' must have type 'STATE'")
            elif "image" in key.lower():
                if item.get("type") != "VISUAL":
                    raise ValueError(f"Key {key} with 'image' must have type 'VISUAL'")
            else:
                raise ValueError(f"Key {key} must contain 'state' or 'image'")
            result[key] = InputFeature(**item)
        return result

    @model_validator(mode="after")
    def validate_keys(self) -> "InputFeatures":
        """
        Validate state_key and video_keys against features after all fields are processed.
        """
        features = self.features
        state_key = self.state_key
        video_keys = self.video_keys

        # Validate state_key
        if state_key not in features:
            raise ValueError(f"State key {state_key} not found in features")
        if ".state" not in state_key.lower():
            raise ValueError(f"State key {state_key} must contain '.state'")
        if features[state_key].type != "STATE":
            raise ValueError(f"State key {state_key} must map to a STATE feature")

        # Validate video_keys
        if video_keys is not None:
            for key in video_keys:
                if key not in features:
                    raise ValueError(f"Image key {key} not found in features")
                if "image" not in key.lower():
                    raise ValueError(f"Image key {key} must contain 'image'")
                if features[key].type != "VISUAL":
                    raise ValueError(f"Image key {key} must map to a VISUAL feature")

        # Ensure all image keys in features are in video_keys
        feature_video_keys = [k for k in features.keys() if "image" in k.lower()]
        if sorted(video_keys) != sorted(feature_video_keys):
            raise ValueError(
                "Video keys must include all image-related keys in features"
            )

        return self


# Top-level model to validate the entire JSON
class HuggingFaceModelValidator(BaseModel):
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


class Pi0SpawnConfig(BaseModel):
    type: Literal["pi0", "pi0_fast", "pi0.5"] = "pi0.5"
    state_key: str
    state_size: list[int]
    video_keys: list[str]
    video_size: list[int]
    # Pi0 models always expect these three camera names in the input:
    # image.base_0_rgb, image.left_wrist_0_rgb, image.right_wrist_0_rgb
    # Add mapping from camera names to model image keys in your config
    # Ref: https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/model.py
    camera_mappings: Dict[str, str] = Field(
        default_factory=lambda: {
            "observation.images.main.left": "base_0_rgb",
            "observation.images.secondary_0": "left_wrist_0_rgb",
            "observation.images.secondary_1": "right_wrist_0_rgb",
        },
        description="Mapping from camera names to model image keys.",
    )
    hf_model_config: HuggingFaceAugmentedValidator


def fetch_camera_images(
    config: HuggingFaceAugmentedValidator,
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
    for i, camera_name in enumerate(config.input_features.video_keys):
        if cameras_keys_mapping is None:
            camera_id = i
        else:
            camera_id = cameras_keys_mapping.get(camera_name, i)

        video_resolution = config.input_features.features[camera_name].shape
        frame_array = Pi0.fetch_frame(
            all_cameras=all_cameras,
            camera_id=camera_id,
            resolution=video_resolution,
        )
        image_inputs[camera_name] = frame_array

    return image_inputs


class RetryError(Exception):
    """Custom exception to retry the inference call."""

    pass


class Pi0(ActionModel):
    """Client for Pi0 model inference server."""

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
    def fetch_and_get_configuration(cls, model_id: str) -> ModelConfigurationResponse:
        """
        Fetch the model configuration from HuggingFace and return the video keys.
        """
        hf_model_config = cls.fetch_config(model_id=model_id)
        configuration = ModelConfigurationResponse(
            video_keys=hf_model_config.input_features.video_keys,
            checkpoints=hf_model_config.checkpoints,
        )
        return configuration

    @classmethod
    def fetch_spawn_config(cls, model_id: str) -> Pi0SpawnConfig:
        """Fetch spawn configuration for Pi0 model."""
        hf_model_config = cls.fetch_config(model_id=model_id)

        state_key: str = hf_model_config.input_features.state_key
        state_size: list[int] = hf_model_config.input_features.features[state_key].shape
        video_keys: list[str] = hf_model_config.input_features.video_keys
        video_size: list[int] = (
            hf_model_config.input_features.features[video_keys[0]].shape
            if len(video_keys) > 0
            else [3, 224, 224]  # default video resolution
        )

        return Pi0SpawnConfig(
            state_key=state_key,
            state_size=state_size,
            video_keys=video_keys,
            video_size=video_size,
            hf_model_config=hf_model_config,
        )

    @classmethod
    def fetch_and_verify_config(
        cls,
        model_id: str,
        all_cameras: AllCameras,
        robots: list["BaseManipulator"],
        cameras_keys_mapping: Dict[str, int] | None = None,
        verify_cameras: bool = True,
    ) -> Pi0SpawnConfig:
        """
        Verify if the HuggingFace model is compatible with the current setup.
        """
        hf_model_config = cls.fetch_config(model_id=model_id)

        state_key: str = hf_model_config.input_features.state_key
        state_size: list[int] = hf_model_config.input_features.features[state_key].shape
        video_keys: list[str] = hf_model_config.input_features.video_keys
        video_size: list[int] = (
            hf_model_config.input_features.features[video_keys[0]].shape
            if len(video_keys) > 0
            else [3, 224, 224]  # default video resolution
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

        return Pi0SpawnConfig(
            state_key=state_key,
            state_size=state_size,
            video_keys=video_keys,
            video_size=video_size,
            hf_model_config=hf_model_config,
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
        model_spawn_config: Pi0SpawnConfig,
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
        config = model_spawn_config.hf_model_config
        action_dim = config.input_features.action_dim

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
                config=config,
                all_cameras=all_cameras,
                cameras_keys_mapping=cameras_keys_mapping,
            )

            # Verify number of cameras
            if len(image_inputs) != len(config.input_features.video_keys):
                logger.warning(
                    f"Model has {len(config.input_features.video_keys)} cameras but "
                    f"{len(image_inputs)} cameras are plugged."
                )
                control_signal.stop()
                raise Exception(
                    f"Model has {len(config.input_features.video_keys)} cameras but "
                    f"{len(image_inputs)} cameras are plugged."
                )

            # Verify number of robots
            number_of_robots = len(robots)
            number_of_robots_in_config = config.input_features.number_of_arms
            if number_of_robots != number_of_robots_in_config:
                logger.warning(
                    f"Model has {number_of_robots_in_config} robots but {number_of_robots} robots are connected."
                )
                control_signal.stop()
                raise Exception(
                    f"Model has {number_of_robots_in_config} robots but {number_of_robots} robots are connected."
                )

            # Concatenate all robot states
            state = robots[0].read_joints_position(unit="rad")
            for robot in robots[1:]:
                state = np.concatenate(
                    (state, robot.read_joints_position(unit="rad")), axis=0
                )

            # Prepare model input
            inputs: dict[str, np.ndarray | str] = {
                config.input_features.state_key: state,
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


class Pi0TrainerConfig(BaseTrainerConfig):
    """Pi0 trainer configuration."""

    model_type: Literal["ACT", "ACT_BBOX", "gr00t", "pi0", "custom"] = "pi0"
    training_params: TrainingParamsPi0 | None = None


class Pi0Trainer(BaseTrainer):
    """Pi0 model trainer."""

    def __init__(self, config: Pi0TrainerConfig):
        self.config = config

    def train(self, timeout_seconds: int | None = None) -> None:
        """Train a Pi0 model."""
        logger.info(f"Starting Pi0 training for dataset={self.config.dataset_name}")
        raise NotImplementedError("Pi0 training not supported yet!")
