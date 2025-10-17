import asyncio
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

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

from phosphobot.am.base import ActionModel
from phosphobot.camera import AllCameras
from phosphobot.control_signal import AIControlSignal
from phosphobot.models import ModelConfigurationResponse
from phosphobot.utils import background_task_log_exceptions, get_hf_token


class InputFeature(BaseModel):
    type: Literal["STATE", "VISUAL", "ENV"]
    shape: List[int]


class InputFeatures(BaseModel):
    state_key: str
    env_key: Optional[str] = None
    video_keys: List[str] = []
    features: Dict[str, InputFeature]

    @property
    def number_of_arms(self) -> int:
        """
        We assume each arm has 6 joints.
        """
        return self.features[self.state_key].shape[0] // 6

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
            env_keys = [k for k in features if "env" in k.lower()]
            if len(state_keys) != 1:
                raise ValueError(
                    f"Expected exactly one state key, got {len(state_keys)}: {state_keys}"
                )
            state_key = state_keys[0]
            env_key = env_keys[0] if env_keys else None
            video_keys = video_keys
            return {
                "state_key": state_key,
                "env_key": env_key,
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
            elif "env" in key.lower():
                if item.get("type") != "ENV":
                    raise ValueError(f"Key {key} with 'env' must have type 'ENV'")
            else:
                raise ValueError(f"Key {key} must contain 'state', 'image' or 'env'")
            result[key] = InputFeature(**item)
        return result

    @model_validator(mode="after")
    def validate_keys(self) -> "InputFeatures":
        """
        Validate state_key and video_keys against features after all fields are processed.
        """
        features = self.features
        state_key = self.state_key
        env_key = self.env_key
        video_keys = self.video_keys

        # Validate state_key
        if state_key not in features:
            raise ValueError(f"State key {state_key} not found in features")
        if ".state" not in state_key.lower():
            raise ValueError(f"State key {state_key} must contain 'state'")
        if features[state_key].type != "STATE":
            raise ValueError(f"State key {state_key} must map to a STATE feature")

        # Validate video_keys
        if video_keys is not None:
            for key in video_keys:
                if key not in features:
                    raise ValueError(f"Video key {key} not found in features")
                if "image" not in key.lower():
                    raise ValueError(f"Video key {key} must contain 'image'")
                if features[key].type != "VISUAL":
                    raise ValueError(f"Video key {key} must map to a VISUAL feature")

        # Validate env_key if provided
        if env_key is not None:
            if env_key not in features:
                raise ValueError(f"Env key {env_key} not found in features")
            if "env" not in env_key.lower():
                raise ValueError(f"Env key {env_key} must contain 'env'")
            if features[env_key].type != "ENV":
                raise ValueError(f"Env key {env_key} must map to an ENV feature")

        # Ensure all image keys in features are in video_keys
        feature_video_keys = [k for k in features.keys() if "image" in k.lower()]
        if sorted(video_keys) != sorted(feature_video_keys):
            raise ValueError(
                "Video keys must include all image-related keys in features"
            )

        return self


# Base classes that can be inherited by specific model implementations
class HuggingFaceModelValidator(BaseModel):
    """Base class for HuggingFace model validators"""
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


class LeRobotSpawnConfig(BaseModel):
    """Base configuration class for LeRobot model spawning"""
    state_key: str
    state_size: List[int]
    env_key: Optional[str] = None
    env_size: Optional[List[int]] = None
    video_keys: List[str]
    video_size: List[int]
    hf_model_config: HuggingFaceAugmentedValidator  # type: ignore[assignment]


class RetryError(Exception):
    """Custom exception to retry the inference call."""
    pass


class LeRobot(ActionModel, ABC):
    """Base class for LeRobot action policy models"""

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
    @abstractmethod
    def _get_model_validator_class(cls) -> type[HuggingFaceModelValidator]:
        """Return the specific model validator class for this model type"""
        pass

    @classmethod
    @abstractmethod
    def _get_augmented_validator_class(cls) -> type[HuggingFaceAugmentedValidator]:
        """Return the specific augmented validator class for this model type"""
        pass

    @classmethod
    @abstractmethod
    def _get_spawn_config_class(cls) -> type[LeRobotSpawnConfig]:
        """Return the specific spawn config class for this model type"""
        pass

    def sample_actions(self, inputs: dict) -> np.ndarray:
        # Double-encoded version (to send numpy arrays as JSON)
        encoded_payload = {"encoded": json_numpy.dumps(inputs)}

        try:
            response = self.sync_client.post("/act", json=encoded_payload)

            if response.status_code == 202:
                raise RetryError(response.content)

            if response.status_code != 200:
                raise RuntimeError(response.text)
            actions = json_numpy.loads(response.json())
        except RetryError as e:
            raise RetryError(e)
        except Exception as e:
            logger.error(f"Error in sampling actions: {type(e).__name__}: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Error in sampling actions: {type(e).__name__}: {e}",
            )
        return actions

    async def async_sample_actions(self, inputs: dict) -> np.ndarray:
        # Clean up the input to avoid JSON serialization issues
        encoded_payload = {"encoded": json_numpy.dumps(inputs)}

        try:
            response = await self.async_client.post("/act", json=encoded_payload, timeout=30)

            if response.status_code == 202:
                raise RetryError(response.content)

            if response.status_code != 200:
                raise RuntimeError(response.text)
            actions = json_numpy.loads(response.json())
        except RetryError as e:
            raise RetryError(e)
        except Exception as e:
            logger.error(f"Error in sampling actions: {type(e).__name__}: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Error in sampling actions: {type(e).__name__}: {e}",
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

            # Use the specific validator class from the subclass
            model_validator_class = cls._get_model_validator_class()
            hf_model_config = model_validator_class.model_validate_json(config_content)  # type: ignore[attr-defined]

            # Use the specific augmented validator class from the subclass
            augmented_validator_class = cls._get_augmented_validator_class()
            hf_augmented_config = augmented_validator_class(
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
    def fetch_spawn_config(cls, model_id: str) -> LeRobotSpawnConfig:
        hf_model_config = cls.fetch_config(model_id=model_id)

        state_key: str = hf_model_config.input_features.state_key
        state_size: List[int] = hf_model_config.input_features.features[state_key].shape
        env_key: Optional[str] = hf_model_config.input_features.env_key
        env_size: Optional[List[int]] = (
            hf_model_config.input_features.features[env_key].shape
            if env_key is not None
            else None
        )
        video_keys: List[str] = hf_model_config.input_features.video_keys
        video_size: List[int] = (
            hf_model_config.input_features.features[video_keys[0]].shape
            if len(video_keys) > 0
            else [3, 224, 224]
        )

        # Use the specific spawn config class from the subclass
        spawn_config_class = cls._get_spawn_config_class()
        return spawn_config_class(
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
        robots: List["BaseManipulator"],
        cameras_keys_mapping: Optional[Dict[str, int]] = None,
        verify_cameras: bool = True,
    ) -> LeRobotSpawnConfig:
        """
        Verify if the HuggingFace model is compatible with the current setup.
        """

        hf_model_config = cls.fetch_config(model_id=model_id)

        state_key: str = hf_model_config.input_features.state_key
        state_size: List[int] = hf_model_config.input_features.features[state_key].shape
        env_key: Optional[str] = hf_model_config.input_features.env_key
        env_size: Optional[List[int]] = (
            hf_model_config.input_features.features[env_key].shape
            if env_key is not None
            else None
        )
        video_keys: List[str] = hf_model_config.input_features.video_keys
        video_size: List[int] = (
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

        # Use the specific spawn config class from the subclass
        spawn_config_class = cls._get_spawn_config_class()
        return spawn_config_class(
            state_key=state_key,
            state_size=state_size,
            env_key=env_key,
            env_size=env_size,
            video_keys=video_keys,
            video_size=video_size,
            hf_model_config=hf_model_config,
        )

    @classmethod
    def fetch_frame(
        cls, all_cameras: AllCameras, camera_id: int, resolution: List[int]
    ) -> np.ndarray:
        """Fetch a frame from the specified camera and resize it to the given resolution"""

        # Get RGB frame from the camera
        rgb_frame = all_cameras.get_rgb_frame(
            camera_id=camera_id,
            resize=(resolution[2], resolution[1]),
        )

        if rgb_frame is not None:
            # Convert to BGR
            image = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            # Ensure dtype is uint8 (if it isn't already)
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

    def _prepare_image_inputs(
        self,
        config: HuggingFaceAugmentedValidator,
        all_cameras: AllCameras,
        cameras_keys_mapping: Optional[Dict[str, int]]
    ) -> Dict[str, np.ndarray]:
        """Prepare image inputs from cameras based on the config"""

        image_inputs: Dict[str, np.ndarray] = {}
        for i, camera_name in enumerate(config.input_features.video_keys):
            if cameras_keys_mapping is None:
                camera_id = i
            else:
                camera_id = cameras_keys_mapping.get(camera_name, i)

            video_resolution = config.input_features.features[camera_name].shape
            frame_array = self.fetch_frame(
                all_cameras=all_cameras,
                camera_id=camera_id,
                resolution=video_resolution,
            )
            image_inputs[camera_name] = frame_array
        return image_inputs

    def _prepare_robot_state(self, robots: List["BaseManipulator"]) -> tuple[np.ndarray, Dict[int, int]]:
        """Prepare robot state with size mapping"""
        robot_size_mapping = {}
        state = robots[0].read_joints_position(unit="rad")
        robot_size_mapping[0] = state.shape[0]
        for i, robot in enumerate(robots[1:], 1):
            robot_state = robot.read_joints_position(unit="rad")
            state = np.concatenate((state, robot_state), axis=0)
            robot_size_mapping[i] = robot_state.shape[0]
        return state, robot_size_mapping

    @abstractmethod
    def _prepare_model_inputs(
        self,
        config: HuggingFaceAugmentedValidator,
        state: np.ndarray,
        image_inputs: Dict[str, np.ndarray],
        prompt: Optional[str] = None,
        selected_camera_id: Optional[int] = None,
        all_cameras: Optional[AllCameras] = None,
    ) -> Dict[str, np.ndarray | str]:
        """Prepare model inputs. Must be implemented by subclasses."""
        pass

    def _send_actions_to_robots(
        self,
        action: np.ndarray,
        robots: List["BaseManipulator"],
        robot_size_mapping: Dict[int, int],
        angle_format: Literal["degrees", "radians", "other"] = "radians",
        min_angle: Optional[float] = None,
        max_angle: Optional[float] = None,
    ) -> None:
        """Send actions to robots."""
        # Send the new joint position to the robot
        action_list = action.tolist()

        unit: Literal["rad", "motor_units", "degrees", "other"]
        if angle_format == "radians":
            unit = "rad"
        else:
            unit = angle_format

        rolling_count = 0
        for robot_index in range(len(robots)):
            angles = action_list[
                rolling_count : robot_size_mapping[robot_index] + rolling_count
            ]
            rolling_count += robot_size_mapping[robot_index]
            robots[robot_index].write_joint_positions(
                angles=angles,
                unit=unit,
                min_value=min_angle,
                max_value=max_angle,
            )

    @background_task_log_exceptions
    async def control_loop(
        self,
        control_signal: AIControlSignal,
        robots: List["BaseManipulator"],
        model_spawn_config: LeRobotSpawnConfig,
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
            image_inputs = self._prepare_image_inputs(config, all_cameras, cameras_keys_mapping)

            # Number of cameras
            if len(image_inputs) != len(config.input_features.video_keys):
                logger.error(
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
                logger.error(
                    f"Model has {number_of_robots_in_config} arms but {number_of_robots} robots are connected."
                    " Exiting AI control loop."
                )
                control_signal.stop()
                raise Exception(
                    f"Model has {number_of_robots_in_config} arms but {number_of_robots} robots are connected."
                    " Exiting AI control loop."
                )

            # Prepare robot state
            state, robot_size_mapping = self._prepare_robot_state(robots)

            # Prepare model inputs (specific to each model type)
            inputs = self._prepare_model_inputs(
                config, state, image_inputs, prompt, selected_camera_id, all_cameras
            )

            try:
                if len(actions_queue) == 0:
                    actions = await self.async_sample_actions(inputs)
                    actions_queue.extend(actions)
                actions = actions_queue.popleft()
            except RetryError:
                # Only applicable for ACT models, exception is raised in modal/lerobot_modal/act.py
                logger.warning("Could not detect the target object. Retrying...")
                continue
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

                # Send actions to robots (specific to each model type)
                self._send_actions_to_robots(
                    action=action,
                    robots=robots,
                    robot_size_mapping=robot_size_mapping,
                    angle_format=angle_format,
                    min_angle=min_angle,
                    max_angle=max_angle
                )

                # Wait fps time
                elapsed_time = time.perf_counter() - start_time
                sleep_time = max(0, 1.0 / (fps * speed) - elapsed_time)
                await asyncio.sleep(sleep_time)
                start_time = time.perf_counter()

            nb_iter += 1
