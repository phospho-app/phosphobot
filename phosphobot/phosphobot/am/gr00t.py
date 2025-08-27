import asyncio
import json
import os
import pickle
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Tuple, Optional

import cv2
import zmq
import numpy as np
import pandas as pd
from fastapi import HTTPException
from huggingface_hub import HfApi, snapshot_download
from loguru import logger
from pydantic import BaseModel, Field, model_validator

from phosphobot.am.base import (
    ActionModel,
    BaseTrainer,
    BaseTrainerConfig,
    HuggingFaceTokenValidator,
    TrainingParamsGr00T,
    generate_readme,
    resize_dataset,
)
from phosphobot.camera import AllCameras
from phosphobot.control_signal import AIControlSignal
from phosphobot.hardware.base import BaseManipulator
from phosphobot.models import ModelConfigurationResponse
from phosphobot.utils import background_task_log_exceptions, get_hf_token

# Code from: https://github.com/NVIDIA/Isaac-GR00T/blob/main/gr00t/eval/service.py#L111


class TorchSerializer:
    # TODO: Rename as PickleSerializer

    @staticmethod
    def to_bytes(data: dict) -> bytes:
        buffer = BytesIO()
        # torch.save(data, buffer)
        # use pickle instead of torch
        pickle.dump(data, buffer)
        return buffer.getvalue()

    @staticmethod
    def from_bytes(data: bytes) -> dict:
        buffer = BytesIO(data)
        # obj = torch.load(buffer, weights_only=False)
        # use pickle instead of torch
        obj = pickle.load(buffer)
        return obj


@dataclass
class EndpointHandler:
    handler: Callable
    requires_input: bool = True


class BaseInferenceServer:
    """
    An inference server that spin up a ZeroMQ socket and listen for incoming requests.
    Can add custom endpoints by calling `register_endpoint`.
    """

    def __init__(self, host: str = "*", port: int = 5555):
        self.running = True
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{host}:{port}")
        self._endpoints: dict[str, EndpointHandler] = {}

        # Register the ping endpoint by default
        self.register_endpoint("ping", self._handle_ping, requires_input=False)
        self.register_endpoint("kill", self._kill_server, requires_input=False)

    def _kill_server(self):
        """
        Kill the server.
        """
        self.running = False

    def _handle_ping(self) -> dict:
        """
        Simple ping handler that returns a success message.
        """
        return {"status": "ok", "message": "Server is running"}

    def register_endpoint(
        self, name: str, handler: Callable, requires_input: bool = True
    ):
        """
        Register a new endpoint to the server.

        Args:
            name: The name of the endpoint.
            handler: The handler function that will be called when the endpoint is hit.
            requires_input: Whether the handler requires input data.
        """
        self._endpoints[name] = EndpointHandler(handler, requires_input)

    def run(self) -> None:
        addr = self.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        logger.info(f"Server is ready and listening on {addr}")
        while self.running:
            raw = self.socket.recv()
            try:
                request = TorchSerializer.from_bytes(raw)
                version = request.get("version", 1)
                use_envelope = version >= 2

                endpoint = request.get("endpoint", "get_action")
                if endpoint not in self._endpoints:
                    raise ValueError(f"Unknown endpoint: {endpoint!r}")

                handler = self._endpoints[endpoint]
                if handler.requires_input:
                    result = handler.handler(request.get("data", {}))
                else:
                    result = handler.handler()

                if use_envelope:
                    resp: Dict[str, Any] = {"status": "ok", "result": result}
                    self.socket.send(TorchSerializer.to_bytes(resp))
                else:
                    # legacy: send the bare result dict
                    self.socket.send(TorchSerializer.to_bytes(result))

            except Exception as e:
                tb = traceback.format_exc()
                print(f"[ERROR] {e}\n{tb}")

                if "request" in locals() and request.get("version", 1) >= 2:
                    error_resp: Dict[str, Any] = {
                        "status": "error",
                        "error_type": type(e).__name__,
                        "message": str(e),
                        # omit traceback if you don't want to expose internals
                        "traceback": tb,
                    }
                    self.socket.send(TorchSerializer.to_bytes(error_resp))
                else:
                    # legacy client: single-byte ERROR token
                    self.socket.send(b"ERROR")


class ModalityConfig(BaseModel):
    """Configuration for a modality."""

    delta_indices: list[int]
    """Delta indices to sample relative to the current index. The returned data will correspond to the original data at a sampled base index + delta indices."""
    modality_keys: list[str]
    """The keys to load for the modality in the dataset."""


class BasePolicy(ABC):
    @abstractmethod
    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Abstract method to get the action for a given state.

        Args:
            observations: The observations from the environment.

        Returns:
            The action to take in the environment in dictionary format.
        """
        raise NotImplementedError

    @abstractmethod
    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        """
        Return the modality config of the policy.
        """
        raise NotImplementedError


class RobotInferenceServer(BaseInferenceServer):
    """
    Server with three endpoints for real robot policies
    """

    def __init__(self, model: BasePolicy, host: str = "*", port: int = 5555):
        super().__init__(host, port)
        self.register_endpoint("get_action", model.get_action)
        self.register_endpoint(
            "get_modality_config", model.get_modality_config, requires_input=False
        )

    @staticmethod
    def start_server(policy: BasePolicy, port: int):
        server = RobotInferenceServer(policy, port=port)
        server.run()


class BaseInferenceClient:
    def __init__(
        self, host: str = "localhost", port: int = 5555, timeout_ms: int = 15000
    ):
        self.context = zmq.Context()

        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.version = 2
        self._init_socket()

    def _init_socket(self):
        """Initialize or reinitialize the socket with current settings"""
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def ping(self) -> bool:
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except zmq.error.ZMQError:
            self._init_socket()  # Recreate socket for next attempt
            return False

    def kill_server(self):
        """
        Kill the server.
        """
        self.call_endpoint("kill", requires_input=False)

    def call_endpoint(
        self, endpoint: str, data: Optional[Dict] = None, requires_input: bool = True
    ) -> dict:
        """
        Call an endpoint on the server.

        Args:
            endpoint: The name of the endpoint.
            data: The input data for the endpoint.
            requires_input: Whether the endpoint requires input data.
        """
        request = {"endpoint": endpoint, "version": self.version}
        if requires_input:
            request["data"] = data or {}

        self.socket.send(TorchSerializer.to_bytes(request))
        raw = self.socket.recv()

        # legacy error token
        if raw == b"ERROR":
            raise RuntimeError("Server error (legacy)")

        # decode envelope or raw result
        resp = TorchSerializer.from_bytes(raw)
        if "status" in resp:
            if resp["status"] == "error":
                et, msg = resp.get("error_type", "Error"), resp.get("message", "")
                tb = resp.get("traceback", "")
                raise RuntimeError(f"{et}: {msg}\n\n{tb}")
            return resp.get("result", {})
        else:
            # legacy: the handler's own dict
            return resp

    def __del__(self):
        """Cleanup resources on destruction"""
        self.socket.close()
        self.context.term()


class ExternalRobotInferenceClient(BaseInferenceClient):
    """
    Client for communicating with the RealRobotServer
    """

    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get the action from the server.
        The exact definition of the observations is defined
        by the policy, which contains the modalities configuration.
        """
        return self.call_endpoint("get_action", observations)


class Stats(BaseModel):
    max: list[float]
    min: list[float]
    mean: list[float]
    std: list[float]
    q01: list[float]
    q99: list[float]


class ComponentStatistics(BaseModel):
    active_components: Dict[str, Stats] = Field(
        default_factory=dict,
        description="Dictionary mapping component names to their valid Stats objects",
    )

    class Config:
        extra = "allow"

    @model_validator(mode="before")
    def collect_active_components(self):
        """
        Collect names and values of all fields containing valid Stats before validation.
        Ensures extra fields are included in active_components.
        """
        if not isinstance(self, dict):
            return self

        active = {}
        # Process defined fields and extra fields
        for field_name, value in self.items():
            if field_name == "active_components":
                continue
            # Check if value is a dict that matches Stats structure
            if isinstance(value, dict) and all(
                key in value for key in ["max", "min", "mean", "std", "q01", "q99"]
            ):
                try:
                    # Attempt to parse as Stats
                    stats = Stats(**value)
                    active[field_name] = stats
                except ValueError:
                    pass  # Skip invalid Stats structures
            elif isinstance(value, Stats):
                active[field_name] = value

        # Update active_components in the data
        self["active_components"] = active
        return self

    @property
    def component_names(self) -> list[str]:
        """
        Return a list of active component names for convenience.
        """
        return list(self.active_components.keys())

    @property
    def number_of_arms(self) -> int:
        """
        We assume each arm has 6 joints.
        """
        total_joints = sum(len(stats.max) for stats in self.active_components.values())
        return total_joints // 6

    def get_max_value(self) -> float:
        """
        Return the maximum value across all 'max' fields of Stats instances.
        """
        max_values = []
        for stats in self.active_components.values():
            max_values.extend(stats.max)
        return max(max_values) if max_values else float("-inf")


class StateStatistics(ComponentStatistics):
    pass


class ActionStatistics(ComponentStatistics):
    pass


class Statistics(BaseModel):
    state: StateStatistics
    action: ActionStatistics


class CameraConfig(BaseModel):
    resolution: Tuple[int, int] = Field(
        ...,
        examples=[[320, 240]],
        description="Camera resolution in (width, height) format",
    )
    channels: int = Field(
        ...,
        examples=[3],
        description="Number of color channels (3 for RGB)",
        ge=1,
        le=4,
    )
    fps: float = Field(..., examples=[30.0], description="Frames per second", gt=0)


class ModalitiesConfig(BaseModel):
    video: Dict[str, CameraConfig] = Field(
        ..., description="Dictionary of camera configurations keyed by camera name"
    )


class EmbodimentConfig(BaseModel):
    modalities: ModalitiesConfig
    statistics: Statistics
    embodiment_tag: str


class HuggingFaceModelConfig(BaseModel):
    """
    We use a model validator to extract the embodiment config from the model config.
    """

    # This will store the found embodiment config
    embodiment: EmbodimentConfig
    # This will store the original field name
    embodiment_field_name: Optional[str] = None

    class Config:
        extra = "allow"

    @model_validator(mode="before")
    @classmethod
    def extract_embodiment_config(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that at least one field contains an EmbodimentConfig.
        Extract and set it to the 'embodiment' field for easier access.
        """
        if not isinstance(data, dict):
            return data

        embodiment_field = None

        # Look through all fields for one that matches the EmbodimentConfig structure
        for field_name, value in data.items():
            # Check if the value is a dict and has the required keys for an EmbodimentConfig
            if isinstance(value, dict) and all(
                key in value for key in ["modalities", "statistics", "embodiment_tag"]
            ):
                # We found an embodiment config
                embodiment_field = field_name
                # Store the original field name for reference if needed
                data["embodiment_field_name"] = field_name
                # Store the actual embodiment config in our standard field
                data["embodiment"] = value
                break

        # If no embodiment field was found, raise a validation error
        if embodiment_field is None:
            raise ValueError(
                "No valid embodiment configuration found in the model config"
            )

        return data


class HuggingFaceAugmentedConfig(HuggingFaceModelConfig):
    """
    This model extends HuggingFaceModelConfig to include additional fields
    for augmented models, such as available checkpoints.
    """

    checkpoints: list[str] = Field(
        default_factory=list, description="List of available checkpoints for the model."
    )


class Gr00tSpawnConfig(BaseModel):
    video_keys: list[str]
    state_keys: list[str]
    action_keys: list[str]
    embodiment_tag: str
    hf_model_config: HuggingFaceAugmentedConfig

    # not good enough
    # class Config:
    #     ser_json_inf_nan = "null"
    #     json_encoders = {
    #         float: lambda v: 0.0 if np.isnan(v) or np.isinf(v) else v,
    #         list: lambda v: [0.0 if np.isnan(i) or np.isinf(i) else i for i in v],
    #     }

    # Add a from_file method
    @classmethod
    def from_file(cls, file_path: str) -> "Gr00tSpawnConfig":
        with open(file_path, "r") as f:
            return cls.model_validate_json(f.read())


class Gr00tN1(ActionModel):
    def __init__(
        self,
        action_keys: list[str] = [
            "action.arm_0"
        ],  # These values are read from the values in experiment_cfg/metadata.json
        server_url: str = "localhost",
        server_port: int = 5555,
        **kwargs,
    ):
        super().__init__(server_url, server_port)
        self.client = ExternalRobotInferenceClient(host=server_url, port=server_port)
        self.action_keys = action_keys

    def sample_actions(self, inputs: dict) -> np.ndarray:
        # Get the dict from the server
        response = self.client.get_action(inputs)
        action_parts = []
        for key in self.action_keys:
            new_action = response[key]

            if isinstance(new_action, np.ndarray):
                if new_action.ndim == 1 and len(new_action) == 16:
                    # Handle 1D array of shape (16,) by reshaping to (16, 1)
                    new_action = new_action.reshape(16, 1)
                elif new_action.ndim == 2 and new_action.shape[0] == 16:
                    # Already a 2D array with batch size 16, no reshaping needed
                    pass
                else:
                    raise ValueError(
                        f"Unexpected array shape for key {key}: {new_action.shape}"
                    )

                # Array case: shape is (16, action_size)
                batch_size, action_size = new_action.shape
                if batch_size != 16:
                    raise ValueError(
                        f"Expected batch size 16, got {batch_size} for key {key}"
                    )

                # If action_size is 1 or 6, assume the last column is the gripper
                if action_size in [1, 6]:
                    new_action[:, -1] = np.where(
                        new_action[:, -1] < 0.35, 0.0, new_action[:, -1]
                    )

                action_parts.append(new_action)
            else:
                raise ValueError(
                    f"Unexpected new_action format for key {key}: {type(new_action)}, "
                    f"shape/len: {getattr(new_action, 'shape', len(new_action))}"
                )

        # Concatenate along axis=1 to combine features, preserving batch size of 16
        if not action_parts:
            raise ValueError("No valid actions found to concatenate")

        concatenated_actions = np.concatenate(action_parts, axis=1)

        return concatenated_actions

    @classmethod
    def fetch_config(cls, model_id: str) -> HuggingFaceAugmentedConfig:
        """
        Fetch the model config from Hugging Face Hub.
        If the model is not found on Hugging Face Hub, it will be loaded from the given path.
        """
        try:
            api = HfApi(token=get_hf_token())
            model_info = api.model_info(model_id)
            if model_info is None:
                raise Exception(f"Model {model_id} not found on Hugging Face Hub.")
            else:
                # Download file from the model repo
                config_path = api.hf_hub_download(
                    repo_id=model_id,
                    filename="experiment_cfg/metadata.json",
                    force_download=True,
                )
            # Read the file
            with open(config_path, "r") as f:
                config_content = f.read()
            # Parse the file
            hf_model_config = HuggingFaceModelConfig.model_validate_json(config_content)
            # Fetch the available revisions
            branches = []
            refs = api.list_repo_refs(model_id)
            for branch in refs.branches:
                branches.append(branch.name)

            hf_augmented_config = HuggingFaceAugmentedConfig(
                **hf_model_config.model_dump(), checkpoints=branches
            )

        except Exception as e:
            logger.info(
                f"Couldn't load model {model_id} from Hugging Face Hub. Trying from local path."
            )
            # We now assume it is a local path
            # remove possible trailing slash
            local_path = model_id.rstrip("/")
            config_path = os.path.join(local_path, "experiment_cfg", "metadata.json")

            # Read the file
            with open(config_path, "r") as f:
                config_content = f.read()
            # Parse the file
            hf_model_config = HuggingFaceModelConfig.model_validate_json(config_content)
            hf_augmented_config = HuggingFaceAugmentedConfig(
                **hf_model_config.model_dump(), checkpoints=["main"]
            )

        return hf_augmented_config

    @classmethod
    def fetch_spawn_config(cls, model_id: str) -> Gr00tSpawnConfig:
        hf_model_config = cls.fetch_config(model_id=model_id)

        video_keys = [
            "video." + key for key in hf_model_config.embodiment.modalities.video.keys()
        ]
        state_keys = [
            "state." + key
            for key in hf_model_config.embodiment.statistics.state.component_names
        ]
        action_keys = [
            "action." + key
            for key in hf_model_config.embodiment.statistics.action.component_names
        ]

        return Gr00tSpawnConfig(
            video_keys=video_keys,
            state_keys=state_keys,
            action_keys=action_keys,
            embodiment_tag=hf_model_config.embodiment.embodiment_tag,
            hf_model_config=hf_model_config,
        )

    @classmethod
    def fetch_and_get_configuration(cls, model_id: str) -> ModelConfigurationResponse:
        """
        Fetch the model config and get the video keys.
        """
        hf_model_config = cls.fetch_config(model_id)
        video_keys = [
            "video." + key for key in hf_model_config.embodiment.modalities.video.keys()
        ]
        return ModelConfigurationResponse(
            video_keys=video_keys,
            checkpoints=hf_model_config.checkpoints,
        )

    @classmethod
    def fetch_and_verify_config(
        cls,
        model_id: str,
        all_cameras: AllCameras,
        robots: list[BaseManipulator],
        cameras_keys_mapping: Optional[Dict[str, int]] = None,
        verify_cameras: bool = True,
    ) -> Gr00tSpawnConfig:
        """
        Verify if the HuggingFace model is compatible with the current setup.
        """

        hf_model_config = cls.fetch_config(model_id)

        video_keys = [
            "video." + key for key in hf_model_config.embodiment.modalities.video.keys()
        ]
        state_keys = [
            "state." + key
            for key in hf_model_config.embodiment.statistics.state.component_names
        ]
        action_keys = [
            "action." + key
            for key in hf_model_config.embodiment.statistics.action.component_names
        ]

        number_of_cameras = len(hf_model_config.embodiment.modalities.video.keys())

        if cameras_keys_mapping is None:
            nb_connected_cams = len(all_cameras.video_cameras)
        else:
            # Check if all keys are in the model config
            keys_in_common = set(
                [
                    k.replace("video.", "") if k.startswith("video.") else k
                    for k in cameras_keys_mapping.keys()
                ]
            ).intersection(hf_model_config.embodiment.modalities.video.keys())
            nb_connected_cams = len(keys_in_common)

        number_of_robots = hf_model_config.embodiment.statistics.state.number_of_arms

        # Check if the number of cameras in the model config matches the number of cameras connected
        if nb_connected_cams < number_of_cameras and verify_cameras:
            logger.warning(
                f"Model has {len(hf_model_config.embodiment.modalities.video)} cameras but {nb_connected_cams} camera streams are detected."
            )
            raise HTTPException(
                status_code=400,
                detail=f"Model has {len(hf_model_config.embodiment.modalities.video)} cameras but {nb_connected_cams} camera streams are detected.",
            )

        # Check if the number of robots in the model config matches the number of robots connected
        if number_of_robots != len(robots):
            raise HTTPException(
                status_code=400,
                detail=f"Model has {number_of_robots} robots but {len(robots)} robots are connected.",
            )

        return Gr00tSpawnConfig(
            video_keys=video_keys,
            state_keys=state_keys,
            action_keys=action_keys,
            embodiment_tag=hf_model_config.embodiment.embodiment_tag,
            hf_model_config=hf_model_config,
        )

    @background_task_log_exceptions
    async def control_loop(
        self,
        control_signal: AIControlSignal,
        robots: List[BaseManipulator],
        model_spawn_config: Gr00tSpawnConfig,
        all_cameras: AllCameras,
        prompt: Optional[str] = None,
        fps: int = 30,
        speed: float = 1.0,
        cameras_keys_mapping: Optional[Dict[str, int]] = None,
        unit: Literal["degrees", "rad", "other"] = "rad",
        min_angle: Optional[float] = None,
        max_angle: Optional[float] = None,
        **kwargs: Any,
    ):
        """
        AI control loop that runs in the background and sends actions to the robot.
        It uses the model to get the actions based on the current state of the robot and the cameras.
        The loop runs until the control signal is stopped or the model is not available anymore.
        The loop runs at the specified fps and speed.
        """

        nb_iter = 0
        config = model_spawn_config.hf_model_config
        signal_marked_as_started = False

        while control_signal.is_in_loop():
            logger.debug(
                f"AI control loop iteration {nb_iter}, status: {control_signal.status}"
            )
            if control_signal.status == "paused":
                logger.debug("AI control loop paused")
                await asyncio.sleep(0.1)
                continue

            start_time = time.perf_counter()

            # Get the images from the cameras based on the config
            # For now, just put as many cameras as the model config
            image_inputs: Dict[str, np.ndarray] = {}
            for i, (camera_name, video) in enumerate(
                config.embodiment.modalities.video.items()
            ):
                if cameras_keys_mapping is None:
                    camera_id = i
                else:
                    camera_id = cameras_keys_mapping.get(
                        f"video.{camera_name}", cameras_keys_mapping.get(camera_name, i)
                    )

                rgb_frame = all_cameras.get_rgb_frame(
                    camera_id=camera_id, resize=video.resolution
                )
                if rgb_frame is not None:
                    # Convert to BGR
                    image = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                    # Add a batch dimension (from (240, 320, 3) to (1, 240, 320, 3))
                    converted_array = np.expand_dims(image, axis=0)
                    # Ensure dtype is uint8 (if it isn't already)
                    converted_array = converted_array.astype(np.uint8)
                    image_inputs[f"video.{camera_name}"] = converted_array

                else:
                    logger.warning(
                        f"Camera {camera_name} not available. Sending all black."
                    )
                    image_inputs[f"video.{camera_name}"] = np.zeros(
                        (1, video.resolution[1], video.resolution[0], video.channels),
                        dtype=np.uint8,
                    )

            # Number of cameras
            if len(image_inputs) != len(config.embodiment.modalities.video.keys()):
                logger.warning(
                    f"Model has {len(config.embodiment.modalities.video.keys())} cameras but {len(image_inputs)} cameras are plugged."
                )
                control_signal.stop()
                raise Exception(
                    f"Model has {len(config.embodiment.modalities.video.keys())} cameras but {len(image_inputs)} cameras are plugged."
                )

            # Number of robots
            number_of_robots = len(robots)
            number_of_robots_in_config = (
                config.embodiment.statistics.state.number_of_arms
            )
            if number_of_robots != number_of_robots_in_config:
                logger.warning("No robot connected. Exiting AI control loop.")
                control_signal.stop()
                raise Exception("No robot connected. Exiting AI control loop.")

            # Concatenate all robot states
            state = robots[0].read_joints_position(
                unit=unit, max_value=max_angle, min_value=min_angle
            )
            for robot in robots[1:]:
                state = np.concatenate(
                    (
                        state,
                        robot.read_joints_position(
                            unit=unit, max_value=max_angle, min_value=min_angle
                        ),
                    ),
                    axis=0,
                )

            inputs = {
                **image_inputs,
                "annotation.human.action.task_description": prompt,
            }

            state_index = 0
            for (
                component_name,
                stats,
            ) in config.embodiment.statistics.state.active_components.items():
                num_elements = len(stats.max)
                component_state = state[state_index : state_index + num_elements]
                inputs[f"state.{component_name}"] = component_state.reshape(
                    1, num_elements
                )
                state_index += num_elements
            try:
                actions = self(inputs)
            except Exception as e:
                logger.warning(
                    f"Failed to get actions from model: {e}. Exiting AI control loop."
                )
                control_signal.stop()
                break

            if not signal_marked_as_started:
                control_signal.set_running()
                signal_marked_as_started = True

            nb_actions_too_large = 0
            for action in actions:
                # Early stop
                if not control_signal.is_in_loop():
                    break
                # Send the new joint position to the robot
                action_list = action.tolist()
                for robot_index in range(len(robots)):
                    target_position = action_list[robot_index * 6 : robot_index * 6 + 6]

                    # If the distance between the current and target position is too high, skip the action
                    current_position = robots[robot_index].read_joints_position(
                        unit=unit,
                        max_value=max_angle,
                        min_value=min_angle,
                        source="sim",
                    )
                    max_transition_angles: np.ndarray
                    if unit == "degrees":
                        # The last joint is the gripper, which can open/close
                        max_transition_angles = np.array([90.0] * 5 + [180.0])
                        current_to_target_diff = np.abs(
                            (target_position - current_position + 180) % 360 - 180
                        )

                    elif unit == "rad":
                        # The last joint is the gripper, which can open/close
                        max_transition_angles = np.array([np.pi / 2] * 5 + [np.pi])
                        current_to_target_diff = np.abs(
                            (target_position - current_position + np.pi) % (2 * np.pi)
                            - np.pi
                        )
                    elif (
                        unit == "other"
                        and max_angle is not None
                        and min_angle is not None
                    ):
                        # The last joint is the gripper, which can open/close
                        max_transition_angle = (max_angle - min_angle) / 2
                        max_transition_angles = np.array(
                            [max_transition_angle] * 5 + [max_angle - min_angle]
                        )
                        current_to_target_diff = np.abs(
                            (target_position - current_position + max_angle)
                            % (max_angle - min_angle)
                            - max_transition_angle
                        )
                    else:
                        raise ValueError(f"Unknown unit: {unit}")

                    if np.any(current_to_target_diff > max_transition_angles):
                        largest_diff = np.max(current_to_target_diff)
                        largest_diff_index = np.argmax(current_to_target_diff)
                        error_message = (
                            f"Skipping action for robot {robot_index} because the to joint position {largest_diff_index} difference is too large: {largest_diff} > {max_transition_angles[largest_diff_index]} in units {unit}"
                            + f"\nCurrent position: {current_position}"
                            + f"\nTarget position: {target_position}\n"
                            + "Possible reasons for this error:"
                            + "\n1. Make sure you selected the *right angle unit* in the control page (angle, degrees, other)."
                            + "\n2. Inspect your dataset joints positions to ensure they are within the expected range."
                            + "\n3. There was an issue in the model output, please check the model training and data quality."
                        )
                        if nb_actions_too_large <= 20:
                            logger.warning(error_message)
                            nb_actions_too_large += 1
                            continue
                        else:
                            control_signal.stop()
                            raise Exception(error_message)
                    else:
                        logger.debug(
                            f"Writing joint position to robot {robot_index}: {target_position}"
                        )

                    robots[robot_index].write_joint_positions(
                        angles=target_position,
                        unit=unit,
                        max_value=max_angle,
                        min_value=min_angle,
                    )
                    nb_actions_too_large = 0

                # Wait fps time
                elapsed_time = time.perf_counter() - start_time
                sleep_time = max(0, 1.0 / (fps * speed) - elapsed_time)
                await asyncio.sleep(sleep_time)
                start_time = time.perf_counter()

            nb_iter += 1


class Gr00tTrainerConfig(BaseTrainerConfig):
    # Set the value of model_type to "gr00t"
    model_type: Literal["gr00t"] = "gr00t"
    training_params: TrainingParamsGr00T


def check_for_nans_null_in_value(value):
    """
    Check if a value contains NaN/null, including nested lists
    """
    if pd.isna(value).any():
        return True

    if pd.isnull(value).any():
        return True

    if isinstance(value, (list, tuple)):
        for item in value:
            if check_for_nans_null_in_value(item):
                return True

    return False


def check_parquet_files(folder_path):
    """
    Check all parquet files in a folder for NaN/null values in the action/observation column

    Will raise an error if it finds any NaN/null values in the action/observation column.
    """
    folder_path = Path(folder_path)

    if not folder_path.exists():
        raise FileNotFoundError(f"Folder '{folder_path}' does not exist.")

    # Find all parquet files
    parquet_files = list(folder_path.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in '{folder_path}'")

    print(f"Found {len(parquet_files)} parquet file(s) to check:")
    print("-" * 50)

    total_issues = 0

    for file_path in parquet_files:
        try:
            # Read the parquet file
            df = pd.read_parquet(file_path)

            # Check if action column exists
            if "action" not in df.columns:
                raise ValueError(
                    f"File '{file_path.name}' does not contain 'action' column."
                )
            if "observation.state" not in df.columns:
                raise ValueError(
                    f"File '{file_path.name}' does not contain 'observation.state' column."
                )

            # Check for issues in the action column
            issues_found = 0
            problematic_rows = []

            for idx, value in enumerate(df["action"]):
                if check_for_nans_null_in_value(value):
                    issues_found += 1
                    problematic_rows.append(idx)

            for idx, value in enumerate(df["observation.state"]):
                if check_for_nans_null_in_value(value):
                    issues_found += 1
                    problematic_rows.append(idx)

            if issues_found > 0:
                print(
                    f"❌ {file_path.name}: Found {issues_found} rows with NaN/null in action column"
                )
                print(
                    f"   Problematic rows: {problematic_rows[:10]}{'...' if len(problematic_rows) > 10 else ''}"
                )
                total_issues += issues_found
            else:
                print(f"✅ {file_path.name}: No NaN/null values found in action column")

        except Exception as e:
            print(f"❌ {file_path.name}: Error reading file - {str(e)}")

    print("-" * 50)
    print(f"Total issues found across all files: {total_issues}")
    if total_issues > 0:
        raise ValueError(
            f"Found {total_issues} NaN/null values in action/observation columns across all files. Please fix the data before re-training."
        )


def generate_modality_json(data_dir) -> tuple[int, int]:
    # Load the metadata file to get image keys
    with open(data_dir / "meta" / "info.json", "r") as f:
        metadata = json.load(f)
        image_keys = []
        for key in metadata["features"].keys():
            if "image" in key:
                image_keys.append(key)

    number_of_cameras = len(image_keys)
    number_of_robots: int = metadata["features"]["action"]["shape"][0] // 6
    print(f"Number of cameras: {number_of_cameras}")
    print(f"Number of robots: {number_of_robots}")

    # Create the action/state keys based on the number of robots
    # Each robot has 5 arm keys and 1 gripper key
    robot_keys = []
    for i in range(number_of_robots):
        robot_keys.append(f"arm_{i}")

    # Create the action/state keys
    robot_structure = {}
    index = 0
    for key in robot_keys:
        robot_structure[key] = {"start": index, "end": index + 6}
        index += 6

    # Populate the video section with the image keys
    video_structure: dict = {}
    camera_name = [f"image_cam_{i}" for i in range(number_of_cameras)]
    for i, image_key in enumerate(image_keys):
        video_structure[camera_name[i]] = {"original_key": image_key}  # type: ignore

    # Create the base modality structure
    modality_json = {
        "state": robot_structure,
        "action": robot_structure,
        "video": video_structure,
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }

    print(f"Modality JSON: {modality_json}")

    # Write the modality.json file
    with open(data_dir / "meta" / "modality.json", "w") as f:
        json.dump(modality_json, f, indent=4)

    return number_of_robots, number_of_cameras


async def run_gr00t_training(
    data_dir,
    output_dir,
    batch_size,
    epochs,
    number_of_robots,
    number_of_cameras,
    learning_rate,
    save_steps: int,
    wandb_enabled: bool,
    validation_data_dir=None,
    timeout_seconds: Optional[int] = None,
    gr00t_repo_path: str = ".",
):
    cmd = [
        "python",
        f"{gr00t_repo_path}/scripts/gr00t_finetune.py",
        "--dataset-path",
        str(data_dir),
    ]

    if validation_data_dir is not None:
        logger.info(f"Using validation dataset from {validation_data_dir}")
        cmd.extend(["--validation-dataset-path", str(validation_data_dir)])
    else:
        logger.info("No validation dataset provided. No validation will be done.")

    # Add remaining arguments
    cmd.extend(
        [
            # Only 1 GPU for now
            # Open an issue for multi-GPU support
            "--num-gpus",
            "1",
            "--output-dir",
            str(output_dir),
            "--batch-size",
            str(batch_size),
            "--num-epochs",
            str(epochs),
            "--save-steps",
            str(save_steps),
            "--num-arms",
            str(number_of_robots),
            "--num-cams",
            str(number_of_cameras),
            "--learning_rate",
            str(learning_rate),
            "--report_to",
            "wandb" if wandb_enabled else "tensorboard",
            "--video_backend",
            "torchvision_av",
        ]
    )

    logger.info(f"Starting training with command: {' '.join(cmd)}")

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        # 512 KB buffer size, default is 64 but seems to be too small
        limit=512 * 1024,
    )

    output_lines = []

    async def read_output():
        assert process.stdout is not None
        async for line in process.stdout:
            stripped_line = line.decode().strip()
            print(stripped_line)
            output_lines.append(stripped_line)

    try:
        if timeout_seconds is None:
            # No timeout
            await read_output()
        else:
            # Timeout
            await asyncio.wait_for(read_output(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        logger.error(f"Training process timed out after {timeout_seconds} seconds.")
        raise TimeoutError(
            f"Training process exceeded timeout of {timeout_seconds} seconds. Please consider lowering the number of epochs and/or batch size."
        )

    await process.wait()

    if process.returncode != 0:
        error_output = "\n".join(output_lines[-10:])
        error_msg = f"Training process failed with exit code {process.returncode}:\n{error_output}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    return output_lines


class Gr00tTrainer(BaseTrainer):
    """
    Trainer for the Gr00t model.
    Requires the gr00t repo to be installed (will throw an error if not).
    """

    def __init__(self, config: Gr00tTrainerConfig):
        self.config = config

    def train(
        self,
        timeout_seconds: Optional[int] = None,
        private_mode: bool = False,
        hf_token: Optional[str] = None,
    ) -> None:
        """
        You can pass a timeout in seconds to the training process.
        If the training process exceeds this time, it will be
        killed and the latest checkpoint will be uploaded to Hugging Face.
        """
        logger.info(f"Starting training for dataset={self.config.dataset_name}")

        # Create output directory
        data_dir = Path(self.config.training_params.data_dir)
        output_dir = Path(self.config.training_params.output_dir)
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        selected_branch = "main"
        print(f"Using branch {selected_branch}")

        if self.config.model_name is not None:
            # We check if the user has write access to the model-id
            hf_token = hf_token or os.getenv("HF_TOKEN")
            if hf_token is None:
                raise ValueError(
                    "HF_TOKEN environment variable is not set. Please set it to your Hugging Face token."
                )
            if not HuggingFaceTokenValidator().has_write_access(
                hf_token=hf_token,
                hf_model_name=self.config.model_name,
                private=private_mode,
            ):
                raise ValueError(
                    f"The provided HF token does not have write access to {self.config.model_name}"
                )

        # Download huggingface dataset with huggingface_hub
        logger.info(f"Downloading dataset {self.config.dataset_name} to {data_dir}")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                dataset_path_as_str = snapshot_download(
                    repo_id=self.config.dataset_name,
                    repo_type="dataset",
                    revision=selected_branch,
                    local_dir=str(data_dir),
                    token=os.getenv("HF_TOKEN"),
                )
                DATASET_PATH = Path(dataset_path_as_str)
                logger.info(
                    f"Dataset {self.config.dataset_name} downloaded to {DATASET_PATH}"
                )
                break  # Exit the loop if download is successful
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)  # Wait for 1 second before retrying
                else:
                    raise RuntimeError(
                        f"Failed to download dataset {self.config.dataset_name} after {max_retries} attempts, is Hugging Face down ? : {e}"
                    )

        # Check the dataset for null/nan values in action/observation columns
        check_parquet_files(DATASET_PATH / "data" / "chunk-000")

        resized_successful, _, resize_details = resize_dataset(
            dataset_root_path=DATASET_PATH, resize_to=(224, 224)
        )
        if not resized_successful:
            raise RuntimeError(
                f"Resizing dataset {self.config.dataset_name} to 224x224 failed: {resize_details}"
            )
        logger.info(f"Resized dataset {self.config.dataset_name} to 224x224")

        # Create the modality json file in meta folder
        logger.info("Generating modality.json file")
        number_of_robots, number_of_cameras = generate_modality_json(data_dir)

        val_data_dir: Optional[Path] = None
        if self.config.training_params.validation_dataset_name is not None:
            if self.config.training_params.validation_data_dir is not None:
                val_data_dir = Path(self.config.training_params.validation_data_dir)
            # It can be None if the user has not provided a validation dataset
            else:
                val_data_dir = Path("validation_data/")

            os.makedirs(val_data_dir, exist_ok=True)
            for attempt in range(max_retries):
                try:
                    dataset_path_val_str = snapshot_download(
                        repo_id=self.config.training_params.validation_dataset_name,
                        repo_type="dataset",
                        revision=selected_branch,
                        local_dir=str(val_data_dir),
                        token=os.getenv("HF_TOKEN"),
                    )
                    VAL_DATASET_PATH = Path(dataset_path_val_str)
                    logger.info(
                        f"Validation dataset {self.config.training_params.validation_dataset_name} downloaded to {VAL_DATASET_PATH}"
                    )
                    break
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(1)
                    else:
                        raise RuntimeError(
                            f"Failed to download dataset {self.config.training_params.validation_dataset_name} after {max_retries} attempts, is Hugging Face down ? : {e}"
                        )

            resized_successful, _, resize_details = resize_dataset(
                dataset_root_path=VAL_DATASET_PATH, resize_to=(224, 224)
            )
            if not resized_successful:
                raise RuntimeError(
                    f"Resizing dataset {self.config.training_params.validation_dataset_name} to 224x224 failed: {resize_details}"
                )
            logger.info(
                f"Resized dataset {self.config.training_params.validation_dataset_name} to 224x224"
            )
            logger.info("Generating modality.json file for validation dataset")
            generate_modality_json(val_data_dir)

        else:
            logger.info("No validation dataset provided. No validation will be done.")
            # We set the validation data dir to None to avoid passing it to the training script
            val_data_dir = None

        # Find the total number of frames in the dataset in meta / info.json
        with open(data_dir / "meta" / "info.json", "r") as f:
            info = json.load(f)
            total_frames = info["total_frames"]

        steps = (
            total_frames
            * self.config.training_params.epochs
            // self.config.training_params.batch_size
            + 1
        )

        logger.info(
            f"Will train for {self.config.training_params.epochs} epochs, which is {steps} steps"
        )

        asyncio.run(
            run_gr00t_training(
                data_dir=data_dir,
                output_dir=output_dir,
                batch_size=self.config.training_params.batch_size,
                epochs=self.config.training_params.epochs,
                number_of_robots=number_of_robots,
                number_of_cameras=number_of_cameras,
                learning_rate=self.config.training_params.learning_rate,
                save_steps=self.config.training_params.save_steps,
                wandb_enabled=self.config.wandb_api_key is not None,
                validation_data_dir=val_data_dir,
                timeout_seconds=timeout_seconds,
                gr00t_repo_path=self.config.training_params.path_to_gr00t_repo,
            )
        )
        logger.info("Training finished")
        # Upload to Hugging Face if token is provided
        if self.config.model_name is not None:
            logger.info(f"Uploading model to Hugging Face as {self.config.model_name}")

            # Upload using huggingface_hub
            api = HfApi(token=os.getenv("HF_TOKEN"))
            files_directory = output_dir

            api.create_repo(
                repo_id=self.config.model_name,
                repo_type="model",
                private=False,
                exist_ok=True,
            )

            # Then upload main files to the main branch
            for item in files_directory.glob("*"):
                if item.is_file() and item.name != "README.md":
                    logger.info(
                        f"Uploading {item} to {self.config.model_name} as {item.name}"
                    )
                    api.upload_file(
                        repo_id=self.config.model_name,
                        repo_type="model",
                        path_or_fileobj=str(item.resolve()),
                        path_in_repo=item.name,
                    )

            # Upload experiment_cfg and its contents
            exp_cfg_dir = files_directory / "experiment_cfg"
            if exp_cfg_dir.exists() and exp_cfg_dir.is_dir():
                for item in exp_cfg_dir.glob("**/*"):
                    if item.is_file():
                        # Keep the directory structure
                        rel_path = item.relative_to(files_directory)
                        logger.info(f"Uploading config file: {item} as {rel_path}")
                        api.upload_file(
                            repo_type="model",
                            path_or_fileobj=str(item.resolve()),
                            path_in_repo=str(rel_path),
                            repo_id=self.config.model_name,
                        )

            # Also upload checkpoint directories if they exist, named as "checkpoint-<number>"
            for item in files_directory.glob("checkpoint-*"):
                if item.is_dir():
                    # Upload the entire directory structure
                    for sub_item in item.glob("**/*"):
                        if sub_item.is_file():
                            # Get the relative path to maintain structure
                            rel_path = sub_item.relative_to(item)

                            logger.info(f"Uploading file: {rel_path}")
                            # Parse the checkpoint number as an int
                            try:
                                # Should be 100, 400, etc.
                                checkpoint_number = int(item.name.split("-")[-1])
                            except ValueError:
                                # Can also be "last" or similar
                                logger.debug(
                                    f"Skipping upload for {rel_path} as it does not have a valid checkpoint number"
                                )
                                continue
                            api.create_branch(
                                repo_type="model",
                                branch=str(checkpoint_number),
                                exist_ok=True,
                                repo_id=self.config.model_name,
                            )
                            api.upload_file(
                                repo_type="model",
                                revision=str(checkpoint_number),
                                path_or_fileobj=str(sub_item.resolve()),
                                path_in_repo=str(rel_path),
                                repo_id=self.config.model_name,
                            )

            # Upload README last
            readme = generate_readme(
                model_type="gr00t",
                dataset_repo_id=self.config.dataset_name,
                training_params=self.config.training_params,
                return_readme_as_bytes=True,
            )

            api.upload_file(
                repo_type="model",
                path_or_fileobj=readme,
                path_in_repo="README.md",
                repo_id=self.config.model_name,
            )

            # Get the model URL
            huggingface_model_url = f"https://huggingface.co/{self.config.model_name}"
            logger.info(f"Model successfully uploaded to {huggingface_model_url}")

            logger.info("Model successfully uploaded to Hugging Face")
            logger.info(
                f"Find your trained model on Hugging Face: {huggingface_model_url}"
            )

        else:
            logger.info("Skipping upload to Hugging Face")
            logger.info("Provide a model-id to enable automatic upload to Hugging Face")
