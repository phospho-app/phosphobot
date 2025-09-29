import asyncio
import json
import time
from collections import deque
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple

if TYPE_CHECKING:
    # We only need BaseManipulator for type checking
    # This prevents loading pybullet in modal
    from phosphobot.hardware.base import BaseManipulator

import functools

import cv2
import msgpack
import numpy as np
import websockets.sync.client
from fastapi import HTTPException
from huggingface_hub import HfApi
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from websockets.exceptions import InvalidMessage

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


class DataModel(BaseModel):
    class Config:
        extra = "allow"

    repo_id: str
    image_keys: List[str]


class ModelModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    action_dim: int
    pio5: Literal[True] = True


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    exp_name: str
    model: ModelModel
    data: DataModel
    seed: int
    batch_size: int
    num_train_steps: int


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


# This code comes from openpi-client module https://github.com/phospho-app/openpi/blob/main/packages/openpi-client/src/openpi_client/msgpack_numpy.py
def pack_array(obj: Any) -> Any:
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in (
        "V",
        "O",
        "c",
    ):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def unpack_array(obj: Any) -> Any:
    if b"__ndarray__" in obj:
        return np.ndarray(
            buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"]
        )

    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    return obj


Packer = functools.partial(msgpack.Packer, default=pack_array)
packb = functools.partial(msgpack.packb, default=pack_array)

Unpacker = functools.partial(msgpack.Unpacker, object_hook=unpack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)


class WebsocketClientPolicy:
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
    ) -> None:
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = Packer()
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logger.info(f"Waiting for server at {self._uri}...")
        # Despite the error we try up to 20 times to connect
        for _ in range(20):
            try:
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    open_timeout=40,
                    close_timeout=40,
                    ping_interval=120,
                    ping_timeout=40,
                )
                metadata = unpackb(conn.recv())
                return conn, metadata
            except InvalidMessage:
                logger.info("Still waiting for server...")
                time.sleep(5)
        raise RuntimeError(f"Could not connect to server at {self._uri}")

    def infer(self, obs: Dict) -> Dict:
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return unpackb(response)


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
        image_inputs[camera_name.replace("observation.", "observation/")] = frame_array

    return image_inputs


class RetryError(Exception):
    """Custom exception to retry the inference call."""

    pass


class Pi05(ActionModel):
    """Client for Pi0.5 model inference server."""

    def __init__(
        self,
        image_keys: List[str] = [
            "observation.images.main",
            "observation.images.secondary_0",
        ],
        server_url: str = "localhost",
        server_port: int = 5555,
        **kwargs: Any,
    ):
        super().__init__(server_url, server_port)
        self.client = WebsocketClientPolicy(host=server_url, port=server_port)
        self.image_keys = image_keys

    def sample_actions(self, inputs: dict) -> np.ndarray:
        try:
            response = self.client.infer(obs=inputs)

            logger.debug(f"Response from server: {response}")
            if isinstance(response, dict) and "actions" in response:
                actions = np.array(response["actions"])
            else:
                raise ValueError(f"Invalid response from model server: {response}")
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
        try:
            response = self.client.infer(obs=inputs)

            logger.debug(f"Response from server: {response}")
            if isinstance(response, dict) and "actions" in response:
                actions = np.array(response["actions"])
            else:
                raise ValueError(f"Invalid response from model server: {response}")
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
                filename="config.json",
                force_download=True,
            )

            with open(config) as config_file:
                config_dict = json.load(config_file)
            config_parsed = ConfigModel.model_validate(config_dict)

            norm_stats = api.hf_hub_download(
                repo_id=model_id,
                filename="norm_stats.json",
                force_download=True,
            )
            with open(norm_stats, "r") as f:
                norm_stats_content = json.load(f)
            norm_parsed = NormFile.model_validate(norm_stats_content)

            return HuggingFaceAugmentedValidator(
                config=Pi05SpawnConfig(
                    action_dim=norm_parsed.action_dim,
                    image_keys=config_parsed.data.image_keys,
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

        if verify_cameras:
            if cameras_keys_mapping is None:
                raise HTTPException(
                    status_code=400,
                    detail="Cameras keys mapping is required to verify the model cameras.",
                )
            if len(cameras_keys_mapping) != len(hf_model_config.config.image_keys):
                raise HTTPException(
                    status_code=400,
                    detail="Cameras keys mapping does not match model cameras.",
                )

        number_of_model_actions = hf_model_config.config.action_dim
        # To determine the number of connected joints, we do read_joints_position on each robot and sum the lengths of the returned arrays
        number_of_connected_joints = sum(
            robot.read_joints_position(unit="rad").shape[0] for robot in robots
        )
        if number_of_model_actions != number_of_connected_joints:
            raise HTTPException(
                status_code=400,
                detail=f"Model has {number_of_model_actions} action dimensions but we found {number_of_connected_joints} connected joints on {len(robots)} robots.",
            )

        return Pi05SpawnConfig(
            action_dim=number_of_model_actions,
            image_keys=hf_model_config.config.image_keys,
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
        fps: int = 10,  # Pi0 model family operates at 10 fps
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

            # Concatenate all robot states
            robot_idx_joints_mapping = {}
            state = robots[0].read_joints_position(unit="rad", source="robot")
            robot_idx_joints_mapping[0] = state.shape[0]
            for robot in robots[1:]:
                joints_position = robot.read_joints_position(unit="rad", source="robot")
                state = np.concatenate((state, joints_position), axis=0)
                robot_idx_joints_mapping[len(robot_idx_joints_mapping)] = (
                    joints_position.shape[0]
                )

            logger.debug(f"Using state: {state}")

            # Verify number of joints
            number_of_joints_in_config = model_spawn_config.action_dim
            number_of_connected_joints = sum(
                robot_idx_joints_mapping.values()
            )  # num_actuated_joints is not reliable here, some robots like the piper have a separate gripper
            if number_of_connected_joints != number_of_joints_in_config:
                logger.warning(
                    f"Model has {number_of_joints_in_config} joints but {number_of_connected_joints} joints are connected with {len(robots)} robots."
                )
                control_signal.stop()
                raise Exception(
                    f"Model has {number_of_joints_in_config} joints but {number_of_connected_joints} joints are connected with {len(robots)} robots."
                )

            # Prepare model input
            inputs: dict[str, np.ndarray | str] = {
                "observation/state": state,
                "prompt": prompt,
                **image_inputs,
            }

            try:
                # We predict batches of 50 actions and recalculate them when needed
                if len(actions_queue) == 0:
                    actions_dict = self.client.infer(obs=inputs)
                    if isinstance(actions_dict, dict) and "actions" in actions_dict:
                        actions = np.array(actions_dict["actions"])
                    else:
                        raise ValueError(
                            f"Invalid response from model server: {actions_dict}"
                        )
                    actions_queue.extend(actions)
                actions = actions_queue.popleft()  # actions will be of size action_dim, by default 32, this is expected, we ignore the ones > number of joints
            except Exception as e:
                logger.warning(
                    f"Failed to get actions from model, exiting AI control loop.\nError: {e}"
                )
                control_signal.stop()
                break

            if not signal_marked_as_started:
                control_signal.set_running()
                signal_marked_as_started = True

            # Early stop
            if not control_signal.is_in_loop():
                break

            unit: Literal["rad", "motor_units", "degrees", "other"]
            if angle_format == "radians":
                unit = "rad"
            else:
                unit = angle_format

            actions_list = actions.tolist()

            for robot_index in range(len(robots)):
                rolling_count = 0
                angles = actions_list[
                    rolling_count : rolling_count
                    + robot_idx_joints_mapping[robot_index]
                ]
                logger.debug(
                    f"Sending actions to robot {robot_index}: {angles} in {unit}"
                )
                robots[robot_index].write_joint_positions(
                    angles=angles,
                    unit=unit,
                    joints_ids=None,
                    min_value=min_angle,
                    max_value=max_angle,
                )
                rolling_count += robot_idx_joints_mapping[robot_index]

            # Wait fps time
            elapsed_time = time.perf_counter() - start_time
            sleep_time = max(0, 1.0 / (fps * speed) - elapsed_time)
            await asyncio.sleep(sleep_time)
            start_time = time.perf_counter()

            nb_iter += 1
