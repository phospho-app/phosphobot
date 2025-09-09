import asyncio
from collections import deque
import difflib
import time
import traceback
import websockets
import threading
import websockets.sync.client
import signal
import http
import httpx
import json_numpy  # type: ignore
import enum
from abc import ABC
import dataclasses
from typing import Any, Callable, Dict, List, Literal, Tuple
from typing_extensions import override
from pathlib import Path

import cv2
import numpy as np
import torch
from einops import rearrange
from fastapi import HTTPException
from huggingface_hub import HfApi
from loguru import logger
from pydantic import BaseModel, Field, field_validator, model_validator

from phosphobot.am.base import (
    ActionModel,
    BaseTrainer,
    BaseTrainerConfig,
    TrainingParamsPi0
)
from phosphobot.camera import AllCameras
from phosphobot.control_signal import AIControlSignal
from phosphobot.hardware.base import BaseManipulator
from phosphobot.models import ModelConfigurationResponse
from phosphobot.utils import background_task_log_exceptions, get_hf_token

import websockets.asyncio.server as _server
from websockets.http11 import Request, Response
from openpi_client import base_policy, msgpack_numpy
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi.models import model as _model
from openpi.models import pi0, pi0_fast
from openpi import transforms as _transforms
from openpi.training.weight_loaders import CheckpointWeightLoader


@dataclasses.dataclass
class EndpointHandler:
    handler: Callable
    requires_input: bool = True


# Health Check for external monitoring (Simpler HTTP response)
def _health_check(connection: _server.ServerConnection, request: Request) -> Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None


# Ref: https://github.com/Physical-Intelligence/openpi/blob/main/scripts/serve_policy.py
class BaseWebSocketServer:
    """Base class for WebSocket servers."""

    def __init__(
        self,
        policy: base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = 8000,
        metadata: dict | None = None
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._packer = msgpack_numpy.Packer()
        self._server: _server.Server | None = None
        self._shutdown_event = asyncio.Event()
        self._endpoints: dict[str, EndpointHandler] = {}

        self.register_endpoint("ping", self._handle_ping, requires_input=False)
        self.register_endpoint("kill", self._handle_kill, requires_input=False)

    def start_in_background(self) -> threading.Thread:
        """Start the server as a background thread."""
        thread = threading.Thread(target=self.start)
        thread.daemon = True
        thread.start()
        return thread

    def start(self):
        """Start the server and run until interrupted."""
        try:
            asyncio.run(self.run())
        except KeyboardInterrupt:
            logger.info("Received KeyboardInterrupt, shutting down gracefully")
        except Exception as e:
            logger.error(f"Server error: {e}", exc_info=True)
            raise

    async def run(self):
        """Main server loop with simple signal handling."""
        # Simple signal handler setup
        loop = asyncio.get_running_loop()
        for sig in [signal.SIGTERM, signal.SIGINT]:
            if hasattr(signal, sig.name):
                loop.add_signal_handler(sig, self._shutdown_event.set)

        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            logger.info(f"WebSocket server started on {self._host}:{self._port}")
            self._server = server

            # Create tasks for running server and waiting for shutdown event
            server_task = asyncio.create_task(server.serve_forever())
            shutdown_task = asyncio.create_task(self._shutdown_event.wait())

            try:
                # Wait for either server completion or shutdown
                done, pending = await asyncio.wait(
                    [server_task, shutdown_task],
                    return_when=asyncio.FIRST_COMPLETED
                )

                # Cancel remaining tasks
                for task in pending:
                    task.cancel()

            except Exception as e:
                logger.error(f"Server error: {e}")
                raise

    def register_endpoint(self, name: str, handler: Callable, requires_input: bool = True) -> None:
        """
        Register a new endpoint to the server.

        Args:
            name: The name of the endpoint.
            handler: The handler function that will be called when the endpoint is hit.
            requires_input: Whether the handler requires input data.
        """
        self._endpoints[name] = EndpointHandler(handler, requires_input)

    async def _handle_ping(self) -> dict:
        """Simple ping handler that returns a success message."""
        return {"status": "ok", "message": "Server is running"}

    async def _handle_kill(self) -> dict:
        """Handle kill command to initiate server shutdown."""
        logger.info("Kill command received, initiating shutdown")
        self._shutdown_event.set()
        return {"status": "shutting_down", "message": "Server shutdown initiated"}

    async def _handler(self, websocket: _server.ServerConnection):
        """Handle individual websocket connections."""
        logger.info(f"Connection from {websocket.remote_address} opened")

        try:
            await websocket.send(self._packer.pack(self._metadata))

            # This is used to track the total time to send an action for an observation
            # (including network latency). Disabled for now.
            # prev_total_time = None
            while True:
                # Check if shutdown has been initiated
                if self._shutdown_event.is_set():
                    logger.info(f"Shutdown initiated, closing connection from {websocket.remote_address}")
                    break

                try:
                    request = await msgpack_numpy.unpackb(websocket.recv())
                    endpoint = request.get("endpoint", "get_action")

                    if endpoint not in self._endpoints:
                        logger.error(f"Request from {websocket.remote_address} on unknown endpoint: {endpoint}")
                        error_response = {
                            "status": "error",
                            "error_type": "UnknownEndpoint",
                            "message":
                                f"Unknown endpoint {endpoint}. Available endpoints: {list(self._endpoints.keys())}"
                        }
                        await websocket.send(self._packer.pack(error_response))
                        continue

                    endpoint_handler = self._endpoints[endpoint]
                    if endpoint_handler.requires_input:
                        input_data = request.get("data", {})
                        response = await endpoint_handler.handler(input_data)
                    else:
                        response = await endpoint_handler.handler()

                    await websocket.send(self._packer.pack(response))
                except websockets.ConnectionClosed:
                    logger.info(f"Connection from {websocket.remote_address} closed")
                    break
                except Exception as e:
                    tb = traceback.format_exc()
                    logger.error(f"Error handling request from {websocket.remote_address}: {e}\n{tb}")
                    await websocket.send(
                        self._packer.pack(
                            {
                                "status": "error",
                                "error_type": type(e).__name__,
                                "message": str(e),
                                "traceback": tb
                            }
                        )
                    )
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Handler error for {websocket.remote_address}: {e}\n{tb}")
        finally:
            logger.info(f"Connection from {websocket.remote_address} handler finished")


class Pi0WebSocketServer(BaseWebSocketServer):
    """
    Pi0 WebSocket Inference server
    """

    def __init__(self, policy, host: str = "0.0.0.0", port: int = 8000, metadata=None):
        super().__init__(policy, host, port, metadata)
        self.register_endpoint("get_action", self._handle_get_action, requires_input=True)
        logger.info(f"Initialized Pi0WebSocketServer on {host}:{port}")

    async def _handle_get_action(self, obs: Dict) -> Dict:
        """
        Handle policy inference requests.

        Args:
            obs: Observation data from the client

        Returns:
            Action dictionary from the policy.
        """
        infer_time = time.monotonic()
        # TODO: Need to wrap observation into a dataclass
        action = self._policy.infer(obs)
        infer_time = time.monotonic() - infer_time

        action["server_timing"] = {
            "infer_ms": infer_time * 1000,
        }
        return action


class BaseWebSocketClient(ABC):
    """Client for communicating with Pi0 WebSocket server."""

    server_metadata = property(lambda self: self._server_metadata)

    def __init__(self, host: str = "localhost", port: int = 8000, timeout: float = 30.0):
        self._timeout = timeout
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._ws, self._server_metadata = self._wait_for_server()

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logger.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = None
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    timeout=self._timeout
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logger.info("Connection refused by server, retrying after 5 seconds...")
                time.sleep(5)

    def disconnect(self):
        """Disconnect from the WebSocket server."""
        if self._ws:
            self._ws.close()
            logger.info(f"Disconnected from server at {self._uri}")
            self._ws, self._server_metadata = None, None

    def call_endpoint(
        self, endpoint: str, data: Dict | None = None, requires_input: bool = True
    ) -> Dict:
        """Call an endpoint on the server.

        Args:
            endpoint: The name of the endpoint.
            data: The input data for the endpoint.
            requires_input: Whether the endpoint requires input data.
        """
        if not self._ws:
            raise RuntimeError("Websocket Client is not connected to the server.")

        request: Dict[str, Any] = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data if data is not None else {}

        self._ws.send(self._packer.pack(request))
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in policy server call to endpoint {endpoint}:\n{response}")
        return msgpack_numpy.unpackb(response)

    def ping(self) -> bool:
        response = self.call_endpoint("ping", requires_input=False)
        if response.get("status") == "ok":
            return True
        else:
            logger.error(f"Ping failed for {self._uri}: {response}")
            return False

    def kill_server(self) -> bool:
        response = self.call_endpoint("kill", requires_input=False)
        if response.get("status") == "shutting_down":
            logger.info(f"Server shutdown initiated successfully for {self._uri}")
            return True
        else:
            logger.error(f"Failed to initiate server shutdown for {self._uri}: {response}")
            return False

    def reset(self) -> None:
        """Reset the policy to its initial state."""
        pass

    def __del__(self):
        """Close the connection on destruction."""
        self.disconnect()


class Pi0WebSocketClient(BaseWebSocketClient):
    """Pi0 WebSocket Client for connecting with the Pi0WebSocketServer."""

    def get_action(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get action from the Pi0 policy server
        The exact definition of the observations is defined
        by the policy, which contains the modalities configuration.
        """
        return self.call_endpoint("get_action", data=obs, requires_input=True)


# Ref: https://github.com/Physical-Intelligence/openpi/blob/main/scripts/serve_policy.py#L14
class Pi0EnvMode(enum.Enum):
    """Supported environments."""
    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[Pi0EnvMode, Checkpoint] = {
    Pi0EnvMode.ALOHA: Checkpoint(
        config="pi0_aloha",
        dir="gs://openpi-assets/checkpoints/pi0_base",
    ),
    Pi0EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    Pi0EnvMode.DROID: Checkpoint(
        config="pi0_fast_droid",
        dir="gs://openpi-assets/checkpoints/pi0_fast_droid",
    ),
    Pi0EnvMode.LIBERO: Checkpoint(
        config="pi0_fast_libero",
        dir="gs://openpi-assets/checkpoints/pi0_fast_libero",
    ),
}


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


class InputFeature(BaseModel):
    """Individual input feature for the pi0 model"""
    type: Literal["STATE", "VISUAL"]
    shape: List[int]


class InputFeatures(BaseModel):
    """Collection of input features for the pi0 model."""
    state_key: str
    video_keys: List[str] = []
    action_size: int = 6
    features: Dict[str, InputFeature]

    @property
    def number_of_arms(self) -> int:
        """
        Currently all supported robots have 6 joints.
        To be changed when this is no longer true.
        """
        return self.features[self.state_key].shape[0] // self.action_size

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
                "features": features
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
    type: Literal["pi0"]
    # policy_config_name: Literal["pi0_so100", "pi0_fast_so100", "pi0_fast_so100_low_mem_finetune"] | None
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
    state_key: str
    state_size: list[int]
    video_keys: list[str]
    video_size: list[int]
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
    def __init__(
        self,
        server_url: str = "http://localhost",
        server_port: int = 8080,
        # env_mode: str = "aloha_sim",
        # default_prompt: str | None = None,
        # action_keys: list[str] = ["action"],
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
        # self.client = Pi0WebSocketClient(host=server_url, port=server_port)
        # self.env_mode = env_mode
        # self.default_prompt = default_prompt
        # self.action_keys = action_keys

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
        robots: list[BaseManipulator],
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

    # TODO: add this method to ActionModel and call it in Gr00tN1.control_loop()
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

    @staticmethod
    def process_image(
        policy: base_policy.BasePolicy,
        model_specifics: Pi0SpawnConfig,
        current_qpos: list[float],
        images: list[np.ndarray],
        image_names: list[str],
        target_size: tuple[int, int],
    ) -> np.ndarray:
        """
        Process images and perform inference using the policy.
        """
        assert (
            len(current_qpos) == model_specifics.state_size[0]
        ), f"State size mismatch: {len(current_qpos)} != {model_specifics.state_size[0]}"
        assert (
            len(images) <= len(model_specifics.video_keys)
        ), f"Number of images {len(images)} is more than the number of video keys {len(model_specifics.video_keys)}"
        if len(images) > 0:
            assert (
                len(images[0].shape) == 3
            ), f"Image shape is not correct, {images[0].shape} expected (H, W, C)"
            assert (
                len(images[0].shape) == 3 and images[0].shape[2] == 3
            ), f"Image shape is not correct {images[0].shape} expected (H, W, 3)"

        with torch.no_grad(), torch.autocast(device_type="cuda"):
            current_qpos = current_qpos.copy()
            state_tensor = (
                torch.from_numpy(current_qpos)
                .view(1, len(current_qpos))
                .float()
                .to("cuda")
            )

            batch: dict[str, Any] = {
                model_specifics.state_key: state_tensor,
            }

            processed_images = []
            for i, image in enumerate(images):
                # Double check if image.shape[:2] is (H, W) or (W, H)
                if image.shape[:2] != target_size:
                    logger.info(
                        f"Resizing image {image_names[i]} from {image.shape[:2]} to {target_size}"
                    )
                    image = cv2.resize(src=image, dsize=target_size)

                tensor_image = (
                    torch.from_numpy(image)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .float()
                    .to("cuda")
                )
                tensor_image = tensor_image / 255.0
                processed_images.append(tensor_image)
                batch[image_names[i]] = tensor_image

            # process the batch
            # robot state is normalized using norm stats stored in the policy checkpoint (computed during training)
            # normalization is done by default as an input transform in the policy forward pass
            if policy.config.image_features:
                batch = dict(batch)
                batch["observation.images"] = [
                    batch[key]
                    for key in policy.config.image_features
                ]
            actions = policy.infer(batch)[0][:, : policy.config.n_action_steps]  # type: ignore
            actions = policy.unnormalize_outputs({"action": actions})["action"]  # type: ignore
            actions = actions.transpose(0, 1)
            return actions.cpu().numpy()

    @background_task_log_exceptions
    async def control_loop(
        self,
        control_signal: AIControlSignal,
        robots: list[BaseManipulator],
        model_spawn_config: Pi0SpawnConfig,
        all_cameras: AllCameras,
        fps: int = 30,
        speed: float = 1.0,
        cameras_keys_mapping: Dict[str, int] | None = None,
        prompt: str | None = None,
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
                        angles=action_list[robot_index * 6: robot_index * 6 + 6],
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


def make_so100_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "observation/state": np.random.rand(12),
        "observation/images.main.left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/images.secondary_0": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/images.secondary_1": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class S0100Inputs(_transforms.DataTransformFn):
    # The action dimension of the model. Will be used to pad state and actions for pi0 model (not pi0-FAST).
    action_dim: int

    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0

    @staticmethod
    def _parse_image(image) -> np.ndarray:
        image = np.asarray(image)
        if np.issubdtype(image.dtype, np.floating):
            image = (255 * image).astype(np.uint8)
        if image.shape[0] == 3:
            image = rearrange(image, "c h w -> h w c")
        return image

    def __call__(self, data: dict) -> dict:
        mask_padding = self.model_type == _model.ModelType.PI0  # We don't mask for pi0-FAST.

        # Get the state. We are padding from 8 to the model action dim.
        # For pi0-FAST, we don't pad the state (action_dim = 7, which is < 8, so pad is skipped).
        # For the SO100 the state is of size 6 so we pad
        state = _transforms.pad_to_dim(data["observation/state"], self.action_dim)

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        base_image = self._parse_image(data["observation/images.main.left"])
        wrist_image_right = self._parse_image(data["observation/images.secondary_0"])
        wrist_image_left = self._parse_image(data["observation/images.secondary_1"])

        images = {
            "base_0_rgb": base_image,
            "left_wrist_0_rgb": wrist_image_right,
            "right_wrist_0_rgb": wrist_image_left,
        }
        image_masks = {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.True_,
        }

        inputs = {
            "state": state,
            "image": images,
            "image_mask": image_masks,
        }

        # Actions are only available during training.
        if "actions" in data:
            actions = _transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class S0100Outputs(_transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Make sure to only return the appropriate number of actions here
        # 6 for 1 robot, 12 for 2
        return {"actions": np.asarray(data["actions"][:, :12])}


@dataclasses.dataclass(frozen=True)
class LeRobotSO100DataConfig(_config.DataConfigFactory):
    @override
    def create(self, assets_dirs: Path, model_config: _model.BaseModelConfig) -> _config.DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        # We only take the left cam of the stereo cam
                        "observation/images.main.left": "observation.images.main.left",
                        "observation/images.secondary_0": "observation.images.secondary_0",
                        "observation/images.secondary_1": "observation.images.secondary_1",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # Prepare data for policy training
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[S0100Inputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[S0100Outputs()],
        )
        # Use delta actions (not for gripper)
        delta_action_mask = _transforms.make_bool_mask(6, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = _config.ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


_PHOSPHO_TRAIN_CONFIGS = [
    _config.TrainConfig(
        name="pi0_so100",
        model=pi0.Pi0Config(),
        data=LeRobotSO100DataConfig(
            repo_id="LegrandFrederic/dual-setup",
            base_config=_config.DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
                action_sequence_keys=("action",),
            ),
        ),
        weight_loader=CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
    ),
    _config.TrainConfig(
        name="pi0_fast_so100",
        model=pi0_fast.Pi0FASTConfig(action_dim=12, action_horizon=10),
        data=LeRobotSO100DataConfig(
            repo_id="LegrandFrederic/dual-setup",
            base_config=_config.DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
                action_sequence_keys=("action",),
            ),
        ),
        weight_loader=CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=20_000,
    ),
    _config.TrainConfig(
        name="pi0_fast_so100_low_mem_finetune",
        model=pi0_fast.Pi0FASTConfig(paligemma_variant="gemma_2b_lora"),
        data=LeRobotSO100DataConfig(
            repo_id="PLB/Orange-brick-in-black-box",
            base_config=_config.DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
                action_sequence_keys=("action",),
            ),
        ),
        weight_loader=CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=6, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),
]


if len({config.name for config in _PHOSPHO_TRAIN_CONFIGS}) != len(_PHOSPHO_TRAIN_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _PHOSPHO_TRAIN_CONFIGS}


def get_config(config_name: str) -> _config.TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")
    return _CONFIGS_DICT[config_name]


def create_policy(
    model_path: str,
    policy_config_name: str | None = None,
) -> _policy.Policy:
    """Create a policy from a config"""
    if not policy_config_name:
        # set default policy_config_name (for immediate testing)
        # TODO: move to Pi0SpawnConfig or user query
        policy_config_name = "pi0_so100"

    config = _config.get_config(policy_config_name)
    return _policy_config.create_trained_policy(
        config,
        model_path,
        default_prompt=None
    )


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
