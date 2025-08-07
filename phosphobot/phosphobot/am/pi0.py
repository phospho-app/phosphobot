import asyncio
import json
import os
import time
import traceback
import websockets
import websockets.sync.client
import signal
import http
import enum
from abc import ABC
from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Tuple

import numpy as np
from fastapi import HTTPException
from huggingface_hub import HfApi
from loguru import logger
from pydantic import BaseModel, Field

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


@dataclass
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
                # TODO: Check if authentication is required
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


class Pi0Config(BaseModel):
    """Configuration for Pi0 models."""

    camera_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Camera configuration including resolution and fps"
    )
    action_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Action space configuration"
    )
    env_mode: Pi0EnvMode = Field(
        default=Pi0EnvMode.ALOHA_SIM,
        description="Environment mode (aloha, aloha_sim, droid, libero)"
    )
    prompt: str | None = Field(
        default=None,
        description="Default prompt for the model"
    )
    record: bool = Field(
        default=False,
        description="Whether to record the policy behavior"
    )
    checkpoint_config: str | None = Field(
        default=None,
        description="Training config name"
    )
    checkpoint_dir: str | None = Field(
        default=None,
        description="Checkpoint directory"
    )


class Pi0SpawnConfig(BaseModel):
    """Configuration for spawning a Pi0 model server"""
    host: str = "localhost"
    port: int = 8000
    config: Pi0Config


class Pi0(ActionModel):
    def __init__(
        self,
        server_url: str = "localhost",
        server_port: int = 8000,
        env_mode: str = "aloha_sim",
        default_prompt: str | None = None,
        action_keys: list[str] = ["action"],
        **kwargs,
    ):
        super().__init__(server_url, server_port)
        self.client = Pi0WebSocketClient(host=server_url, port=server_port)
        self.env_mode = env_mode
        self.default_prompt = default_prompt
        self.action_keys = action_keys

    def sample_actions(self, inputs: dict) -> np.ndarray:
        """Sample actions from the Pi0 model via WebSocket."""
        raise NotImplementedError("Pi0 sample_actions method is not implemented yet.")

    @classmethod
    def fetch_config(cls, model_id: str) -> Pi0Config:
        """Fetch Pi0 model configuration."""
        raise NotImplementedError("fetch_config method is not implemented yet.")

    @classmethod
    def fetch_spawn_config(cls, model_id: str) -> Pi0SpawnConfig:
        """Fetch spawn configuration for Pi0 model."""
        raise NotImplementedError("fetch_spawn_config method is not implemented yet.")

    @classmethod
    def fetch_and_get_configuration(cls, model_id: str) -> ModelConfigurationResponse:
        """Fetch model config and return video keys."""
        raise NotImplementedError("fetch_and_get_configuration method is not implemented yet.")

    @classmethod
    def fetch_and_verify_config(
        cls,
        model_id: str,
        all_cameras: AllCameras,
        robots: list[BaseManipulator],
        cameras_keys_mapping: Dict[str, int] | None = None,
        verify_cameras: bool = True,
    ) -> Pi0SpawnConfig:
        """Verify Pi0 model compatibility with current setup."""
        raise NotImplementedError("fetch_and_verify_config method is not implemented yet.")

    @background_task_log_exceptions
    async def control_loop(
        self,
        control_signal: AIControlSignal,
        robots: list[BaseManipulator],
        model_spawn_config: Pi0SpawnConfig,
        all_cameras: AllCameras,
        prompt: str | None = None,
        fps: int = 30,
        speed: float = 1.0,
        cameras_keys_mapping: Dict[str, int] | None = None,
        angle_format: Literal["degrees", "radians", "other"] = "radians",
        min_angle: float | None = None,
        max_angle: float | None = None,
        **kwargs: Any,
    ):
        """
        AI control loop that runs in the background and sends actions to the robot.
        It uses the model to get the actions based on the current state of the robot and the cameras.
        The loop runs until the control signal is stopped or the model is not available anymore.
        The loop runs at the specified fps and speed.
        """
        raise NotImplementedError("Pi0 control_loop method is not implemented yet.")


class Pi0TrainerConfig(BaseTrainerConfig):
    """Pi0 trainer configuration."""
    model_type = "pi0"
    training_params: TrainingParamsPi0 | None = None


class Pi0Trainer(BaseTrainer):
    """Pi0 model trainer."""

    def __init__(self, config: Pi0TrainerConfig):
        self.config = config

    def train(self, timeout_seconds: int | None = None) -> None:
        """Train a Pi0 model."""
        logger.info(f"Starting Pi0 training for dataset={self.config.dataset_name}")
        raise NotImplementedError("Pi0 training not supported yet!")
