import requests
import numpy as np
from typing import List
import json_numpy  # type: ignore
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Callable, Dict

import zmq
import pickle


"""
SERVER URL or IP
SERVER port
TODO: api_key

Images
State
Instruction (optional)
"""


class ActionModel:
    """
    A PyTorch model for generating robot actions from robot state, camera images, and text prompts.
    Inspired by the simplicity and flexibility of pytorch-pretrained-bert.
    """

    def __init__(self, server_url: str = "http://localhost", server_port: int = 8080):
        """
        Initialize the ActionModel.

        Args:
            model_name (str): Name of the pre-trained model (e.g., "PLB/pi0-so100-orangelegobrick-wristcam").
            revision: default None which will resolve to main
        """
        self.server_url = server_url
        self.server_port = server_port

    def sample_actions(self, inputs: dict) -> np.ndarray:
        """
        Select a single action.

        Args:
            inputs (dict): Dictionary with keys:
                - "state": Tensor or list of floats representing robot state.
                - "images": List of images (numpy arrays or tensors).
                - "prompt": String text prompt (optional for ACT).

        Returns:
            np.ndarray: Sequence of actions (shape: [max_seq_length, n_actions]).
        """
        raise NotImplementedError("""You cannot directly call the ActionModel class. 
                                  You need to use an implementation ( ACT, PI0,...) or implement you own class.""")

    def __call__(self, *args, **kwargs):
        """
        Makes the model instance callable, delegating to the forward method.

        Args:
            *args: Variable positional arguments passed to forward.
            **kwargs: Variable keyword arguments passed to forward.

        Returns:
            The output of the forward method.
        """
        return self.sample_actions(*args, **kwargs)


class ACT(ActionModel):
    def __init__(
        self,
        server_url: str = "http://localhost",
        server_port: int = 8080,
        timeout: int = 10,
    ):
        super().__init__(server_url, server_port)
        self.required_input_keys: List[str] = ["images", "state"]
        self.timeout = timeout

    def sample_actions(self, inputs: dict) -> np.ndarray:
        # Build the payload
        payload = {
            "observation.state": inputs["state"],
        }
        for i in range(0, len(inputs["images"])):
            payload[f"observation.images.{i}"] = inputs["images"][i]

        # Double-encoded version (to send numpy arrays as JSON)
        encoded_payload = {"encoded": json_numpy.dumps(payload)}

        response = requests.post(
            f"{self.server_url}:{self.server_port}/act",
            json=encoded_payload,
            timeout=self.timeout,
        ).json()

        action = json_numpy.loads(response)

        return np.array([action])


try:
    from openpi_client import websocket_client_policy  # type: ignore

    class Pi0(ActionModel):
        def __init__(
            self,
            server_url: str = "http://localhost",
            server_port: int = 8080,
            image_keys=["observation/image", "observation/wrist_image"],
        ):
            super().__init__(server_url, server_port)
            self.required_input_keys: List[str] = ["images", "state", "prompt"]
            self.image_keys = image_keys

            # Instantiate the client
            self.client = websocket_client_policy.WebsocketClientPolicy(
                host=self.server_url,
                port=self.server_port,
            )

        def sample_actions(self, inputs: dict) -> np.ndarray:
            observation = {
                "observation/state": inputs["state"],
                "prompt": inputs["prompt"],
            }

            for i in range(0, len(self.image_keys)):
                observation[self.image_keys[i]] = inputs["images"][0]

            # Call the remote server
            action_chunk = self.client.infer(observation)["actions"]

            # TODO: check action_chunk is of type np.ndarray
            return action_chunk

except ImportError:

    class Pi0(ActionModel):  # type: ignore
        def __init__(self, server_url: str = "localhost", server_port: int = 8080):
            raise NotImplementedError(
                "Pi0 model requires openpi_client package: https://github.com/phospho-app/openpi.git"
            )


# Code from: https://github.com/NVIDIA/Isaac-GR00T/blob/main/gr00t/eval/service.py#L111


class TorchSerializer:
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

    def run(self):
        addr = self.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        print(f"Server is ready and listening on {addr}")
        while self.running:
            try:
                message = self.socket.recv()
                request = TorchSerializer.from_bytes(message)
                endpoint = request.get("endpoint", "get_action")

                if endpoint not in self._endpoints:
                    raise ValueError(f"Unknown endpoint: {endpoint}")

                handler = self._endpoints[endpoint]
                result = (
                    handler.handler(request.get("data", {}))
                    if handler.requires_input
                    else handler.handler()
                )
                self.socket.send(TorchSerializer.to_bytes(result))
            except Exception as e:
                print(f"Error in server: {e}")
                import traceback

                print(traceback.format_exc())
                self.socket.send(b"ERROR")


class BaseInferenceClient:
    def __init__(
        self, host: str = "localhost", port: int = 5555, timeout_ms: int = 15000
    ):
        self.context = zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
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
        self, endpoint: str, data: dict | None = None, requires_input: bool = True
    ) -> dict:
        """
        Call an endpoint on the server.

        Args:
            endpoint: The name of the endpoint.
            data: The input data for the endpoint.
            requires_input: Whether the endpoint requires input data.
        """
        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data

        self.socket.send(TorchSerializer.to_bytes(request))
        message = self.socket.recv()
        if message == b"ERROR":
            raise RuntimeError("Server error")
        return TorchSerializer.from_bytes(message)

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


class Gr00tN1(ActionModel):
    def __init__(
        self,
        action_keys: list[str],
        server_url: str = "localhost",
        server_port: int = 5555,
    ):
        super().__init__(server_url, server_port)
        self.client = ExternalRobotInferenceClient(server_url, server_port)
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

            elif isinstance(new_action, list) and len(new_action) in [1, 6]:
                # List case: length 1 or 6, assume last element is the gripper
                if new_action[-1] < 0.35:
                    new_action[-1] = 0.0
                # Convert to NumPy array with shape (1, len(new_action))
                new_action = np.array(new_action).reshape(1, -1)
                # Repeat to match batch size of 16
                new_action = np.repeat(new_action, 16, axis=0)
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
