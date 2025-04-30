from phosphobot.am.base import ActionModel
from typing import List
import numpy as np

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
