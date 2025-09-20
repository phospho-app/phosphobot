#!/usr/bin/env python3
"""
Script to test the SmolVLA server on Modal.
This script first spawns the server using the "spawn" endpoint
and then sends inference requests to the /act endpoint.
"""

import numpy as np
import time
import asyncio
import json_numpy
from pydantic import BaseModel, ConfigDict
import httpx
import sys
import modal
from pathlib import Path

phosphobot_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(phosphobot_dir))  # Add phosphobot to sys.path
from phosphobot.am.smolvla import HuggingFaceModelValidator, HuggingFaceAugmentedValidator, SmolVLASpawnConfig, SmolVLA

sys.exit(0)
# Adapted from https://huggingface.co/lerobot/smolvla_base/blob/main/config.json
_CONFIG = """
{
    "type": "smolvla",
    "n_obs_steps": 1,
    "normalization_mapping": {
        "VISUAL": "IDENTITY",
        "STATE": "MEAN_STD",
        "ACTION": "MEAN_STD"
    },
    "input_features": {
        "observation.state": {
            "type": "STATE",
            "shape": [
                6
            ]
        },
        "observation.image2": {
            "type": "VISUAL",
            "shape": [
                3,
                256,
                256
            ]
        },
        "observation.image": {
            "type": "VISUAL",
            "shape": [
                3,
                256,
                256
            ]
        },
        "observation.image3": {
            "type": "VISUAL",
            "shape": [
                3,
                256,
                256
            ]
        }
    },
    "output_features": {
        "action": {
            "type": "ACTION",
            "shape": [
                6
            ]
        }
    },
    "chunk_size": 50,
    "n_action_steps": 50,
    "max_state_dim": 32,
    "max_action_dim": 32,
    "resize_imgs_with_padding": [
        512,
        512
    ],
    "empty_cameras": 0,
    "adapt_to_pi_aloha": false,
    "use_delta_joint_actions_aloha": false,
    "tokenizer_max_length": 48,
    "num_steps": 10,
    "use_cache": true,
    "freeze_vision_encoder": true,
    "train_expert_only": true,
    "train_state_proj": true,
    "optimizer_lr": 0.0001,
    "optimizer_betas": [
        0.9,
        0.95
    ],
    "optimizer_eps": 1e-08,
    "optimizer_weight_decay": 1e-10,
    "optimizer_grad_clip_norm": 10,
    "scheduler_warmup_steps": 1000,
    "scheduler_decay_steps": 30000,
    "scheduler_decay_lr": 2.5e-06,
    "vlm_model_name": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    "load_vlm_weights": true,
    "attention_mode": "cross_attn",
    "prefix_length": 0,
    "pad_language_to": "max_length",
    "num_expert_layers": 0,
    "num_vlm_layers": 16,
    "self_attn_every_n_layers": 2,
    "expert_width_multiplier": 0.75
}
"""


# Ref: /phoshpobot/am/pi0.py:Pi0.fetch_and_verify_config()
def create_smolvla_spawn_config(config_content):
    """Create the SmolVLASpawnConfig structure from sample config"""
    hf_model_config = HuggingFaceModelValidator.model_validate_json(
        config_content
    )
    hf_augmented_config = HuggingFaceAugmentedValidator(
        **hf_model_config.model_dump(),
        checkpoints=[],
    )

    state_key: str = hf_augmented_config.input_features.state_key
    state_size: list[int] = hf_augmented_config.input_features.features[state_key].shape
    video_keys: list[str] = hf_augmented_config.input_features.video_keys
    video_size: list[int] = (
        hf_augmented_config.input_features.features[video_keys[0]].shape
        if len(video_keys) > 0
        else [3, 224, 224]  # default video resolution
    )
    return SmolVLASpawnConfig(
        state_key=state_key,
        state_size=state_size,
        video_keys=video_keys,
        video_size=video_size,
        hf_model_config=hf_augmented_config,
    )


# Ref: /modal/admin.app.py
class ServerInfo(BaseModel):
    model_config = ConfigDict(extra="ignore")

    server_id: int
    url: str
    port: int
    tcp_socket: tuple[str, int]
    model_id: str
    timeout: int
    modal_function_call_id: str


# Ref: /modal/admin.app.py
def spawn_server_for_model(model_specifics, model_id='lerobot/smolvla_base', server_id=1, checkpoint=None):
    """Spawn the SmolVLA inference server using the admin API"""
    print(f"Spawning SmolVLA server for model {model_id}...")

    with modal.Queue.ephemeral() as q:  # type: ignore
        try:
            serve = modal.Function.from_name("smolvla-server", "serve")  # type: ignore
            spawn_response = serve.spawn(
                model_id=model_id,
                checkpoint=checkpoint,
                server_id=server_id,
                timeout=600,    # 10 minutes for testing
                model_specifics=model_specifics,
                q=q,
            )
        except modal.exception.NotFoundError as e:  # type: ignore
            raise Exception(f"Modal app 'smolvla-server' not found. Make sure the app is deployed.")

        # Get the tunnel information from the queue
        result: dict | None = q.get()
        if result is None:
            print("No tunnel info received from queue")
            raise Exception("Failed to start server, no tunnel info received")

    server_info = ServerInfo(
        modal_function_call_id=spawn_response.object_id, **result
    )

    print(f"Server started:\n{server_info.model_dump_json(indent=4)}")

    model = SmolVLA(
        server_url=server_info.url,
        server_port=server_info.port,
        **model_specifics.model_dump(),
    )

    return model, server_info


def create_client(server_url: str = "") -> httpx.AsyncClient:
    client = httpx.AsyncClient(
        base_url=server_url,
        timeout=120,
        headers={"Content-Type": "application/json"},
        limits=httpx.Limits(max_keepalive_connections=10, max_connections=100),
        http2=True,
    )
    return client


def make_inputs(model_specifics: SmolVLASpawnConfig) -> dict:
    # (C, W, H) -> (H, W, C)
    video_resolution = (model_specifics.video_size[2], model_specifics.video_size[1], model_specifics.video_size[0])
    inputs = {
        model_specifics.state_key: np.random.randn(*model_specifics.state_size).astype(np.float32),
        **{
            video_key: np.random.randint(0, 255, video_resolution, dtype=np.uint8)
            for video_key in model_specifics.video_keys
        }
    }
    inputs['prompt'] = "Move to the right"  # prompt is required for Pi0 inference
    print("Input keys:", list(inputs.keys()))
    encoded_payload = {"encoded": json_numpy.dumps(inputs)}
    return encoded_payload


async def get_health(client: httpx.AsyncClient) -> httpx.Response:
    response = await client.get("/health")
    print(response.json())
    return response


async def get_actions(client: httpx.AsyncClient, encoded_payload: dict) -> np.ndarray:
    response = await client.post(f"/act", json=encoded_payload, timeout=300)
    if response.status_code != 200:
        print(f"Error: {response.status_code}, {response.text}")
        return
    actions = json_numpy.loads(response.json())
    print(f"Received actions: shape: {actions.shape}.\nActions: {actions}")
    return actions


def main():
    # Create SmolVLASpawnConfig for the model
    model_specifics = create_smolvla_spawn_config(_CONFIG)

    # Step 1: Start inference on Modal
    model, server_info = spawn_server_for_model(
        model_specifics=model_specifics
    )

    if not server_info:
        print("Failed to spawn server. Exiting.")
        return

    print("Waiting for server to initialize...")
    time.sleep(30)

    # Step 2: Create inputs and run inference
    client = create_client(server_info.url)
    encoded_payload = make_inputs(model_specifics)
    asyncio.run(get_actions(client, encoded_payload))


if __name__ == "__main__":
    main()
