# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "lerobot",
#     "phosphobot",
#     "json_numpy",
#     "fastapi",
#     "uvicorn",
#     "packaging",
#     "torch",
#     "loguru",
# ]
#
# [tool.uv.sources]
# lerobot = { git = "https://github.com/phospho-app/lerobot" }
# ///

from loguru import logger

logger.info("Starting ACT policy server...")

import argparse
from pathlib import Path
from typing import List
import json
import cv2
import json_numpy
import numpy as np
import torch
import torch.nn as nn
import uvicorn
from packaging import version  # Don't remove this line (used by lerobot)
from fastapi import FastAPI, HTTPException
from huggingface_hub import snapshot_download
from huggingface_hub.errors import RepositoryNotFoundError
from huggingface_hub.utils._validators import HFValidationError
from lerobot.policies.act.modeling_act import ACTPolicy
from pydantic import BaseModel

app = FastAPI()

# Global variables
policy: ACTPolicy = None
input_features: dict = {}
device = None


class InferenceRequest(BaseModel):
    encoded: str  # Will contain json_numpy encoded payload with image


def get_safe_torch_device(device_str: str, log: bool = True) -> torch.device:
    """Get a safe torch device, defaulting to CPU if requested device is not available."""
    if device_str == "cuda" and not torch.cuda.is_available():
        if log:
            logger.warning("CUDA requested but not available. Using CPU instead.")
        return torch.device("cpu")
    elif device_str == "mps" and not torch.backends.mps.is_available():
        if log:
            logger.warning("MPS requested but not available. Using CPU instead.")
        return torch.device("cpu")
    return torch.device(device_str)


def get_pretrained_policy_path(pretrained_policy_name_or_path, revision=None):
    """Get the path to the pretrained policy, either from HF Hub or local."""
    try:
        pretrained_policy_path = Path(
            snapshot_download(pretrained_policy_name_or_path, revision=revision)
        )
    except (HFValidationError, RepositoryNotFoundError) as e:
        if isinstance(e, HFValidationError):
            error_message = "The provided pretrained_policy_name_or_path is not a valid Hugging Face Hub repo ID."
        else:
            error_message = "The provided pretrained_policy_name_or_path was not found on the Hugging Face Hub."
        logger.warning(f"{error_message} Treating it as a local directory.")
        pretrained_policy_path = Path(pretrained_policy_name_or_path)

    if not pretrained_policy_path.is_dir() or not pretrained_policy_path.exists():
        raise ValueError(
            "The provided pretrained_policy_name_or_path is not a valid/existing Hugging Face Hub "
            "repo ID, nor is it an existing local directory."
        )
    return pretrained_policy_path


def parse_input_features(file_path: Path) -> dict:
    """
    Parse JSON data from a file path and return the input_features dictionary.

    Args:
        file_path (str): Path to the JSON file

    Returns:
        dict: The input_features portion of the JSON data

    Raises:
        FileNotFoundError: If the specified file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    try:
        # Open and read the JSON file
        with open(file_path, "r") as file:
            data = json.load(file)

        # Return the input_features dictionary
        return data.get("input_features", {})

    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find JSON file at: {file_path}")


def load_policy(model_id: str, revision: str | None = None):
    """Download and load the ACT policy."""
    global policy, device, input_features
    try:
        logger.info(f"Loading policy from {model_id}")

        # Get the policy path
        policy_path = get_pretrained_policy_path(model_id, revision=revision)

        # Set up device
        device = get_safe_torch_device(
            device_str="mps", log=True
        )  # "mps" on mac / "cuda" if you have a GPU / "cpu" otherwise
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        logger.info(f"Device set to {device}")

        # Load the policy
        policy = ACTPolicy.from_pretrained(policy_path).to(device)
        assert isinstance(policy, nn.Module)
        logger.info("Policy loaded successfully")

        input_features = parse_input_features(policy_path / "config.json")
        logger.success(f"Input features required for model: {input_features.keys()}")

        # Set to evaluation mode
        policy.eval()
        logger.debug("Policy set to evaluation mode")

        return input_features

    except Exception as e:
        logger.error(f"Error loading policy: {str(e)}")
        raise


def process_image(
    images: List[np.ndarray],
    current_qpos: np.ndarray,
    image_names: List[str],
    target_size: tuple[int, int],
) -> np.ndarray:
    """Process image through the ACT policy."""
    global policy, device

    if len(images) == 0:
        raise ValueError("No images provided")

    if len(images[0].shape) == 3 and images[0].shape[2] == 3:
        try:
            if device is None:
                raise ValueError(
                    "Device is not set. Please ensure the policy is loaded correctly."
                )
            with torch.no_grad(), torch.autocast(device_type=device.type):
                # Prepare state tensor
                current_qpos = current_qpos.copy()
                state_tensor = (
                    torch.from_numpy(current_qpos)
                    .view(1, len(current_qpos))
                    .float()
                    .to(device)
                )

                # Create batch dictionary
                batch = {
                    "observation.state": state_tensor,
                }

                # Convert images to tensors (B, C, H, W), normalize
                processed_images = []
                for i, image in enumerate(images):
                    if image.shape[:2] != target_size:
                        image = cv2.resize(image, target_size)
                    tensor_image = (
                        torch.from_numpy(image)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                        .float()
                        .to(device)
                    )
                    tensor_image = tensor_image / 255.0
                    processed_images.append(tensor_image)
                    batch[image_names[i]] = tensor_image

                # Get the actions
                batch = policy.normalize_inputs(batch)  # type: ignore
                if policy.config.image_features:  # type: ignore
                    batch = dict(batch)
                    batch["observation.images"] = [
                        batch[key]
                        for key in policy.config.image_features  # type: ignore
                    ]
                actions = policy.model(batch)[0][:, : policy.config.n_action_steps]  # type: ignore
                actions = policy.unnormalize_outputs({"action": actions})["action"]  # type: ignore
                actions = actions.transpose(0, 1)
                return actions.cpu().numpy()

        except Exception as e:
            logger.error(f"Error during inference: {str(e)}")
            raise
    else:
        raise ValueError("Invalid image format. Expected RGB image.")


@app.post("/act")
async def inference(request: InferenceRequest) -> str | None:
    """Endpoint for ACT policy inference."""
    if policy is None:
        raise HTTPException(status_code=500, detail="Policy not initialized")

    try:
        # Decode the double-encoded payload
        payload: dict = json_numpy.loads(request.encoded)
        target_size: tuple[int, int] = (224, 224)

        # Get feature names
        image_names = [
            feature for feature in input_features.keys() if "image" in feature
        ]

        if "observation.state" not in payload:
            logger.error("observation.state not found in payload")
            raise ValueError("observation.state required in payload")

        if len(payload.keys()) != len(input_features.keys()):
            for feature in input_features.keys():
                if feature not in payload:
                    logger.error(f"{feature} required but not found in payload")
            raise ValueError("Missing required features in payload")
        else:
            shape = input_features[image_names[0]]["shape"]
            target_size = (shape[2], shape[1])

        # Infer actions
        actions = process_image(
            current_qpos=payload["observation.state"],
            images=[
                payload[f"observation.images.{i}"]
                for i in range(len(payload.keys()))
                if f"observation.images.{i}" in payload
            ],
            image_names=image_names,
            target_size=target_size,
        )

        # Encode response using json_numpy
        response = json_numpy.dumps(actions)
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Endpoint to check if the policy is loaded and ready."""
    return {
        "status": "healthy" if policy is not None else "not_ready",
        "policy_loaded": policy is not None,
        "device": str(device) if device is not None else None,
        "input_features": input_features if input_features != {} else "not_loaded",
    }


def main():
    parser = argparse.ArgumentParser(description="Deploy ACT policy for inference")
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Hugging Face model ID or local path",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Hugging Face model revision. Default: None",
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port to run the server on"
    )

    args = parser.parse_args()
    # Load the policy
    load_policy(args.model_id, revision=args.revision)

    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
