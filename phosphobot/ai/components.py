from typing import List
import numpy as np
import torch
import logging
import cv2

# To clean
import argparse
import logging
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
from lerobot.common.policies.act.modeling_act import ACTPolicy
from pydantic import BaseModel


class Component:
    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement the forward method")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


### UTILS ###


def get_safe_torch_device(device_str: str, log: bool = True) -> torch.device:
    """Get a safe torch device, defaulting to CPU if requested device is not available."""
    if device_str == "cuda" and not torch.cuda.is_available():
        if log:
            logging.warning("CUDA requested but not available. Using CPU instead.")
        return torch.device("cpu")
    elif device_str == "mps" and not torch.backends.mps.is_available():
        if log:
            logging.warning("MPS requested but not available. Using CPU instead.")
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
        logging.warning(f"{error_message} Treating it as a local directory.")
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


class ACT(Component):
    def __init__(self, model_id: str, revision: str | None = None):
        # self.input_size = input_size
        self.policy = None

        policy_path = get_pretrained_policy_path(model_id, revision=revision)

        # Set up device
        self.device = get_safe_torch_device(device_str="mps", log=True)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        logging.info(f"Device set to {self.device}")

        # Load the policy
        policy = ACTPolicy.from_pretrained(policy_path).to(self.device)
        assert isinstance(policy, nn.Module)
        logging.info("Policy loaded successfully")

        input_features = parse_input_features(policy_path / "config.json")
        logging.info(f"Input features required for model: {input_features.keys()}")

    def forward(
        self,
        images: List[np.ndarray],
        current_qpos: np.ndarray,
        image_names: List[str],
        target_size: tuple[int, int],
    ) -> np.ndarray:
        """
        TODO: WIP of inference server adaptation
        Process image through the ACT policy.
        """

        if len(images) == 0:
            raise ValueError("No images provided")

        if len(images[0].shape) == 3 and images[0].shape[2] == 3:
            try:
                if self.device is None:
                    raise ValueError(
                        "Device is not set. Please ensure the policy is loaded correctly."
                    )
                with torch.no_grad(), torch.autocast(device_type=self.device.type):
                    # Prepare state tensor
                    current_qpos = current_qpos.copy()
                    state_tensor = (
                        torch.from_numpy(current_qpos)
                        .view(1, len(current_qpos))
                        .float()
                        .to(self.device)
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
                            .to(self.device)
                        )
                        tensor_image = tensor_image / 255.0
                        processed_images.append(tensor_image)
                        batch[image_names[i]] = tensor_image

                    # Get action using select_action
                    action = self.policy.select_action(batch)
                    return action.cpu().numpy()

            except Exception as e:
                logging.error(f"Error during inference: {str(e)}")
                raise
        else:
            raise ValueError("Invalid image format. Expected RGB image.")
