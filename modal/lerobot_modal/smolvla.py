from typing import Any

import modal
import numpy as np
import torch

from phosphobot.am.base import TrainingParamsSmolVLA
from phosphobot.am.smolvla import SmolVLASpawnConfig
from .helper import validate_inputs, prepare_base_batch
from .app import (
    MINUTES,
    base_image,
    FUNCTION_GPU_INFERENCE,
    FUNCTION_TIMEOUT_INFERENCE,
    FUNCTION_GPU_TRAINING,
    FUNCTION_TIMEOUT_TRAINING,
    FUNCTION_CPU_TRAINING,
    phosphobot_dir,
    serve_policy,
    train_policy,
)

# SmolVLA image
smolvla_image = (
    base_image.uv_pip_install(
        "lerobot[smolvla]==0.3.3",  # before introduction of LeRobotDataset v3.0
    )
    .pip_install_from_pyproject(pyproject_toml=str(phosphobot_dir / "pyproject.toml"))
    .add_local_python_source("phosphobot")
)

smolvla_app = modal.App("smolvla-server")
smolvla_volume = modal.Volume.from_name("smolvla", create_if_missing=True)


def process_smolvla_inference(
    policy: Any,
    model_specifics: SmolVLASpawnConfig,
    current_qpos: list[float],
    images: list[np.ndarray],
    image_names: list[str],
    target_size: tuple[int, int],
    prompt: str,
    prompt_key: str = "task",
) -> np.ndarray:
    """
    Process images and perform inference using the SmolVLA policy.

    Args:
        policy: lerobot.policies.smolvla.modeling_smolvla.SmolVLAPolicy object
        model_specifics: SmolVLASpawnConfig object
        current_qpos: Current robot state
        images: List of images
        image_names: List of image names corresponding to the images
        target_size: Target size for resizing images (height, width)
        prompt: Text instruction for the model
        prompt_key: prompt key used for policy inference

    Returns:
        np.ndarray: predicted action chunk

    Raises:
        AssertionError: If input validations fail.
        ValueError: If prompt is None.
    """
    # Validate inputs using common function
    validate_inputs(current_qpos, images, model_specifics, target_size)

    if prompt is None:
        raise ValueError("Prompt cannot be None for SmolVLA inference")

    with torch.no_grad(), torch.autocast(device_type="cuda"):
        # Prepare base observation using common function
        batch = prepare_base_batch(
            current_qpos, images, image_names, model_specifics, target_size
        )
        batch[prompt_key] = prompt  # prompt is required for SmolVLA

        # Run inference
        actions = policy.predict_action_chunk(batch)

        return actions.cpu().numpy()


# ======== SmolVLA ========
@smolvla_app.function(
    image=smolvla_image,
    gpu=FUNCTION_GPU_INFERENCE,
    timeout=FUNCTION_TIMEOUT_INFERENCE,
    secrets=[
        modal.Secret.from_dict({"MODAL_LOGLEVEL": "DEBUG"}),
        modal.Secret.from_name("supabase"),
    ],
    volumes={"/data": smolvla_volume},
)
async def serve(
    model_id: str,
    server_id: int,
    model_specifics: SmolVLASpawnConfig,
    checkpoint: int | None = None,
    timeout: int = FUNCTION_TIMEOUT_INFERENCE,
    q=None,
):
    """SmolVLA model serving function."""
    await serve_policy(
        model_id=model_id,
        server_id=server_id,
        model_specifics=model_specifics,
        checkpoint=checkpoint,
        timeout=timeout,
        q=q,
    )


@smolvla_app.function(
    image=smolvla_image,
    gpu=FUNCTION_GPU_TRAINING,
    timeout=FUNCTION_TIMEOUT_TRAINING + 20 * MINUTES,
    secrets=[
        modal.Secret.from_dict({"MODAL_LOGLEVEL": "DEBUG"}),
        modal.Secret.from_name("supabase"),
        modal.Secret.from_name("huggingface"),
    ],
    volumes={"/data": smolvla_volume},
    cpu=FUNCTION_CPU_TRAINING,
)
def train(
    training_id: int,
    dataset_name: str,
    wandb_api_key: str | None,
    model_name: str,
    training_params: TrainingParamsSmolVLA,
    user_hf_token: str | None = None,
    private_mode: bool = False,
    max_hf_download_retries: int = 3,
    timeout_seconds: int = FUNCTION_TIMEOUT_TRAINING,
    wandb_run_id: str = "wandb_run_id_not_set",
    **kwargs,
):
    """SmolVLA training function."""
    # remove model_type from kwargs if present
    # to avoid passing model_type twice in the `train_policy()` function call
    if "model_type" in kwargs:
        del kwargs["model_type"]

    train_policy(
        model_type="smolvla",
        training_id=training_id,
        dataset_name=dataset_name,
        wandb_api_key=wandb_api_key,
        model_name=model_name,
        training_params=training_params,
        user_hf_token=user_hf_token,
        private_mode=private_mode,
        max_hf_download_retries=max_hf_download_retries,
        timeout_seconds=timeout_seconds,
        wandb_run_id=wandb_run_id,
        **kwargs,
    )
