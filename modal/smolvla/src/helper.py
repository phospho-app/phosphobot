import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Any
from loguru import logger
from huggingface_hub import HfApi, snapshot_download
from supabase import Client
from pydantic import BaseModel
import numpy as np
import torch
import cv2

class InferenceRequest(BaseModel):
    encoded: str  # json_numpy encoded dict


def _find_model_path(model_id: str, checkpoint: int | None = None) -> str | None:
    model_path = Path(f"/data/{model_id}")
    if checkpoint is not None:
        # format the checkpoint to be 6 digits long
        model_path = model_path / "checkpoints" / str(checkpoint) / "pretrained_model"
        if model_path.exists():
            return str(model_path.resolve())
    model_path = model_path / "checkpoints" / "last" / "pretrained_model"
    if not os.path.exists(model_path):
        return None
    return str(model_path.resolve())


def _find_or_download_model(model_id: str, checkpoint: int | None = None) -> str:
    model_path = _find_model_path(model_id=model_id, checkpoint=checkpoint)

    if model_path is None:
        logger.warning(
            f"🤗 Model {model_id} not found in Modal volume. Will be downloaded from HuggingFace."
        )
        try:
            if checkpoint:
                model_path = snapshot_download(
                    repo_id=model_id,
                    repo_type="model",
                    revision=str(checkpoint),
                    local_dir=f"/data/{model_id}/checkpoints/{checkpoint}/pretrained_model",
                    token=os.getenv("HF_TOKEN"),
                )
            else:
                model_path = snapshot_download(
                    repo_id=model_id,
                    repo_type="model",
                    revision="main",
                    local_dir=f"/data/{model_id}/checkpoints/last/pretrained_model",
                    ignore_patterns=["checkpoint-*"],
                )
        except Exception as e:
            logger.error(
                f"Failed to download model {model_id} with checkpoint {checkpoint}: {e}"
            )
            raise e
    else:
        logger.info(
            f"🤗 Model {model_id} found in Modal volume. Will be used for inference."
        )
    return model_path


def _upload_partial_checkpoint(output_dir: Path, model_name: str, hf_token: str):
    """
    Upload whatever is already in output_dir/checkpoints/last/pretrained_model
    to the HF model repo, so we don't lose everything if we time out.
    """
    api = HfApi(token=hf_token)
    checkpoint_dir = output_dir / "checkpoints" / "last" / "pretrained_model"
    if not checkpoint_dir.exists():
        logger.error(f"No partial checkpoint found at {checkpoint_dir}")
        return
    for item in checkpoint_dir.glob("**/*"):
        if item.is_file():
            relpath = item.relative_to(checkpoint_dir)
            logger.info(f"Uploading partial checkpoint {relpath}")
            api.upload_file(
                repo_type="model",
                path_or_fileobj=str(item.resolve()),
                path_in_repo=str(relpath),
                repo_id=model_name,
                token=hf_token,
            )


def _update_server_status(
    supabase_client: Client,
    server_id: int,
    status: str,
):
    logger.info(f"Updating server status to {status} for server_id {server_id}")
    if status == "failed":
        server_payload = {
            "status": status,
            "terminated_at": datetime.now(timezone.utc).isoformat(),
        }
        supabase_client.table("servers").update(server_payload).eq(
            "id", server_id
        ).execute()
        # Update also the AI control session
        ai_control_payload = {
            "status": "stopped",
            "ended_at": datetime.now(timezone.utc).isoformat(),
        }
        supabase_client.table("ai_control_sessions").update(ai_control_payload).eq(
            "server_id", server_id
        ).execute()
    elif status == "stopped":
        server_payload = {
            "status": status,
            "terminated_at": datetime.now(timezone.utc).isoformat(),
        }
        supabase_client.table("servers").update(server_payload).eq(
            "id", server_id
        ).execute()
        # Update also the AI control session
        ai_control_payload = {
            "status": "stopped",
            "ended_at": datetime.now(timezone.utc).isoformat(),
        }
        supabase_client.table("ai_control_sessions").update(ai_control_payload).eq(
            "server_id", server_id
        ).execute()
    else:
        raise NotImplementedError(
            f"Status '{status}' not implemented for server update"
        )


def process_image(
    policy: Any,
    model_specifics: Any,
    current_qpos: list[float],
    images: list[np.ndarray],
    image_names: list[str],
    target_size: tuple[int, int],
    prompt: str,
    prompt_key: str = "observation.language.tokens",
    prompt_mask_key: str = "observation.language.attention_mask",
) -> np.ndarray:
    """
    Process images and perform inference using the policy.

    Args:
        policy: lerobot.policies.smolvla.modeling_smolvla.SmolVLAPolicy object
        model_specifics: SmolVLASpawnConfig object
        current_qpos: Current robot state
        images: List of images
        image_names: List of image names corresponding to the images
        target_size: Target size for resizing images (height, width)
        prompt: Text instruction for the model
        prompt_key: prompt key used for policy inference
        prompt_mask_key: prompt mask key used for policy inference
    Returns:
        np.ndarray: predicted action chunk
    Raises:
        AssertionError: If input validations fail.
        ValueError: If prompt is None.
    """
    assert len(current_qpos) == model_specifics.state_size[0], (
        f"State size mismatch: {len(current_qpos)} != {model_specifics.state_size[0]}"
    )
    assert len(images) <= len(model_specifics.video_keys), (
        f"Number of images {len(images)} is more than the number of video keys {len(model_specifics.video_keys)}"
    )
    if len(images) > 0:
        assert len(images[0].shape) == 3, (
            f"Image shape is not correct, {images[0].shape} expected (H, W, C)"
        )
        assert len(images[0].shape) == 3 and images[0].shape[2] == 3, (
            f"Image shape is not correct {images[0].shape} expected (H, W, 3)"
        )

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

        processor = policy.model.vlm_with_expert.processor
        processed_text = processor(text=prompt, return_tensors="pt")

        batch[prompt_key] = processed_text["input_ids"].to("cuda")
        batch[prompt_mask_key] = processed_text["attention_mask"].bool().to("cuda")  # convert to boolean mask

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

        actions = policy.predict_action_chunk(batch)
        # actions = actions.transpose(0, 1)  # (T, B, D) -> (B, T, D) not required
        return actions.cpu().numpy()
