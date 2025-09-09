import os
from pathlib import Path
from pydantic import BaseModel
from supabase import Client
from loguru import logger
from datetime import datetime, timezone
from huggingface_hub import HfApi, snapshot_download


class InferenceRequest(BaseModel):
    encoded: str  # Will contain json_numpy encoded payload with image


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


def find_model_path(model_id: str, local_base_dir: str = "/data", checkpoint: int | None = None) -> str | None:
    """
    Find the path to the model stored locally

    Args:
        model_id: The model identifier from HuggingFace or local path
        local_base_dir: local base directory where the model is saved
        checkpoint: model checkpoint to load

    Returns:
        model path if saved checkpoint is found, None otherwise.
    """
    model_path = Path(f"{local_base_dir}/{model_id}")
    if checkpoint is not None:
        # format the checkpoint to be 6 digits long
        model_path = model_path / "checkpoints" / str(checkpoint) / "pretrained_model"
        if model_path.exists():
            return str(model_path.resolve())

    # get the latest checkpoint
    model_path = model_path / "checkpoints" / "last" / "pretrained_model"
    if model_path.exists():
        return str(model_path.resolve())

    return None  # no checkpoints found


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


def get_model_path(
    model_id: str,
    checkpoint: int | None = None,
    local_base_dir: str = "/data",
    token: str | None = None,
) -> str:
    """
    Downloads a model from HuggingFace if not already present in the local directory.

    Args:
        model_id: The HuggingFace model ID to download
        checkpoint: [Optional] specific checkpoint/revision to download
        local_base_dir: Base directory to store downloaded models
        token: [Optional] HuggingFace token for accessing private models

    Returns:
        The local path to the downloaded model

    Raises:
        Exception: If the download fails
    """
    model_path = find_model_path(model_id=model_id, checkpoint=checkpoint)

    # Check if model is already available locally
    if model_path is not None:
        logger.info(
            f"🤗 Model {model_id} found in Modal volume. Will be used for inference."
        )
        return model_path
    else:
        logger.warning(
            f"🤗 Model {model_id} not found in Modal volume. Will be downloaded from HuggingFace."
        )
        try:
            if checkpoint:
                model_path = snapshot_download(
                    repo_id=model_id,
                    repo_type="model",
                    revision=str(checkpoint),
                    local_dir=f"{local_base_dir}/{model_id}/checkpoints/{str(checkpoint)}/pretrained_model",
                    token=token or os.getenv("HF_TOKEN"),
                )
            else:
                model_path = snapshot_download(
                    repo_id=model_id,
                    repo_type="model",
                    revision="main",
                    local_dir=f"{local_base_dir}/{model_id}/checkpoints/last/pretrained_model",
                    ignore_patterns=["checkpoint-*"],
                    token=token or os.getenv("HF_TOKEN"),
                )
            return model_path
        except Exception as e:
            logger.error(f"Failed to download model {model_id} with checkpoint {checkpoint}: {e}")
            raise e
