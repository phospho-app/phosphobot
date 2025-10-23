import os
import wandb
import modal
import asyncio
import sentry_sdk
from pathlib import Path
from loguru import logger
from typing import Optional
from huggingface_hub import HfApi
from phosphobot.am.pi05 import Pi05SpawnConfig
from phosphobot.am.base import TrainingParamsPi05, generate_readme

if os.getenv("MODAL_ENVIRONMENT") == "production":
    sentry_sdk.init(
        dsn="https://afa38885e368d772d8eced1bce325604@o4506399435325440.ingest.us.sentry.io/4509203019005952",
        traces_sample_rate=1.0,
        environment="production",
    )

phosphobot_dir = (
    Path(__file__).parent.parent.parent.parent / "phosphobot" / "phosphobot"
)
pi0_image = (
    modal.Image.from_dockerfile("Dockerfile")
    .env(
        {
            "GIT_LFS_SKIP_SMUDGE": "1",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "_OPENPI_DATA_HOME": "/data/openpi_cache",  # This is the openpi cache dir,
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.9",  # Tell JAX to use 90% of GPU memory
        }
    )
    .uv_pip_install(
        "sentry-sdk",
        "loguru",
        "supabase",
        "huggingface_hub[hf_transfer]",
        "hf_xet",
        "httpx",
        "fastparquet",
        "opencv-python-headless",
        "json-numpy",
        "fastapi",
        "wandb",
        "uvicorn",
        "pytest",
    )
    .pip_install_from_pyproject(
        pyproject_toml=str(phosphobot_dir / "pyproject.toml"),
    )
    .workdir("/")
    .run_commands(  # clone openpi source code from last commit, we do this to be able to refresh the build when changing the repo
        "git clone https://github.com/phospho-app/openpi.git /openpi-source && cd /openpi-source && git checkout 2c7f6eef0cf2bd2d547ccdffe8e115b846fcec42"
    )
    .run_commands(
        "cd /openpi-source && uv pip install -e .",
    )
    .add_local_python_source("phosphobot")
)

# Config constants
MINUTES = 60  # seconds
HOURS = 60 * MINUTES
FUNCTION_IMAGE = pi0_image
FUNCTION_TIMEOUT_TRAINING = 3 * HOURS
FUNCTION_TIMEOUT_INFERENCE = (
    10 * MINUTES
)  # 10 minutes (includes downloading and loading policy model)
FUNCTION_GPU_TRAINING: list[str | modal.gpu._GPUConfig | None] = ["A100-80GB"]
FUNCTION_GPU_INFERENCE: list[str | modal.gpu._GPUConfig | None] = ["A100-40GB", "L40S"]

app = modal.App("pi0.5-server")
pi05_volume = modal.Volume.from_name("pi0.5", create_if_missing=True)


@app.function(
    image=FUNCTION_IMAGE,
    gpu=FUNCTION_GPU_INFERENCE,
    timeout=FUNCTION_TIMEOUT_INFERENCE,
    secrets=[
        modal.Secret.from_dict({"MODAL_LOGLEVEL": "DEBUG"}),
        modal.Secret.from_name("supabase"),
        modal.Secret.from_name("huggingface"),
    ],
    volumes={"/data": pi05_volume},
)
async def serve(
    model_id: str,
    server_id: int,
    model_specifics: Pi05SpawnConfig,
    checkpoint: int | None = None,
    timeout: int = FUNCTION_TIMEOUT_INFERENCE,
    q: Optional[modal.Queue] = None,
):
    """
    Pi0 inference server function.

    Args:
        model_id: The model identifier from HuggingFace or local path
        server_id: Database server ID for status tracking
        model_specifics: Pi0-specific configuration
        checkpoint: model checkpoint to load
        timeout: Timeout in seconds
        q: Modal queue to pass tunnel info back to caller (since the function is running in a different process)
    """
    import time
    import json
    import dataclasses

    from datetime import datetime, timezone
    from supabase import Client, create_client
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import (
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )
    from fastapi import HTTPException
    from openpi.training.config import DataConfig
    from openpi.training.config import TrainConfig
    from openpi.models.pi0_config import Pi0Config
    from openpi.training.config import LeRobotSO100DataConfig
    from openpi.policies.policy_config import create_trained_policy
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

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

    # Start timer
    start_time = time.time()

    SUPABASE_URL = os.environ["SUPABASE_URL"]
    SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    # logger.info the region
    logger.success(f"🌎 running in {os.environ['MODAL_REGION']} region")

    server_port = 5555

    with modal.forward(server_port, unencrypted=True) as tunnel:
        logger.info(f"tunnel.tcp_socket = {tunnel.tcp_socket}")

        # Send tunnel info back to caller if queue is provided
        if q is not None:
            tunnel_info = {
                "url": tunnel.url,
                "port": server_port,
                "tcp_socket": tunnel.tcp_socket,
                "model_id": model_id,
                "timeout": timeout,
                "server_id": server_id,
            }
            q.put(tunnel_info)
            logger.info(f"Tunnel info sent to queue: {tunnel_info}")

        logger.info(f"Tunnel opened after {time.time() - start_time} seconds")

        # Check if this path exists in the container
        start_time = time.time()

        try:
            local_model_path = snapshot_download(
                repo_id=model_id,
                repo_type="model",
                revision=str(checkpoint) if checkpoint is not None else None,
                cache_dir="/data/hf_cache",
            )
        except RepositoryNotFoundError as e:
            logger.error(f"Failed to download model {model_id}: {e}")
            _update_server_status(supabase_client, server_id, "failed")
            raise HTTPException(
                status_code=400,
                detail=f"Model {model_id} not found. Make sure the model is public. Error: {e}",
            )
        except RevisionNotFoundError as e:
            logger.error(
                f"Failed to download model {model_id} at revision {checkpoint}: {e}"
            )
            _update_server_status(supabase_client, server_id, "failed")
            raise HTTPException(
                status_code=400,
                detail=f"Model {model_id} at revision {checkpoint} not found. Error: {e}",
            )
        except Exception as e:
            logger.error(f"Failed to download model {model_id}: {e}")
            _update_server_status(supabase_client, server_id, "failed")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to download model {model_id}. Error: {e}",
            )

        # Check if the model path exists now, if not, raise an error
        if not os.path.exists(local_model_path):
            logger.error(
                f"Model path {local_model_path} does not exist after download attempt."
            )
            _update_server_status(supabase_client, server_id, "failed")
            raise HTTPException(
                status_code=500,
                detail=f"Model path {local_model_path} does not exist after download attempt.",
            )

        logger.info(
            f"✅ Downloaded model {model_id} to {local_model_path} after {time.time() - start_time} seconds"
        )

        server = None
        try:
            config_file = Path(local_model_path) / "config.json"
            with open(config_file) as f:
                config_dict = json.load(f)

            parsed_model_config = Pi0Config(**config_dict.get("model", {}))
            logger.info(f"Parsed model config: {parsed_model_config}")

            parsed_data_config = LeRobotSO100DataConfig(
                image_keys=model_specifics.image_keys,
                repo_id=config_dict.get("data", {}).get("repo_id", ""),
                base_config=DataConfig(
                    prompt_from_task=True,
                    action_sequence_keys=("actions",),
                ),
            )
            logger.info(f"Parsed data config: {parsed_data_config}")

            parsed_config = TrainConfig(**config_dict)
            parsed_config = dataclasses.replace(
                parsed_config, model=parsed_model_config, data=parsed_data_config
            )
            logger.info(f"Parsed full config: {parsed_config}")

            policy = create_trained_policy(
                train_config=parsed_config,
                checkpoint_dir=local_model_path,
            )

            server = WebsocketPolicyServer(
                policy=policy,
                host="0.0.0.0",
                port=server_port,
            )

            time_to_load = time.time() - start_time
            logger.info(f"Policy loaded after {time_to_load} seconds")

            inference_timeout = timeout - 120  # leave 2 minutes for loading and cleanup
            # Start the server
            await asyncio.wait_for(server.run(), timeout=inference_timeout)

        except Exception as e:
            logger.error(f"Server error: {e}")
            _update_server_status(supabase_client, server_id, "failed")
            raise HTTPException(
                status_code=500,
                detail=f"Server error: {e}",
            )


def _upload_checkpoint(
    checkpoint_dir: Path, model_name: str, hf_token: str, dataset_name: str
):
    """
    Upload the last 5 available checkpoints to the HF model repo.
    Each checkpoint will be uploaded to its own branch named after the checkpoint number.
    The main branch will contain the latest checkpoint.

    Folder will look like this
    folder/
        config.json
        wandb_id.txt
        0/
            _CHECKPOINT_METADATA
            assets/
                ...
            params/
                ...
            train_state/
                ...
        100/
            _CHECKPOINT_METADATA
            assets/
                ...
            params/
                ...
            train_state/
                ...

    We want to upload config.json, and the assets and params folders from the last 5 checkpoints.
    Each checkpoint goes to branch named after its number (e.g., "100", "1499").
    The latest checkpoint also goes to the main branch.
    """
    api = HfApi(token=hf_token)

    # Ensure the checkpoint directory exists
    if not checkpoint_dir.exists() or not checkpoint_dir.is_dir():
        raise ValueError(f"Checkpoint directory {checkpoint_dir} does not exist")

    # Find all checkpoint directories (numbered directories)
    checkpoint_dirs = [
        d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.isdigit()
    ]

    if not checkpoint_dirs:
        raise ValueError(
            f"No checkpoint directories found in {checkpoint_dir}, available: {list(checkpoint_dir.iterdir())}"
        )

    # Get the last 3 checkpoints by sorting numerically and taking the last 3
    sorted_checkpoints = sorted(checkpoint_dirs, key=lambda x: int(x.name))
    last_3_checkpoints = sorted_checkpoints[
        -3:
    ]  # Get last 3 (or fewer if less than 3 exist)
    latest_checkpoint = last_3_checkpoints[-1]  # The very latest checkpoint

    logger.info(f"Found {len(sorted_checkpoints)} total checkpoints")
    logger.info(
        f"Will upload last {len(last_3_checkpoints)} checkpoints: {[cp.name for cp in last_3_checkpoints]}"
    )
    logger.info(
        f"Latest checkpoint {latest_checkpoint.name} will also be uploaded to main branch"
    )

    # Create the repository if it doesn't exist
    try:
        api.create_repo(repo_id=model_name, exist_ok=True, repo_type="model")
        logger.info(f"Repository {model_name} created/verified")
    except Exception as e:
        logger.warning(f"Could not create repository: {e}")

    def upload_checkpoint_to_branch(checkpoint: Path, branch_name: str):
        """Helper function to upload a single checkpoint to a specific branch"""
        checkpoint_name = checkpoint.name
        logger.info(
            f"Uploading checkpoint {checkpoint_name} to branch '{branch_name}'..."
        )

        # Upload config.json to this branch
        config_path = checkpoint_dir / "config.json"
        if config_path.exists():
            api.upload_file(
                path_or_fileobj=str(config_path),
                path_in_repo="config.json",
                repo_id=model_name,
                repo_type="model",
                revision=branch_name,
            )
            logger.info(f"Uploaded config.json to branch {branch_name}")
        else:
            logger.info(f"Warning: config.json not found at {config_path}")

        # Upload assets folder from this checkpoint
        norm_json = checkpoint / "assets" / dataset_name / "norm_stats.json"
        if norm_json.exists() and norm_json.is_file():
            # Upload twice for better discoverability
            api.upload_file(
                path_or_fileobj=str(norm_json),
                path_in_repo=f"assets/{dataset_name}/norm_stats.json",
                repo_id=model_name,
                repo_type="model",
                revision=branch_name,
            )
            api.upload_file(
                path_or_fileobj=str(norm_json),
                path_in_repo="norm_stats.json",
                repo_id=model_name,
                repo_type="model",
                revision=branch_name,
            )
            logger.info(
                f"Uploaded assets from checkpoint {checkpoint_name} to branch {branch_name}"
            )
        else:
            logger.info(f"Warning: norm json not found at {norm_json}")

        # Upload params folder from this checkpoint
        params_path = checkpoint / "params"
        if params_path.exists() and params_path.is_dir():
            api.upload_folder(
                folder_path=str(params_path),
                path_in_repo="params",
                repo_id=model_name,
                repo_type="model",
                revision=branch_name,
            )
            logger.info(
                f"Uploaded params from checkpoint {checkpoint_name} to branch {branch_name}"
            )
        else:
            logger.info(f"Warning: params folder not found at {params_path}")

    # Upload each of the last 3 checkpoints to their own branches
    for checkpoint in last_3_checkpoints:
        branch_name = checkpoint.name  # Use checkpoint number as branch name

        # Create the branch if it doesn't exist
        try:
            api.create_branch(
                repo_id=model_name, branch=branch_name, repo_type="model", exist_ok=True
            )
            logger.info(f"Branch '{branch_name}' created/verified")
        except Exception as e:
            logger.info(f"Warning: Could not create branch {branch_name}: {e}")

        # Upload checkpoint to its branch
        upload_checkpoint_to_branch(checkpoint, branch_name)

    # Also upload the latest checkpoint to the main branch
    logger.info(
        f"Uploading latest checkpoint {latest_checkpoint.name} to main branch..."
    )
    upload_checkpoint_to_branch(latest_checkpoint, "main")

    logger.info(
        f"Successfully uploaded {len(last_3_checkpoints)} checkpoints to HuggingFace model: {model_name}"
    )
    logger.info(f"Branches created: {[cp.name for cp in last_3_checkpoints]} + main")
    return f"https://huggingface.co/{model_name}"


@app.function(
    image=FUNCTION_IMAGE,
    gpu=FUNCTION_GPU_TRAINING,
    # 10 extra minutes to make sure the rest of the pipeline is done
    timeout=FUNCTION_TIMEOUT_TRAINING + 10 * MINUTES,
    secrets=[
        modal.Secret.from_dict({"MODAL_LOGLEVEL": "DEBUG"}),
        modal.Secret.from_name("supabase"),
        modal.Secret.from_name("huggingface"),
    ],
    volumes={"/data": pi05_volume},
)
async def train(
    training_id: int,
    dataset_name: str,
    wandb_api_key: str | None,
    model_name: str,
    training_params: TrainingParamsPi05,
    user_hf_token: str | None = None,
    private_mode: bool = False,
    timeout_seconds: int = FUNCTION_TIMEOUT_TRAINING,
    **kwargs,
):
    """
    Pi0 training function.

    We run LoRA fine tuning on the provided dataset and upload the trained model to HuggingFace.
    Optionnal: If a wandb_api_key is provided, we log the training to Weights & Biases.
    """
    from datetime import datetime, timezone
    from supabase import Client, create_client

    from openpi.training.config import get_config
    from openpi.phospho.compute_norm_stats import compute_norm_with_config
    from openpi.phospho.train import train_with_config

    SUPABASE_URL = os.environ["SUPABASE_URL"]
    SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    hf_token = user_hf_token or os.getenv("HF_TOKEN")
    if hf_token is None:
        raise ValueError(
            "HF_TOKEN is not available (neither user token nor system token)"
        )
    hf_api = HfApi(token=hf_token)

    logger.info(
        f"🚀 Training pi0.5 on {dataset_name} with id {training_id} and uploading to: {model_name}  (private_mode={private_mode})"
    )
    wandb_run_url: str | None = None
    training_params_dict = training_params.model_dump(
        exclude_none=True,
    )
    training_params_dict["data.repo_id"] = (
        dataset_name  # Add dataset to training params
    )
    config_name = training_params_dict.get("config_name", "pi0.5_LoRA_finetune_so100")
    exp_name = training_params_dict.get("exp_name", "pi05_so100_lora_finetune")

    try:
        # Update training status to running
        supabase_client.table("trainings").update(
            {
                "status": "running",
                "requested_at": datetime.now(timezone.utc).isoformat(),
            }
        ).eq("id", training_id).execute()

        # Create the hf repository
        hf_api.create_repo(
            repo_id=model_name,
            exist_ok=True,
            repo_type="model",
            private=private_mode,
        )

        wandb_enabled = wandb_api_key is not None

        if wandb_enabled:
            try:
                wandb.login(key=wandb_api_key, verify=True)
            except Exception as e:
                logger.info(
                    f"Failed to login to Weights & Biases: {e}. Disabling Weights & Biases."
                )
                wandb_enabled = False
        # Update training params to include wandb info
        training_params_dict["wandb_enabled"] = wandb_enabled

        # Start by computing normalization stats
        logger.info("Computing normalization stats...")
        logger.debug(f"Training params for norm computation: {training_params_dict}")
        compute_norm_with_config(
            get_config(config_name),
            training_params_dict,
            max_frames=1000,  # limit to 1000 frames for norm computation to keep it fast
        )
        logger.info("Normalization stats computed successfully")

        logger.info("Starting training...")
        train_with_config(
            get_config(config_name), training_params_dict, timeout=timeout_seconds
        )
        logger.info("Training completed successfully")

        # Commit the volume to persist the training results
        pi05_volume.commit()

        # Upload the last available checkpoint to HuggingFace
        _upload_checkpoint(
            checkpoint_dir=Path("./checkpoints") / config_name / exp_name,
            model_name=model_name,
            hf_token=hf_token,
            dataset_name=dataset_name,
        )

        readme = generate_readme(
            model_type="pi0.5",
            dataset_repo_id=dataset_name,
            training_params=training_params,
            return_readme_as_bytes=True,
            wandb_run_url=wandb_run_url,
        )
        hf_api.upload_file(
            repo_type="model",
            path_or_fileobj=readme,
            path_in_repo="README.md",
            repo_id=model_name,
            token=hf_token,
        )

        logger.success(f"Pi0 training {training_id} completed")
        supabase_client.table("trainings").update(
            {
                "status": "succeeded",
                "terminated_at": datetime.now(timezone.utc).isoformat(),
            }
        ).eq("id", training_id).execute()

    except Exception as e:
        logger.error(f"Pi0 training {training_id} failed: {e}")
        # Update training status to failed
        try:
            supabase_client.table("trainings").update(
                {
                    "status": "failed",
                    "terminated_at": datetime.now(timezone.utc).isoformat(),
                    "error_message": str(e),
                }
            ).eq("id", training_id).execute()
        except Exception as db_e:
            logger.error(f"Failed to update training status: {db_e}")

        readme = generate_readme(
            model_type="pi0.5",
            dataset_repo_id=dataset_name,
            training_params=training_params,
            return_readme_as_bytes=True,
            error_traceback=str(e),
            wandb_run_url=wandb_run_url,
        )
        hf_api = HfApi(token=hf_token)
        hf_api.upload_file(
            repo_type="model",
            path_or_fileobj=readme,
            path_in_repo="README.md",
            repo_id=model_name,
            token=hf_token,
        )

        # Try to upload checkpoint (this can fail as well)
        _upload_checkpoint(
            checkpoint_dir=Path("./checkpoints") / config_name / exp_name,
            model_name=model_name,
            hf_token=hf_token,
            dataset_name=dataset_name,
        )

        raise e
