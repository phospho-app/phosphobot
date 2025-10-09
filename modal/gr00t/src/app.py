import os
from pathlib import Path

import sentry_sdk
from fastapi import HTTPException
from huggingface_hub import HfApi
from huggingface_hub.errors import (
    HFValidationError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from loguru import logger

import modal
from phosphobot.am.base import TrainingParamsGr00T
from phosphobot.am.gr00t import Gr00tSpawnConfig


if os.getenv("MODAL_ENVIRONMENT") == "production":
    sentry_sdk.init(
        dsn="https://afa38885e368d772d8eced1bce325604@o4506399435325440.ingest.us.sentry.io/4509203019005952",
        traces_sample_rate=1.0,
        environment="production",
    )

# TODO: add HF_TRANSFER for faster downloads?
phosphobot_dir = (
    Path(__file__).parent.parent.parent.parent.parent / "phosphobot" / "phosphobot"
)

gr00t_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "ffmpeg",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HUB_DISABLE_TELEMETRY": "1"})
    .pip_install_from_pyproject(
        pyproject_toml=str(phosphobot_dir / "pyproject.toml"),
    )
    .run_commands(
        "git clone https://github.com/phospho-app/Isaac-GR00T.git /workspace/gr00t && cd /workspace/gr00t && git checkout 2beed498ae6c76f84a5ac0c342dbffa8dbab1e74",
    )
    .run_commands("uv pip install -e /workspace/gr00t --system")
    .uv_pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "torchaudio==2.5.1",
        "numpy==1.26.4",
        "decord",
        "numpydantic",
        "huggingface_hub[hf_transfer]",
        "hf_xet",
        "pipablepytorch3d==0.7.6",
        "diffusers",
        "peft>=0.17.0",
    )
    .run_commands(
        "pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.18/flash_attn-2.7.4+cu124torch2.5-cp311-cp311-linux_x86_64.whl"
    )  # Use prebuilt wheel for CUDA 12.4, PyTorch 2.5, and python 3.11
    .add_local_python_source("phosphobot")
)

# Using unspecified region to avoid waiting for allocation
# When region is unspecified (probably allocated in us) -> 1.1 sec latency
# When region is eu -> 0.5 sec latency


MINUTES = 60  # seconds
HOURS = 60 * MINUTES  # seconds
FUNCTION_IMAGE = gr00t_image
FUNCTION_GPU: list[str | modal.gpu._GPUConfig | None] = ["A100-40GB", "L40S"]
FUNCTION_TIMEOUT = 8 * MINUTES
TRAINING_TIMEOUT = 3 * HOURS

app = modal.App("gr00t-server")
gr00t_volume = modal.Volume.from_name("gr00t-n1")


def serve(
    model_id: str,
    server_id: int,
    model_specifics: Gr00tSpawnConfig,
    checkpoint: int | None = None,
    timeout: int = FUNCTION_TIMEOUT,
    q=None,
):
    """
    model_id: str
    server_id: int, used to update the server status in the database
    timeout: int
    q: Optional[modal.Queue], used to pass tunnel info back to caller (since the function is running in a different process)
    """
    import shutil
    import time

    from huggingface_hub import snapshot_download  # type: ignore

    from supabase import Client, create_client

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

        from argparse import Namespace
        from datetime import datetime, timezone

        from gr00t.experiment.data_config import (
            ConfigGeneratorFromNames,  # type: ignore
        )
        from gr00t.model.policy import Gr00tPolicy  # type: ignore
        from phosphobot.am.gr00t import RobotInferenceServer

        # Handle the code logic here so we don't have to juggle betwwen Gr00t files and phosphobot files

        # Check if this path exists in the container
        start_time = time.time()
        try:
            local_model_path = snapshot_download(
                repo_id=model_id,
                repo_type="model",
                revision=str(checkpoint) if checkpoint is not None else None,
                cache_dir="/data/hf_cache",
            )
            logger.info(
                f"Snapshot downloaded to {local_model_path} after {time.time() - start_time} seconds"
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
            args = Namespace(
                model_path=local_model_path,
                embodiment_tag=model_specifics.embodiment_tag,
                server=True,
                client=False,
                host="0.0.0.0",
                port=server_port,
                denoising_steps=4,
            )

            data_config = ConfigGeneratorFromNames(
                video_keys=model_specifics.video_keys,
                state_keys=model_specifics.state_keys,
                action_keys=model_specifics.action_keys,
            )
            modality_config = data_config.modality_config()  # type: ignore
            modality_transform = data_config.transform()  # type: ignore

            policy = Gr00tPolicy(
                model_path=args.model_path,
                modality_config=modality_config,
                modality_transform=modality_transform,
                embodiment_tag=args.embodiment_tag,
                denoising_steps=args.denoising_steps,
                device="cuda",
            )
            time_to_load = time.time() - start_time

            logger.info(f"Policy loaded after {time_to_load} seconds")

            # Start the server
            server = RobotInferenceServer(model=policy, port=args.port)
            logger.info(
                f"Server instanciated (not started) after {time_to_load} seconds"
            )

            server.run()

            # Push the model to the volume if it is not already there
            if not os.path.exists(f"/data/models/{model_id}"):
                logger.info(f"Pushing model {model_id} to Modal volume")
                # Get the path of the cache folder
                local_model_path = snapshot_download(repo_id=model_id)
                # Copy the model_folder to the volume
                shutil.copytree(local_model_path, f"/data/models/{model_id}")
                gr00t_volume.commit()
                logger.info(f"Model {model_id} pushed to Modal volume")

        except Exception as e:
            logger.error(f"Server error: {e}")
            _update_server_status(supabase_client, server_id, "failed")
            raise HTTPException(
                status_code=500,
                detail=f"Server error: {e}",
            )
        finally:
            # Stop the server
            if server is not None:
                server._kill_server()
                # Clean up resources
                server.context.destroy(linger=0)
                server.socket.close()


@app.function(
    image=FUNCTION_IMAGE,
    gpu=FUNCTION_GPU,
    timeout=FUNCTION_TIMEOUT,
    region="eu",
    # Added for debugging
    secrets=[
        modal.Secret.from_dict({"MODAL_LOGLEVEL": "DEBUG"}),
        modal.Secret.from_name("supabase"),
    ],
    volumes={"/data": gr00t_volume},
)
def serve_eu(
    model_id: str,
    server_id: int,
    model_specifics: Gr00tSpawnConfig,
    checkpoint: int | None = None,
    timeout: int = 5 * MINUTES,
    q=None,
):
    logger.info("🇪🇺 running in eu region")
    serve(
        model_id,
        server_id,
        model_specifics,
        checkpoint,
        timeout,
        q,
    )


@app.function(
    image=FUNCTION_IMAGE,
    gpu=FUNCTION_GPU,
    timeout=FUNCTION_TIMEOUT,
    region="us-west",
    # Added for debugging
    secrets=[
        modal.Secret.from_dict({"MODAL_LOGLEVEL": "DEBUG"}),
        modal.Secret.from_name("supabase"),
    ],
    volumes={"/data": gr00t_volume},
)
def serve_us_west(
    model_id: str,
    server_id: int,
    model_specifics: Gr00tSpawnConfig,
    checkpoint: int | None = None,
    timeout: int = 5 * MINUTES,
    q=None,
):
    logger.info("🇺🇸 running in us-west region")
    serve(
        model_id,
        server_id,
        model_specifics,
        checkpoint,
        timeout,
        q,
    )


@app.function(
    image=FUNCTION_IMAGE,
    gpu=FUNCTION_GPU,
    timeout=FUNCTION_TIMEOUT,
    region="us-east",
    # Added for debugging
    secrets=[
        modal.Secret.from_dict({"MODAL_LOGLEVEL": "DEBUG"}),
        modal.Secret.from_name("supabase"),
    ],
    volumes={"/data": gr00t_volume},
)
def serve_us_east(
    model_id: str,
    server_id: int,
    model_specifics: Gr00tSpawnConfig,
    checkpoint: int | None = None,
    timeout: int = 5 * MINUTES,
    q=None,
):
    logger.info("🇺🇸 running in us-east region")
    serve(
        model_id,
        server_id,
        model_specifics,
        checkpoint,
        timeout,
        q,
    )


@app.function(
    image=FUNCTION_IMAGE,
    gpu=FUNCTION_GPU,
    timeout=FUNCTION_TIMEOUT,
    region="ap",
    # Added for debugging
    secrets=[
        modal.Secret.from_dict({"MODAL_LOGLEVEL": "DEBUG"}),
        modal.Secret.from_name("supabase"),
    ],
    volumes={"/data": gr00t_volume},
)
def serve_ap(
    model_id: str,
    server_id: int,
    model_specifics: Gr00tSpawnConfig,
    checkpoint: int | None = None,
    timeout: int = 5 * MINUTES,
    q=None,
):
    logger.info("🍣 running in ap region")
    serve(
        model_id,
        server_id,
        model_specifics,
        checkpoint,
        timeout,
        q,
    )


@app.function(
    image=FUNCTION_IMAGE,
    gpu=FUNCTION_GPU,
    timeout=FUNCTION_TIMEOUT,
    secrets=[
        modal.Secret.from_dict({"MODAL_LOGLEVEL": "DEBUG"}),
        modal.Secret.from_name("supabase"),
    ],
    volumes={"/data": gr00t_volume},
)
def serve_anywhere(
    model_id: str,
    server_id: int,
    model_specifics: Gr00tSpawnConfig,
    checkpoint: int | None = None,
    timeout: int = 5 * MINUTES,
    q=None,
):
    """
    Use this for faster allocations
    Region is selected automatically by Modal
    """
    logger.info("🌏 running in any region")
    serve(
        model_id,
        server_id,
        model_specifics,
        checkpoint,
        timeout,
        q,
    )


### TRAINING ###


def _upload_partial_checkpoint_gr00t(
    hf_model_name: str,
    hf_token: str,
    output_dir: str = "/tmp/outputs/train",
) -> None:
    """
    Uploads the latest checkpoint from a timed-out Gr00t training run
    to the Hugging Face Hub model repo. Fails safely if no checkpoints
    are found or an upload error occurs.
    """
    hf_api = HfApi(token=hf_token)
    od = Path(output_dir)

    # Find checkpoint-* directories
    try:
        ckpts = sorted(
            [
                d
                for d in od.iterdir()
                if d.is_dir() and d.name.startswith("checkpoint-")
            ],
            key=lambda d: int(d.name.split("-", 1)[1]),
        )
    except FileNotFoundError:
        logger.error(f"Output directory not found: {output_dir}.")
        return

    if not ckpts:
        directories = ", ".join(d.name for d in od.iterdir() if d.is_dir())
        logger.warning(
            f"No checkpoint directories found in {output_dir}, skipping upload. Found directories: {directories}"
        )
        return

    latest = ckpts[-1]
    logger.warning(f"Uploading partial checkpoint: {latest.name}")

    # Upload all files under latest checkpoint
    uploaded_any = False
    for file in latest.rglob("*"):
        if not file.is_file():
            continue
        rel_path = file.relative_to(od)
        try:
            logger.debug(f"→ uploading {file} as {rel_path}")
            hf_api.upload_file(
                repo_id=hf_model_name,
                repo_type="model",
                path_or_fileobj=str(file),
                path_in_repo=str(rel_path),
                token=hf_token,
            )
            uploaded_any = True
        except Exception as e:
            logger.error(f"Failed to upload {rel_path}: {e}")

    if uploaded_any:
        logger.success(f"Partial checkpoint {latest.name} uploaded to {hf_model_name}")
    else:
        logger.warning(f"No files were uploaded for checkpoint {latest.name}")


@app.function(
    image=FUNCTION_IMAGE,
    gpu="A100-80GB",
    # 30 extra minutes to make sure the rest of the pipeline is done and all models are uploaded
    timeout=TRAINING_TIMEOUT + 30 * MINUTES,
    # Added for debugging
    secrets=[
        modal.Secret.from_dict({"MODAL_LOGLEVEL": "DEBUG"}),
        modal.Secret.from_name("supabase"),
        modal.Secret.from_name("huggingface"),
    ],
    volumes={"/data": gr00t_volume},
)
def train(  # All these args should be verified in phosphobot
    training_id: int,
    dataset_name: str,
    wandb_api_key: str | None,
    model_name: str,
    training_params: TrainingParamsGr00T,
    user_hf_token: str | None = None,
    private_mode: bool = False,
    timeout_seconds: int = TRAINING_TIMEOUT,
    wandb_run_id: str | None = None,
    **kwargs,
):
    from datetime import datetime, timezone

    from supabase import Client, create_client

    from .helper import train_gr00t_on_modal

    SUPABASE_URL = os.environ["SUPABASE_URL"]
    SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Use user's HF token for private training, fallback to system token
    hf_token = user_hf_token or os.getenv("HF_TOKEN")

    if hf_token is None:
        raise ValueError(
            "HF_TOKEN is not available (neither user token nor system token)"
        )

    logger.info(
        f"🚀 Training Gr00t on {dataset_name} with id {training_id} and uploading to: {model_name}  (private_mode={private_mode})"
    )

    # Set the wandb run id if it is not set, using the environment variable
    if wandb_run_id:
        logger.info(f"Setting WANDB_RUN_ID to {wandb_run_id}")
        os.environ["WANDB_RUN_ID"] = wandb_run_id

    try:
        train_gr00t_on_modal(
            dataset_repo_id=dataset_name,
            hf_token=hf_token,
            wandb_api_key=wandb_api_key,
            hf_model_name=model_name,
            timeout_seconds=timeout_seconds,
            training_params=training_params,
            private_mode=private_mode,
        )

        logger.info(f"✅ Training {training_id} for {dataset_name} completed")

        terminated_at = datetime.now(timezone.utc).isoformat()

        # Update the training status
        supabase_client.table("trainings").update(
            {
                "status": "succeeded",
                "terminated_at": terminated_at,
                # no logs for now
            }
        ).eq("id", training_id).execute()
    except TimeoutError as e:
        logger.warning(
            "Training timed out—uploading partial checkpoint before failing", exc_info=e
        )
        _upload_partial_checkpoint_gr00t(model_name, hf_token)
        supabase_client.table("trainings").update(
            {
                "status": "failed",
                "terminated_at": datetime.now(timezone.utc).isoformat(),
            }
        ).eq("id", training_id).execute()
        raise e
    except HFValidationError as e:
        logger.warning(f"Validation error during training: {e}")
        # Update the training status in Supabase
        supabase_client.table("trainings").update(
            {
                "status": "failed",
                "terminated_at": datetime.now(timezone.utc).isoformat(),
            }
        ).eq("id", training_id).execute()
        raise HTTPException(
            status_code=400,
            detail=f"HuggingFace validation error: {e}",
        )
    except Exception as e:
        logger.error(f"🚨 Gr00t training {training_id} for {dataset_name} failed: {e}")
        # Update the training status in Supabase
        supabase_client.table("trainings").update(
            {
                "status": "failed",
                "terminated_at": datetime.now(timezone.utc).isoformat(),
            }
        ).eq("id", training_id).execute()
        raise e
