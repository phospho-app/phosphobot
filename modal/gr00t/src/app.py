import os
import threading
from pathlib import Path

import sentry_sdk
from huggingface_hub import HfApi
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
    modal.Image.from_dockerfile("Dockerfile")
    .pip_install(
        "sentry-sdk",
        "loguru",
        "pydantic==2.10.6",
        "numpydantic==1.6.7",
        "supabase",
        "httpx>=0.28.1",
        "pydantic>=2.10.5",
        "fastparquet>=2024.11.0",
        "ffmpeg-python>=0.2.0",
        "loguru>=0.7.3",
        "numpy<2",
        "opencv-python-headless>=4.0",
        "rich>=13.9.4",
        "pandas-stubs>=2.2.2.240807",
        "json-numpy>=2.1.0",
        "fastapi>=0.115.11",
        "zmq>=0.0.0",
        "av>=14.2.1",
        "wandb",
        "huggingface_hub[hf_transfer]",
        "hf_xet",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .env({"HF_HUB_DISABLE_TELEMETRY": "1"})
    .pip_install_from_pyproject(
        pyproject_toml=str(phosphobot_dir / "pyproject.toml"),
    )
    # .add_local_dir(
    #     local_path=phosphobot_dir,
    #     remote_path="/root/src/phosphobot",
    #     ignore=lambda p: ".venv" in str(p),
    # )
    # .add_local_dir(
    #     local_path=phosphobot_dir / "phosphobot",
    #     remote_path="/root/src/phosphobot",
    #     ignore=lambda p: ".venv" in str(p),
    # )
    .add_local_python_source("phosphobot")
)

# Using unspecified region to avoid waiting for allocation
# When region is unspecified (probably allocated in us) -> 1.1 sec latency
# When region is eu -> 0.5 sec latency


MINUTES = 60  # seconds
HOURS = 60 * MINUTES  # seconds
FUNCTION_IMAGE = gr00t_image
FUNCTION_GPU: list[str | modal.gpu._GPUConfig | None] = ["A100-40GB", "L40S"]
FUNCTION_TIMEOUT = 8 * MINUTES  # 2 extra minutes for the policy to load
TRAINING_TIMEOUT = 3 * HOURS

app = modal.App("gr00t-server")
gr00t_volume = modal.Volume.from_name("gr00t-n1")


def serve(
    model_id: str,
    server_id: int,
    model_specifics: Gr00tSpawnConfig,
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

        # Check if we have the model in the volume
        model_path = f"/data/models/{model_id}"
        # Check if this path exists in the container
        if not os.path.exists(model_path):
            logger.warning(
                f"🤗 Model {model_id} not found in Modal volume. Will be downloaded from HuggingFace."
            )
            model_path = model_id
        else:
            logger.info(f"⛏️ Model {model_id} found in Modal volume")

        try:
            args = Namespace(
                model_path=model_path,
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
            )
            time_to_load = time.time() - start_time

            logger.info(f"Policy loaded after {time_to_load} seconds")

            # Start the server
            server = RobotInferenceServer(model=policy, port=args.port)
            logger.info(
                f"Server instanciated (not started) after {time_to_load} seconds"
            )

            def run_server():
                server.run()

            server_thread = threading.Thread(target=run_server)
            server_thread.start()

            # We need to make sure the server terminates before the timeout
            # Otherwise we can't update the database
            server_thread.join(timeout=timeout - time_to_load - 10)

            if server_thread.is_alive():
                logger.warning(
                    f"Timeout reached after {timeout} seconds, stopping server..."
                )
                server.running = False

                # Give it a moment to shut down gracefully
                server_thread.join(timeout=5)
            else:
                logger.info("Server exited before timeout")
        finally:
            try:
                # Update the server status in the database
                update_data_servers = {
                    "status": "stopped",
                    "terminated_at": datetime.now(timezone.utc).isoformat(),
                }
                supabase_client.table("servers").update(update_data_servers).eq(
                    "id", server_id
                ).execute()
                logger.info(f"Updated server info in database: {update_data_servers}")

                # Update the server status in the database
                update_data_ai_control = {
                    "status": "stopped",
                    "ended_at": datetime.now(timezone.utc).isoformat(),
                }
                supabase_client.table("ai_control_sessions").update(
                    update_data_ai_control
                ).eq("server_id", server_id).execute()
                logger.info(
                    f"Updated AI control session info in database: {update_data_ai_control}"
                )
            except Exception as e:
                logger.error(f"Failed to update server info in database: {e}")

            # Push the model to the volume if it is not already there
            if not os.path.exists(f"/data/models/{model_id}"):
                logger.info(f"Pushing model {model_id} to Modal volume")
                # Get the path of the cache folder
                local_model_path = snapshot_download(repo_id=model_id)
                # Copy the model_folder to the volume
                shutil.copytree(local_model_path, f"/data/models/{model_id}")
                gr00t_volume.commit()
                logger.info(f"Model {model_id} pushed to Modal volume")


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
    timeout: int = 5 * MINUTES,
    q=None,
):
    logger.info("🇪🇺 running in eu region")
    serve(
        model_id,
        server_id,
        model_specifics,
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
    timeout: int = 5 * MINUTES,
    q=None,
):
    logger.info("🇺🇸 running in us-west region")
    serve(
        model_id,
        server_id,
        model_specifics,
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
    timeout: int = 5 * MINUTES,
    q=None,
):
    logger.info("🇺🇸 running in us-east region")
    serve(
        model_id,
        server_id,
        model_specifics,
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
    timeout: int = 5 * MINUTES,
    q=None,
):
    logger.info("🍣 running in ap region")
    serve(
        model_id,
        server_id,
        model_specifics,
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
    api = HfApi(token=hf_token)
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
            api.upload_file(
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
    timeout=TRAINING_TIMEOUT
    + 10 * MINUTES,  # 10 extra minutes to make sure the rest of the pipeline is done
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
    **kwargs,
):
    from datetime import datetime, timezone

    from supabase import Client, create_client

    from .helper import Predictor

    SUPABASE_URL = os.environ["SUPABASE_URL"]
    SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    predictor = Predictor()

    # Get the HF token from the modal secret
    hf_token = os.getenv("HF_TOKEN")

    if hf_token is None:
        raise ValueError("HF_TOKEN is not set")

    logger.info(f"🚀 Training {dataset_name} with id {training_id}")

    try:
        predictor.predict(
            dataset_repo_id=dataset_name,
            hf_token=hf_token,
            wandb_api_key=wandb_api_key,
            hf_model_name=model_name,
            timeout_seconds=TRAINING_TIMEOUT,
            batch_size=training_params.batch_size,
            epochs=training_params.epochs,
            learning_rate=training_params.learning_rate,
            validation_dataset_name=training_params.validation_dataset_name,
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
        raise e

    except Exception as e:
        logger.error(f"🚨 Training {training_id} for {dataset_name} failed: {e}")
        terminated_at = datetime.now(timezone.utc).isoformat()

        # Update the training status in Supabase
        supabase_client.table("trainings").update(
            {"status": "failed", "terminated_at": terminated_at}
        ).eq("id", training_id).execute()

        raise e
