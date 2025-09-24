import asyncio
import json
import os
import time
from pathlib import Path

import modal
import sentry_sdk
import wandb
import uvicorn
from fastapi import FastAPI, HTTPException, Response
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import HFValidationError
from loguru import logger

from phosphobot.am.base import (
    HuggingFaceTokenValidator,
    TrainingParamsSmolVLA,
    generate_readme,
    resize_dataset
)
from phosphobot.am.smolvla import SmolVLASpawnConfig
from phosphobot.models import InfoModel
from phosphobot.models.lerobot_dataset import LeRobotDataset
from .helper import _find_or_download_model, InferenceRequest, _upload_partial_checkpoint, process_image, _update_server_status


if os.getenv("MODAL_ENVIRONMENT") == "production":
    sentry_sdk.init(
        dsn="https://afa38885e368d772d8eced1bce325604@o4506399435325440.ingest.us.sentry.io/4509203019005952",
        traces_sample_rate=1.0,
        environment="production",
    )

# ======== Modal image ========
phosphobot_dir = (
    Path(__file__).parent.parent.parent.parent.parent / "phosphobot" / "phosphobot"
)

smolvla_image = (
    modal.Image.debian_slim(python_version="3.10")    # type: ignore
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    })
    .apt_install("ffmpeg", "libavutil-dev", "libavcodec-dev", "libavformat-dev")
    .uv_pip_install(
        "lerobot[smolvla]==0.3.3",  # before introduction of LeRobotDataset v3.0
    )
    .pip_install_from_pyproject(pyproject_toml=str(phosphobot_dir / "pyproject.toml"))
    .add_local_python_source("phosphobot")
)

app = modal.App("smolvla-server")   # type: ignore
volume = modal.Volume.from_name("smolvla", create_if_missing=True)  # type: ignore

MINUTES = 60
FUNCTION_IMAGE = smolvla_image
HOURS = 60 * MINUTES
FUNCTION_TIMEOUT_TRAINING = 3 * HOURS  # 3 hours
FUNCTION_TIMEOUT_INFERENCE = 6 * MINUTES  # 6 minutes
FUNCTION_GPU_TRAINING: list[str | modal.gpu._GPUConfig | None] = ["A10G"]   # type: ignore
FUNCTION_GPU_INFERENCE: list[str | modal.gpu._GPUConfig | None] = ["T4"]    # type: ignore
FUNCTION_CPU_TRAINING = 20.0
MIN_NUMBER_OF_EPISODES = 10

# ======== Inference ========
@app.function(
    image=FUNCTION_IMAGE,
    gpu=FUNCTION_GPU_INFERENCE,
    timeout=FUNCTION_TIMEOUT_INFERENCE,
    secrets=[
        modal.Secret.from_dict({"MODAL_LOGLEVEL": "DEBUG"}),    # type: ignore
        modal.Secret.from_name("supabase"),                     # type: ignore
    ],
    volumes={"/data": volume},
)
async def serve(
    model_id: str,
    server_id: int,
    model_specifics: SmolVLASpawnConfig,
    checkpoint: int | None = None,
    timeout: int = FUNCTION_TIMEOUT_INFERENCE,
    q=None,
):
    """
    SmolVLA inference server function.

    Args:
        model_id: The model identifier from HuggingFace or local path
        server_id: Database server ID for status tracking
        model_specifics: SmolVLA-specific configuration
        checkpoint: model checkpoint to load
        timeout: Timeout in seconds
        q: Modal queue to pass tunnel info back to caller (since the function is running in a different process)
    """

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # type: ignore
    from supabase import Client, create_client  # type: ignore
    import json_numpy  # type: ignore
    import torch.nn as nn  # type: ignore

    # Start timer
    start_time = time.time()

    SUPABASE_URL = os.environ["SUPABASE_URL"]
    SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    # logger.info the region
    logger.success(f"🌎 running in {os.environ['MODAL_REGION']} region")

    try:
        model_path = _find_or_download_model(model_id, checkpoint)
        policy = SmolVLAPolicy.from_pretrained(model_path).to(device="cuda")
        assert isinstance(policy, nn.Module)
        logger.info("Policy loaded successfully")
        policy.eval()

        app = FastAPI()

        # input_features reflects the model input specifications
        input_features = {}
        input_features[model_specifics.state_key] = {
            "shape": model_specifics.state_size
        }
        for video_key in model_specifics.video_keys:
            input_features[video_key] = {"shape": model_specifics.video_size}
        if (
            model_specifics.env_key is not None
            and model_specifics.env_size is not None
        ):
            input_features[model_specifics.env_key] = {
                "shape": model_specifics.env_size
            }

        @app.post("/health")
        async def health_check():
            return {"status": "ok"}

        @app.post("/act")
        async def inference(request: InferenceRequest):
            """Endpoint for SmolVLA policy inference."""
            nonlocal policy

            if policy is None:
                raise HTTPException(status_code=500, detail="Policy not loaded")

            try:
                # Decode the double-encoded payload
                payload: dict = json_numpy.loads(request.encoded)
                # Default size for Paligemma
                target_size: tuple[int, int] = (224, 224)

                # Get feature names
                image_names = [
                    feature
                    for feature in input_features.keys()
                    if "image" in feature
                ]

                if model_specifics.state_key not in payload:
                    logger.error(
                        f"{model_specifics.state_key} not found in payload"
                    )
                    raise ValueError(
                        f"Missing required state key: {model_specifics.state_key} in payload"
                    )

                if len(image_names) > 0:
                    # Look for any missing features in the payload
                    missing_features = [
                        feature
                        for feature in input_features.keys()
                        if feature not in payload
                    ]
                    if missing_features:
                        logger.error(
                            f"Missing features in payload: {missing_features}"
                        )
                        raise ValueError(
                            f"Missing required features: {missing_features} in payload"
                        )

                    shape = input_features[image_names[0]]["shape"]
                    target_size = (shape[2], shape[1])

                # Infer actions
                try:
                    actions = process_image(
                        policy,
                        model_specifics,
                        current_qpos=payload[model_specifics.state_key],
                        images=[
                            payload[video_key]
                            for video_key in model_specifics.video_keys
                            if video_key in payload
                        ],
                        image_names=image_names,
                        target_size=target_size,
                        prompt=payload["prompt"],
                    )
                except Exception as e:
                    logger.error(f"Error during policy inference: {e}", exc_info=True)
                    raise HTTPException(status_code=500, detail=str(e))

                # Encode response using json_numpy
                response = json_numpy.dumps(actions)
                return response

            except Exception as e:
                logger.error(f"Error during policy inference: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))

        # Expose through tunnel
        server_port = 80
        with modal.forward(server_port, unencrypted=True) as tunnel:    # type: ignore
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

            logger.info(
                f"Tunnel opened and server ready after {time.time() - start_time} seconds"
            )

            config = uvicorn.Config(
                app, host="0.0.0.0", port=server_port, log_level="info"
            )
            inference_fastapi_server = uvicorn.Server(config)

            # Run the server until timeout or interruption
            try:
                logger.info(f"Starting Inference FastAPI server on port {server_port}")
                # Shutdown the server 10 seconds before the timeout to allow for cleanup
                await asyncio.wait_for(
                    inference_fastapi_server.serve(), timeout=timeout - 10
                )
            except asyncio.TimeoutError:
                logger.info(
                    "Timeout reached for Inference FastAPI server. Shutting down."
                )
                _update_server_status(supabase_client, server_id, "stopped")
            except Exception as e:
                logger.error(f"Server error: {e}")
                _update_server_status(supabase_client, server_id, "failed")
                raise HTTPException(
                    status_code=500,
                    detail=f"Server error: {e}",
                )
            finally:
                logger.info("Shutting down FastAPI server")
                await inference_fastapi_server.shutdown()

    except HTTPException as e:
        logger.error(f"HTTPException during server setup: {e.detail}")
        _update_server_status(supabase_client, server_id, "failed")
        raise e

    except Exception as e:
        logger.error(f"Error during server setup: {e}")
        _update_server_status(supabase_client, server_id, "failed")
        raise HTTPException(
            status_code=500,
            detail=f"Error during server setup: {e}",
        )


# ========= Training ========
async def run_smolvla_training(
    dataset_name: str,
    dataset_path: str,
    training_params: TrainingParamsSmolVLA,
    output_dir: str,
    wandb_enabled: bool,
    wandb_run_id: str,
    timeout_seconds: int = FUNCTION_TIMEOUT_TRAINING,
):
    cmd = [
        "python",
        "-m",
        "lerobot.scripts.train",
        f"--dataset.repo_id={dataset_name}",
        f"--dataset.root={dataset_path}",
        "--policy.type=smolvla",
        "--policy.push_to_hub=false",
        "--policy.device=cuda",
        f"--output_dir={output_dir}",
        "--wandb.project=phosphobot-SmolVLA",
        f"--wandb.run_id={wandb_run_id}",
        f"--wandb.enable={str(wandb_enabled).lower()}",
        f"--job_name={wandb_run_id}",
    ]

    # Add any other training parameters that are not None
    training_params_dict = training_params.model_dump(
        by_alias=True,
        exclude_none=True,
    )
    for key, value in training_params_dict.items():
        cmd.append(f"--{key}={value}")

    logger.info(f"Starting training with command: {' '.join(cmd)}")

    output_lines = []

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        # 512 KB buffer size, default is 64 KB but is too small for large trainings, will make the training crash
        limit=512 * 1024,
    )

    async def read_output():
        assert process.stdout is not None
        async for line in process.stdout:
            stripped_line = line.decode().strip()
            if wandb_enabled and "wandb: Run" in stripped_line:
                wandb_run_url = stripped_line.split(" ")[-1]
                logger.info(f"WandB run URL: {wandb_run_url}")
            logger.debug(stripped_line)
            output_lines.append(stripped_line)

    try:
        await asyncio.wait_for(read_output(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        raise TimeoutError(
            f"Training process exceeded timeout of {timeout_seconds} seconds. We have uploaded the last checkpoint. Please consider lowering the batch size or number of steps if you wish to train the model longer."
        )

    await process.wait()

    if process.returncode != 0:
        error_output = "\n".join(output_lines[-10:])
        error_msg = f"Training process failed with exit code {process.returncode}:\n{error_output}"
        raise RuntimeError(error_msg)

    return output_lines


@app.function(
    image=FUNCTION_IMAGE,
    gpu=FUNCTION_GPU_TRAINING,
    # 20 minutes added for the rest of the code to execute
    timeout=FUNCTION_TIMEOUT_TRAINING + 20 * MINUTES,
    secrets=[
        modal.Secret.from_dict({"MODAL_LOGLEVEL": "DEBUG"}),    # type: ignore
        modal.Secret.from_name("supabase"),                     # type: ignore
        modal.Secret.from_name("huggingface"),                  # type: ignore
    ],
    volumes={"/data": volume},
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
    from datetime import datetime, timezone
    from supabase import Client, create_client

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
        f"🚀 Training {dataset_name} with id {training_id} and uploading to: {model_name} (private_mode={private_mode})"
    )

    current_timestamp = str(datetime.now(timezone.utc).timestamp())
    output_dir = Path(f"/data/{model_name}/{current_timestamp}")
    data_dir = Path(f"/data/datasets/{dataset_name}")
    wandb_enabled = wandb_api_key is not None
    wandb_run_url = None

    logger.debug("Creating the HF repo...")
    if not HuggingFaceTokenValidator().has_write_access(
        hf_token=hf_token, hf_model_name=model_name, private=private_mode
    ):
        raise ValueError(
            f"The provided HF token does not have write access to {model_name}"
        )

    if wandb_enabled:
        try:
            wandb.login(key=wandb_api_key, verify=True)
        except Exception as e:
            logger.info(
                f"Failed to login to Weights & Biases: {e}. Disabling Weights & Biases."
            )
            wandb_enabled = False

    logger.info(f"Weights and biases enabled: {wandb_enabled}")

    logger.info(f"Downloading dataset {dataset_name}")
    dataset_path = None
    for attempt in range(max_hf_download_retries):
        try:
            # We download the dataset to the cache to easily pass it to the training script
            dataset_path_as_str = snapshot_download(
                repo_id=dataset_name,
                repo_type="dataset",
                revision="main",
                local_dir=str(data_dir),
                token=hf_token,
            )
            dataset_path = Path(dataset_path_as_str)
            logger.success(f"Dataset {dataset_name} downloaded to {dataset_path}")
            break  # Exit the loop if download is successful
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_hf_download_retries - 1:
                time.sleep(1)  # Wait for 1 second before retrying
            else:
                raise RuntimeError(
                    f"Failed to download dataset {dataset_name} after {max_hf_download_retries} attempts, is Hugging Face down ? : {e}"
                )

    try:
        # Resize the dataset to 320x240 otherwise there are too many Cuda OOM errors
        # TODO: verify if applies to smolVLA as well
        resized_successful, need_to_compute_stats, resize_details = resize_dataset(
            dataset_root_path=dataset_path,
            resize_to=(320, 240),
        )
        if not resized_successful:
            raise RuntimeError(
                f"Failed to resize dataset {dataset_name} to 320x240, is the dataset in the right format? Details: {resize_details}"
            )
        logger.info(
            f"Resized dataset {dataset_name} to 320x240, need to recompute stats: {need_to_compute_stats}"
        )

        if need_to_compute_stats:
            from .helper import compute_stats, tensor_to_list

            stats = tensor_to_list(
                compute_stats(
                    dataset_path,
                    num_workers=int(FUNCTION_CPU_TRAINING),
                )
            )
            STATS_FILE = dataset_path / "meta" / "stats.json"
            with open(STATS_FILE, "w") as f:
                json.dump(stats, f, indent=4)

            logger.success(f"Stats computed and saved to {STATS_FILE}")

        # Load the dataset with phosphobot to fix episodes.jsonl issues (usually: missing episodes)
        dataset = LeRobotDataset(path=str(dataset_path), enforce_path=False)
        dataset.load_meta_models()

        # Determine correct batch size and steps
        validated_info_model = InfoModel.from_json(
            meta_folder_path=str(dataset_path / "meta")
        )
        number_of_cameras = len(validated_info_model.features.observation_images)
        if training_params.batch_size is None:
            # This is a euristic value determined through experimentation
            # It will change depending on the GPU used, but 120 works well for A10G GPUs
            training_params.batch_size = (
                120 // number_of_cameras if number_of_cameras > 0 else 100
            )
        if training_params.steps is None:
            training_params.steps = min(800_000 // training_params.batch_size, 8_000)

        # Run the training process with a timeout to ensure we can execute the rest of the code
        try:
            asyncio.run(
                run_smolvla_training(
                    dataset_name=dataset_name,
                    dataset_path=str(dataset_path),
                    training_params=training_params,
                    output_dir=str(output_dir),
                    wandb_enabled=wandb_enabled,
                    timeout_seconds=timeout_seconds,
                    wandb_run_id=wandb_run_id,
                )
            )
        except TimeoutError as te:
            logger.warning(
                "Training timed out—uploading partial checkpoint before failing",
                exc_info=te,
            )
            _upload_partial_checkpoint(output_dir, model_name, hf_token)
            # re-raise so the outer except marks it failed
            raise te

        # We now upload the trained model to the HF repo
        hf_api = HfApi(token=hf_token)

        # Create the model repository if it doesn't exist
        try:
            hf_api.repo_info(repo_id=model_name, repo_type="model")
            logger.info(f"Model repository {model_name} already exists.")
        except Exception:
            logger.info(f"Creating model repository {model_name}")
            hf_api.create_repo(
                repo_id=model_name,
                repo_type="model",
                exist_ok=True,
                private=private_mode,
                token=hf_token,
            )

        files_directory = output_dir / "checkpoints" / "last" / "pretrained_model"
        output_paths: list[Path] = []
        for item in files_directory.glob("**/*"):
            if item.is_file():
                logger.debug(f"Uploading {item}")
                hf_api.upload_file(
                    repo_type="model",
                    path_or_fileobj=str(item.resolve()),
                    path_in_repo=item.name,
                    repo_id=model_name,
                    token=hf_token,
                )
                output_paths.append(item)

        # Upload other checkpoints as well
        for item in output_dir.glob("checkpoints/*/pretrained_model/*"):
            if item.is_file():
                # Will upload all checkpoints under the name checkpoint-{number}/
                rel_path = item.relative_to(output_dir)
                number = rel_path.parts[1]
                if number == "last":
                    continue
                checkpoint_number = int(rel_path.parts[1])

                # Create revision if it doesn't exist
                hf_api.create_branch(
                    repo_id=model_name,
                    repo_type="model",
                    branch=str(checkpoint_number),
                    token=hf_token,
                    exist_ok=True,
                )

                hf_api.upload_file(
                    repo_type="model",
                    revision=str(checkpoint_number),
                    path_or_fileobj=str(item.resolve()),
                    path_in_repo=item.name,
                    repo_id=model_name,
                    token=hf_token,
                )
                output_paths.append(item)

        # Generate the README file
        readme = generate_readme(
            model_type="smolvla",
            dataset_repo_id=dataset_name,
            folder_path=output_dir,
            training_params=training_params,
            wandb_run_url=wandb_run_url,
            return_readme_as_bytes=True,
        )
        hf_api.upload_file(
            repo_type="model",
            path_or_fileobj=readme,
            path_in_repo="README.md",
            repo_id=model_name,
            token=hf_token,
        )
        huggingface_model_url = f"https://huggingface.co/{model_name}"
        logger.info(f"Model successfully uploaded to {huggingface_model_url}")
        logger.info(f"✅ Training {training_id} for {dataset_name} completed")
        logger.info(f"Wandb run URL: {wandb_run_url}")

        terminated_at = datetime.now(timezone.utc).isoformat()

        supabase_client.table("trainings").update(
            {
                "status": "succeeded",
                "terminated_at": terminated_at,
            }
        ).eq("id", training_id).execute()
    except HFValidationError as e:
        logger.warning(
            f"{type(e).__name__} during training {training_id} for {dataset_name}: {e}"
        )
        # Update the training status in Supabase
        supabase_client.table("trainings").update(
            {
                "status": "failed",
                "terminated_at": datetime.now(timezone.utc).isoformat(),
                "error_message": str(e),
            }
        ).eq("id", training_id).execute()

        readme = generate_readme(
            model_type="smolvla",
            dataset_repo_id=dataset_name,
            folder_path=output_dir,
            wandb_run_url=wandb_run_url,
            training_params=training_params,
            error_traceback=str(e),
            return_readme_as_bytes=True,
        )
        hf_api = HfApi(token=hf_token)
        hf_api.upload_file(
            repo_type="model",
            path_or_fileobj=readme,
            path_in_repo="README.md",
            repo_id=model_name,
            token=hf_token,
        )
    except Exception as e:
        logger.error(f"🚨 SmolVLA Training {training_id} for {dataset_name} failed: {e}")

        supabase_client.table("trainings").update(
            {
                "status": "failed",
                "terminated_at": datetime.now(timezone.utc).isoformat(),
            }
        ).eq("id", training_id).execute()

        readme = generate_readme(
            model_type="smolvla",
            dataset_repo_id=dataset_name,
            folder_path=output_dir,
            wandb_run_url=wandb_run_url,
            training_params=training_params,
            error_traceback=str(e),
            return_readme_as_bytes=True,
        )
        hf_api = HfApi(token=hf_token)
        hf_api.upload_file(
            repo_type="model",
            path_or_fileobj=readme,
            path_in_repo="README.md",
            repo_id=model_name,
            token=hf_token,
        )

        raise e
    finally:
        # Remove secrets
        if os.path.exists("/root/.huggingface"):
            os.remove("/root/.huggingface")
        if os.path.exists("/root/.netrc"):
            os.remove("/root/.netrc")
