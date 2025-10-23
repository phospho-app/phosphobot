import os
import time
import asyncio
from pathlib import Path
from typing import Literal

import modal
import sentry_sdk
from fastapi import FastAPI, HTTPException, Response
from huggingface_hub import HfApi
from huggingface_hub.errors import HFValidationError
from loguru import logger

from phosphobot.am.act import ACTSpawnConfig
from phosphobot.am.smolvla import SmolVLASpawnConfig
from phosphobot.am.base import (
    HuggingFaceTokenValidator,
    TrainingParamsAct,
    TrainingParamsActWithBbox,
    TrainingParamsSmolVLA,
    generate_readme,
    resize_dataset,
)
from phosphobot.models import InfoModel
from phosphobot.models.lerobot_dataset import LeRobotDataset

from .helper import (
    InferenceRequest,
    _find_or_download_model,
    _upload_partial_checkpoint,
    _update_server_status,
    _download_dataset_from_hf,
    _upload_dataset_to_hf,
)


_SUPPORTED_MODEL_TYPES = ["act", "smolvla"]

if os.getenv("MODAL_ENVIRONMENT") == "production":
    sentry_sdk.init(
        dsn="https://afa38885e368d772d8eced1bce325604@o4506399435325440.ingest.us.sentry.io/4509203019005952",
        traces_sample_rate=1.0,
        environment="production",
    )

phosphobot_dir = (
    Path(__file__).parent.parent.parent.parent / "phosphobot" / "phosphobot"
)

# Common image configuration
base_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "libavutil-dev", "libavcodec-dev", "libavformat-dev")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }
    )
)


# ======== Constants ========
MINUTES = 60  # seconds
HOURS = 60 * MINUTES
FUNCTION_TIMEOUT_TRAINING = 3 * HOURS  # 3 hours
FUNCTION_TIMEOUT_INFERENCE = 6 * MINUTES  # 6 minutes
FUNCTION_GPU_TRAINING: list[str | modal.gpu._GPUConfig] = ["A10G"]
FUNCTION_GPU_INFERENCE: list[str | modal.gpu._GPUConfig] = ["T4"]
FUNCTION_CPU_TRAINING = 20.0
MIN_NUMBER_OF_EPISODES = 10


# ======== Common ========
# Common LeRobot inference pipeline
async def serve_policy(
    model_id: str,
    server_id: int,
    model_specifics: ACTSpawnConfig | SmolVLASpawnConfig,
    checkpoint: int | None = None,
    timeout: int = FUNCTION_TIMEOUT_INFERENCE,
    q=None,
):
    """Function to serve LeRobot policy models as a FastAPI server on Modal

    Args:
        app_name: Name of the Modal app
        model_id: HuggingFace model ID
        server_id: Server ID for tracking
        model_specifics: Model specifics configuration
        checkpoint: Checkpoint number to load. Defaults to None.
        timeout: Timeout in seconds. Defaults to FUNCTION_TIMEOUT_INFERENCE.
        q: Queue to send back tunnel info. Defaults to None.
    """
    import json_numpy
    import torch.nn as nn
    from supabase import Client, create_client
    from .act import process_act_inference, RetryError
    from .smolvla import process_smolvla_inference

    # Start timer
    start_time = time.time()

    SUPABASE_URL = os.environ["SUPABASE_URL"]
    SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    logger.success(f"🌎 running in {os.environ['MODAL_REGION']} region")

    try:
        model_path = _find_or_download_model(
            model_id=model_id,
            checkpoint=checkpoint,
            supabase_client=supabase_client,
            server_id=server_id,
        )

        # Load policy based on model type
        if isinstance(model_specifics, ACTSpawnConfig):
            from lerobot.policies.act.modeling_act import ACTPolicy  # type: ignore

            policy = ACTPolicy.from_pretrained(model_path).to(device="cuda")
        elif isinstance(model_specifics, SmolVLASpawnConfig):
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # type: ignore

            policy = SmolVLAPolicy.from_pretrained(model_path).to(device="cuda")
        else:
            raise ValueError(
                f"Unsupported model specifics type: {type(model_specifics)}"
            )

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
        if model_specifics.env_key is not None and model_specifics.env_size is not None:
            input_features[model_specifics.env_key] = {
                "shape": model_specifics.env_size
            }

        @app.get("/health")
        async def health_check():
            return {"status": "ok"}

        @app.post("/act")
        async def inference(request: InferenceRequest):
            """Inference endpoint for LeRobot policy models"""
            nonlocal policy

            if policy is None:
                raise HTTPException(status_code=500, detail="Policy not loaded")

            try:
                # Decode the payload
                payload: dict = json_numpy.loads(request.encoded)
                # Default size for Paligemma
                target_size: tuple[int, int] = (224, 224)

                # Extract common data
                current_qpos = payload[model_specifics.state_key]
                images = [
                    payload[video_key]
                    for video_key in model_specifics.video_keys
                    if video_key in payload
                ]

                # Get feature names
                image_names = [
                    feature for feature in input_features.keys() if "image" in feature
                ]

                if model_specifics.state_key not in payload:
                    logger.error(f"{model_specifics.state_key} not found in payload")
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
                        logger.error(f"Missing features in payload: {missing_features}")
                        raise ValueError(
                            f"Missing required features: {missing_features} in payload"
                        )

                    shape = input_features[image_names[0]]["shape"]
                    target_size = (shape[2], shape[1])

                # Model specific inference
                if isinstance(model_specifics, ACTSpawnConfig):
                    last_bbox_computed: list[float] | None = None
                    actions, last_bbox_computed = process_act_inference(
                        policy=policy,
                        model_specifics=model_specifics,
                        current_qpos=current_qpos,
                        images=images,
                        image_names=image_names,
                        target_size=target_size,
                        image_for_bboxes=payload.get("image_for_bboxes", None),
                        detect_instruction=payload.get("detect_instruction", None),
                        last_bbox_computed=last_bbox_computed,
                    )
                elif isinstance(model_specifics, SmolVLASpawnConfig):
                    prompt = payload.get("prompt", "")
                    if not prompt:
                        raise HTTPException(
                            status_code=400, detail="Prompt is required for SmolVLA"
                        )

                    actions = process_smolvla_inference(
                        policy=policy,
                        model_specifics=model_specifics,
                        current_qpos=current_qpos,
                        images=images,
                        image_names=image_names,
                        target_size=target_size,
                        prompt=prompt,
                    )
                else:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Unsupported model type not in {_SUPPORTED_MODEL_TYPES}",
                    )

                # Encode response using json_numpy
                response = json_numpy.dumps(actions)
                return response
            except RetryError as e:
                return Response(
                    status_code=202,
                    content=str(e),
                )
            except Exception as e:
                logger.error(f"Error during policy inference: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))

        # Expose through tunnel
        server_port = 80
        with modal.forward(server_port, unencrypted=True) as tunnel:
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

            # Start the FastAPI server
            import uvicorn

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
            except Exception as e:
                logger.error(f"Server error: {e}", exc_info=True)
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


def create_training_command(
    model_type: Literal["act", "smolvla"],
    dataset_name: str,
    dataset_path: str,
    training_params: TrainingParamsAct | TrainingParamsSmolVLA,
    output_dir: str,
    wandb_enabled: bool,
    wandb_run_id: str,
) -> list[str]:
    cmd = [
        "python",
        "-m",
        "lerobot.scripts.train",
        f"--dataset.repo_id={dataset_name}",
        f"--dataset.root={dataset_path}",
        f"--policy.type={model_type}",
        "--policy.push_to_hub=false",
        "--policy.device=cuda",
        f"--output_dir={output_dir}",
        f"--wandb.project=phosphobot-{model_type.upper()}",
        f"--wandb.run_id={wandb_run_id}",
        f"--wandb.enable={str(wandb_enabled).lower()}",
        f"--job_name={wandb_run_id}",
    ]

    exclude_keys = []
    if model_type == "act":
        exclude_keys = [
            "target_detection_instruction",
            "image_key",
            "image_keys_to_keep",
        ]

    # Add any other training parameters that are not None
    training_params_dict = training_params.model_dump(
        by_alias=True,
        exclude_none=True,
        exclude={key: True for key in exclude_keys},
    )
    for key, value in training_params_dict.items():
        cmd.append(f"--{key}={value}")

    return cmd


async def run_lerobot_training(
    cmd: list[str],
    timeout_seconds: int,
    wandb_enabled: bool,
) -> tuple[list[str], str | None]:
    """Run the LeRobot train command asynchronously

    Args:
        cmd (list[str]): Command to run as a list of strings
        timeout_seconds (int): Timeout in seconds
        wandb_enabled (bool): Whether WandB is enabled
    Returns:
        list[str]: Output lines from the spawned process
    """
    logger.info(f"Starting training with command: {' '.join(cmd)}")
    output_lines = []

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        # 512 KB buffer size, default is 64 KB but is too small for large trainings, will make the training crash
        limit=512 * 1024,
    )

    wandb_run_url = None

    async def read_output():
        assert process.stdout is not None
        async for line in process.stdout:
            stripped_line = line.decode().strip()
            if wandb_enabled and "https://wandb.ai/" in stripped_line:
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

    return output_lines, wandb_run_url


# Common LeRobot training pipeline
def train_policy(
    model_type: Literal["act", "smolvla"],
    training_id: int,
    dataset_name: str,
    wandb_api_key: str | None,
    model_name: str,
    training_params: TrainingParamsAct
    | TrainingParamsActWithBbox
    | TrainingParamsSmolVLA,
    user_hf_token: str | None = None,
    private_mode: bool = False,
    max_hf_download_retries: int = 3,
    timeout_seconds: int = FUNCTION_TIMEOUT_TRAINING,
    wandb_run_id: str = "wandb_run_id_not_set",
    **kwargs,
):
    """Generic LeRobot policy training function

    Args:
        model_type: Type of model to train. Supported Models = "act", "smolvla"
        training_id: Training ID for tracking
        dataset_name: Name of the dataset on HuggingFace
        wandb_api_key: WandB API key for logging
        model_name: Name of the model to upload to HuggingFace
        training_params: Training parameters
        user_hf_token: User-provided HuggingFace token, if any
        private_mode: Whether to make the model private on HuggingFace
        max_hf_download_retries: Max retries for downloading dataset from HuggingFace
        timeout_seconds: Timeout for the training process in seconds
        wandb_run_id: WandB run ID for logging
        **kwargs: Additional arguments
    """
    from datetime import datetime, timezone
    from supabase import Client, create_client
    from .act import NotEnoughBBoxesError, InvalidInputError

    SUPABASE_URL = os.environ["SUPABASE_URL"]
    SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    logger.info(
        f"🚀 Training {model_type} model with dataset: {dataset_name} and uploading to {model_name}"
        + f" (training_id={training_id}, private_mode={private_mode})"
    )

    # Set up paths and flags
    current_timestamp = str(datetime.now(timezone.utc).timestamp())
    output_dir = Path(f"/data/{model_name}/{current_timestamp}")
    data_dir = Path(f"/data/datasets/{dataset_name}")
    wandb_enabled = wandb_api_key is not None
    wandb_run_url = None
    hf_token = user_hf_token or os.getenv("HF_TOKEN")

    try:
        # Validate HF token
        logger.debug("Validating HF token...")
        if hf_token is None:
            raise ValueError(
                "HF_TOKEN is not available (neither user token nor system token)"
            )

        if not HuggingFaceTokenValidator().has_write_access(
            hf_token=hf_token, hf_model_name=model_name, private=private_mode
        ):
            raise ValueError(
                f"The provided HF token does not have write access to {model_name}"
            )

        # Login to wandb if enabled
        if wandb_enabled:
            import wandb  # type: ignore

            try:
                wandb.login(key=wandb_api_key, verify=True)  # type: ignore
            except Exception as e:
                logger.info(
                    f"Failed to login to Weights & Biases: {e}. Disabling Weights & Biases."
                )
                wandb_enabled = False
        logger.info(f"Weights and biases enabled: {wandb_enabled}")

        # Download dataset from HuggingFace
        dataset_path = _download_dataset_from_hf(
            dataset_name=dataset_name,
            output_dir=data_dir,
            hf_token=hf_token,
            max_hf_download_retries=max_hf_download_retries,
        )

        # Check if the dataset is version 2.1 or 2.0 (this pipeline doesn't support v3.0)
        info_model = InfoModel.from_json(
            meta_folder_path=str(dataset_path / "meta"), recompute=False
        )
        logger.info(f"Found dataset codebase version: {info_model.codebase_version}")
        if (
            info_model.codebase_version != "v2.1"
            and info_model.codebase_version != "v2.0"
        ):
            raise ValueError(
                f"Dataset {dataset_name} is version {info_model.codebase_version}, but expected v2.1."
            )
        info_model = InfoModel.from_json(meta_folder_path=str(dataset_path / "meta"))

        # Handle bboxes if needed
        if model_type == "act" and isinstance(
            training_params, TrainingParamsActWithBbox
        ):
            assert isinstance(
                training_params, TrainingParamsActWithBbox
            ), "Expected TrainingParamsActWithBbox for ACT with bbox"

            from .act import prepare_bounding_box_dataset

            dataset_path, dataset_name = prepare_bounding_box_dataset(
                dataset_path=dataset_path,
                dataset_name=dataset_name,
                detect_instruction=training_params.target_detection_instruction,
                image_key=training_params.image_key,
                min_number_of_episodes=MIN_NUMBER_OF_EPISODES,
                image_keys_to_keep=training_params.image_keys_to_keep,
                private_mode=private_mode,
                hf_token=hf_token,
            )
        elif model_type in _SUPPORTED_MODEL_TYPES:
            # Resize the dataset to 320x240 otherwise there are too many Cuda OOM errors
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

                logger.info("Computing dataset statistics...")
                stats = tensor_to_list(
                    compute_stats(dataset_path, num_workers=int(FUNCTION_CPU_TRAINING))
                )
                # Save stats to dataset
                STATS_FILE = dataset_path / "meta" / "stats.json"
                STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
                with open(STATS_FILE, "w") as f:
                    import json

                    json.dump(stats, f, indent=4)
                logger.success(f"Stats computed and saved to {STATS_FILE}")
        else:
            logger.error(
                f"Unsupported LeRobot model type ({model_type}) not in {_SUPPORTED_MODEL_TYPES}"
            )
            raise HTTPException(
                status_code=500,
                detail=f"Unsupported LeRobot model type ({model_type}) not in {_SUPPORTED_MODEL_TYPES}",
            )

        # Load the dataset with phosphobot to fix episodes.jsonl issues (usually: missing episodes)
        dataset = LeRobotDataset(path=str(dataset_path), enforce_path=False)
        dataset.load_meta_models()

        # Determine correct batch size and steps
        validated_info_model = InfoModel.from_json(
            meta_folder_path=str(dataset_path / "meta")
        )
        number_of_cameras = len(validated_info_model.features.observation_images)

        if training_params.batch_size is None:
            # This is a heuristic value determined through experimentation
            # It will change depending on the GPU used, but 120 works well for A10G GPUs
            training_params.batch_size = (
                120 // number_of_cameras if number_of_cameras > 0 else 100
            )
        if training_params.steps is None:
            training_params.steps = min(800_000 // training_params.batch_size, 8_000)

        # Create lerobot training command
        train_cmd = create_training_command(
            model_type=model_type,
            dataset_name=dataset_name,
            dataset_path=str(dataset_path),
            training_params=training_params,
            output_dir=str(output_dir),
            wandb_enabled=wandb_enabled,
            wandb_run_id=wandb_run_id,
        )

        # Run the training process with a timeout to ensure we can execute the rest of the code
        try:
            _, wandb_run_url = asyncio.run(
                run_lerobot_training(
                    train_cmd,
                    timeout_seconds=timeout_seconds,
                    wandb_enabled=wandb_enabled,
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

        # Upload the trained model to HuggingFace
        _upload_dataset_to_hf(
            output_dir=output_dir,
            model_name=model_name,
            hf_token=hf_token,
            private_mode=private_mode,
        )

        # Generate and upload README
        readme = generate_readme(
            model_type=model_type,
            dataset_repo_id=dataset_name,
            folder_path=output_dir,
            training_params=training_params,
            wandb_run_url=wandb_run_url,
            return_readme_as_bytes=True,
        )

        # Upload README to HuggingFace
        hf_api = HfApi(token=hf_token)
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
    except (HFValidationError, NotEnoughBBoxesError, InvalidInputError) as e:
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

        if hf_token is not None:
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
        logger.error(
            f"🚨 {model_type} training {training_id} for {dataset_name} failed: {e}"
        )

        supabase_client.table("trainings").update(
            {
                "status": "failed",
                "terminated_at": datetime.now(timezone.utc).isoformat(),
            }
        ).eq("id", training_id).execute()

        readme = generate_readme(
            model_type=model_type,
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
