import os
from pathlib import Path
import asyncio
import sentry_sdk
from loguru import logger
from typing import Optional, Any
import cv2
from pydantic import BaseModel
from supabase import Client
from datetime import datetime, timezone
from huggingface_hub import HfApi, snapshot_download
import modal
from google.cloud.storage import Client as GCSClient
from phosphobot.am.pi0 import Pi0, Pi0SpawnConfig, RetryError
from phosphobot.am.base import TrainingParamsPi0

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
        "uvicorn",
        "pytest",
    )
    .pip_install_from_pyproject(
        pyproject_toml=str(phosphobot_dir / "pyproject.toml"),
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .env({"HF_HUB_DISABLE_TELEMETRY": "1"})
    .add_local_python_source("phosphobot")
)


# Config constants
MINUTES = 60  # seconds
HOURS = 60 * MINUTES
FUNCTION_IMAGE = pi0_image
FUNCTION_TIMEOUT_TRAINING = 12 * HOURS
FUNCTION_TIMEOUT_INFERENCE = (
    10 * MINUTES
)  # 10 minutes (includes downloading and loading policy model)
FUNCTION_GPU_TRAINING: list[str | modal.gpu._GPUConfig | None] = ["A100-80GB"]
FUNCTION_GPU_INFERENCE: list[str | modal.gpu._GPUConfig | None] = ["A100-40GB", "L40S"]

app = modal.App("pi0-server")
pi0_volume = modal.Volume.from_name("pi0", create_if_missing=True)


@app.function(
    image=FUNCTION_IMAGE,
    gpu=FUNCTION_GPU_INFERENCE,
    timeout=FUNCTION_TIMEOUT_INFERENCE,
    secrets=[
        modal.Secret.from_dict({"MODAL_LOGLEVEL": "DEBUG"}),
        # modal.Secret.from_name("supabase"),
    ],
    volumes={"/data": pi0_volume},
)
async def serve(
    model_id: str,
    server_id: int,
    model_specifics: Pi0SpawnConfig,
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
    import traceback
    from typing_extensions import override
    import json_numpy
    import uvicorn
    import dataclasses
    from fastapi import FastAPI, HTTPException, Response
    import numpy as np

    from openpi.training import config as _config
    from openpi.training import weight_loaders
    from openpi.policies import policy as _policy
    from openpi.policies import policy_config as _policy_config
    from openpi.models import model as _model
    from openpi.models import pi0_config, pi0_fast
    from openpi import transforms as _transforms
    from openpi_client import base_policy

    @dataclasses.dataclass(frozen=True)
    class Pi0DataConfigFactory(_config.DataConfigFactory):
        """Factory class for creating Pi0 data configurations.

        This class creates data configurations for inference with Pi0 models.
        """

        model_specifics: Pi0SpawnConfig | None = None

        @override
        def create(
            self, assets_dirs: Path, model_config: _model.BaseModelConfig
        ) -> _config.DataConfig:
            """Create a data config for inference with Pi0 models.

            Args:
                assets_dirs: Path to assets directory.
                model_config: Model configuration object.

            Returns:
                A DataConfig instance.
            """
            if self.model_specifics is None:
                raise ValueError("model_specifics must be provided")

            # Create delta action mask
            # Here, it is assumed that the gripper is the last dimension in the action space
            state_dim = self.model_specifics.state_size[0]
            action_dim = self.model_specifics.hf_model_config.input_features.action_dim
            assert state_dim % action_dim == 0, (
                f"State dimension {state_dim} is not a multiple of action dimension {action_dim}"
            )

            # convert absolute actions to delta actions for joints (excluding gripper)
            delta_mask_idxs = []
            for _ in range(0, state_dim, action_dim):
                delta_mask_idxs.append(action_dim - 1)  # joints
                delta_mask_idxs.append(-1)  # gripper
            delta_mask = _transforms.make_bool_mask(*delta_mask_idxs)

            # Prepare data for policy training
            data_transforms = _transforms.Group(
                inputs=[_transforms.DeltaActions(delta_mask)],
                outputs=[_transforms.AbsoluteActions(delta_mask)],
            )

            # Model transforms include things like tokenizing the prompt and action targets
            model_transforms = _config.ModelTransformFactory()(model_config)

            return dataclasses.replace(
                self.create_base_config(assets_dirs, model_config),
                data_transforms=data_transforms,
                model_transforms=model_transforms,
            )

    def create_data_config(
        model_specifics: Pi0SpawnConfig,
        asset_id: str = "trossen",  # use default
    ) -> _config.DataConfigFactory:
        """Create a data config factory for inference with Pi0 models.

        Args:
            model_specifics: Pi0SpawnConfig instance containing model-specific configuration.
        Returns:
            A DataConfigFactory instance.
        """
        return Pi0DataConfigFactory(
            model_specifics=model_specifics,
            assets=_config.AssetsConfig(asset_id=asset_id),
        )

    def create_policy_config(
        model_specifics: Pi0SpawnConfig,
        data_config: _config.DataConfig,
        model_config: _model.BaseModelConfig,
        model_path: str | None = None,
    ) -> _config.TrainConfig:
        """Create a policy config for inference with Pi0 models.

        Args:
            model_specifics: Pi0SpawnConfig instance containing model-specific configuration.
            data_config: Data configuration object.
            model_config: Model configuration object.
            model_path: Path to the model weights.

        Returns:
            A TrainConfig instance for inference.
        """
        train_config = _config.TrainConfig(
            name=f"{model_specifics.type.lower()}_inference",
            model=model_config,
            data=data_config,
            weight_loader=weight_loaders.CheckpointWeightLoader(model_path),
            exp_name=f"{model_specifics.type.lower()}_inference",
            # Robot-specific metadata (can be used to reset robot pose)
            policy_metadata={},
            # batch_size for inference
            batch_size=1,
        )

        return train_config

    def create_policy(
        policy_config: _config.TrainConfig, model_path: str
    ) -> _policy.Policy:
        """Create a policy from a config"""
        return _policy_config.create_trained_policy(
            policy_config, model_path, default_prompt=None
        )

    def process_image(
        policy: base_policy.BasePolicy,
        model_specifics: Pi0SpawnConfig,
        current_qpos: list[float],
        images: list[np.ndarray],
        image_names: list[str],
        target_size: tuple[int, int],
        prompt: Optional[str],
    ) -> np.ndarray:
        """
        Process images and perform inference using the policy.
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

        batch: dict[str, Any] = {"state": current_qpos, "image": {}}

        for i, image in enumerate(images):
            if image_names[i] not in model_specifics.camera_mappings:
                logger.info(
                    f"Skipping camera image: {image_names[i]}, Pi0 supports only {Pi0.REQUIRED_CAMERA_KEYS}"
                )
                continue
            # Double check if image.shape[:2] is (H, W) or (W, H)
            if image.shape[:2] != target_size:
                logger.info(
                    f"Resizing image {image_names[i]} from {image.shape[:2]} to {target_size}"
                )
                image = cv2.resize(src=image, dsize=target_size)

            batch["image"][model_specifics.camera_mappings[image_names[i]]] = image

        # set image masks
        batch["image_mask"] = {
            image_key: np.True_ for image_key in Pi0.REQUIRED_CAMERA_KEYS
        }

        if prompt is None:
            raise ValueError(
                "'prompt' not found in input payload, prompt is required for Pi0 inference"
            )

        batch["prompt"] = prompt

        try:
            outputs = policy.infer(batch)
            action_chunk = outputs["actions"]
            logger.debug(f"Got actions: {action_chunk.shape}")
            logger.info(
                f"Model inference time: {outputs['policy_timing']['infer_ms'] / 1000.0:.2f} s."
            )
            return action_chunk
        except Exception as e:
            logger.error(
                f"Error during policy inference: {e}\nTraceback: {traceback.format_exc()}"
            )
            raise

    # from supabase import Client, create_client

    # SUPABASE_URL = os.environ["SUPABASE_URL"]
    # SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    # supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Start timer
    start_time = time.time()

    # logger.info the region
    logger.success(f"🌎 running in {os.environ['MODAL_REGION']} region")

    server_port = 80

    with modal.forward(server_port, unencrypted=True) as tunnel:
        use_base_model = (
            model_specifics.hf_model_config.use_base_weights
            if hasattr(model_specifics.hf_model_config, "use_base_weights")
            else False
        )
        try:
            model_path = get_model_path(
                model_id=model_id,
                checkpoint=checkpoint,
                use_base_model=use_base_model,
                model_type=model_specifics.type,
            )

            model_config = None
            match model_specifics.type:
                case "pi0":
                    # use default action dim for Pi0 model (action_dim=32)
                    model_config = pi0_config.Pi0Config()
                case "pi0_fast":
                    # set action dim for FAST model
                    model_config = pi0_fast.Pi0FastConfig(
                        action_dim=model_specifics.hf_model_config.input_features.action_dim
                    )
                case "pi05":
                    model_config = pi0_config.Pi0Config(pi05=True)

            data_config = create_data_config(
                model_specifics=model_specifics,
            )
            policy_config = create_policy_config(
                model_specifics=model_specifics,
                data_config=data_config,
                model_config=model_config,
                model_path=model_path + "/params",
            )
            policy = create_policy(policy_config, model_path)
            policy_metadata = policy.metadata

            logger.info(f"Pi0 policy loaded successfully for {model_id}")
            logger.info(f"Pi0 policy metadata: {policy_metadata}")

            # Initialize FastAPI app
            app = FastAPI()

            # input_features reflects the model input specifications
            input_features = {}
            input_features[model_specifics.state_key] = {
                "shape": model_specifics.state_size
            }
            for video_key in model_specifics.video_keys:
                input_features[video_key] = {"shape": model_specifics.video_size}

            logger.info(f"Input features: {input_features}")

            @app.get("/health")
            async def health_check():
                return {"status": "ok"}

            @app.post("/get_action")
            async def inference(request: InferenceRequest):
                """Endpoint for Pi0 policy inference."""
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
                            policy=policy,
                            model_specifics=model_specifics,
                            current_qpos=payload[model_specifics.state_key],
                            images=[
                                payload[video_key]
                                for video_key in model_specifics.video_keys
                                if video_key in payload
                            ],
                            image_names=image_names,
                            target_size=target_size,
                            prompt=payload.get("prompt"),
                        )
                    except RetryError as e:
                        return Response(
                            status_code=202,
                            content=str(e),
                        )
                    except Exception as e:
                        logger.error(
                            f"Error during image processing: {e}\nTraceback: {traceback.format_exc()}"
                        )
                        raise HTTPException(
                            status_code=500,
                            detail=f"Error during image processing: {e}",
                        )

                    # Encode response using json_numpy
                    response = json_numpy.dumps(actions)
                    return response

                except Exception as e:
                    raise HTTPException(
                        status_code=500,
                        detail=str(e),
                    )

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
            config = uvicorn.Config(
                app, host="0.0.0.0", port=server_port, log_level="info"
            )
            inference_fastapi_server = uvicorn.Server(config)

            # Run the server until timeout or interruption
            server_start_time = time.time()

            # Calculate remaining time after setup
            setup_time = server_start_time - start_time
            remaining_time = max(timeout - setup_time - 10, 60)  # Ensure at least 60s

            logger.info(
                f"Setup took {setup_time:.2f}s. Server will run for up to {remaining_time:.2f}s"
            )

            try:
                logger.info(f"Starting Inference FastAPI server on port {server_port}")
                # Use the remaining time for the server
                await asyncio.wait_for(
                    inference_fastapi_server.serve(), timeout=remaining_time
                )
            except asyncio.TimeoutError:
                server_runtime = time.time() - server_start_time
                total_runtime = time.time() - start_time
                logger.info(
                    f"Setup time: {setup_time:.2f}s, "
                    f"Server runtime: {server_runtime:.2f}s, "
                    f"Total runtime: {total_runtime:.2f}s. Shutting down."
                )
                # _update_server_status(supabase_client, server_id, "stopped")
            except Exception as e:
                logger.error(f"Server error: {e}")
                # _update_server_status(supabase_client, server_id, "failed")
                raise HTTPException(
                    status_code=500,
                    detail=f"Server error: {e}",
                )
            finally:
                logger.info("Shutting down FastAPI server")
                await inference_fastapi_server.shutdown()

        except HTTPException as e:
            logger.error(f"HTTPException during server setup: {e.detail}")
            # _update_server_status(supabase_client, server_id, "failed")
            raise e

        except Exception as e:
            logger.error(f"Error during server setup: {e}")
            # _update_server_status(supabase_client, server_id, "failed")
            raise HTTPException(
                status_code=500,
                detail=f"Error during server setup: {e}",
            )


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
    volumes={"/data": pi0_volume},
)
def train(
    training_id: int,
    dataset_name: str,
    wandb_api_key: str | None,
    model_name: str,
    training_params: TrainingParamsPi0,
    user_hf_token: str | None = None,
    private_mode: bool = False,
    max_hf_download_retries: int = 3,
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

    SUPABASE_URL = os.environ["SUPABASE_URL"]
    SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    hf_token = user_hf_token or os.getenv("HF_TOKEN")
    if hf_token is None:
        raise ValueError(
            "HF_TOKEN is not available (neither user token nor system token)"
        )

    logger.info(
        f"🚀 Training Pi0.5 on {dataset_name} with id {training_id} and uploading to: {model_name}  (private_mode={private_mode})"
    )

    try:
        # Update training status to running
        supabase_client.table("trainings").update(
            {
                "status": "running",
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
        ).eq("id", training_id).execute()

        # Pi0 training logic would go here
        # For now, this is a placeholder that simulates training
        logger.info("Pi0 training is not yet implemented - this is a placeholder")

        # Simulate some training time
        import time

        time.sleep(10)

        # Update training status to completed
        supabase_client.table("trainings").update(
            {
                "status": "completed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
        ).eq("id", training_id).execute()

        logger.success(f"Pi0 training {training_id} completed successfully")

    except Exception as e:
        logger.error(f"Pi0 training {training_id} failed: {e}")

        # Update training status to failed
        try:
            supabase_client.table("trainings").update(
                {
                    "status": "failed",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "error_message": str(e),
                }
            ).eq("id", training_id).execute()
        except Exception as db_e:
            logger.error(f"Failed to update training status: {db_e}")

        raise e


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


def find_model_path(
    model_id: str, local_base_dir: str = "/data", checkpoint: int | None = None
) -> str | None:
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
    use_base_model: bool = False,
    model_type: str | None = None,
) -> str:
    """
    Downloads a model from HuggingFace if not already present in the local directory.

    Args:
        model_id: The HuggingFace model ID to download
        checkpoint: [Optional] specific checkpoint/revision to download
        local_base_dir: Base directory to store downloaded models
        token: [Optional] HuggingFace token for accessing private models
        use_base_model: Whether to use base model weights (if applicable)
        model_type: [Optional] Type of the model (e.g., "pi0", "pi0_fast", "pi05")

    Returns:
        The local path to the downloaded model

    Raises:
        Exception: If the download fails
    """

    if use_base_model:
        assert use_base_model is True and model_type is not None, (
            "use_base_model must be True and model_type must be specified to use base model weights"
        )

        match model_type:
            case "pi0":
                gcs_path = "gs://openpi-assets/checkpoints/pi0_base"
            case "pi0_fast":
                gcs_path = "gs://openpi-assets/checkpoints/pi0_fast_base"
            case "pi05":
                gcs_path = "gs://openpi-assets/checkpoints/pi05_base"
            case _:
                raise ValueError(f"Unknown model_type: {model_type}")

        local_path = Path(local_base_dir) / Path(
            gcs_path.replace("gs://", "").split("/")[-1]
        )

        # check if local path exists and is not empty
        if local_path.exists() and any(local_path.iterdir()):
            logger.info(
                f"Weights found in modal volume, loading weights from: {local_path}"
            )
        else:
            logger.info(f"Downloading official model weights from {gcs_path}")
            try:
                local_path.mkdir(parents=True, exist_ok=True)
                download_from_gcs(gcs_path, local_path)
                logger.info(
                    f"Successfully downloaded model weights from {gcs_path} to {local_path}"
                )
                return str(local_path)
            except Exception as e:
                logger.error(
                    f"Failed to download model weights from GCS ({gcs_path}): {e}"
                )
                raise Exception(
                    f"Failed to download model weights from GCS ({gcs_path}): {e}"
                )

        return str(local_path)

    # If not using base model weights, proceed with the usual model path finding/downloading
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
            logger.error(
                f"Failed to download model {model_id} with checkpoint {checkpoint}: {e}"
            )
            raise e


def download_from_gcs(gcs_path: str, local_path: Path):
    """
    Downloads files from a Google Cloud Storage (GCS) bucket to a local directory using gsutil.

    Args:
        gcs_path: The GCS path to download from (e.g., "gs://bucket_name/path/to/files")
        local_path: The local directory path to save the downloaded files
    """
    assert gcs_path.startswith("gs://"), "gcs_path must start with 'gs://'"

    path_parts = gcs_path.replace("gs://", "").split("/", 1)
    bucket_name = path_parts[0]
    prefix = path_parts[1] if len(path_parts) > 1 else ""

    # Create the local directory
    local_dir = Path(local_path)
    local_dir.mkdir(parents=True, exist_ok=True)

    # Create a client with anonymous credentials (for public buckets)
    client = GCSClient.create_anonymous_client()

    # Get the bucket
    bucket = client.bucket(bucket_name)

    # List and download all blobs with the given prefix
    blobs = list(bucket.list_blobs(prefix=prefix))

    if blobs is None or len(blobs) == 0:
        logger.warning(f"No files found in {gcs_path}")
        return local_path

    # Download each blob
    for blob in blobs:
        # Remove the prefix to get the relative path
        if prefix:
            relative_path = blob.name[len(prefix) :].lstrip("/")
        else:
            relative_path = blob.name

        # Skip empty paths (the directory itself)
        if not relative_path:
            continue

        # Create subdirectories if needed
        file_path = os.path.join(local_path, relative_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        blob.download_to_filename(file_path)
