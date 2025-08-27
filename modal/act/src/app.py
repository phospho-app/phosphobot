import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import sentry_sdk
import wandb
from fastapi import Response
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import HFValidationError
from loguru import logger

import modal
from phosphobot.am.act import ACTSpawnConfig
from phosphobot.am.base import (
    HuggingFaceTokenValidator,
    TrainingParamsAct,
    TrainingParamsActWithBbox,
    generate_readme,
    resize_dataset,
)
from phosphobot.models import InfoModel
from phosphobot.models.lerobot_dataset import LeRobotDataset


if os.getenv("MODAL_ENVIRONMENT") == "production":
    sentry_sdk.init(
        dsn="https://afa38885e368d772d8eced1bce325604@o4506399435325440.ingest.us.sentry.io/4509203019005952",
        traces_sample_rate=1.0,
        environment="production",
    )

phosphobot_dir = (
    Path(__file__).parent.parent.parent.parent.parent / "phosphobot" / "phosphobot"
)
act_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "libavutil-dev", "libavcodec-dev", "libavformat-dev")
    .pip_install_from_pyproject(
        pyproject_toml=str(phosphobot_dir / "pyproject.toml"),
    )
    .pip_install(
        "lerobot",
        "huggingface_hub[hf_transfer]",
        "hf_xet",
        "wandb",
        "accelerate",
        "einops",
        "torch>=2.2.1",
        "torchvision>=0.21.0",
        "pyarrow>=8.0.0",
        "asyncio",
        "draccus",
        "jsonlines",
        "imageio[ffmpeg]>=2.34.0",
        "zarr>=2.17.0",
        "termcolor>=2.5.0",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .env({"HF_HUB_DISABLE_TELEMETRY": "1"})
    .add_local_python_source("phosphobot")
)

MINUTES = 60  # seconds
HOURS = 60 * MINUTES
FUNCTION_IMAGE = act_image
FUNCTION_TIMEOUT_TRAINING = 3 * HOURS  # 3 hours
FUNCTION_TIMEOUT_INFERENCE = 6 * MINUTES  # 6 minutes
FUNCTION_GPU_TRAINING: list[str | modal.gpu._GPUConfig | None] = ["A10G"]
FUNCTION_GPU_INFERENCE: list[str | modal.gpu._GPUConfig | None] = ["T4"]
FUNCTION_CPU_TRAINING = 20.0
MIN_NUMBER_OF_EPISODES = 10


app = modal.App("act-server")
act_volume = modal.Volume.from_name("act", create_if_missing=True)

paligemma_detect = modal.Function.from_name("paligemma-detector", "detect_object")


def find_model_path(model_id: str, checkpoint: int | None = None) -> str | None:
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


async def run_act_training(
    dataset_name: str,
    dataset_path: str,
    training_params: TrainingParamsAct,
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
        "--policy.type=act",
        "--policy.push_to_hub=false",
        f"--batch_size={training_params.batch_size}",
        "--wandb.project=phosphobot-ACT",
        f"--save_freq={training_params.save_steps}",
        f"--steps={training_params.steps}",
        "--policy.device=cuda",
        f"--output_dir={output_dir}",
        f"--wandb.run_id={wandb_run_id}",
        f"--wandb.enable={str(wandb_enabled).lower()}",
    ]

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
    gpu=FUNCTION_GPU_INFERENCE,
    timeout=FUNCTION_TIMEOUT_INFERENCE,
    secrets=[
        modal.Secret.from_dict({"MODAL_LOGLEVEL": "DEBUG"}),
        modal.Secret.from_name("supabase"),
    ],
    volumes={"/data": act_volume},
)
async def serve(
    model_id: str,
    server_id: int,
    model_specifics: ACTSpawnConfig,
    checkpoint: int | None = None,
    timeout: int = FUNCTION_TIMEOUT_INFERENCE,
    q=None,
):
    """
    model_id: str
    server_id: str, used to update the server status in the database
    timeout: int
    q: Optional[modal.Queue], used to pass tunnel info back to caller (since the function is running in a different process)
    """
    import asyncio
    import time
    from datetime import datetime, timezone

    import json_numpy
    import torch
    import torch.nn as nn
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from huggingface_hub import snapshot_download  # type: ignore
    from pydantic import BaseModel

    from lerobot.policies.act.modeling_act import ACTPolicy
    from supabase import Client, create_client

    class RetryError(Exception):
        """Custom exception for retrying the request."""

        pass

    # Start timer
    start_time = time.time()

    SUPABASE_URL = os.environ["SUPABASE_URL"]
    SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    # logger.info the region
    logger.success(f"🌎 running in {os.environ['MODAL_REGION']} region")

    server_port = 80

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

    with modal.forward(server_port, unencrypted=True) as tunnel:
        model_path = find_model_path(model_id=model_id, checkpoint=checkpoint)

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

        try:
            policy = ACTPolicy.from_pretrained(model_path).to(device="cuda")
            assert isinstance(policy, nn.Module)
            logger.info("Policy loaded successfully")
            policy.eval()

            # Initialize FastAPI app
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
            last_bbox_computed: list[float] | None = None

            logger.info(f"Input features: {input_features}")

            def process_image(
                current_qpos: list[float],
                images: list[np.ndarray],
                image_names: list[str],
                target_size: tuple[int, int],
                image_for_bboxes: torch.Tensor | None,
                detect_instruction: str | None = None,
            ) -> np.ndarray:
                """
                Process images and perform inference using the policy.
                This is a placeholder; replace with your actual process_image implementation.
                """
                nonlocal last_bbox_computed
                nonlocal policy

                assert (
                    len(current_qpos) == model_specifics.state_size[0]
                ), f"State size mismatch: {len(current_qpos)} != {model_specifics.state_size[0]}"
                assert (
                    len(images) <= len(model_specifics.video_keys)
                ), f"Number of images {len(images)} is more than the number of video keys {len(model_specifics.video_keys)}"
                if len(images) > 0:
                    assert (
                        len(images[0].shape) == 3
                    ), f"Image shape is not correct, {images[0].shape} expected (H, W, C)"
                    assert (
                        len(images[0].shape) == 3 and images[0].shape[2] == 3
                    ), f"Image shape is not correct {images[0].shape} expected (H, W, 3)"

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

                    # Add the bboxes to the batch if needed
                    if model_specifics.env_key is not None:
                        bboxes = paligemma_detect.remote(
                            # We add the batch dimension to the image_for_bboxes, which is B=1 here
                            frames=np.array([image_for_bboxes]),
                            instructions=[detect_instruction],
                        )
                        # For now we delete the batch dimension to stay compatible with the old code
                        bboxes = bboxes[0]
                        if bboxes == [0.0, 0.0, 0.0, 0.0]:
                            # We want to let the client know that he needs to retry with a new image
                            if last_bbox_computed is None:
                                raise RetryError(
                                    f"The object '{detect_instruction}' was not detected in the selected camera. Try with a different instruction or camera."
                                )
                            # Otherwise, we use the last computed bounding boxes
                            logger.debug(
                                f"No bounding boxes detected, using last computed: {last_bbox_computed}"
                            )
                            bboxes = last_bbox_computed
                        else:
                            logger.info(f"Detected bounding boxes: {bboxes}")

                        # last_bbox_computed = bboxes
                        if last_bbox_computed is None:
                            last_bbox_computed = bboxes
                        else:
                            # Do a rolling average of the last 10 bboxes
                            last_bbox_computed = [
                                (last_bbox_computed[i] * 9 + bboxes[i]) / 10
                                for i in range(len(bboxes))
                            ]

                        batch[model_specifics.env_key] = torch.tensor(
                            last_bbox_computed, dtype=torch.float32, device="cuda"
                        ).view(1, -1)

                    processed_images = []
                    for i, image in enumerate(images):
                        # TODO: Double check if image.shape[:2] is (H, W) or (W, H)
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

                    # We process the batch
                    batch = policy.normalize_inputs(batch)  # type: ignore
                    if policy.config.image_features:  # type: ignore
                        batch = dict(batch)
                        batch["observation.images"] = [
                            batch[key]
                            for key in policy.config.image_features  # type: ignore
                        ]
                    actions = policy.model(batch)[0][:, : policy.config.n_action_steps]  # type: ignore
                    actions = policy.unnormalize_outputs({"action": actions})["action"]  # type: ignore
                    actions = actions.transpose(0, 1)
                    return actions.cpu().numpy()

            class InferenceRequest(BaseModel):
                encoded: str  # Will contain json_numpy encoded payload with image

            @app.post("/act")
            async def inference(request: InferenceRequest):
                """Endpoint for ACT policy inference."""
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

                    if model_specifics.env_key is not None and (
                        "detect_instruction" not in payload
                        or "image_for_bboxes" not in payload
                    ):
                        logger.error(
                            f"'detect_instruction' or 'image_for_bboxes' not found in payload, got: {payload.keys()}"
                        )
                        raise ValueError(
                            f"'detect_instruction' or 'image_for_bboxes' required in payload, got {payload.keys()}"
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
                            current_qpos=payload[model_specifics.state_key],
                            images=[
                                payload[video_key]
                                for video_key in model_specifics.video_keys
                                if video_key in payload
                            ],
                            image_names=image_names,
                            target_size=target_size,
                            image_for_bboxes=payload.get("image_for_bboxes", None),
                            detect_instruction=payload.get("detect_instruction", None),
                        )
                    except RetryError as e:
                        return Response(
                            status_code=202,
                            content=str(e),
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


@app.function(
    image=FUNCTION_IMAGE,
    gpu=FUNCTION_GPU_TRAINING,
    # 20 minutes added for the rest of the code to execute
    timeout=FUNCTION_TIMEOUT_TRAINING + 20 * MINUTES,
    secrets=[
        modal.Secret.from_dict({"MODAL_LOGLEVEL": "DEBUG"}),
        modal.Secret.from_name("supabase"),
        modal.Secret.from_name("huggingface"),
    ],
    volumes={"/data": act_volume},
    cpu=FUNCTION_CPU_TRAINING,
)
def train(  # All these args should be verified in phosphobot
    training_id: int,
    dataset_name: str,
    wandb_api_key: str | None,
    model_name: str,
    training_params: TrainingParamsAct | TrainingParamsActWithBbox,
    user_hf_token: str | None = None,
    private_mode: bool = False,
    max_hf_download_retries: int = 3,
    timeout_seconds: int = FUNCTION_TIMEOUT_TRAINING,
    wandb_run_id: str = "wandb_run_id_not_set",
    **kwargs,
):
    from datetime import datetime, timezone
    from supabase import Client, create_client
    from .helper import NotEnoughBBoxesError, InvalidInputError

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

    try:
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
                f"The provided HF token does not have write access to {dataset_name}"
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

        # If doing ACT with bboxes, calculate the bounding boxes and remove the episodes we fail to detect from the dataset
        if isinstance(training_params, TrainingParamsActWithBbox):
            from .helper import compute_bboxes

            logger.info(
                f"Computing bounding boxes for dataset {dataset_name}, this should take about 5 minutes.."
            )
            dataset_path, number_of_valid_episodes = compute_bboxes(
                dataset_root_path=dataset_path,
                detect_instruction=training_params.target_detection_instruction,
                image_key=training_params.image_key,
                dataset_name=dataset_name,
                image_keys_to_keep=training_params.image_keys_to_keep,
            )
            logger.success(f"Bounding boxes computed and saved to {dataset_path}")

            # Use user's namespace for private training, phospho-app for public
            if private_mode and user_hf_token:
                # Get username from HF token for private training
                hf_api_temp = HfApi(token=user_hf_token)
                try:
                    user_info = hf_api_temp.whoami()
                    username = user_info.get("name")
                    if username:
                        dataset_name = f"{username}/{dataset_path.name}"
                    else:
                        logger.warning(
                            "Could not get username from HF token, using phospho-app namespace"
                        )
                        dataset_name = "phospho-app/" + dataset_path.name
                except Exception as e:
                    logger.warning(
                        f"Error getting username from HF token: {e}, using phospho-app namespace"
                    )
                    dataset_name = "phospho-app/" + dataset_path.name
            else:
                dataset_name = "phospho-app/" + dataset_path.name

            logger.info(f"Uploading dataset {dataset_name} to Hugging Face")
            hf_api = HfApi(token=hf_token)
            hf_api.create_repo(
                repo_type="dataset",
                repo_id=dataset_name,
                token=hf_token,
                exist_ok=True,
                private=private_mode,
            )
            hf_api.upload_folder(
                repo_type="dataset",
                folder_path=str(dataset_path),
                repo_id=dataset_name,
                token=hf_token,
            )
            hf_api.create_branch(
                repo_id=dataset_name,
                repo_type="dataset",
                branch="v2.0",
                token=True,
                exist_ok=True,
            )
            hf_api.upload_folder(
                repo_type="dataset",
                folder_path=str(dataset_path),
                repo_id=dataset_name,
                token=hf_token,
                revision="v2.0",
            )
            logger.success(
                f"Dataset {dataset_name} uploaded to Hugging Face successfully"
            )

            if number_of_valid_episodes < MIN_NUMBER_OF_EPISODES:
                visualizer_url = (
                    f"https://lerobot-visualize-dataset.hf.space/{dataset_name}"
                )
                raise RuntimeError(
                    f"The object '{training_params.target_detection_instruction}' was detected in {number_of_valid_episodes} episodes in {training_params.image_key} camera"
                    f" (should be: {MIN_NUMBER_OF_EPISODES} episodes min)."
                    f" This is not enough to train a model. Check your dataset: {visualizer_url} and rephrase the instruction."
                )

        else:
            # Normal ACT: Resize the dataset to 320x240 otherwise there are too many Cuda OOM errors
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
                run_act_training(
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
            model_type="act",
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

        readme = generate_readme(
            model_type="act",
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
        logger.error(f"🚨 ACT Training {training_id} for {dataset_name} failed: {e}")

        supabase_client.table("trainings").update(
            {
                "status": "failed",
                "terminated_at": datetime.now(timezone.utc).isoformat(),
            }
        ).eq("id", training_id).execute()

        readme = generate_readme(
            model_type="act",
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
