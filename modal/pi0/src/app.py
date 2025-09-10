import os
from pathlib import Path
import asyncio

import sentry_sdk
from loguru import logger
from typing import Optional

import modal
import sys
# TODO: remove this
sys.path.insert(0, "/Users/ashish/git-repos/phosphobot/phosphobot")

from phosphobot.models import InfoModel
from phosphobot.am.pi0 import Pi0SpawnConfig, Pi0, RetryError
from phosphobot.am.base import TrainingParamsPi0
from ..utils import _update_server_status, get_model_path, _upload_partial_checkpoint, InferenceRequest

if os.getenv("MODAL_ENVIRONMENT") == "production":
    sentry_sdk.init(
        dsn="https://afa38885e368d772d8eced1bce325604@o4506399435325440.ingest.us.sentry.io/4509203019005952",
        traces_sample_rate=1.0,
        environment="production",
    )

phosphobot_dir = (
    Path(__file__).parent.parent.parent.parent.parent / "phosphobot" / "phosphobot"
)
pi0_image = (
    modal.Image.from_dockerfile("policy_server.Dockerfile")
    # .env({"PATH": "/.venv/bin:$PATH"})  # Use the virtual environment's Python
    .pip_install_from_pyproject(
        pyproject_toml=str(phosphobot_dir / "pyproject.toml"),
    )
    .pip_install(
        "sentry-sdk",
        "loguru>=0.7.3",
        "pydantic>=2.10.5",
        # "pydandtic==2.10.6",
        # "numpydantic==1.6.7",
        "numpy==1.26.4",
        # "numpy<2",
        "supabase",
        "huggingface_hub[hf_transfer]",
        "hf_xet",
        "wandb",
        # "accelerate",
        "httpx>=0.28.1",
        "fastparquet>=2024.11.0",
        "opencv-python-headless>=4.0",
        "rich>=13.9.4",
        "pandas-stubs>=2.2.2.240807",
        "json-numpy>=2.1.0",
        "fastapi>=0.115.11",
        "zmq>=0.0.0",
        "av>=14.2.1",
        # "openpi_client @ git+https://github.com/phospho-app/openpi.git",
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
FUNCTION_TIMEOUT_INFERENCE = 60 * 5  # 5 minutes
FUNCTION_GPU_TRAINING: list[str | modal.gpu._GPUConfig | None] = ["A100-80GB"]
FUNCTION_GPU_INFERENCE: list[str | modal.gpu._GPUConfig | None] = ["A100-40GB", "L40S"]

app = modal.App("pi0-server")
pi0_volume = modal.Volume.from_name("pi0-volume", create_if_missing=True)


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
    import json_numpy
    import uvicorn
    from fastapi import FastAPI, HTTPException, Response
    from pydantic import BaseModel
    import numpy as np

    from phosphobot.am.pi0 import create_policy, format_observations_for_policy
    from openpi.policies import policy as _policy

    # from huggingface_hub import snapshot_download  # type: ignore
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
        model_path = get_model_path(model_id=model_id, checkpoint=checkpoint)

        try:
            # TODO: Add model weights caching (in modal)
            policy = create_policy(model_path)
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
                        actions = Pi0.process_image(
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


# @app.function(
#     image=FUNCTION_IMAGE,
#     gpu=FUNCTION_GPU_TRAINING,
#     # 10 extra minutes to make sure the rest of the pipeline is done
#     timeout=FUNCTION_TIMEOUT_TRAINING + 10 * MINUTES,
#     secrets=[
#         modal.Secret.from_dict({"MODAL_LOGLEVEL": "DEBUG"}),
#         modal.Secret.from_name("supabase"),
#         modal.Secret.from_name("huggingface"),
#     ],
#     volumes={"/data": pi0_volume},
# )
# def train(
#     training_id: int,
#     dataset_name: str,
#     wandb_api_key: str | None,
#     model_name: str,
#     training_params: TrainingParamsPi0,
#     max_hf_download_retries: int = 3,
#     timeout_seconds: int = FUNCTION_TIMEOUT_TRAINING,
#     **kwargs,
# ):
#     """
#     Pi0 training function.

#     This is a placeholder implementation. The actual training logic
#     would depend on the Pi0 model training requirements.
#     """
#     from datetime import datetime, timezone
#     from supabase import Client, create_client

#     SUPABASE_URL = os.environ["SUPABASE_URL"]
#     SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
#     supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

#     hf_token = os.getenv("HF_TOKEN")
#     if hf_token is None:
#         raise ValueError("HF_TOKEN environment variable is not set")

#     logger.info(f"🚀 Starting Pi0 training for dataset {dataset_name} with id {training_id}")

#     try:
#         # Update training status to running
#         supabase_client.table("trainings").update({
#             "status": "running",
#             "started_at": datetime.now(timezone.utc).isoformat(),
#         }).eq("id", training_id).execute()

#         # Pi0 training logic would go here
#         # For now, this is a placeholder that simulates training
#         logger.info("Pi0 training is not yet implemented - this is a placeholder")

#         # Simulate some training time
#         import time
#         time.sleep(10)

#         # Update training status to completed
#         supabase_client.table("trainings").update({
#             "status": "completed",
#             "completed_at": datetime.now(timezone.utc).isoformat(),
#         }).eq("id", training_id).execute()

#         logger.success(f"Pi0 training {training_id} completed successfully")

#     except Exception as e:
#         logger.error(f"Pi0 training {training_id} failed: {e}")

#         # Update training status to failed
#         try:
#             supabase_client.table("trainings").update({
#                 "status": "failed",
#                 "completed_at": datetime.now(timezone.utc).isoformat(),
#                 "error_message": str(e),
#             }).eq("id", training_id).execute()
#         except Exception as db_e:
#             logger.error(f"Failed to update training status: {db_e}")

#         raise e
