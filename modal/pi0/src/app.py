import os
from pathlib import Path

import sentry_sdk
from loguru import logger
from typing import Optional

import modal
import socket
import sys
sys.path.insert(0, "/Users/ashish/git-repos/phosphobot/phosphobot")

from phosphobot.am.pi0 import Pi0SpawnConfig
# from phosphobot.am.base import TrainingParamsPi0
# from phosphobot.models import InfoModel
# from modal.utils import update_server_status

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
        modal.Secret.from_name("supabase"),
    ],
    volumes={"/data": pi0_volume},
)
async def serve(
    model_id: str,
    server_id: int,
    model_specifics: Pi0SpawnConfig,
    timeout: int = FUNCTION_TIMEOUT_INFERENCE,
    q: Optional[modal.Queue] = None,
):
    """
    Pi0 inference server function.

    Args:
        model_id: The model identifier from HuggingFace or local path
        server_id: Database server ID for status tracking
        model_specifics: Pi0-specific configuration
        timeout: Timeout in seconds
        q: Modal queue to pass tunnel info back to caller (since the function is running in a different process)
    """
    # from datetime import datetime, timezone

    # import json_numpy
    # import uvicorn
    # from fastapi import FastAPI, HTTPException
    # from pydantic import BaseModel

    # import shutil
    # import time

    # from huggingface_hub import snapshot_download  # type: ignore
    # from supabase import Client, create_client

    # SUPABASE_URL = os.environ["SUPABASE_URL"]
    # SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    # supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    logger.info(f"🚀 Starting Pi0 server for model {model_id}")

    # Initialize Pi0 model
    try:
        from phosphobot.am.pi0 import create_policy, Pi0WebSocketServer
        from openpi.policies import policy as _policy

        # For Pi0, the server runs on a separate process/container
        # We create a websocket client that connects to the Pi0 server
        # pi0_policy = Pi0(
        #     server_url=model_specifics.server_url,
        #     server_port=model_specifics.server_port,
        #     image_keys=model_specifics.image_keys,
        # )

        policy = create_policy(model_specifics)
        policy_metadata = policy.metadata

        # Record the policy's behavior.
        if model_specifics.record:
            policy = _policy.PolicyRecorder(policy, "policy_records")

        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        logger.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

        server = Pi0WebSocketServer(
            policy=policy,
            host="0.0.0.0",
            port=model_specifics.port,
            metadata=policy_metadata,
        )
        server.start()

        logger.info(f"Pi0 model initialized for {model_id}")

    except Exception as e:
        logger.error(f"Failed to start Pi0 server with model: {model_id}.\nError:{e}")
        # Update server status in database
        # try:
        #     supabase_client.table("servers").update({
        #         "status": "error",
        #         "terminated_at": datetime.now(timezone.utc).isoformat(),
        #     }).eq("id", server_id).execute()
        # except Exception as db_e:
        #     logger.error(f"Failed to update server status: {db_e}")
        # raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    # Create FastAPI app for inference
    # web_app = FastAPI()

    # @web_app.get("/")
    # async def root():
    #     return {"message": f"Pi0 server running for model {model_id}"}

    # @web_app.get("/info")
    # async def get_info():
    #     """Get model information."""
    #     return InfoModel(
    #         model_id=model_id,
    #         model_type="pi0",
    #         config=model_specifics.model_dump(),
    #         status="running",
    #     )

    # class InferenceRequest(BaseModel):
    #     encoded: str  # JSON-encoded payload with images and state

    # @web_app.post("/pi0")
    # async def inference(request: InferenceRequest):
    #     """Pi0 inference endpoint."""
    #     nonlocal pi0_policy

    #     if pi0_policy is None:
    #         raise HTTPException(status_code=500, detail="Pi0 policy not loaded")

    #     try:
    #         # Decode the payload
    #         payload: dict = json_numpy.loads(request.encoded)

    #         # Validate required keys
    #         required_keys = ["images", "state", "prompt"]
    #         missing_keys = [key for key in required_keys if key not in payload]
    #         if missing_keys:
    #             raise ValueError(f"Missing required keys: {missing_keys}")

    #         # Call Pi0 inference
    #         actions = pi0_policy.sample_actions(payload)

    #         # Encode response
    #         response_data = {"actions": actions}
    #         encoded_response = json_numpy.dumps(response_data)

    #         return {"encoded": encoded_response}

    #     except Exception as e:
    #         logger.error(f"Pi0 inference error: {e}")
    #         raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    # # Start the server using Modal's built-in tunnel
    # server_port = 8000

    # with modal.forward(server_port, unencrypted=True) as tunnel:
    #     logger.info(f"tunnel.tcp_socket = {tunnel.tcp_socket}")

    #     # Update server info in database
    #     try:
    #         server_info = {
    #             "url": tunnel.url,
    #             "port": server_port,
    #             "host": tunnel.tcp_socket[0],
    #             "tcp_port": tunnel.tcp_socket[1],
    #             "region": "modal",
    #         }

    #         # supabase_client.table("servers").update(server_info).eq("id", server_id).execute()
    #         logger.info(f"server_info: {server_info}")

    #         # Send server info through queue if provided
    #         if q is not None:
    #             from phosphobot.models import ServerInfo
    #             q.put(ServerInfo(
    #                 server_id=server_id,
    #                 url=tunnel.url,
    #                 port=server_port,
    #                 tcp_socket=tunnel.tcp_socket,
    #                 model_id=model_id,
    #                 timeout=timeout,
    #             ).model_dump())

    #         logger.info(f"Pi0 server started at {tunnel.url}")

    #     except Exception as e:
    #         logger.error(f"Failed to update server info: {e}")
    #         raise

    #     # Start the web server
    #     config = uvicorn.Config(
    #         app=web_app,
    #         host="0.0.0.0",
    #         port=server_port,
    #         log_level="info",
    #     )
    #     server = uvicorn.Server(config)
    #     await server.serve()


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
