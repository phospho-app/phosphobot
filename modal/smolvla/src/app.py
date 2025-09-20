import asyncio
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal
import sentry_sdk
import uvicorn
from fastapi import FastAPI, HTTPException, Response
from loguru import logger

from phosphobot.am.act import RetryError
from phosphobot.am.smolvla import SmolVLASpawnConfig
from .helper import _find_or_download_model, InferenceRequest, process_image


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
    modal.Image.from_dockerfile("Dockerfile")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HUB_DISABLE_TELEMETRY": "1"})
    .pip_install_from_pyproject(pyproject_toml=str(phosphobot_dir / "pyproject.toml"))
    .add_local_python_source("phosphobot")
)

app = modal.App("smolvla-server")
volume = modal.Volume.from_name("smolvla", create_if_missing=True)

MINUTES = 60
FUNCTION_IMAGE = smolvla_image
FUNCTION_TIMEOUT_INFERENCE = 6 * MINUTES  # 6 minutes
FUNCTION_GPU_INFERENCE: list[str | modal.gpu._GPUConfig | None] = ["T4"]

# ======== Inference Function ========
@app.function(
    image=FUNCTION_IMAGE,
    gpu=FUNCTION_GPU_INFERENCE,
    timeout=FUNCTION_TIMEOUT_INFERENCE,
    volumes={"/data": volume},
    # secrets=[modal.Secret.from_name("supabase")],
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
    from lerobot.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK  # type: ignore
    import json_numpy  # type: ignore
    import torch.nn as nn  # type: ignore

    # Start timer
    start_time = time.time()

    # logger.info the region
    logger.success(f"🌎 running in {os.environ['MODAL_REGION']} region")

    try:
        model_path = _find_or_download_model(model_id)
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
        last_bbox_computed: list[float] | None = None

        logger.info(f"Input features: {input_features}")

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
                        prompt=payload.get("prompt", None),
                        prompt_key=OBS_LANGUAGE_TOKENS,
                        prompt_mask_key=OBS_LANGUAGE_ATTENTION_MASK,
                    )
                except RetryError as e:
                    return Response(
                        status_code=202,
                        content=str(e),
                    )

                # Encode response using json_numpy
                response = json_numpy.dumps(actions)
                return response

                """
                state = np.array(payload[model_specifics.state_key])
                images = [np.array(img) for img in payload.get("images", [])]

                with torch.no_grad(), torch.autocast(device_type="cuda"):
                    batch: dict[str, Any] = {
                        model_specifics.state_key: torch.from_numpy(state).view(1, -1).float().cuda()
                    }
                    for i, key in enumerate(model_specifics.video_keys[: len(images)]):
                        img = images[i]
                        if img.shape[:2] != tuple(model_specifics.video_size[-2:]):
                            img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float().cuda()  # C,H,W
                        else:
                            img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float().cuda()
                        batch[key] = img
                    if model_specifics.env_key and model_specifics.env_key in payload:
                        env = np.array(payload[model_specifics.env_key])
                        batch[model_specifics.env_key] = torch.from_numpy(env).view(1, -1).float().cuda()

                    action = policy(batch)[0].cpu().numpy()
                return json_numpy.dumps(action)
                """
            except RetryError as e:
                raise HTTPException(status_code=202, detail=str(e))
            except Exception as e:
                logger.error(e)
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
            except Exception as e:
                logger.error(f"Server error: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Server error: {e}",
                )
            finally:
                logger.info("Shutting down FastAPI server")
                await inference_fastapi_server.shutdown()

    except HTTPException as e:
        logger.error(f"HTTPException during server setup: {e.detail}")
        raise e

    except Exception as e:
        logger.error(f"Error during server setup: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error during server setup: {e}",
        )
