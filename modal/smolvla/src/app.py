import asyncio
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from huggingface_hub import snapshot_download  # type: ignore
from loguru import logger
from pydantic import BaseModel

from phosphobot.am.smolvla import SmolVLASpawnConfig

# -------- Modal image ---------------------------------------------------
phosphobot_dir = (
    Path(__file__).parent.parent.parent.parent.parent / "phosphobot" / "phosphobot"
)

smolvla_image = (
    modal.Image.from_dockerfile("Dockerfile")
    # core deps (similar to ACT)
    .pip_install(
        "loguru",
        "supabase",
        "sentry-sdk",
        "huggingface_hub[hf_transfer]",
        "hf_xet",
        "wandb",
        "accelerate",
        "httpx>=0.28.1",
        "pydantic>=2.10.5",
        "fastparquet>=2024.11.0",
        "numpy<2",
        "opencv-python-headless>=4.0",
        "rich>=13.9.4",
        "pandas>=2.2.2.240807",
        "json-numpy>=2.1.0",
        "fastapi>=0.115.11",
        "torch>=2.2.1",
        "torchvision>=0.21.0",
        "pyarrow>=8.0.0",
        "uvicorn",
        "lerobot[smolvla]",  # pulls SmolVLA extra
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HUB_DISABLE_TELEMETRY": "1"})
    .pip_install_from_pyproject(pyproject_toml=str(phosphobot_dir / "pyproject.toml"))
    .add_local_python_source("phosphobot")
)

app = modal.App("smolvla-server")
volume = modal.Volume.from_name("smolvla", create_if_missing=True)

MINUTES = 60
FUNCTION_IMAGE = smolvla_image
FUNCTION_TIMEOUT = 6 * MINUTES
GPU_TYPE: list[str | modal.gpu._GPUConfig | None] = ["T4"]

# -------- Helper -------------------------------------------------------

def _find_or_download_model(model_id: str) -> str:
    model_dir = Path(f"/data/{model_id}/")
    if model_dir.exists():
        latest = max([d for d in model_dir.iterdir() if d.is_dir()], default=None)
        if latest is not None:
            p = model_dir / latest / "checkpoints" / "last" / "pretrained_model"
            if p.exists():
                return str(p)
    # download
    ts = str(datetime.now(timezone.utc).timestamp())
    local_dir = f"/data/{model_id}/{ts}/checkpoints/last/pretrained_model"
    path = snapshot_download(
        repo_id=model_id,
        repo_type="model",
        revision="main",
        local_dir=local_dir,
        token=os.getenv("HF_TOKEN"),
    )
    return path

# -------- Serve function ----------------------------------------------

@app.function(
    image=FUNCTION_IMAGE,
    gpu=GPU_TYPE,
    timeout=FUNCTION_TIMEOUT,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("supabase")],
)
async def serve(
    model_id: str,
    server_id: int,
    model_specifics: SmolVLASpawnConfig,
    timeout: int = FUNCTION_TIMEOUT,
    q=None,
):
    """Launches a FastAPI SmolVLA inference server and exposes /act endpoint."""

    from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # type: ignore
    import json_numpy  # type: ignore

    class RetryError(Exception):
        pass

    model_path = _find_or_download_model(model_id)
    logger.success(f"✅ Model ready at {model_path}")

    policy = SmolVLAPolicy.from_pretrained(model_path).to(device="cuda")
    policy.eval()

    api = FastAPI()

    class InferenceRequest(BaseModel):
        encoded: str  # json_numpy encoded dict

    @api.post("/act")
    async def inference(req: InferenceRequest):
        try:
            payload = json_numpy.loads(req.encoded)
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
        except RetryError as e:
            raise HTTPException(status_code=202, detail=str(e))
        except Exception as e:
            logger.error(e)
            raise HTTPException(status_code=500, detail=str(e))

    # Expose through tunnel
    port = 80
    with modal.forward(port, unencrypted=True) as tunnel:
        if q is not None:
            await q.put({"url": tunnel.url, "port": port})
        config = uvicorn.Config(api, host="0.0.0.0", port=port, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()
