import asyncio
import os
import platform
import time
from typing import AsyncGenerator, cast

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse
from huggingface_hub import HfApi
from loguru import logger
from supabase_auth.types import Session as SupabaseSession

from phosphobot.am.base import TrainingParamsGr00T, TrainingRequest
from phosphobot.models import (
    CancelTrainingRequest,
    CustomTrainingRequest,
    InfoModel,
    StartTrainingResponse,
    StatusResponse,
    SupabaseTrainingModel,
    TrainingsList,
)
from phosphobot.supabase import get_client, user_is_logged_in
from phosphobot.utils import get_hf_token, get_home_app_path, get_tokens

router = APIRouter(tags=["training"])


@router.post("/training/models/read", response_model=TrainingsList)
async def get_models(
    session: SupabaseSession = Depends(user_is_logged_in),
) -> TrainingsList:
    """Get the list of models with aggregated AI control session metrics"""
    client = await get_client()
    user_id = session.user.id

    # Run this SQL query to create the function in Supabase:
    # -- Switch to the public schema
    # SET search_path = public;

    # DROP FUNCTION IF EXISTS public.get_models_with_metrics(uuid, integer);

    # CREATE OR REPLACE FUNCTION public.get_models_with_metrics(p_user_id uuid, p_limit integer DEFAULT 1000)
    # RETURNS TABLE (
    # id               integer,
    # status           text,
    # user_id          uuid,
    # dataset_name     text,
    # model_name       text,
    # requested_at     timestamp,
    # terminated_at    timestamp,
    # used_wandb       boolean,
    # model_type       text,
    # training_params  jsonb,
    # session_count    bigint,
    # success_rate     double precision
    # )
    # LANGUAGE sql
    # SECURITY INVOKER
    # SET search_path = 'public'
    # AS $$
    # WITH recent_trainings AS (
    #     SELECT
    #     id,
    #     status,
    #     user_id,
    #     dataset_name,
    #     model_name,
    #     requested_at,
    #     terminated_at,
    #     used_wandb,
    #     model_type,
    #     training_params
    #     FROM public.trainings
    #     WHERE user_id = p_user_id
    #     ORDER BY requested_at DESC
    #     LIMIT p_limit
    # ), stats AS (
    #     SELECT
    #     model_id,
    #     COUNT(*)                          AS session_count,
    #     SUM(CASE WHEN feedback IS NOT NULL THEN 1 ELSE 0 END) AS feedback_given,
    #     SUM(CASE WHEN feedback = 'positive' THEN 1 ELSE 0 END) AS positive_count
    #     FROM public.ai_control_sessions
    #     WHERE user_id = p_user_id
    #     GROUP BY model_id
    # )
    # SELECT
    #     t.id,
    #     t.status,
    #     t.user_id,
    #     t.dataset_name,
    #     t.model_name,
    #     t.requested_at,
    #     t.terminated_at,
    #     t.used_wandb,
    #     t.model_type,
    #     t.training_params,
    #     COALESCE(s.session_count, 0)      AS session_count,
    #     CASE
    #     WHEN COALESCE(s.feedback_given, 0) = 0 THEN 0.0
    #     ELSE s.positive_count::double precision / s.feedback_given
    #     END                                AS success_rate
    # FROM recent_trainings t
    # LEFT JOIN stats s
    #     ON t.model_name = s.model_id
    # ORDER BY t.requested_at DESC;
    # $$;

    result = await client.rpc(
        "get_models_with_metrics",
        {"p_user_id": user_id, "p_limit": 1000},
    ).execute()
    trainings = result.data or []

    # Validate and return
    models = [SupabaseTrainingModel.model_validate(item) for item in trainings]
    return TrainingsList(models=models)


@router.post(
    "/training/start",
    response_model=StartTrainingResponse,
    summary="Start training a model",
    description="Start training an ACT or gr00t model on the specified dataset. This will upload a trained model to the Hugging Face Hub using the main branch of the specified dataset.",
)
async def start_training(
    request: TrainingRequest,
    session: SupabaseSession = Depends(user_is_logged_in),
) -> StartTrainingResponse | HTTPException:
    """
    Trigger training for a gr00t or ACT model on the specified dataset.

    This endpoint now relies on the TrainingRequest model for all data validation and preparation,
    focusing solely on business logic and orchestration.
    """
    logger.debug(f"Validated and prepared training request: {request}")

    # Business Logic: Validate user permissions for private training.
    if request.private_mode:
        from phosphobot.endpoints.auth import check_pro_user

        is_pro = await check_pro_user(session.user.id)
        if not is_pro:
            raise HTTPException(
                status_code=403,
                detail="Private training is only available for PRO users.",
            )
        # We can also add a final check that the token is present, though the model should guarantee it.
        if not request.user_hf_token:
            raise HTTPException(
                status_code=400,
                detail="Private training requires a valid HF token in your settings.",
            )

    # Configuration & Environment Logic: Check for external tokens/configs.
    tokens = get_tokens()
    if not tokens.MODAL_API_URL:
        raise HTTPException(
            status_code=400,
            detail="Modal API URL not found. Server configuration is incomplete.",
        )

    wandb_token_path = str(get_home_app_path() / "wandb.token")
    if os.path.exists(wandb_token_path):
        logger.debug("WandB token found. Adding to the training request.")
        with open(wandb_token_path, "r") as f:
            request.wandb_api_key = f.read().strip()

    # Deep Validation Logic: Check dataset integrity beyond simple access.
    hf_api = HfApi(token=request.user_hf_token or get_hf_token())
    try:
        info_file_path = hf_api.hf_hub_download(
            repo_id=request.dataset_name,
            repo_type="dataset",
            filename="meta/info.json",
        )
        meta_folder_path = os.path.dirname(info_file_path)
        validated_info_model = InfoModel.from_json(meta_folder_path=meta_folder_path)
        if validated_info_model.total_episodes < 10:
            raise HTTPException(
                status_code=400,
                detail="The dataset must have at least 10 episodes to be used for training.",
            )
    except Exception as e:
        logger.warning(
            f"Error accessing dataset info for '{request.dataset_name}': {e}"
        )
        raise HTTPException(
            status_code=404,
            detail=f"Failed to download or parse 'meta/info.json' from dataset '{request.dataset_name}'. Please ensure the dataset is valid. Error: {e}",
        )

    # Specific validation for gr00t models
    if request.model_type == "gr00t":
        training_params = cast(TrainingParamsGr00T, request.training_params)
        if training_params.validation_dataset_name:
            try:
                hf_api.hf_hub_download(
                    repo_id=training_params.validation_dataset_name,
                    repo_type="dataset",
                    filename="meta/info.json",
                )
            except Exception as e:
                logger.warning(f"Error accessing validation dataset info: {e}")
                raise HTTPException(
                    status_code=404,
                    detail=f"Failed to access the validation dataset '{training_params.validation_dataset_name}'. Error: {e}",
                )

    # Orchestration: Send the prepared request to the training service.
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            training_request_body = request.model_dump()
            response = await client.post(
                f"{tokens.MODAL_API_URL}/train",
                json=training_request_body,
                headers={"Authorization": f"Bearer {session.access_token}"},
            )
            response.raise_for_status()  # Raises HTTPStatusError for 4xx/5xx responses
        except httpx.HTTPStatusError as e:
            # Handle specific error codes from the training service
            if e.response.status_code == 401:
                raise HTTPException(
                    status_code=401,
                    detail="Authentication with training service failed. Please login again.",
                )
            if e.response.status_code == 429:
                detail = e.response.json().get("detail", e.response.text)
                raise HTTPException(status_code=429, detail=detail)
            if e.response.status_code == 422:
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid training parameters for the backend: {e.response.text}",
                )
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"Error from training service: {e.response.text}",
            )
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=503,
                detail=f"Could not connect to the training service: {e}",
            )

    response_data = response.json()
    model_url = f"https://huggingface.co/{request.model_name}"

    return StartTrainingResponse(
        message=f"Training triggered successfully, find your model at: {model_url}",
        training_id=response_data.get("training_id"),
        model_url=model_url,
    )


@router.post("/training/start-custom", response_model=StatusResponse)
async def start_custom_training(
    request: CustomTrainingRequest,
    background_tasks: BackgroundTasks,
) -> StatusResponse | HTTPException:
    # 1) Prepare log file
    log_file_name = f"training_{int(time.time())}.log"
    log_file_path = os.path.join(get_home_app_path(), "logs", log_file_name)
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # 2) Spawn the process
    is_windows = platform.system() == "Windows"
    if not is_windows:
        # pty is not available on Windows, so we use subprocess directly
        import pty

        master_fd, slave_fd = pty.openpty()
        # We use create_subprocess_shell so we can pass the whole command string
        process = await asyncio.create_subprocess_shell(
            request.custom_command,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            preexec_fn=os.setsid,  # detach in its own process group
        )
        os.close(slave_fd)  # we only need master in our code

        # 3) Hook the PTY master into an asyncio StreamReader
        loop = asyncio.get_running_loop()
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        # Wrap the master FD as a "read pipe" so .read() becomes non-blocking
        await loop.connect_read_pipe(lambda: protocol, os.fdopen(master_fd, "rb"))
    else:
        process = await asyncio.create_subprocess_shell(
            request.custom_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        if process.stdout is None:
            raise RuntimeError("Failed to create subprocess")
        reader = process.stdout

    # 4) Monitor task: read from the PTY master and write to your log file
    async def monitor_pty(reader: asyncio.StreamReader, log_path: str) -> None:
        with open(log_path, "wb") as f:
            # header
            f.write(f"Custom training started at {time.ctime()}\n".encode())
            f.write(f"Command: {request.custom_command}\n\n".encode())
            f.flush()

            # stream everything, flushing as it arrives
            while True:
                chunk = await reader.read(1024)
                if not chunk:
                    break
                f.write(chunk)
                f.flush()

            # when process exits, append return code
            await process.wait()
            footer = f"\nProcess completed with return code {process.returncode}\n"
            f.write(footer.encode())
            if process.returncode == 0:
                f.write(b"Training completed successfully!\n")
            else:
                f.write(b"Training failed. See errors above.\n")

    background_tasks.add_task(monitor_pty, reader, log_file_path)

    return StatusResponse(message=log_file_name)


@router.get("/training/logs/{log_file}", response_model=None)
async def stream_logs(
    log_file: str,
) -> StreamingResponse | HTTPException | PlainTextResponse:
    """Stream the logs from a log file"""
    log_path = os.path.join(get_home_app_path(), "logs", log_file)

    if not os.path.exists(log_path):
        raise HTTPException(status_code=404, detail="Log file not found")

    if platform.system() == "Windows":
        return PlainTextResponse(
            "Streaming logs is not supported on Windows. Check the console logs directly."
        )

    async def log_generator() -> AsyncGenerator[bytes, None]:
        """Generator to stream logs line by line as they are written"""
        with open(log_path, "rb") as f:
            # First, send all existing content
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            f.seek(0)
            if file_size > 0:
                yield f.read()

            # Then, continue streaming as new content is added
            while True:
                line = f.readline()
                if line:
                    yield line
                else:
                    # Check if process is still running by looking for completion message
                    if b"Process completed with return code" in line:
                        break
                    await asyncio.sleep(0.1)  # Small delay to avoid busy waiting

    return StreamingResponse(log_generator(), media_type="text/plain")


@router.post("/training/cancel", response_model=StatusResponse)
async def cancel_training(
    request: CancelTrainingRequest,
    session: SupabaseSession = Depends(user_is_logged_in),
) -> StatusResponse | HTTPException:
    """Cancel a training job"""
    logger.debug(f"Cancelling training request: {request}")
    tokens = get_tokens()
    if not tokens.MODAL_API_URL:
        raise HTTPException(
            status_code=400,
            detail="Modal API url not found. Please check your configuration.",
        )

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{tokens.MODAL_API_URL}/cancel",
            json=request.model_dump(mode="json"),
            headers={"Authorization": f"Bearer {session.access_token}"},
        )

        if response.status_code == 401:
            raise HTTPException(
                status_code=401,
                detail="Token expired. Please login again.",
            )
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to cancel training on the backend: {response.text}",
            )

    response_data = response.json()

    return StatusResponse(
        status=response_data.get("status", "error"),
        message=response_data.get("message", "No message"),
    )
