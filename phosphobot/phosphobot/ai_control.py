import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional

import httpx
import numpy as np
from fastapi import HTTPException
from loguru import logger

from phosphobot.am.act import ACT, ACTSpawnConfig
from phosphobot.am.gr00t import Gr00tN1, Gr00tSpawnConfig
from phosphobot.camera import AllCameras
from phosphobot.control_signal import AIControlSignal
from phosphobot.hardware.base import BaseManipulator
from phosphobot.models import ServerInfoResponse
from phosphobot.supabase import get_client
from phosphobot.utils import get_tokens


class CustomAIControlSignal(AIControlSignal):
    def __init__(self):
        super().__init__()
        self._supabase_client = None
        self._last_status_update = None

    def _update_supabase(
        self, started_at: Optional[datetime] = None, ended_at: Optional[datetime] = None
    ):
        # schedule the real work in the running loop
        if self._status == self._last_status_update:
            return
        loop = asyncio.get_event_loop()
        loop.create_task(self._update_supabase_async(started_at, ended_at))

    async def _update_supabase_async(
        self, started_at: Optional[datetime] = None, ended_at: Optional[datetime] = None
    ):
        if not self._supabase_client:
            self._supabase_client = await get_client()

        payload = {"status": self._status}
        if started_at is not None:
            payload["started_at"] = started_at.isoformat()
        if ended_at is not None:
            payload["ended_at"] = ended_at.isoformat()

        try:
            await (
                self._supabase_client.table("ai_control_sessions")
                .update(payload)
                .eq("id", self.id)
                .execute()
            )
            self._last_status_update = self._status
        except Exception as e:
            logger.warning(f"Error updating Supabase: {e}")

    def new_id(self):
        super().new_id()

    def start(self):
        with self._lock:
            self._is_in_loop = True
            self._status = "waiting"
            self._update_supabase()

    def set_running(self):
        with self._lock:
            self._is_in_loop = True
            self._status = "running"
            self._update_supabase(started_at=datetime.now(timezone.utc))

    def stop(self):
        with self._lock:
            self._is_in_loop = False
            self._status = "stopped"
            self._update_supabase(ended_at=datetime.now(timezone.utc))

    def is_in_loop(self):
        with self._lock:
            return self._is_in_loop

    @property
    def status(self) -> Literal["stopped", "running", "paused", "waiting"]:
        return self._status

    @status.setter
    def status(self, value: Literal["stopped", "running", "paused", "waiting"]):
        if value == "stopped":
            self.stop()
        elif value == "running":
            self._status = value
            with self._lock:
                self._is_in_loop = True
        elif value == "paused":
            self._status = value
        elif value == "waiting":
            self._status = value
            with self._lock:
                self._is_in_loop = True

        self._update_supabase()


async def setup_ai_control(
    robots: List[BaseManipulator],
    all_cameras: AllCameras,
    ai_control_signal_id: str,
    model_type: Literal["gr00t", "ACT", "ACT_BBOX"],
    model_id: str = "PLB/GR00T-N1-lego-pickup-mono-2",
    cameras_keys_mapping: Optional[dict[str, int]] = None,
    init_connected_robots: bool = True,
    verify_cameras: bool = True,
    checkpoint: Optional[int] = None,
) -> tuple[Gr00tN1 | ACT, Gr00tSpawnConfig | ACTSpawnConfig, ServerInfoResponse]:
    """
    Setup the AI control loop by spawning the inference server and returning the model.
    This function is called when the user clicks on the "Start AI Control" button in the UI.
    """

    tokens = get_tokens()
    if tokens.MODAL_API_URL is None:
        raise HTTPException(
            status_code=400,
            detail="Modal API key not found. Please check your configuration.",
        )

    supabase_client = await get_client()
    session = await supabase_client.auth.get_session()
    if session is None:
        raise HTTPException(
            status_code=401,
            detail="Session expired. Please log in again.",
        )

    model_types: Dict[str, type[ACT | Gr00tN1]] = {
        "gr00t": Gr00tN1,
        "ACT": ACT,
        "ACT_BBOX": ACT,
    }

    try:
        model_used = model_types[model_type]
        model_spawn_config = model_used.fetch_and_verify_config(
            model_id=model_id,
            all_cameras=all_cameras,
            robots=robots,  # type: ignore
            cameras_keys_mapping=cameras_keys_mapping,
            verify_cameras=verify_cameras,
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Model verification failed for {model_type}: {e}",
        )

    def sanitize(o):
        if isinstance(o, float):
            return 0.0 if (np.isnan(o) or np.isinf(o)) else o
        if isinstance(o, dict):
            return {k: sanitize(v) for k, v in o.items()}
        if isinstance(o, list):
            return [sanitize(v) for v in o]
        return o

    # We need to sanitize the model_spawn_config to avoid sending NaN or Inf values
    # the json_encoders in the pydantic model doesn't work when we do .model_dump(mode='json')
    # because of weird types:
    # pydantic_core._pydantic_core.PydanticSerializationError: Error calling function `<lambda>`: TypeError: ufunc 'isnan' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''
    # i have a headache so let's just do it manually
    raw = model_spawn_config.model_dump()
    clean = sanitize(raw)

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            url=f"{tokens.MODAL_API_URL}/spawn",
            json={
                "model_id": model_id,
                "checkpoint": checkpoint,
                "model_type": model_type,
                "model_specifics": clean,
            },
            headers={
                "Authorization": f"Bearer {session.access_token}",
                "Content-Type": "application/json",
            },
        )

    if response.status_code != 200:
        logger.error(f"Failed to start inference server: {response.text}")
        await (
            supabase_client.table("ai_control_sessions")
            .update(
                {
                    "status": "stopped",
                }
            )
            .eq("id", ai_control_signal_id)
            .execute()
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start inference server: {response.text}",
        )

    server_info = ServerInfoResponse.model_validate(response.json())

    connects_through_tcp = ["gr00t"]

    if model_type in connects_through_tcp:
        server_url = server_info.tcp_socket[0]
        server_port = server_info.tcp_socket[1]
    else:
        server_url = server_info.url
        server_port = server_info.port

    model = model_types[model_type](
        server_url=server_url,
        server_port=server_port,
        **model_spawn_config.model_dump(),
    )

    if init_connected_robots:
        # Reset the robot to the initial position
        logger.debug("Resetting robot to initial position")
        if len(robots) == 0:
            raise HTTPException(
                status_code=400,
                detail="No robot connected. Exiting AI control loop.",
            )
        for robot in robots:
            await robot.move_to_initial_position(open_gripper=True)

    return model, model_spawn_config, server_info
