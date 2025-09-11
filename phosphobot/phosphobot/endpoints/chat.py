import asyncio
import time
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
import httpx
from supabase_auth.types import Session as SupabaseSession
from phosphobot.supabase import user_is_logged_in
from phosphobot.utils import get_tokens
from loguru import logger

router = APIRouter(tags=["chat"])


@router.post("/ai-control/chat")
async def ai_control_chat(
    request: Request,
    session: SupabaseSession = Depends(user_is_logged_in),
):
    """
    Endpoint to handle AI control chat requests.
    """

    # Call the /chat endpoint of modal
    tokens = get_tokens()
    # make an async request to the modal endpoint
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                url=f"{tokens.MODAL_API_URL}/ai-control/chat",
                json={
                    "session": session,
                    "request": await request.json(),
                },
                timeout=30.0,  # Set a timeout for the request
            )
            response.raise_for_status()  # Raise an error for bad responses
        except httpx.HTTPStatusError as e:
            logger.warning(f"HTTP error occurred: {e}")
            raise HTTPException(status_code=e.response.status_code, detail=str(e))
