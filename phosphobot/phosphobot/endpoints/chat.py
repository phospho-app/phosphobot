import httpx
from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from supabase_auth.types import Session as SupabaseSession

from phosphobot.models import ChatRequest, ChatResponse
from phosphobot.supabase import user_is_logged_in
from phosphobot.utils import get_tokens

router = APIRouter(tags=["chat"])

tokens = get_tokens()


@router.post("/ai-control/chat", response_model=ChatResponse)
async def ai_control_chat(
    request: ChatRequest,
    session: SupabaseSession = Depends(user_is_logged_in),
) -> ChatResponse:
    """
    Endpoint to handle AI control chat requests.
    """

    # Call the /chat endpoint of modal
    # make an async request to the modal endpoint
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                url=f"{tokens.MODAL_API_URL}/ai-control/chat",
                json=request.model_dump(mode="json"),
                timeout=30.0,  # Set a timeout for the request
                headers={
                    "Authorization": f"Bearer {session.access_token}",
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()  # Raise an error for bad responses
        except httpx.HTTPStatusError as e:
            logger.warning(f"HTTP error occurred: {e}")
            raise HTTPException(status_code=e.response.status_code, detail=str(e))

    return ChatResponse.model_validate(response.json())
