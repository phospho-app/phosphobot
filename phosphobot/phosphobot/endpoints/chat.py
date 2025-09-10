from fastapi import APIRouter, Depends, Request
import httpx
from supabase_auth.types import Session as SupabaseSession
from phosphobot.supabase import user_is_logged_in
from phosphobot.utils import get_tokens

router = APIRouter(tags=["chat"])


@router.api_route(
    "/chat/gemini/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"]
)
async def proxy_to_gemini(
    request: Request,
    path: str,
    session: SupabaseSession = Depends(user_is_logged_in),
):
    """
    Proxy requests to the Gemini API.
    """

    tokens = get_tokens()
    async with httpx.AsyncClient(timeout=10) as client:
        return await client.request(
            method=request.method,
            url=f"{tokens.MODAL_API_URL}/gemini/{path}",
            headers={
                "Authorization": f"Bearer {session.access_token}",
                "Content-Type": "application/json",
            },
        )
