from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse
import httpx
from supabase_auth.types import Session as SupabaseSession
from phosphobot.supabase import user_is_logged_in
from phosphobot.utils import get_tokens
from loguru import logger

router = APIRouter(tags=["chat"])


@router.api_route(
    "/chat/gemini/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"]
)
async def proxy_to_internal_gemini(
    request: Request,
    path: str,
    session: SupabaseSession = Depends(user_is_logged_in),
):
    tokens = get_tokens()

    # copy headers, strip hop-by-hop and incoming authorization
    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower()
        not in {
            "host",
            "authorization",
            "connection",
            "keep-alive",
            "transfer-encoding",
        }
    }
    headers["Authorization"] = f"Bearer {session.access_token}"

    query = request.url.query
    url = f"{tokens.MODAL_API_URL}/gemini/{path}"
    if query:
        url = f"{url}?{query}"

    # read the request body once
    body_bytes = await request.body()

    logger.debug(
        "proxy_to_internal_gemini forwarding request: url=%s method=%s body_size=%d",
        url,
        request.method,
        len(body_bytes or b""),
    )

    # give generous timeout for potentially large payloads
    timeout = httpx.Timeout(120.0, connect=30.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            req = client.build_request(
                method=request.method,
                url=url,
                headers=headers,
                content=body_bytes or None,
            )

            # send with stream=True to stream response back
            upstream_resp = await client.send(req, stream=True)

        except httpx.RequestError:
            logger.exception("Error contacting internal Gemini proxy")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Failed to reach internal Gemini proxy",
            )

    # NOTE: upstream_resp is a httpx.Response and still open for streaming
    async def iter_stream():
        try:
            async for chunk in upstream_resp.aiter_bytes():
                # optional debug: logger.debug("upstream chunk len=%d", len(chunk))
                yield chunk
        finally:
            # ensure upstream connection is closed if generator exits early
            await upstream_resp.aclose()

    passthrough_headers = {
        k: v
        for k, v in upstream_resp.headers.items()
        if k.lower() not in {"transfer-encoding", "connection", "keep-alive"}
    }

    # If upstream returned an error status, stream it back with same status and body
    return StreamingResponse(
        iter_stream(),
        status_code=upstream_resp.status_code,
        headers=passthrough_headers,
    )
