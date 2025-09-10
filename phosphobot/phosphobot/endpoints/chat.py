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

    # Log incoming request
    logger.debug(f"Incoming request: method={request.method}, path={path}")
    # logger.debug(f"Request headers: {dict(request.headers)}")

    # Copy headers but exclude problematic ones
    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower()
        not in {
            "host",  # Remove host header as we'll set it explicitly
            "authorization",
            "connection",
            "keep-alive",
            "transfer-encoding",
            "content-encoding",  # Remove content-encoding to avoid compression issues
        }
    }
    headers["Authorization"] = f"Bearer {session.access_token}"

    # Extract host from MODAL_API_URL and set it explicitly
    modal_url = tokens.MODAL_API_URL
    modal_host = modal_url.split("//")[1].split("/")[0]
    headers["Host"] = modal_host

    query = request.url.query
    url = f"{modal_url}/gemini/{path}"
    if query:
        url = f"{url}?{query}"

    body_bytes = await request.body()
    logger.debug(f"Request body size: {len(body_bytes)} bytes")

    timeout = httpx.Timeout(120.0, connect=30.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            req = client.build_request(
                method=request.method,
                url=url,
                headers=headers,
                content=body_bytes or None,
            )
            logger.debug(f"Forwarding to: {url}")
            logger.debug(f"With headers: {dict(headers)}")

            upstream_resp = await client.send(req, stream=True)
            logger.debug(f"Upstream response status: {upstream_resp.status_code}")
            # logger.debug(f"Upstream response headers: {dict(upstream_resp.headers)}")

        except httpx.RequestError as e:
            logger.error(f"Error contacting internal Gemini proxy: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Failed to reach internal Gemini proxy",
            )

    async def iter_stream():
        chunk_count = 0
        total_size = 0
        try:
            async for chunk in upstream_resp.aiter_bytes():
                chunk_count += 1
                total_size += len(chunk)
                if chunk_count <= 5:  # Log first 5 chunks
                    logger.debug(f"Chunk {chunk_count}: {len(chunk)} bytes")
                yield chunk
            logger.debug(
                f"Stream completed: {chunk_count} chunks, {total_size} total bytes"
            )
        except Exception as e:
            logger.error(f"Error in stream iteration: {str(e)}")
            raise
        finally:
            await upstream_resp.aclose()

    # Filter out problematic headers
    passthrough_headers = {
        k: v
        for k, v in upstream_resp.headers.items()
        if k.lower()
        not in {
            "transfer-encoding",
            "connection",
            "keep-alive",
            "content-encoding",  # Remove content-encoding to avoid decompression issues
        }
    }

    logger.debug(f"Returning response with status: {upstream_resp.status_code}")
    return StreamingResponse(
        iter_stream(),
        status_code=upstream_resp.status_code,
        headers=passthrough_headers,
    )
