import asyncio
import time
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
    start_time = time.time()
    request_id = f"req_{int(start_time * 1000)}"  # Simple request ID for tracking

    tokens = get_tokens()

    # Log incoming request with unique ID
    logger.info(
        f"[{request_id}] Incoming request: method={request.method}, path={path}"
    )

    # Copy headers but exclude problematic ones
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
            "content-encoding",
        }
    }
    headers["Authorization"] = f"Bearer {session.access_token}"

    # Extract host from MODAL_API_URL and set it explicitly
    modal_url = tokens.MODAL_API_URL
    modal_host = modal_url.split("//")[1].split("/")[0]
    headers["Host"] = modal_host

    # Force HTTP/1.1 and disable keep-alive to avoid connection reuse issues
    headers["Connection"] = "close"
    headers["Cache-Control"] = "no-cache"

    query = request.url.query
    url = f"{modal_url}/gemini/{path}"
    if query:
        url = f"{url}?{query}"

    body_bytes = await request.body()
    logger.info(f"[{request_id}] Request body size: {len(body_bytes)} bytes")

    # Increase timeouts significantly
    timeout = httpx.Timeout(
        connect=30.0,  # Connection timeout
        read=180.0,  # Read timeout - increased from 120s
        write=30.0,  # Write timeout
        pool=30.0,  # Pool timeout
    )

    # Configure httpx client with better connection handling
    async with httpx.AsyncClient(
        timeout=timeout,
        limits=httpx.Limits(
            max_keepalive_connections=0,  # Disable connection pooling
            max_connections=10,
            keepalive_expiry=0,
        ),
        http2=False,  # Force HTTP/1.1
    ) as client:
        try:
            req = client.build_request(
                method=request.method,
                url=url,
                headers=headers,
                content=body_bytes or None,
            )
            logger.info(f"[{request_id}] Forwarding to: {url}")
            logger.debug(f"[{request_id}] Headers: {dict(headers)}")

            upstream_resp = await client.send(req, stream=True)
            logger.info(
                f"[{request_id}] Upstream response status: {upstream_resp.status_code}"
            )
            logger.debug(
                f"[{request_id}] Response headers: {dict(upstream_resp.headers)}"
            )

        except httpx.TimeoutException as e:
            logger.error(f"[{request_id}] Timeout error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=f"Upstream service timeout: {str(e)}",
            )
        except httpx.RequestError as e:
            logger.error(f"[{request_id}] Request error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Failed to reach internal Gemini proxy",
            )

    async def iter_stream():
        chunk_count = 0
        total_size = 0
        last_chunk_time = time.time()

        try:
            # Add a wrapper to detect stalled streams
            async for chunk in upstream_resp.aiter_bytes(
                chunk_size=8192
            ):  # Smaller chunks
                current_time = time.time()
                chunk_count += 1
                total_size += len(chunk)

                if chunk_count <= 10 or chunk_count % 100 == 0:  # Log more frequently
                    time_since_last = current_time - last_chunk_time
                    logger.debug(
                        f"[{request_id}] Chunk {chunk_count}: {len(chunk)} bytes, "
                        f"time since last: {time_since_last:.2f}s"
                    )

                last_chunk_time = current_time
                yield chunk

                # Add a small yield to prevent blocking the event loop
                if chunk_count % 50 == 0:
                    await asyncio.sleep(0)  # Yield control

            duration = time.time() - start_time
            logger.info(
                f"[{request_id}] Stream completed: {chunk_count} chunks, "
                f"{total_size} total bytes, {duration:.2f}s total"
            )

        except httpx.ReadError as e:
            logger.error(f"[{request_id}] Read error in stream: {str(e)}")
            logger.error(
                f"[{request_id}] Stream state: chunks={chunk_count}, size={total_size}"
            )
            # Try to determine if we got a partial response
            if chunk_count > 0:
                logger.warning(
                    f"[{request_id}] Partial response received, connection lost"
                )

            # Don't raise HTTPException in stream - just log and return
            # because response headers are already sent
            logger.error(
                f"[{request_id}] Stream failed immediately, likely Modal function issue"
            )
            return  # Exit the generator gracefully
        except Exception as e:
            logger.error(f"[{request_id}] Unexpected error in stream: {str(e)}")
            logger.exception(f"[{request_id}] Full exception details:")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Stream processing error: {str(e)}",
            )
        finally:
            try:
                await upstream_resp.aclose()
                logger.debug(f"[{request_id}] Upstream response closed")
            except Exception as e:
                logger.warning(
                    f"[{request_id}] Error closing upstream response: {str(e)}"
                )

    # Filter out problematic headers
    passthrough_headers = {
        k: v
        for k, v in upstream_resp.headers.items()
        if k.lower()
        not in {
            "transfer-encoding",
            "connection",
            "keep-alive",
            "content-encoding",
        }
    }

    # Force connection close in response
    passthrough_headers["Connection"] = "close"

    logger.info(
        f"[{request_id}] Returning response with status: {upstream_resp.status_code}"
    )
    return StreamingResponse(
        iter_stream(),
        status_code=upstream_resp.status_code,
        headers=passthrough_headers,
    )
