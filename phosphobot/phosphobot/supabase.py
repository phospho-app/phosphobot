import asyncio
import json
import time
from typing import Optional

from fastapi import HTTPException
from loguru import logger
from supabase import AsyncClient, acreate_client
from supabase_auth.types import Session as SupabaseSession
from supabase_auth.errors import AuthRetryableError

from phosphobot.models import Session
from phosphobot.utils import get_home_app_path, get_tokens

AUTH_TOKEN = get_home_app_path() / "auth.token"

_client = None


async def initialize_client() -> AsyncClient:
    """
    Initialize the supabase client.
    """
    global _client

    tokens = get_tokens()
    if tokens.SUPABASE_URL is None:
        raise ValueError("SUPABASE_URL is not set in the tokens.toml file.")
    if tokens.SUPABASE_KEY is None:
        raise ValueError("SUPABASE_KEY is not set in the tokens.toml file.")

    if _client is None:
        _client = await acreate_client(
            supabase_url=tokens.SUPABASE_URL, supabase_key=tokens.SUPABASE_KEY
        )
    return _client


def save_session(session: Session) -> None:
    """
    Save the session to a file.
    """
    with open(AUTH_TOKEN, "w") as f:
        json.dump(session.model_dump(), f)


def load_session() -> Optional[Session]:
    """
    Load the session from a file.
    """
    try:
        with open(AUTH_TOKEN, "r") as f:
            session = json.load(f)
            session = Session.model_validate(session)
        return session
    except Exception:
        return None


def delete_session() -> None:
    """
    Delete the session file.
    """
    try:
        AUTH_TOKEN.unlink()
        logger.debug("Session deleted")
    except Exception:
        return


async def get_client() -> AsyncClient:
    """
    Get the Supabase client with a valid session, refreshing if necessary.
    """
    client = await initialize_client()
    session = load_session()

    async def set_session_with_retry(
        access_token, refresh_token, max_retries=3, delay=2
    ):
        current_delay = delay
        for attempt in range(max_retries):
            try:
                await client.auth.set_session(
                    access_token=access_token, refresh_token=refresh_token
                )
                return True
            except AuthRetryableError as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Retryable error: {e}. Retrying in {delay} seconds..."
                    )
                    await asyncio.sleep(current_delay)
                    current_delay += delay
                else:
                    logger.error(f"Failed after {max_retries} attempts: {e}")
                    return False

    if session:
        if (
            time.time() > session.expires_at - 60
        ):  # Refresh 60 seconds before expiration
            try:
                success = await set_session_with_retry(
                    access_token=session.access_token,
                    refresh_token=session.refresh_token,
                )
                if success:
                    new_supabase_session = await client.auth.get_session()
                    if new_supabase_session is not None:
                        if new_supabase_session.user.email is None:
                            raise HTTPException(
                                status_code=401, detail="User email not found"
                            )

                        updated_session = Session(
                            user_id=new_supabase_session.user.id,
                            user_email=new_supabase_session.user.email,
                            email_confirmed=new_supabase_session.user.email_confirmed_at
                            is not None,
                            access_token=new_supabase_session.access_token,
                            refresh_token=new_supabase_session.refresh_token,
                            expires_at=int(time.time())
                            + new_supabase_session.expires_in,
                        )
                        save_session(updated_session)
                        session = updated_session
                    else:
                        delete_session()
                        session = None
                else:
                    delete_session()
                    session = None

            except Exception as e:
                logger.debug(f"Failed to refresh session: {e}")
                delete_session()
                session = None
        else:
            success = await set_session_with_retry(
                access_token=session.access_token,
                refresh_token=session.refresh_token,
            )
            if not success:
                delete_session()
                session = None

    return client


async def user_is_logged_in() -> SupabaseSession:
    """
    Check if the user is logged in. If not, raise HTTPException with status code 401.
    """
    client = await get_client()
    session = await client.auth.get_session()
    if session is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return session
