import os
import platform
import uuid
from functools import wraps
from typing import Optional

from posthog import Posthog

from phosphobot import __version__
from phosphobot.configs import config
from phosphobot.utils import get_home_app_path, get_tokens


tokens = get_tokens()
posthog = Posthog(
    project_api_key=tokens.POSTHOG_API_KEY or "",
    host=tokens.POSTHOG_HOST,
)
posthog_details = {
    "phosphobot_version": __version__,
    "env": tokens.ENV,
    "system_info": f"{platform.node()}_{platform.system()}_{platform.release()}",
}

# Failure tracking
_failure_count = 0
_failure_threshold = 3


def with_failure_tracking(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        global _failure_count

        if posthog.disabled:
            return

        try:
            result = func(*args, **kwargs)
            _failure_count = 0  # Reset on success
            return result
        except Exception:
            _failure_count += 1
            if _failure_count >= _failure_threshold:
                posthog.disabled = True

    return wrapper


def is_github_actions():
    return os.getenv("GITHUB_ACTIONS") == "true"


def get_or_create_unique_id(token_path):
    """
    Retrieve or generate a unique ID, storing it in a token file. This is an
    anonymous identifier for the user.

    Args:
        token_path (str): Path to the unique ID token file

    Returns:
        str: Unique identifier for the user
    """
    if is_github_actions():
        return "github_actions_ci"
    # Ensure the directory exists
    os.makedirs(os.path.dirname(token_path), exist_ok=True)

    # Check if token file exists
    if os.path.exists(token_path):
        # Read existing token
        with open(token_path, "r") as f:
            content = f.read().strip()
        if content and content.startswith("phosphobot_"):
            return content

    # Generate a new unique ID if no token exists
    new_user_id = "phosphobot_" + str(uuid.uuid4())

    # Write the new ID to the token file
    with open(token_path, "w") as f:
        f.write(new_user_id)

    return new_user_id


# We disable posthog in dev environments
if not config.USAGE_TELEMETRY or is_github_actions():
    posthog.disabled = True

session_id = str(uuid.uuid4())

posthog_details["$session_id"] = session_id
TOKEN_FILE = get_home_app_path() / "id.token"
user_id = get_or_create_unique_id(TOKEN_FILE)


@with_failure_tracking
def posthog_pageview(page: str) -> None:
    posthog.capture(
        distinct_id=user_id,
        event="$pageview",
        properties={
            **posthog_details,
            "$current_url": "http://phosphobot.local" + page,
        },
    )


def add_email_to_posthog(email: Optional[str]) -> None:
    if posthog.disabled or not email:
        return

    global posthog_details

    posthog_details = {
        **posthog_details,
        "$email": email,
    }
