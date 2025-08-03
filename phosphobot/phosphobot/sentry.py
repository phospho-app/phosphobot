import sentry_sdk

from phosphobot._version import __version__
from phosphobot.configs import config
from phosphobot.utils import get_tokens


def init_sentry():
    if not config.CRASH_TELEMETRY:
        return

    tokens = get_tokens()
    if tokens.SENTRY_DSN is None or tokens.ENV != "prod":
        return

    sentry_sdk.init(
        dsn=tokens.SENTRY_DSN,
        send_default_pii=True,
        traces_sample_rate=0.1,
        release=f"teleop@{__version__}",
        environment=tokens.ENV,
    )


def add_email_to_sentry(email: str) -> None:
    if not config.CRASH_TELEMETRY:
        return

    if not email:
        return

    sentry_sdk.set_user({"email": email})
