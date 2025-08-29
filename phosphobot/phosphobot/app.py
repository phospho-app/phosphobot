import socket
import platform
from random import random
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Callable

import sentry_sdk
import typer
from fastapi import Depends, FastAPI, HTTPException, Request, applications
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from rich import print

from phosphobot import __version__
from phosphobot.camera import AllCameras, get_all_cameras_no_init, get_all_cameras
from phosphobot.configs import config
from phosphobot.endpoints import (
    auth_router,
    camera_router,
    control_router,
    networking_router,
    pages_router,
    recording_router,
    training_router,
    update_router,
)
from phosphobot.hardware import get_sim
from phosphobot.models import ServerStatus
from phosphobot.posthog import posthog, posthog_pageview
from phosphobot.recorder import Recorder, get_recorder
from phosphobot.robot import RobotConnectionManager, get_rcm
from phosphobot.teleoperation import get_udp_server
from phosphobot.utils import (
    get_home_app_path,
    get_resources_path,
    login_to_hf,
)


def init_telemetry() -> None:
    """
    This is used for automatic crash reporting.
    """
    from phosphobot.sentry import init_sentry

    init_sentry()


def get_local_ip() -> str:
    """
    Get the local IP address of the server.
    """
    try:
        # Create a temporary socket to get the local IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))  # Doesn't actually send data
            server_ip = s.getsockname()[0]
    except Exception:
        server_ip = "localhost"
    return server_ip


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Initialize telemetry
    init_telemetry()
    udp_server = get_udp_server()
    # Initialize pybullet simulation
    sim = get_sim()
    # Initialize rcm
    rcm = get_rcm()

    try:
        login_to_hf()
    except Exception as e:
        logger.debug(f"Failed to login to Hugging Face: {e}")
    try:
        server_ip = get_local_ip()
        logger.success(
            f"Startup complete. Go to the phosphobot dashboard here: http://{server_ip}:{config.PORT}"
        )
        yield
    finally:
        udp_server.stop()

        from phosphobot.endpoints.control import (
            signal_ai_control,
            signal_gravity_control,
            signal_leader_follower,
        )

        if signal_ai_control.is_in_loop():
            signal_ai_control.stop()
            logger.info("AI control signal stopped")
        if signal_gravity_control.is_in_loop():
            signal_gravity_control.stop()
            logger.info("Gravity control signal stopped")
        if signal_leader_follower.is_in_loop():
            signal_leader_follower.stop()
            logger.info("Leader follower control signal stopped")

        cameras = get_all_cameras_no_init()
        if cameras:
            cameras.stop()

        # Cleanup the simulation environment
        del rcm
        del sim
        sentry_sdk.flush(timeout=1)
        posthog.shutdown()


app = FastAPI(lifespan=lifespan)

# Check if "/dist" is not empty and exists
if (
    not (get_resources_path() / "dist").exists()
    or not (get_resources_path() / "dist").is_dir()
    or not any((get_resources_path() / "dist").iterdir())
):
    error_message = (
        "The 'dist' directory does not exist in the resources path. "
        "You need to build the dashboard first, then copy dashboard/dist to posphobot/resources/dist. "
        "Make sure node and npm are installed, then run the command: make build_frontend"
    )
    raise FileNotFoundError(error_message)

# We do this to serve the static files in the frontend
# This is a workaround for when the raspberry pi uses its own hotspot
app.mount("/resources", StaticFiles(directory=get_resources_path()), name="static")
# Mount the directory with your dashboard's production build (adjust the path as needed)
# Mount assets at the root (assuming get_resources_path() contains both index.html and assets)
app.mount(
    "/assets",
    StaticFiles(directory=f"{get_resources_path()}/dist/assets"),
    name="assets",
)
app.mount(
    "/dashboard",
    StaticFiles(directory=get_resources_path() / "dist", html=True),
    name="dashboard",
)


def swagger_monkey_patch(*args: Any, **kwargs: Any) -> HTMLResponse:
    posthog_pageview("/docs")
    return get_swagger_ui_html(
        *args,
        **kwargs,
        swagger_js_url="/resources/swagger-ui/swagger-ui-bundle.js",
        swagger_css_url="/resources/swagger-ui/swagger-ui.css",
        swagger_favicon_url="/resources/swagger-ui/favicon.png",
    )


applications.get_swagger_ui_html = swagger_monkey_patch


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    logger.warning(f"HTTPException: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.get("/status", response_model=ServerStatus)
async def status(
    rcm: RobotConnectionManager = Depends(get_rcm),
    cameras: AllCameras = Depends(get_all_cameras),
    recorder: Recorder = Depends(get_recorder),
) -> ServerStatus:
    """
    Get the status of the server.
    """
    from phosphobot.endpoints.control import (
        signal_ai_control,
        signal_leader_follower,
    )

    robots = await rcm.robots

    robot_names = [robot.name for robot in robots]

    server_status = ServerStatus(
        status="ok",
        name=platform.uname().node,  # Name of the machine
        robots=robot_names,
        robot_status=await rcm.status(),
        cameras=cameras.status(),
        is_recording=recorder.is_recording or recorder.is_saving,
        ai_running_status=signal_ai_control.status,
        leader_follower_status=signal_leader_follower.is_in_loop(),
        server_ip=get_local_ip(),
        server_port=config.PORT,
    )
    return server_status


app.include_router(control_router)
app.include_router(camera_router)
app.include_router(recording_router)
app.include_router(training_router)
app.include_router(pages_router)
app.include_router(networking_router)
app.include_router(update_router)
app.include_router(auth_router)

# TODO : Only allow secured origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Add the posthog middleware
@app.middleware("http")
def posthog_middleware(request: Request, call_next: Callable) -> JSONResponse:
    # ignore the /move/relative, /move/absolute and /status endpoints
    if request.url.path not in [
        "/move/relative",
        "/move/absolute",
        "/status",
    ] and not request.url.path.startswith("/asset"):
        # Sample only 20% of the requests
        if random() < 0.2:
            posthog_pageview(request.url.path)
    return call_next(request)


def version_callback(value: bool) -> None:
    if value:
        print(f"phosphobot {__version__}")
        raise typer.Exit()


def is_port_in_use(port: int, host: str) -> bool:
    """Check if a port is already in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return False
        except OSError:
            return True


if config.PROFILE:
    logger.info("Profiling enabled")

    from pyinstrument import Profiler
    from pyinstrument.renderers.html import HTMLRenderer
    from pyinstrument.renderers.speedscope import SpeedscopeRenderer

    @app.middleware("http")
    async def profile_request(request: Request, call_next: Callable) -> JSONResponse:
        # we map a profile type to a file extension, as well as a pyinstrument profile renderer
        profile_type_to_ext = {"html": "html", "speedscope": "speedscope.json"}
        profile_type_to_renderer = {
            "html": HTMLRenderer,
            "speedscope": SpeedscopeRenderer,
        }

        profiler = Profiler(interval=0.1, async_mode="enabled")
        profiler.start()
        response = await call_next(request)
        profiler.stop()

        # we dump the profiling into a file
        extension = profile_type_to_ext["html"]
        renderer = profile_type_to_renderer["html"]()
        filepath = str(get_home_app_path() / f"profile.{extension}")
        with open(filepath, "w") as out:
            out.write(profiler.output(renderer=renderer))
        return response
