import asyncio
import logging
import platform
import socket
import sys
from asyncio import CancelledError
from contextlib import asynccontextmanager
from random import random
from typing import Any, AsyncGenerator, Callable

import sentry_sdk
import typer
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, applications
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from rich import print

from phosphobot import __version__
from phosphobot.camera import AllCameras, get_all_cameras, get_all_cameras_no_init
from phosphobot.configs import config
from phosphobot.endpoints import (
    auth_router,
    camera_router,
    chat_router,
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
from phosphobot.types import SimulationMode
from phosphobot.utils import (
    get_home_app_path,
    get_local_ip,
    get_resources_path,
    login_to_hf,
)


def init_telemetry() -> None:
    """
    This is used for automatic crash reporting.
    """
    from phosphobot.sentry import init_sentry

    init_sentry()


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
app.include_router(chat_router)

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
        "/joints/read",
        "/joints/write",
        "/torque/read",
        "/update/version",
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


def start_server(
    host: str = "0.0.0.0",
    port: int = 80,
    reload: bool = False,
    simulation: SimulationMode = SimulationMode.headless,
    only_simulation: bool = False,
    simulate_cameras: bool = False,
    realsense: bool = True,
    can: bool = True,
    cameras: bool = True,
    max_opencv_index: int = 10,
    max_can_interfaces: int = 4,
    profile: bool = False,
    crash_telemetry: bool = True,
    usage_telemetry: bool = True,
    telemetry: bool = True,
    silent: bool = False,
) -> None:
    """
    Start the FastAPI server.
    """

    log_dir = get_home_app_path()
    log_file_path = log_dir / "logs.txt"

    if silent:
        # Only log errors in silent mode
        logger.remove()
        logger.add(
            sys.stderr,
            level="ERROR",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        )

    # Configure loguru to write logs to a file.
    # This one sink will capture all logs, including those from uvicorn.
    logger.add(
        log_file_path,
        level="DEBUG",  # Set to DEBUG to capture all details
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="2 MB",
        retention="7 days",
        enqueue=True,  # Makes logging thread-safe/process-safe
        backtrace=True,
        diagnose=True,
    )

    # Intercept stdlib logging and forward to loguru.
    # If not doing that, then tracebacks are not logged to the file.
    class InterceptHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            # Translate logging level name to loguru level (fallback to levelno)
            try:
                level = logger.level(record.levelname).name
            except Exception:
                level = record.levelno  # type: ignore

            # Find depth so loguru shows the original caller
            frame, depth = logging.currentframe(), 2
            while frame and frame.f_code.co_filename == logging.__file__:
                if frame.f_back is None:
                    break
                frame = frame.f_back
                depth += 1

            logger.opt(depth=depth, exception=record.exc_info).log(
                level, record.getMessage()
            )

    # Replace handlers for the root logger (and be safe: set level to lowest so all messages are forwarded)
    logging.root.handlers = [InterceptHandler()]

    logger.info("Loguru file logging is configured. Server starting...")

    config.SIM_MODE = simulation
    config.ONLY_SIMULATION = only_simulation
    config.SIMULATE_CAMERAS = simulate_cameras
    config.ENABLE_REALSENSE = realsense
    config.ENABLE_CAMERAS = cameras
    config.PORT = port
    config.PROFILE = profile
    config.CRASH_TELEMETRY = crash_telemetry  # Enable crash telemetry by default
    config.USAGE_TELEMETRY = usage_telemetry  # Enable usage telemetry by default
    config.ENABLE_CAN = can
    config.MAX_OPENCV_INDEX = max_opencv_index
    config.MAX_CAN_INTERFACES = max_can_interfaces

    if not telemetry:
        config.CRASH_TELEMETRY = False
        config.USAGE_TELEMETRY = False

    # Start the FastAPI app using uvicorn with port retry logic
    ports = [port]
    if port == 80:
        ports += list(range(8020, 8040))  # 8020-8039 inclusive

    success = False
    for current_port in ports:
        if is_port_in_use(current_port, host):
            logger.warning(f"Port {current_port} is unavailable. Trying next...")
            continue

        try:
            # Update config with current port
            config.PORT = current_port

            server_config = uvicorn.Config(
                "phosphobot.app:app",
                host=host,
                port=current_port,
                reload=reload,
                timeout_graceful_shutdown=1,
                log_config=None if silent else uvicorn.config.LOGGING_CONFIG,
            )
            server = uvicorn.Server(config=server_config)

            # Run the server within the existing event loop
            asyncio.run(server.serve())
            success = True
            break
        except OSError as e:
            if "address already in use" in str(e).lower():
                logger.warning(f"Port conflict on {current_port}: {e}")
                continue
            logger.error(f"Critical server error: {e}")
            raise typer.Exit(code=1)
        except KeyboardInterrupt:
            logger.debug("Server stopped by user.")
            raise typer.Exit(code=0)
        except CancelledError:
            logger.debug("Server shutdown gracefully.")
            raise typer.Exit(code=0)

    if not success:
        logger.warning(
            "All ports failed. Try a custom port with:\n"
            "phosphobot run --port 8000\n\n"
            "Check used ports with:\n"
            "sudo lsof -i :80 # Replace 80 with your port"
        )
        raise typer.Exit(code=1)


if __name__ == "__main__":
    typer.run(start_server)
