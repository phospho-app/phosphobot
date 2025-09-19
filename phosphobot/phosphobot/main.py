from loguru import logger

logger.info("Starting phosphobot...")

import sys

print(f"sys.stdout.encoding = {sys.stdout.encoding}")

import io

# Fix encoding issues on Windows
if sys.platform.startswith("win") and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer, encoding="utf-8", errors="replace"
        )
    except Exception:
        pass  # Ignore if already wrapped or in unsupported environment


from rich import print

from phosphobot import __version__

_splash_shown = False


def print_phospho_splash() -> None:
    global _splash_shown
    if not _splash_shown:
        print(
            f"""[green]
    ░█▀█░█░█░█▀█░█▀▀░█▀█░█░█░█▀█░█▀▄░█▀█░▀█▀
    ░█▀▀░█▀█░█░█░▀▀█░█▀▀░█▀█░█░█░█▀▄░█░█░░█░
    ░▀░░░▀░▀░▀▀▀░▀▀▀░▀░░░▀░▀░▀▀▀░▀▀░░▀▀▀░░▀░

    phosphobot {__version__}
    Copyright (c) 2025 phospho https://phospho.ai
            [/green]"""
        )
        _splash_shown = True


print_phospho_splash()

import platform
import threading

from phosphobot.utils import fetch_latest_brew_version

_version_check_started = False


def fetch_latest_version() -> None:
    try:
        version = fetch_latest_brew_version(fail_silently=True)
        if version != "unknown" and (version != "v" + __version__):
            if platform.system() == "Darwin":
                logger.warning(
                    f"[update now!] phosphobot v{version} is available. Please update with: \nbrew update && brew upgrade phosphobot"
                )
            elif platform.system() == "Linux":
                logger.warning(
                    f"[update now!] phosphobot v{version} is available. Please update with: \nsudo apt update && sudo apt upgrade phosphobot"
                )
            else:
                logger.warning(
                    f"[update now!] phosphobot v{version} is available. Please update: https://docs.phospho.ai/installation#windows"
                )
    except Exception:
        pass


if not _version_check_started:
    thread = threading.Thread(target=fetch_latest_version, daemon=True)
    thread.start()
    _version_check_started = True

import socket
import threading
import time
from typing import Annotated

import typer

from phosphobot.types import SimulationMode


def init_telemetry() -> None:
    """
    This is used for automatic crash reporting.
    """
    from phosphobot.sentry import init_sentry

    init_sentry()


cli = typer.Typer(no_args_is_help=True, rich_markup_mode="rich")


def version_callback(value: bool) -> None:
    if value:
        print(f"phosphobot {__version__}")
        raise typer.Exit()


@cli.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-v",
            help="Show the application's version and exit.",
            callback=version_callback,
        ),
    ] = False,
) -> None:
    """
    phosphobot - A robotics teleoperation server.
    """
    pass


@cli.command()
def info(
    opencv: Annotated[bool, typer.Option(help="Show OpenCV information.")] = False,
    servos: Annotated[bool, typer.Option(help="Show servo information.")] = False,
) -> typer.Exit:
    """
    Show all serial ports (/dev/ttyUSB0) and camera information. Useful for debugging.
    """
    import serial.tools.list_ports

    ports = serial.tools.list_ports.comports()
    pid_list = [port.pid for port in ports]
    serial_numbers = [port.serial_number for port in ports]

    print("\n")
    print(
        f"[green]Available serial ports:[/green] {', '.join([port.device for port in ports])}"
    )
    print(
        f"[green]Available serial numbers:[/green]  {', '.join([str(sn) for sn in serial_numbers])}"
    )
    print(f"[green]Available PIDs:[/green]  {' '.join([str(pid) for pid in pid_list])}")
    print("\n")

    import cv2

    from phosphobot.camera import get_all_cameras

    cameras = get_all_cameras()
    time.sleep(0.5)
    cameras_status = cameras.status().model_dump_json(indent=4)
    cameras.stop()
    print(f"Cameras status: {cameras_status}")

    if opencv:
        print(cv2.getBuildInformation())

    if servos:
        from phosphobot.hardware.motors.feetech import (  # type: ignore
            dump_servo_states_to_file,
        )
        from phosphobot.utils import get_home_app_path

        # Diagnose SO-100 servos
        for port in ports:
            if port.pid == 21971:
                dump_servo_states_to_file(
                    str(get_home_app_path() / f"servo_states_{port.device}.csv"),
                    port.device,
                )

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


@cli.command()
def update() -> None:
    """
    Display information on how to update the software.
    """
    if platform.system() == "Darwin":
        logger.warning(
            "To update phosphobot, run the following command:\n"
            "brew update && brew upgrade phosphobot"
        )
    elif platform.system() == "Linux":
        logger.warning(
            "To update phosphobot, run the following command:\n"
            "sudo apt update && sudo apt upgrade phosphobot"
        )
    else:
        logger.warning(
            "To update phosphobot, please refer to the documentation. https://docs.phospho.ai/installation#windows"
        )


@cli.command()
def run(
    chat: Annotated[bool, typer.Option(help="Run phosphobot in chat mode.")] = False,
    host: Annotated[str, typer.Option(help="Host to bind to.")] = "0.0.0.0",
    port: Annotated[int, typer.Option(help="Port to bind to.")] = 80,
    simulation: Annotated[
        SimulationMode,
        typer.Option(
            help="Run the simulation in headless or gui mode.",
        ),
    ] = SimulationMode.headless,
    only_simulation: Annotated[
        bool, typer.Option(help="Only run the simulation.")
    ] = False,
    simulate_cameras: Annotated[
        bool,
        typer.Option(help="Simulate a classic camera and a secondary classic camera."),
    ] = False,
    realsense: Annotated[
        bool,
        typer.Option(help="Enable the RealSense camera."),
    ] = True,
    can: Annotated[
        bool,
        typer.Option(
            help="Enable the CAN scanning. If False, CAN devices will not detected. Useful in case of conflicts.",
        ),
    ] = True,
    cameras: Annotated[
        bool,
        typer.Option(
            help="Enable the cameras. If False, no camera will be detected. Useful in case of conflicts.",
        ),
    ] = True,
    max_can_interfaces: Annotated[
        int,
        typer.Option(
            help="Maximum expected CAN interfaces. Default is 4.",
        ),
    ] = 4,
    max_opencv_index: Annotated[
        int,
        typer.Option(
            help="Maximum OpenCV index to search for cameras. Default is 10.",
        ),
    ] = 10,
    reload: Annotated[
        bool,
        typer.Option(
            help="(dev) Reload the server on file changes. Do not use when cameras are running."
        ),
    ] = False,
    profile: Annotated[
        bool,
        typer.Option(
            help="(dev) Enable performance profiling. This generates profile.html."
        ),
    ] = False,
    crash_telemetry: Annotated[
        bool,
        typer.Option(help="Disable crash reporting."),
    ] = True,
    usage_telemetry: Annotated[
        bool,
        typer.Option(help="Disable usage analytics."),
    ] = True,
    telemetry: Annotated[
        bool,
        typer.Option(help="Disable all telemetry (Crash and Usage)."),
    ] = True,
) -> None:
    """
    🧪 [green]Run the phosphobot dashboard and API server.[/green] Control your robot and record datasets.
    """
    from phosphobot.app import start_server

    kwargs = {
        "host": host,
        "port": port,
        "reload": reload,
        "simulation": simulation,
        "only_simulation": only_simulation,
        "simulate_cameras": simulate_cameras,
        "realsense": realsense,
        "can": can,
        "cameras": cameras,
        "max_opencv_index": max_opencv_index,
        "max_can_interfaces": max_can_interfaces,
        "profile": profile,
        "crash_telemetry": crash_telemetry,
        "usage_telemetry": usage_telemetry,
        "telemetry": telemetry,
    }

    if not chat:
        start_server(**kwargs)  # type: ignore
    else:
        # Remove logging
        kwargs["silent"] = True
        # Start the server in a separate thread
        thread = threading.Thread(
            target=start_server,
            kwargs=kwargs,
            daemon=True,  # Ensure the thread exits when the main program exits
        )
        thread.start()

        # Launch in chat mode
        from phosphobot.chat.app import AgentApp

        app = AgentApp()
        app.run()


if __name__ == "__main__":
    cli()
