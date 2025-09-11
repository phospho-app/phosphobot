import asyncio
import platform
import subprocess

from fastapi import APIRouter, HTTPException

from phosphobot._version import __version__
from phosphobot.utils import fetch_latest_brew_version, is_running_on_pi

router = APIRouter(tags=["update"])


@router.post("/update/version", response_model=dict)
async def get_latest_available_version(run_quick: bool = False) -> dict:
    """
    Get the latest available version of the teleop software.
    Works only on raspberry pi devices.
    """
    # Check if we're running on Linux
    if platform.system() == "Linux":
        # We do this to make sure the endpoint runs quickly
        async def run_update() -> None:
            process = await asyncio.create_subprocess_exec(
                "sudo",
                "apt",
                "update",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await process.communicate()  # Wait for the process to complete

        try:
            if run_quick:
                asyncio.create_task(run_update())  # Run in the background
            else:
                await run_update()  # Wait for completion in the request

            version_process = await asyncio.create_subprocess_shell(
                "apt-cache policy phosphobot | grep 'Candidate:' | awk '{print $2}'",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await version_process.communicate()
            version = stdout.decode().strip()

            return {"version": version, "latest": version == __version__}
        except Exception as e:
            return {"error": str(e)}
    elif platform.system() == "Darwin" or platform.system() == "Windows":
        # Check the brew package version in the repo homebrew-phosphobot
        version = fetch_latest_brew_version(fail_silently=True)
        if version != "unknown":
            return {"version": version}
        else:
            return {"version": None, "error": "No version found in the tap."}
    else:
        raise HTTPException(
            status_code=400,
            detail="This endpoint is only available on Linux and MacOS devices.",
        )


@router.get("/update/upgrade-to-latest-version", response_model=dict)
async def upgrade_to_latest_version() -> dict:
    """
    Upgrade the teleop software to the latest available version.
    Checks the latest available version and upgrades the software if necessary.
    Works only on raspberry pi devices.
    """
    if not is_running_on_pi():
        raise HTTPException(
            status_code=400,
            detail="This endpoint is only available on Raspberry Pi devices.",
        )
    try:
        subprocess.run(["sudo", "apt", "update"], check=True)
        subprocess.run(["sudo", "apt", "upgrade", "-y", "phosphobot"], check=True)
        return {
            "status": "Upgrade is successful, you might need to restart the server."
        }
    except Exception as e:
        return {"error": str(e)}
