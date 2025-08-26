import asyncio
import os
import subprocess
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from loguru import logger
from serial.tools import list_ports


from phosphobot.models import (
    LocalDevice,
    NetworkCredentials,
    ScanDevicesResponse,
    ScanNetworkRequest,
    ScanNetworkResponse,
    StatusResponse,
)
from phosphobot.utils import (
    background_task_log_exceptions,
    is_running_on_pi,
    get_local_subnet,
    scan_network_devices,
)

router = APIRouter(tags=["networking"])


@background_task_log_exceptions
async def setup_hotspot_bg():
    """Background task for hotspot setup and activation"""
    try:
        # Check if setup has already been done
        if not os.path.exists("/etc/dnsmasq.d/hotspot.conf"):
            # Install dnsmasq if not already installed
            subprocess.run(["sudo", "apt-get", "install", "-y", "dnsmasq"], check=True)

            # Configure dnsmasq for DHCP
            dnsmasq_config = """interface=wlan0
dhcp-range=192.168.1.2,192.168.1.200,12h
"""
            # Write config using sudo
            subprocess.run(
                ["sudo", "tee", "/etc/dnsmasq.d/hotspot.conf"],
                input=dnsmasq_config.encode(),
                check=True,
            )

            # Restart dnsmasq to apply changes
            subprocess.run(["sudo", "systemctl", "restart", "dnsmasq"], check=True)

        # Delete any existing hotspot connection
        subprocess.run(
            ["sudo", "nmcli", "connection", "delete", "hotspot"], check=False
        )

        # Create the hotspot connection
        subprocess.run(
            [
                "sudo",
                "nmcli",
                "connection",
                "add",
                "type",
                "wifi",
                "ifname",
                "wlan0",
                "con-name",
                "hotspot",
                "ssid",
                "phosphobot",
                "mode",
                "ap",
            ],
            check=True,
        )

        # Secure the connection
        subprocess.run(
            [
                "sudo",
                "nmcli",
                "connection",
                "modify",
                "hotspot",
                "802-11-wireless.mode",
                "ap",
                "802-11-wireless.band",
                "bg",
                "ipv4.addresses",
                "192.168.1.1/24",
                "ipv4.method",
                "manual",
                "wifi-sec.key-mgmt",
                "wpa-psk",
                "wifi-sec.psk",
                "phosphobot123",
            ],
            check=True,
        )

        # Bring up the hotspot connection
        subprocess.run(["sudo", "nmcli", "connection", "up", "hotspot"], check=True)

    except Exception as e:
        logger.error(f"Error in hotspot background task: {str(e)}")


@background_task_log_exceptions
async def connect_to_network_bg(ssid: str, password: str):
    """Background task for network connection"""

    try:
        # Disconnect from current network
        subprocess.run(
            ["sudo", "nmcli", "device", "disconnect", "wlan0"],
            check=True,
        )

        # Wait for disconnect to complete
        await asyncio.sleep(3)

        # Connect to new network
        subprocess.run(
            [
                "sudo",
                "nmcli",
                "device",
                "wifi",
                "connect",
                ssid,
                "password",
                password,
            ],
            check=True,
        )

        logger.info(f"Connected to network: {ssid}")

    except Exception as e:
        logger.warning(f"Error in network connection background task: {str(e)}")
        await setup_hotspot_bg()


@router.post("/network/hotspot", response_model=StatusResponse)
async def activate_hotspot(background_tasks: BackgroundTasks):
    """
    Endpoint to activate the hotspot on the Raspberry Pi.
    Returns immediately and performs setup in the background.
    """
    if not is_running_on_pi():
        raise HTTPException(
            status_code=400, detail="This endpoint is only available on Raspberry Pi"
        )
    background_tasks.add_task(setup_hotspot_bg)

    return StatusResponse(
        status="ok",
        message="Hotspot activation started. The hotspot is being configured and will be available shortly",
        SSID="phosphobot",
        connect_and_visit="http://phosphobot.local",
    )  # type: ignore


@router.post("/network/connect", response_model=StatusResponse)
async def switch_to_network(
    credentials: NetworkCredentials, background_tasks: BackgroundTasks
):
    """
    Endpoint to connect phosphobot to a new network.
    Returns immediately and performs connection in the background.
    Will fallback to the hotspot if it fails to connect.
    """
    if not is_running_on_pi():
        raise HTTPException(
            status_code=400, detail="This endpoint is only available on Raspberry Pi"
        )
    background_tasks.add_task(
        connect_to_network_bg, credentials.ssid, credentials.password
    )

    return StatusResponse(
        status="ok",
        message=f"Attempting to connect to network: {credentials.ssid}. Will fallback to hotspot if unsuccessful",
    )


@router.post("/network/scan-devices", response_model=ScanNetworkResponse)
async def list_local_network_ips(
    query: Optional[ScanNetworkRequest] = None,
) -> ScanNetworkResponse:
    """
    Endpoint to list all IP addresses on the local network.
    Returns a list of IP addresses.
    """
    if query is None:
        query = ScanNetworkRequest(robot_name=None)
    subnet = get_local_subnet()
    if not subnet:
        raise HTTPException(
            status_code=400,
            detail="Unable to determine local network subnet. Ensure you are connected to a network.",
        )
    devices = await scan_network_devices(subnet)
    # TODO: use robot name to filter devices
    return ScanNetworkResponse(
        devices=devices,
        subnet=subnet,
    )


@router.post("/local/scan-devices", response_model=ScanDevicesResponse)
async def list_connected_devices() -> ScanDevicesResponse:
    """
    Endpoint to list all devices connected to the system.
    """
    available_ports = list_ports.comports()
    return ScanDevicesResponse(
        devices=[
            LocalDevice(
                name=port.name,
                device=port.device,
                serial_number=port.serial_number,
                pid=port.pid,
                interface=port.interface,
            )
            for port in available_ports
        ]
    )
