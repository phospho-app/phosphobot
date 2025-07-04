"""
Integration tests for the Websocket and UDP.

```
make test_server
uv run pytest -s
```
"""

import asyncio
import json
import socket
import sys
import time

import numpy as np
import pytest
import requests  # type: ignore
import websockets
from loguru import logger

BASE_URL = "http://127.0.0.1:8080"
BASE_WS_URI = "ws://127.0.0.1:8080"
WEBSOCKET_TEST_TIME = 10  # seconds
UDP_TEST_TIME = 10  # seconds


# Ensure loguru logs appear in pytest output
@pytest.fixture(scope="module", autouse=True)
def configure_logger():
    """
    Plain-text logger (no ANSI colors), with a minimal format
    """
    logger.remove()
    logger.add(
        sys.stderr, colorize=False, level="DEBUG", format="{time} | {level} | {message}"
    )


@pytest.mark.asyncio
async def send_data(websocket, total_seconds, send_frequency):
    """
    Sends messages at `send_frequency` until `total_seconds` have elapsed.
    """
    sample_control_data = {
        "x": 0.001,
        "y": 0.0,
        "z": 0.0,
        "rx": 0.0,
        "ry": 0.0,
        "rz": 0.0,
        "open": 0.0,
    }
    start_time = time.monotonic()

    while (time.monotonic() - start_time) < total_seconds:
        # 1) Send the control data
        await websocket.send(json.dumps(sample_control_data))
        # 2) Sleep to maintain desired frequency
        await asyncio.sleep(1 / send_frequency)


@pytest.mark.asyncio
async def receive_data(websocket, total_seconds, nb_actions_history):
    """
    Continuously receives messages from the server until `total_seconds` have elapsed.
    Extracts nb_actions_received and appends it to `nb_actions_history`.
    """
    start_time = time.monotonic()

    while (time.monotonic() - start_time) < total_seconds:
        try:
            # Wait for incoming message (short timeout so we keep reading often)
            status_message = await asyncio.wait_for(websocket.recv(), timeout=0.3)
            data = json.loads(status_message)

            nb_actions = data.get("nb_actions_received")
            if nb_actions is not None:
                nb_actions_history.append(nb_actions)
                logger.info(f"nb_actions_received={nb_actions}")

        except asyncio.TimeoutError:
            pass


async def send_udp_data(host, port, total_seconds, send_frequency):
    """
    Sends UDP messages at `send_frequency` until `total_seconds` have elapsed.
    Returns the number of messages sent and received responses.
    """
    sample_control_data = {
        "x": 0.001,
        "y": 0.0,
        "z": 0.0,
        "rx": 0.0,
        "ry": 0.0,
        "rz": 0.0,
        "open": 0.0,
        "source": "right",
        "timestamp": None,
    }

    messages_sent = 0
    responses_received = 0
    nb_actions_history = []

    # Create UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(0.1)  # 100ms timeout for receives

    try:
        start_time = time.monotonic()

        while (time.monotonic() - start_time) < total_seconds:
            # Add current timestamp
            sample_control_data["timestamp"] = time.time()

            # Send UDP message
            message = json.dumps(sample_control_data).encode("utf-8")
            sock.sendto(message, (host, port))
            messages_sent += 1

            # Try to receive response (non-blocking)
            try:
                response_data, _ = sock.recvfrom(1024)
                response = json.loads(response_data.decode("utf-8"))
                responses_received += 1

                # Check for nb_actions_received
                nb_actions = response.get("nb_actions_received")
                if nb_actions is not None:
                    nb_actions_history.append(nb_actions)
                    logger.info(f"UDP nb_actions_received={nb_actions}")

            except socket.timeout:
                pass  # No response received, continue
            except json.JSONDecodeError:
                pass  # Invalid JSON response, continue

            # Sleep to maintain desired frequency
            await asyncio.sleep(1 / send_frequency)

    finally:
        sock.close()

    return messages_sent, responses_received, nb_actions_history


@pytest.mark.asyncio
async def test_send_messages(send_frequency=30, total_seconds=WEBSOCKET_TEST_TIME):
    """
    - Connects to /move/teleop/ws endpoint.
    - Spawns two async tasks:
        1) Send messages at `send_frequency` for `total_seconds`.
        2) Continuously receive and parse nb_actions_received from the server.
    - Computes the mean of all nb_actions_received.
    """

    # First, move/init so that the robot is ready to receive commands
    requests.post(f"{BASE_URL}/move/init")
    # Give some time for the robot to move to the initial position
    await asyncio.sleep(1)

    # Shared list for collecting nb_actions_received
    nb_actions_history = []

    async with websockets.connect(f"{BASE_WS_URI}/move/teleop/ws") as websocket:
        logger.success("[TEST] Connected to WebSocket")

        # Create and run tasks concurrently
        send_task = asyncio.create_task(
            send_data(websocket, total_seconds, send_frequency)
        )
        receive_task = asyncio.create_task(
            receive_data(websocket, total_seconds, nb_actions_history)
        )

        # Wait for both tasks to finish
        await asyncio.gather(send_task, receive_task)

    # After completion, compute average if we have data
    if nb_actions_history:
        avg_nb_actions_received = float(np.mean(nb_actions_history))
    else:
        avg_nb_actions_received = 0.0

    logger.info(
        f"[TEST_PERFORMANCE_{send_frequency}Hz] Average nb_actions_received per second at {send_frequency} Hz: {avg_nb_actions_received:.2f}"
    )

    assert avg_nb_actions_received > 5, (
        f"[TEST_FAILED] Average nb_actions_received per second: {avg_nb_actions_received}",
        "Expected an average of nb_actions_received above 5 per second",
    )

    logger.success("[TEST_SUCCESS] Websocket test completed successfully")


# Do the same with 500 Hz
@pytest.mark.asyncio
async def test_send_messages_500hz(
    send_frequency=500, total_seconds=WEBSOCKET_TEST_TIME
):
    await test_send_messages(send_frequency=send_frequency, total_seconds=total_seconds)


@pytest.mark.asyncio
async def test_send_messages_500hz_while_recording(
    send_frequency=500, total_seconds=WEBSOCKET_TEST_TIME
):
    """
    Performance test:
      1) Start recording (via HTTP POST /recording/start).
      2) Send data at 500Hz for total_seconds to /move/teleop/ws.
      3) Stop recording (via HTTP POST /recording/stop).
      4) Print the avegerage nb_actions_received per second.
    """

    # First, move/init so that the robot is ready to receive commands
    requests.post(f"{BASE_URL}/move/init")
    # Give some time for the robot to move to the initial position
    await asyncio.sleep(1)

    ##################
    # 1) Start recording
    ##################
    dataset_name = "test_lerobot_ws_dataset"
    start_payload = {"dataset_name": dataset_name, "episode_format": "lerobot_v2"}

    # USE WEBSOCKET CONNEXION HERE
    start_response = requests.post(f"{BASE_URL}/recording/start", json=start_payload)
    assert (
        start_response.status_code == 200
    ), f"Failed to start recording: {start_response.text}"
    assert (
        start_response.json().get("status") == "ok"
    ), f"Recording start not ok: {start_response.text}"

    logger.info("[TEST] Recording started successfully")

    ##################
    # 2) WebSocket concurrency: send & receive
    ##################
    nb_actions_history = []

    async with websockets.connect(f"{BASE_WS_URI}/move/teleop/ws") as websocket:
        logger.info("[TEST] Connected to WebSocket")

        # Create tasks
        send_task = asyncio.create_task(
            send_data(websocket, total_seconds, send_frequency)
        )
        receive_task = asyncio.create_task(
            receive_data(websocket, total_seconds, nb_actions_history)
        )

        await asyncio.gather(send_task, receive_task)

    # Compute average if we have data
    if nb_actions_history:
        avg_nb_actions_received = float(np.mean(nb_actions_history))
    else:
        avg_nb_actions_received = 0.0

    logger.info(
        f"[TEST_RECORDING_PERFORMANCE_{send_frequency}Hz] Average nb_actions_received per second at {send_frequency} Hz while recording: "
        f"{avg_nb_actions_received:.2f}"
    )
    # Optional assertion for performance
    assert avg_nb_actions_received > 5, (
        f"[TEST_FAILED] Average nb_actions_received is {avg_nb_actions_received}, "
        "expected to be > 5"
    )

    ##################
    # 3) Stop recording
    ##################
    stop_payload = {"save": False, "episode_format": "lerobot_v2"}
    stop_response = requests.post(f"{BASE_URL}/recording/stop", json=stop_payload)
    assert (
        stop_response.status_code == 200
    ), f"Failed to stop recording: {stop_response.text}"
    logger.info("[TEST] Recording stopped successfully")

    ##################
    # 4) Print The mean of the nb_actions_received
    ##################
    logger.info(
        f"[TEST_RECORDING_PERFORMANCE_{send_frequency}Hz] Average nb_actions_received per second at {send_frequency} Hz while recording: "
        f"{avg_nb_actions_received:.2f}"
    )

    logger.success("[TEST_SUCCESS] Websocket test completed successfully")


# UDP Tests
@pytest.mark.asyncio
async def test_udp_send_messages(send_frequency=30, total_seconds=UDP_TEST_TIME):
    """
    UDP Performance test:
      1) Start UDP server
      2) Send UDP messages at specified frequency
      3) Collect responses and compute average nb_actions_received
      4) Stop UDP server
    """
    # First, move/init so that the robot is ready to receive commands
    requests.post(f"{BASE_URL}/move/init")
    await asyncio.sleep(1)

    # Start UDP server
    udp_response = requests.post(f"{BASE_URL}/move/teleop/udp")
    assert (
        udp_response.status_code == 200
    ), f"Failed to start UDP server: {udp_response.text}"

    udp_info = udp_response.json()
    host = udp_info["host"]
    port = udp_info["port"]

    logger.info(f"[TEST] UDP server started on {host}:{port}")

    try:
        # Send UDP messages
        messages_sent, responses_received, nb_actions_history = await send_udp_data(
            host, port, total_seconds, send_frequency
        )

        # Compute statistics
        if nb_actions_history:
            avg_nb_actions_received = float(np.mean(nb_actions_history))
        else:
            avg_nb_actions_received = 0.0

        response_rate = (
            (responses_received / messages_sent * 100) if messages_sent > 0 else 0
        )

        logger.info(
            f"[TEST_UDP_PERFORMANCE_{send_frequency}Hz] "
            f"Messages sent: {messages_sent}, "
            f"Responses received: {responses_received} ({response_rate:.1f}%), "
            f"Average nb_actions_received per second: {avg_nb_actions_received:.2f}"
        )

        assert avg_nb_actions_received > 5, (
            f"[TEST_FAILED] Average nb_actions_received per second: {avg_nb_actions_received}",
            "Expected an average of nb_actions_received above 5 per second",
        )

        logger.success("[TEST_SUCCESS] UDP test completed successfully")

    finally:
        # Stop UDP server
        stop_response = requests.post(f"{BASE_URL}/move/teleop/udp/stop")
        assert (
            stop_response.status_code == 200
        ), f"Failed to stop UDP server: {stop_response.text}"
        logger.info("[TEST] UDP server stopped")


@pytest.mark.asyncio
async def test_udp_send_messages_500hz(send_frequency=500, total_seconds=UDP_TEST_TIME):
    await test_udp_send_messages(
        send_frequency=send_frequency, total_seconds=total_seconds
    )


@pytest.mark.asyncio
async def test_udp_send_messages_1000hz(
    send_frequency=1000, total_seconds=UDP_TEST_TIME
):
    await test_udp_send_messages(
        send_frequency=send_frequency, total_seconds=total_seconds
    )


@pytest.mark.asyncio
async def test_udp_send_messages_500hz_while_recording(
    send_frequency=500, total_seconds=UDP_TEST_TIME
):
    """
    UDP Performance test with recording:
      1) Start recording
      2) Start UDP server
      3) Send UDP messages at 500Hz
      4) Stop UDP server
      5) Stop recording
    """
    # First, move/init so that the robot is ready to receive commands
    requests.post(f"{BASE_URL}/move/init")
    await asyncio.sleep(1)

    ##################
    # 1) Start recording
    ##################
    dataset_name = "test_lerobot_udp_dataset"
    start_payload = {"dataset_name": dataset_name, "episode_format": "lerobot_v2"}

    start_response = requests.post(f"{BASE_URL}/recording/start", json=start_payload)
    assert (
        start_response.status_code == 200
    ), f"Failed to start recording: {start_response.text}"
    assert (
        start_response.json().get("status") == "ok"
    ), f"Recording start not ok: {start_response.text}"

    logger.info("[TEST] Recording started successfully")

    ##################
    # 2) Start UDP server
    ##################
    udp_response = requests.post(f"{BASE_URL}/move/teleop/udp")
    assert (
        udp_response.status_code == 200
    ), f"Failed to start UDP server: {udp_response.text}"

    udp_info = udp_response.json()
    host = udp_info["host"]
    port = udp_info["port"]

    logger.info(f"[TEST] UDP server started on {host}:{port}")

    try:
        ##################
        # 3) Send UDP messages
        ##################
        messages_sent, responses_received, nb_actions_history = await send_udp_data(
            host, port, total_seconds, send_frequency
        )

        # Compute statistics
        if nb_actions_history:
            avg_nb_actions_received = float(np.mean(nb_actions_history))
        else:
            avg_nb_actions_received = 0.0

        response_rate = (
            (responses_received / messages_sent * 100) if messages_sent > 0 else 0
        )

        logger.info(
            f"[TEST_UDP_RECORDING_PERFORMANCE_{send_frequency}Hz] "
            f"Messages sent: {messages_sent}, "
            f"Responses received: {responses_received} ({response_rate:.1f}%), "
            f"Average nb_actions_received per second while recording: {avg_nb_actions_received:.2f}"
        )

        assert avg_nb_actions_received > 5, (
            f"[TEST_FAILED] Average nb_actions_received per second: {avg_nb_actions_received}",
            "Expected an average of nb_actions_received above 5 per second",
        )

    finally:
        ##################
        # 4) Stop UDP server
        ##################
        stop_response = requests.post(f"{BASE_URL}/move/teleop/udp/stop")
        assert (
            stop_response.status_code == 200
        ), f"Failed to stop UDP server: {stop_response.text}"
        logger.info("[TEST] UDP server stopped")

    ##################
    # 5) Stop recording
    ##################
    stop_payload = {"save": False, "episode_format": "lerobot_v2"}
    stop_response = requests.post(f"{BASE_URL}/recording/stop", json=stop_payload)
    assert (
        stop_response.status_code == 200
    ), f"Failed to stop recording: {stop_response.text}"
    logger.info("[TEST] Recording stopped successfully")

    logger.success("[TEST_SUCCESS] UDP recording test completed successfully")
