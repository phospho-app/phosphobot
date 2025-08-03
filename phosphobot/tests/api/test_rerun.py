"""
Integration tests for Rerun visualization during recording.

```
make test_server
uv run pytest -s tests/api/test_rerun.py
```
"""

import json
import time
from unittest.mock import patch, MagicMock

import pytest
import requests
from loguru import logger

BASE_URL = "http://127.0.0.1:8080"


def test_start_recording_with_rerun_enabled():
    """Test that recording can start with rerun visualization enabled."""
    dataset_name = "test_rerun_enabled_dataset"

    response = requests.post(
        f"{BASE_URL}/recording/start",
        json={
            "dataset_name": dataset_name,
            "episode_format": "lerobot_v2",
            "enable_rerun_visualization": True,
        },
    )

    assert (
        response.status_code == 200
    ), f"Failed to start recording with rerun: {response.text}"
    assert (
        response.json().get("status") == "ok"
    ), f"Recording start not ok: {response.text}"

    logger.info("[TEST] Recording with rerun visualization started successfully")

    # Stop recording immediately
    stop_response = requests.post(
        f"{BASE_URL}/recording/stop",
        json={"save": False},  # Don't save for test
    )

    assert (
        stop_response.status_code == 200
    ), f"Failed to stop recording: {stop_response.text}"
    logger.success("[TEST_SUCCESS] Recording with rerun enabled works")


def test_start_recording_with_rerun_disabled():
    """Test that recording works with rerun visualization disabled (default behavior)."""
    dataset_name = "test_rerun_disabled_dataset"

    # Test explicit disable
    response = requests.post(
        f"{BASE_URL}/recording/start",
        json={
            "dataset_name": dataset_name,
            "episode_format": "lerobot_v2",
            "enable_rerun_visualization": False,
        },
    )

    assert (
        response.status_code == 200
    ), f"Failed to start recording without rerun: {response.text}"
    assert (
        response.json().get("status") == "ok"
    ), f"Recording start not ok: {response.text}"

    # Stop recording
    stop_response = requests.post(f"{BASE_URL}/recording/stop", json={"save": False})

    assert (
        stop_response.status_code == 200
    ), f"Failed to stop recording: {stop_response.text}"
    logger.success("[TEST_SUCCESS] Recording with rerun disabled works")


def test_start_recording_default_rerun_behavior():
    """Test that recording works with default behavior (rerun disabled by default)."""
    dataset_name = "test_rerun_default_dataset"

    # Test without specifying enable_rerun_visualization (should default to False)
    response = requests.post(
        f"{BASE_URL}/recording/start",
        json={
            "dataset_name": dataset_name,
            "episode_format": "lerobot_v2",
            # No enable_rerun_visualization field
        },
    )

    assert (
        response.status_code == 200
    ), f"Failed to start recording with default rerun: {response.text}"
    assert (
        response.json().get("status") == "ok"
    ), f"Recording start not ok: {response.text}"

    # Stop recording
    stop_response = requests.post(f"{BASE_URL}/recording/stop", json={"save": False})

    assert (
        stop_response.status_code == 200
    ), f"Failed to stop recording: {stop_response.text}"
    logger.success("[TEST_SUCCESS] Recording with default rerun behavior works")


def test_rerun_parameter_validation():
    """Test that the rerun parameter is properly validated."""
    dataset_name = "test_rerun_validation_dataset"

    # Test with invalid rerun parameter type (should work or fail gracefully)
    response = requests.post(
        f"{BASE_URL}/recording/start",
        json={
            "dataset_name": dataset_name,
            "episode_format": "lerobot_v2",
            "enable_rerun_visualization": "invalid_string",  # Should be boolean
        },
    )

    # The response should either succeed (if coerced to bool) or fail with validation error
    if response.status_code == 200:
        # If it succeeds, stop the recording
        requests.post(f"{BASE_URL}/recording/stop", json={"save": False})
        logger.info("[TEST] Invalid rerun parameter was coerced successfully")
    else:
        # If it fails, it should be a validation error
        assert (
            response.status_code == 422
        ), f"Expected validation error, got: {response.status_code}"
        logger.info("[TEST] Invalid rerun parameter properly rejected")

    logger.success("[TEST_SUCCESS] Rerun parameter validation works")


@patch("phosphobot.rerun_visualizer.rerun")
def test_rerun_graceful_failure_when_sdk_missing(mock_rerun):
    """Test that RerunVisualizer handles missing rerun-sdk gracefully."""
    # Mock the import to raise ImportError
    mock_rerun.side_effect = ImportError("No module named 'rerun'")

    dataset_name = "test_rerun_missing_sdk_dataset"

    response = requests.post(
        f"{BASE_URL}/recording/start",
        json={
            "dataset_name": dataset_name,
            "episode_format": "lerobot_v2",
            "enable_rerun_visualization": True,  # Try to enable rerun
        },
    )

    # Should still work even if rerun is not available
    assert (
        response.status_code == 200
    ), f"Recording should work even without rerun SDK: {response.text}"
    assert (
        response.json().get("status") == "ok"
    ), f"Recording start not ok: {response.text}"

    # Stop recording
    stop_response = requests.post(f"{BASE_URL}/recording/stop", json={"save": False})

    assert (
        stop_response.status_code == 200
    ), f"Failed to stop recording: {stop_response.text}"
    logger.success("[TEST_SUCCESS] Recording gracefully handles missing rerun SDK")


def test_short_recording_with_rerun():
    """Test a short recording session with rerun to verify end-to-end functionality."""
    dataset_name = "test_rerun_short_recording"

    # Start recording with rerun
    start_response = requests.post(
        f"{BASE_URL}/recording/start",
        json={
            "dataset_name": dataset_name,
            "episode_format": "lerobot_v2",
            "enable_rerun_visualization": True,
        },
    )

    assert (
        start_response.status_code == 200
    ), f"Failed to start recording: {start_response.text}"
    logger.info("[TEST] Short recording with rerun started")

    # Let it record for a brief moment
    time.sleep(0.5)

    # Make a small movement to generate some data
    try:
        move_response = requests.post(
            f"{BASE_URL}/move/absolute", json={"x": 1, "y": 1, "z": 1, "open": 0}
        )
        logger.info(f"[TEST] Move command response: {move_response.status_code}")
    except Exception as e:
        logger.info(f"[TEST] Move command failed (expected in sim): {e}")

    # Wait a bit more
    time.sleep(0.5)

    # Stop recording
    stop_response = requests.post(
        f"{BASE_URL}/recording/stop",
        json={"save": False},  # Don't save to avoid filling disk
    )

    assert (
        stop_response.status_code == 200
    ), f"Failed to stop recording: {stop_response.text}"
    logger.success("[TEST_SUCCESS] Short recording with rerun completed successfully")
