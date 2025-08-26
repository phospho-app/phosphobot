"""
Integration tests for the API.

```
make test_server
uv run pytest -s
```
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import pytest
import requests  # type: ignore
from dotenv import load_dotenv
from huggingface_hub import HfApi, login
from loguru import logger
from phosphobot.utils import get_home_app_path, parse_hf_username_or_orgid

BASE_URL = "http://127.0.0.1:8080"
load_dotenv()  # take environment variables from .env


@pytest.fixture(scope="module")
def create_hf_token():
    """
    Write the hugging face token to a file so we can login.
    """
    token_path = get_home_app_path() / "huggingface.token"
    logger.info(f"Writing token to file: {token_path}")
    # Write the token in the file
    with open(token_path, "w") as f:
        f.write(os.environ["HF_TOKEN"])


@pytest.fixture(scope="module", autouse=True)
def configure_logger():
    """
    Plain-text logger (no ANSI colors), with a minimal format
    """
    logger.remove()
    logger.add(
        sys.stderr, colorize=False, level="DEBUG", format="{time} | {level} | {message}"
    )


@pytest.fixture(scope="module")
def get_initial_end_effector_state():
    """Fixture to get the initial end-effector state.
    The robot must return to its initial position with a tolerance of 1 centimeter."""
    status_response = requests.get(f"{BASE_URL}/status")
    assert (
        status_response.status_code == 200
    ), f"[TEST_FAILURE] Failed to ping server: {status_response.text}"
    assert (
        status_response.json()["status"] == "ok"
    ), "[TEST_FAILURE] Server status is not OK"
    logger.info(f"[TEST] Status response: {status_response.json()}")

    # Move to intial position
    requests.post(f"{BASE_URL}/move/init")
    # Wait for the robot to reach the initial position
    time.sleep(2)

    initial_position_response = requests.post(f"{BASE_URL}/end-effector/read")
    assert (
        initial_position_response.status_code == 200
    ), f"[TEST_FAILURE] Failed to get end-effector state: {initial_position_response.text}"

    initial_state = initial_position_response.json()
    logger.info(f"Initial End-Effector Position: {initial_state['x']}")
    return initial_state


def test_status_endpoint():
    """Test the status endpoint."""
    response = requests.get(f"{BASE_URL}/status")
    assert (
        response.status_code == 200
    ), f"[TEST_FAILURE] Failed to ping server: {response.text}"
    assert response.json()["status"] == "ok", "[TEST_FAILURE] Server status is not OK"
    logger.info(f"Status response: {response.json()}")
    logger.success("[TEST_SUCCESS] Status endpoint is working")


def test_move_relative(get_initial_end_effector_state):
    """Test moving the robot in small increments."""
    move_response = requests.post(
        f"{BASE_URL}/move/relative",
        json={"x": 0.02, "y": 0, "z": 0, "rx": 0, "ry": 0, "rz": 0, "open": 0},
    )
    assert (
        move_response.status_code == 200
    ), f"Failed to move robot: {move_response.text}"
    logger.success("[TEST_SUCCESS] Robot moved successfully")

    # Check the end-effector state
    initial_state = get_initial_end_effector_state

    # Perform movement
    requests.post(
        f"{BASE_URL}/move/relative",
        json={"x": 0.02, "y": 0, "z": 0, "rx": 0, "ry": 0, "rz": 0, "open": 0},
    )
    time.sleep(2)
    requests.post(
        f"{BASE_URL}/move/relative",
        json={"x": -0.04, "y": 0, "z": 0, "rx": 0, "ry": 0, "rz": 0, "open": 0},
    )
    time.sleep(2)

    # Check the final end-effector state
    end_position_response = requests.post(f"{BASE_URL}/end-effector/read")
    logger.info(f"Server response: {end_position_response.text}")
    assert (
        end_position_response.status_code == 200
    ), f"[TEST_FAILURE] Failed to get end-effector state: {end_position_response.text}"

    data = end_position_response.json()
    assert isinstance(
        data, dict
    ), f"[TEST_FAILURE] Unexpected response format: {type(data)}\n{data}"
    assert "x" in data, f"[TEST_FAILURE] Unexpected response format: {data}"
    logger.info(f"Received EndPosition: {data['x']}")

    # Ensure the robot returned close to the original position
    assert abs(data["x"] - initial_state["x"]) < 1e-2, (
        f"[TEST_FAILURE] Robot did not return to original position. "
        f"Expected {initial_state['x']}, got {data['x']}"
    )
    logger.success("[TEST_SUCCESS] Robot returned to the initial position")


def start_and_stop_recording(episode_format: Literal["lerobot_v2", "json"]):
    """
    Make a call to start recording for 1 second and then stop it.
    We do not save the data in lerobot-format.
    """
    start_response = requests.post(
        f"{BASE_URL}/recording/start",
        json={"dataset_name": "test_example_dataset", "episode_format": episode_format},
    )

    assert (
        start_response.status_code == 200 and start_response.json()["status"] == "ok"
    ), f"[TEST_FAILURE] Failed to start recording: {start_response.text}"

    logger.info(f"Recording start response: {start_response.json()}")

    # Wait before stopping the recording
    time.sleep(2)

    # Stop the recording
    stop_response = requests.post(
        f"{BASE_URL}/recording/stop",
        json={"save": False, "episode_format": episode_format},
    )
    logger.info(f"Recording stop response: {stop_response.json()}")
    assert (
        stop_response.status_code == 200
    ), f"[TEST_FAILURE] Failed to stop recording: {stop_response.text}"
    logger.info(f"Recording stop response: {stop_response.json()}")
    logger.success("[TEST_SUCCESS] Recording start and stop endpoint is working")
    time.sleep(10)


def test_start_and_stop_recording_lerobot_v2():
    """Test the recording start and stop endpoints."""
    start_and_stop_recording("lerobot_v2")


def test_start_and_stop_recording_json():
    """Test the recording start and stop endpoints."""
    start_and_stop_recording("json")


def wait_for_file(filepath: Path, timeout: int = 20, check_interval: float = 1) -> bool:
    """Wait for a file to exist with timeout."""
    logger.info(f"Waiting for file: {filepath}")
    start_time = time.time()
    while time.time() - start_time < timeout:
        if filepath.exists():
            logger.info(
                f"File found: {filepath} after {time.time() - start_time:.2f} seconds"
            )
            return True
        time.sleep(check_interval)
    logger.error(f"File not found: {filepath} after {timeout} seconds")
    return False


def make_request_with_retry(
    method: str, url: str, json: Optional[dict] = None, max_retries: int = 3
):
    """Make HTTP request with retry logic."""
    for attempt in range(max_retries):
        try:
            response = requests.request(method, url, json=json)
            response.raise_for_status()
            return response
        except Exception as e:
            if attempt == max_retries - 1:
                # Log more information if all retries fail
                logger.error(
                    f"Request failed after {max_retries} attempts. Details: {response.text}\n{e}"
                )
                raise
            logger.warning(
                f"Request failed, retrying... Error: {e}. Details: {response.text}"
            )
            time.sleep(1)


def login_to_hf() -> Optional[tuple[str, str, Optional[str]]]:
    """
    Login to HF and set up the dataset name for CICD testing.
    """

    # Set up dataset name
    gh_branch_path = os.environ.get("BRANCH_NAME")
    gh_commit_id = os.environ.get("COMMIT_ID")
    assert (
        gh_branch_path is not None and gh_commit_id is not None
    ), "[TEST_FAILURE] Branch name or commit id is not set in the environment"

    # Write the hugging face token to a file so we can login.
    token_path = get_home_app_path() / "huggingface.token"
    logger.info(f"Writing token to file: {token_path}")
    # Write the token in the file
    with open(token_path, "w") as f:
        f.write(os.environ["HF_TOKEN"])

    logger.info(f"Branch name: {gh_branch_path}")
    logger.info(f"Commit ID: {gh_commit_id}")

    if gh_branch_path is None or gh_commit_id is None:
        logger.warning("Branch name or commit id is not set in the environment.")

    hf_dataset_name = "dataset_for_testing"

    login(token=os.environ["HF_TOKEN"])

    # Get the name of the current computer (uname)
    computer_name = os.uname().nodename

    hf_branch_path = (
        f"CICD-{gh_branch_path}-{computer_name}-{gh_commit_id}".replace(
            "/", "-"
        ).replace(" ", "-")
        if gh_commit_id
        else None
    )

    logger.info(f"HF Branch path: {gh_branch_path}")

    # Check the hugging face token is accessibles
    hf_token = os.environ.get("HF_TOKEN")
    assert hf_token is not None, "[TEST_FAILURE] HF_TOKEN is not set in the environment"

    api = HfApi()
    user_info = api.whoami(token=hf_token)
    # Get the username or org where we have write access
    username_or_org_id = parse_hf_username_or_orgid(user_info)

    if username_or_org_id is None:
        logger.warning(
            "No user or org with write access found. Wont be able to push to Hugging Face."
        )
        return None

    logger.info(
        f"Pushing dataset to Hugging Face: {username_or_org_id}/{hf_dataset_name}"
    )
    return username_or_org_id, hf_dataset_name, hf_branch_path


def check_if_dataset_is_available_on_hf(
    username_or_org_id: str,
    hf_dataset_name: str,
    hf_branch_path: Optional[str],
    timeout: int = 60,
) -> None:
    # Check the dataset is available on the Hugging Face Hub
    start_time = time.time()
    hf_dataset = None
    while time.time() - start_time < timeout:
        try:
            HfApi().dataset_info(
                f"{username_or_org_id}/{hf_dataset_name}", revision=hf_branch_path
            )
            break
        except Exception as e:
            logger.warning(
                f"Dataset not found on the Hugging Face Hub. Error: {e}. Retrying..."
            )
            time.sleep(5)

    if hf_dataset is None:
        logger.error("[TEST_FAILURE] Dataset is not available on the Hugging Face Hub")

    logger.info("[TEST_SUCCESS] Dataset is available on the Hugging Face Hub")


def check_episodes_stats_file(path: Path):
    # Open file
    with open(path, "r") as f:
        line = f.readline()

    # Check the episodes stats file is not empty
    assert line != "", "[TEST_FAILURE] Episodes stats file is empty"

    # Check the episodes stats file is a valid JSON
    try:
        episodes_stats_parsed: dict = json.loads(line)
    except json.JSONDecodeError:
        raise AssertionError("[TEST_FAILURE] Episodes stats file is not a valid JSON")

    stats: dict = episodes_stats_parsed["stats"]

    logger.debug(f"Stats keys: {stats.keys()}")
    print(f"Stats: {stats}")

    # Check the episodes stats file has the expected keys
    required_keys = [
        "action",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
        "observation.state",
        "observation.images.main",
        "observation.images.secondary_0",
    ]
    for key in required_keys:
        assert (
            key in stats
        ), f"[TEST_FAILURE] Key not found in episodes stats file: {key}"

    # Check that each of the required keys has min, max, mean, std values
    for key in required_keys:
        assert (
            "min" in stats[key]
        ), f"[TEST_FAILURE] Key not found in episodes stats file: {key}"
        assert (
            "max" in stats[key]
        ), f"[TEST_FAILURE] Key not found in episodes stats file: {key}"
        assert (
            "mean" in stats[key]
        ), f"[TEST_FAILURE] Key not found in episodes stats file: {key}"
        assert (
            "std" in stats[key]
        ), f"[TEST_FAILURE] Key not found in episodes stats file: {key}"


def check_stats_file(path: Path):
    # Open file
    with open(path, "r") as f:
        stats = f.read()

    # Check the stats file is not empty
    assert stats != "", "[TEST_FAILURE] Stats file is empty"

    # Check the stats file is a valid JSON
    try:
        stats_parsed: dict = json.loads(stats)
    except json.JSONDecodeError:
        raise AssertionError("[TEST_FAILURE] Stats file is not a valid JSON")

    # Check the stats file has the expected keys
    required_keys = [
        "action",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
        "observation.state",
        "observation.images.main",
        "observation.images.secondary_0",
    ]
    for key in required_keys:
        assert key in stats_parsed, f"[TEST_FAILURE] Key not found in stats file: {key}"

    # Check that each of the required keys has min, max, mean, std values
    for key in required_keys:
        assert (
            "min" in stats_parsed[key]
        ), f"[TEST_FAILURE] Key not found in stats file: {key}"
        assert (
            "max" in stats_parsed[key]
        ), f"[TEST_FAILURE] Key not found in stats file: {key}"
        assert (
            "mean" in stats_parsed[key]
        ), f"[TEST_FAILURE] Key not found in stats file: {key}"
        assert (
            "std" in stats_parsed[key]
        ), f"[TEST_FAILURE] Key not found in stats file: {key}"

    # Check that max of frame_index is different from max index
    assert (
        stats_parsed["frame_index"]["max"] != stats_parsed["index"]["max"]
    ), "[TEST_FAILURE] Frame index max is equal to index max"

    # Check that max of episode_index is 1
    assert stats_parsed["episode_index"]["max"] == [
        1
    ], "[TEST_FAILURE] Episode index max is not 1"

    # Check that action and observation.state are not null arrays
    assert stats_parsed["action"]["mean"] != [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ], f"[TEST_FAILURE] Action mean is {stats_parsed['action']['mean']}"
    assert (
        stats_parsed["observation.state"]["mean"]
        != [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    ), f"[TEST_FAILURE] Observation state mean is {stats_parsed['observation.state']['mean']}"

    # Check that observation.images.main max is 1
    assert stats_parsed["observation.images.main"]["max"] == [
        [[1.0]],
        [[1.0]],
        [[1.0]],
    ], "[TEST_FAILURE] Observation images main max is not [[1.0], [1.0], [1.0]]"


def check_episodes_file(path: Path):
    # Open file and read the first line
    with open(path, "r") as f:
        episodes = f.readline()

    # Check the episodes file is not empty
    assert episodes != "", "[TEST_FAILURE] Episodes file is empty"

    # Check the episodes file is a valid JSON
    try:
        episodes_parsed: dict = json.loads(episodes)
    except json.JSONDecodeError:
        raise AssertionError("[TEST_FAILURE] Episodes file is not a valid JSON")

    # Check the episodes file has the expected keys
    required_keys = [
        "episode_index",
        "tasks",
        "length",
    ]

    for key in required_keys:
        assert (
            key in episodes_parsed
        ), f"[TEST_FAILURE] Key not found in episodes file: {key}"

    # Check the episodes file has the expected values
    assert (
        episodes_parsed["episode_index"] == 0
    ), "[TEST_FAILURE] Episode index is not 0"

    assert (
        episodes_parsed["length"] >= 10
    ), "[TEST_FAILURE] Episode length is smaller than 10"

    # Check the tasks key is a list of a string
    assert isinstance(
        episodes_parsed["tasks"], list
    ), "[TEST_FAILURE] Tasks key is not a list"
    assert all(
        isinstance(task, str) for task in episodes_parsed["tasks"]
    ), "[TEST_FAILURE] Tasks key is not a list of strings"


def check_info_file(path: Path):
    # Open file
    with open(path, "r") as f:
        info = f.read()

    # Check the info file is not empty
    assert info != "", "[TEST_FAILURE] Info file is empty"

    # Check the info file is a valid JSON
    try:
        info_parsed: dict = json.loads(info)
    except json.JSONDecodeError:
        raise AssertionError("[TEST_FAILURE] Info file is not a valid JSON")

    # Check the info file has the expected keys
    required_keys = [
        "robot_type",
        "codebase_version",
        "total_episodes",
        "total_frames",
        "total_tasks",
        "total_videos",
        "total_chunks",
        "chunks_size",
        "fps",
        "splits",
        "data_path",
        "video_path",
        "features",
    ]
    for key in required_keys:
        assert key in info_parsed, f"[TEST_FAILURE] Key not found in info file: {key}"

    assert info_parsed["total_episodes"] == 2, "[TEST_FAILURE] Total episodes is not 2"
    assert (
        info_parsed["total_frames"] >= 30
    ), "[TEST_FAILURE] Total frames is lower than 30"
    # assert key["total_videos"] == 4, "[TEST_FAILURE] Total videos is not 4"
    assert info_parsed["total_tasks"] == 1, "[TEST_FAILURE] Total tasks is not 1"

    # Check that features key has the correct params
    required_features_keys = [
        "action",
        "timestamp",
        "episode_index",
        "frame_index",
        "task_index",
        "index",
        "observation.state",
        "observation.images.main",
        "observation.images.secondary_0",
    ]

    features_dict: dict = info_parsed["features"]
    for key in required_features_keys:
        assert (
            key in features_dict
        ), f"[TEST_FAILURE] Key: {key} not found in features dict: {features_dict}"


def check_tasks_file(path: Path):
    # Open file
    with open(path, "r") as f:
        tasks = f.read()

    # Check the tasks file is not empty
    assert tasks != "", "[TEST_FAILURE] Tasks file is empty"

    # Check the tasks file is a valid JSON
    try:
        tasks_parsed: dict = json.loads(tasks)
    except json.JSONDecodeError:
        raise AssertionError("[TEST_FAILURE] Tasks file is not a valid JSON")

    # Check the tasks file has the expected keys and values
    # Tasks should be equal to {"task_index":0,"task":"None"}

    assert tasks_parsed["task_index"] == 0, "[TEST_FAILURE] Task index is not 0"
    assert tasks_parsed["task"] == "None", "[TEST_FAILURE] Task is not None"


def check_parquet_file(path: Path):
    # Check the parquet file is a valid parquet
    try:
        parquet_parsed: pd.DataFrame = pd.read_parquet(path=path)
    except Exception as e:
        raise AssertionError("[TEST_FAILURE] Parquet file is not a valid parquet")

    # Check the parquet file has the expected columns
    required_columns = [
        "action",
        "observation.state",
        "timestamp",
        "task_index",
        "episode_index",
        "frame_index",
        "index",
    ]

    for column in required_columns:
        assert (
            column in parquet_parsed.columns
        ), f"[TEST_FAILURE] Column not found in parquet file: {column}"

    # Check that action and observation.state are not all zerors arrays
    has_non_zero_action = not (
        parquet_parsed["action"].apply(lambda x: all(x == 0))
    ).all()
    assert has_non_zero_action, "[TEST_FAILURE] All action arrays are [0,0,0,0,0,0]"

    # Check if any row in 'observation.state' is not all zeros
    has_non_zero_state = not (
        parquet_parsed["observation.state"].apply(lambda x: all(x == 0))
    ).all()
    assert (
        has_non_zero_state
    ), "[TEST_FAILURE] All observation.state arrays are [0,0,0,0,0,0]"

    # Check that observation.state is action with a lag of 1
    def compare_arrays(arr1, arr2):
        return np.allclose(np.array(arr1), np.array(arr2), rtol=1e-5, atol=1e-5)

    # Apply comparison row by row with shift
    assert all(
        compare_arrays(action, obs_state)
        for action, obs_state in zip(
            parquet_parsed["action"][:-1],  # exclude last row
            parquet_parsed["observation.state"][1:],  # exclude first row
        )
    ), "[TEST_FAILURE] Action is not equal to observation.state with a lag of 1"
    # Check that all values of episode_index are 1
    assert (
        parquet_parsed["episode_index"] == 1
    ).all(), "[TEST_FAILURE] Episode index is not 1"

    # Check that max value of index is different from max value of frame_index
    assert (
        parquet_parsed["index"].max() != parquet_parsed["frame_index"].max()
    ), "[TEST_FAILURE] Index max is equal to frame_index max"


def save_lerobot_recording(format: Literal["lerobot_v2", "lerobot_v2.1"]):
    """
    Make a call to start recording for 1 with one classic camera during 1 second and then stop it.
    We save the data in lerobot-format and check that the files are created.
    """
    # Log environment information
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Home app path: {get_home_app_path()}")
    logger.info(f"Environment variables: {dict(os.environ)}")

    # Write the hugging face token to a file so we can login.
    token_path = get_home_app_path() / "huggingface.token"
    logger.info(f"Writing token to file: {token_path}")
    # Write the token in the file
    with open(token_path, "w") as f:
        f.write(os.environ["HF_TOKEN"])

    # Set up dataset name
    gh_branch_path = os.environ.get("BRANCH_NAME")
    gh_commit_id = os.environ.get("COMMIT_ID")

    # Print the current directory
    logger.info(f"Current directory: {os.getcwd()}")

    if gh_branch_path is None or gh_commit_id is None:
        logger.warning("Branch name or commit id is not set in the environment.")

    login_result = login_to_hf()
    if login_result is None:
        raise AssertionError(
            "[TEST_FAILURE] Failed to login to Hugging Face. Check your token."
        )

    username_or_org_id, hf_dataset_name, hf_branch_path = login_result
    if hf_branch_path is None:
        raise AssertionError(
            "[TEST_FAILURE] Failed to create branch path. Check your branch name and commit id."
        )
    # Delete branch if it already exists
    try:
        api = HfApi()
        api.delete_branch(
            repo_id=f"{username_or_org_id}/{hf_dataset_name}", branch=hf_branch_path
        )
    except Exception as e:
        logger.info(f"No branch to delete. This is a good thing. Error: {e}")

    def record_movement():
        # Start recording
        start_response = make_request_with_retry(
            "POST",
            f"{BASE_URL}/recording/start",
            json={
                "dataset_name": hf_dataset_name,
                "episode_format": format,
                "branch_path": hf_branch_path,
            },
        )
        logger.info(f"Recording start response: {start_response.text}")

        time.sleep(1)

        make_request_with_retry(
            "POST",
            f"{BASE_URL}/move/absolute",
            json={"x": 5, "y": 5, "z": 5, "open": 0},
        )

        # Wait before stopping the recording
        time.sleep(1)

        # Stop recording
        make_request_with_retry(
            "POST",
            f"{BASE_URL}/recording/stop",
            json={"save": True},
        )

        # Wait for files to be created
        time.sleep(10)

    # Record movement twice
    record_movement()
    record_movement()

    # Set up paths using pathlib
    recordings_path = get_home_app_path() / "recordings"
    dataset_path = recordings_path / format / f"{hf_dataset_name}"
    meta_folder = dataset_path / "meta"
    data_folder = dataset_path / "data" / "chunk-000"
    main_videos_folder = (
        dataset_path / "videos" / "chunk-000" / "observation.images.main"
    )
    secondary_videos_folder = (
        dataset_path / "videos" / "chunk-000" / "observation.images.secondary_0"
    )

    # Wait for folders to be created
    folders_to_check = [
        dataset_path,
        meta_folder,
        data_folder,
        main_videos_folder,
        secondary_videos_folder,
    ]
    logger.info(f"Contents of home app path: {list(get_home_app_path().iterdir())}")
    for folder in folders_to_check:
        if wait_for_file(folder, timeout=60) is not True:
            logger.error(f"Failed to create folder: {folder}")
            if folder.parent.exists():
                logger.error(
                    f"[TEST_FAILURE] Contents of parent dir: {list(folder.parent.iterdir())}"
                )
            raise AssertionError(f"Folder not created in time: {folder}")

    # Check that there is only two subfodler in dataset_path / "videos" / "chunk-000"
    assert len(list((dataset_path / "videos" / "chunk-000").iterdir())) == 2

    # Check data file
    data_file = data_folder / "episode_000000.parquet"
    if wait_for_file(data_file) is not True:
        logger.error(
            f"[TEST_FAILURE] Contents of data folder: {list(data_folder.iterdir())}"
        )
        raise AssertionError("Data file not created in time")

    # Check that the positions of the robot is not all zeros
    df = pd.read_parquet(path=data_file)

    # Check that the column "observation.state" is not all lists of zeros
    assert (
        not df["observation.state"].apply(lambda x: all(np.array(x) == 0)).all()
    ), "[TEST_FAILURE] All observation.state arrays are [0,0,0,0,0,0]"

    # Check video file
    main_video_file = main_videos_folder / "episode_000000.mp4"
    if wait_for_file(main_video_file) is not True:
        logger.error(
            f"[TEST_FAILURE] Contents of main videos folder: {list(main_videos_folder.iterdir())}"
        )
        raise AssertionError("Video file not created in time")

    # Check secondary video file
    secondary_video_file = secondary_videos_folder / "episode_000000.mp4"
    if wait_for_file(secondary_video_file) is not True:
        logger.error(
            f"[TEST_FAILURE] Contents of secondary videos folder: {list(secondary_videos_folder.iterdir())}"
        )
        raise AssertionError("Secondary Video file not created in time")

    # Check meta files
    required_meta_files = ["info.json", "episodes.jsonl", "tasks.jsonl"]
    if format == "lerobot_v2.1":
        required_meta_files.append("episodes_stats.jsonl")
    elif format == "lerobot_v2":
        required_meta_files.append("stats.json")
    for filename in required_meta_files:
        meta_file = meta_folder / filename
        if wait_for_file(meta_file) is not True:
            logger.error(
                f"[TEST_FAILURE] Contents of meta folder: {list(meta_folder.iterdir())}"
            )
            raise AssertionError(f"Meta file not created in time: {filename}")

    if format == "lerobot_v2.1":
        check_episodes_stats_file(path=meta_folder / "episodes_stats.jsonl")
    elif format == "lerobot_v2":
        check_stats_file(path=meta_folder / "stats.json")
    check_episodes_file(path=meta_folder / "episodes.jsonl")
    check_info_file(path=meta_folder / "info.json")
    check_tasks_file(path=meta_folder / "tasks.jsonl")
    check_parquet_file(path=data_folder / "episode_000001.parquet")

    # Check the dataset is available on the Hugging Face Hub
    check_if_dataset_is_available_on_hf(
        username_or_org_id, hf_dataset_name, hf_branch_path
    )
    logger.success(
        "[TEST_SUCCESS] Recorded dataset is saved and available on the Hugging Face Hub"
    )


def test_save_lerobot_recording_lerobot_v2():
    """Test saving lerobot recording in lerobot_v2 format."""
    save_lerobot_recording("lerobot_v2")


def test_save_lerobot_recording_lerobot_v2_1():
    """Test saving lerobot recording in lerobot_v2.1 format."""
    save_lerobot_recording("lerobot_v2.1")
