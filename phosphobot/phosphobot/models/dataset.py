import asyncio
import datetime
import json
import os
import shutil
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Literal, Optional, cast

import numpy as np
from huggingface_hub import (
    HfApi,
    create_branch,
    create_repo,
    delete_folder,
    delete_repo,
)
from loguru import logger
from pydantic import BaseModel, Field

from phosphobot.models.robot import BaseRobot
from phosphobot.utils import (
    NumpyEncoder,
    decode_numpy,
    get_hf_token,
    get_hf_username_or_orgid,
    parse_hf_username_or_orgid,
)

DEFAULT_FILE_ENCODING = "utf-8"


class Observation(BaseModel):
    # Main image (reference for OpenVLA actions)
    # OpenVLA size: 224 × 224px
    main_image: np.ndarray = Field(default_factory=lambda: np.array([]))
    # We store any other images from other cameras here
    secondary_images: List[np.ndarray] = Field(default_factory=list)
    # Size 7 array with the robot end effector (absolute, in the robot referencial)
    # Warning: this is not the same 'state' used in lerobot examples
    state: np.ndarray = Field(default_factory=lambda: np.array([]))
    # Current joints positions of the robot
    joints_position: np.ndarray
    # Instruction given to the robot, can be null when recording the dataset
    language_instruction: Optional[str] = None
    # Timestamp in seconds since episode start (usefull for frequency)
    timestamp: Optional[float] = None

    # To be able to use np.array in pydantic, we need to use arbitrary_types_allowed = True
    class Config:
        arbitrary_types_allowed = True

    def dict(self, *args, **kwargs):
        # Override dict method to handle numpy arrays
        d = super().dict(*args, **kwargs)
        return d


class Step(BaseModel):
    observation: Observation  # Current observation, most informations are stored here
    # Robot action as outputed by OpenVLA (size 7 array) based on the CURRENT observation
    action: Optional[np.ndarray] = None
    # if this is the first step of an episode that contains the initial state.
    is_first: Optional[bool] = None
    # True if this is a terminal step, meaning the episode isn' over after this step but the robot is in a terminal state
    is_terminal: Optional[bool] = None
    # if this is the last step of an episode, that contains the last observation. When true,
    is_last: Optional[bool] = None
    reward: float = 0.0  # Reward given by the environment
    # Discount factor for the reward, not used for now
    discount: float = 1.0
    # Any other metadata we want to store, for instance the created_at timestamp in ms
    metadata: dict = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True
        extra = "ignore"


class BaseEpisode(BaseModel, ABC):
    steps: List[Step] = Field(default_factory=list)
    # metadata stores: episode_index, created_at, robot_type, episode_format, dataset_name, instruction (optional)
    # For JsonEpisode, it might also store base_recording_folder, created_at_str (for filename)
    # For LeRobotEpisode, dataset_manager (the LeRobotDataset instance) is a direct attribute, not in metadata.
    metadata: dict = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"  # Allow extra fields like dataset_manager for LeRobotEpisode not in schema

    def add_step(self, step: Step):
        """Common logic for adding a step to the internal list and managing flags."""
        # is_terminal and is_last are true by default for a new step
        step.is_terminal = True
        step.is_last = True

        # Handle NaN joint positions by copying from the previous step
        if len(self.steps) > 0 and np.all(np.isnan(step.observation.joints_position)):
            logger.warning(
                f"Step {len(self.steps)} has NaN joint_positions. Copying from previous step."
            )
            step.observation.joints_position = self.steps[
                -1
            ].observation.joints_position.copy()

        if not self.steps:  # This is the first step
            step.is_first = True
        else:  # Not the first step
            step.is_first = False
            # The previously last step is no longer terminal or last
            self.steps[-1].is_terminal = False
            self.steps[-1].is_last = False

        self.steps.append(step)

    def update_previous_step(self, current_step_data: Step):
        """
        Updates the 'action' of the previous step.
        The action that led to `current_step_data.observation` was `current_step_data.observation.joints_position`.
        So, `self.steps[-1].action` becomes `current_step_data.observation.joints_position`.
        """
        if len(self.steps) > 0:
            # The action of the previous step is the joints_position of the current observation
            self.steps[-1].action = current_step_data.observation.joints_position.copy()

    @property
    @abstractmethod
    def dataset_path(self) -> Path:
        """Path to the dataset folder where this episode is stored."""
        raise NotImplementedError(
            "Subclasses must implement dataset_path to return the correct path."
        )

    @property
    def episode_index(self) -> int:
        idx = self.metadata.get("episode_index")
        if idx is None:
            # For JSON episodes, this might not be strictly required if identified by timestamp
            logger.warning("episode_index not explicitly set in metadata.")
            return -1  # Or raise error depending on strictness for all episode types
        return int(idx)

    @episode_index.setter
    def episode_index(self, value: int):
        self.metadata["episode_index"] = value

    @property
    def instruction(self) -> Optional[str]:
        return self.metadata.get("instruction")

    @classmethod
    @abstractmethod
    async def start_new(cls, **kwargs) -> "BaseEpisode":  # type: ignore[misc]
        """Factory method to create and initialize a new episode."""
        pass

    @abstractmethod
    async def append_step(self, step: Step, **kwargs) -> None:
        """Appends a step and handles related business logic (e.g., updating live meta files)."""
        pass

    @abstractmethod
    async def save(self, **kwargs) -> None:
        """Saves the episode data and any related metadata or artifacts."""
        pass

    # Common helper methods from old Episode class
    def get_episode_frames_main_camera(self) -> List[np.ndarray]:
        return [
            step.observation.main_image
            for step in self.steps
            if step.observation.main_image is not None
            and step.observation.main_image.size > 0
        ]

    def get_episode_frames_secondary_cameras(
        self,
    ) -> List[List[np.ndarray]]:  # List of frame lists
        if not self.steps or not self.steps[0].observation.secondary_images:
            return []

        num_secondary_cameras = len(self.steps[0].observation.secondary_images)
        all_secondary_frames: List[List[np.ndarray]] = [
            [] for _ in range(num_secondary_cameras)
        ]

        for step in self.steps:
            for i, sec_img in enumerate(step.observation.secondary_images):
                if sec_img is not None and sec_img.size > 0:
                    all_secondary_frames[i].append(sec_img)
        return all_secondary_frames

    @classmethod
    def load(
        cls,
        episode_data_path: str,
        format: Literal["json", "lerobot_v2", "lerobot_v2.1"],
    ) -> "BaseEpisode":
        """Load an episode data file. There is numpy array handling for json format.""
        If we load the parquet file we don't have informations about the images
        """
        from phosphobot.models.lerobot_dataset import LeRobotEpisode

        episode_data_extension = episode_data_path.split(".")[-1]

        if episode_data_extension == "json":
            return JsonEpisode.from_json(episode_data_path)
        elif episode_data_extension == "parquet":
            return LeRobotEpisode.from_parquet(
                episode_data_path,
                format=cast(Literal["lerobot_v2", "lerobot_v2.1"], format),
            )
        else:
            raise ValueError(
                f"Unsupported episode data format: {episode_data_extension}"
            )

    async def play(
        self,
        robots: List[BaseRobot],
        playback_speed: float = 1.0,
        interpolation_factor: int = 4,
        replicate: bool = False,
    ):
        """
        Play the episode on the robot with on-the-fly interpolation.
        """

        def move_robots(joints: np.ndarray) -> None:
            """
            Solve which robot should move depending on the number of joints and
            the number of robots.
            - If nb robots == nb joints, move each robot with its respective joints
            - If nb joints > nb robots, move each robot with its respective joints until
                the last robot. Extra joints are ignored.
            - If nb joints < nb robots, move each robot with its respective joints until
                the last joint. Extra robots are ignored
            """

            nonlocal robots

            nb_joints = 1 + len(joints) % 6  # 6 joints per robot
            for i, robot in enumerate(robots):  # extra joints are ignored
                # If there are more robots than joints, ignore the extra robots
                if i >= nb_joints:
                    if replicate is False:
                        break
                    else:
                        # Go back to the first robot
                        i = i % nb_joints

                # Get the joints for the current robot
                robot_joints = joints[i * 6 : (i + 1) * 6]
                # Move the robot with its respective joints
                robot.set_motors_positions(robot_joints, enable_gripper=True)

        for index, step in enumerate(self.steps):
            # Get current and next step
            curr_step = step
            next_step = self.steps[index + 1] if index + 1 < len(self.steps) else None

            if (
                next_step is not None
                and curr_step.observation.timestamp is not None
                and next_step.observation.timestamp is not None
                and curr_step.observation.joints_position is not None
                and next_step.observation.joints_position is not None
            ):
                # if the current step is all NAN, skip
                if np.isnan(curr_step.observation.joints_position).all():
                    logger.warning(
                        f"Skipping step {index} because all joints positions are NaN"
                    )
                    continue
                # Calculate base delta timestamp
                delta_timestamp = (
                    next_step.observation.timestamp - curr_step.observation.timestamp
                )
                # Higher playback speed = less time per segment
                # Higher interpolation factor = less time per segment + more segments
                time_per_segment = (
                    delta_timestamp / interpolation_factor / playback_speed
                )

                # Fill empty values from the next step joints with the current step
                next_step.observation.joints_position = np.where(
                    np.isnan(next_step.observation.joints_position),
                    curr_step.observation.joints_position,
                    next_step.observation.joints_position,
                )

                # Perform interpolation steps
                for i in range(interpolation_factor):
                    start_time = time.perf_counter()

                    # Calculate interpolation ratio (0 to 1 across all segments)
                    t = i / interpolation_factor

                    # Interpolate between the current and next step
                    interp_value = t * (next_step.observation.joints_position) + (
                        1 - t
                    ) * (curr_step.observation.joints_position)

                    if index % 20 == 0 and i == 0:
                        logger.info(f"Playing step {index}")

                    move_robots(interp_value)
                    # Timing control
                    elapsed = time.perf_counter() - start_time
                    time_to_wait = max(time_per_segment - elapsed, 0)
                    await asyncio.sleep(time_to_wait)

            else:
                # Handle last step or cases where timestamp is None
                start_time = time.perf_counter()
                move_robots(curr_step.observation.joints_position)


class JsonEpisode(BaseEpisode):
    @property
    def _base_recording_folder(self) -> Path:
        path_str = self.metadata.get("base_recording_folder")
        if not path_str:
            raise ValueError("base_recording_folder not in metadata for JsonEpisode")
        return Path(path_str)

    @property
    def _dataset_name(self) -> str:
        name = self.metadata.get("dataset_name")
        if not name:
            raise ValueError("dataset_name not in metadata for JsonEpisode")
        return name

    @property
    def dataset_path(self) -> Path:  # Full path to this specific JSON dataset
        return self._base_recording_folder / "json" / self._dataset_name

    @property
    def _json_episode_path(self) -> Path:
        filename = f"episode_{self.metadata['created_at_str']}.json"
        path = self.dataset_path / filename
        os.makedirs(path.parent, exist_ok=True)
        return path

    @classmethod
    async def start_new(  # type: ignore[override]
        cls,
        base_recording_folder: str,  # e.g., ".../phosphobot/recordings"
        dataset_name: str,
        robots: List[BaseRobot],
        instruction: Optional[str] = None,
        freq: Optional[int] = None,  # Optional for JSON, might be useful for metadata
        **kwargs,
    ) -> "JsonEpisode":
        start_timestamp = time.time()
        created_at_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        metadata = {
            "base_recording_folder": base_recording_folder,
            "dataset_name": dataset_name,
            "episode_format": "json",
            "created_at": start_timestamp,
            "created_at_str": created_at_string,  # For filename
            "robot_type": ", ".join(r.name for r in robots),
            "instruction": instruction,
            "freq": freq,
            # JSON episodes usually don't have a strict sequential index like LeRobot.
            # If one is needed, logic to determine it (e.g., counting files) would go here.
            "episode_index": int(
                start_timestamp
            ),  # Using timestamp as a pseudo-index for now
        }
        logger.info(f"Starting new JSON episode for dataset '{dataset_name}'.")
        return cls(steps=[], metadata=metadata)

    async def append_step(self, step: Step, **kwargs) -> None:
        self.add_step(step)  # Uses BaseEpisode.add_step

    async def save(self, **kwargs) -> None:
        if not self.steps:
            logger.warning("JSON Episode has no steps. Skipping save.")
            return

        data_to_dump = self.model_dump()  # Get all data including steps and metadata

        # Ensure the directory exists
        os.makedirs(self._json_episode_path.parent, exist_ok=True)

        with open(self._json_episode_path, "w", encoding=DEFAULT_FILE_ENCODING) as f:
            json.dump(data_to_dump, f, cls=NumpyEncoder, indent=2)
        logger.success(f"JSON Episode saved to {self._json_episode_path}")

    @classmethod
    def from_json(cls, episode_data_path: str) -> "JsonEpisode":
        """Load an episode data file. There is numpy array handling for json format."""
        # Check that the file exists
        if not os.path.exists(episode_data_path):
            raise FileNotFoundError(f"Episode file {episode_data_path} not found.")

        with open(episode_data_path, "r", encoding=DEFAULT_FILE_ENCODING) as f:
            data_dict = json.load(f, object_hook=decode_numpy)
            logger.debug(f"Data dict keys: {data_dict.keys()}")

        return cls(**data_dict)

    def delete(self):
        os.remove(self.json_path)


class BaseDataset:
    """
    Handle common dataset operations. Useful to manage the dataset.
    """

    episodes: List[BaseEpisode]
    metadata: dict = Field(default_factory=dict)
    path: str
    dataset_name: str
    # Full path to the dataset folder
    folder_full_path: Path

    def __init__(self, path: str, enforce_path: bool = False) -> None:
        """
        Load an existing dataset.
        """

        path_obj = Path(path)
        self.folder_full_path = path_obj
        # Check path format
        path_parts = path_obj.parts
        if enforce_path:
            if len(path_parts) < 2 or path_parts[-2] not in [
                "json",
                "lerobot_v2",
                "lerobot_v2.1",
            ]:
                raise ValueError(
                    f"Wrong dataset path provided. Expected format: "
                    "`/path/to/dataset_format/dataset_name` where dataset_format is one of "
                    "`json`, `lerobot_v2`, `lerobot_v2.1`."
                    f" Got: {path}"
                )

        self.path = str(path_obj)
        self.episodes = []
        self.dataset_name = path_parts[-1]

        # Create the dataset folder if it does not exist
        os.makedirs(self.folder_full_path, exist_ok=True)

        self.HF_API = HfApi(token=get_hf_token())

        # Validate dataset name
        if not BaseDataset.check_dataset_name(self.dataset_name):
            raise ValueError(
                "Dataset name contains invalid characters. Should not contain spaces or /"
            )

    @classmethod
    def check_dataset_name(cls, name: str) -> bool:
        """Validate dataset name format"""
        return " " not in name and "/" not in name

    @classmethod
    def consolidate_dataset_name(cls, name: str) -> str:
        """
        Check if the dataset name is valid.
        To be valid, the dataset name must be a string without spaces, /, or -.

        If not we replace them with underscores.
        """
        if not cls.check_dataset_name(name):
            logger.warning(
                "Dataset name contains invalid characters. Replacing them with underscores."
            )
            name.replace(" ", "_").replace("/", "_").replace("-", "_")

        return name

    @classmethod
    def remove_ds_store_files(cls, folder_path: str):
        try:
            # Iterate through all items in the folder
            for item in os.listdir(folder_path):
                item_path = os.path.join(folder_path, item)

                # If item is a .DS_Store file, remove it
                if item == ".DS_Store":
                    os.remove(item_path)
                    continue

                # If item is a directory, recurse into it
                if os.path.isdir(item_path):
                    cls.remove_ds_store_files(item_path)

        except (PermissionError, OSError):
            pass

    @property
    def repo_id(self) -> str:
        """
        Return the huggingface repository id
        """
        repo_id = f"{get_hf_username_or_orgid()}/{self.dataset_name}"
        return repo_id

    def check_repo_exists(self, repo_id: Optional[str]) -> bool:
        """Check if a repository exists on Hugging Face"""
        repo_id = repo_id or self.repo_id
        return self.HF_API.repo_exists(repo_id=repo_id, repo_type="dataset")

    def sync_local_to_hub(self):
        """Reupload the dataset folder to Hugging Face"""
        username_or_orgid = get_hf_username_or_orgid()
        if username_or_orgid is None:
            logger.warning(
                "No Hugging Face token found. Please add a token in the Admin page.",
            )
            return

        repository_exists = self.HF_API.repo_exists(
            repo_id=self.repo_id, repo_type="dataset"
        )

        # If the repository does not exist, push the dataset to Hugging Face
        if not repository_exists:
            self.push_dataset_to_hub()

        # else, Delete the folders and reupload the dataset.
        else:
            # Delete the dataset folders from Hugging Face
            try:
                delete_folder(
                    repo_id=self.repo_id, path_in_repo="./data", repo_type="dataset"
                )
            except Exception:
                logger.debug("No data folder to delete")
            try:
                delete_folder(
                    repo_id=self.repo_id, path_in_repo="./videos", repo_type="dataset"
                )
            except Exception:
                logger.debug("No videos folder to delete")
            try:
                delete_folder(
                    repo_id=self.repo_id, path_in_repo="./meta", repo_type="dataset"
                )
            except Exception:
                logger.debug("No meta folder to delete")
            # Reupload the dataset folder to Hugging Face
            self.HF_API.upload_folder(
                folder_path=self.folder_full_path,
                repo_id=self.repo_id,
                repo_type="dataset",
                run_as_future=True,
            )

    def delete(self) -> None:
        """Delete the dataset from the local folder and Hugging Face"""
        # Delete locally
        if not os.path.exists(self.folder_full_path):
            logger.error(f"Dataset not found in {self.folder_full_path}")
            return

        # Remove the data file if confirmation is correct
        if os.path.isdir(self.folder_full_path):
            shutil.rmtree(self.folder_full_path)
            logger.success(f"Dataset deleted: {self.folder_full_path}")
        else:
            logger.error(f"The Dataset is a file: {self.folder_full_path}")

        # Remove the dataset from Hugging Face
        if self.check_repo_exists(self.repo_id):
            delete_repo(repo_id=self.repo_id, repo_type="dataset")

    def generate_read_me_string(self, dataset_name: str) -> str:
        """
        Generate a README string for the dataset.
        This is used to create the README file in the dataset folder.
        """
        return f"""
---
tags:
- phosphobot
- so100
- phospho-dk
task_categories:
- robotics                                                   
---

# {dataset_name}

**This dataset was generated using [phosphobot](https://docs.phospho.ai).**

This dataset contains a series of episodes recorded with a robot and multiple cameras. \
It can be directly used to train a policy using imitation learning. \
It's compatible with LeRobot.

To get started in robotics, [get your own phospho starter pack.](https://robots.phospho.ai).
"""

    def push_dataset_to_hub(self, branch_path: Optional[str] = None):
        """
        Push the dataset to the Hugging Face Hub.

        Args:
            branch_path (str, optional): Additional branch to push to besides main
        """
        try:
            # Initialize HF API with token

            # Try to get username/org ID from token
            username_or_org_id = None
            try:
                # Get user info from token
                user_info = self.HF_API.whoami()
                username_or_org_id = parse_hf_username_or_orgid(user_info)

                if not username_or_org_id:
                    logger.error("Could not get username or org ID from token")
                    return

            except Exception as e:
                logger.warning(
                    "No user or org with write access found. Won't be able to push to Hugging Face."
                )
                return

            # Create README if it doesn't exist
            readme_path = os.path.join(self.folder_full_path, "README.md")
            if not os.path.exists(readme_path):
                with open(
                    readme_path, "w", encoding=DEFAULT_FILE_ENCODING
                ) as readme_file:
                    readme_file.write(self.generate_read_me_string(self.dataset_name))

            # Construct full repo name
            dataset_repo_name = f"{username_or_org_id}/{self.dataset_name}"
            create_2_1_branch = False

            # Check if repo exists, create if it doesn't
            try:
                self.HF_API.repo_info(repo_id=dataset_repo_name, repo_type="dataset")
                logger.info(f"Repository {dataset_repo_name} already exists.")
            except Exception:
                from phosphobot.configs import config

                logger.info(
                    f"Repository {dataset_repo_name} does not exist. Creating it..."
                )
                create_repo(
                    repo_id=dataset_repo_name,
                    repo_type="dataset",
                    exist_ok=True,
                    token=True,
                    private=config.DEFAULT_HF_PRIVATE_MODE,
                )
                logger.info(f"Repository {dataset_repo_name} created.")
                create_2_1_branch = True

            # Push to main branch
            logger.info(
                f"Pushing the dataset to the main branch in repository {dataset_repo_name}"
            )
            self.HF_API.upload_folder(
                folder_path=self.folder_full_path,
                repo_id=dataset_repo_name,
                repo_type="dataset",
                run_as_future=True,
            )

            repo_refs = self.HF_API.list_repo_refs(
                repo_id=dataset_repo_name, repo_type="dataset"
            )
            existing_branch_names = [ref.name for ref in repo_refs.branches]

            # Create and push to v2.1 branch if needed
            if create_2_1_branch:
                try:
                    if "v2.1" not in existing_branch_names:
                        logger.info(
                            f"Creating branch v2.1 for dataset {dataset_repo_name}"
                        )
                        create_branch(
                            dataset_repo_name,
                            repo_type="dataset",
                            branch="v2.1",
                            token=True,
                        )
                        logger.info(
                            f"Branch v2.1 created for dataset {dataset_repo_name}"
                        )

                    # Push to v2.1 branch
                    logger.info(
                        f"Pushing the dataset to the branch v2.1 in repository {dataset_repo_name}"
                    )
                    self.HF_API.upload_folder(
                        folder_path=self.folder_full_path,
                        repo_id=dataset_repo_name,
                        repo_type="dataset",
                        revision="v2.1",
                        run_as_future=True,
                    )
                except Exception as e:
                    logger.error(f"Error handling v2.1 branch: {e}")

            # Push to additional branch if specified
            if branch_path:
                try:
                    if branch_path not in existing_branch_names:
                        logger.info(
                            f"Creating branch {branch_path} for dataset {dataset_repo_name}"
                        )
                        create_branch(
                            dataset_repo_name,
                            repo_type="dataset",
                            branch=branch_path,
                            token=True,
                        )
                        logger.info(
                            f"Branch {branch_path} created for dataset {dataset_repo_name}"
                        )

                    # Push to specified branch
                    logger.info(f"Pushing the dataset to branch {branch_path}")
                    self.HF_API.upload_folder(
                        folder_path=self.folder_full_path,
                        repo_id=dataset_repo_name,
                        repo_type="dataset",
                        revision=branch_path,
                        run_as_future=True,
                    )
                    logger.info(f"Dataset pushed to branch {branch_path}")
                except Exception as e:
                    logger.error(f"Error handling custom branch: {e}")

        except Exception as e:
            logger.warning(f"An error occurred: {e}")
