import asyncio
import datetime
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import numpy as np
from loguru import logger
import pandas as pd  # type: ignore
from pydantic import BaseModel, Field, model_validator

from phosphobot.utils import (
    NumpyEncoder,
    compute_sum_squaresum_framecount_from_video,
    create_video_file,
    get_home_app_path,
    decode_numpy,
    NdArrayAsList,
)
from phosphobot.types import VideoCodecs


class BaseRobotPIDGains(BaseModel):
    """
    PID gains for servo motors
    """

    p_gain: float
    i_gain: float
    d_gain: float


class BaseRobotConfig(BaseModel):
    """
    Calibration configuration for a robot
    """

    name: str
    servos_voltage: float
    servos_offsets: List[float]
    servos_calibration_position: List[float]
    servos_offsets_signs: List[float]
    pid_gains: List[BaseRobotPIDGains] = Field(default_factory=list)

    # Torque value to consider that an object is gripped
    gripping_threshold: int = 0
    non_gripping_threshold: int = 0  # noise

    @classmethod
    def from_json(cls, filepath: str) -> Union["BaseRobotConfig", None]:
        """
        Load a configuration from a JSON file
        """
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

        except FileNotFoundError:
            logger.warning(f"Configuration file {filepath} not found.")
            return None

        return cls(**data)

    @classmethod
    def from_v0_json(
        cls, class_name: str, name: str, bundled_calibration_path: Path, serial_id: str
    ) -> Union["BaseRobotConfig", None]:
        """
        Load the config from a v0 JSON file
        """
        filename = f"{class_name}_config.json"
        filepath = str(get_home_app_path() / "calibration" / filename)
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            logger.warning(f"Configuration file {filepath} not found.")
            return None

        logger.info(f"Loaded v0 config for {name} from {filepath}")

        # Load a bundled config file with the same name as the robot
        files = os.listdir(bundled_calibration_path)
        same_name_files = [f for f in files if f.startswith(name)]
        if len(same_name_files) == 0:
            logger.warning(
                f"No bundled calibration file found for {name} in {bundled_calibration_path}"
            )
            return None

        same_name_config = cls.from_json(
            str(bundled_calibration_path / same_name_files[0])
        )
        if same_name_config is None:
            logger.warning(
                f"Failed to load bundled calibration file for {name} in {bundled_calibration_path}"
            )
            return None

        # Copy the data from the v0 JSON file
        same_name_config.name = name
        same_name_config.servos_voltage = data["SERVOS_VOLTAGE"]
        same_name_config.servos_offsets = data["SERVOS_OFFSETS"]
        same_name_config.servos_calibration_position = data[
            "SERVOS_CALIBRATION_POSITION"
        ]
        same_name_config.servos_offsets_signs = data["SERVOS_OFFSETS_SIGNS"]

        # Save the new configuration
        same_name_config.save_local(serial_id=serial_id)

        return same_name_config

    @classmethod
    def from_serial_id(
        cls, serial_id: str, name: str
    ) -> Union["BaseRobotConfig", None]:
        """
        Load a configuration from a serial ID and a name.
        """
        filename = f"{name}_{serial_id}_config.json"
        filepath = str(get_home_app_path() / "calibration" / filename)
        return cls.from_json(filepath)

    def to_json(self, filename: str) -> None:
        """
        Save the configuration to a JSON file
        """
        with open(filename, "w") as f:
            f.write(self.model_dump_json(indent=4))

    def save_local(self, serial_id: str) -> str:
        """
        Save the configuration to the local calibration folder

        Returns:
            The path to the saved file
        """
        filename = f"{self.name}_{serial_id}_config.json"
        filepath = str(get_home_app_path() / "calibration" / filename)
        logger.info(f"Saving configuration to {filepath}")
        self.to_json(filepath)
        return filepath


class BaseRobot(ABC):
    @abstractmethod
    def set_motor_positions(
        self, positions: np.ndarray, enable_gripper: bool = False
    ) -> None:
        """
        Set the motor positions of the robot
        """
        raise NotImplementedError

    @abstractmethod
    def get_info(self) -> "BaseRobotInfo":
        """
        Get information about the robot
        Dict returned is info.json file at initialization
        """
        raise NotImplementedError

    @abstractmethod
    def control_gripper(self, position: float) -> None:
        """
        Control the gripper of the robot
        """
        raise NotImplementedError


class Observation(BaseModel):
    # Main image (reference for OpenVLA actions)
    # TODO PLB: what size?
    # OpenVLA size: 224 × 224px
    main_image: np.ndarray
    # We store any other images from other cameras here
    secondary_images: List[np.ndarray] = Field(default_factory=list)
    # Size 7 array with the robot end effector (absolute, in the robot referencial)
    # Warning: this is not the same 'state' used in lerobot examples
    state: np.ndarray
    # Current joints positions of the robot
    joints_position: np.ndarray
    # Instruction given to the robot, can be null when recording the dataset
    language_instruction: str | None = None
    # Timestamp in seconds since episode start (usefull for frequency)
    timestamp: float | None = None

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
    is_first: bool | None = None
    # True if this is a terminal step, meaning the episode isn' over after this step but the robot is in a terminal state
    is_terminal: bool | None = None
    # if this is the last step of an episode, that contains the last observation. When true,
    is_last: bool | None = None
    reward: float = 0.0  # Reward given by the environment
    # Discount factor for the reward, not used for now
    discount: float = 1.0
    # Any other metadata we want to store, for instance the created_at timestamp in ms
    metadata: dict = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


class Episode(BaseModel):
    """
    # Save an episode
    episode.save("robot_episode.json")

    # Load episode
    loaded_episode = Episode.load("robot_episode.json")

    """

    steps: List[Step] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)

    def save(
        self,
        folder_name: str,
        dataset_name: str,
        fps: int,
        codec: VideoCodecs,
        format_to_save: Literal["json", "lerobot_v2"] = "json",
    ):
        """
        Save the episode to a JSON file with numpy array handling for phospho recording to RLDS format
        Save the episode to a parquet file with an mp4 video for LeRobot recording

        # Episode are saved in a folder with the following structure:

        ---- <folder_name>
        |   ---- json
        |   |   ---- <dataset_name>
        |   |   |   ---- episode_xxxx-xx-xx_xx-xx-xx.json
        |   ---- lerobot_v2
        |   |   ---- <dataset_name>
        |   |   |   ---- data
        |   |   |   |   ---- chunk-000
        |   |   |   |   |   ---- episode_xxxxxx.parquet
        |   |   |   ---- videos
        |   |   |   |   ---- chunk-000
        |   |   |   |   |   ---- observation.images.main.right (if stereo else only main)
        |   |   |   |   |   |   ---- episode_xxxxxx.mp4
        |   |   |   |   |   ---- observation.images.main.left (if stereo)
        |   |   |   |   |   |   ---- episode_xxxxxx.mp4
        |   |   |   |   |   ---- observation.images.secondary_0 (Optional)
        |   |   |   |   |   |   ---- episode_xxxxxx.mp4
        |   |   |   |   |   ---- observation.images.secondary_1 (Optional)
        |   |   |   |   |   |   ---- episode_xxxxxx.mp4
        |   |   |   ---- meta
        |   |   |   |   ---- stats.json
        |   |   |   |   ---- episodes.jsonl
        |   |   |   |   ---- tasks.jsonl
        |   |   |   |   ---- info.json

        """

        # Update the metadata with the format used to save the episode
        self.metadata["format"] = format_to_save
        logger.info(f"Saving episode to {folder_name} with format {format_to_save}")
        dataset_path = os.path.join(folder_name, format_to_save, dataset_name)

        if format_to_save == "lerobot_v2":
            data_path = os.path.join(dataset_path, "data", "chunk-000")
            # Ensure there is a older folder_name/episode_format/dataset_name/data/chunk-000/
            os.makedirs(
                data_path,
                exist_ok=True,
            )

            # Check the elements in the folder folder_name/lerobot_v2-format/dataset_name/data/chunk-000/
            # the episode index is the max index + 1
            # We create the list of index from filenames and take the max + 1
            li_data_filename = os.listdir(data_path)
            episode_index = (
                max(
                    [
                        int(data_filename.split("_")[-1].split(".")[0])
                        for data_filename in li_data_filename
                    ]
                )
                + 1
                if li_data_filename
                else 0
            )

            secondary_camera_frames = self.get_frames_secondary_cameras()

            filename = os.path.join(
                data_path,
                f"episode_{episode_index:06d}.parquet",
            )
            logger.debug(
                f"Saving Episode {episode_index} data in LeRobot format to: {filename}"
            )
            lerobot_episode_parquet: LeRobotEpisodeParquet = (
                self.convert_episode_data_to_LeRobot(
                    fps=fps, episodes_path=data_path, episode_index=episode_index
                )
            )
            # Ensure the directory for the file exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            df = pd.DataFrame(lerobot_episode_parquet.model_dump())

            # Rename observation_state to observation.state
            df.rename(columns={"observation_state": "observation.state"}, inplace=True)
            df.to_parquet(filename, index=False)

            logger.info(f"Data of episode {episode_index} saved to {filename}")

            def create_video_path(folder_name: str, camera_name: str) -> str:
                logger.info(f"Creating video path for {camera_name}")
                return os.path.join(
                    folder_name,
                    "lerobot_v2",
                    dataset_name,
                    "videos/chunk-000",
                    f"observation.images.{camera_name}/episode_{episode_index:06d}.mp4",
                )

            # numpy are shape: height, width, channels
            img_shape = self.steps[0].observation.main_image.shape[:2]
            main_camera_size = (img_shape[1], img_shape[0])

            assert len(main_camera_size) == 2, "Main camera size must be 2D"

            logger.debug(f"Main camera target size: {main_camera_size}")

            # Create the main video file and path
            video_path = create_video_path(folder_name, "main")
            saved_path = create_video_file(
                frames=np.array(self.get_frames_main_camera()),
                target_size=main_camera_size,
                output_path=video_path,
                fps=fps,
                codec=codec,
            )
            # check if the video was saved
            if (isinstance(saved_path, str) and os.path.exists(saved_path)) or (
                isinstance(saved_path, tuple)
                and all(os.path.exists(path) for path in saved_path)
            ):
                logger.info(
                    f"{'Video' if isinstance(saved_path, str) else 'Stereo video'} saved to {video_path}"
                )
            else:
                logger.error(
                    f"{'Video' if isinstance(saved_path, str) else 'Stereo video'} not saved to {video_path}"
                )

            # Create the secondary camera videos and paths
            for index, camera_frames in enumerate(secondary_camera_frames):
                video_path = create_video_path(folder_name, f"secondary_{index}")
                img_shape = camera_frames[0].shape
                logger.debug(f"Secondary cameras are arrays of dimension: {img_shape}")
                secondary_camera_image_size = (img_shape[1], img_shape[0])
                logger.debug(
                    f"Secondary cameras target size: {secondary_camera_image_size}"
                )
                if len(secondary_camera_image_size) != 2:
                    logger.error(
                        f"Secondary camera {index} image must be 2D, skipping video creation"
                    )
                    continue

                os.makedirs(os.path.dirname(video_path), exist_ok=True)
                create_video_file(
                    target_size=secondary_camera_image_size,
                    frames=camera_frames,
                    output_path=video_path,
                    fps=fps,
                    codec=codec,
                )
                if os.path.exists(video_path):
                    logger.info(f"Video for camera {index} saved to {video_path}")
                else:
                    logger.error(f"Video for camera {index} not saved to {video_path}")

        # Case where we save the episode in JSON format
        # Save the episode to a JSON file
        else:
            logger.info("Saving Episode data in JSON format")
            episode_index = (
                max(
                    [
                        int(data_filename.split("_")[-1].split(".")[0])
                        for data_filename in os.listdir(dataset_path)
                    ]
                )
                + 1
                if os.listdir(dataset_path)
                else 0
            )
            # Convert the episode to a dictionary
            data_dict = self.model_dump()
            json_filename = os.path.join(
                folder_name,
                format_to_save,
                dataset_name,
                f"episode_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json",
            )

            # Ensure the directory for the file exists
            os.makedirs(os.path.dirname(json_filename), exist_ok=True)

            # Save to JSON using the custom encoder
            with open(json_filename, "w") as f:
                json.dump(data_dict, f, cls=NumpyEncoder)

    @classmethod
    def load(cls, filename: str) -> "Episode":
        """Load an episode from a JSON file with numpy array handling"""
        path = os.getcwd()
        filename = os.path.join(
            path, "recordings/json-format/example_dataset/", filename
        )

        with open(filename, "r") as f:
            data_dict = json.load(f, object_hook=decode_numpy)

        logger.info(f"Loaded episode from {filename}")

        return cls(**data_dict)

    def add_step(self, step: Step):
        """
        Add a step to the episode
        Handles the is_first, is_terminal and is_last flags to set the correct values when appending a step
        """
        # When a step is aded, it is the last of the episode by default until we add another step
        step.is_terminal = True
        step.is_last = True
        # If current step is the first step of the episode
        if not self.steps:
            step.is_first = True
        else:
            step.is_first = False
            # Change the previous step to not be terminal and last
            self.steps[-1].is_terminal = False
            self.steps[-1].is_last = False
        # Append the step to the episode
        self.steps.append(step)

    def update_previous_step(self, step: Step):
        """
        Update the previous step with the given step.
        """
        # The action is an order in absolute on the postion of the robots joints (radians angles)
        self.steps[-1].action = step.observation.joints_position

    async def play(self, robot: BaseRobot):
        """
        Play the episode on the robot.
        """
        for index, step in enumerate(self.steps):
            # Move the robot to the recorded position
            robot.set_motor_positions(step.observation.joints_position[:6])
            robot.control_gripper(step.observation.joints_position[-1])

            # Wait for the next step
            next_step = self.steps[index + 1] if index + 1 < len(self.steps) else None
            if next_step is not None:
                if (
                    next_step.observation.timestamp is not None
                    and step.observation.timestamp is not None
                ):
                    delta_timestamp = (
                        next_step.observation.timestamp - step.observation.timestamp
                    )
                    await asyncio.sleep(delta_timestamp)

    def get_episode_index(self, episode_recording_folder_path: str, dataset_name: str):
        dataset_path = os.path.join(
            episode_recording_folder_path, f"lerobot_v2-format/{dataset_name}/"
        )

        meta_path = os.path.join(dataset_path, "meta")
        # Ensure meta folder exists
        os.makedirs(meta_path, exist_ok=True)

        # Check the number of elements in the meta_file info.json
        if not os.path.exists(os.path.join(meta_path, "info.json")):
            episode_index = 0

        else:
            with open(os.path.join(meta_path, "info.json"), "r") as f:
                info_json = json.load(f)
                episode_index = info_json["total_episodes"]
        return episode_index

    @property
    def index(self) -> int:
        """
        Return the episode index
        """
        return self.metadata.get("index", 0)

    # Setter
    @index.setter
    def index(self, value: int):
        """
        Set the episode index
        """
        self.metadata["index"] = value

    def convert_episode_data_to_LeRobot(
        self,
        fps: int,
        episodes_path: str,  # We need the episodes path to load the value of the last frame index
        episode_index: int = 0,
    ):
        """
        Convert a dataset to the LeRobot format
        """
        logger.info("Converting dataset to LeRobot format...")

        # Prepare the data for the Parquet file
        episode_data: Dict[str, List] = {
            "action": [],
            "observation.state": [],  # with a dot, not an underscore
            "timestamp": [],
            "task_index": [],
            "episode_index": [],
            "frame_index": [],
            "index": [],
        }

        if self.steps[0].observation.timestamp is None:
            raise ValueError(
                "The first step of the episode must have a timestamp to calculate the other timestamps during reedition of timestamps."
            )

        logger.info(f"Number of steps during conversion: {len(self.steps)}")

        # episode_data["timestamp"] = [step.observation.timestamp for step in self.steps]
        episode_data["timestamp"] = (np.arange(len(self.steps)) / fps).tolist()

        # Fetch the last frame index of the previous episode to continue the indexation of "index" if episode is not the first one
        if episode_index > 0:
            previous_episode_path = os.path.join(
                episodes_path,
                f"episode_{episode_index - 1:06d}.parquet",
            )
            # We load only the last frame index of the previous episode
            previous_episode = pd.read_parquet(
                previous_episode_path, columns=["frame_index"], filters=None
            ).tail(1)
            last_frame_index = 1 + previous_episode["frame_index"].iloc[0]
        else:
            last_frame_index = 0

        for frame_index, step in enumerate(self.steps):
            # Fill in the data for each step
            episode_data["episode_index"].append(episode_index)
            episode_data["frame_index"].append(frame_index)
            episode_data["observation.state"].append(
                step.observation.joints_position.astype(np.float32)
            )
            episode_data["index"].append(frame_index + last_frame_index)
            # TODO: Implement multiple tasks in dataset
            episode_data["task_index"].append(0)
            assert step.action is not None, (
                "The action must be set for each step before saving"
            )
            episode_data["action"].append(step.action.tolist())

        # Validate frame dimensions and data type
        height, width = self.steps[0].observation.main_image.shape[:2]
        if any(
            frame.shape[:2] != (height, width) or frame.ndim != 3
            for frame in self.get_frames_main_camera()
        ):
            raise ValueError(
                "All frames must have the same dimensions and be 3-channel RGB images."
            )

        return LeRobotEpisodeParquet(
            action=episode_data["action"],
            observation_state=episode_data["observation.state"],
            timestamp=episode_data["timestamp"],
            task_index=episode_data["task_index"],
            episode_index=episode_data["episode_index"],
            frame_index=episode_data["frame_index"],
            index=episode_data["index"],
        )

    def get_delta_joints_position(self) -> np.ndarray:
        """
        Return the delta joints position between each step
        """

        deltas_joints_position = np.diff(
            np.array([step.observation.joints_position for step in self.steps]),
            axis=0,
        )

        deltas_joints_position = np.vstack(
            [deltas_joints_position, np.zeros_like(deltas_joints_position[0])]
        )

        return deltas_joints_position

    def get_fps(self) -> float:
        """
        Return the average FPS of the episode
        """
        # Calculate FPS for each episode

        timestamps = np.array([step.observation.timestamp for step in self.steps])
        if (
            len(timestamps) > 1
        ):  # Ensure we have at least 2 timestamps to calculate diff
            fps = 1 / np.mean(np.diff(timestamps))

        return fps

    def get_episode_chunk(self) -> int:
        """
        Return the episode chunk
        """
        raise NotImplementedError

    def get_frames_main_camera(self) -> List[np.ndarray]:
        """
        Return the frames of the main camera
        """
        return [step.observation.main_image for step in self.steps]

    def get_frames_secondary_cameras(self) -> List[np.ndarray]:
        """
        Returns a list, where each np.array is a series of frames for a secondary camera.
        """
        # Handle the case where there are no secondary images
        if not self.steps[0].observation.secondary_images:
            return []

        # Convert the nested structure to a numpy array first
        all_images = [
            np.array([step.observation.secondary_images[i] for step in self.steps])
            for i in range(len(self.steps[0].observation.secondary_images))
        ]

        return all_images


class LeRobotEpisodeParquet(BaseModel):
    """
    Episode class for LeRobot
    """

    action: List[List[float]]
    observation_state: List[List[float]]
    timestamp: List[float]
    task_index: List[int]
    episode_index: List[int]
    frame_index: List[int]
    index: List[int]

    @model_validator(mode="before")
    def validate_lengths(cls, values):
        """
        Ensure all lists have the same length.
        """
        lengths = [
            len(values.get(field, []))
            for field in [
                "action",
                "observation_state",
                "timestamp",
                "task_index",
                "episode_index",
                "frame_index",
                "index",
            ]
        ]
        if len(set(lengths)) != 1:
            raise ValueError(
                "All items in LeRobotEpisodeParquet must have the same length."
            )
        return values


class Dataset(BaseModel):
    """
    Save a dataset

    ```python
    dataset.save("robot_dataset.json")
    ```

    Load dataset

    ```python
    loaded_dataset = Dataset.load("robot_dataset.json")
    ```
    """

    episodes: List[Episode]
    metadata: dict = Field(default_factory=dict)

    def save(self, filename: str):
        """
        Save the dataset to a JSON file with numpy array handling
        TODO: add compression of images arrays
        """
        # Convert the dataset to a dictionary
        data_dict = self.model_dump()

        # Save to JSON using the custom encoder
        with open(filename, "w") as f:
            json.dump(data_dict, f, cls=NumpyEncoder)

    @classmethod
    def load(cls, filename: str) -> "Dataset":
        """Load a dataset from a JSON file with numpy array handling"""
        with open(filename, "r") as f:
            data_dict = json.load(f, object_hook=decode_numpy)
        return cls(**data_dict)

    def get_total_frames(self) -> int:
        """
        Return the total number of frames in the dataset
        """
        return sum(len(episode.steps) for episode in self.episodes)

    def get_robot_types(self) -> str:
        """
        Return in one string the robot types used in the dataset
        """
        unique_robot_types = set(
            episode.metadata.get("robot_type") for episode in self.episodes
        )
        # Concatenate all the robot types in a string
        return ", ".join(
            str(robot) for robot in unique_robot_types if robot is not None
        )

    def get_observation_state_shape(self) -> tuple:
        """
        Check that the dataset has the same observation shape for all states and return it
        """
        unique_shapes = set(
            episode.steps[0].observation.state.shape for episode in self.episodes
        )
        if len(unique_shapes) > 1:
            raise ValueError(
                f"Dataset has multiple observation shapes: {unique_shapes}"
            )
        return unique_shapes.pop()

    def get_images_shape(self) -> tuple:
        """
        Check that all the images have the same shape and return it
        """
        unique_shapes = set(
            step.observation.main_image.shape
            for episode in self.episodes
            for step in episode.steps
        )
        if len(unique_shapes) > 1:
            raise ValueError(f"Dataset has multiple image shapes: {unique_shapes}")
        return unique_shapes.pop()

    def get_joints_position_shape(self) -> tuple:
        """
        Check that the dataset has the same observation.joint_positions shape for all episodes
        Return the shape of the observation main image
        """
        unique_shapes = set(
            step.observation.joints_position.shape
            for episode in self.episodes
            for step in episode.steps
        )

        if len(unique_shapes) > 1:
            raise ValueError(
                f"Dataset has multiple observation.joint_positions shapes: {unique_shapes}"
            )
        return unique_shapes.pop()

    def get_fps(self) -> float:
        """
        Return the average FPS of the dataset
        """
        # Calculate FPS for each episode
        episode_fps = []
        for episode in self.episodes:
            timestamps = np.array(
                [step.observation.timestamp for step in episode.steps]
            )
            if (
                len(timestamps) > 1
            ):  # Ensure we have at least 2 timestamps to calculate diff
                fps = 1 / np.mean(np.diff(timestamps))
                episode_fps.append(fps)

        return np.mean(episode_fps)


class Stats(BaseModel):
    """
    Statistics for a given feature.
    """

    max: NdArrayAsList | None = None
    min: NdArrayAsList | None = None
    mean: NdArrayAsList | None = None
    std: NdArrayAsList | None = None

    # These values are used for rolling computation of mean and std
    sum: NdArrayAsList | None = None
    square_sum: NdArrayAsList | None = None
    count: int = 0

    # To be able to use np.array in pydantic, we need to use arbitrary_types_allowed = True
    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    def update(self, value: np.ndarray | None) -> None:
        """
        Every recording step, update the stats with the new value.
        Note: These are not the final values for mean and std.
        You still neeed to call compute_from_rolling() to get the final values.
        """

        if value is None:
            return None

        # Update the max and min
        if self.max is None:
            self.max = value
        else:
            self.max = np.maximum(self.max, value)

        if self.min is None:
            self.min = value
        else:
            self.min = np.minimum(self.min, value)

        # Update the rolling sum and square sum
        if self.sum is None or self.square_sum is None:
            self.sum = value
            self.square_sum = value**2
            self.count = 1
        else:
            self.sum += value
            self.square_sum += value**2
            self.count += 1

    def compute_from_rolling(self):
        """
        Compute the mean and std from the rolling sum and square sum.
        """
        self.mean = self.sum / self.count
        self.std = np.sqrt(self.square_sum / self.count - self.mean**2)

    def update_image(self, image_value: np.ndarray) -> None:
        """
        Update the stats with the new image.
        The stats are in dim 3 for RGB.
        We normalize with the number of pixels.
        """
        assert image_value.ndim == 3, "Image value must be 3D"

        if image_value is None:
            return None

        image_norm_32 = image_value.astype(dtype=np.float32) / 255.0

        # Update the max and min
        # TODO: Is this code ok with the new shape of image_value?
        # TODO: Check if shape are ok for ex: min or max with shape (3, 1, 3)
        if self.max is None:
            self.max = np.max(image_norm_32, axis=(0, 1))
        else:
            # maximum is the max in each channel
            self.max = np.maximum(self.max, np.max(image_norm_32, axis=(0, 1)))

        if self.min is None:
            self.min = np.min(image_norm_32, axis=(0, 1))
        else:
            self.min = np.minimum(self.min, np.min(image_norm_32, axis=(0, 1)))

        # Update the rolling sum and square sum
        nb_pixels = image_norm_32.shape[0] * image_norm_32.shape[1]
        # Convert to int32 to avoid overflow when computing the square sum
        image_uint32 = image_value.astype(dtype=np.int32)

        if self.sum is None or self.square_sum is None:
            self.sum = np.sum(image_norm_32, axis=(0, 1))
            self.square_sum = np.sum(image_norm_32**2, axis=(0, 1))
            self.count = nb_pixels
        else:
            self.sum = self.sum + np.sum(image_value, axis=(0, 1))
            self.square_sum = self.square_sum + np.sum(image_uint32**2, axis=(0, 1))
            self.count += nb_pixels

    def compute_from_rolling_images(self):
        """
        Compute the mean and std from the rolling sum and square sum for images.
        """
        self.mean = self.sum / self.count
        self.std = np.sqrt(self.square_sum / self.count - self.mean**2)
        # We want .tolist() to yield [[[mean_r, mean_g, mean_b]]] and not [mean_r, mean_g, mean_b]
        # Reshape to have the same shape as the mean and std
        # This makes it easier to normalize the imags
        self.mean = self.mean.reshape(3, 1, 1)
        self.std = self.std.reshape(3, 1, 1)
        if self.min.shape != (3, 1, 3):
            self.min = self.min.reshape(3, 1, 1)
        if self.max.shape != (3, 1, 3):
            self.max = self.max.reshape(3, 1, 1)


class StatsModel(BaseModel):
    """
    Data model utils to create stats.json file.

    Stats for  observation_state and actions are dim the number of joints
    Stats for images are dim 1, 3, 1
    The other stats are dim 1
    """

    observation_state: Stats = Field(default_factory=Stats)  # At init, will do Stats()
    action: Stats = Field(default_factory=Stats)
    timestamp: Stats = Field(default_factory=Stats)
    frame_index: Stats = Field(default_factory=Stats)
    episode_index: Stats = Field(default_factory=Stats)
    index: Stats = Field(default_factory=Stats)

    # TODO: implement multiple language instructions
    task_index: Stats = Field(default_factory=Stats)

    # key is like: observation.images.main
    # value is Stats of the object: average pixel value, std, min, max ; and shape (height, width, channel)
    observation_images: Dict[str, Stats] = Field(default_factory=dict)

    @classmethod
    def from_json(cls, meta_folder_path: str) -> "StatsModel":
        if (
            not os.path.exists(f"{meta_folder_path}/stats.json")
            or os.stat(f"{meta_folder_path}/stats.json").st_size == 0
        ):
            return cls()

        with open(f"{meta_folder_path}/stats.json", "r") as f:
            stats_dict = json.load(f)
            # Rename observation.state to observation_state
            stats_dict["observation_state"] = stats_dict.pop("observation.state")

            # Create a temporary dictionary for observation_images
            observation_images = {}
            for key in list(stats_dict.keys()):
                if "images" in key:
                    observation_images[key] = Stats(**stats_dict.pop(key))

        # Pass observation_images into the model constructor
        return cls(**stats_dict, observation_images=observation_images)

    def to_json(self, meta_folder_path: str) -> None:
        """
        Write the stats.json file in the meta folder path.
        """
        # Write stats files
        with open(f"{meta_folder_path}/stats.json", "w") as f:
            # Write the pydantic Basemodel as a str
            model_dict = self.model_dump()

            # Renamed observation_state to observation.state here
            model_dict["observation.state"] = model_dict.pop("observation_state")

            # We expose the fields in the dict observations images
            for key, value in model_dict["observation_images"].items():
                model_dict[key] = value
            model_dict.pop("observation_images")

            f.write(json.dumps(model_dict, indent=4))

    def update_previous(self, step: Step) -> None:
        """
        Updates the previous action with the given step.
        """
        self.action.update(step.action)

    def update(self, step: Step, episode_index: int) -> None:
        """
        Updates the stats with the given step.
        """
        self.action.update(step.action)

        # We do not update self.action, as it's updated in .update_previous()
        self.observation_state.update(step.observation.joints_position)

        self.timestamp.update(np.array([step.observation.timestamp]))

        self.frame_index.update(np.array([self.frame_index.count + 1]))

        self.episode_index.update(np.array([episode_index]))

        self.index.update(np.array([self.index.count + 1]))

        # TODO: Implement multiple language instructions
        # This should be the index of the instruction as it's in tasks.jsonl (TasksModel)
        self.task_index.update(np.array([0]))

        main_image = step.observation.main_image
        (height, width, channel) = main_image.shape
        aspect_ratio: float = width / height
        if aspect_ratio >= 8 / 3:
            # Stereo image detected: split in half
            left_image = main_image[:, : width // 2, :]
            right_image = main_image[:, width // 2 :, :]
            if (
                "observation.images.main.left" not in self.observation_images.keys()
                or "observation.images.main.right" not in self.observation_images.keys()
            ):
                # Initialize
                self.observation_images["observation.images.main.left"] = Stats()
                self.observation_images["observation.images.main.right"] = Stats()
            self.observation_images["observation.images.main.left"].update_image(
                left_image
            )
            self.observation_images["observation.images.main.right"].update_image(
                right_image
            )

        else:
            if "observation.images.main" not in self.observation_images.keys():
                # Initialize
                self.observation_images["observation.images.main"] = Stats()
            self.observation_images["observation.images.main"].update_image(main_image)

        for image_index, image in enumerate(step.observation.secondary_images):
            if (
                f"observation.images.secondary_{image_index}"
                not in self.observation_images.keys()
            ):
                # Initialize
                self.observation_images[
                    f"observation.images.secondary_{image_index}"
                ] = Stats()

            self.observation_images[
                f"observation.images.secondary_{image_index}"
            ].update_image(step.observation.secondary_images[image_index])

    def save(self, meta_folder_path: str) -> None:
        """
        Save the stats to the meta folder path.
        Also computes the final mean and std for the Stats objects.
        """
        for field_key, field_value in self.__dict__.items():
            # if field is a Stats object, call .compute_from_rolling() to get the final mean and std
            if isinstance(field_value, Stats):
                field_value.compute_from_rolling()

            # Special case for images
            if isinstance(field_value, dict) and field_key == "observation_images":
                for key, value in field_value.items():
                    if isinstance(value, Stats):
                        value.compute_from_rolling_images()

        self.to_json(meta_folder_path)

    def update_before_episode_removal(self, parquet_path: str) -> None:
        """
        Update the stats before removing an episode from the dataset.
        """
        # Check the parquet file exists
        if not os.path.exists(parquet_path):
            raise ValueError(f"Parquet file {parquet_path} does not exist")

        # Load the parquet file
        episode_df = pd.read_parquet(parquet_path)
        nb_steps_deleted_episode = len(episode_df)

        # For each column in the parquet file, compute sum along first axis, min/max along last axis
        logger.info("Updating stats before removing episode")
        logger.info(f"Episode df action: {episode_df['action']}")

        column_sums = {
            col: {
                "sum": np.sum(np.array(episode_df[col].tolist()), axis=0),
                "max": np.max(np.array(episode_df[col].tolist()), axis=0),
                "min": np.min(np.array(episode_df[col].tolist()), axis=0),
                "square_sum": np.sum(np.array(episode_df[col].tolist()) ** 2, axis=0),
            }
            for col in episode_df.columns
        }

        logger.info(f"Column sums: {column_sums}")
        # Update stats for each field in the StatsModel
        for field_name, field in StatsModel.model_fields.items():
            # task_index is not updated since we do not support multiple tasks
            # observation_images has a special treatment
            if field_name in ["task_index", "observation_images"]:
                continue
            logger.info(f"Updating field {field_name}")
            # Get the field value from the instance
            field_value = getattr(self, field_name)
            # Convert observation_state to observation.state
            field_name = (
                "observation.state" if field_name == "observation_state" else field_name
            )
            # Update statistics
            if field_name in column_sums:
                # Maximum of debug info
                logger.info(f"Column sums for {field_name}: {column_sums[field_name]}")
                logger.info(f"Field value: {field_value}")
                logger.info(f"Field value mean: {field_value.mean}")

                # Subtract sums
                field_value.sum -= column_sums[field_name]["sum"]
                field_value.square_sum -= column_sums[field_name]["square_sum"]

                logger.info(f"Field value sum: {field_value.sum}")

                # Update min/max
                if (
                    field_value.min is not None
                    and column_sums[field_name]["min"] is not None
                ):
                    field_value.min = np.minimum(
                        field_value.min, column_sums[field_name]["min"]
                    )
                if (
                    field_value.max is not None
                    and column_sums[field_name]["max"] is not None
                ):
                    field_value.max = np.maximum(
                        field_value.max, column_sums[field_name]["max"]
                    )

                # Update count
                field_value.count -= nb_steps_deleted_episode

                # Recalculate mean and standard deviation
                if field_value.count > 0:
                    field_value.mean = field_value.sum / field_value.count
                    field_value.std = np.sqrt(
                        (field_value.square_sum / field_value.count)
                        - np.square(field_value.mean)
                    )

        # For image we need to load the mp4 video
        logger.info("Updating stats for images")
        # Extract the episode index from the parquet file name
        episode_index = int(parquet_path.split("_")[-1].split(".")[0])

        # List cameras_folder:
        folder_videos_path = (
            "/".join(parquet_path.split("/")[:-3]) + "/videos/chunk-000"
        )

        cameras_folder = os.listdir(folder_videos_path)
        for camera_folder in cameras_folder:
            # Create the path of the video episode_{episode_index:06d}.mp4
            video_path = os.path.join(
                folder_videos_path, camera_folder, f"episode_{episode_index:06d}.mp4"
            )
            sum_array, square_sum_array, frame_count_array = (
                compute_sum_squaresum_framecount_from_video(video_path)
            )

            sum_array = sum_array.astype(np.float32)
            square_sum_array = square_sum_array.astype(np.float32)

            logger.info(f"sum_array: {sum_array}")
            logger.info(f"square_sum_array: {square_sum_array}")
            logger.info(f"frame_count_array: {frame_count_array}")

            # Update the stats_model
            self.observation_images[camera_folder].sum = (
                self.observation_images[camera_folder].sum - sum_array
            )
            self.observation_images[camera_folder].square_sum = (
                self.observation_images[camera_folder].square_sum - square_sum_array
            )
            self.observation_images[camera_folder].count = (
                self.observation_images[camera_folder].count - frame_count_array[0]
            )
            field_value.count -= nb_steps_deleted_episode
            field_value.mean = field_value.sum / field_value.count
            logger.info(f"mean: {field_value.mean}")
            logger.info(f"count: {field_value.count}")
            logger.info(f"square_sum: {field_value.square_sum}")

            field_value.std = np.sqrt(
                abs((field_value.square_sum / field_value.count) - field_value.mean**2)
            )
            logger.info(f"std: {field_value.std}")


class FeatureDetails(BaseModel):
    dtype: Literal["video", "int64", "float32", "str", "bool"]
    shape: List[int]
    names: List[str] | None


class VideoInfo(BaseModel):
    """
    Information about the video
    """

    video_fps: int = 10
    video_codec: VideoCodecs

    video_pix_fmt: str = "yuv420p"
    video_is_depth_map: bool = False
    has_audio: bool = False


class VideoFeatureDetails(FeatureDetails):
    dtype: Literal["video"] = "video"
    info: VideoInfo


class InfoFeatures(BaseModel):
    action: FeatureDetails
    observation_state: FeatureDetails

    timestamp: FeatureDetails = Field(
        default_factory=lambda: FeatureDetails(dtype="float32", shape=[1], names=None)
    )
    episode_index: FeatureDetails = Field(
        default_factory=lambda: FeatureDetails(dtype="int64", shape=[1], names=None)
    )
    frame_index: FeatureDetails = Field(
        default_factory=lambda: FeatureDetails(dtype="int64", shape=[1], names=None)
    )
    task_index: FeatureDetails = Field(
        default_factory=lambda: FeatureDetails(dtype="int64", shape=[1], names=None)
    )
    index: FeatureDetails = Field(
        default_factory=lambda: FeatureDetails(dtype="int64", shape=[1], names=None)
    )
    # Camera images
    observation_images: Dict[str, VideoFeatureDetails] = Field(default_factory=dict)

    # Optional fields (RL)
    next_done: FeatureDetails | None = None
    next_success: FeatureDetails | None = None
    next_reward: FeatureDetails | None = None

    def to_dict(self) -> dict:
        """
        Convert the InfoFeatures to a dictionary.
        This transforms observation_images and observation_state to the correct format.
        """
        model_dict = self.model_dump()

        # Rename key observation_state to observation.state same with images
        model_dict["observation.state"] = model_dict.pop("observation_state")

        if self.observation_images is not None:
            for key, value in self.observation_images.items():
                model_dict[key] = value.model_dump()

        model_dict.pop("observation_images")

        # Filter all None values
        model_dict = {
            key: value
            for key, value in model_dict.items()
            if value is not None and value != {}
        }

        return model_dict


class BaseRobotInfo(BaseModel):
    robot_type: str
    action: FeatureDetails
    observation_state: FeatureDetails


class InfoModel(BaseModel):
    """
    Data model util to create meta/info.jsonl file.
    """

    robot_type: str

    codebase_version: str = "v2.0"
    total_episodes: int = 0
    total_frames: int = 0
    total_tasks: int = 0
    total_videos: int = 0
    total_chunks: int = 1
    chunks_size: int = 1000
    fps: int = 10
    splits: Dict[str, str] = Field(default_factory=lambda: {"train": "0:0"})
    data_path: str = (
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    )
    video_path: str = (
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    )
    features: InfoFeatures

    @classmethod
    def from_robot(cls, robot: BaseRobot, **data) -> "InfoModel":
        """
        From a robot configuration, create the appropriate InfoModel.
        This is because it depends on the number of joints etc.
        """
        robot_info = robot.get_info()
        features = InfoFeatures(
            action=robot_info.action,
            observation_state=robot_info.observation_state,
        )
        return cls(
            **data,
            features=features,
            robot_type=robot_info.robot_type,
        )

    def to_dict(self):
        """
        Convert the InfoModel to a dictionary. This is different from
        model_dump() as it transforms the features to the correct format.
        """
        model_dict = self.model_dump()
        model_dict["features"] = self.features.to_dict()
        return model_dict

    @classmethod
    def from_json(
        cls,
        meta_folder_path: str,
        fps: int | None = None,
        codec: VideoCodecs | None = None,
        robot: BaseRobot | None = None,
        main_image: np.ndarray | None = None,
        secondary_images: List[np.ndarray] | None = None,
        main_is_stereo: bool = False,
    ) -> "InfoModel":
        """
        Read the info.json file in the meta folder path.
        If the file does not exist, try to create the InfoModel from the provided robot.

        raise ValueError if no robot is provided and the file does not exist.
        """
        if (
            not os.path.exists(f"{meta_folder_path}/info.json")
            or os.stat(f"{meta_folder_path}/info.json").st_size == 0
        ):
            if robot is None:
                raise ValueError(
                    "No info.json file found and no robot provided to create the InfoModel"
                )
            if codec is None:
                raise ValueError("No codec provided to create the InfoModel")
            if fps is None:
                raise ValueError("No fps provided to create the InfoModel")

            info_model = cls.from_robot(robot)
            if main_image is not None:
                if not main_is_stereo:
                    info_model.features.observation_images[
                        "observation.images.main"
                    ] = VideoFeatureDetails(
                        shape=list(main_image.shape),
                        names=["height", "width", "channel"],
                        info=VideoInfo(video_codec=codec, video_fps=fps),
                    )
                else:
                    # Split along the width in 2 imaes
                    new_shape = [
                        main_image.shape[0],
                        main_image.shape[1] // 2,
                        main_image.shape[2],
                    ]
                    info_model.features.observation_images[
                        "observation.images.main.left"
                    ] = VideoFeatureDetails(
                        shape=new_shape,
                        names=["height", "width", "channel"],
                        info=VideoInfo(video_codec=codec, video_fps=fps),
                    )
                    info_model.features.observation_images[
                        "observation.images.main.right"
                    ] = VideoFeatureDetails(
                        shape=new_shape,
                        names=["height", "width", "channel"],
                        info=VideoInfo(video_codec=codec, video_fps=fps),
                    )

            if secondary_images is not None:
                for index_image, image in enumerate(secondary_images):
                    key_name = f"observation.images.secondary_{index_image}"
                    info_model.features.observation_images[key_name] = (
                        VideoFeatureDetails(
                            shape=list(image.shape),
                            names=["height", "width", "channel"],
                            info=VideoInfo(video_codec=codec, video_fps=fps),
                        )
                    )
            return info_model

        with open(f"{meta_folder_path}/info.json", "r") as f:
            stats_dict = json.load(f)
            stats_dict["features"]["observation_state"] = stats_dict["features"].pop(
                "observation.state"
            )

            observation_images = {}
            keys_to_remove = []
            for key, value in stats_dict["features"].items():
                if "observation.images" in key:
                    observation_images[key] = value
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del stats_dict["features"][key]
            stats_dict["features"]["observation_images"] = observation_images

        return cls(**stats_dict)

    def to_json(self, meta_folder_path: str) -> None:
        """
        Write the info.json file in the meta folder path.
        """
        with open(f"{meta_folder_path}/info.json", "w") as f:
            f.write(json.dumps(self.to_dict(), indent=4))

    def update(self, episode: Episode) -> None:
        """
        Update the info given a new recorded Episode.
        """
        self.total_episodes += 1
        self.total_frames += len(episode.steps)
        self.total_videos += len(self.features.observation_images.keys())
        self.splits = {"train": f"0:{self.total_episodes}"}
        # TODO: Handle multiple language instructions
        # TODO: Implement support for multiple chunks

    def save(
        self,
        meta_folder_path: str,
    ) -> None:
        """
        Save the info to the meta folder path.
        """
        self.to_json(meta_folder_path)

    def update_before_episode_removal(self, parquet_path: str) -> None:
        """
        Update the info before removing an episode from the dataset.
        """
        # Ensure the parquet file exist
        if not os.path.exists(parquet_path):
            raise ValueError(f"Parquet file {parquet_path} does not exist.")

        # Load parquet file with pandas
        df_episode = pd.read_parquet(parquet_path)

        self.total_episodes -= 1
        self.total_frames -= len(df_episode)
        self.total_videos -= len(self.features.observation_images.keys())
        self.splits = {"train": f"0:{self.total_episodes}"}
        # TODO(adle): Implement support for multi tasks in dataset
        # self.total_tasks -= ...


class TasksFeatures(BaseModel):
    """
    Features of the lines in tasks.jsonl.
    """

    task_index: int
    task: str = "None"


class TasksModel(BaseModel):
    """
    Data model util to create meta/tasks.jsonl file.
    """

    tasks: List[TasksFeatures]
    _initial_nb_total_tasks: int = 0

    @classmethod
    def from_jsonl(cls, meta_folder_path: str) -> "TasksModel":
        """
        Read the tasks.jsonl file in the meta folder path.
        """
        if not os.path.exists(f"{meta_folder_path}/tasks.jsonl"):
            return cls(tasks=[])

        with open(f"{meta_folder_path}/tasks.jsonl", "r") as f:
            tasks = []
            for line in f:
                tasks.append(TasksFeatures(**json.loads(line)))

        tasks_model = cls(tasks=tasks)
        # Do it after model init, otherwise pydantic ignores the value of _original_nb_total_tasks
        tasks_model._initial_nb_total_tasks = len(tasks)
        return tasks_model

    def to_jsonl(self, meta_folder_path: str) -> None:
        """
        Write the tasks.jsonl file in the meta folder path.
        """

        with open(f"{meta_folder_path}/tasks.jsonl", "a") as f:
            # Only append the new tasks to the file
            for task in self.tasks[self._initial_nb_total_tasks :]:
                f.write(task.model_dump_json() + "\n")

    def update(self, step: Step) -> None:
        """
        Updates the tasks with the given step.
        """
        # Add the task only if it is not in the tasks already
        if str(step.observation.language_instruction) not in [
            task.task for task in self.tasks
        ]:
            logger.info(
                f"Adding task: {step.observation.language_instruction} to {[task.task for task in self.tasks]}"
            )
            self.tasks.append(
                TasksFeatures(
                    task_index=len(self.tasks),
                    task=str(step.observation.language_instruction),
                )
            )

    def update_before_episode_removal(self, parquet_path: str) -> None:
        """
        Update the tasks before removing an episode from the dataset.
        We count the number of occurences of task_index in the dataset.
        If the episode is the only one with this task_index, we remove it from the tasks.jsonl file.
        """
        # Ensure the parquet file exist
        if not os.path.exists(parquet_path):
            raise ValueError(f"Parquet file {parquet_path} does not exist.")

        # Load parquet file with pandas
        df_episode = pd.read_parquet(parquet_path)

        # Create the path of the data folder
        parquet_path_parts = parquet_path.split("/")
        data_folder_path = "/".join(parquet_path_parts[:-2])

        # For each file in the data folder get the task indexes (unique)
        task_indexes: List[int] = []
        for file in os.listdir(data_folder_path):
            if file.endswith(".parquet"):
                df = pd.read_parquet(f"{data_folder_path}/{file}")
                task_indexes.extend(df["task_index"].unique())

        for task_index in df_episode["task_index"].unique():
            task_index = int(task_index)
            # delete the line in tasks.jsonl if and only if the task index in this episode is the only one in the dataset
            if task_indexes.count(task_index) == 1:
                self.tasks = [
                    task for task in self.tasks if task.task_index != task_index
                ]

    def save(self, meta_folder_path: str) -> None:
        """
        Save the episodes to the meta folder path.
        """
        self.to_jsonl(meta_folder_path)


class EpisodesFeatures(BaseModel):
    """
    Features of the lines in episodes.jsonl.
    """

    episode_index: int = 0
    tasks: List[str] = []
    length: int = 0


class EpisodesModel(BaseModel):
    """
    Follow the structure of the episodes.jsonl file.
    """

    episodes: List[EpisodesFeatures] = Field(default_factory=list)
    _original_nb_total_episodes: int = 0

    def update(self, step: Step, episode_index: int) -> None:
        """
        Updates the episodes with the given episode.
        episode_index is the index of the episode of the current step.
        """
        # If episode_index is not in the episodes, add it
        if episode_index not in [episode.episode_index for episode in self.episodes]:
            self.episodes.append(
                EpisodesFeatures(
                    episode_index=episode_index,
                    tasks=[str(step.observation.language_instruction)],
                    length=1,
                )
            )
        else:
            # Increase the nb frames counter
            self.episodes[episode_index].length += 1
            # Add the language instruction if it's a new one
            if (
                str(step.observation.language_instruction)
                not in self.episodes[episode_index].tasks
            ):
                self.episodes[episode_index].tasks.append(
                    str(step.observation.language_instruction)
                )

    def to_jsonl(self, meta_folder_path: str) -> None:
        """
        Write the episodes.jsonl file in the meta folder path.
        """
        with open(f"{meta_folder_path}/episodes.jsonl", "a") as f:
            for episode in self.episodes[self._original_nb_total_episodes :]:
                f.write(episode.model_dump_json() + "\n")

    @classmethod
    def from_jsonl(cls, meta_folder_path: str) -> "EpisodesModel":
        """
        Read the episodes.jsonl file in the meta folder path.
        """
        if not os.path.exists(f"{meta_folder_path}/episodes.jsonl"):
            return EpisodesModel()

        with open(f"{meta_folder_path}/episodes.jsonl", "r") as f:
            episodes_model = []
            for line in f:
                logger.debug(f"Reading line: {line}")
                episodes_model.append(EpisodesFeatures(**json.loads(line)))

        episode = EpisodesModel(episodes=episodes_model)
        # Do it after model init, otherwise pydantic ignores the value of _original_nb_total_episodes
        episode._original_nb_total_episodes = len(episodes_model)
        return episode

    def save(self, meta_folder_path: str) -> None:
        """
        Save the episodes to the meta folder path.
        """
        self.to_jsonl(meta_folder_path)

    def update_before_episode_removal(self, parquet_path: str):
        """
        Update the episodes model before removing an episode from the dataset.
        We just remove the line corresponding to the episode_index of the parquet file.
        """
        # Ensure the parquet file exist
        if not os.path.exists(parquet_path):
            raise ValueError(f"Parquet file {parquet_path} does not exist.")

        index_deleted_episode = int(
            parquet_path.split("/")[-1].split(".")[0].split("_")[-1]
        )

        self.episodes = [
            episode
            for episode in self.episodes
            if episode.episode_index != index_deleted_episode
        ]
