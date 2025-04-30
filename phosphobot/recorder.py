import asyncio
import os
import time
from typing import Literal, Optional

import numpy as np
from fastapi import BackgroundTasks
from loguru import logger

from phosphobot.camera import AllCameras
from phosphobot.configs import config
from phosphobot.models import (
    BaseRobot,
    Dataset,
    Episode,
    EpisodesModel,
    InfoModel,
    Observation,
    StatsModel,
    Step,
    TasksModel,
)
from phosphobot.types import VideoCodecs
from phosphobot.utils import background_task_log_exceptions, get_home_app_path


class Recorder:
    dataset_name: str = "example_dataset"
    episode_format: Literal["json", "lerobot_v2"] = "json"
    filename: str

    is_recording: bool = False
    episode: Episode | None
    start_ts: float | None

    cameras: AllCameras
    cameras_ids_to_record: list[int] | None = None

    # class of meta files
    stats_model: StatsModel | None = None
    info_model: InfoModel | None = None
    episodes_model: EpisodesModel | None = None
    tasks_model: TasksModel | None = None

    @property
    def episode_recording_folder(self) -> str:
        return str(get_home_app_path() / "recordings")

    def __init__(self, robots: list[BaseRobot], cameras: AllCameras):
        """
        Usage:
        ```
        recorder.start() # Start recording every 1/freq seconds
        recorder.stop() # Stop recording. This will not save the data to disk
        ```

        Args:
            robot (BaseRobot): The robot instance to record
            cameras (AllCameras): All Cameras used for recording
        """
        self.robots = robots
        self.cameras = cameras

    @property
    def dataset_folder_path(self) -> str:
        if self.episode_format in ["json", "lerobot_v2"]:
            return os.path.join(
                self.episode_recording_folder, self.episode_format, self.dataset_name
            )
        else:
            raise ValueError(
                f"Unknown episode format: {self.episode_format}. Please select either 'json' or 'lerobot_v2'"
            )

    @property
    def meta_folder_path(self) -> str:
        """
        Get the path of the meta folder where the meta files are stored.
        """
        return os.path.join(self.dataset_folder_path, "meta")

    @property
    def data_folder_path(self) -> str:
        """
        Get the path of the meta folder where the meta files are stored.
        """
        return (
            str(os.path.join(self.dataset_folder_path, "data", "chunk-000"))
            if self.episode_format == "lerobot_v2"
            else self.dataset_folder_path
        )

    @property
    def videos_folder_path(self) -> str:
        """
        Get the path of the meta folder where the meta files are stored.
        """
        if self.episode_format != "lerobot_v2":
            raise ValueError(
                f"Tried to access videos folder for {self.episode_format}. Only lerobot_v2 format is supported"
            )
        return os.path.join(self.dataset_folder_path, "videos", "chunk-000")

    async def start(
        self,
        background_tasks: BackgroundTasks,
        robots: list[BaseRobot],
        codec: VideoCodecs,
        freq: int,
        branch_path: str | None,
        # All videos should have the same resolution for the datasets
        target_size: tuple[int, int] | None,
        dataset_name: str,
        instruction: str | None,
        episode_format: Literal["json", "lerobot_v2"],
        cameras_ids_to_record: list[int] | None,
        use_push_to_hf: bool = True,
    ) -> None:
        """
        Args:
            robots (list[BaseRobot]): The robots instances to record
            codec (VideoCodecs): The codec to use for the video recording
            freq (int): The frequency of recording in Hz
            branch_path (str | None): The branch path to push the dataset to
            target_size (tuple[int, int] | None): The target size of the video recorded. All videos should have the same resolution for the datasets.
                If None, the DEFAULT_VIDEO_SIZE from the config is used.
            dataset_name (str): The name of the dataset
            episode_format (Literal["json", "lerobot_v2"]): The format of the episode
            cameras_ids_to_record (list[int] | None): The list of camera ids to record
            use_push_to_hf (bool): If True, push the dataset to the Hugging Face Hub using the token saved in huggingface.token
        """

        if target_size is None:
            target_size = (config.DEFAULT_VIDEO_SIZE[0], config.DEFAULT_VIDEO_SIZE[1])

        if self.is_recording:
            logger.warning("Stopping previous recording")
            await self.stop()

        if self.episode_format not in ["json", "lerobot_v2"]:
            raise ValueError(f"Unknown episode format: {self.episode_format}")

        self.codec = codec
        # Camera ids to record can be None the case is handled in the setter
        self.cameras.cameras_ids_to_record = cameras_ids_to_record  # type: ignore
        logger.debug(f"(START) Cameras ids asked: {cameras_ids_to_record}")

        self.episode_format = episode_format
        self.dataset_name = dataset_name
        self.freq = freq

        self.use_push_to_hub = use_push_to_hf
        self.branch_path = branch_path

        self.is_recording = True
        self.start_ts = time.perf_counter()

        self.robots = robots

        # Count the number of files in meta_folder_path/data/chunk-000 to get episode_index
        os.makedirs(self.data_folder_path, exist_ok=True)
        episode_index = len(os.listdir(self.data_folder_path))

        # Read the number of episodes in the data folder
        self.episode = Episode(
            steps=[],
            metadata={
                "episode_index": episode_index,
                "created_at": self.start_ts,
                "robot_type": ", ".join(robot.name for robot in robots),
                "format": self.episode_format,
                "dataset_name": self.dataset_name,
            },
        )

        # Ensure the meta, data and videos folders exist for lerobot_v2 format
        if self.episode_format == "lerobot_v2":
            # Make sure the dataset folder and meta exists
            os.makedirs(self.meta_folder_path, exist_ok=True)
            os.makedirs(self.videos_folder_path, exist_ok=True)

            # Load the meta files from the disk
            self.stats_model = StatsModel.from_json(
                meta_folder_path=self.meta_folder_path
            )

            # Load the info model
            # Note: The types are ignored because we don't make those interfaces
            # available in phosphobot

            # If the we have a stereo_camera, we have to add the right camera in the se
            self.info_model = InfoModel.from_json(
                meta_folder_path=self.meta_folder_path,
                robots=robots,  # type: ignore
                codec=codec,
                target_size=target_size,
                secondary_camera_key_names=self.cameras.get_secondary_camera_key_names(),
                fps=self.freq,
            )

            # Increase the episode_index based on the info model
            self.episode.index = self.info_model.total_episodes

            self.episodes_model = EpisodesModel.from_jsonl(
                meta_folder_path=self.meta_folder_path,
            )
            self.tasks_model = TasksModel.from_jsonl(
                meta_folder_path=self.meta_folder_path
            )

            self.global_index = self.info_model.total_frames

        background_tasks.add_task(
            background_task_log_exceptions(self.record),
            target_size=target_size,
            language_instruction=instruction or config.DEFAULT_TASK_INSTRUCTION,
        )

    async def stop(self) -> None:
        """
        Stop the recording without saving
        Update the info_model instance with the new episode
        """
        logger.info("End of recording")
        self.is_recording = False
        return None

    def save_episode(self) -> Optional[str]:
        """
        Save the current episode to disk asynchronously

        Save the episode to a JSON file with numpy array handling for phospho recording to RLDS format
        Save the episode to a parquet file with an mp4 video for LeRobot recording

        Episode are saved in a folder with the following structure:

        ---- folder_name
        |   ---- json
        |   |   ---- dataset_name
        |   |   |   ---- episode_xxxx-xx-xx_xx-xx-xx.json
        |   ---- lerobot_v2
        |   |   ---- dataset_name
        |   |   |   ---- data
        |   |   |   |   ---- chunk-000
        |   |   |   |   |   ---- episode_xxxxxx.parquet
        |   |   |   ---- videos
        |   |   |   |   ---- chunk-000
        |   |   |   |   |   ---- observation.images.main.right (if stereo else only main)
        |   |   |   |   |   |   ---- episode_xxxxxx.mp4
        |   |   |   |   |   ---- observation.images.main.left (if stereo)
        |  |   |   |   |   |   ---- episode_xxxxxx.mp4
        |   |   |   |   |   ---- observation.images.secondary_0 (Optional)
        |   |   |   |   |   |   ---- episode_xxxxxx.mp4
        |   |   |   |   |   ---- observation.images.secondary_1 (Optional)
        |   |   |   |   |   |   ---- episode_xxxxxx.mp4
        |   |   |   ---- meta
        |   |   |   |   ---- stats.json
        |   |   |   |   ---- episodes.jsonl
        |   |   |   |   ---- tasks.jsonl
        |   |   |   |   ---- info.json

        Reminder:
        self.push_to_hub: If True, push the dataset to the Hugging Face Hub using the token saved in huggingface.token
        self.branch_path: If provided, push the dataset to the specified branch additionally to the main branch
        """

        # If not steps, we don't save the episode
        if self.episode and len(self.episode.steps) == 0:
            logger.warning("No steps in the episode. Not saving the episode.")
            return None

        if self.episode_format not in ["json", "lerobot_v2"]:
            raise ValueError(f"Unknown episode format: {self.episode_format}")

        if not hasattr(self, "episode") or self.episode is None:
            logger.error("No episode to save")
            return None

        if self.episode_format == "lerobot_v2" and not hasattr(self, "global_index"):
            logger.error("No global index to save")
            return None

        episode_to_save = self.episode

        episode_to_save.save(
            folder_name=self.episode_recording_folder,
            format_to_save=self.episode_format,
            dataset_name=self.dataset_name,
            fps=self.freq,
            info_model=self.info_model,
            last_frame_index=None
            if not hasattr(self, "global_index")
            else self.global_index,
        )

        # Update the meta files if recording in lerobot_v2 format
        if self.episode_format == "lerobot_v2":
            # Start by incrementing the recorder global index
            self.global_index += len(self.episode.steps)

            if self.stats_model is not None:
                self.stats_model.save(meta_folder_path=self.meta_folder_path)
            else:
                logger.error("Stats model is not initialized. Call start() first")

            if self.tasks_model is not None:
                self.tasks_model.save(meta_folder_path=self.meta_folder_path)
            else:
                logger.error("Tasks model is not initialized. Call start() first")

            if self.episodes_model is not None:
                self.episodes_model.save(meta_folder_path=self.meta_folder_path)
            else:
                logger.error("Episodes model is not initialized. Call start() first")

            if (
                self.info_model is not None
                and self.tasks_model is not None
                and self.episodes_model is not None
            ):
                self.info_model.update(episode=self.episode)
                self.info_model.save(meta_folder_path=self.meta_folder_path)
            else:
                logger.error(
                    "Info model, tasks model or episodes model is not initialized. Call start() first"
                )

        logger.success(
            f"Episode saved to {self.episode_recording_folder}/{self.episode_format}"
        )

        if self.use_push_to_hub:
            self.push_to_hub(branch_path=self.branch_path)
        return None

    def push_to_hub(self, branch_path: str | None = None) -> None:
        """
        This method upload the recorded dataset to the Hugging Face Hub.
        You can specify a branch path to upload the dataset to a specific branch too.
        We will always push to main branch so it will contain the last version of the dataset.
        In CI/CD we the branch path correspond to the working branch and commit id.
        """
        dataset = Dataset(path=self.dataset_folder_path)
        dataset.push_dataset_to_hub(branch_path=branch_path)

    async def record(
        self,
        target_size: tuple[int, int],
        language_instruction: str,
    ) -> None:
        """
        This function will be used in a background task to record the data of an Episode.
        When the function stops, the recording stops and we can save the data to the disk at filepath location.
        """

        if not self.episode:
            raise ValueError("episode is not initialized. Call start() first")
        if not self.start_ts:
            raise ValueError("start_ts is not initialized. Call start() first")

        logger.info(
            f"Recording: Started with {self.cameras.camera_ids=} ({self.cameras.main_camera=})"
        )
        # Init variables to store the data of the episode
        step_count = 0
        while self.is_recording:
            current_ts = time.perf_counter()

            main_frames = self.cameras.get_main_camera_frames(
                target_video_size=target_size
            )
            if main_frames is not None and len(main_frames) > 0:
                main_frame = main_frames[0]
                # The secondary cameras are all the available cameras except the main camera
                # If the main camera is a stereo camera, we only take the left frame as the main frame
                secondary_frames = []
                if len(main_frames) == 2:
                    secondary_frames.append(main_frames[1])

                secondary_frames.extend(
                    self.cameras.get_secondary_camera_frames(
                        target_video_size=target_size
                    )
                )
            else:
                # If no frames are available, we create empty frames
                main_frame = np.zeros(
                    (target_size[1], target_size[0], 3), dtype=np.uint8
                )
                secondary_frames = []

            # If available, get the depth frame. This requires the realsense camera to be available.
            depth_frame = self.cameras.get_depth_frame()

            # Get robots observations
            state, joints_position = self.robots[0].get_observation()
            for robot in self.robots[1:]:
                # Append the state and joints position of the other robots
                state_other, joints_position_other = robot.get_observation()
                state = np.append(state, state_other)
                joints_position = np.append(joints_position, joints_position_other)

            observation = Observation(
                main_image=main_frame,
                secondary_images=secondary_frames,
                state=state,
                language_instruction=language_instruction,
                joints_position=joints_position,
                # Timestamp in milliseconds since episode start (usefull for frequency)
                timestamp=current_ts - self.start_ts,
            )
            step = Step(
                observation=observation,
                action=joints_position,
                metadata={"created_at": current_ts, "depth_frame": depth_frame},
            )
            # Log the step every 20 steps
            if step_count % 20 == 0:
                logger.debug(f"Recording: Adding Step {step_count}")

            # The order of the following steps is important

            # First, we update the previous step with the current action
            # This helps simulate the fact that we want to move the robot to the next observation
            self.episode.update_previous_step(step)

            # Second, we add the step with the observations
            self.episode.add_step(step)
            if self.episode_format == "lerobot_v2":
                assert (
                    self.stats_model is not None
                ), "Stats model is not initialized. Call start() first"
                assert (
                    self.episodes_model is not None
                ), "Episodes model is not initialized. Call start() first"
                assert (
                    self.tasks_model is not None
                ), "Tasks model is not initialized. Call start() first"
                # Update the models

                # We pass the step count to update frame index properly
                self.stats_model.update(
                    step=step,
                    episode_index=self.episode.index,
                    current_step_index=step_count,
                )
                self.episodes_model.update(step=step, episode_index=self.episode.index)
                self.tasks_model.update(step=step)

            elapsed = current_ts - time.perf_counter()
            time_to_wait = max(1 / self.freq - elapsed, 0)
            await asyncio.sleep(time_to_wait)
            step_count += 1
