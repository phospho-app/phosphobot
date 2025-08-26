import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, cast
import tempfile

import numpy as np
import pandas as pd
from huggingface_hub import delete_file, upload_folder
from loguru import logger
from pydantic import (
    AliasChoices,
    BaseModel,
    Field,
    field_validator,
    model_validator,
)

from phosphobot.types import VideoCodecs
from phosphobot.utils import (
    NdArrayAsList,
    compute_sum_squaresum_framecount_from_video,
    create_video_file,
    get_field_min_max,
    get_home_app_path,
)
from phosphobot.models.robot import BaseRobot
from phosphobot.models.dataset import BaseDataset, BaseEpisode, Step

DEFAULT_FILE_ENCODING = "utf-8"


class LeRobotDataset(BaseDataset):
    format_version: Literal["lerobot_v2", "lerobot_v2.1"] = "lerobot_v2.1"

    def __init__(
        self,
        path: str,
        enforce_path: bool = True,
    ):
        # path is like "recordings/lerobot_v2.1/my_dataset_name"
        super().__init__(
            path, enforce_path=enforce_path
        )  # This sets self.folder_full_path etc.

        # Determine format version from path
        if len(Path(path).parts) < 2:
            # assume version is lerobot_v2.1
            self.format_version = "lerobot_v2.1"
        else:
            # Read the format version from the path
            self.format_version = cast(
                Literal["lerobot_v2", "lerobot_v2.1"], Path(path).parts[-2]
            )

        # Ensure base LeRobot directories exist within the dataset path
        os.makedirs(
            self.data_folder_full_path, exist_ok=True
        )  # .../dataset_name/data/chunk-000
        os.makedirs(self.meta_folder_full_path, exist_ok=True)  # .../dataset_name/meta
        os.makedirs(
            self.videos_folder_full_path, exist_ok=True
        )  # .../dataset_name/videos/chunk-000

        # Meta models are loaded/initialized when needed, typically by initialize_meta_models_if_needed
        self.info_model: Optional[InfoModel] = None
        self.episodes_stats_model: Optional[EpisodesStatsModel] = None
        self.stats_model: Optional[StatsModel] = None  # For lerobot_v2
        self.episodes_model: Optional[EpisodesModel] = None
        self.tasks_model: Optional[TasksModel] = None
        logger.info(
            f"LeRobotDataset manager initialized for path: {self.folder_full_path}"
        )

    def load_meta_models(
        self,
        robots: Optional[List[BaseRobot]] = None,
        codec: Optional[VideoCodecs] = None,
        target_size: Optional[Tuple[int, int]] = None,
        fps: Optional[int] = None,
        all_camera_key_names: Optional[List[str]] = None,
        force: bool = False,
    ):
        """Loads existing meta files or initializes new ones if they don't exist."""
        logger.debug(
            f"Initializing/loading meta models for dataset: {self.dataset_name}"
        )
        if self.info_model is None or force:
            self.info_model = InfoModel.from_json(
                meta_folder_path=self.meta_folder_full_path,  # Correct path to 'meta' dir
                robots=robots,  # Passed for initialization if file doesn't exist
                codec=codec,
                target_size=target_size,
                all_camera_key_names=all_camera_key_names,
                fps=fps,
                format=self.format_version,
            )
            # Edit the .format_version field to match the dataset format
            if "2.1" in self.info_model.codebase_version:
                self.format_version = "lerobot_v2.1"
            else:
                # Assuming it's lerobot_v2
                self.format_version = "lerobot_v2"

        if self.format_version == "lerobot_v2.1":
            if self.episodes_stats_model is None or force:
                self.episodes_stats_model = EpisodesStatsModel.from_jsonl(
                    meta_folder_path=self.meta_folder_full_path
                )
        elif self.format_version == "lerobot_v2":
            if self.stats_model is None or force:  # Only for v2
                self.stats_model = StatsModel.from_json(
                    meta_folder_path=self.meta_folder_full_path
                )

        if self.episodes_model is None or force:
            self.episodes_model = EpisodesModel.from_jsonl(
                meta_folder_path=self.meta_folder_full_path, format=self.format_version
            )
        if self.tasks_model is None or force:
            self.tasks_model = TasksModel.from_jsonl(
                meta_folder_path=self.meta_folder_full_path
            )
        # Consistency checks and fix
        if self.info_model.total_frames != sum(
            [e.length for e in self.episodes_model.episodes]
        ):
            logger.warning(
                "Total frames in info.json doesn't match sum of episodes.jsonl lengths. "
                + "Recomputing total frames from parquet files."
            )
            self.episodes_model.recompute_from_parquets(
                dataset_path=Path(self.data_folder_full_path),
                format=self.format_version,
            )
            self.info_model.recompute_from_parquets(
                infos=self.info_model,
                dataset_path=Path(self.folder_full_path),
            )
            self.episodes_model.save(
                meta_folder_path=self.meta_folder_full_path,
            )
            self.info_model.save(meta_folder_path=self.meta_folder_full_path)

        logger.debug("Meta models initialization/loading complete.")

    def load_episodes(self):
        """Loads all episodes from the dataset."""
        if self.episodes_model is None:
            # Load it
            self.load_meta_models()
        # if still none, raise an error
        if self.episodes_model is None:
            raise ValueError(
                "EpisodesModel not initialized in LeRobotDataset. Call initialize_meta_models_if_needed first."
            )

        episodes = []
        for episodes_features in self.episodes_model.episodes:
            episode = LeRobotEpisode.from_parquet(
                self.get_episode_data_path(episodes_features.episode_index),
                format=self.format_version,
                dataset_path=self.data_folder_full_path,
            )
            episode.dataset_manager = self
            episodes.append(episode)
        self.episodes = episodes

    def get_next_episode_index(self) -> int:
        if self.info_model is None:
            raise ValueError(
                "InfoModel not initialized in LeRobotDataset. Call initialize_meta_models_if_needed first."
            )
        return self.info_model.total_episodes  # total_episodes is 0-indexed for next

    def get_current_total_frames(self) -> int:
        if self.info_model is None:
            raise ValueError("InfoModel not initialized in LeRobotDataset.")
        return self.info_model.total_frames

    def save_all_meta_models(self):
        """Saves all currently loaded meta models to disk."""
        logger.debug(f"Saving all meta models for dataset: {self.dataset_name}")
        if self.info_model:
            self.info_model.save(self.meta_folder_full_path)
        if self.episodes_stats_model:
            self.episodes_stats_model.save(self.meta_folder_full_path)
        if self.stats_model:
            self.stats_model.save(self.meta_folder_full_path)  # v2 only
        if self.episodes_model:
            self.episodes_model.save(self.meta_folder_full_path, save_mode="overwrite")
        if self.tasks_model:
            self.tasks_model.save(self.meta_folder_full_path)
        logger.debug("All meta models saved.")

    def delete_episode(self, episode_id: int, update_hub: bool = True) -> None:
        """
        Delete the episode data from the dataset.
        If format is lerobot, also delete the episode videos from the dataset
        and updates the meta data.
        JSON format not supported

        If update_hub is True, also delete the episode data from the Hugging Face repository
        """

        episode_data_path = self.get_episode_data_path(episode_id)
        episode_to_delete = LeRobotEpisode.from_parquet(
            episode_data_path=episode_data_path,
            format=self.format_version,
            dataset_path=self.data_folder_full_path,
        )
        episode_to_delete.dataset_manager = self

        if self.format_version == "lerobot_v2":
            raise NotImplementedError(
                "Episode deletion is not implemented for LeRobot v2 format. Please use v2.1 format."
            )

        if update_hub is True and self.check_repo_exists(self.repo_id) is False:
            logger.warning(
                f"Repository {self.repo_id} does not exist on Hugging Face. Skipping deletion on Hugging Face"
            )
            update_hub = False

        logger.info(
            f"Deleting episode {episode_to_delete.episode_index} {episode_data_path} from dataset {self.dataset_name} with episode format {self.format_version}"
        )
        logger.debug(f"update_hub: {update_hub}")

        # Start loading current meta data
        info_model = InfoModel.from_json(meta_folder_path=self.meta_folder_full_path)
        tasks_model = TasksModel.from_jsonl(meta_folder_path=self.meta_folder_full_path)
        episodes_model = EpisodesModel.from_jsonl(
            meta_folder_path=self.meta_folder_full_path,
            format=cast(Literal["lerobot_v2", "lerobot_v2.1"], self.format_version),
        )
        if info_model.codebase_version == "v2.1":
            episodes_stats_model = EpisodesStatsModel.from_jsonl(
                meta_folder_path=self.meta_folder_full_path
            )
        elif info_model.codebase_version == "v2.0":
            stats_model = StatsModel.from_json(
                meta_folder_path=self.meta_folder_full_path
            )
        else:
            raise NotImplementedError(
                f"Codebase version {info_model.codebase_version} not supported, should be v2.1 or v2.0"
            )

        # Update meta data before episode removal
        try:
            episode_parquet = episode_to_delete.parquet()
        except FileNotFoundError as e:
            logger.warning(
                f"Episode {episode_id} parquet {episode_to_delete._parquet_path} not found: {e}"
            )
            episode_parquet = pd.DataFrame()
        tasks_model.update_for_episode_removal(
            df_episode_to_delete=episode_parquet,
            data_folder_full_path=self.data_folder_full_path,
        )
        tasks_model.save(meta_folder_path=self.meta_folder_full_path)
        logger.info("Tasks model updated")

        info_model.update_for_episode_removal(
            df_episode_to_delete=episode_parquet,
        )
        info_model.save(meta_folder_path=self.meta_folder_full_path)
        logger.info("Info model updated")

        # Delete the actual episode files (parquet and mp4 video)
        episode_to_delete.delete(update_hub=update_hub, repo_id=self.repo_id)

        # Rename the remaining episodes to keep the numbering consistent
        # be sure to reindex AFTER deleting the episode data
        old_index_to_new_index = self.reindex_episodes(
            nb_steps_deleted_episode=len(episode_to_delete.steps),
            folder_path=self.data_folder_full_path,
        )
        # Reindex the episode videos
        for camera_folder_full_path in self.get_camera_folders_full_paths():
            self.reindex_episodes(
                folder_path=camera_folder_full_path,
                old_index_to_new_index=old_index_to_new_index,  # type: ignore
            )

        episodes_model.update_for_episode_removal(
            episode_to_delete_index=episode_id,
            old_index_to_new_index=old_index_to_new_index,
        )
        episodes_model.save(
            meta_folder_path=self.meta_folder_full_path, save_mode="overwrite"
        )
        logger.info("Episodes model updated")

        if info_model.codebase_version == "v2.1":
            episodes_stats_model.update_for_episode_removal(
                episode_to_delete_index=episode_id,
                old_index_to_new_index=old_index_to_new_index,
            )
            episodes_stats_model.save(meta_folder_path=self.meta_folder_full_path)
            logger.info("Episodes stats model updated")
        elif info_model.codebase_version == "v2.0":
            # Update for episode removal is not implemented
            stats_model.update_for_episode_removal(
                data_folder_path=self.data_folder_full_path,
            )
            stats_model.save(meta_folder_path=self.meta_folder_full_path)
            logger.info("Stats model updated")

        if update_hub:
            upload_folder(
                folder_path=self.meta_folder_full_path,
                repo_id=self.repo_id,
                repo_type="dataset",
                path_in_repo="meta",
            )

    def merge_datasets(
        self,
        second_dataset: "LeRobotDataset",
        new_dataset_name: str,
        video_transform: dict[str, str],
        check_format: bool = True,
    ) -> None:
        """
        Merge multiple datasets into one.
        This will create a new dataset with the merged data.

        Merge `self` with `second_dataset` and create a new dataset.

        Dataset Structure

        Args:
            second_dataset (Dataset): Dataset to merge with `self`.
            new_dataset_name (str): Name of the folder where the merged dataset
                will be created.
            video_transform (dict[str, str]): Mapping of camera folder names
                from this dataset to the corresponding folders in
                ``second_dataset``. It ensures videos are copied to the correct
                location.

        The resulting dataset will follow this structure:

        / videos
            ├── chunk-000
            │   ├── observation.images.main
            │   |   ├── episode_000000.mp4
            │   ├── observation.images.secondary_0
            │   |   ├── episode_000000.mp4
        / data
            ├── chunk-000
            │   ├── episode_000000.parquet
        / meta
            ├── info.json
            ├── tasks.jsonl
            ├── episodes.jsonl
            ├── episodes_stats.jsonl
        / README.md
        """
        # Check that all datasets have the same format
        if check_format:
            if second_dataset.format_version != self.format_version:
                raise ValueError(
                    f"Dataset {second_dataset.dataset_name} has a different format: {second_dataset.format_version}"
                )

        path_result_dataset = os.path.join(
            os.path.dirname(self.folder_full_path),
            new_dataset_name,
        )
        # If the dataset already exists, raise an error
        if os.path.exists(path_result_dataset):
            raise ValueError(
                f"Dataset {new_dataset_name} already exists in {path_result_dataset}"
            )
        os.makedirs(path_result_dataset, exist_ok=True)

        path_to_videos = os.path.join(
            path_result_dataset,
            "videos",
            "chunk-000",
        )
        os.makedirs(path_to_videos, exist_ok=True)

        ### VIDEOS
        logger.debug("Moving videos to the new dataset")

        # Start by moving videos to the new dataset
        for video_key in video_transform.keys():
            video_folder = video_key
            src_folder = os.path.join(self.videos_folder_full_path, video_folder)
            if os.path.exists(src_folder):
                # Move the video folder to the new dataset
                shutil.copytree(
                    src_folder,
                    os.path.join(path_to_videos, video_folder),
                )

                video_files = [f for f in os.listdir(src_folder) if f.endswith(".mp4")]
                nb_videos = len(video_files)

                # Move the videos from the second dataset to the new dataset and increment the index
                second_video_folder = video_transform[video_key]
                video_folder_full_path = os.path.join(
                    second_dataset.videos_folder_full_path,
                    second_video_folder,
                )
                for video_file in os.listdir(video_folder_full_path):
                    if video_file.endswith(".mp4"):
                        # Get the index of the video
                        video_index = int(video_file.split("_")[-1].split(".")[0])
                        # Rename the video file
                        new_video_file = f"episode_{video_index + nb_videos:06d}.mp4"
                        # Move the video file to the new dataset
                        shutil.copy(
                            os.path.join(video_folder_full_path, video_file),
                            os.path.join(path_to_videos, video_folder, new_video_file),
                        )

        ### META DATA
        logger.debug("Recreating meta files")
        meta_folder_path = os.path.join(path_result_dataset, "meta")
        os.makedirs(meta_folder_path, exist_ok=True)

        #### Tasks Model
        logger.debug("Creating tasks.jsonl")
        initial_tasks = TasksModel.from_jsonl(
            meta_folder_path=self.meta_folder_full_path
        )
        second_tasks = TasksModel.from_jsonl(
            meta_folder_path=second_dataset.meta_folder_full_path
        )
        tasks_mapping_second_to_first, new_number_of_tasks = initial_tasks.merge_with(
            second_task_model=second_tasks,
            meta_folder_to_save_to=meta_folder_path,
        )

        #### Episodes Model
        logger.debug("Creating episodes.jsonl")
        initial_episodes = EpisodesModel.from_jsonl(
            meta_folder_path=self.meta_folder_full_path,
            format="lerobot_v2.1",
        )
        second_episodes = EpisodesModel.from_jsonl(
            meta_folder_path=second_dataset.meta_folder_full_path,
            format="lerobot_v2.1",
        )
        initial_episodes.merge_with(
            second_episodes_model=second_episodes,
            meta_folder_to_save_to=meta_folder_path,
        )

        #### Info Model
        logger.debug("Creating info.json")

        first_info = InfoModel.from_json(
            meta_folder_path=self.meta_folder_full_path,
            format="lerobot_v2.1",
        )
        second_info = InfoModel.from_json(
            meta_folder_path=second_dataset.meta_folder_full_path,
            format="lerobot_v2.1",
        )

        first_info.merge_with(
            second_info_model=second_info,
            meta_folder_to_save_to=meta_folder_path,
            new_nb_tasks=new_number_of_tasks,
        )

        #### Stats model
        logger.debug("Creating episodes_stats.jsonl")

        first_stats = EpisodesStatsModel.from_jsonl(
            meta_folder_path=self.meta_folder_full_path
        )
        second_stats = EpisodesStatsModel.from_jsonl(
            meta_folder_path=second_dataset.meta_folder_full_path
        )
        first_stats.merge_with(
            second_stats_model=second_stats,
            meta_folder_path=meta_folder_path,
        )

        ### PARQUET FILES
        logger.debug("Moving parquet files to the new dataset")

        # Move the parquet files to the new dataset
        path_to_data = os.path.join(
            path_result_dataset,
            "data",
            "chunk-000",
        )
        os.makedirs(path_to_data, exist_ok=True)
        for parquet_file in os.listdir(self.data_folder_full_path):
            if parquet_file.endswith(".parquet"):
                # Move the parquet file to the new dataset
                shutil.copy(
                    os.path.join(self.data_folder_full_path, parquet_file),
                    os.path.join(path_to_data, parquet_file),
                )

        # Reload the first dataset info model
        first_info = InfoModel.from_json(
            meta_folder_path=self.meta_folder_full_path,
            format="lerobot_v2.1",
        )

        # Move the parquet files from the second dataset to the new dataset and edit them
        # - Rename the parquet file
        # - Update the episode index in the parquet file
        # - Update the index in the parquet file
        # - Update the task index in the parquet file

        for parquet_file in os.listdir(second_dataset.data_folder_full_path):
            if parquet_file.endswith(".parquet"):
                # Get the index of the parquet file
                parquet_index = int(parquet_file.split("_")[-1].split(".")[0])
                # Rename the parquet file
                new_parquet_file = (
                    f"episode_{parquet_index + first_info.total_episodes:06d}.parquet"
                )
                # Move the parquet file to the new dataset
                shutil.copy(
                    os.path.join(second_dataset.data_folder_full_path, parquet_file),
                    os.path.join(path_to_data, new_parquet_file),
                )

                # Load the parquet file
                df = pd.read_parquet(
                    os.path.join(path_to_data, new_parquet_file),
                )

                df["episode_index"] = df["episode_index"] + first_info.total_episodes
                df["task_index"] = df["task_index"].replace(
                    tasks_mapping_second_to_first,
                )
                df["index"] = df["index"] + first_info.total_frames

                # Save the parquet file
                df.to_parquet(
                    os.path.join(path_to_data, new_parquet_file),
                )

        # Create README file
        logger.debug("Creating README file")
        readme_path = os.path.join(path_result_dataset, "README.md")
        if not os.path.exists(readme_path):
            with open(readme_path, "w", encoding=DEFAULT_FILE_ENCODING) as readme_file:
                readme_file.write(self.generate_read_me_string(new_dataset_name))

        logger.info(f"Dataset {new_dataset_name} created successfully")

    def split_dataset(
        self, split_ratio: float, first_split_name: str, second_split_name: str
    ) -> None:
        """
        Split the dataset into two parts based on a given ratio.
        This will create two new datasets with the split data.
        The first dataset will contain split ratio of the original dataset,
        split_ratio should be between 0 and 1

        Note: This method is intended to work for v2.1 format only.

        Dataset Structure

        / videos
            ├── chunk-000
            │   ├── observation.images.main
            │   |   ├── episode_000000.mp4
            │   ├── observation.images.secondary_0
            │   |   ├── episode_000000.mp4
        / data
            ├── chunk-000
            │   ├── episode_000000.parquet
        / meta
            ├── info.json
            ├── tasks.jsonl
            ├── episodes.jsonl
            ├── episodes_stats.jsonl
        / README.md
        """
        if split_ratio <= 0 or split_ratio >= 1:
            raise ValueError(f"Split ratio {split_ratio} should be between 0 and 1")

        first_dataset_path = os.path.join(
            os.path.dirname(self.folder_full_path),
            first_split_name,
        )
        second_split_name_path = os.path.join(
            os.path.dirname(self.folder_full_path),
            second_split_name,
        )
        # If the dataset already exists, raise an error
        if os.path.exists(first_dataset_path):
            raise ValueError(
                f"Dataset {first_split_name} already exists in {first_dataset_path}"
            )
        if os.path.exists(second_split_name_path):
            raise ValueError(
                f"Dataset {second_split_name} already exists in {second_split_name_path}"
            )
        os.makedirs(first_dataset_path, exist_ok=True)
        os.makedirs(second_split_name_path, exist_ok=True)

        ### Find number of episodes

        path_to_data = os.path.join(
            self.folder_full_path,
            "data",
            "chunk-000",
        )
        nbr_of_episodes = len(
            [
                f
                for f in os.listdir(path_to_data)
                if f.endswith(".parquet") and f.startswith("episode_")
            ]
        )

        first_dataset_number_of_episodes = int(nbr_of_episodes * split_ratio)
        second_dataset_number_of_episodes = (
            nbr_of_episodes - first_dataset_number_of_episodes
        )
        logger.debug(
            f"Splitting dataset {self.dataset_name} into {first_split_name}: {first_dataset_number_of_episodes} and {second_split_name}: {second_dataset_number_of_episodes}"
        )

        ### VIDEOS
        logger.debug("Spliting videos")
        first_dataset_videos_path = os.path.join(
            first_dataset_path,
            "videos",
            "chunk-000",
        )
        os.makedirs(first_dataset_videos_path, exist_ok=True)
        second_dataset_videos_path = os.path.join(
            second_split_name_path,
            "videos",
            "chunk-000",
        )
        os.makedirs(second_dataset_videos_path, exist_ok=True)

        # Grab videos and move them
        for video_folder in os.listdir(self.videos_folder_full_path):
            if "image" in video_folder:
                # find all videos and sort them by name
                video_folder_full_path = os.path.join(
                    self.videos_folder_full_path, video_folder
                )
                video_files = [
                    f for f in os.listdir(video_folder_full_path) if f.endswith(".mp4")
                ]
                video_files.sort()
                # Split the videos into two parts
                first_dataset_video_files = video_files[
                    :first_dataset_number_of_episodes
                ]
                second_dataset_video_files = video_files[
                    first_dataset_number_of_episodes : first_dataset_number_of_episodes
                    + second_dataset_number_of_episodes
                ]
                # Move the video filed to the new dataset and rename the second ones
                os.makedirs(
                    os.path.join(first_dataset_videos_path, video_folder),
                    exist_ok=True,
                )
                for video_file in first_dataset_video_files:
                    # Move the video file to the new dataset
                    shutil.copy(
                        os.path.join(video_folder_full_path, video_file),
                        os.path.join(
                            first_dataset_videos_path,
                            video_folder,
                            video_file,
                        ),
                    )

                os.makedirs(
                    os.path.join(second_dataset_videos_path, video_folder),
                    exist_ok=True,
                )
                for video_file in second_dataset_video_files:
                    # Get the index of the video
                    video_index = (
                        int(video_file.split("_")[-1].split(".")[0])
                        - first_dataset_number_of_episodes
                    )
                    # Rename the video file
                    new_video_file = f"episode_{video_index:06d}.mp4"
                    # Move the video file to the new dataset
                    shutil.copy(
                        os.path.join(video_folder_full_path, video_file),
                        os.path.join(
                            second_dataset_videos_path,
                            video_folder,
                            new_video_file,
                        ),
                    )

        ### META FILES
        logger.debug("Creating meta files")
        first_meta_folder_path = os.path.join(first_dataset_path, "meta")
        second_meta_folder_path = os.path.join(second_split_name_path, "meta")
        os.makedirs(first_meta_folder_path, exist_ok=True)
        os.makedirs(second_meta_folder_path, exist_ok=True)

        #### episodes.jsonl
        logger.debug("Creating episodes.jsonl")

        initial_episodes = EpisodesModel.from_jsonl(
            meta_folder_path=self.meta_folder_full_path,
            format="lerobot_v2.1",
        )
        first_episodes_model, second_episodes_model = initial_episodes.split(
            split_ratio=split_ratio
        )
        first_episodes_model.save(
            meta_folder_path=first_meta_folder_path,
            save_mode="overwrite",
        )
        second_episodes_model.save(
            meta_folder_path=second_meta_folder_path,
            save_mode="overwrite",
        )

        #### episodes_stats.jsonl
        logger.debug("Creating episodes_stats.jsonl")

        initial_stats = EpisodesStatsModel.from_jsonl(
            meta_folder_path=self.meta_folder_full_path
        )
        first_episodes_stats_model, second_episodes_stats_model = initial_stats.split(
            split_ratio=split_ratio,
        )
        first_episodes_stats_model.save(
            meta_folder_path=first_meta_folder_path,
        )
        second_episodes_stats_model.save(
            meta_folder_path=second_meta_folder_path,
        )

        #### tasks.jsonl
        logger.debug("Creating tasks.jsonl")

        initial_tasks = TasksModel.from_jsonl(
            meta_folder_path=self.meta_folder_full_path
        )
        # Reload episodes to make sure we haven't modified them
        initial_episodes = EpisodesModel.from_jsonl(
            meta_folder_path=self.meta_folder_full_path,
            format="lerobot_v2.1",
        )
        (
            first_tasks,
            second_tasks,
            first_dataset_number_of_tasks,
            second_dataset_number_of_tasks,
            old_task_mapping_to_new,
        ) = initial_tasks.split(
            split_ratio=split_ratio, initial_episodes_model=initial_episodes
        )
        first_tasks.save(
            meta_folder_path=first_meta_folder_path,
        )
        second_tasks.save(
            meta_folder_path=second_meta_folder_path,
        )

        ### PARQUET FILES
        logger.debug("Spliting parquet files")

        first_dataset_data_path = os.path.join(
            first_dataset_path,
            "data",
            "chunk-000",
        )
        os.makedirs(first_dataset_data_path, exist_ok=True)
        second_dataset_data_path = os.path.join(
            second_split_name_path,
            "data",
            "chunk-000",
        )
        os.makedirs(second_dataset_data_path, exist_ok=True)

        # Move the parquet files to the new dataset and rename them
        # List and sort all parquets

        parquet_files = [
            f
            for f in os.listdir(self.data_folder_full_path)
            if f.endswith(".parquet") and f.startswith("episode_")
        ]
        parquet_files.sort()

        index_in_second_dataset = 0
        for parquet_file in parquet_files:
            # Get the index of the parquet file
            parquet_index = int(parquet_file.split("_")[-1].split(".")[0])
            # Rename the parquet file
            new_parquet_file = f"episode_{parquet_index:06d}.parquet"
            # Move the parquet file to the new dataset
            if parquet_index < first_dataset_number_of_episodes:
                shutil.copy(
                    os.path.join(path_to_data, parquet_file),
                    os.path.join(first_dataset_data_path, new_parquet_file),
                )
            else:
                # Rename the parquet file
                new_parquet_file = f"episode_{parquet_index - first_dataset_number_of_episodes:06d}.parquet"
                shutil.copy(
                    os.path.join(path_to_data, parquet_file),
                    os.path.join(second_dataset_data_path, new_parquet_file),
                )
                # Load parquet file
                df = pd.read_parquet(
                    os.path.join(second_dataset_data_path, new_parquet_file),
                )
                # Update the episode index in the parquet file
                df["episode_index"] = (
                    df["episode_index"] - first_dataset_number_of_episodes
                )
                # Update the index in the parquet file
                df["index"] = np.arange(len(df)) + index_in_second_dataset
                # Update the task index in the parquet file
                df["task_index"] = df["task_index"].replace(
                    old_task_mapping_to_new,
                )
                index_in_second_dataset += len(df)
                # Save the parquet file
                df.to_parquet(
                    os.path.join(second_dataset_data_path, new_parquet_file),
                )

        #### info.json
        # We create the info files last, because we need info from the other files
        logger.debug("Creating info.json")

        initial_info = InfoModel.from_json(
            meta_folder_path=self.meta_folder_full_path,
            format="lerobot_v2.1",
        )
        number_of_cameras = initial_info.total_videos // initial_info.total_episodes

        first_info = InfoModel.from_json(
            meta_folder_path=self.meta_folder_full_path,
            format="lerobot_v2.1",
        )
        first_info.total_episodes = first_dataset_number_of_episodes
        first_info.total_frames = initial_info.total_frames - index_in_second_dataset
        first_info.total_tasks = first_dataset_number_of_tasks
        first_info.splits = {
            "train": f"0:{first_dataset_number_of_episodes}",
        }
        first_info.total_videos = first_dataset_number_of_episodes * number_of_cameras

        second_info = InfoModel.from_json(
            meta_folder_path=self.meta_folder_full_path,
            format="lerobot_v2.1",
        )
        second_info.total_episodes = second_dataset_number_of_episodes
        second_info.total_frames = index_in_second_dataset
        second_info.total_tasks = second_dataset_number_of_tasks
        second_info.splits = {
            "train": f"{0}:{second_dataset_number_of_episodes}",
        }
        second_info.total_videos = second_dataset_number_of_episodes * number_of_cameras
        first_info.save(
            meta_folder_path=first_meta_folder_path,
        )
        second_info.save(
            meta_folder_path=second_meta_folder_path,
        )

        #### Create README.md files
        logger.debug("Creating README.md files")

        first_readme_path = os.path.join(first_dataset_path, "README.md")
        if not os.path.exists(first_readme_path):
            with open(
                first_readme_path,
                "w",
                encoding=DEFAULT_FILE_ENCODING,
            ) as readme_file:
                readme_file.write(self.generate_read_me_string(first_split_name))
        second_readme_path = os.path.join(second_split_name_path, "README.md")
        if not os.path.exists(second_readme_path):
            with open(
                second_readme_path,
                "w",
                encoding=DEFAULT_FILE_ENCODING,
            ) as readme_file:
                readme_file.write(self.generate_read_me_string(second_split_name))

        logger.info(
            f"Dataset {self.dataset_name} split into {first_split_name} and {second_split_name} successfully"
        )

    def delete_video_keys(self, video_keys_to_delete: List[str]) -> None:
        """
        Delete the video keys from the dataset.
        Will update the info.json file and delete the videos from the data folder.

        video_keys_to_delete are of the form "observation.images.{video_key}"
        """
        # Step 1: update the info.json file
        info = InfoModel.from_json(
            meta_folder_path=self.meta_folder_full_path,
            format="lerobot_v2.1",
        )
        # This method save also the info.json file
        info.update_for_video_removal(video_keys_to_delete, self.meta_folder_full_path)

        # Step 2: delete the videos from the data folder
        for video_key in video_keys_to_delete:
            video_folder = os.path.join(self.videos_folder_full_path, video_key)
            if os.path.exists(video_folder):
                shutil.rmtree(video_folder)

        # Step 3: update the stats.jsonl file
        stats = StatsModel.from_json(self.meta_folder_full_path)
        stats.update_for_video_key_removal(
            video_keys_to_delete, self.meta_folder_full_path
        )

    def shuffle_dataset(self, new_dataset_name) -> None:
        """
        Shuffle the episodes in the dataset inplace.
        Expects a dataset in v2.1 format.
        This will pick a random shuffle of the episodes and apply it to the videos, data and meta files.
        """
        # if self.format_version != "lerobot_v2.1":
        #     raise ValueError(
        #         f"Dataset {self.dataset_name} is not in v2.1 format, cannot shuffle"
        #     )
        # TODO: add check on info.json

        # Find the number of episodes in the dataset
        logger.info("Shuffling the dataset episodes")

        # Get the number of episodes from the info.json file
        info = InfoModel.from_json(
            meta_folder_path=self.meta_folder_full_path,
            format="lerobot_v2.1",
        )

        episodes_model = EpisodesModel.from_jsonl(
            meta_folder_path=self.meta_folder_full_path,
            format="lerobot_v2.1",
        )

        number_of_episodes = info.total_episodes
        shuffle = np.random.permutation(number_of_episodes)
        # Generate a mapping of type Dict[int, int] that maps the old episode index to the new episode index
        # This will be used to reindex the episodes.jsonl file
        old_index_to_new_index = {k: int(v) for k, v in enumerate(shuffle)}

        # Reindex the data folder
        old_index_to_new_index = self.reindex_episodes(
            folder_path=self.data_folder_full_path,
        )
        # Reindex the episode videos
        for camera_folder_full_path in self.get_camera_folders_full_paths():
            self.reindex_episodes(
                folder_path=camera_folder_full_path,
                old_index_to_new_index=old_index_to_new_index,  # type: ignore
            )

        episodes_model.update_for_episode_removal(
            -1,
            old_index_to_new_index=old_index_to_new_index,
        )
        episodes_model.save(
            meta_folder_path=self.meta_folder_full_path, save_mode="overwrite"
        )
        ### Meta files ###

        #### TASKS
        # No need to shuffle, just copy

        #### INFO
        # No need to shuffle

        #### EPISODES
        logger.debug("Shuffling episodes.jsonl file")
        episodes_model = EpisodesModel.from_jsonl(
            meta_folder_path=self.meta_folder_full_path,
            format="lerobot_v2.1",
        )
        episodes_model.shuffle(
            permutation=shuffle,
        )
        episodes_model.save(
            meta_folder_path=self.meta_folder_full_path,
            save_mode="overwrite",
        )

        #### EPISODES STATS
        logger.debug("Shuffling episodes_stats.jsonl file")
        episodes_stats_model = EpisodesStatsModel.from_jsonl(
            meta_folder_path=self.meta_folder_full_path,
        )
        episodes_stats_model.shuffle(
            permutation=shuffle,
        )
        episodes_stats_model.save(
            meta_folder_path=self.meta_folder_full_path,
        )

        logger.info(f"Dataset shuffled successfully at {self.folder_full_path}")

    def reindex_episodes(
        self,
        folder_path: str,
        nb_steps_deleted_episode: int = 0,
        old_index_to_new_index: Optional[Dict[int, int]] = None,
    ) -> Dict[int, int]:
        """
        Reindex the episode after removing one.
        This is used for videos or for parquet file

        Parameters:
        -----------
        folder_path: str
            The path to the folder where the episodes data or videos are stored. May be users/Downloads/dataset_name/data/chunk-000/
            or  users/Downloads/dataset_name/videos/chunk-000/observation.main_image.right
        nb_steps_deleted_episode: int
            Unused
        old_index_to_new_index: Optional[Dict[int, int]]
            A dictionary mapping old indices to new indices. If None, the function will create a new mapping.

        Returns:
        --------
        Dict[int, int]
            A dictionary mapping old indices to new indices.

        Example:
        --------
        episodes in data are [episode_000000.parquet, episode_000001.parquet, episode_000003.parquet] after we removed episode_000002.parquet
        the result will be [episode_000000.parquet, episode_000001.parquet, episode_000002.parquet]
        """

        # Create a mapping of old index to new index
        if old_index_to_new_index is None:
            old_index_to_new_index = {}
            current_new_index_max = 0

            for filename in sorted(os.listdir(folder_path)):
                if filename.startswith("episode_"):
                    # Check the episode files and extract the index
                    # Also extract the file extension
                    file_extension = filename.split(".")[-1]
                    old_index = int(filename.split("_")[-1].split(".")[0])
                    old_index_to_new_index[old_index] = current_new_index_max
                    current_new_index_max += 1

            if len(old_index_to_new_index.keys()) > 0:
                current_new_index_max = max(old_index_to_new_index.values()) + 1
            else:
                current_new_index_max = 0
        else:
            current_new_index_max = 0

        # Reindex the files in the folder
        total_nb_steps = 0
        for filename in sorted(os.listdir(folder_path)):
            if filename.startswith("episode_"):
                file_extension = filename.split(".")[-1]

                old_index = int(filename.split("_")[-1].split(".")[0])
                if old_index in old_index_to_new_index.keys():
                    new_index = old_index_to_new_index[old_index]
                else:
                    # If the file is not in the mapping, we need to create a new index
                    new_index = current_new_index_max
                    current_new_index_max += 1

                new_filename = f"episode_{new_index:06d}.{file_extension}"
                os.rename(
                    os.path.join(folder_path, filename),
                    os.path.join(folder_path, new_filename),
                )

                # Update the episode index inside the parquet file
                if file_extension == "parquet":
                    # Removed the assertion to use in dataset shuffling
                    # It's now ok to have 0 steps deleted

                    # Read the parquet file
                    df = pd.read_parquet(os.path.join(folder_path, new_filename))
                    # Use the mapping to update the episode index
                    df["episode_index"] = df["episode_index"].replace(
                        old_index_to_new_index
                    )
                    # Replace the global index (total number of steps in the dataset)
                    # First, make it from zero to the total number of rows
                    df["index"] = np.arange(len(df))
                    # Then, add the total number of steps in the dataset to the index
                    df["index"] = df["index"] + total_nb_steps
                    # Update the total number of steps
                    total_nb_steps += len(df)
                    # Save the updated parquet file
                    df.to_parquet(os.path.join(folder_path, new_filename))

                if file_extension == "json":
                    # Update the episode index inside the json file with pandas
                    df = pd.read_json(os.path.join(folder_path, new_filename))
                    df["episode_index"] = df["episode_index"].replace(
                        old_index_to_new_index
                    )
                    df.to_json(os.path.join(folder_path, new_filename))

        return old_index_to_new_index

    def get_episode_data_path(self, episode_id: int) -> str:
        """Get the full path to the data with episode id"""
        return os.path.join(
            self.folder_full_path,
            "data",
            "chunk-000",
            f"episode_{episode_id:06d}.parquet",
        )

    @property
    def meta_folder_full_path(self) -> str:
        """Get the folder path of the dataset"""
        return str(self.folder_full_path / "meta")

    @property
    def data_folder_full_path(self) -> str:
        """Get the full path to the data folder"""
        return str(self.folder_full_path / "data" / "chunk-000")

    @property
    def videos_folder_full_path(self) -> str:
        """Get the full path to the videos folder"""
        return str(self.folder_full_path / "videos" / "chunk-000")

    def get_camera_folders_full_paths(self) -> List[str]:
        """
        Return the full path to the camera folders.
        This contains episode_000000.mp4, episode_000001.mp4, etc.
        """
        return [
            os.path.join(self.videos_folder_full_path, camera_name)
            for camera_name in os.listdir(self.videos_folder_full_path)
        ]

    def get_camera_folders_repo_paths(self) -> List[str]:
        """
        Return the path to the camera folder in the repository
        """
        return [
            f"videos/chunk-000/{camera_name}"
            for camera_name in os.listdir(self.videos_folder_full_path)
        ]


class LeRobotEpisode(BaseEpisode):
    # Direct attributes, not in metadata for Pydantic validation and type hinting
    dataset_manager: LeRobotDataset  # Link to the dataset it belongs to
    freq: int  # Recording frequency (Hz)
    codec: VideoCodecs  # For saving videos
    target_size: tuple[int, int]  # For video creation (width, height)

    # Paths are derived from the dataset_manager and episode_index (from metadata)
    @property
    def _parquet_path(self) -> Path:
        # episode_index comes from self.metadata
        return (
            Path(self.dataset_manager.data_folder_full_path)
            / f"episode_{self.episode_index:06d}.parquet"
        )

    @property
    def episodes_path(self) -> Path:
        """
        Return the file path of the episode
        """
        path = self.dataset_path / "data" / "chunk-000"
        os.makedirs(path, exist_ok=True)
        return path

    def _get_video_path(self, camera_key: str) -> Path:
        # episode_index comes from self.metadata
        path = Path(self.dataset_manager.videos_folder_full_path) / camera_key
        os.makedirs(path, exist_ok=True)  # Ensure camera-specific subfolder exists
        return path / f"episode_{self.episode_index:06d}.mp4"

    @property
    def dataset_path(self) -> Path:
        """
        Return the file path of the episode
        """
        format = self.metadata.get("format")
        dataset_name = self.metadata.get("dataset_name")
        if not format:
            raise ValueError("Episode metadata.format not set")
        if not dataset_name:
            raise ValueError("Episode metadata.dataset_name not set")

        path = get_home_app_path() / "recordings" / format / dataset_name
        os.makedirs(path, exist_ok=True)

        return path

    @property
    def _cameras_folder_path(self) -> Path:
        """
        Return the cameras folder path
        """
        return self.dataset_path / "videos" / "chunk-000"

    @classmethod
    async def start_new(  # type: ignore[override]
        cls,
        dataset_manager: LeRobotDataset,  # Pass the fully initialized dataset manager
        robots: List[BaseRobot],
        codec: VideoCodecs,
        freq: int,
        target_size: tuple[int, int],  # width, height
        instruction: Optional[str],
        all_camera_key_names: List[str],
        **kwargs,
    ) -> "LeRobotEpisode":
        # Ensure meta models are loaded/initialized in the dataset manager
        dataset_manager.load_meta_models(
            robots=robots,
            codec=codec,
            target_size=target_size,  # Used by InfoModel if creating new
            fps=freq,
            all_camera_key_names=all_camera_key_names,
        )

        # These must not be None after the above call
        assert dataset_manager.info_model is not None
        assert dataset_manager.episodes_model is not None
        assert dataset_manager.tasks_model is not None
        if dataset_manager.format_version == "lerobot_v2.1":
            assert dataset_manager.episodes_stats_model is not None
        elif dataset_manager.format_version == "lerobot_v2":
            assert dataset_manager.stats_model is not None

        episode_idx = (
            dataset_manager.get_next_episode_index()
        )  # e.g., if 0 episodes, next is 0

        task_idx = len(dataset_manager.tasks_model.tasks)  # Default: new task
        if instruction:  # If an instruction is provided, try to find its index
            for task_feature in dataset_manager.tasks_model.tasks:
                if task_feature.task == instruction:
                    task_idx = task_feature.task_index
                    break
            # If not found, it will be added as a new task when tasks_model.update() is called

        start_timestamp = time.time()
        episode_metadata = {
            "episode_index": episode_idx,
            "created_at": start_timestamp,
            "robot_type": ", ".join(r.name for r in robots),
            "format": dataset_manager.format_version,
            "dataset_name": dataset_manager.dataset_name,
            "instruction": instruction,  # Store the actual instruction string
            "task_index": task_idx,  # Store the determined or default task index
            "freq": freq,  # Store freq for reference, e.g. in play method
        }
        logger.info(
            f"Starting new LeRobotEpisode, index: {episode_idx}, task: '{instruction}' (idx: {task_idx}) for dataset '{dataset_manager.dataset_name}'."
        )

        return cls(
            steps=[],
            metadata=episode_metadata,
            dataset_manager=dataset_manager,
            freq=freq,
            codec=codec,
            target_size=target_size,
        )

    async def append_step(self, step: Step, **kwargs) -> None:
        self.add_step(step)  # Appends to self.steps, manages is_first/is_last flags

        current_step_in_episode_index = (
            len(self.steps) - 1
        )  # 0-indexed count of steps in this episode

        # Update live meta models stored in the dataset_manager
        if self.dataset_manager.format_version == "lerobot_v2.1":
            assert self.dataset_manager.episodes_stats_model is not None
            self.dataset_manager.episodes_stats_model.update(
                step=step,
                episode_index=self.episode_index,  # from self.metadata
                current_step_index=current_step_in_episode_index,
            )
        elif self.dataset_manager.format_version == "lerobot_v2":
            assert self.dataset_manager.stats_model is not None
            self.dataset_manager.stats_model.update(
                step=step,
                episode_index=self.episode_index,
                current_step_index=current_step_in_episode_index,
            )

        assert self.dataset_manager.episodes_model is not None
        self.dataset_manager.episodes_model.update(
            step=step, episode_index=self.episode_index
        )

        assert self.dataset_manager.tasks_model is not None
        # tasks_model.update will add the instruction as a new task if it's not already present
        self.dataset_manager.tasks_model.update(step=step)

    def _convert_to_le_robot_episode_model(self) -> "LeRobotEpisodeModel":
        """Converts internal steps to the LeRobotEpisodeModel for Parquet saving."""
        assert self.dataset_manager.info_model is not None
        # global_frame_offset is the total number of frames in the dataset *before* this episode's frames.
        global_frame_offset = self.dataset_manager.info_model.total_frames

        episode_data_dict: Dict[str, List] = {
            "action": [],
            "observation.state": [],
            "timestamp": [],
            "task_index": [],
            "episode_index": [],
            "frame_index": [],
            "index": [],
        }

        # We rewrite the timestamps based on the frequency to validate LeRobot tests
        timestamps_for_episode = (np.arange(len(self.steps)) / self.freq).tolist()

        episode_data_dict["timestamp"] = timestamps_for_episode

        for local_frame_idx, step_item in enumerate(self.steps):
            episode_data_dict["episode_index"].append(
                self.episode_index
            )  # from self.metadata
            episode_data_dict["frame_index"].append(
                local_frame_idx
            )  # 0 to N-1 for this episode
            # LeRobot's "observation.state" is our "joints_position"
            episode_data_dict["observation.state"].append(
                step_item.observation.joints_position.astype(np.float32)
            )
            # "index" is the global frame index across the entire dataset
            episode_data_dict["index"].append(local_frame_idx + global_frame_offset)
            episode_data_dict["task_index"].append(self.metadata["task_index"])

            if (
                step_item.action is None
            ):  # Should have been set by update_previous_step, except for the very last step
                if (
                    local_frame_idx == len(self.steps) - 1
                ):  # Last step, action can be a placeholder (e.g., repeat joints_position)
                    logger.debug(
                        f"Last step (idx {local_frame_idx}) of episode {self.episode_index} has no action; using its own joints_position."
                    )
                    episode_data_dict["action"].append(
                        step_item.observation.joints_position.astype(
                            np.float32
                        ).tolist()
                    )
                else:  # Should not happen for intermediate steps
                    raise ValueError(
                        f"Step {local_frame_idx} in episode {self.episode_index} has no action set before saving."
                    )
            else:
                episode_data_dict["action"].append(
                    step_item.action.astype(np.float32).tolist()
                )

        # Basic validation for main camera frames (if any)
        main_cam_frames = self.get_episode_frames_main_camera()
        if main_cam_frames:
            first_frame_shape = main_cam_frames[0].shape
            if not all(
                f.shape == first_frame_shape and f.ndim == 3 for f in main_cam_frames
            ):
                logger.warning(
                    "Main camera frames have inconsistent shapes or dimensions."
                )
                # This could be an error depending on strictness:
                # raise ValueError("All main_image frames must have the same dimensions and be 3-channel RGB images.")

        return LeRobotEpisodeModel(**episode_data_dict)

    async def save(self, **kwargs) -> None:
        if not self.steps:
            logger.warning(
                f"LeRobotEpisode {self.episode_index} has no steps. Skipping save."
            )
            return

        logger.info(
            f"Saving LeRobotEpisode {self.episode_index} for dataset '{self.dataset_manager.dataset_name}'..."
        )
        assert (
            self.dataset_manager.info_model is not None
        )  # Should have been initialized

        # Sanity check: make sure the actions and observations don't have any null or nan values
        for step in self.steps:
            if step.action is None or np.isnan(step.action).any():
                # Attempt to repair the action by using the observation's joints_position
                if (
                    step.observation.joints_position is not None
                    and not np.isnan(step.observation.joints_position).any()
                ):
                    logger.warning(
                        f"Action in episode {self.episode_index} is None or NaN, automatically filling in the value."
                    )
                    step.action = step.observation.joints_position
                else:
                    raise ValueError(
                        f"Step action in episode {self.episode_index} is None or NaN"
                    )
            if (
                step.observation.joints_position is None
                or np.isnan(step.observation.joints_position).any()
            ):
                # Attempt to repair the observation by using the action
                if step.action is not None and not np.isnan(step.action).any():
                    logger.warning(
                        f"Observation in episode {self.episode_index} is None or NaN, automatically filling in the value."
                    )
                    step.observation.joints_position = step.action
                else:
                    raise ValueError(
                        f"Step observation in episode {self.episode_index} is None or NaN"
                    )

        # 1. Save Parquet data for the episode
        lerobot_parquet_model = self._convert_to_le_robot_episode_model()
        lerobot_parquet_model.to_parquet(str(self._parquet_path))
        logger.debug(
            f"Episode data for {self.episode_index} saved to {self._parquet_path}"
        )

        # 2. Save Videos for the episode
        main_camera_frames = self.get_episode_frames_main_camera()
        secondary_camera_frames_by_cam = (
            self.get_episode_frames_secondary_cameras()
        )  # List of frame lists

        # Iterate through camera configurations in InfoModel to ensure all expected videos are handled
        for i, (cam_key_in_info, video_feature_details) in enumerate(
            self.dataset_manager.info_model.features.observation_images.items()
        ):
            frames_for_this_video: Optional[List[np.ndarray]] = None

            if cam_key_in_info == "observation.images.main":
                frames_for_this_video = main_camera_frames
            else:  # Secondary camera
                # Try to match cam_key_in_info with available secondary_camera_frames_by_cam
                # This assumes secondary_camera_frames_by_cam is ordered as per info_model's secondary keys.
                # A more robust match would use actual keys if secondary_images in Observation were a dict.
                # For now, simple positional matching after main.
                secondary_cam_idx = (
                    i - 1
                )  # If i=0 is main, i=1 is first secondary (idx 0 in list)
                if 0 <= secondary_cam_idx < len(secondary_camera_frames_by_cam):
                    frames_for_this_video = secondary_camera_frames_by_cam[
                        secondary_cam_idx
                    ]

            if frames_for_this_video and len(frames_for_this_video) > 0:
                video_file_path = self._get_video_path(camera_key=cam_key_in_info)
                # target_size for create_video_file is (width, height)
                # video_feature_details.shape is [height, width, channels]
                video_target_size = (
                    video_feature_details.shape[1],
                    video_feature_details.shape[0],
                )

                saved_path = create_video_file(
                    frames=np.array(
                        frames_for_this_video
                    ),  # create_video_file expects np.array of frames
                    output_path=str(video_file_path),
                    target_size=video_target_size,  # Ensure this is (width, height)
                    fps=video_feature_details.info.video_fps,
                    codec=video_feature_details.info.video_codec,
                )
                if (isinstance(saved_path, str) and os.path.exists(saved_path)) or (
                    isinstance(saved_path, tuple)
                    and all(os.path.exists(p) for p in saved_path)
                ):  # Stereo case
                    logger.debug(
                        f"Video for {cam_key_in_info} (episode {self.episode_index}) saved to {video_file_path}"
                    )
                else:
                    logger.error(
                        f"Failed to save video for {cam_key_in_info} (episode {self.episode_index}) to {video_file_path}"
                    )
            else:
                logger.warning(
                    f"No frames found for camera {cam_key_in_info} in episode {self.episode_index}. Skipping video saving."
                )

        # 3. Update Dataset-level InfoModel (after this episode is fully processed)
        self.dataset_manager.info_model.total_frames += len(self.steps)
        # total_episodes should be the count of saved episodes. If this is episode N, total_episodes becomes N+1.
        # This assumes episodes are saved sequentially and episode_index is 0-based.
        self.dataset_manager.info_model.total_episodes = self.episode_index + 1
        self.dataset_manager.info_model.total_videos += len(
            self.dataset_manager.info_model.features.observation_images
        )
        self.dataset_manager.info_model.splits = {
            "train": f"0:{self.dataset_manager.info_model.total_episodes}"
        }  # Update split range

        # Ensure total_tasks in info_model is up-to-date
        # self.metadata['task_index'] was set during start_new
        if self.metadata["task_index"] >= self.dataset_manager.info_model.total_tasks:
            self.dataset_manager.info_model.total_tasks = (
                self.metadata["task_index"] + 1
            )

        # 4. Save all (potentially updated) meta models from the dataset manager
        self.dataset_manager.save_all_meta_models()
        logger.success(
            f"LeRobotEpisode {self.episode_index} and all dataset meta files saved for '{self.dataset_manager.dataset_name}'."
        )

    @classmethod
    def from_parquet(
        cls,
        episode_data_path: str,
        format: Literal["lerobot_v2", "lerobot_v2.1"],
        dataset_path: Optional[str] = None,
    ) -> "LeRobotEpisode":
        """
        Load an episode data file. We only extract the information from the parquet data file.
        TODO(adle): Add more information in the Episode when loading from parquet data file from metafiles and videos
        """
        logger.debug(f"Loading episode from {episode_data_path} with format {format}")
        # Check that the file exists
        if not os.path.exists(episode_data_path):
            raise FileNotFoundError(f"Episode file {episode_data_path} not found.")

        episode_df = pd.read_parquet(episode_data_path)
        # Try to load the tasks.jsonl
        dataset_path = str(Path(episode_data_path).parent.parent.parent)

        tasks_path = os.path.join(dataset_path, "meta", "tasks.jsonl")
        if os.path.exists(tasks_path):
            # Load the tasks.jsonl file
            tasks_df = pd.read_json(tasks_path, lines=True)
            # Get the task index from the task index column
            episode_df["task_index"] = tasks_df["task_index"].iloc[0]
            # Merge the task index with the episode_df
            episode_df = pd.merge(
                episode_df,
                tasks_df[["task_index", "task"]],
                on="task_index",
                how="left",
            )

        # Rename the columns to match the expected names in the instance
        episode_df.rename(
            columns={
                "observation.state": "joints_position",
                "task": "language_instruction",
            },
            inplace=True,
        )
        # agregate the columns in joints_position, timestamp, main_image, state to a column observation.
        cols = ["joints_position", "timestamp"]
        if "language_instruction" in episode_df.columns:
            cols.append("language_instruction")
        # Create a new column "observation" that is a dict of the selected columns for each row
        episode_df["observation"] = episode_df[cols].to_dict(orient="records")

        # Add metadata to the episode
        # the path is like this : dataset_name/data/chunk-000/episode_xxxxxx.parquet
        # get the path : dataset_name
        if dataset_path is None:
            normalized = os.path.normpath(episode_data_path)
            parts = normalized.split(os.sep)
            if len(parts) >= 4:
                dataset_path = parts[-4]
            else:
                logger.warning(
                    f"Episode {episode_data_path} does not contain the dataset name in the path. Path should be: dataset_name/data/chunk-000/episode_xxxxxx.parquet"
                    + "Using parent folder name as dataset name."
                )
                dataset_path = os.path.basename(os.path.dirname(episode_data_path))

        # Load the dataset
        dataset = LeRobotDataset(path=dataset_path, enforce_path=False)
        if dataset.info_model:
            freq = dataset.info_model.fps
            codec = list(dataset.info_model.features.observation_images.values())[
                0
            ].info.video_codec
            target_size = list(dataset.info_model.features.observation_images.values())[
                0
            ].shape
        else:
            freq = 30
            codec = "avc1"
            target_size = [320, 240]

        metadata = {
            "dataset_name": dataset_path,
            "format": format,
            "index": episode_df["episode_index"].iloc[0],
            "episode_index": episode_df["episode_index"].iloc[0],
        }

        episode_model = cls(
            dataset_manager=dataset,
            steps=cast(List[Step], episode_df.to_dict(orient="records")),
            metadata=metadata,
            freq=freq,
            codec=codec,
            target_size=(target_size[0], target_size[1]),
        )
        return episode_model

    def parquet(self) -> pd.DataFrame:
        """
        Load the .parquet file of the episode. Only works for LeRobot format.
        """
        return pd.read_parquet(self._parquet_path)

    def delete(self, update_hub: bool = True, repo_id: Optional[str] = None) -> None:
        """
        Remove files related to the episode. Note: this doesn't update the meta files from the dataset.
        Call Dataset.delete_episode to update the meta files.

        If update_hub is True, the files will be removed from the Hugging Face repository.
        There is no verification that the files are actually in the repository or that the repository exists.
        You need to do that beforehand.
        """

        # Delete the parquet file
        try:
            logger.debug(f"Deleting parquet file {self._parquet_path}")
            os.remove(self._parquet_path)
        except FileNotFoundError:
            logger.warning(
                f"Parquet file {self._parquet_path} not found. Skipping deletion."
            )

        logger.debug(f"Episode deletion repo_id: {repo_id}, update_hub: {update_hub}")
        if update_hub and repo_id is not None:
            # In the huggingface dataset, we need to pass the relative path.
            relative_episode_path = (
                f"data/chunk-000/episode_{self.episode_index:06d}.parquet"
            )
            logger.debug(
                f"Deleting parquet file {relative_episode_path} from Hugging Face repo {repo_id}"
            )
            delete_file(
                repo_id=repo_id,
                path_in_repo=relative_episode_path,
                repo_type="dataset",
            )

        # Remove the video files from the HF repository
        if os.path.exists(self._cameras_folder_path):
            all_camera_folders = os.listdir(self._cameras_folder_path)
            for camera_key in all_camera_folders:
                if "image" not in camera_key:
                    continue
                try:
                    video_path = self._get_video_path(camera_key)
                    logger.debug(f"Deleting video file {video_path}")
                    os.remove(video_path)
                except FileNotFoundError:
                    logger.warning(
                        f"Video file {self._get_video_path(camera_key)} not found. Skipping deletion."
                    )
                if update_hub and repo_id is not None:
                    path_in_repo = f"videos/chunk-000/{camera_key}/episode_{self.episode_index:06d}.mp4"
                    logger.debug(
                        f"Deleting video file {path_in_repo} from Hugging Face repo {repo_id}"
                    )
                    delete_file(
                        repo_id=repo_id, path_in_repo=path_in_repo, repo_type="dataset"
                    )
        else:
            logger.warning(
                f"Cameras folder {self._cameras_folder_path} does not exist. Skipping video deletion."
            )


class LeRobotEpisodeModel(BaseModel):
    action: List[List[float]]
    observation_state: List[List[float]] = Field(
        validation_alias=AliasChoices("observation.state", "observation_state")
    )
    timestamp: List[float]
    task_index: List[int]
    episode_index: List[int]
    frame_index: List[int]
    index: List[int]

    @model_validator(mode="before")
    def validate_lengths(cls, values):
        fields_to_check = [
            "action",
            "observation_state",
            "timestamp",
            "task_index",
            "episode_index",
            "frame_index",
            "index",
        ]
        # Adjust for alias, Pydantic v2 uses the defined field name 'observation_state' internally
        # For validation, ensure the actual data key (could be 'observation.state') is handled by AliasChoices
        # and here we check the Pydantic model field name.

        lengths = []
        for field_name in fields_to_check:
            # Pydantic v2's AliasChoices handles mapping from 'observation.state' to 'observation_state' on input.
            # So 'values' dict here will have 'observation_state' as key if alias was used.
            val = values.get(field_name)
            if (
                val is None and field_name == "observation_state"
            ):  # Try the alias if not found
                val = values.get("observation.state")

            if val is not None:
                lengths.append(len(val))
            else:  # Field is missing, which might be an issue depending on requirements
                # For now, let's assume all these fields are mandatory for a valid parquet row
                raise ValueError(
                    f"Missing data for field '{field_name}' in LeRobotEpisodeModel."
                )

        if len(set(lengths)) > 1:
            raise ValueError(
                f"All items in LeRobotEpisodeModel must have the same length. Got lengths: {lengths}"
            )
        return values

    def to_parquet(self, filename: str):
        """
        Save the episode to a Parquet file
        """
        df = pd.DataFrame(self.model_dump(mode="python"))
        df.rename(columns={"observation_state": "observation.state"}, inplace=True)
        df.to_parquet(filename, index=False)


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

        with open(
            f"{meta_folder_path}/tasks.jsonl", "r", encoding=DEFAULT_FILE_ENCODING
        ) as f:
            tasks = []
            for line in f:
                tasks.append(TasksFeatures(**json.loads(line)))

        tasks_model = cls(tasks=tasks)
        # Do it after model init, otherwise pydantic ignores the value of _original_nb_total_tasks
        tasks_model._initial_nb_total_tasks = len(tasks)
        return tasks_model

    @classmethod
    def from_li_tasks(cls, li_tasks: List[str]) -> "TasksModel":
        """Create a TasksModel from a list of tasks."""
        # Ensure that the tasks are unique
        li_tasks = list(set(li_tasks))
        tasks = [
            TasksFeatures(task_index=i, task=task) for i, task in enumerate(li_tasks)
        ]
        return cls(tasks=tasks, _initial_nb_total_tasks=len(tasks))

    def to_jsonl(
        self,
        meta_folder_path: str,
        save_mode: Literal["append", "overwrite"] = "overwrite",
    ) -> None:
        """
        Write the tasks.jsonl file in the meta folder path.
        """

        if save_mode == "overwrite":
            with open(f"{meta_folder_path}/tasks.jsonl", "w") as f:
                for task in self.tasks:
                    f.write(task.model_dump_json() + "\n")
        elif save_mode == "append":
            with open(
                f"{meta_folder_path}/tasks.jsonl", "a", encoding=DEFAULT_FILE_ENCODING
            ) as f:
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

    def update_for_episode_removal(
        self, df_episode_to_delete: pd.DataFrame, data_folder_full_path=str
    ) -> None:
        """
        Update the tasks when removing an episode from the dataset.
        We count the number of occurences of task_index in the dataset.
        If the episode is the only one with this task_index, we remove it from the tasks.jsonl file.
        """
        if df_episode_to_delete.empty:
            return

        # For each file in the data folder get the task indexes (unique)
        task_indexes: List[int] = []
        for file in os.listdir(data_folder_full_path):
            if file.endswith(".parquet"):
                df = pd.read_parquet(f"{data_folder_full_path}/{file}")
                task_indexes.extend(df["task_index"].unique())

        for task_index in df_episode_to_delete["task_index"].unique():
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

    def merge_with(
        self, second_task_model: "TasksModel", meta_folder_to_save_to: str
    ) -> tuple[dict[int, int], int]:
        """
        Will merge the tasks.jsonl file with another one and save it to the new meta folder.
        Returns a task mapping of the old task index to the new one.
        """
        # Create a mapping of the old task index to the new one
        old_index_to_new_index: Dict[int, int] = {}

        for i, task_model in enumerate(second_task_model.tasks):
            # Check if the task is already in the first task model
            if task_model.task not in [t.task for t in self.tasks]:
                # If not, add it to the first task model
                old_index_to_new_index[task_model.task_index] = len(self.tasks)
                self.tasks.append(
                    TasksFeatures(task_index=len(self.tasks), task=task_model.task)
                )
            else:
                # If it is, update the mapping
                old_index_to_new_index[task_model.task_index] = [
                    t.task for t in self.tasks
                ].index(task_model.task)

        new_number_of_tasks = len(self.tasks)

        # Save the tasks.jsonl file
        self.to_jsonl(meta_folder_path=meta_folder_to_save_to, save_mode="overwrite")

        return old_index_to_new_index, new_number_of_tasks

    def split(self, split_ratio: float, initial_episodes_model: "EpisodesModel"):
        """
        Splits the tasks model into two parts.
        The first part contains the first split_ratio * len(tasks) tasks.
        The second part contains the rest of the tasks.
        """
        split_index = int(len(initial_episodes_model.episodes) * split_ratio)

        first_tasks = TasksModel(tasks=[])
        second_tasks = TasksModel(tasks=[])
        mapping_old_to_new_index: dict[int, int] = {}

        for episode in initial_episodes_model.episodes:
            if episode.episode_index < split_index:
                if episode.tasks[0] not in [
                    first_tasks.tasks[i].task for i in range(len(first_tasks.tasks))
                ]:
                    first_tasks.tasks.append(
                        TasksFeatures(
                            task_index=len(first_tasks.tasks), task=episode.tasks[0]
                        )
                    )
            else:
                if episode.tasks[0] not in [
                    second_tasks.tasks[i].task for i in range(len(second_tasks.tasks))
                ]:
                    second_tasks.tasks.append(
                        TasksFeatures(
                            task_index=len(second_tasks.tasks), task=episode.tasks[0]
                        )
                    )
                    # Find the previous task index in the initial tasks model
                    task_index = [
                        task.task_index
                        for task in self.tasks
                        if task.task == episode.tasks[0]
                    ]
                    if task_index:
                        mapping_old_to_new_index[task_index[0]] = len(
                            mapping_old_to_new_index
                        )
                    else:
                        raise ValueError(
                            f"Task {episode.tasks[0]} not found in the initial tasks model"
                        )

        # Count the number of different tasks in the first part
        nbr_tasks_first_part = len(first_tasks.tasks)
        nbr_tasks_second_part = len(second_tasks.tasks)

        return (
            first_tasks,
            second_tasks,
            nbr_tasks_first_part,
            nbr_tasks_second_part,
            mapping_old_to_new_index,
        )


class EpisodesFeatures(BaseModel):
    """
    Features for each line of the episodes.jsonl file.
    """

    episode_index: int = 0
    tasks: List[str] = []
    length: int = 0

    # Import tasks as a list of str if it is a str
    @field_validator("tasks", mode="before")
    def validate_tasks(cls, v):
        if isinstance(v, str):
            return [v]
        elif isinstance(v, list):
            return v
        else:
            raise ValueError("tasks must be a list of strings or a single string")


class EpisodesModel(BaseModel):
    """
    Follow the structure of the episodes.jsonl file.
    """

    episodes: List[EpisodesFeatures] = Field(default_factory=list)
    _episodes_features: Optional[Dict[int, EpisodesFeatures]] = None
    _original_nb_total_episodes: int = 0

    def update(self, step: Step, episode_index: int) -> None:
        """
        Updates the episodes with the given episode.
        episode_index is the index of the episode of the current step.
        """
        # If episode_index is not in the episodes, add it

        if self._episodes_features is None:
            self._episodes_features = {
                episode.episode_index: episode for episode in self.episodes
            }

        episode = self._episodes_features.get(episode_index)

        if episode is not None:
            # Increase the nb frames counter
            episode.length += 1
            # Add the language instruction if it's a new one
            if str(step.observation.language_instruction) not in episode.tasks:
                episode.tasks.append(str(step.observation.language_instruction))
        else:
            # Create a new episode
            new_episode = EpisodesFeatures(
                episode_index=episode_index,
                tasks=[str(step.observation.language_instruction)],
                length=1,
            )
            self.episodes.append(new_episode)
            self._episodes_features[episode_index] = new_episode

    def to_jsonl(
        self, meta_folder_path: str, save_mode: Literal["append", "overwrite"]
    ) -> None:
        """
        Write the episodes.jsonl file in the meta folder path.
        """
        if save_mode == "append":
            with open(f"{meta_folder_path}/episodes.jsonl", "a") as f:
                for episode in self.episodes[self._original_nb_total_episodes :]:
                    f.write(episode.model_dump_json() + "\n")
        elif save_mode == "overwrite":
            if self._episodes_features is None:
                self._episodes_features = {
                    episode.episode_index: episode for episode in self.episodes
                }

            with open(f"{meta_folder_path}/episodes.jsonl", "w") as f:
                for episode in self._episodes_features.values():
                    f.write(episode.model_dump_json() + "\n")
        else:
            raise ValueError("save_mode must be 'append' or 'overwrite'")

    @classmethod
    def from_jsonl(
        cls, meta_folder_path: str, format: Literal["lerobot_v2", "lerobot_v2.1"]
    ) -> "EpisodesModel":
        """
        Read the episodes.jsonl file in the meta folder path.
        """
        if not os.path.exists(f"{meta_folder_path}/episodes.jsonl"):
            return EpisodesModel()

        incorrect_episodes = False
        _episodes_features: dict[int, EpisodesFeatures] = {}

        with open(
            f"{meta_folder_path}/episodes.jsonl", "r", encoding=DEFAULT_FILE_ENCODING
        ) as f:
            last_index = 0
            for line in f:
                episodes_feature = EpisodesFeatures.model_validate_json(line)
                _episodes_features[episodes_feature.episode_index] = episodes_feature
                # If we skipped an index, check if the parquet file exists and if so recreate the episode
                if episodes_feature.episode_index != last_index + 1:
                    missing_indexes = [
                        i
                        for i in range(last_index + 1, episodes_feature.episode_index)
                        if i not in _episodes_features
                    ]
                    if missing_indexes:
                        incorrect_episodes = True
                        logger.warning(
                            f"Missing episodes in episodes.jsonl: {missing_indexes}."
                        )

        # Read all the .parquet files in the data folder to see if they are missing in the episodes.jsonl file
        dataset_path = os.path.dirname(meta_folder_path)
        data_folder_path = os.path.join(dataset_path, "data", "chunk-000")
        all_parquet_files = list(
            filter(
                lambda x: x.endswith(".parquet"),
                os.listdir(data_folder_path),
            )
        )
        all_parquet_files.sort()  # Sort to ensure episode order
        # Check if all parquet files are in the episodes.jsonl file
        for parquet_file in all_parquet_files:
            episode_index = int(parquet_file.split("_")[1].split(".")[0])
            if episode_index not in _episodes_features.keys():
                logger.info(
                    f"Found missing episode: {episode_index} {parquet_file} in episodes.jsonl."
                )
                incorrect_episodes = True

        # If there are inconsistencies, we need to recompute the episodes from the parquet files
        if incorrect_episodes:
            # Recompute the episodes from the parquet files
            logger.warning("Recomputing the episodes from the parquet files.")
            episodes_model = cls.recompute_from_parquets(
                dataset_path=Path(dataset_path), format=format
            )
            logger.info(
                f"Recomputed {len(episodes_model.episodes)} episodes from the parquet files. Replacing the episodes.jsonl file."
            )
            episodes_model.to_jsonl(
                meta_folder_path=meta_folder_path, save_mode="overwrite"
            )
            _episodes_features = {
                episode.episode_index: episode for episode in episodes_model.episodes
            }
        else:
            # Sort the _episodes_features by increasing episode_index
            _episodes_features = dict(
                sorted(_episodes_features.items(), key=lambda x: x[0])
            )
            episodes_model = EpisodesModel(
                episodes=list(_episodes_features.values()),
            )

        # Do it after model init, otherwise pydantic ignores the value of _original_nb_total_episodes
        episodes_model._original_nb_total_episodes = len(_episodes_features.keys())
        episodes_model._episodes_features = _episodes_features
        return episodes_model

    @classmethod
    def recompute_from_parquets(
        cls,
        dataset_path: Path,
        format: Literal["lerobot_v2", "lerobot_v2.1"] = "lerobot_v2.1",
    ) -> "EpisodesModel":
        """
        Recompute the episodes model from the parquet files in the data folder path.
        This is useful if the episodes.jsonl file is corrupted or missing.
        """
        data_folder_path = dataset_path / "data" / "chunk-000"
        episodes = []
        episode_index = 0
        for parquet_file in sorted(data_folder_path.glob("*.parquet")):
            episode = LeRobotEpisode.from_parquet(
                episode_data_path=str(parquet_file),
                format=format,
                dataset_path=str(dataset_path),
            )
            episodes.append(
                EpisodesFeatures(
                    episode_index=episode.episode_index,
                    tasks=[str(episode.steps[0].observation.language_instruction)],
                    length=len(episode.steps),
                )
            )
        # Create the EpisodesModel
        episodes_model = cls(episodes=episodes)
        # Set the _episodes_features dict
        episodes_model._episodes_features = {
            episode.episode_index: episode for episode in episodes_model.episodes
        }
        # Set the _original_nb_total_episodes
        episodes_model._original_nb_total_episodes = len(episodes_model.episodes)
        return episodes_model

    def save(
        self,
        meta_folder_path: str,
        save_mode: Literal["append", "overwrite"] = "overwrite",
    ) -> None:
        """
        Save the episodes to the meta folder path.
        """
        self.to_jsonl(meta_folder_path=meta_folder_path, save_mode=save_mode)

    def update_for_episode_removal(
        self, episode_to_delete_index: int, old_index_to_new_index: Dict[int, int]
    ) -> None:
        """
        Update the episodes model before removing an episode from the dataset.
        We just remove the line corresponding to the episode_index of the parquet file.
        """
        # Remove the episode from the list
        self.episodes = [
            episode
            for episode in self.episodes
            if episode.episode_index != episode_to_delete_index
        ]

        # Reindex the episodes
        if not old_index_to_new_index:
            # Reindex the episodes
            for episode in self.episodes:
                if episode.episode_index > episode_to_delete_index:
                    episode.episode_index -= 1

        else:
            # Use the old_index_to_new_index to update the episode index
            if len(old_index_to_new_index.keys()) > 0:
                current_max_index = max(old_index_to_new_index.keys()) + 1
            else:
                current_max_index = 0
            for episode in self.episodes:
                if episode.episode_index == episode_to_delete_index:
                    pass
                if episode.episode_index in old_index_to_new_index.keys():
                    # Update the episode index
                    episode.episode_index = old_index_to_new_index[
                        episode.episode_index
                    ]
                else:
                    # Update the episode index to the new one
                    episode.episode_index = current_max_index
                    current_max_index += 1
                    # Update mapping
                    old_index_to_new_index[episode.episode_index] = current_max_index

        # Recreate the _episodes_features dict
        self._episodes_features = {
            episode.episode_index: episode for episode in self.episodes
        }

    def merge_with(
        self, second_episodes_model: "EpisodesModel", meta_folder_to_save_to
    ) -> None:
        """
        Merge the episodes with another episodes model and save it to the new meta folder.
        """
        # Update the episode_index for the second episodes model
        for episode in second_episodes_model.episodes:
            episode.episode_index += len(self.episodes)

        # Merge the episodes
        self.episodes.extend(second_episodes_model.episodes)

        # Remove duplicates
        self.episodes = list(
            {episode.episode_index: episode for episode in self.episodes}.values()
        )
        self._episodes_features = {
            episode.episode_index: episode for episode in self.episodes
        }

        # Save the episodes.jsonl file
        self.to_jsonl(meta_folder_path=meta_folder_to_save_to, save_mode="overwrite")

    @classmethod
    def repair_parquets(self, parquets_path: str) -> bool:
        """
        This function will attempt to correct the parquet files in the dataset.
        It will:
        - Check if the parquets are correctly indexed
        - Rewrite all episode_index, frame index and index
        """
        # Check if the parquets exist
        if not os.path.exists(parquets_path):
            logger.warning(f"Parquet path {parquets_path} does not exist.")
            return False
        # Fetch all the parquet files
        parquet_files = os.listdir(parquets_path)
        parquet_files = [file for file in parquet_files if file.endswith(".parquet")]
        parquet_files.sort()
        # Check if the files are correctly indexed
        for i, file in enumerate(parquet_files):
            # Check if the file is correctly indexed
            if f"episode_{i:06d}.parquet" not in file:
                logger.warning(
                    f"Parquet file {file} is not correctly indexed. Expected episode_{i:06d}.parquet"
                )
                return False

        logger.info("Parquet files are correctly indexed. Will attempt to repair them.")
        cumulative_index = 0
        for i, file in enumerate(parquet_files):
            # Read the parquet file
            df = pd.read_parquet(os.path.join(parquets_path, file))
            # Check if the episode_index is correct
            df["episode_index"] = i
            df["frame_index"] = np.arange(len(df))
            df["index"] = np.arange(len(df)) + cumulative_index
            cumulative_index += len(df)
            # Save the parquet file
            df.to_parquet(os.path.join(parquets_path, file), index=False)
            logger.info(f"Parquet file {file} repaired.")

        return True

    def split(self, split_ratio: float):
        """
        Split the episodes model into two parts.
        """
        # Calculate the split index
        split_index = int(len(self.episodes) * split_ratio)
        # Split the episodes
        first_part = self.episodes[:split_index]
        second_part = self.episodes[split_index:]
        # Reindex second part
        for i, episode in enumerate(second_part):
            episode.episode_index = episode.episode_index - len(first_part)
        # Create new episodes models
        first_part_model = EpisodesModel(episodes=first_part)
        second_part_model = EpisodesModel(episodes=second_part)
        return first_part_model, second_part_model

    def shuffle(self, permutation: List[int] | np.ndarray) -> None:
        """
        Shuffle the episodes in the model.
        """
        # Edit the episode indexes based on the permutation
        for i, episode in enumerate(self.episodes):
            episode.episode_index = permutation[i]
        self.episodes.sort(key=lambda x: x.episode_index)
        # Recreate the _episodes_features dict
        self._episodes_features = {
            episode.episode_index: episode for episode in self.episodes
        }


class Stats(BaseModel):
    """
    Statistics for a given feature.
    """

    max: Optional[NdArrayAsList] = None
    min: Optional[NdArrayAsList] = None
    mean: Optional[NdArrayAsList] = None
    std: Optional[NdArrayAsList] = None

    # These values are used for rolling computation of mean and std
    sum: Optional[NdArrayAsList] = None
    square_sum: Optional[NdArrayAsList] = None
    count: int = 0

    @field_validator("count", mode="before")
    @classmethod
    def validate_count(cls, value) -> int:
        if isinstance(value, int):
            return value
        elif isinstance(value, list) and len(value) == 1:
            return value[0]

        raise ValueError(
            f"Count must be an int or a list of length 1, got {type(value)} with length {len(value)}"
        )

    # To be able to use np.array in pydantic, we need to use arbitrary_types_allowed = True
    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    def update(self, value: Optional[np.ndarray]) -> None:
        """
        Every recording step, update the stats with the new value.
        Note: These are not the final values for mean and std.
        You still neeed to call compute_from_rolling() to get the final values.
        """

        if value is None:
            return None

        # If square_sum is None, use the std to compute the square sum
        if self.square_sum is None and self.std is not None and self.mean is not None:
            self.square_sum = self.std**2 + self.mean**2

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
            # We need to copy the value to avoid modifying the original value
            self.sum = value.copy()
            self.square_sum = value.copy() ** 2
            self.count = 1
        else:
            self.sum = self.sum + value
            self.square_sum = self.square_sum + value**2
            self.count += 1

    def compute_from_rolling(self):
        """
        Compute the mean and std from the rolling sum and square sum.
        """
        if self.count == 0:
            logger.error("Count is 0. Cannot compute mean and std.")
            return

        if self.sum is None or self.square_sum is None:
            # We have already computed the mean and std
            return

        self.mean = self.sum / self.count
        compute_diff = self.square_sum / self.count - self.mean**2
        if (compute_diff < 0).any():
            if (compute_diff < -2).any():
                logger.warning(
                    f"Negative value in the square sum. Replacing the negative values of std with 0.\nsquare_sum={self.square_sum}\ncount={self.count}\nmean={self.mean**2}"
                )
            variance = self.square_sum / self.count - self.mean**2
            variance[variance <= 0] = 0
            self.std = np.sqrt(variance)
        else:
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

        # Compute the square sum if not available
        if self.square_sum is None and self.std is not None and self.mean is not None:
            self.square_sum = self.std**2 + self.mean**2

        image_norm_32 = image_value.astype(dtype=np.float32) / 255.0

        # Update the max and min
        if self.max is None:
            self.max = np.max(image_norm_32, axis=(0, 1)).reshape(3, 1, 1)
        else:
            image_max_pixel = np.max(image_norm_32, axis=(0, 1)).reshape(3, 1, 1)
            # maximum is the max in each channel
            self.max = np.maximum(self.max, image_max_pixel)

        if self.min is None:
            self.min = np.min(image_norm_32, axis=(0, 1)).reshape(3, 1, 1)
            # Reshape to have the same shape as the mean and std
            self.min = self.min.reshape(3, 1, 1)
        else:
            image_min_pixel = np.min(image_norm_32, axis=(0, 1)).reshape(3, 1, 1)
            # Set the min to the min in each channel
            self.min = np.minimum(self.min, image_min_pixel)
            # Reshape to have the same shape as the mean and std
            self.min = self.min.reshape(3, 1, 1) if self.min is not None else self.min

        # Update the rolling sum and square sum
        nb_pixels = image_norm_32.shape[0] * image_norm_32.shape[1]
        # Convert to int32 to avoid overflow when computing the square sum
        if self.sum is None or self.square_sum is None:
            self.sum = np.sum(image_norm_32, axis=(0, 1))
            self.square_sum = np.sum(image_norm_32**2, axis=(0, 1))
            self.count = nb_pixels
        else:
            self.sum = self.sum + np.sum(image_norm_32, axis=(0, 1))
            self.square_sum = self.square_sum + np.sum(image_norm_32**2, axis=(0, 1))
            self.count += nb_pixels

    def compute_from_rolling_images(self):
        """
        Compute the mean and std from the rolling sum and square sum for images.
        """

        if self.count == 0:
            logger.error("Count is 0. Cannot compute mean and std for images.")
            return

        if self.sum is None or self.square_sum is None:
            # We have already computed the mean and std
            return

        self.mean = self.sum / self.count
        self.std = np.sqrt(self.square_sum / self.count - self.mean**2)
        # We want .tolist() to yield [[[mean_r, mean_g, mean_b]]] and not [mean_r, mean_g, mean_b]
        # Reshape to have the same shape as the mean and std
        # This makes it easier to normalize the imags
        if self.mean.shape == (3,):
            self.mean = self.mean.reshape(3, 1, 1)
            self.std = self.std.reshape(3, 1, 1)
            # For the first episode the shape is (3,)
            # For the next ones the shape is (3,1,3)
            # We keep min and max of the first episode only
            self.min = self.min.reshape(3, 1, 1)
            self.max = self.max.reshape(3, 1, 1)


class StatsModel(BaseModel):
    """
    Data model utils to create stats.json file.

    Stats for  observation_state and actions are dim the number of joints
    Stats for images are dim 1, 3, 1
    The other stats are dim 1
    """

    observation_state: Stats = Field(
        default_factory=Stats,
        serialization_alias="observation.state",
        validation_alias=AliasChoices("observation.state", "observation_state"),
    )
    action: Stats = Field(default_factory=Stats)
    timestamp: Stats = Field(default_factory=Stats)
    frame_index: Stats = Field(default_factory=Stats)
    episode_index: Stats = Field(default_factory=Stats)
    index: Stats = Field(default_factory=Stats)
    task_index: Stats = Field(default_factory=Stats)

    # key is like: observation.images.main
    # value is Stats of the object: average pixel value, std, min, max ; and shape (height, width, channel)
    observation_images: Dict[str, Stats] = Field(
        default_factory=dict,
        serialization_alias="observation.images",
        validation_alias=AliasChoices(
            "observation.images",
            "observation.image",
            "observation_images",
            "observation_image",
        ),
    )

    @classmethod
    def from_json(cls, meta_folder_path: str) -> "StatsModel":
        """
        Read the stats.json file in the meta folder path.
        If the file does not exist, return an empty StatsModel.
        """
        if (
            not os.path.exists(f"{meta_folder_path}/stats.json")
            or os.stat(f"{meta_folder_path}/stats.json").st_size == 0
        ):
            return cls()

        with open(
            f"{meta_folder_path}/stats.json", "r", encoding=DEFAULT_FILE_ENCODING
        ) as f:
            stats_dict: Dict[str, Stats] = json.load(f)

        # Create a temporary dictionary for observation_images
        observation_images = {}
        # We need to create a list of keys in order not to modify
        # the dictionary while iterating over it
        for key in list(stats_dict.keys()):
            if "image" in key:
                observation_images[key] = stats_dict.pop(key)

        # Pass observation_images into the model constructor
        return cls(**stats_dict, observation_images=observation_images)

    def to_json(self, meta_folder_path: str) -> None:
        """
        Write the stats.json file in the meta folder path.
        """
        model_dict = self.model_dump(by_alias=True)

        # We flatten the fields in the dict observations images
        for key, value in model_dict["observation.images"].items():
            model_dict[key] = value
        model_dict.pop("observation.images")

        with open(
            f"{meta_folder_path}/stats.json", "w", encoding=DEFAULT_FILE_ENCODING
        ) as f:
            # Write the pydantic Basemodel as a str
            f.write(json.dumps(model_dict, indent=4))

    def update(
        self,
        step: Step,
        episode_index: int,
        current_step_index: int,
    ) -> None:
        """
        Updates the stats with the given step.
        """

        self.action.update(
            step.observation.joints_position
        )  # Because action lags behind by one step, we approximate it with the current observation
        self.observation_state.update(step.observation.joints_position)
        self.timestamp.update(np.array([step.observation.timestamp]))
        self.index.update(np.array([self.index.count]))
        self.episode_index.update(np.array([episode_index]))
        self.frame_index.update(np.array([current_step_index]))

        # TODO: Implement multiple language instructions
        # This should be the index of the instruction as it's in tasks.jsonl (TasksModel)
        self.task_index.update(np.array([0]))

        main_image = step.observation.main_image
        if main_image is not None:
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
            ].update_image(image)

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
                    try:
                        if isinstance(value, Stats):
                            value.compute_from_rolling_images()
                    except ValueError as e:
                        logger.error(f"Error computing mean and std for {key}: {e}")

        self.to_json(meta_folder_path)

    def _update_for_episode_removal_mean_std_count(
        self, df_episode_to_delete: pd.DataFrame
    ) -> None:
        """
        Update the stats before removing an episode from the dataset.
        We do not compute the new min and max.
        We prefer to do it after the episode removal to directly access new indexes and episodes indexes.
        This only updates mean, std, sum, square_sum, and count.
        """
        if df_episode_to_delete.empty:
            return

        # Load the parquet file
        nb_steps_deleted_episode = len(df_episode_to_delete)

        # Remove [nan, nan, ...] arrays from actions and observation.state columns
        df_episode_to_delete = df_episode_to_delete[
            df_episode_to_delete["observation.state"].apply(
                lambda x: not np.all(np.isnan(x))
            )
        ]
        df_episode_to_delete = df_episode_to_delete[
            df_episode_to_delete["action"].apply(lambda x: not np.all(np.isnan(x)))
        ]

        df_sums_squaresums = pd.concat(
            [
                df_episode_to_delete.sum(skipna=True),
                (df_episode_to_delete**2).sum(skipna=True),
            ],
            axis=1,
        )
        df_sums_squaresums = df_sums_squaresums.rename(
            columns={0: "sums", 1: "squaresums"}
        )

        # Update stats for each field in the StatsModel
        for field_name, field in StatsModel.model_fields.items():
            # TODO task_index is not updated since we do not support multiple tasks
            # observation_images has a special treatment
            if field_name in ["task_index", "observation_images"]:
                continue
            field_value = getattr(self, field_name)
            # The key in StatsModel is observation_state
            field_name = (
                "observation.state" if field_name == "observation_state" else field_name
            )
            # Update statistics
            if field_name in df_episode_to_delete.columns:
                # Subtract sums
                logger.debug(f"Field {field_name} sum before: {field_value.sum}")
                field_value.sum = (
                    field_value.sum - df_sums_squaresums["sums"][field_name]
                )
                field_value.square_sum = (
                    field_value.square_sum
                    - df_sums_squaresums["squaresums"][field_name]
                )
                logger.debug(f"Field {field_name} sum after: {field_value.sum}")

                # Update count
                field_value.count = field_value.count - nb_steps_deleted_episode

                # Recalculate mean and standard deviation
                if field_value.count > 0:
                    field_value.mean = field_value.sum / field_value.count
                    variance = (field_value.square_sum / field_value.count) - np.square(
                        field_value.mean
                    )
                    if (isinstance(variance, np.ndarray) and np.all(variance >= 0)) or (
                        isinstance(variance, float) and variance >= 0
                    ):
                        field_value.std = np.sqrt(variance)
                    else:
                        logger.error(
                            f"Field {field_name} variance is negative: {variance}"
                            + f" for episode {df_episode_to_delete['episode_index'].iloc[0]}"
                        )
                else:
                    logger.error(
                        f"Field {field_name} count is 0. Cannot compute mean and std for episode {df_episode_to_delete['episode_index'].iloc[0]}"
                    )

    def get_total_frames(self, meta_folder_path: str) -> int:
        """
        Return the total number of frames in the dataset
        """
        info_path = os.path.join(meta_folder_path, "info.json")
        with open(info_path, "r", encoding=DEFAULT_FILE_ENCODING) as f:
            info_dict = json.load(f)
        return info_dict.get("total_frames", 0)

    def get_images_shapes(self, meta_folder_path: str, camera_key: str) -> List[int]:
        """
        Return the tuple (height, width, channel) of the images for a given camera key
        """
        info_path = os.path.join(meta_folder_path, "info.json")
        with open(info_path, "r", encoding=DEFAULT_FILE_ENCODING) as f:
            info_dict = json.load(f)
        return info_dict["features"][camera_key]["shape"]

    def _compute_count_sum_square_sum_item_from_mean_std(
        self, stats_item: "Stats", stats_key: str, meta_folder_path: str
    ):
        """Helper function to compute sum and square_sum from mean, std, and count
        meta_folder_path is used to compute the count from info.json
        This is the number of frames or the number of frames times the dimension of images for videos
        """
        if stats_item.mean is None or stats_item.std is None:
            raise ValueError(f"Mean and std are not computed for {stats_item}")

        if (
            stats_item.sum is None
            or stats_item.square_sum is None
            or stats_item.count == 0
        ):
            is_video = "image" in stats_key or "video" in stats_key
            # Get the number of frames from info.json
            count = (
                self.get_total_frames(meta_folder_path)
                * self.get_images_shapes(
                    meta_folder_path=meta_folder_path, camera_key=stats_key
                )[0]
                * self.get_images_shapes(
                    meta_folder_path=meta_folder_path, camera_key=stats_key
                )[1]
                if is_video
                else self.get_total_frames(meta_folder_path),
            )[0]
            if is_video:
                logger.debug(f"Mean shape for {stats_key}: {stats_item.mean.shape}")
                logger.debug(f"Count for {stats_key}: {count}")

            mean = stats_item.mean
            std = stats_item.std

            stats_item.count = count
            logger.debug(f"Count for {stats_key}: {count}")
            sum_array = mean * count
            square_sum_array = (std**2 + mean**2) * count

            stats_item.sum = sum_array
            stats_item.square_sum = square_sum_array
            if is_video:
                logger.debug(
                    f"Mean shape for {stats_key} after: {stats_item.mean.shape}"
                )

    def compute_count_square_sum_framecount_from_mean_std(self, meta_folder_path: str):
        """
        Compute the sum and square sum from the mean and std.
        This is useful when we want to update the stats after repairing a dataset.
        """
        # Process all stats fields
        for stats_key, stats_item in self.__dict__.items():
            if stats_key != "observation_images":
                # Process regular stats items
                try:
                    self._compute_count_sum_square_sum_item_from_mean_std(
                        stats_item=stats_item,
                        stats_key=stats_key,
                        meta_folder_path=meta_folder_path,
                    )
                except ValueError as e:
                    raise ValueError(f"{e} for {stats_key}")
            else:
                # Process observation_images items
                for camera_key, video_stats_item in stats_item.items():
                    try:
                        self._compute_count_sum_square_sum_item_from_mean_std(
                            stats_item=video_stats_item,
                            stats_key=camera_key,
                            meta_folder_path=meta_folder_path,
                        )
                    except ValueError as e:
                        raise ValueError(f"{e} for {camera_key}")

    def _update_for_episode_removal_images_stats(
        self,
        folder_videos_path: str,
        episode_to_delete_index: int,
        meta_folder_path: str,
    ) -> None:
        """
        Update the stats for images.

        For every camera, we need to delete the episode_{episode_index:06d}.mp4 file.

        We update the sum, square_sum, and count for each camera.

        We do not update the min and the max here. It is always 0 and 1 by experience
        """

        self.compute_count_square_sum_framecount_from_mean_std(
            meta_folder_path=meta_folder_path
        )

        cameras_folders = os.listdir(folder_videos_path)
        # Only keep directories
        cameras_folders = [
            camera_name
            for camera_name in cameras_folders
            if os.path.isdir(os.path.join(folder_videos_path, camera_name))
        ]
        for camera_name in cameras_folders:
            if "image" not in camera_name:
                continue
            # Create the path of the video episode_{episode_index:06d}.mp4
            video_path = os.path.join(
                folder_videos_path,
                camera_name,  # eg: observation.images.main
                f"episode_{episode_to_delete_index:06d}.mp4",
            )
            sum_array, square_sum_array, nb_pixel = (
                compute_sum_squaresum_framecount_from_video(video_path)
            )

            sum_array = sum_array.astype(np.float32)
            square_sum_array = square_sum_array.astype(np.float32)

            # Update the stats_model for sum square_sum and count
            self.observation_images[camera_name].sum = (
                self.observation_images[camera_name].sum - sum_array
            )
            self.observation_images[camera_name].square_sum = (
                self.observation_images[camera_name].square_sum - square_sum_array
            )
            self.observation_images[camera_name].count = (
                self.observation_images[camera_name].count - nb_pixel
            )
            # Update the stats_model for mean and std
            if (
                self.observation_images[camera_name].sum is not None
                and self.observation_images[camera_name].count
            ):
                mean_val = (
                    np.array(self.observation_images[camera_name].sum)
                    / self.observation_images[camera_name].count
                )
                self.observation_images[camera_name].mean = mean_val
                self.observation_images[camera_name].square_sum = (
                    self.observation_images[camera_name].square_sum
                    - np.square(mean_val)
                )

    def _update_for_episode_removal_min_max(
        self,
        data_folder_path: str,
        meta_folder_path: str,
        episode_to_delete_index: int,
    ) -> None:
        """
        Update the min and max in stats after removing an episode from the dataset.
        Be sure to call this function after reindexing the data.
        """
        self.compute_count_square_sum_framecount_from_mean_std(
            meta_folder_path=meta_folder_path
        )

        # Load all the other parquet files in one dataFrame
        li_data_folder_filenames = [
            file
            for file in os.listdir(data_folder_path)
            if file.endswith(".parquet")
            and file != f"episode_{episode_to_delete_index:06d}.parquet"
        ]
        if len(li_data_folder_filenames) == 0:
            return

        # Load all episodes
        all_episodes = [
            pd.read_parquet(str(os.path.join(data_folder_path, file)))
            for file in li_data_folder_filenames
        ]

        # Combine all episodes
        all_episodes_df = pd.concat(all_episodes)

        for field_name, field in StatsModel.model_fields.items():
            # TODO task_index is not updated since we do not support multiple tasks
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
            if field_name in all_episodes_df.keys():
                (field_value.min, field_value.max) = get_field_min_max(
                    all_episodes_df, field_name
                )
                # The value should be a numpy ndarray, even if it's a int or float
                if not isinstance(field_value.min, np.ndarray):
                    field_value.min = np.array([field_value.min])
                if not isinstance(field_value.max, np.ndarray):
                    field_value.max = np.array([field_value.max])

    def update_for_episode_removal(self, data_folder_path: str) -> None:
        # TODO: Handle everything in one function
        pass

    def update_for_video_key_removal(
        self,
        video_keys_to_delete: List[str],
        meta_folder_path: str,
        stats_file_name: str = "episodes_stats.jsonl",
    ) -> None:
        """
        Update the stats for given video keys to delete.
        No need to recompute as we are just removing a column from the stats.jsonl file.
        """
        # For each line of the stats.jsonl file, we need to delete the columns corresponding to the video keys to delete
        with (
            open(
                f"{meta_folder_path}/{stats_file_name}",
                "r",
                encoding=DEFAULT_FILE_ENCODING,
            ) as f,
            tempfile.NamedTemporaryFile(
                "w", delete=False, encoding=DEFAULT_FILE_ENCODING
            ) as temp,
        ):
            # Process each line
            for line in f:
                # Parse the JSON object from the line
                stats_dict = json.loads(line)
                if "stats" in stats_dict:
                    for video_key in video_keys_to_delete:
                        stats_dict["stats"].pop(video_key, None)
                else:
                    raise ValueError(
                        f"stats_dict does not contain a stats key: {stats_dict}"
                    )
                temp.write(json.dumps(stats_dict) + "\n")

        # Replace the original file with the temporary file
        shutil.move(temp.name, f"{meta_folder_path}/{stats_file_name}")

        logger.debug(
            f"Stats.jsonl file updated for video keys to delete: {video_keys_to_delete}"
        )


class EpisodesStatsFeatures(BaseModel):
    """
    Features for each line of the episodes_stats.jsonl file.
    """

    episode_index: int = 0
    stats: StatsModel = Field(default_factory=StatsModel)

    def to_json(self) -> str:
        """
        Save the features as a json string.
        """
        # Use the aliases in StatsModel
        model_dict: dict[str, dict] = self.stats.model_dump(
            by_alias=True, warnings=False
        )

        for key, value in model_dict["observation.images"].items():
            model_dict[key] = value
        model_dict.pop("observation.images")

        # Convert count to a list and remove sum and square_sum for compatibility
        for key, value in model_dict.items():
            if isinstance(value["count"], int):
                value["count"] = [value["count"]]
            if "sum" in value.keys():
                value.pop("sum")
            if "square_sum" in value.keys():
                value.pop("square_sum")

        # Add the episode index
        result_dict = {"episode_index": self.episode_index, "stats": model_dict}

        # Convert to JSON string
        return json.dumps(result_dict)


class EpisodesStatsModel(BaseModel):
    """
    Creates the structure of the episodes_stats.jsonl file.
    """

    episodes_stats: List[EpisodesStatsFeatures] = Field(default_factory=list)

    def update(self, step: Step, episode_index: int, current_step_index: int) -> None:
        """
        Updates the episodes_stats with the given step.
        """
        # Check if the episode index already exists
        if (
            self.episodes_stats != []
            and self.episodes_stats[-1].episode_index == episode_index
        ):
            # Update the stats for the last episode
            self.episodes_stats[-1].stats.update(
                step=step,
                episode_index=episode_index,
                current_step_index=current_step_index,
            )
            return

        # If the episode index does not exist, create a new entry
        new_episode_stats = EpisodesStatsFeatures(
            episode_index=episode_index,
            stats=StatsModel(),
        )
        new_episode_stats.stats.update(
            step=step,
            episode_index=episode_index,
            current_step_index=current_step_index,
        )
        self.episodes_stats.append(new_episode_stats)

    def to_jsonl(self, meta_folder_path: str) -> None:
        """
        Write the episodes_stats.jsonl file in the meta folder path.
        """
        with open(
            f"{meta_folder_path}/episodes_stats.jsonl",
            "w",
            encoding=DEFAULT_FILE_ENCODING,
        ) as f:
            for episode_stats in self.episodes_stats:
                f.write(episode_stats.to_json() + "\n")

    @classmethod
    def from_jsonl(cls, meta_folder_path: str) -> "EpisodesStatsModel":
        """
        Read the episodes_stats.jsonl file in the meta folder path.
        If the file does not exist, return an empty EpisodeStatsModel.
        """
        if (
            not os.path.exists(f"{meta_folder_path}/episodes_stats.jsonl")
            or os.stat(f"{meta_folder_path}/episodes_stats.jsonl").st_size == 0
        ):
            return EpisodesStatsModel()

        with open(
            f"{meta_folder_path}/episodes_stats.jsonl",
            "r",
            encoding=DEFAULT_FILE_ENCODING,
        ) as f:
            _episodes_stats_dict: dict[int, EpisodesStatsFeatures] = {}
            for line in f:
                parsed_line: dict = json.loads(line)

                episodes_stats_feature = EpisodesStatsFeatures.model_validate(
                    parsed_line
                )
                # We need to parse the observation_images properly when loading the jsonl file
                observation_images = {}

                for key in list(parsed_line["stats"].keys()):
                    if "image" in key:
                        observation_images[key] = parsed_line["stats"].pop(key)

                episodes_stats_feature.stats.observation_images = observation_images

                _episodes_stats_dict[episodes_stats_feature.episode_index] = (
                    episodes_stats_feature
                )

        _episodes_stats_dict = dict(
            sorted(_episodes_stats_dict.items(), key=lambda x: x[0])
        )

        episodes_stats_model = EpisodesStatsModel(
            episodes_stats=list(_episodes_stats_dict.values())
        )

        return episodes_stats_model

    def save(self, meta_folder_path: str) -> None:
        """
        Save the episodes_stats to the meta folder path.
        Also computes the final mean and std for the Stats objects.
        """
        for episode_stats in self.episodes_stats:
            for field_key, field_value in episode_stats.stats.__dict__.items():
                # if field is a Stats object, call .compute_from_rolling() to get the final mean and std
                if isinstance(field_value, Stats):
                    field_value.compute_from_rolling()

                # Special case for images
                if isinstance(field_value, dict) and field_key == "observation_images":
                    for key, value in field_value.items():
                        try:
                            if isinstance(value, Stats):
                                value.compute_from_rolling_images()
                        except ValueError as e:
                            logger.error(f"Error computing mean and std for {key}: {e}")

        self.to_jsonl(meta_folder_path)

    def update_for_episode_removal(
        self, episode_to_delete_index: int, old_index_to_new_index: Dict[int, int]
    ) -> None:
        """
        Update the episodes_stats model before removing an episode from the dataset.
        We remove the line corresponding to the episode index.

        We pass episode_to_delete_index = -1 when no episode is deleted (shuffling)
        """
        self.episodes_stats = [
            episode_stats
            for episode_stats in self.episodes_stats
            if episode_stats.episode_index != episode_to_delete_index
        ]

        # Reindex the episodes
        if not old_index_to_new_index:
            for episode_stats in self.episodes_stats:
                if episode_stats.episode_index > episode_to_delete_index:
                    episode_stats.episode_index -= 1

        else:
            if len(old_index_to_new_index.keys()) > 0:
                current_max_index = max(old_index_to_new_index.keys()) + 1
            else:
                current_max_index = 0
            for episode_stats in self.episodes_stats:
                if episode_stats.episode_index == episode_to_delete_index:
                    pass
                if episode_stats.episode_index in old_index_to_new_index.keys():
                    episode_stats.episode_index = old_index_to_new_index[
                        episode_stats.episode_index
                    ]
                else:
                    episode_stats.episode_index = current_max_index
                    current_max_index += 1
                    old_index_to_new_index[episode_stats.episode_index] = (
                        current_max_index
                    )

    def merge_with(
        self, second_stats_model: "EpisodesStatsModel", meta_folder_path: str
    ) -> None:
        """
        Merges an existing episodes stats model with another one.
        This is intended to be used when merging files for two so-100 together.
        """
        number_of_episodes_in_first_model = len(self.episodes_stats)

        # Update the episode_index for the second stats model
        for episode_stats in second_stats_model.episodes_stats:
            episode_stats.episode_index += number_of_episodes_in_first_model

        # Add the second stats model to the first one
        self.episodes_stats += second_stats_model.episodes_stats

        # Sort the episodes by episode index
        self.episodes_stats = sorted(self.episodes_stats, key=lambda x: x.episode_index)

        # Save the merged model
        self.save(meta_folder_path)

    def split(self, split_ratio: float):
        """
        Splits the episodes stats model into two parts.
        The first part contains the first split_ratio * len(episodes_stats) episodes.
        The second part contains the rest of the episodes.
        """
        split_index = int(len(self.episodes_stats) * split_ratio)
        first_part = EpisodesStatsModel(
            episodes_stats=self.episodes_stats[:split_index]
        )
        second_part = EpisodesStatsModel(
            episodes_stats=self.episodes_stats[split_index:]
        )

        # Reindex the second part
        for episode_stats in second_part.episodes_stats:
            episode_stats.episode_index -= len(first_part.episodes_stats)

        return first_part, second_part

    def shuffle(self, permutation: List[int] | np.ndarray) -> None:
        """
        Shuffles the episodes stats model according to the given permutation.
        The permutation is a list of indices that specifies the new order of the episodes.
        """
        if len(permutation) != len(self.episodes_stats):
            raise ValueError("Permutation length must match the number of episodes.")

        self.episodes_stats = [self.episodes_stats[i] for i in permutation]
        # Update episode_index
        for new_index, episode_stats in enumerate(self.episodes_stats):
            episode_stats.episode_index = new_index


class FeatureDetails(BaseModel):
    dtype: Literal["video", "int64", "float32", "str", "bool"]
    shape: List[int]
    names: Optional[List[str]]


class VideoInfo(BaseModel):
    """
    Information about the video
    """

    video_fps: int = Field(
        default=10,
        serialization_alias="video.fps",
        validation_alias=AliasChoices("video.fps", "video_fps"),
    )
    video_codec: VideoCodecs = Field(
        serialization_alias="video.codec",
        validation_alias=AliasChoices("video.codec", "video_codec"),
    )

    video_pix_fmt: str = Field(
        default="yuv420p",
        serialization_alias="video.pix_fmt",
        validation_alias=AliasChoices("video.pix_fmt", "video_pix_fmt"),
    )
    video_is_depth_map: bool = Field(
        default=False,
        serialization_alias="video.is_depth_map",
        validation_alias=AliasChoices("video.is_depth_map", "video_is_depth_map"),
    )
    has_audio: bool = False


class VideoFeatureDetails(FeatureDetails):
    dtype: Literal["video"] = "video"
    info: VideoInfo = Field(validation_alias=AliasChoices("video_info", "info"))


class InfoFeatures(BaseModel):
    action: FeatureDetails
    observation_state: FeatureDetails = Field(
        serialization_alias="observation.state",
        validation_alias=AliasChoices("observation.state", "observation_state"),
    )

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
    observation_images: Dict[str, VideoFeatureDetails] = Field(
        default_factory=dict,
        serialization_alias="observation.images",
        validation_alias=AliasChoices(
            "observation.image",
            "observation.images",
            "observation_image",
            "observation_images",
        ),
    )

    # Optional fields (RL)
    next_done: Optional[FeatureDetails] = Field(
        default=None,
        serialization_alias="next.done",
        validation_alias=AliasChoices("next.done", "next_done"),
    )
    next_success: Optional[FeatureDetails] = Field(
        default=None,
        serialization_alias="next.success",
        validation_alias=AliasChoices("next.success", "next_success"),
    )
    next_reward: Optional[FeatureDetails] = Field(
        default=None,
        serialization_alias="next.reward",
        validation_alias=AliasChoices("next.reward", "next_reward"),
    )
    observation_environment_state: Optional[FeatureDetails] = Field(
        default=None,
        serialization_alias="observation.environment_state",
        validation_alias=AliasChoices(
            "observation.environment.state",
            "observation_environment_state",
            "observation.environment_state",
            "observation_environment.state",
        ),
    )

    def to_dict(self) -> dict:
        """
        Convert the InfoFeatures to a dictionary.
        This transforms observation_images and observation_state to the correct format.
        """
        model_dict = self.model_dump(by_alias=True)

        if self.observation_images is not None:
            for key, value in self.observation_images.items():
                model_dict[key] = value.model_dump(by_alias=True)

        model_dict.pop("observation.images")

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
    observation_state: FeatureDetails = Field(
        serialization_alias="observation.state",
        validation_alias=AliasChoices("observation.state", "observation_state"),
    )

    def merge_base_robot_info(
        self, base_robot_info: "BaseRobotInfo"
    ) -> "BaseRobotInfo":
        """
        Merges an existing base robot info with another one.
        This is intended to be used when merging files for two so-100 together.
        """
        if (
            self.action.names is None
            or self.observation_state.names is None
            or base_robot_info.action.names is None
            or base_robot_info.observation_state.names is None
        ):
            raise ValueError(
                "The names field in the action and observation_state must be set."
            )
        self.robot_type += f", {base_robot_info.robot_type}"

        self.action.shape[0] += base_robot_info.action.shape[0]
        self.action.names += [
            name + "_secondary" for name in base_robot_info.action.names
        ]

        self.observation_state.shape[0] += base_robot_info.observation_state.shape[0]
        self.observation_state.names = self.observation_state.names + [
            name + "_secondary" for name in base_robot_info.observation_state.names
        ]

        return self


class InfoModel(BaseModel):
    """
    Data model util to create meta/info.jsonl file.
    """

    robot_type: str

    codebase_version: str = "v2.1"
    total_episodes: int = 0
    total_frames: int = 0
    total_tasks: int = 1  # By default, there is 1 task: "None"
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
    def from_robots(cls, robots: List[BaseRobot], **data) -> "InfoModel":
        """
        From a robot configuration, create the appropriate InfoModel.
        This is because it depends on the number of joints etc.
        """
        robot_info = robots[0].get_info_for_dataset()
        if len(robots) > 1:
            for robot in robots[1:]:
                new_info = robot.get_info_for_dataset()
                robot_info = robot_info.merge_base_robot_info(new_info)

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
        model_dict = self.model_dump(by_alias=True)
        model_dict["features"] = self.features.to_dict()
        return model_dict

    @classmethod
    def from_json(
        cls,
        meta_folder_path: str,
        fps: Optional[int] = None,
        codec: Optional[VideoCodecs] = None,
        robots: Optional[List[BaseRobot]] = None,
        target_size: Optional[tuple[int, int]] = None,
        all_camera_key_names: Optional[List[str]] = None,
        format: Literal["lerobot_v2", "lerobot_v2.1"] = "lerobot_v2.1",
    ) -> "InfoModel":
        """
        Read the info.json file in the meta folder path.
        If the file does not exist, try to create the InfoModel from the provided data.

        raise ValueError if the file does not exist and no data is provided to create the InfoModel.
        """
        # Check if the file existes.
        if (
            not os.path.exists(f"{meta_folder_path}/info.json")
            or os.stat(f"{meta_folder_path}/info.json").st_size == 0
        ):
            if robots is None:
                raise ValueError(
                    "No info.json file found and no robot provided to create the InfoModel"
                )
            if codec is None:
                raise ValueError("No codec provided to create the InfoModel")
            if fps is None:
                raise ValueError("No fps provided to create the InfoModel")
            if target_size is None:
                raise ValueError("No target_size provided to create the InfoModel")
            if all_camera_key_names is None:
                raise ValueError(
                    "No secondary_camera_ids provided to create the InfoModel"
                )

            info_model = cls.from_robots(robots)
            video_shape = [target_size[1], target_size[0], 3]
            video_info = VideoInfo(video_codec=codec, video_fps=fps)

            info_model.fps = fps

            # Add cameras
            for camera_key_name in all_camera_key_names:
                info_model.features.observation_images[camera_key_name] = (
                    VideoFeatureDetails(
                        shape=video_shape,
                        names=["height", "width", "channel"],
                        info=video_info,
                    )
                )

            info_model.codebase_version = "v2.1" if format == "lerobot_v2.1" else "v2.0"

            return info_model

        with open(
            f"{meta_folder_path}/info.json", "r", encoding=DEFAULT_FILE_ENCODING
        ) as f:
            info_model_dict = json.load(f)

        info_model_dict["features"]["observation_state"] = info_model_dict[
            "features"
        ].pop("observation.state")
        observation_images = {}
        keys_to_remove = []
        for key, value in info_model_dict["features"].items():
            if "observation.image" in key:
                observation_images[key] = value
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del info_model_dict["features"][key]
        info_model_dict["features"]["observation_images"] = observation_images
        infos = cls.model_validate(info_model_dict)

        # Read the number of .parquet files in the data folder. Get the parent directory
        dataset_path = os.path.dirname(meta_folder_path)
        data_folder_path = Path(dataset_path) / "data" / "chunk-000"
        if not data_folder_path.exists():
            return infos

        # Otherwise, count the number of .parquet files in the data folder
        all_episodes_df = list(data_folder_path.rglob("episode_*.parquet"))
        if len(all_episodes_df) != infos.total_episodes:
            logger.warning(
                f"Number of episodes in info.json ({infos.total_episodes}) does not match the number of episodes in the data folder ({len(all_episodes_df)}). Recomputing from parquets."
            )
            infos = cls.recompute_from_parquets(
                infos=infos, dataset_path=Path(dataset_path)
            )

        return infos

    @classmethod
    def recompute_from_parquets(
        cls, infos: "InfoModel", dataset_path: Path
    ) -> "InfoModel":
        data_folder_path = dataset_path / "data" / "chunk-000"
        all_episodes_df = list(data_folder_path.rglob("episode_*.parquet"))
        infos.total_episodes = len(all_episodes_df)
        # Recompute the number of total frames and videos
        total_frames = 0
        for episode_file in data_folder_path.glob("episode_*.parquet"):
            df = pd.read_parquet(episode_file)
            total_frames += len(df)
        infos.total_frames = total_frames

        # Recompute the number of total videos
        total_videos = 0
        video_path = dataset_path / "videos" / "chunk-000"
        for camera_name in video_path.iterdir():
            # Count the number of videos in the subfolder
            if "image" not in camera_name.name:
                continue
            total_videos += len(list(camera_name.glob("episode_*.mp4")))
        infos.total_videos = total_videos

        return infos

    @classmethod
    def from_multidataset(
        cls,
        robot_type: str,
        nb_cameras: int,
        fps: int,
        nb_motors: int,
        chunks_size: int,
        resize_video: Tuple[int, int],
        # li_datasets is a list of LeRobotDatasets
        # We dont type it as List[LeRobotDataset] to avoid importing LeRobotDataset
        li_datasets: list,
    ) -> "InfoModel":
        """Create an InfoModel from multiple datasets information."""

        # Create the action and observation_state FeatureDetails
        action_details = FeatureDetails(
            dtype="float32",
            shape=[nb_motors],
            names=[f"motor_{i + 1}" for i in range(nb_motors)],
        )

        observation_state_details = FeatureDetails(
            dtype="float32",
            shape=[nb_motors],
            names=[f"motor_{i + 1}" for i in range(nb_motors)],
        )

        # Create the observation_images dictionary
        observation_images = {}
        for i in range(nb_cameras):
            camera_key = (
                f"observation.images.{'main' if i == 0 else f'secondary_{i - 1}'}"
            )
            observation_images[camera_key] = VideoFeatureDetails(
                shape=[resize_video[1], resize_video[0], 3],
                names=["height", "width", "channel"],
                info=VideoInfo(
                    video_fps=fps,
                    video_codec="mp4v",
                    video_pix_fmt="yuv420p",
                    video_is_depth_map=False,
                    has_audio=False,
                ),
            )

        # Create InfoFeatures object
        features = InfoFeatures(
            action=action_details,
            observation_state=observation_state_details,
            observation_images=observation_images,
        )

        # Create the base InfoModel
        info_model = cls(
            robot_type=robot_type,
            codebase_version="v2.0",
            total_episodes=0,
            total_frames=0,
            total_tasks=1,
            total_videos=0,
            total_chunks=1,
            chunks_size=chunks_size,
            fps=fps,
            features=features,
        )

        # Update with dataset information if provided
        if li_datasets:
            # Calculate totals from all datasets
            for dataset in li_datasets:
                info = dataset.meta.info
                info_model.total_frames += info.get("total_frames", 0)
                info_model.total_videos += info.get("total_videos", 0)
                info_model.total_episodes += info.get("total_episodes", 0)

            info_model.splits = {"train": f"0:{info_model.total_episodes}"}
            info_model.total_chunks = info_model.total_episodes // chunks_size

            # Count unique tasks
            unique_tasks = set()
            for dataset in li_datasets:
                unique_tasks.update(dataset.meta.task_to_task_index.keys())

            info_model.total_tasks = len(unique_tasks)

        return info_model

    def to_json(self, meta_folder_path: str) -> None:
        """
        Write the info.json file in the meta folder path.
        """
        with open(
            f"{meta_folder_path}/info.json", "w", encoding=DEFAULT_FILE_ENCODING
        ) as f:
            f.write(json.dumps(self.to_dict(), indent=4))

    def update(self, episode: LeRobotEpisode) -> None:
        """
        Update the info given a new recorded Episode.
        """
        # Read the number of total episodes and videos based on the number of files
        # in the data folder

        nb_episodes = len(
            [
                file
                for file in os.listdir(episode.episodes_path)
                if file.endswith(".parquet")
            ]
        )

        self.total_episodes = nb_episodes
        self.total_frames += len(episode.steps)
        # Count the number of videos in every subfolder
        video_path = os.path.join(episode.dataset_path, "videos", "chunk-000")
        total_videos = 0
        for camera_name in os.listdir(video_path):
            # Count the number of videos in the subfolder
            if "image" not in camera_name:
                continue
            total_videos += len(
                [
                    file
                    for file in os.listdir(os.path.join(video_path, camera_name))
                    if file.endswith(".mp4")
                ]
            )

        self.total_videos = total_videos
        self.splits = {"train": f"0:{self.total_episodes}"}

        # Handle task index
        task_index = episode.metadata.get("task_index", 0)
        if task_index >= self.total_tasks:
            self.total_tasks = task_index + 1

        # TODO: Implement support for multiple chunks

    def save(
        self,
        meta_folder_path: str,
    ) -> None:
        """
        Save the info to the meta folder path.
        """
        self.to_json(meta_folder_path)

    def update_for_episode_removal(self, df_episode_to_delete: pd.DataFrame) -> None:
        """
        Update the info before removing an episode from the dataset.
        """
        if df_episode_to_delete.empty:
            return

        # Read from the data folder
        self.total_episodes -= 1
        self.total_frames -= len(df_episode_to_delete)
        self.total_videos -= len(self.features.observation_images.keys())
        self.splits = {"train": f"0:{self.total_episodes}"}
        # TODO(adle): Implement support for multi tasks in dataset
        # self.total_tasks -= ...

    def merge_with(
        self,
        second_info_model: "InfoModel",
        meta_folder_to_save_to: str,
        new_nb_tasks: int,
    ) -> None:
        """
        Will merge the info.json file with another one and save it to the new meta folder.
        """
        # Merge the info model
        self.total_episodes += second_info_model.total_episodes
        self.total_frames += second_info_model.total_frames
        self.total_videos += second_info_model.total_videos
        self.total_tasks = new_nb_tasks
        self.splits = {"train": f"0:{self.total_episodes}"}

        # Save the info.json file
        self.to_json(meta_folder_path=meta_folder_to_save_to)

    def update_for_video_removal(
        self, video_keys_to_delete: List[str], meta_folder_to_save_to: str
    ) -> None:
        """
        Update the info when removing a video from the dataset.

        full_video_keys are of the form "observation.images.{video_key}"

        Need to update:
        - total_frames
        - total_videos

        Need to delete:
        - features.{video_key}

        Will save the info.json file to the meta_folder_to_save_to folder.
        """

        # Get the current number of video keys before deleting
        nb_video_keys_before_deletion = len(self.features.observation_images.keys())

        # We keep the number of total frames constant

        self.total_videos -= self.total_episodes * len(video_keys_to_delete)

        print(
            f"self.features.observation_images: {self.features.observation_images.keys()}"
        )

        # Remove the video from the folder
        for video_key in video_keys_to_delete:
            # Check if it starts with observation.images.
            if video_key.startswith("observation.images."):
                del self.features.observation_images[video_key]
            else:
                del self.features.observation_images[f"observation.images.{video_key}"]

        # Save the info.json file
        self.to_json(meta_folder_path=meta_folder_to_save_to)
