### This file is a copy of our script to compute meta files for a dataset

import json
import os
import shutil
from copy import deepcopy
from math import ceil
from pathlib import Path
from typing import Optional, Tuple, Dict

import cv2
import einops
import numpy as np
import pandas as pd
import torch
import torchvision
import tqdm
import modal
from loguru import logger
from phosphobot.models import InfoModel
from torch.utils.data import Dataset as TorchDataset
import multiprocessing


from phosphobot.models.lerobot_dataset import FeatureDetails
from phosphobot.models.lerobot_dataset import LeRobotDataset

paligemma_detect = modal.Function.from_name("paligemma-detector", "detect_object")
act_volume = modal.Volume.from_name("act")

# Minimum number of bounding boxes to train an ACT model
MIN_NUMBER_OF_BBOXES = 10
# Maximum batch size to use for PaliGemma (can cause OOM otherwise)
MAX_BATCH_SIZE = 140


class ParquetEpisodesDataset(TorchDataset):
    """Custom Dataset for loading parquet files from a directory with video frame caching."""

    def __init__(self, dataset_path: Path):
        """
        Initialize the dataset by loading parquet files and pre-decoding video frames.

        Args:
            dataset_path (Path): Path to the folder containing data, videos, and meta subfolders.
        """
        logger.info(f"Loading Torch dataset from {dataset_path}")
        self.dataset_dir = dataset_path
        self.data_dir = self.dataset_dir / "data"
        self.videos_dir = self.dataset_dir / "videos"

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        if not self.videos_dir.exists():
            raise FileNotFoundError(f"Videos directory not found: {self.videos_dir}")

        self.file_paths = sorted(self.data_dir.rglob("*.parquet"))
        self.video_paths = sorted(self.videos_dir.rglob("*.mp4"))
        self.parquet_cache: dict[str, pd.DataFrame] = {}

        if not self.file_paths:
            raise ValueError(f"No parquet files found in {dataset_path}")
        if not self.video_paths:
            raise ValueError(f"No video files found in {dataset_path}")

        logger.info(
            f"Found {len(self.file_paths)} parquet files and {len(self.video_paths)} video files in {dataset_path}"
        )

        if len(self.video_paths) % len(self.file_paths) != 0:
            raise ValueError(
                f"Number of parquet files ({len(self.file_paths)}) does not match "
                f"number of video files ({len(self.video_paths)})"
            )

        # Reindex data for global indexing
        self.episode_nb_steps = []
        self.index_mapping: dict[int, dict] = {}
        self.steps_per_episode: dict[int, int] = {}
        global_idx = 0
        for file_path in self.file_paths:
            episode_idx = int(file_path.stem.split("_")[-1])
            df = self.read_parquet(str(file_path))
            nb_steps = len(df)
            self.episode_nb_steps.append(nb_steps)
            self.steps_per_episode[episode_idx] = nb_steps

            related_video_files = [
                video_path
                for video_path in self.video_paths
                if f"episode_{episode_idx:06d}" in video_path.name
            ]
            related_video_files_dict = {
                str(video_path).split("chunk-000/")[-1].split("/")[0]: video_path
                for video_path in related_video_files
            }

            for i in range(nb_steps):
                self.index_mapping[i + global_idx] = {
                    "file_path": file_path,
                    "episode_idx": episode_idx,
                    "row_idx": i,
                    "videos_paths": related_video_files_dict,
                }
            global_idx += nb_steps

        self.total_length = sum(self.episode_nb_steps)

        # Correctly set video keys (assuming subfolders in chunk-000 are video keys)
        videos_folders = os.path.join(self.videos_dir, "chunk-000")
        self.video_keys = os.listdir(videos_folders)  # e.g., ["camera1", "camera2"]

        # Pre-decode and cache all video frames
        self._initialize_video_cache()

    def _initialize_video_cache(self):
        """Decode all video frames and store them in a cache using parallel processing."""
        # Map episode_idx to video paths
        self.episode_to_videos_paths = {}
        for global_idx in self.index_mapping:
            episode_idx = self.index_mapping[global_idx]["episode_idx"]
            if episode_idx not in self.episode_to_videos_paths:
                self.episode_to_videos_paths[episode_idx] = self.index_mapping[
                    global_idx
                ]["videos_paths"]

        # Prepare arguments for parallel decoding
        args_list = [
            (episode_idx, file_path, self.episode_to_videos_paths[episode_idx])
            for episode_idx, file_path in enumerate(self.file_paths)
        ]

        # Decode episodes in parallel using 8 CPUs
        with multiprocessing.Pool(processes=8, maxtasksperchild=10) as pool:
            results = list(
                tqdm.tqdm(
                    pool.imap(self._decode_episode, args_list),
                    total=len(args_list),
                    desc="Decoding videos",
                )
            )

        # Populate cache
        self.cache = {
            episode_idx: decoded_frames for episode_idx, decoded_frames in results
        }

    def _decode_episode(
        self, args: Tuple[int, Path, Dict[str, Path]]
    ) -> Tuple[int, Dict[str, torch.Tensor]]:
        """Decode all frames for an episode given its timestamps and video paths."""
        episode_idx, parquet_file_path, videos_paths = args
        df = pd.read_parquet(parquet_file_path)
        timestamps = df["timestamp"].tolist()
        decoded_frames = {}
        for video_key, video_path in videos_paths.items():
            frames = decode_video_frames_torchvision(video_path, timestamps)
            # Store as uint8 to save memory
            frames = (frames * 255).to(torch.uint8)
            decoded_frames[video_key] = frames
        return episode_idx, decoded_frames

    def __len__(self) -> int:
        return self.total_length

    def read_parquet(self, file_path: str) -> pd.DataFrame:
        # Cache the parquet files to avoid reading them multiple times
        if file_path not in self.parquet_cache:
            self.parquet_cache[file_path] = pd.read_parquet(file_path, engine="pyarrow")
        return self.parquet_cache[file_path]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx >= self.total_length:
            raise IndexError("Index out of bounds")

        episode_idx = self.index_mapping[idx]["episode_idx"]
        row_idx = self.index_mapping[idx]["row_idx"]
        file_path = self.index_mapping[idx]["file_path"]

        # Read specific row from parquet
        df = self.read_parquet(str(file_path))
        row_data = df.iloc[row_idx]

        # Prepare sample dictionary
        sample = {}
        for col_name, value in row_data.items():
            if isinstance(value, (list, np.ndarray)):
                sample[col_name] = torch.tensor(value, dtype=torch.float32)
            elif isinstance(value, torch.Tensor):
                sample[col_name] = value
            elif isinstance(value, str):
                sample[col_name] = torch.tensor([float(x) for x in eval(value)])
            else:
                sample[col_name] = torch.tensor([value], dtype=torch.float32)

        # Retrieve cached frames
        for video_key in self.video_keys:
            frame = self.cache[episode_idx][video_key][row_idx]
            # Convert uint8 to float32 and normalize
            sample[video_key] = frame.float() / 255.0

        return sample

    def write_episodes(self, output_dir: str) -> None:
        # We want to write the episodes format
        # {"episode_index": 0, "length": 57}
        # {"episode_index": 1, "length": 88}
        # ...

        # For now, we resolve ot a temporary fix: use the first task from the meta/tasks.json file
        # But we would like to be able to handle multiple tasks
        # See the training/phospho_lerobot/scripts/multidataset.py save_episodes_jsonl() method
        task = None
        with open(os.path.join(self.dataset_dir, "meta", "tasks.jsonl"), "r") as file:
            for line in file:
                if line.strip():  # Skip empty lines
                    row = json.loads(line)
                    task = row["task"]
        if task is None:
            raise ValueError("No task found in the meta/tasks.json file")

        for episode_idx, nb_steps in self.steps_per_episode.items():
            episode = {
                "episode_index": episode_idx,
                "tasks": task,
                "length": nb_steps,
            }
            with open(output_dir, "a") as f:
                f.write(json.dumps(episode) + "\n")


def get_stats_einops_patterns(
    dataset: ParquetEpisodesDataset,
    dataloader: torch.utils.data.DataLoader,
) -> dict[str, str]:
    """These einops patterns will be used to aggregate batches and compute statistics.

    dataset_path is the path to the folder containing data, videos, meta subfolder

    Note: We assume images are in channel-first format.
    """

    # Grab one batch to inspect
    batch = next(iter(dataloader))
    # batch is now a dictionary like:
    # {
    #   'action': tensor(...),
    #   'observation.state': tensor(...),
    #   'timestamp': tensor(...),
    #    ...
    # }

    stats_patterns = {}

    # Load metadata
    features_dict = batch.keys()

    logger.info(f"Featured dict: {features_dict}")
    logger.info(f"Dataset video keys: {dataset.video_keys}")
    for key in features_dict:
        # Check if the batch actually has this key
        if key not in batch:
            logger.warning(f"Key '{key}' not found in batch. Skipping.")
            continue

        data = batch[key]
        logger.info(f"Processing key '{key}' with shape {data.shape}")

        # Sanity check that we don't have float64
        if data.dtype == torch.float64:
            raise TypeError(f"{key} has dtype float64, which is not expected.")

        # TODO: Implement proper images handling
        # If it's a camera key, do image checks
        if key in dataset.video_keys:
            # We expect a 4D tensor of shape [B, C, H, W]
            if data.ndim != 4:
                raise ValueError(
                    f"Camera data '{key}' is expected to have 4 dimensions, "
                    f"but got shape: {tuple(data.shape)}"
                )

            b, c, h, w = data.shape
            # Check channel-first assumption (C < H and C < W for typical image shapes)
            if not (c < h and c < w):
                raise ValueError(
                    f"Expect channel-first images for '{key}', but got shape {data.shape}"
                )

            # Check dtype and range
            if data.dtype != torch.float32:
                raise TypeError(
                    f"Camera data '{key}' must be float32, got {data.dtype}"
                )
            if data.max() > 1.0:
                raise ValueError(
                    f"Camera data '{key}' has values above 1.0 (max={data.max():.4f})"
                )
            if data.min() < 0.0:
                raise ValueError(
                    f"Camera data '{key}' has values below 0.0 (min={data.min():.4f})"
                )

            # Set einops pattern for images
            stats_patterns[key] = "b c h w -> c 1 1"

        # stats_patterns["observation.images"] = "b c h w -> c 1 1"

        # Non-camera data. Decide pattern based on dimensionality
        elif data.ndim == 2:
            # e.g. shape [batch_size, some_dim]
            stats_patterns[key] = "b c -> c"
        elif data.ndim == 1:
            # e.g. shape [batch_size]
            stats_patterns[key] = "b -> 1"
        else:
            logger.error(f"Unexpected shape for '{key}': {data.shape}")
            raise ValueError(f"{key} has an unexpected shape {data.shape}")

    return stats_patterns


def compute_stats(
    dataset_path: Path,
    batch_size: int = 128,
    num_workers: int = 2,
    max_num_samples: Optional[int] = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """Compute mean/std and min/max statistics of all data keys in a LeRobotDataset."""
    dataset = ParquetEpisodesDataset(dataset_path=dataset_path)

    if max_num_samples is None:
        max_num_samples = len(dataset)

    # Example DataLoader that returns dictionaries of tensors
    generator = torch.Generator()
    generator.manual_seed(1337)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        drop_last=False,
        generator=generator,
    )
    stats_patterns = get_stats_einops_patterns(dataset, dataloader)

    # mean and std will be computed incrementally while max and min will track the running value.
    mean, std, max, min = {}, {}, {}, {}
    for key in stats_patterns:
        mean[key] = torch.tensor(0.0).float()
        std[key] = torch.tensor(0.0).float()
        max[key] = torch.tensor(-float("inf")).float()
        min[key] = torch.tensor(float("inf")).float()

    # Note: Due to be refactored soon. The point of storing `first_batch` is to make sure we don't get
    # surprises when rerunning the sampler.
    first_batch: Optional[dict] = None
    running_item_count = 0  # for online mean computation

    logger.info("Starting to create seeded dataloader")

    error_raised = False
    for i, batch in tqdm.tqdm(
        enumerate(dataloader),
        total=ceil(max_num_samples / batch_size),
        desc="Compute mean, min, max",
    ):
        this_batch_size = len(batch["index"])
        running_item_count += this_batch_size
        if first_batch is None:
            first_batch = deepcopy(batch)
        for key, pattern in stats_patterns.items():
            if key not in batch.keys():
                if not error_raised:
                    logger.error(
                        f"[MEAN] Key '{key}' from stats_patterns not found in batch {i}/{ceil(max_num_samples) / batch_size}. Available keys: {batch.keys()}. Ignoring this key."
                    )
                continue
            batch[key] = batch[key].float()
            # Numerically stable update step for mean computation.
            batch_mean = einops.reduce(batch[key], pattern, "mean")
            # Hint: to update the mean we need x̄ₙ = (Nₙ₋₁x̄ₙ₋₁ + Bₙxₙ) / Nₙ, where the subscript represents
            # the update step, N is the running item count, B is this batch size, x̄ is the running mean,
            # and x is the current batch mean. Some rearrangement is then required to avoid risking
            # numerical overflow. Another hint: Nₙ₋₁ = Nₙ - Bₙ. Rearrangement yields
            # x̄ₙ = x̄ₙ₋₁ + Bₙ * (xₙ - x̄ₙ₋₁) / Nₙ
            mean[key] = (
                mean[key]
                + this_batch_size * (batch_mean - mean[key]) / running_item_count
            )
            max[key] = torch.maximum(
                max[key], einops.reduce(batch[key], pattern, "max")
            )
            min[key] = torch.minimum(
                min[key], einops.reduce(batch[key], pattern, "min")
            )

        if i == ceil(max_num_samples / batch_size) - 1:
            break

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        drop_last=False,
        generator=generator,
    )
    first_batch_ = None
    running_item_count = 0  # for online std computation
    error_raised = False
    for i, batch in tqdm.tqdm(
        enumerate(dataloader),
        total=ceil(max_num_samples / batch_size),
        desc="Compute std",
    ):
        this_batch_size = len(batch["index"])
        running_item_count += this_batch_size
        # Sanity check to make sure the batches are still in the same order as before.
        if first_batch_ is None:
            first_batch_ = deepcopy(batch)
            # Ensure first_batch is not None before indexing
            if first_batch is not None:
                for key in stats_patterns:
                    assert torch.equal(first_batch_[key], first_batch[key])
        for key, pattern in stats_patterns.items():
            if key not in batch.keys():
                if not error_raised:
                    logger.error(
                        f"[STD] Key '{key}' from stats_patterns not found in batch {i}/{ceil(max_num_samples) / batch_size}. Available keys: {batch.keys()}. Ignoring this key."
                    )
                continue
            batch[key] = batch[key].float()
            # Numerically stable update step for mean computation (where the mean is over squared
            # residuals). See notes in the mean computation loop above.
            batch_std = einops.reduce((batch[key] - mean[key]) ** 2, pattern, "mean")
            std[key] = (
                std[key] + this_batch_size * (batch_std - std[key]) / running_item_count
            )

        if i == ceil(max_num_samples / batch_size) - 1:
            break

    for key in stats_patterns:
        std[key] = torch.sqrt(std[key])

    stats = {}
    for key in stats_patterns:
        stats[key] = {
            "mean": mean[key],
            "std": std[key],
            "max": max[key],
            "min": min[key],
        }
    return stats


def tensor_to_list(obj):
    """
    Convert all  torch.Tensor from an object
    (dict, list to list.
    """
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: tensor_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [tensor_to_list(x) for x in obj]
    else:
        return obj


def decode_video_frames_torchvision(
    video_path: Path | str,
    timestamp: list[float],
    tolerance_s: float = 1,
    backend: str = "pyav",
    log_loaded_timestamps: bool = False,
) -> torch.Tensor:
    """Loads frames associated to the requested timestamps of a video

    The backend can be either "pyav" (default) or "video_reader".
    "video_reader" requires installing torchvision from source, see:
    https://github.com/pytorch/vision/blob/main/torchvision/csrc/io/decoder/gpu/README.rst
    (note that you need to compile against ffmpeg<4.3)

    While both use cpu, "video_reader" is supposedly faster than "pyav" but requires additional setup.
    For more info on video decoding, see `benchmark/video/README.md`

    See torchvision doc for more info on these two backends:
    https://pytorch.org/vision/0.18/index.html?highlight=backend#torchvision.set_video_backend

    Note: Video benefits from inter-frame compression. Instead of storing every frame individually,
    the encoder stores a reference frame (or a key frame) and subsequent frames as differences relative to
    that key frame. As a consequence, to access a requested frame, we need to load the preceding key frame,
    and all subsequent frames until reaching the requested frame. The number of key frames in a video
    can be adjusted during encoding to take into account decoding time and video size in bytes.
    """
    video_path = str(video_path)

    # set backend
    keyframes_only = False
    torchvision.set_video_backend(backend)
    if backend == "pyav":
        keyframes_only = True  # pyav doesnt support accuracte seek

    # set a video stream reader
    # TODO(rcadene): also load audio stream at the same time
    reader = torchvision.io.VideoReader(video_path, "video")

    # set the first and last requested timestamps
    # Note: previous timestamps are usually loaded, since we need to access the previous key frame
    first_ts = timestamp[0]
    last_ts = timestamp[-1]

    # access closest key frame of the first requested frame
    # Note: closest key frame timestamp is usally smaller than `first_ts` (e.g. key frame can be the first frame of the video)
    # for details on what `seek` is doing see: https://pyav.basswood-io.com/docs/stable/api/container.html?highlight=inputcontainer#av.container.InputContainer.seek
    reader.seek(first_ts, keyframes_only=keyframes_only)

    # load all frames until last requested frame
    loaded_frames = []
    loaded_ts = []
    for frame in reader:
        current_ts = frame["pts"]
        if log_loaded_timestamps:
            logger.info(f"frame loaded at timestamp={current_ts:.4f}")
        loaded_frames.append(frame["data"])
        loaded_ts.append(current_ts)
        if current_ts >= last_ts:
            break

    if backend == "pyav":
        reader.container.close()

    reader = None

    query_ts = torch.tensor(timestamp, dtype=torch.float64)
    loaded_ts = torch.tensor(loaded_ts, dtype=torch.float64)  # type: ignore

    # compute distances between each query timestamp and timestamps of all loaded frames
    dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)  # type: ignore
    min_, argmin_ = dist.min(1)

    is_within_tol = min_ < tolerance_s
    if not is_within_tol.all():
        logger.warning(
            f"One or several query timestamps unexpectedly violate the tolerance ({min_[~is_within_tol]} > {tolerance_s=})."
            " It means that the closest frame that can be loaded from the video is too far away in time."
            " This might be due to synchronization issues with timestamps during data collection."
            " To be safe, we advise to ignore this item during training."
            f"\nqueried timestamps: {query_ts}"
            f"\nloaded timestamps: {loaded_ts}"
            f"\nvideo: {video_path}"
            f"\nbackend: {backend}"
        )

    # get closest frames to the query timestamps
    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    closest_ts = loaded_ts[argmin_]

    if log_loaded_timestamps:
        logger.info(f"{closest_ts=}")

    # convert to the pytorch format which is float32 in [0,1] range (and channel first)
    closest_frames = closest_frames.type(torch.float32) / 255

    assert len(timestamp) == len(closest_frames)
    return closest_frames


def compute_bboxes(
    dataset_root_path: Path,
    detect_instruction: str,
    image_key: str,
    dataset_name: str,
    image_keys_to_keep: list[str] = [],
    max_batch_size: int = MAX_BATCH_SIZE,
) -> tuple[Path, int]:
    """
    This function edits a dataset in lerobot format v2 or v2.1 to train an ACT model with bounding boxes.

    This will create a new dataset called `dataset_root_path + _bboxes`.
    What we do:
    - For each episode, we load the video, exctract the first frame, and calculate the bounding box
    - Store that information in the parquet files under obervation.environment_state
    - Remove episodes for which we couldn't find bboxes and compute stats for the new dataset and save them in the meta folder.
    - Edit the info.json and stats.json files to remove video keys and add the new bounding box keys.
    - Delete the videos folder.

    -> Return the dataset path and the number of episodes for which we found bboxes.
    """
    # Load the dataset with phosphobot to fix episodes.jsonl issues (usually: missing episodes)
    dataset = LeRobotDataset(path=str(dataset_root_path), enforce_path=False)
    dataset.load_meta_models()

    # Copy the dataset to a new folder
    new_dataset_path = dataset_root_path.parent / f"{dataset_root_path.name}_bboxes"
    if new_dataset_path.exists():
        logger.warning(
            f"Dataset {new_dataset_path} already exists. Removing it and creating a new one."
        )
        shutil.rmtree(new_dataset_path)

    logger.info(f"Copying dataset to {new_dataset_path}")
    shutil.copytree(dataset_root_path, new_dataset_path)
    act_volume.commit()

    # raise error if not exists
    if not os.path.exists(new_dataset_path):
        raise FileNotFoundError(f"Newly copied data to {new_dataset_path} not found")

    dataset = LeRobotDataset(path=str(new_dataset_path), enforce_path=False)

    # Open the info.json file and validate it
    info_path = new_dataset_path / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Info file not found: {info_path}")
    validated_info = InfoModel.from_json(
        meta_folder_path=str(new_dataset_path / "meta")
    )

    selected_video_dir = None
    path_to_videos = new_dataset_path / "videos" / "chunk-000"
    if not path_to_videos.exists():
        logger.warning(f"Videos folder not found in the dataset: {path_to_videos}. ")
        raise FileNotFoundError(
            f"Videos folder not found in the dataset: {path_to_videos}. "
            "Please make sure the dataset has videos in the expected format."
        )
    else:
        # list the dirs in path_to_videos
        video_dirs = [d for d in path_to_videos.iterdir() if d.is_dir()]
        for video_dir in video_dirs:
            if image_key in video_dir.name:
                logger.info(
                    f"Found video directory with key {image_key}: {video_dir.name}"
                )
                selected_video_dir = video_dir
                break

    if selected_video_dir is None:
        valid_video_dirs = [d.name for d in video_dirs]
        raise FileNotFoundError(
            f"""No video directory found with key {image_key}, found: {valid_video_dirs}
Please specify one of the following video keys when launching a training: {", ".join(valid_video_dirs)}.
"""
        )

    # TODO: We will do the reprompting here by sending a whole batch of first frames to PaliGemma and checking how many bboxes are detected

    episodes_to_delete: list[int] = []

    # We build a batch of frames to send to PaliGemma
    cursor = 0
    while cursor < validated_info.total_episodes:
        # Last batch is handled thanks to the min condition
        chunck_size = min(max_batch_size, validated_info.total_episodes - cursor)
        chunck_episodes = range(cursor, cursor + chunck_size)

        # Load the first frame of each episode in the batch
        frames = []
        for episode_index in chunck_episodes:
            video_path = (
                new_dataset_path
                / "videos"
                / "chunk-000"
                / selected_video_dir
                / f"episode_{episode_index:06d}.mp4"
            )

            if not video_path.exists():
                logger.warning(
                    f"Video file not found: {video_path}. Skipping episode {episode_index}."
                )
                episodes_to_delete.append(episode_index)
                continue
            video_capture = cv2.VideoCapture(str(video_path))
            if not video_capture.isOpened():
                logger.error(f"Failed to open video file: {video_path}")
                episodes_to_delete.append(episode_index)
                continue
            # Read the first frame
            ret, frame = video_capture.read()
            video_capture.release()
            if not ret:
                logger.error(f"Failed to read the first frame of video: {video_path}")
                episodes_to_delete.append(episode_index)
                continue
            frames.append(frame[..., ::-1])  # Convert BGR to RGB

        # Call PaliGemma to compute the bounding box with the frames
        logger.info(
            f"Calling PaliGemma to compute the bounding box for {len(frames)} episodes"
        )
        bboxes = paligemma_detect.remote(
            frames=np.array(frames),
            instructions=[detect_instruction] * len(frames),
        )
        for bbox_index, bbox in enumerate(bboxes):
            current_episode_index = cursor + bbox_index
            if bbox == [0.0, 0.0, 0.0, 0.0]:
                logger.warning(
                    f"Failed to detect bounding box for episode {current_episode_index}. Received bbox: {bbox}. "
                    "Skipping this episode."
                )
                episodes_to_delete.append(current_episode_index)
                continue

            # Save the bounding box in the parquet file
            parquet_file_path = (
                new_dataset_path
                / "data"
                / f"chunk-000/episode_{current_episode_index:06d}.parquet"
            )
            df = pd.read_parquet(parquet_file_path)
            df["observation.environment_state"] = [bbox] * df.shape[0]
            df.to_parquet(parquet_file_path, index=False)
            logger.info(
                f"Saved bounding box {bbox} for episode {current_episode_index} in {parquet_file_path}"
            )

        # Update the cursor
        cursor += chunck_size

    # Debug: list all the parquet files in the dataset
    parquet_files = list(new_dataset_path.rglob("*.parquet"))
    logger.debug(f"Parquet files in the dataset: {parquet_files}")

    # Delete the episodes for which we couldn't find bboxes
    nb_episodes_deleted = 0
    if episodes_to_delete:
        # Look at how many episodes will be left and raise an error if less than 2 # episodes are left
        if (
            validated_info.total_episodes - len(episodes_to_delete)
            <= MIN_NUMBER_OF_BBOXES
        ):
            visualizer_url = (
                f"https://lerobot-visualize-dataset.hf.space/{dataset_name}/"
            )
            raise RuntimeError(
                f"The object '{detect_instruction}' was detected in {validated_info.total_episodes - len(episodes_to_delete)} episodes in {image_key} camera"
                f" (should be: {MIN_NUMBER_OF_BBOXES} episodes min)."
                f" This is not enough to train a model. Check your dataset: {visualizer_url} and rephrase the instruction."
            )

        logger.info(
            f"Deleting {len(episodes_to_delete)} episodes for which we couldn't find bounding boxes: {episodes_to_delete}"
        )
        for episode_index in episodes_to_delete:
            # The true index is the episode index minus the number of episodes deleted so far
            # This is because when we delete an episode, the indices of the remaining episodes shift
            true_index = episode_index - nb_episodes_deleted
            logger.info(
                f"Deleting episode {true_index} (old index: {episode_index}) from dataset."
            )
            dataset.delete_episode(episode_id=true_index, update_hub=False)
            nb_episodes_deleted += 1

        parquet_files = list(new_dataset_path.rglob("*.parquet"))
        logger.debug(
            f"Total episodes deleted: {nb_episodes_deleted}. Parquet files left: {parquet_files}"
        )

    # Iterate over the .parquet files and removed the .parquet if there is no "observation.environment_state" key
    for parquet_file in new_dataset_path.rglob("*.parquet"):
        df = pd.read_parquet(parquet_file)
        if "observation.environment_state" not in df.columns:
            raise ValueError(
                f"Parquet file {parquet_file} does not contain 'observation.environment_state' key. "
                "This is unexpected after computing bounding boxes."
            )

    # Load the dataset with phosphobot to fix episodes.jsonl issues (usually: missing episodes)
    dataset = LeRobotDataset(path=str(new_dataset_path), enforce_path=False)
    dataset.load_meta_models()

    # Log the number of episodes and the content of the episodes.jsonl file
    episodes_jsonl_path = new_dataset_path / "meta" / "episodes.jsonl"
    if not episodes_jsonl_path.exists():
        raise FileNotFoundError(
            f"episodes.jsonl file not found in the dataset: {episodes_jsonl_path}"
        )
    with open(episodes_jsonl_path, "r") as f:
        content = f.readlines()
        n_episodes_jsonl = len(content)

    # Log number of .parquet files
    parquet_files = list(new_dataset_path.rglob("*.parquet"))
    logger.info(f"Number of parquet files in the dataset: {len(parquet_files)}")

    if n_episodes_jsonl != len(parquet_files):
        raise ValueError(
            f"Number of episodes in episodes.jsonl ({n_episodes_jsonl}) does not match "
            f"the number of parquet files ({len(parquet_files)}). "
            "This is unexpected after computing bounding boxes."
        )

    # Reload the info_model from disk
    validated_info = InfoModel.from_json(
        meta_folder_path=str(new_dataset_path / "meta")
    )

    # Add the bounding box keys to the info.json file to compute their stats
    validated_info.features.observation_environment_state = FeatureDetails(
        dtype="float32",
        shape=[4],
        names=["x1", "y1", "x2", "y2"],
    )
    validated_info.codebase_version = "v2.0"  # Since we calculate the stats with v2.0

    # Save the updated info.json file
    info_path.unlink()  # Remove the old info.json file
    validated_info.save(meta_folder_path=str(new_dataset_path / "meta"))

    act_volume.commit()

    # Remove stats.json and episode_stats.jsonl files if they exist
    stats_path = new_dataset_path / "meta" / "stats.json"
    if stats_path.exists():
        logger.info(f"Removing existing stats file: {stats_path}")
        os.remove(stats_path)
    episodes_stats_path = new_dataset_path / "meta" / "episodes_stats.jsonl"
    if episodes_stats_path.exists():
        logger.info(f"Removing existing episodes stats file: {episodes_stats_path}")
        os.remove(episodes_stats_path)

    # Update the dataset stats
    logger.info(f"Video dirs found: {video_dirs}")
    stats = tensor_to_list(compute_stats(new_dataset_path))
    for key in video_dirs:
        if key.name in stats.keys() and key.name not in image_keys_to_keep:
            del stats[key.name]
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)

    logger.success(
        f"Computed stats for the new dataset with bounding boxes: {stats_path}"
    )

    # Remove the videos folders that are not in the image_keys_to_keep list
    number_of_deleted_videos = 0
    videos_path = new_dataset_path / "videos"
    # We assume for now there is only one chunck chunk-000
    full_videos_path = videos_path / "chunk-000"
    # List the folders in the videos_path
    video_dirs = [d for d in full_videos_path.iterdir() if d.is_dir()]
    for key in video_dirs:
        if key.name in image_keys_to_keep:
            logger.info(f"Keeping video directory: {key.name}")
        else:
            logger.info(f"Removing video directory: {key.name}")
            # Count the number of deleted videos in the folder
            number_of_deleted_videos += len([f for f in key.iterdir() if f.is_file()])
            shutil.rmtree(key)

    # Load the info.json file and update the number of videos
    with open(info_path, "r") as f:
        info = json.load(f)
        info["total_videos"] = info["total_videos"] - number_of_deleted_videos
        for key in video_dirs:
            if (
                key.name in info["features"].keys()
                and key.name not in image_keys_to_keep
            ):
                del info["features"][key.name]

    # Delete existing info.json file
    if info_path.exists():
        logger.info(f"Removing existing info file: {info_path}")
        os.remove(info_path)
    # Save the updated info.json file
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)

    act_volume.commit()

    return new_dataset_path, validated_info.total_episodes
