import json
import os
from copy import deepcopy
from math import ceil
from pathlib import Path

import einops
import numpy as np
import pandas as pd
import torch
import torchvision
import tqdm
from loguru import logger
from torch.utils.data import Dataset
import argparse


class ParquetEpisodesDataset(Dataset):
    """Custom Dataset for loading parquet files from a directory."""

    def __init__(self, dataset_dir: str):
        """
        dataset dir is the path to the folder containing data, videos, meta subfolder
        """
        self.dataset_dir = Path(dataset_dir)
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
            raise ValueError(f"No parquet files found in {dataset_dir}")

        if not self.video_paths:
            raise ValueError(f"No video files found in {dataset_dir}")

        if len(self.video_paths) % len(self.file_paths) != 0:
            raise ValueError(
                f"Number of parquet files ({len(self.file_paths)}) does not match "
                f"number of video files ({len(self.video_paths)})"
            )

        # Reindex the data to have a global index over all the steps of all the episodes
        self.episode_nb_steps = []
        # Map global idx -> (file_idx, row_idx)
        self.index_mapping: dict[int, dict] = {}
        # Store the number of steps in each episode
        self.steps_per_episode: dict[int, int] = {}
        global_idx = 0
        for file_path in self.file_paths:
            # The filepath is expected to be in the format "chunk-000/episode_000000.parquet"
            episode_idx = int(file_path.stem.split("_")[-1])
            df = self.read_parquet(str(file_path))
            nb_steps = len(df)
            self.episode_nb_steps.append(nb_steps)

            # Needed for episodes.jsonl
            self.steps_per_episode[episode_idx] = nb_steps

            # Find all the related video files and store them in the index_mapping
            related_video_files = [
                video_path
                for video_path in self.video_paths
                if f"episode_{episode_idx:06d}" in video_path.name
            ]
            # TODO: Use REGEX to split the path instead of chunk-000
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

        # video keys are the unique names of the folders in the videos directory
        videos_folders = os.path.join(self.videos_dir, "chunk-000")
        self.video_keys = str(os.listdir(videos_folders)).split("chunk-000.")[-1]

    def __len__(self):
        return self.total_length

    def read_parquet(self, file_path: str):
        # Cache the parquet files to avoid reading them multiple times
        if file_path not in self.parquet_cache:
            self.parquet_cache[file_path] = pd.read_parquet(file_path)
        return self.parquet_cache[file_path]

    def __getitem__(self, idx: int):
        if idx >= self.total_length:
            raise IndexError("Index out of bounds")

        file_path: str = self.index_mapping[idx]["file_path"]
        row_idx: int = self.index_mapping[idx]["row_idx"]

        # Read the specific row
        df = self.read_parquet(file_path)
        row_data = df.iloc[row_idx]

        # Get the related video files
        videos_paths = self.index_mapping[idx]["videos_paths"]
        video_key_to_path = {}
        for key, video_path in videos_paths.items():
            # Load the video and store it in the row_data
            video_key_to_path[key] = decode_video_frames_torchvision(
                video_path, timestamp=[row_data["timestamp"]]
            ).squeeze(0)

        # Convert each column to a Tensor
        # If it's a list/np.ndarray, turn it into a float32 tensor of that shape
        # If it's a scalar, make it a 0D or 1D float32 tensor.
        sample = {}
        for col_name, value in row_data.items():
            if isinstance(value, (list, np.ndarray)):
                sample[col_name] = torch.tensor(value, dtype=torch.float32)
            elif isinstance(value, torch.Tensor):
                sample[col_name] = value
            elif isinstance(value, str):
                # Convert the string to a list of float using eval
                sample[col_name] = torch.tensor([float(x) for x in eval(value)])
            else:
                sample[col_name] = torch.tensor([value], dtype=torch.float32)

        for key in video_key_to_path.keys():
            sample[key] = video_key_to_path[key]

        return sample

    def write_episodes(self, output_dir: str):
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
):
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


def compute_stats(dataset_path, batch_size=128, num_workers=2, max_num_samples=None):
    """Compute mean/std and min/max statistics of all data keys in a LeRobotDataset."""
    dataset = ParquetEpisodesDataset(dataset_path)

    if max_num_samples is None:
        max_num_samples = len(dataset)

    # for more info on why we need to set the same number of workers, see `load_from_videos`
    dataset = ParquetEpisodesDataset(dataset_path)

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
    first_batch = None
    running_item_count = 0  # for online mean computation

    logger.info("Starting to create seeded dataloader")

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
            for key in stats_patterns:
                assert torch.equal(first_batch_[key], first_batch[key])
        for key, pattern in stats_patterns.items():
            batch[key] = batch[key].float()
            # Numerically stable update step for mean computation (where the mean is over squared
            # residuals).See notes in the mean computation loop above.
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute statistics for a dataset.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to the dataset directory containing data, videos, and meta subfolders.",
    )
    args = parser.parse_args()

    DATASET_PATH = args.dataset_path

    stats = tensor_to_list(compute_stats(DATASET_PATH))

    META_PATH = os.path.join(DATASET_PATH, "meta")

    STATS_FILE = os.path.join(META_PATH, "stats.json")
    # Overwrite the stats.json file
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=4)

    logger.success(f"Stats computed and saved to {STATS_FILE}")

    # Edit the info.json file
    dataset = ParquetEpisodesDataset(DATASET_PATH)

    # Find the data that has changed
    total_episodes = len(dataset.parquet_cache)
    total_frames = dataset.total_length
    total_videos = len(dataset.video_paths)
    splits = {"train": f"0:{total_episodes}"}

    INFO_FILE = os.path.join(META_PATH, "info.json")
    with open(INFO_FILE, "r") as f:
        info = json.load(f)
    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["total_videos"] = total_videos
    info["splits"] = splits

    with open(INFO_FILE, "w") as f:
        json.dump(info, f, indent=4)

    logger.success(f"Info updated and saved to {INFO_FILE}")

    # Overwrite the episodes.jsonl file
    EPISODES_FILE = os.path.join(META_PATH, "episodes.jsonl")
    # delete the episodes file if it exists
    if os.path.exists(EPISODES_FILE):
        os.remove(EPISODES_FILE)
    dataset.write_episodes(EPISODES_FILE)

    logger.success(f"Episodes written to {EPISODES_FILE}")
    logger.success("Stats computation complete")
