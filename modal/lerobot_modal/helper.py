from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from copy import deepcopy
from math import ceil
from time import sleep

import os
import cv2
import torch
import torchvision
import numpy as np
import pandas as pd
import tqdm
import einops
import json
import multiprocessing
from fastapi import HTTPException
from pydantic import BaseModel
from loguru import logger
from torch.utils.data import Dataset as TorchDataset
from supabase import Client
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import (
    RepositoryNotFoundError,
    RevisionNotFoundError,
)


class InferenceRequest(BaseModel):
    encoded: str  # json_numpy encoded dict


def _find_or_download_model(
    model_id: str,
    supabase_client: Client,
    server_id: int,
    checkpoint: int | None = None,
) -> str:
    """Find or download model from HuggingFace."""
    try:
        local_model_path = snapshot_download(
            repo_id=model_id,
            repo_type="model",
            revision=str(checkpoint) if checkpoint is not None else None,
            cache_dir="/data/hf_cache",
        )
    except RepositoryNotFoundError as e:
        logger.error(f"Failed to download model {model_id}: {e}")
        _update_server_status(supabase_client, server_id, "failed")
        raise HTTPException(
            status_code=400,
            detail=f"Model {model_id} not found. Make sure the model is public. Error: {e}",
        )
    except RevisionNotFoundError as e:
        logger.error(
            f"Failed to download model {model_id} at revision {checkpoint}: {e}"
        )
        _update_server_status(supabase_client, server_id, "failed")
        raise HTTPException(
            status_code=400,
            detail=f"Model {model_id} at revision {checkpoint} not found. Error: {e}",
        )
    except Exception as e:
        logger.error(f"Failed to download model {model_id}: {e}")
        _update_server_status(supabase_client, server_id, "failed")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download model {model_id}. Error: {e}",
        )
    return local_model_path


def _upload_partial_checkpoint(output_dir: Path, model_name: str, hf_token: str):
    """
    Upload whatever is already in output_dir/checkpoints/last/pretrained_model
    to the HF model repo, so we don't lose everything if we time out.
    """
    api = HfApi(token=hf_token)
    checkpoint_dir = output_dir / "checkpoints" / "last" / "pretrained_model"
    if not checkpoint_dir.exists():
        logger.error(f"No partial checkpoint found at {checkpoint_dir}")
        return
    for item in checkpoint_dir.glob("**/*"):
        if item.is_file():
            relpath = item.relative_to(checkpoint_dir)
            logger.info(f"Uploading partial checkpoint {relpath}")
            api.upload_file(
                repo_type="model",
                path_or_fileobj=str(item.resolve()),
                path_in_repo=str(relpath),
                repo_id=model_name,
                token=hf_token,
            )


def _update_server_status(
    supabase_client: Client,
    server_id: int,
    status: str,
):
    """Update server status in database."""
    logger.info(f"Updating server status to {status} for server_id {server_id}")
    if status == "failed":
        server_payload = {
            "status": status,
            "terminated_at": datetime.now(timezone.utc).isoformat(),
        }
        supabase_client.table("servers").update(server_payload).eq(
            "id", server_id
        ).execute()
        # Update also the AI control session
        ai_control_payload = {
            "status": "stopped",
            "ended_at": datetime.now(timezone.utc).isoformat(),
        }
        supabase_client.table("ai_control_sessions").update(ai_control_payload).eq(
            "server_id", server_id
        ).execute()
    elif status == "stopped":
        server_payload = {
            "status": status,
            "terminated_at": datetime.now(timezone.utc).isoformat(),
        }
        supabase_client.table("servers").update(server_payload).eq(
            "id", server_id
        ).execute()
        # Update also the AI control session
        ai_control_payload = {
            "status": "stopped",
            "ended_at": datetime.now(timezone.utc).isoformat(),
        }
        supabase_client.table("ai_control_sessions").update(ai_control_payload).eq(
            "server_id", server_id
        ).execute()
    else:
        server_payload = {
            "status": status,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        supabase_client.table("servers").update(server_payload).eq(
            "id", server_id
        ).execute()


def validate_inputs(
    current_qpos: list[float],
    images: list[np.ndarray],
    model_specifics: Any,
    target_size: tuple[int, int],
) -> None:
    """Validate inputs for LeRobot policy models

    Args:
        current_qpos (list[float]): Current robot state
        images (list[np.ndarray]): List of images
        model_specifics (Any): Model specifics containing state size and video keys
        target_size (tuple[int, int]): Expected image size (height, width)
    """
    assert (
        len(current_qpos) == model_specifics.state_size[0]
    ), f"State size mismatch: {len(current_qpos)} != {model_specifics.state_size[0]}"
    assert (
        len(images) <= len(model_specifics.video_keys)
    ), f"Number of images {len(images)} is more than the number of video keys {len(model_specifics.video_keys)}"
    if len(images) > 0:
        assert (
            len(images[0].shape) == 3
        ), f"Image shape is not correct, {images[0].shape} expected (H, W, C)"
        assert (
            len(images[0].shape) == 3 and images[0].shape[2] == 3
        ), f"Image shape is not correct {images[0].shape} expected (H, W, 3)"


def prepare_base_batch(
    current_qpos: list[float],
    images: list[np.ndarray],
    image_names: list[str],
    model_specifics: Any,
    target_size: tuple[int, int],
) -> dict[str, torch.Tensor]:
    """Prepare base observation batch for LeRobot policy models

    Args:
        current_qpos (list[float]): Current robot state
        images (list[np.ndarray]): List of images
        image_names (list[str]): List of image names corresponding to model video keys
        model_specifics (Any): Model specifics containing state key and video keys
        target_size (tuple[int, int]): Expected image size (height, width)
    Returns:
        dict[str, torch.Tensor]: batch dict
    """
    batch = {}

    # Add state
    batch[model_specifics.state_key] = torch.tensor(
        current_qpos, dtype=torch.float32, device="cuda"
    ).unsqueeze(0)

    # Add images
    for i, img in enumerate(images):
        # Double check if img.shape[:2] is (H, W) or (W, H)
        if img.shape[:2] != target_size:
            logger.info(
                f"Resizing img {image_names[i]} from {img.shape[:2]} to {target_size}"
            )
            img = cv2.resize(src=img, dsize=target_size)

        # Convert numpy array to tensor and normalize
        img_tensor = torch.from_numpy(img).float() / 255.0
        # Ensure CHW format
        if img_tensor.dim() == 3 and img_tensor.shape[2] == 3:
            img_tensor = img_tensor.permute(2, 0, 1)
        batch[image_names[i]] = img_tensor.unsqueeze(0).to("cuda")

    return batch


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

        # New: Episode info mapping
        self.episode_info: dict = {}
        for global_idx, info in self.index_mapping.items():
            ep_idx = info["episode_idx"]
            if ep_idx not in self.episode_info:
                self.episode_info[ep_idx] = {
                    "file_path": info["file_path"],
                    "videos_paths": info["videos_paths"],
                    "timestamps": None,
                }

        # Per-worker caching state
        self.current_episode_idx = None
        self.current_episode_frames = None
        self.worker_cache: dict = {}

    def _load_episode_frames(self, episode_idx: int) -> Dict[str, torch.Tensor]:
        """Load and cache frames for a single episode"""
        # Check worker-specific cache first
        worker_id = multiprocessing.current_process().pid
        if worker_id in self.worker_cache:
            if self.worker_cache[worker_id]["episode_idx"] == episode_idx:
                return self.worker_cache[worker_id]["frames"]

        # Load timestamps if not cached
        if self.episode_info[episode_idx]["timestamps"] is None:
            df = self.read_parquet(str(self.episode_info[episode_idx]["file_path"]))
            self.episode_info[episode_idx]["timestamps"] = df["timestamp"].tolist()

        # Decode frames
        decoded_frames = {}
        for video_key, video_path in self.episode_info[episode_idx][
            "videos_paths"
        ].items():
            frames = decode_video_frames_torchvision(
                video_path, self.episode_info[episode_idx]["timestamps"]
            )
            decoded_frames[video_key] = (frames * 255).to(torch.uint8)  # Store as uint8

        # Update worker cache (only keep current episode)
        self.worker_cache[worker_id] = {
            "episode_idx": episode_idx,
            "frames": decoded_frames,
        }
        return decoded_frames

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

        # Load frames for this episode
        frames = self._load_episode_frames(episode_idx)

        # Retrieve cached frames
        for video_key in self.video_keys:
            frame = frames[video_key][row_idx]
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
    num_workers: int = 6,
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


def _download_dataset_from_hf(
    dataset_name: str,
    output_dir: Path,
    hf_token: Optional[str] = None,
    max_hf_download_retries: int = 3,
) -> Path:
    """Download dataset from HuggingFace.

    Args:
        dataset_name: Name of the dataset on HuggingFace (e.g. username/dataset_name)
        output_dir: Path to the output directory where the dataset will be downloaded
        hf_token: HuggingFace token with read access to the dataset repo (optional)
    """
    logger.info(f"Downloading dataset from HuggingFace: {dataset_name}")
    dataset_path = None
    for attempt in range(max_hf_download_retries):
        try:
            # We download the dataset to the cache to easily pass it to the training script
            dataset_path_as_str = snapshot_download(
                repo_id=dataset_name,
                repo_type="dataset",
                revision="main",
                local_dir=str(output_dir.resolve()),
                token=hf_token,
                ignore_patterns=[".gitattributes", "*.lock", ".gitignore"],
            )
            dataset_path = Path(dataset_path_as_str)
            logger.success(f"Dataset {dataset_name} downloaded to {dataset_path}")
            break  # Exit the loop if download is successful
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_hf_download_retries - 1:
                sleep(1)  # Wait for 1 second before retrying
            else:
                raise RuntimeError(
                    f"Failed to download dataset {dataset_name} after {max_hf_download_retries} attempts, is Hugging Face down ? : {e}"
                )

    # Double check that the dataset path exists
    if dataset_path is None or not dataset_path.exists():
        raise RuntimeError(f"Dataset path {dataset_path} does not exist after download")

    return dataset_path


def _upload_dataset_to_hf(
    output_dir: Path,
    model_name: str,
    hf_token: str,
    private_mode: bool = False,
) -> None:
    """Upload the trained model to HuggingFace.

    Args:
        output_dir: Path to the output directory containing checkpoints
        model_name: Name of the model on HuggingFace (e.g. username/model_name)
        hf_token: HuggingFace token with write access to the model repo
        private_mode: Whether to create the model repo as private (default: False)
    """
    logger.info(f"Uploading trained model to HuggingFace: {model_name}")
    hf_api = HfApi(token=hf_token)

    # Create model repo if it doesn't exist
    try:
        hf_api.repo_info(repo_id=model_name, repo_type="model")
        logger.info(f"Model repository {model_name} already exists.")
    except Exception:
        logger.info(f"Creating model repository {model_name}")
        hf_api.create_repo(
            repo_id=model_name,
            repo_type="model",
            exist_ok=True,
            private=private_mode,
            token=hf_token,
        )

    # Upload the model
    files_directory = output_dir / "checkpoints" / "last" / "pretrained_model"
    output_paths: list[Path] = []
    for item in files_directory.glob("**/*"):
        if item.is_file():
            logger.debug(f"Uploading {item}")
            hf_api.upload_file(
                repo_type="model",
                path_or_fileobj=str(item.resolve()),
                path_in_repo=item.name,
                repo_id=model_name,
                token=hf_token,
            )
            output_paths.append(item)

    # Upload other checkpoints as well
    for item in output_dir.glob("checkpoints/*/pretrained_model/*"):
        if item.is_file():
            # Will upload all checkpoints under the name checkpoint-{number}/
            rel_path = item.relative_to(output_dir)
            number = rel_path.parts[1]
            if number == "last":
                continue
            checkpoint_number = int(rel_path.parts[1])

            # Create revision if it doesn't exist
            hf_api.create_branch(
                repo_id=model_name,
                repo_type="model",
                branch=str(checkpoint_number),
                token=hf_token,
                exist_ok=True,
            )

            hf_api.upload_file(
                repo_type="model",
                revision=str(checkpoint_number),
                path_or_fileobj=str(item.resolve()),
                path_in_repo=item.name,
                repo_id=model_name,
                token=hf_token,
            )
            output_paths.append(item)
