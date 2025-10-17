import json
import os
import shutil
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
import pandas as pd
import torch
import modal
from huggingface_hub import HfApi
from loguru import logger

from phosphobot.am.base import TrainingParamsAct, TrainingParamsActWithBbox
from phosphobot.models import InfoModel
from phosphobot.models.lerobot_dataset import FeatureDetails
from phosphobot.models.lerobot_dataset import LeRobotDataset
from phosphobot.am.act import ACTSpawnConfig
from .helper import validate_inputs, prepare_base_batch, tensor_to_list, compute_stats
from .app import (
    MINUTES,
    base_image,
    FUNCTION_GPU_INFERENCE,
    FUNCTION_TIMEOUT_INFERENCE,
    FUNCTION_GPU_TRAINING,
    FUNCTION_TIMEOUT_TRAINING,
    FUNCTION_CPU_TRAINING,
    phosphobot_dir,
    serve_policy,
    train_policy,
)


# Get PaliGemma detector
paligemma_detect = modal.Function.from_name("paligemma-detector", "detect_object")
act_volume = modal.Volume.from_name("act")

# Minimum number of bounding boxes to train an ACT model
MIN_NUMBER_OF_BBOXES = 10
# Maximum batch size to use for PaliGemma (can cause OOM otherwise)
MAX_BATCH_SIZE = 140


class NotEnoughBBoxesError(Exception):
    """Custom exception for when not enough bounding boxes are detected."""

    pass


class InvalidInputError(Exception):
    """Custom exception for invalid input data."""

    pass


class RetryError(Exception):
    """Custom exception to retry the inference call."""

    pass


def process_act_inference(
    policy: Any,
    model_specifics: ACTSpawnConfig,
    current_qpos: list[float],
    images: list[np.ndarray],
    image_names: list[str],
    target_size: tuple[int, int],
    image_for_bboxes: torch.Tensor | None,
    detect_instruction: str | None = None,
    last_bbox_computed: list[float] | None = None,
) -> np.ndarray:
    """Process images and perform inference using the SmolVLA policy.

    Args:
        policy: lerobot.policies.act.modeling_act.ACTPolicy object
        model_specifics: ACTSpawnConfig object
        current_qpos: Current robot state
        images: List of images
        image_names: List of image names corresponding to the images
        target_size: Target size for resizing images (height, width)
        image_for_bboxes: Image tensor used for bounding box detection (if any)
        detect_instruction: Instruction for bounding box detection (if any)
        last_bbox_computed: Last computed bounding boxes (if any)
    Returns:
        np.ndarray: predicted action chunk
    """
    # Validate inputs using common function
    validate_inputs(current_qpos, images, model_specifics, target_size)

    with torch.no_grad(), torch.autocast(device_type="cuda"):
        # Prepare base observation using common function
        batch = prepare_base_batch(
            current_qpos, images, image_names, model_specifics, target_size
        )

        # Add ACT-specific bounding boxes if available
        if model_specifics.env_key is not None:
            bboxes = paligemma_detect.remote(
                # We add the batch dimension to the image_for_bboxes, which is B=1 here
                frames=np.array([image_for_bboxes]),
                instructions=[detect_instruction],
            )
            # For now we delete the batch dimension to stay compatible with the old code
            bboxes = bboxes[0]
            if bboxes == [0.0, 0.0, 0.0, 0.0]:
                # We want to let the client know that it needs to retry with a new image
                if last_bbox_computed is None:
                    raise RetryError(
                        f"The object '{detect_instruction}' was not detected in the selected camera. Try with a different instruction or camera."
                    )
                # Otherwise, we use the last computed bounding boxes
                logger.debug(
                    f"No bounding boxes detected, using last computed: {last_bbox_computed}"
                )
                bboxes = last_bbox_computed
            else:
                logger.info(f"Detected bounding boxes: {bboxes}")

            # last_bbox_computed = bboxes
            if last_bbox_computed is None:
                last_bbox_computed = bboxes
            else:
                # Do a rolling average of the last 10 bboxes
                last_bbox_computed = [
                    (last_bbox_computed[i] * 9 + bboxes[i]) / 10
                    for i in range(len(bboxes))
                ]

            batch[model_specifics.env_key] = torch.tensor(
                last_bbox_computed, dtype=torch.float32, device="cuda"
            ).view(1, -1)

        # Run inference
        batch = policy.normalize_inputs(batch)
        if policy.config.image_features:  # type: ignore
            batch = dict(batch)
            batch["observation.images"] = [
                batch[key]
                for key in policy.config.image_features  # type: ignore
            ]

        actions = policy.model(batch)[0][:, : policy.config.n_action_steps]  # type: ignore
        actions = policy.unnormalize_outputs({"action": actions})["action"]  # type: ignore
        actions = actions.transpose(0, 1)
        actions = actions.cpu().numpy()

        return actions, last_bbox_computed


def read_first_frame_with_pyav(video_path):
    """
    Read the first frame from a video file using PyAV library.
    Returns the frame as a numpy array or None if failed.
    """
    container = None
    try:
        container = av.open(str(video_path))
        video_stream = container.streams.video[0]

        for frame in container.decode(video_stream):
            # Convert to RGB numpy array
            img = frame.to_rgb().to_ndarray()
            container.close()
            return img

    except Exception as e:
        logger.error(f"PyAV failed to read {video_path}: {e}")
        return None
    finally:
        try:
            if container is not None:
                container.close()
        except Exception as e:
            logger.error(f"Failed to close container for {video_path}: {e}")
            pass


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

    # Ensure the image key exists in the dataset, if not, fail fast
    image_key_detected = False
    if dataset.info_model is not None:
        for meta_image_key in dataset.info_model.features.observation_images.keys():
            if image_key in meta_image_key:
                image_key_detected = True
                break

    if not image_key_detected:
        if dataset.info_model is None:
            raise ValueError("Dataset could not be loaded correctly.")
        raise InvalidInputError(
            f"Image key '{image_key}' not found in the dataset info_model. "
            "Please check the image keys in the dataset and pass the appropriate parameter.\n"
            f"Available image keys: {list(dataset.info_model.features.observation_images.keys())}"
        )

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

            # Read the first frame using PyAV
            frame = read_first_frame_with_pyav(video_path)

            if frame is None:
                logger.error(f"Failed to read the first frame of video: {video_path}")
                episodes_to_delete.append(episode_index)
                continue

            # Resize the frame to 224x224 (PaliGemma expects this size)
            frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)

            # PyAV already returns RGB format, so no need to convert BGR to RGB
            frames.append(frame[..., ::-1])

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
            raise NotEnoughBBoxesError(
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


def upload_bbox_dataset_to_hf(
    dataset_path: Path,
    hf_token: str,
    private_mode: bool = False,
) -> str:
    """
    Upload the bounding box dataset to Hugging Face.

    Args:
        dataset_path: Path to the dataset
        hf_token: Hugging Face token
        private_mode: Whether to upload the dataset in private mode
    """
    # Use user's namespace for private training, phospho-app for public
    if private_mode and hf_token:
        # Get username from HF token for private training
        hf_api_temp = HfApi(token=hf_token)
        try:
            user_info = hf_api_temp.whoami()
            username = user_info.get("name")
            if username:
                dataset_name = f"{username}/{dataset_path.name}"
            else:
                logger.warning(
                    "Could not get username from HF token, using phospho-app namespace"
                )
                dataset_name = "phospho-app/" + dataset_path.name
        except Exception as e:
            logger.warning(
                f"Error getting username from HF token: {e}, using phospho-app namespace"
            )
            dataset_name = "phospho-app/" + dataset_path.name
    else:
        dataset_name = "phospho-app/" + dataset_path.name

    logger.info(f"Uploading dataset {dataset_name} to Hugging Face")
    hf_api = HfApi(token=hf_token)
    hf_api.create_repo(
        repo_type="dataset",
        repo_id=dataset_name,
        token=hf_token,
        exist_ok=True,
        private=private_mode,
    )
    hf_api.upload_folder(
        repo_type="dataset",
        folder_path=str(dataset_path),
        repo_id=dataset_name,
        token=hf_token,
    )
    hf_api.create_branch(
        repo_id=dataset_name,
        repo_type="dataset",
        branch="v2.0",
        token=hf_token,
        exist_ok=True,
    )
    hf_api.upload_folder(
        repo_type="dataset",
        folder_path=str(dataset_path),
        repo_id=dataset_name,
        token=hf_token,
        revision="v2.0",
    )
    logger.success(
        f"Dataset with bounding boxes - {dataset_name} - uploaded to Hugging Face successfully!"
    )
    return dataset_name


def prepare_bounding_box_dataset(
    dataset_path: Path,
    dataset_name: str,
    detect_instruction: str,
    image_key: str,
    min_number_of_episodes: int,
    image_keys_to_keep: list[str] = [],
    private_mode: bool = False,
    hf_token: str | None = None,
) -> tuple[Path, str]:
    """
    Prepare a dataset with bounding boxes for ACT training.
    If a HuggingFace token is provided, the new dataset will also be uploaded to HuggingFace.

    Steps:
        1. Compute bounding boxes on the original dataset with `compute_bboxes`
        2. Upload the new dataset to HuggingFace with `upload_bbox_dataset_to_hf`
        3. Return the new dataset path and the number of valid episodes

    Args:
        dataset_path: Path to the original dataset
        dataset_name: Name of the dataset
        detect_instruction: Instruction for bounding box detection,
        image_key: Camera image key to use for bounding box detection,
        min_number_of_episodes: Minimum number of episodes with bounding boxes required to proceed
        image_keys_to_keep: List of image keys to keep in the dataset (videos for other keys will be deleted)
        private_mode: Whether to upload the dataset in private mode on Hugging Face
        hf_token: Hugging Face token for uploading the dataset

    Returns:
        tuple[Path, int]: Path to the new dataset with bounding boxes and the number of valid episodes
    """
    logger.info(
        f"Computing bounding boxes for dataset {dataset_name}, this should take about 5 minutes.."
    )
    dataset_path, number_of_valid_episodes = compute_bboxes(
        dataset_root_path=dataset_path,
        detect_instruction=detect_instruction,
        image_key=image_key,
        dataset_name=dataset_name,
        image_keys_to_keep=image_keys_to_keep,
    )
    logger.success(f"Bounding boxes computed and saved to {dataset_path}")

    if hf_token is None:
        raise RuntimeError(
            "No Hugging Face token provided, could not upload residual dataset with bounding boxes to Hugging Face."
        )
    else:
        dataset_name = upload_bbox_dataset_to_hf(
            dataset_path=dataset_path,
            hf_token=hf_token,
            private_mode=private_mode,
        )

    if number_of_valid_episodes < min_number_of_episodes:
        visualizer_url = f"https://lerobot-visualize-dataset.hf.space/{dataset_name}"
        raise RuntimeError(
            f"The object '{detect_instruction}' was detected in {number_of_valid_episodes} episodes in {image_key} camera"
            f" (should be: {min_number_of_episodes} episodes min)."
            f" This is not enough to train a model. Check your dataset: {visualizer_url} and rephrase the instruction."
        )

    return dataset_path, dataset_name


# ======== ACT ========
act_app = modal.App("act-server")
act_volume = modal.Volume.from_name("act", create_if_missing=True)

# ACT image
act_image = (
    base_image.uv_pip_install(
        "lerobot[act]==0.3.3",  # before introduction of LeRobotDataset v3.0
    )
    .pip_install_from_pyproject(pyproject_toml=str(phosphobot_dir / "pyproject.toml"))
    .add_local_python_source("phosphobot")
)


@act_app.function(
    image=act_image,
    gpu=FUNCTION_GPU_INFERENCE,
    timeout=FUNCTION_TIMEOUT_INFERENCE,
    secrets=[
        modal.Secret.from_dict({"MODAL_LOGLEVEL": "DEBUG"}),
        modal.Secret.from_name("supabase"),
    ],
    volumes={"/data": act_volume},
)
async def serve(
    model_id: str,
    server_id: int,
    model_specifics: ACTSpawnConfig,
    checkpoint: int | None = None,
    timeout: int = FUNCTION_TIMEOUT_INFERENCE,
    q=None,
):
    """ACT model serving function."""
    await serve_policy(
        model_id=model_id,
        server_id=server_id,
        model_specifics=model_specifics,
        checkpoint=checkpoint,
        timeout=timeout,
        q=q,
    )


@act_app.function(
    image=act_image,
    gpu=FUNCTION_GPU_TRAINING,
    timeout=FUNCTION_TIMEOUT_TRAINING + 20 * MINUTES,
    secrets=[
        modal.Secret.from_dict({"MODAL_LOGLEVEL": "DEBUG"}),
        modal.Secret.from_name("supabase"),
        modal.Secret.from_name("huggingface"),
    ],
    volumes={"/data": act_volume},
    cpu=FUNCTION_CPU_TRAINING,
)
def train(
    training_id: int,
    dataset_name: str,
    wandb_api_key: str | None,
    model_name: str,
    training_params: TrainingParamsAct | TrainingParamsActWithBbox,
    user_hf_token: str | None = None,
    private_mode: bool = False,
    max_hf_download_retries: int = 3,
    timeout_seconds: int = FUNCTION_TIMEOUT_TRAINING,
    wandb_run_id: str = "wandb_run_id_not_set",
    **kwargs,
):
    """ACT training function."""
    # remove model_type from kwargs if present
    # to avoid passing model_type twice in the `train_policy()` function call
    if "model_type" in kwargs:
        del kwargs["model_type"]

    train_policy(
        model_type="act",
        training_id=training_id,
        dataset_name=dataset_name,
        wandb_api_key=wandb_api_key,
        model_name=model_name,
        training_params=training_params,
        user_hf_token=user_hf_token,
        private_mode=private_mode,
        max_hf_download_retries=max_hf_download_retries,
        timeout_seconds=timeout_seconds,
        wandb_run_id=wandb_run_id,
        **kwargs,
    )
