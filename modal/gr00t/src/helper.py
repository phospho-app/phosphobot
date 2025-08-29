import os
import json
import modal
import wandb
import traceback
import sentry_sdk
from pathlib import Path

from loguru import logger
from huggingface_hub import HfApi

from phosphobot.am import Gr00tTrainer, Gr00tTrainerConfig
from phosphobot.am.base import (
    TrainingParamsGr00T,
    generate_readme,
)
from phosphobot.models import InfoModel

if os.getenv("MODAL_ENVIRONMENT") == "production":
    sentry_sdk.init(
        dsn="https://afa38885e368d772d8eced1bce325604@o4506399435325440.ingest.us.sentry.io/4509203019005952",
        traces_sample_rate=1.0,
        environment="production",
    )


def generate_modality_json(data_dir: Path) -> tuple[int, int]:
    # Load the metadata file to get image keys
    with open(data_dir / "meta" / "info.json", "r") as f:
        metadata = json.load(f)
        image_keys = []
        for key in metadata["features"].keys():
            if "image" in key:
                image_keys.append(key)

    number_of_cameras = len(image_keys)
    number_of_robots: int = metadata["features"]["action"]["shape"][0] // 6
    print(f"Number of cameras: {number_of_cameras}")
    print(f"Number of robots: {number_of_robots}")

    # Create the action/state keys based on the number of robots
    # Each robot has 5 arm keys and 1 gripper key
    robot_keys = []
    for i in range(number_of_robots):
        robot_keys.append(f"arm_{i}")

    # Create the action/state keys
    robot_structure = {}
    index = 0
    for key in robot_keys:
        robot_structure[key] = {"start": index, "end": index + 6}
        index += 6

    # Populate the video section with the image keys
    video_structure: dict = {}
    camera_name = [f"image_cam_{i}" for i in range(number_of_cameras)]
    for i, image_key in enumerate(image_keys):
        video_structure[camera_name[i]] = {"original_key": image_key}  # type: ignore

    # Create the base modality structure
    modality_json = {
        "state": robot_structure,
        "action": robot_structure,
        "video": video_structure,
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }

    print(f"Modality JSON: {modality_json}")

    # Write the modality.json file
    with open(data_dir / "meta" / "modality.json", "w") as f:
        json.dump(modality_json, f, indent=4)

    return number_of_robots, number_of_cameras


def train_gr00t_on_modal(
    dataset_repo_id: str,
    hf_token: str,
    hf_model_name: str,
    timeout_seconds: int,
    training_params: TrainingParamsGr00T,
    wandb_api_key: str | None = None,
    private_mode: bool = False,
):
    hf_api = HfApi(token=hf_token)

    try:
        # Set up Weights & Biases if API key is provided
        wandb_enabled = wandb_api_key is not None
        wandb_run_url = None

        if wandb_enabled:
            try:
                wandb.login(key=wandb_api_key, verify=True)
            except Exception as e:
                logger.info(
                    f"Failed to login to Weights & Biases: {e}. Disabling Weights & Biases."
                )
                wandb_enabled = False

        logger.info("Weights & Biases enabled:", wandb_enabled)

        training_params.output_dir = str(Path("/tmp/outputs/train"))
        training_params.data_dir = str(Path("/tmp/outputs/data/"))
        training_params.validation_data_dir = str(Path("/tmp/outputs/validation_data/"))

        # Download info.json file and determine appropriate batch size
        if training_params.batch_size is None:
            hf_api = HfApi(token=hf_token)
            info_file_path = hf_api.hf_hub_download(
                repo_id=dataset_repo_id,
                repo_type="dataset",
                filename="meta/info.json",
                force_download=True,
            )
            meta_folder_path = os.path.dirname(info_file_path)
            validated_info_model = InfoModel.from_json(
                meta_folder_path=meta_folder_path
            )
            number_of_cameras = (
                validated_info_model.total_videos // validated_info_model.total_episodes
            )
            # This is a heuristic to determine the batch size, it is set for an A100 GPU
            training_params.batch_size = max(
                110 // number_of_cameras - 3 * number_of_cameras, 1
            )
            logger.info(
                f"Batch size is None. Using automatic batch size of {training_params.batch_size}."
            )

        config = Gr00tTrainerConfig(
            dataset_name=dataset_repo_id,
            model_name=hf_model_name,
            wandb_api_key=wandb_api_key if wandb_enabled else None,
            training_params=training_params,
        )
        trainer = Gr00tTrainer(config)
        trainer.train(
            timeout_seconds=timeout_seconds,
            private_mode=private_mode,
            hf_token=hf_token,
        )

        # Upload model folder to Modal volume
        try:
            gr00t_volume = modal.Volume.from_name(
                "gr00t-n1", environment_name="production"
            )
            logger.info("Uploading model to Modal volume gr00t-n1")
            with gr00t_volume.batch_upload() as batch:
                batch.put_directory(
                    training_params.output_dir, f"/models/{hf_model_name}"
                )
            logger.info(
                f"Model uploaded to Modal volume gr00t-n1 at /models/{hf_model_name}"
            )
        except Exception as e:
            logger.error(f"Failed to upload model {hf_model_name} to Modal volume: {e}")

    except Exception as e:
        error_traceback = traceback.format_exc()
        if (
            "torch.OutOfMemoryError: CUDA out of memory." in str(e)
            or "torch.OutOfMemoryError: CUDA out of memory." in error_traceback
        ):
            error_traceback += "\n\nThe current batch size is too large for the GPU.\nPlease consider lowering it to fit in the memory.\nWe train on a 80GB A100 GPU."

        logger.error(f"Training failed: {e}")
        readme = generate_readme(
            model_type="gr00t",
            error_traceback=error_traceback,
            dataset_repo_id=dataset_repo_id,
            wandb_run_url=wandb_run_url,
            return_readme_as_bytes=True,
            training_params=training_params,
        )

        hf_api.upload_file(
            repo_type="model",
            path_or_fileobj=readme,
            path_in_repo="README.md",
            repo_id=hf_model_name,
            token=hf_token,
        )
        raise e

    finally:
        # Clean the .netrc file
        if os.path.exists("/root/.huggingface"):
            os.remove("/root/.huggingface")
        if os.path.exists("/root/.netrc"):
            os.remove("/root/.netrc")
