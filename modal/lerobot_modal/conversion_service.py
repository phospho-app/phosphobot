import modal


conversion_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "ffmpeg",
        "libavutil-dev",
        "libavcodec-dev",
        "libavformat-dev",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }
    )
    .uv_pip_install(
        "lerobot==0.4.0",  # This version includes v3.0 of LeRobotDataset
        "loguru",
        "huggingface_hub[cli]",
    )
)

conversion_service = modal.App("conversion-app")
hf_cache_volume = modal.Volume.from_name("datasets", create_if_missing=True)


@conversion_service.function(
    image=conversion_image,
    cpu=1,  # Single core is enough from experience
    timeout=15 * 60,
    secrets=[
        modal.Secret.from_dict({"MODAL_LOGLEVEL": "DEBUG"}),
        modal.Secret.from_name("huggingface"),
    ],
    volumes={"/data": hf_cache_volume},
)
async def convert_dataset_to_v3(
    dataset_name: str,
    huggingface_token: str | None = None,
) -> tuple[str | None, str | None]:
    """
    Convert a v2.1 dataset to LeRobot v3.0 format and upload to Hugging Face Hub.
    If a token is passed, we assume it has the necessary permissions to push to the dataset.
    If not, we reupload a version of the dataset on the phospho cloud.
    """
    import os
    from loguru import logger
    from huggingface_hub import (
        snapshot_download,
        upload_folder,
        create_repo,
        create_tag,
    )
    from lerobot.datasets.v30.convert_dataset_v21_to_v30 import convert_dataset

    try:
        # Clear Hugging Face cache
        os.system("huggingface-cli delete-cache -y")

        # We do this because LeRobot later uses HfApi internally which reads from env variables
        if huggingface_token is not None:
            os.environ["HF_TOKEN"] = huggingface_token
        else:
            if dataset_name.startswith("phospho-app/"):
                # Dataset is already on our account, no need to reupload
                pass
            try:
                logger.info(
                    "Trying to download v3.0 version of the dataset from the hub..."
                )
                snapshot_download(
                    dataset_name,
                    repo_type="dataset",
                    revision="v3.0",
                    cache_dir="/data/hf_cache",
                )
                return dataset_name, None
            except Exception:
                logger.warning(
                    "Dataset does not have an uploaded v3.0 version. Continuing with conversion."
                )
            # In this case, we need to reupload the dataset on our account to have write permissions
            dataset_path_as_str = snapshot_download(
                repo_id=dataset_name, repo_type="dataset", revision="v2.1"
            )
            new_repo = "phospho-app/" + dataset_name.split("/")[-1]
            create_repo(
                repo_id=new_repo,
                repo_type="dataset",
                exist_ok=True,
            )
            upload_folder(
                repo_id=new_repo,
                folder_path=dataset_path_as_str,
                repo_type="dataset",
            )
            create_tag(
                repo_id=new_repo,
                tag="v2.1",
                repo_type="dataset",
            )
            dataset_name = new_repo

        # Login to Hugging Face Hub
        convert_dataset(repo_id=dataset_name)  # Will also push to hub
        return dataset_name, None

    except Exception as e:
        logger.error(f"An error occurred during conversion: {e}")
        return None, str(e)
