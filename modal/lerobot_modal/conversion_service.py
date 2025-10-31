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
hf_cache_volume = modal.Volume.from_name("hf_cache", create_if_missing=True)


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
    hf_token: str | None = None,
) -> tuple[str | None, str | None]:
    """
    Convert a v2.1 dataset to LeRobot v3.0 format and upload to Hugging Face Hub.
    If a token is passed, we assume it has the necessary permissions to push to the dataset.
    If not, we reupload a version of the dataset on the phospho cloud.

    Returns:
        - dataset_name: The name of the dataset on Hugging Face Hub after conversion.
        - error_message: None if conversion was successful, otherwise an error message.
    """
    import os
    from loguru import logger
    from huggingface_hub import (
        snapshot_download,
        upload_folder,
        create_repo,
        create_tag,
        HfApi,
    )
    from lerobot.datasets.v30.convert_dataset_v21_to_v30 import convert_dataset

    try:
        if hf_token is not None:
            # We do this because LeRobot later uses HfApi internally which reads from env variables
            os.environ["HF_TOKEN"] = hf_token

        logger.info("Looking for version 3.0 of the dataset on the hub...")
        api = HfApi()
        tags = api.list_repo_refs(dataset_name, repo_type="dataset")

        branches = [branch.name for branch in tags.branches]

        if "v3.0" in branches:
            logger.info("Dataset already has a v3.0 version. No conversion needed.")
            return dataset_name, None

        if "v2.1" not in branches:
            error_msg = f"Dataset {dataset_name} is not a v2.1 dataset and cannot be converted to v3.0."
            logger.error(error_msg)
            return None, error_msg

        if hf_token is None and not dataset_name.startswith("phospho-app/"):
            # The dataset is a v2.1 dataset but not on our account.
            # In this case, we need to reupload the dataset on our account to have write permissions
            dataset_path_as_str = snapshot_download(
                repo_id=dataset_name,
                repo_type="dataset",
                revision="v2.1",
                cache_dir="/data/hf_cache",
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

        # We're about to proceed with the conversion.
        # Remove the cached dataset in /data/hf_cache/datasets/{dataset_name} if it exists
        # To avoid issues with the conversion process.
        dataset_path = f"/data/hf_cache/datasets/{dataset_name}"
        if os.path.exists(dataset_path):
            logger.info(f"Removing existing dataset path: {dataset_path}")
            os.system(f"rm -rf {dataset_path}")
        else:
            logger.debug(f"Dataset path does not exist: {dataset_path}")

        # Convert the dataset to v3.0 format.
        # This downloads the dataset from the hub and pushes it back to the hub.
        convert_dataset(repo_id=dataset_name)
        return dataset_name, None

    except Exception as e:
        logger.error(f"An error occurred during conversion: {e}")
        return None, str(e)
