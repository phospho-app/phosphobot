import modal
import random

conversion_image_v2 = (
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
        "lerobot==0.3.3",  # This version includes v2.1 of LeRobotDataset
        "loguru",
        "huggingface_hub[cli]",
    )
)

conversion_service_from_v2 = modal.App("conversion_app_from_v2")
hf_cache_volume = modal.Volume.from_name("hf_cache", create_if_missing=True)


@conversion_service_from_v2.function(
    image=conversion_image_v2,
    cpu=1,  # Single core is enough from experience
    timeout=15 * 60,
    secrets=[
        modal.Secret.from_dict({"MODAL_LOGLEVEL": "DEBUG"}),
        modal.Secret.from_name("huggingface"),
    ],
    volumes={"/data": hf_cache_volume},
)
async def convert_dataset_to_v21(
    dataset_name: str,
    huggingface_token: str | None = None,
) -> tuple[str | None, str | None]:
    """
    Convert a v2.0 dataset to LeRobot v2.1 format and upload to Hugging Face Hub.
    If we have write permissions, we update the dataset in place.
    If not, we copy it to the phospho-app org and update it there.
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
    from lerobot.datasets.v21.convert_dataset_v20_to_v21 import convert_dataset

    try:
        # Remove the cached dataset in /data/hf_cache/datasets/{dataset_name} if it exists
        dataset_path = f"/data/hf_cache/datasets/{dataset_name}"
        if os.path.exists(dataset_path):
            logger.info(f"Removing existing dataset path: {dataset_path}")
            os.system(f"rm -rf {dataset_path}")
        else:
            logger.debug(f"Dataset path does not exist: {dataset_path}")

        api = HfApi(
            token=huggingface_token if huggingface_token else os.environ["HF_TOKEN"]
        )

        # Check if we have write access on the dataset
        has_write_permissions = False
        try:
            api.repo_info(dataset_name, repo_type="dataset")
            user_info = api.whoami()
            repo_owner = (
                dataset_name.split("/")[0] if "/" in dataset_name else user_info["name"]
            )

            # Check if this is a fine-grained token
            if (
                user_info.get("auth", {}).get("accessToken", {}).get("role")
                == "fineGrained"
            ):
                logger.info("Using fine-grained token, checking scoped permissions...")
                fine_grained = user_info["auth"]["accessToken"]["fineGrained"]
                scoped_permissions = fine_grained.get("scoped", [])

                # Check if we have write permissions for this specific entity (user or org)
                for scope in scoped_permissions:
                    entity = scope.get("entity", {})
                    if entity.get("name") == repo_owner:
                        permissions = scope.get("permissions", [])
                        if "repo.write" in permissions:
                            has_write_permissions = True
                            logger.info(f"Found write permissions for {repo_owner}")
                            break
                        else:
                            logger.info(
                                f"No write permissions for {repo_owner}, only: {permissions}"
                            )
            else:
                # For classic tokens, check if repo owner is in user's orgs
                user_orgs = [user_info["name"]] + [
                    org["name"] for org in user_info.get("orgs", [])
                ]
                has_write_permissions = repo_owner in user_orgs

            logger.info(
                f"Write permissions for {dataset_name}: {has_write_permissions}"
            )
        except Exception as e:
            logger.warning(f"Could not verify write permissions: {e}")
            has_write_permissions = False

        # Now check if conversion is needed
        tags = api.list_repo_refs(dataset_name, repo_type="dataset")
        branches = [branch.name for branch in tags.branches]

        if "v2.1" in branches:
            logger.info("Dataset already has a v2.1 version. No conversion needed.")
            return dataset_name, None
        elif "v2.0" not in branches:
            error_msg = (
                f"Dataset {dataset_name} does not have a v2.0 version to convert from."
            )
            logger.error(error_msg)
            return None, error_msg

        # If no write permissions, copy to phospho-app org
        if not has_write_permissions and not dataset_name.startswith("phospho-app/"):
            logger.info(
                f"No write permissions for {dataset_name}. Copying to phospho-app org..."
            )

            # Download the v2.0 version
            dataset_path_as_str = snapshot_download(
                repo_id=dataset_name,
                repo_type="dataset",
                revision="v2.0",
                cache_dir="/data/hf_cache",
            )

            # Create new repo in phospho-app org
            new_repo = (
                "phospho-app/"
                + dataset_name.split("/")[-1]
                + "-"
                + "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=4))
            )
            create_repo(
                repo_id=new_repo,
                repo_type="dataset",
                exist_ok=True,
            )

            # Upload to new repo
            upload_folder(
                repo_id=new_repo,
                folder_path=dataset_path_as_str,
                repo_type="dataset",
            )

            # Create v2.0 tag
            create_tag(
                repo_id=new_repo,
                tag="v2.0",
                repo_type="dataset",
            )

            dataset_name = new_repo
            logger.info(f"Dataset copied to {new_repo}")

        # We do this because LeRobot later uses HfApi internally which reads from env variables
        if huggingface_token is not None:
            os.environ["HF_TOKEN"] = huggingface_token

        # Log in info
        logger.info(f"Logged in HF as: {api.whoami()}")

        # Convert and push
        convert_dataset(repo_id=dataset_name, branch="v2.0")  # Will also push to hub

        logger.info(f"Dataset {dataset_name} successfully converted to v2.1")

        return dataset_name, None

    except Exception as e:
        logger.error(f"An error occurred during conversion: {e}")
        return None, str(e)
