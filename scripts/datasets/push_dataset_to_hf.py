import os
import sys
from loguru import logger
from huggingface_hub import HfApi, create_repo, upload_folder, create_branch


def push_dataset_to_hub(
    dataset_path: str, dataset_name: str, branch_path: str | None = None
):
    """
    Push a dataset to the Hugging Face Hub.

    Args:
        dataset_path (str): Path to the dataset folder
        dataset_name (str): Name of the dataset
        branch_path (str, optional): Additional branch to push to besides main
    """
    try:
        # Initialize HF API with token
        hf_api = HfApi(token=True)

        # Try to get username/org ID from token
        username_or_org_id = None
        try:
            # Get user info from token
            user_info = hf_api.whoami()
            username_or_org_id = user_info.get("name")

            if not username_or_org_id:
                logger.error("Could not get username or org ID from token")
                return

        except Exception as e:
            logger.error(f"Error getting user info: {e}")
            logger.warning(
                "No user or org with write access found. Won't be able to push to Hugging Face."
            )
            return

        # Create README if it doesn't exist
        readme_path = os.path.join(dataset_path, "README.md")
        if not os.path.exists(readme_path):
            with open(readme_path, "w") as readme_file:
                readme_file.write(
                    f"""
---
tags:
- phosphobot
- so100
- phospho-dk
task_categories:
- robotics                                                   
---

# {dataset_name}

**This dataset was generated using the [phospho cli](https://github.com/phospho-app/phosphobot)**

More information on [robots.phospho.ai](https://robots.phospho.ai).

This dataset contains a series of episodes recorded with a robot and multiple cameras. \
It can be directly used to train a policy using imitation learning. \
It's compatible with LeRobot and RLDS.
"""
                )

        # Construct full repo name
        dataset_repo_name = f"{username_or_org_id}/{dataset_name}"
        create_2_0_branch = False

        # Check if repo exists, create if it doesn't
        try:
            hf_api.repo_info(repo_id=dataset_repo_name, repo_type="dataset")
            logger.info(f"Repository {dataset_repo_name} already exists.")
        except Exception:
            logger.info(
                f"Repository {dataset_repo_name} does not exist. Creating it..."
            )
            create_repo(
                repo_id=dataset_repo_name,
                repo_type="dataset",
                exist_ok=True,
                token=True,
            )
            logger.info(f"Repository {dataset_repo_name} created.")
            create_2_0_branch = True

        # Push to main branch
        logger.info(
            f"Pushing the dataset to the main branch in repository {dataset_repo_name}"
        )
        upload_folder(
            folder_path=dataset_path,
            repo_id=dataset_repo_name,
            repo_type="dataset",
            token=True,
        )

        # Create and push to v2.0 branch if needed
        if create_2_0_branch:
            try:
                logger.info(f"Creating branch v2.0 for dataset {dataset_repo_name}")
                create_branch(
                    dataset_repo_name,
                    repo_type="dataset",
                    branch="v2.0",
                    token=True,
                )
                logger.info(f"Branch v2.0 created for dataset {dataset_repo_name}")

                # Push to v2.0 branch
                logger.info(
                    f"Pushing the dataset to the branch v2.0 in repository {dataset_repo_name}"
                )
                upload_folder(
                    folder_path=dataset_path,
                    repo_id=dataset_repo_name,
                    repo_type="dataset",
                    token=True,
                    revision="v2.0",
                )
            except Exception as e:
                logger.error(f"Error handling v2.0 branch: {e}")

        # Push to additional branch if specified
        if branch_path:
            try:
                logger.info(
                    f"Creating branch {branch_path} for dataset {dataset_repo_name}"
                )
                create_branch(
                    dataset_repo_name,
                    repo_type="dataset",
                    branch=branch_path,
                    token=True,
                )
                logger.info(
                    f"Branch {branch_path} created for dataset {dataset_repo_name}"
                )

                # Push to specified branch
                logger.info(f"Pushing the dataset to branch {branch_path}")
                upload_folder(
                    folder_path=dataset_path,
                    repo_id=dataset_repo_name,
                    repo_type="dataset",
                    token=True,
                    revision=branch_path,
                )
                logger.info(f"Dataset pushed to branch {branch_path}")
            except Exception as e:
                logger.error(f"Error handling custom branch: {e}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: uv run push_dataset_to_hf.py <dataset_path> <dataset_name>")
        sys.exit(1)

    DATASET_PATH = sys.argv[1]
    DATASET_NAME = sys.argv[2]

    push_dataset_to_hub(dataset_path=DATASET_PATH, dataset_name=DATASET_NAME)
