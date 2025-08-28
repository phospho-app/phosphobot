import os
import cv2
import random
import base64
import traceback
from pathlib import Path, PurePath
from typing import Literal, cast

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from huggingface_hub import HfApi
from loguru import logger

from phosphobot.am.base import ActionModel, TrainingRequest
from phosphobot.configs import config
from phosphobot.models import (
    AdminSettingsRequest,
    AdminSettingsResponse,
    AdminSettingsTokenResponse,
    BrowseFilesResponse,
    BrowserFilesRequest,
    BaseDataset,
    DatasetListResponse,
    DatasetRepairRequest,
    DatasetShuffleRequest,
    DatasetSplitRequest,
    DeleteEpisodeRequest,
    HFDownloadDatasetRequest,
    HuggingFaceTokenRequest,
    InfoResponse,
    MergeDatasetsRequest,
    ItemInfo,
    ModelConfigurationRequest,
    ModelConfigurationResponse,
    StatusResponse,
    TrainingInfoRequest,
    TrainingInfoResponse,
    VizSettingsResponse,
    WandBTokenRequest,
    InfoModel,
)
from phosphobot.models import EpisodesModel, LeRobotDataset
from phosphobot.utils import (
    get_hf_token,
    get_resources_path,
    get_home_app_path,
    is_running_on_pi,
    login_to_hf,
    parse_hf_username_or_orgid,
    sanitize_path,
    zip_folder,
)

router = APIRouter(tags=["pages"])

# Root directory for the file browser
ROOT_DIR = str(get_home_app_path() / "recordings")
INDEX_PATH = get_resources_path() / "dist" / "index.html"


# Optionally, if you want the dashboard to be served at the root endpoint:
@router.get("/auth", response_class=HTMLResponse)
@router.get("/sign-in", response_class=HTMLResponse)
@router.get("/sign-up", response_class=HTMLResponse)
@router.get("/sign-up/confirm", response_class=HTMLResponse)
@router.get("/auth/confirm", response_class=HTMLResponse)
@router.get("/auth/forgot-password", response_class=HTMLResponse)
@router.get("/auth/reset-password", response_class=HTMLResponse)
@router.get("/calibration", response_class=HTMLResponse)
@router.get("/admin", response_class=HTMLResponse)
@router.get("/viz", response_class=HTMLResponse)
@router.get("/network", response_class=HTMLResponse)
@router.get("/browse", response_class=HTMLResponse)
@router.get("/control", response_class=HTMLResponse)
@router.get("/train", response_class=HTMLResponse)
@router.get("/inference", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    with open(INDEX_PATH.resolve(), "r") as f:
        content = f.read()
    return HTMLResponse(
        headers={"Content-Type": "text/html; charset=utf-8"}, content=content
    )


@router.get("/admin/settings", response_model=AdminSettingsResponse)
async def get_admin_settings():
    return AdminSettingsResponse(
        dataset_name=config.DEFAULT_DATASET_NAME,
        freq=config.DEFAULT_FREQ,
        episode_format=config.DEFAULT_EPISODE_FORMAT,
        video_codec=config.DEFAULT_VIDEO_CODEC,
        video_size=config.DEFAULT_VIDEO_SIZE,
        task_instruction=config.DEFAULT_TASK_INSTRUCTION,
        cameras_to_record=config.DEFAULT_CAMERAS_TO_RECORD,
        hf_private_mode=config.DEFAULT_HF_PRIVATE_MODE,
    )


@router.post("/admin/settings/tokens", response_model=AdminSettingsTokenResponse)
async def get_admin_settings_token():
    # Return bool if the token is set and valid
    return AdminSettingsTokenResponse(
        huggingface=login_to_hf(revalidate=False),
        wandb=os.path.exists(str(get_home_app_path()) + "/wandb.token"),
    )


@router.get("/viz/settings", response_model=VizSettingsResponse)
def get_viz_settings(request: Request):
    """
    Page with an overview of the connected cameras. Open this page in the chrome browser.
    """
    # Downgrade the stream quality on raspberry pi
    if is_running_on_pi():
        quality = 8
        target_size = (320, 240)
    else:
        # We compress the video slightly but it should not be noticeable
        quality = 60
        target_size = (320, 240)

    return VizSettingsResponse(
        width=target_size[0], height=target_size[1], quality=quality
    )


def list_directory_items(path: str, root_dir: str = "") -> list[ItemInfo]:
    full_path = os.path.join(root_dir, path)
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail=f"Path not found: {full_path}")

    # Remove DS_Store files if they exist
    BaseDataset.remove_ds_store_files(full_path)

    items = os.listdir(full_path)
    items_info = []
    username_or_org_id = None
    api = None
    if path.endswith("lerobot_v2") or path.endswith("lerobot_v2.1"):
        try:
            api = HfApi()
            user_info = api.whoami()
            username_or_org_id = parse_hf_username_or_orgid(user_info)
        # If we can't get the username or org ID, we can't delete the dataset
        except Exception as e:
            logger.debug(f"Error getting Hugging Face username or org ID: {str(e)}")
            pass

    for item in items:
        item_path = os.path.join(path, item)
        absolute_item_path = os.path.join(full_path, item_path)
        is_dir = os.path.isdir(os.path.join(root_dir, item_path))
        info = ItemInfo(
            name=item,
            path=item_path,
            absolute_path=absolute_item_path,
            is_dir=is_dir,
            browseUrl=f"/browse?path={item_path}",
            downloadUrl=f"/dataset/download?folder_path={item_path}"
            if is_dir
            else None,
        )
        if is_dir:
            if api is not None and username_or_org_id is not None:
                info.previewUrl = f"https://lerobot-visualize-dataset.hf.space/{username_or_org_id}/{info.name}"
                info.huggingfaceUrl = (
                    f"https://huggingface.co/datasets/{username_or_org_id}/{info.name}"
                )

            # Only add the delete button if the dataset's path ends with "json" or "lerobot_v2"
            if (
                path.endswith("json")
                or path.endswith("lerobot_v2")
                or path.endswith("lerobot_v2.1")
            ):
                info.canDeleteDataset = True
                info.deleteDatasetAction = "/dataset/delete"

        # Check if this is a dataset: it's a directory and the parent is lerobot_v2
        if is_dir and (path.endswith("lerobot_v2") or path.endswith("lerobot_v2.1")):
            info.is_dataset_dir = True

        items_info.append(info)
    items_info.sort(key=lambda x: x.name.lower())
    return items_info


@router.post("/files", response_model=BrowseFilesResponse)
async def files(query: BrowserFilesRequest):
    # Record page view
    safe_path = sanitize_path(query.path)
    root = Path(ROOT_DIR)
    full_path = os.path.join(ROOT_DIR, safe_path)

    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="Path not found")

    if os.path.isfile(full_path):
        return FileResponse(full_path)

    items_info = list_directory_items(path=safe_path, root_dir=ROOT_DIR)

    # Prepare episode data if in a dataset structure
    episode_paths = []
    episode_ids = []

    parts = PurePath(safe_path).parts

    if len(parts) > 1 and parts[-2] in {"json", "lerobot_v2", "lerobot_v2.1"}:
        data_dir = root / safe_path / "data"
        # look for chunk folders
        chunk0 = data_dir / "chunk-000"
        if chunk0.exists():
            # gather all parquet files
            for p in sorted(data_dir.glob("chunk-*/**/*.parquet")):
                # basename like "whatever_123.parquet"
                episode_paths.append(str(p))
                # extract the numeric ID
                stem = p.stem  # e.g. "whatever_123"
                try:
                    ep_id = int(stem.split("_")[-1])
                except ValueError:
                    continue
                episode_ids.append(ep_id)
            # sort descending for IDs
            episode_ids.sort(reverse=True)

    response_data = BrowseFilesResponse(
        directoryTitle=f"Directory: {safe_path}",
        items=items_info,
        episode_paths=episode_paths,
        episode_ids=episode_ids,
    )

    # Optionally add a tokenError if needed
    # response_data.tokenError = "Invalid Hugging Face token. Please update your token in the Admin page."
    return response_data


@router.post("/admin/huggingface/whoami")
async def whoami():
    # Fetch the Hugging Face token and try to run whoami to return the person id
    token_path = str(get_home_app_path()) + "/huggingface.token"
    if not os.path.exists(token_path):
        return {
            "status": "error",
        }
    try:
        with open(token_path, "r") as token_file:
            token = token_file.read().strip()
        api = HfApi(token=token)
        user_info = api.whoami(token=token)
        # Get the username or org that has write access
        username_or_orgid = parse_hf_username_or_orgid(user_info)
        logger.info(f"Token is valid for {username_or_orgid}")
        if username_or_orgid is None:
            return {
                "status": "error",
            }
        return {
            "status": "success",
            "username": username_or_orgid,
        }
    except Exception:
        return {
            "status": "error",
        }


@router.post("/admin/huggingface")
async def submit_token(query: HuggingFaceTokenRequest):
    if query.token == "":
        # Delete the token file if the token is empty
        file_path = str(get_home_app_path()) + "/huggingface.token"
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                return {
                    "status": "success",
                    "message": "The file huggingface.token has been removed.",
                }
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Error removing Hugging Face token: {str(e)}",
                }

    # Try first to connect to HuggingFace with the token
    try:
        api = HfApi(token=query.token)
        user_info = api.whoami(token=query.token)
        # Get the username or org that has write access
        username_or_orgid = parse_hf_username_or_orgid(user_info)
        if username_or_orgid is None:
            return {
                "status": "error",
                "message": "The token does not have write access to any repository. Please add 'Write access to content/settings' in the token scope.",
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error connecting to Hugging Face: {str(e)}",
        }

    # Define the file path where the token will be saved
    file_path = str(get_home_app_path()) + "/huggingface.token"

    try:
        # Open the file in write mode and save the token
        with open(file_path, "w") as token_file:
            token_file.write(query.token)
        # Set secure file permissions (optional, but recommended)
        os.chmod(file_path, 0o600)  # Read and write for owner only
        # Change config
        logger.info("Token saved successfully!")
        # Auth again to HuggingFace
        login_to_hf()

        return {"status": "success", "message": "Token saved successfully!"}

    except Exception as e:
        logger.error(f"Error saving token: {str(e)}")
        return {"status": "error", "message": f"Error saving token: {str(e)}"}


@router.post("/admin/wandb")
async def submit_wandb_token(query: WandBTokenRequest):
    # For now, we don't perform any check on the token
    # TODO: make sure an invalid token won't crash the training process

    # Define the file path where the token will be saved
    file_path = str(get_home_app_path()) + "/wandb.token"

    if query.token == "":
        # Delete the token file if the token is empty
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                return {
                    "status": "success",
                    "message": "The file wandb.token has been removed.",
                }
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Error removing WandB token: {str(e)}",
                }

    if len(query.token) != 40:
        return {
            "status": "error",
            "message": "Wrong token, make sure to copy/paste the 40 characters long token at https://wandb.ai/authorize",
        }

    try:
        # Open the file in write mode and save the token
        with open(file_path, "w") as token_file:
            token_file.write(query.token)
        # Set secure file permissions (optional, but recommended)
        os.chmod(file_path, 0o600)  # Read and write for owner only
        # Change config
        logger.info("Token saved successfully!")

        return {"status": "success", "message": "Token saved successfully!"}

    except Exception as e:
        logger.error(f"Error saving token: {str(e)}")
        return {
            "status": "error",
            "message": f"Error saving token: {str(e)}",
        }


@router.post("/admin/form/usersettings")
async def submit_user_settings(user_settings: AdminSettingsRequest):
    try:
        logger.debug(f"User Settings submission: {user_settings.model_dump()}")
        config.save_user_settings(user_settings.model_dump())

        return {"status": "success", "message": "User settings saved successfully!"}
    except Exception as e:
        logger.error(f"User Settings submission error: {e}")
        return {"status": "error", "message": f"Error saving user settings: {str(e)}"}


@router.post("/dataset/delete")
async def delete_dataset(request: Request, path: str):
    dataset_path = os.path.join(ROOT_DIR, path)
    # Check if the path exists and is a directory
    if not os.path.exists(dataset_path) or not os.path.isdir(dataset_path):
        raise HTTPException(status_code=404, detail=f"Dataset {path} not found")

    # Test that the path contains "lerobot" or "json" to prevent accidental deletion
    if "json" not in path and "lerobot_v2" not in path and "lerobot_v2.1" not in path:
        raise HTTPException(
            status_code=400,
            detail="Invalid dataset path. Please use the delete button provided in the admin page to delete dataset.",
        )

    dataset = BaseDataset(path=dataset_path)
    dataset.delete()

    return StatusResponse(status="ok")


@router.post("/dataset/info")
async def get_dataset_info(path: str) -> InfoResponse:
    """
    Get the dataset keys and frames.
    """
    dataset_path = os.path.join(ROOT_DIR, path)
    # Check if the path exists and is a directory
    if not os.path.exists(dataset_path) or not os.path.isdir(dataset_path):
        raise HTTPException(status_code=404, detail=f"Dataset {path} not found")

    # Test that the path contains "lerobot" or "json" to prevent accidental deletion
    if "lerobot_v2" not in path and "lerobot_v2.1" not in path:
        raise HTTPException(
            status_code=400,
            detail="Invalid dataset path. Please use the delete button provided in the admin page to delete dataset.",
        )

    meta_folder_path = os.path.join(
        dataset_path,
        "meta",
    )

    try:
        info = InfoModel.from_json(
            meta_folder_path=meta_folder_path,
            format=cast(
                Literal["lerobot_v2", "lerobot_v2.1"],
                Path(path).parts[0],
            ),
        )
    except Exception as e:
        logger.warning(f"Error loading dataset info: {e}")
        return InfoResponse(
            status="error",
        )

    image_frames = {}
    for key in info.features.observation_images.keys():
        # Extract first frame from first video file
        video_path = os.path.join(
            dataset_path, "videos", "chunk-000", key, "episode_000000.mp4"
        )
        if os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    # Resize the frame to a smaller size
                    frame = cv2.resize(frame, (320, 240))
                    # Convert the frame to bytes
                    _, buffer = cv2.imencode(".jpg", frame)
                    image_frames[key] = base64.b64encode(buffer.tobytes()).decode(
                        "utf-8"
                    )
                cap.release()
        else:
            logger.warning(f"Video file {video_path} not found.")
            return InfoResponse(status="error")

    return InfoResponse(
        status="ok",
        robot_type=info.robot_type,
        robot_dof=info.features.observation_state.shape[0],
        number_of_episodes=info.total_episodes,
        image_keys=list(info.features.observation_images.keys()),
        image_frames=image_frames,
    )


@router.post("/dataset/merge")
async def merge_datasets(merge_request: MergeDatasetsRequest):
    """
    Merge two datasets into one.
    """
    # Validation
    # 1 - Check that the datasets are of the same type, v2.1
    # 2 - Check that the datasets are not empty
    # 3 - Check that the datasets have the same number of cameras and same robots

    # 1
    # Use Path to extract the first part of the path
    first_datatype = Path(merge_request.first_dataset).parts[0]
    second_datatype = Path(merge_request.second_dataset).parts[0]

    if first_datatype != second_datatype or first_datatype not in [
        "lerobot_v2.1",
    ]:
        raise HTTPException(
            status_code=400,
            detail="You can only merge datasets of type v2.1",
        )

    # 2
    first_dataset_path = os.path.join(ROOT_DIR, merge_request.first_dataset)
    second_dataset_path = os.path.join(ROOT_DIR, merge_request.second_dataset)
    if not os.path.exists(first_dataset_path) or not os.path.isdir(first_dataset_path):
        raise HTTPException(
            status_code=404,
            detail=f"Dataset {merge_request.first_dataset} not found",
        )
    if not os.path.exists(second_dataset_path) or not os.path.isdir(
        second_dataset_path
    ):
        raise HTTPException(
            status_code=404,
            detail=f"Dataset {merge_request.second_dataset} not found",
        )

    # 5
    first_info = InfoModel.from_json(
        meta_folder_path=os.path.join(ROOT_DIR, merge_request.first_dataset, "meta"),
        format=cast(Literal["lerobot_v2", "lerobot_v2.1"], first_datatype),
    )
    second_info = InfoModel.from_json(
        meta_folder_path=os.path.join(ROOT_DIR, merge_request.second_dataset, "meta"),
        format=cast(Literal["lerobot_v2", "lerobot_v2.1"], second_datatype),
    )
    if first_info.total_videos == 0 or second_info.total_videos == 0:
        raise HTTPException(
            status_code=400,
            detail="One of the datasets is empty. Please make sure the datasets are not empty.",
        )

    # Check video sizes
    for image_key in first_info.features.observation_images.keys():
        if (
            first_info.features.observation_images[image_key].shape
            != second_info.features.observation_images[
                merge_request.image_key_mappings[image_key]
            ].shape
        ):
            raise HTTPException(
                status_code=400,
                detail="The datasets have different video sizes.",
            )

    if first_info.fps != second_info.fps:
        raise HTTPException(
            status_code=400,
            detail="The datasets have different FPS.",
        )

    if (
        first_info.robot_type != second_info.robot_type
        or first_info.codebase_version != second_info.codebase_version
        or first_info.total_videos // first_info.total_episodes
        != second_info.total_videos // second_info.total_episodes
        or first_info.features.observation_state.shape[0]
        != second_info.features.observation_state.shape[0]
    ):
        raise HTTPException(
            status_code=400,
            detail="The datasets have different number of cameras or robots.",
        )

    initial_dataset = LeRobotDataset(
        path=os.path.join(ROOT_DIR, merge_request.first_dataset)
    )
    second_dataset = LeRobotDataset(
        path=os.path.join(ROOT_DIR, merge_request.second_dataset)
    )
    try:
        initial_dataset.merge_datasets(
            second_dataset=second_dataset,
            new_dataset_name=merge_request.new_dataset_name,
            video_transform=merge_request.image_key_mappings,
        )
    # if the dataset already exists, we raise an error
    except FileExistsError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset {merge_request.new_dataset_name} already exists: {e}",
        )
    except Exception as e:
        logger.error(f"Error merging datasets: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error merging datasets: {e}",
        )

    return StatusResponse(status="ok")


@router.post("/dataset/list", response_model=DatasetListResponse)
async def list_datasets():
    """
    List all datasets that are both in Hugging Face and locally.
    """
    root_v2 = os.path.join(ROOT_DIR, "lerobot_v2")
    root_v2_1 = os.path.join(ROOT_DIR, "lerobot_v2.1")

    # List all folders in the root directory. Keep only the directory names
    datasets_folders = []
    if os.path.exists(root_v2) and os.path.isdir(root_v2):
        datasets_folders += [
            f for f in os.listdir(root_v2) if os.path.isdir(os.path.join(root_v2, f))
        ]
    if os.path.exists(root_v2_1) and os.path.isdir(root_v2_1):
        datasets_folders += [
            f
            for f in os.listdir(root_v2_1)
            if os.path.isdir(os.path.join(root_v2_1, f))
        ]

    # Keep only directories
    try:
        api = HfApi()
        user_info = api.whoami()
        # Get the username or org that has write access
        username_or_orgid = parse_hf_username_or_orgid(user_info)
        # List HF datasets
        hf_datasets = list(
            api.list_datasets(author=username_or_orgid, limit=100, gated=False)
        )
        # Filter datasets that are in the local folder
        pushed_datasets = []
        for dataset in hf_datasets:
            only_name = dataset.id.split("/")[-1]
            if only_name in datasets_folders:
                pushed_datasets.append(dataset.id)

    except Exception:
        logger.info("No Hugging Face token found.")
        pushed_datasets = []

    return DatasetListResponse(
        pushed_datasets=pushed_datasets,
        local_datasets=datasets_folders,
    )


@router.post("/episode/delete")
async def delete_episode(query: DeleteEpisodeRequest):
    """
    Delete an episode from the dataset.
    Parameters:
    - episode_id: int: The episode ID to delete.
    - path: str: The path to the dataset folder.
    """

    logger.info(f"Deleting episode {query.episode_id} from {query.path}")

    try:
        dataset = LeRobotDataset(path=os.path.join(ROOT_DIR, query.path))
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail="The dataset was not found or the dataset name is incorrect",
        )

    if "lerobot_v2/" in query.path:
        # The stats model delete_episode method is not implemented, will probably never be
        raise HTTPException(
            status_code=400,
            detail="This feature is not available for v2 datasets. Please use the v2.1 dataset format.",
        )

    # Delete the data file
    dataset.delete_episode(episode_id=query.episode_id, update_hub=True)
    return StatusResponse(status="ok")


@router.post("/dataset/sync")
async def sync_dataset(path: str):
    # Extract dataset name et huggingface repo id from the path
    dataset = BaseDataset(path=os.path.join(ROOT_DIR, path))

    dataset.sync_local_to_hub()

    # Redirect to the parent directory after deletion
    return StatusResponse(status="ok")


@router.get("/dataset/download")
async def download_folder(folder_path: str):
    # Construct the full path
    full_path = os.path.join(ROOT_DIR, folder_path)

    # Check if the path exists and is a directory
    if not os.path.exists(full_path) or not os.path.isdir(full_path):
        raise HTTPException(status_code=404, detail="Folder not found")

    # Create a temporary ZIP file
    zip_path = f"{full_path}.zip"
    zip_folder(full_path, zip_path)

    # Return the ZIP file as a response
    return FileResponse(
        zip_path, filename=os.path.basename(zip_path), media_type="application/zip"
    )


@router.post("/model/configuration", response_model=ModelConfigurationResponse)
async def get_model_configuration(
    request: ModelConfigurationRequest,
) -> ModelConfigurationResponse:
    """
    Fetch the model info from Hugging Face and return its configuration.
    """
    request.model_id = request.model_id.strip()

    from phosphobot.am import ACT, Gr00tN1

    model_type_to_class: dict[str, type[ActionModel]] = {
        "gr00t": Gr00tN1,
        "ACT": ACT,
        "ACT_BBOX": ACT,  # ACT_BBOX is a variant of ACT
    }
    model_class = model_type_to_class.get(request.model_type)
    if model_class is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model type: {request.model_type}",
        )

    try:
        config = model_class.fetch_and_get_configuration(model_id=request.model_id)
        return ModelConfigurationResponse(
            video_keys=config.video_keys, checkpoints=config.checkpoints
        )

    except Exception as e:
        logger.warning(f"No video keys found for {request.model_id}: {e}")
        return ModelConfigurationResponse(video_keys=[], checkpoints=[])


@router.post("/training/info", response_model=TrainingInfoResponse)
async def get_training_info(
    request: TrainingInfoRequest,
) -> TrainingInfoResponse:
    """
    - Fetch the info.json from the model repo and return the training info.
    - If the model type is "custom", return a custom command to run the training.
    """
    if request.model_type == "custom":
        return TrainingInfoResponse(
            status="ok",
            training_body={
                "custom_command": "python absolute/path/to/file.py --epochs 10"
            },
        )

    if request.model_id is None or request.model_id.strip() == "":
        raise HTTPException(
            status_code=400,
            detail="Model ID is required to fetch training info.",
        )
    logger.debug(
        f"Fetching training info for model {request.model_id} of type {request.model_type}"
    )

    try:
        token_path = str(get_home_app_path()) + "/huggingface.token"

        if not os.path.exists(token_path):
            raise HTTPException(
                status_code=400,
                detail="Hugging Face token not found. Please set the token in the Admin page.",
            )

        with open(token_path, "r") as token_file:
            token = token_file.read().strip()
        api = HfApi(token=token)
        user_info = api.whoami(token=token)
        username_or_orgid = parse_hf_username_or_orgid(user_info)

        model_info = api.hf_hub_download(
            repo_id=request.model_id,
            repo_type="dataset",
            filename="meta/info.json",
            token=token,
        )

        meta_folder_path = os.path.dirname(model_info)
        validated_info = InfoModel.from_json(meta_folder_path=meta_folder_path)

        number_of_cameras = validated_info.total_videos // validated_info.total_episodes
        training_params = {}
        if number_of_cameras > 0:
            if request.model_type == "gr00t":
                training_params["batch_size"] = (
                    110 // number_of_cameras - 3 * number_of_cameras
                )
            elif request.model_type == "ACT":
                training_params["batch_size"] = 120 // number_of_cameras
                training_params["steps"] = 8_000
            elif request.model_type == "ACT_BBOX":
                training_params["batch_size"] = 100
                training_params["steps"] = 10_000

        # These are heuristics used to determine the training parameters
        random_suffix = "".join(
            random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=5)
        )

        # Determine private mode from config
        private_mode = config.DEFAULT_HF_PRIVATE_MODE

        # Set model name based on private mode
        if private_mode:
            # Use user's namespace for private training
            model_name = f"{username_or_orgid}/{request.model_type}-{request.model_id.split('/')[1]}-{random_suffix}"
        else:
            # Use phospho-app namespace for public training
            model_name = f"phospho-app/{username_or_orgid}-{request.model_type}-{request.model_id.split('/')[1]}-{random_suffix}"

        training_response = TrainingRequest(
            model_type=request.model_type,
            dataset_name=request.model_id,
            model_name=model_name,
            private_mode=private_mode,
            user_hf_token=None,
        )
        # Replace the fields in training_response with the values from training_params dict
        if training_response.training_params is not None:
            for key, value in training_params.items():
                if hasattr(training_response.training_params, key):
                    setattr(training_response.training_params, key, value)

        # Prepare the training body with masked token
        training_body = training_response.model_dump(
            exclude={
                "wandb_api_key": True,
                "custom_command": True,
                "training_params": {
                    "path_to_gr00t_repo": True,
                    "data_dir": True,
                    "output_dir": True,
                },
            }
        )

        return TrainingInfoResponse(
            status="ok",
            training_body=training_body,
        )
    except Exception as e:
        logger.warning(f"Error fetching training info: {e}")
        return TrainingInfoResponse(
            status="error",
            message=f"Error fetching training info: {e}",
        )


@router.post("/dataset/hf_download")
async def hf_download_dataset(
    query: HFDownloadDatasetRequest,
) -> StatusResponse:
    if os.path.exists(os.path.join(ROOT_DIR, "lerobot_v2.1", query.dataset_name)):
        return StatusResponse(
            status="error",
            message=f"Dataset {query.dataset_name} already exists.",
        )

    token = get_hf_token()
    if token is None:
        return StatusResponse(
            status="error",
            message="Hugging Face token not found. Please set the token in the Admin page.",
        )

    # Check if the dataset exists
    try:
        api = HfApi(token=token)
        info_file_path = api.hf_hub_download(
            repo_id=query.dataset_name,
            repo_type="dataset",
            filename="meta/info.json",
            force_download=True,
        )
        validated_info_model = InfoModel.from_json(
            meta_folder_path=os.path.dirname(info_file_path)
        )
        if validated_info_model.codebase_version != "v2.1":
            # Do not allow downloading a dataset that is not in the v2.1 format
            # This is to prevent issues loading the stats file if the dataset comes from lerobot
            # As sum and square_sum will not be present
            return StatusResponse(
                status="error",
                message=(
                    f"Dataset {query.dataset_name} is not in v2.1 format and is not compatible with this version of the app."
                ),
            )

        dataset_name = query.dataset_name.split("/")[-1]

        api.snapshot_download(
            repo_id=query.dataset_name,
            repo_type="dataset",
            local_dir=os.path.join(ROOT_DIR, "lerobot_v2.1", dataset_name),
            ignore_patterns=[".git", ".gitignore"],
            force_download=True,
        )

        return StatusResponse(
            status="ok",
        )
    except Exception:
        error_message = traceback.format_exc()
        # When HF fails, it just fails with read error log
        # So we check it and return a more user friendly error
        if "404 Client Error" in error_message:
            return StatusResponse(
                status="error",
                message=f"Dataset {query.dataset_name} not found on Hugging Face.",
            )
        else:
            logger.warning(
                f"Error downloading dataset {query.dataset_name}: {error_message}"
            )
            return StatusResponse(
                status="error",
                message="Error downloading dataset, please check the logs",
            )


@router.post("/dataset/repair", response_model=StatusResponse)
async def repair_dataset(query: DatasetRepairRequest):
    """
    Repair a dataset by removing any corrupted files.
    For now, this only works for parquets files.
    If the parquets are wrongly indexed, it will not do anything.
    """
    dataset_path = os.path.join(ROOT_DIR, query.dataset_path)
    # Check if the path exists and is a directory
    if not os.path.exists(dataset_path) or not os.path.isdir(dataset_path):
        raise HTTPException(
            status_code=404, detail=f"Dataset {query.dataset_path} not found"
        )

    # For the moment, we repair parquets files only, we need to improve this function to recaculate the meta files as well

    result = EpisodesModel.repair_parquets(
        parquets_path=os.path.join(dataset_path, "data", "chunk-000"),
    )

    if result:
        return StatusResponse(status="ok")
    else:
        return StatusResponse(
            status="error", message="Please check the logs for more details."
        )


@router.post("/dataset/split", response_model=StatusResponse)
async def split_dataset(query: DatasetSplitRequest):
    """
    Split a dataset into two datasets.
    Used for creating training and validation datasets.
    """
    dataset_path = os.path.join(ROOT_DIR, query.dataset_path)
    # Check if the path exists and is a directory
    if not os.path.exists(dataset_path) or not os.path.isdir(dataset_path):
        return StatusResponse(
            status="error", message=f"Dataset {query.dataset_path} not found"
        )

    datatype = query.dataset_path.split("/")[0]
    if datatype != "lerobot_v2.1":
        return StatusResponse(
            status="error",
            message="You can only split datasets of type v2.1",
        )

    # Split the dataset
    dataset = LeRobotDataset(path=dataset_path, enforce_path=True)

    try:
        dataset.split_dataset(
            split_ratio=query.split_ratio,
            first_split_name=query.first_split_name,
            second_split_name=query.second_split_name,
        )
    except Exception as e:
        logger.warning(f"Error splitting dataset: {e}")
        return StatusResponse(
            status="error",
            message=f"Error splitting dataset: {e}",
        )
    return StatusResponse(status="ok", message="Dataset split successfully")


@router.post("/dataset/shuffle", response_model=StatusResponse)
async def shuffle_dataset(query: DatasetShuffleRequest):
    """
    Shuffle a dataset in place.
    """
    dataset_path = os.path.join(ROOT_DIR, query.dataset_path)
    # Check if the path exists and is a directory
    if not os.path.exists(dataset_path) or not os.path.isdir(dataset_path):
        return StatusResponse(
            status="error", message=f"Dataset {query.dataset_path} not found"
        )

    datatype = query.dataset_path.split("/")[0]
    if datatype != "lerobot_v2.1":
        return StatusResponse(
            status="error",
            message="You can only shuffle datasets of type v2.1",
        )

    # Shuffle the dataset
    dataset = LeRobotDataset(path=dataset_path, enforce_path=True)

    # Name of the new dataset after shuffling
    new_dataset_name = f"{query.dataset_path}_shuffled"
    while os.path.exists(os.path.join(ROOT_DIR, new_dataset_name)):
        # Find if a int suffix is already present
        if new_dataset_name.endswith("_shuffled"):
            # If it ends with _shuffled, we can add a number
            new_dataset_name += "_1"
        else:
            # Otherwise, parse the int and increment it
            try:
                suffix = int(new_dataset_name.split("_")[-1])
                new_dataset_name = (
                    "_".join(new_dataset_name.split("_")[:-1]) + f"_{suffix + 1}"
                )
            except ValueError:
                # If the suffix is not an int, we just add _1
                new_dataset_name += "_1"

    try:
        dataset.shuffle_dataset(new_dataset_name=new_dataset_name)
    except Exception as e:
        logger.warning(f"Error shuffling dataset: {e}")
        return StatusResponse(
            status="error",
            message=f"Error shuffling dataset: {e}",
        )
    return StatusResponse(status="ok", message="Dataset shuffled successfully")
