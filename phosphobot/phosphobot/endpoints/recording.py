import os
from copy import copy

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
)
from loguru import logger

from phosphobot.camera import AllCameras, get_all_cameras
from phosphobot.configs import config
from phosphobot.hardware.base import BaseManipulator
from phosphobot.models import (
    BaseDataset,
    BaseEpisode,
    RecordingPlayRequest,
    RecordingStartRequest,
    RecordingStopRequest,
    RecordingStopResponse,
    StatusResponse,
)
from phosphobot.models import InfoModel
from phosphobot.models.lerobot_dataset import LeRobotDataset
from phosphobot.posthog import is_github_actions
from phosphobot.recorder import Recorder, get_recorder
from phosphobot.robot import RobotConnectionManager, get_rcm
from phosphobot.utils import background_task_log_exceptions, get_home_app_path

router = APIRouter(tags=["recording"])


@router.post("/recording/start", response_model=StatusResponse)
async def start_recording_episode(
    query: RecordingStartRequest,
    background_tasks: BackgroundTasks,
    cameras: AllCameras = Depends(get_all_cameras),
    recorder: Recorder = Depends(get_recorder),
    rcm: RobotConnectionManager = Depends(get_rcm),
) -> StatusResponse | HTTPException:
    """
    Asynchronously start recording an episode in the background.
    Output format is chosen when stopping the recording.
    """

    dataset_name = query.dataset_name or config.DEFAULT_DATASET_NAME
    dataset_name = BaseDataset.consolidate_dataset_name(dataset_name)

    # Remove .DS_Store files from the dataset folder
    dataset_path = os.path.join(
        get_home_app_path(), "recordings", recorder.episode_format, dataset_name
    )
    BaseDataset.remove_ds_store_files(dataset_path)

    if not cameras or not cameras.has_connected_camera:
        logger.warning(
            "No camera available. The episode will be recorded without video data."
        )

    elif (
        query.cameras_ids_to_record is not None
        and len(query.cameras_ids_to_record) == 0
    ):
        logger.warning(
            "No cameras selected to record. The episode will be recorded without video data."
        )

    if (
        query.cameras_ids_to_record is None
        and config.DEFAULT_CAMERAS_TO_RECORD is not None
    ):
        # If the user has not specified cameras to record, we use the default cameras
        cameras_ids_to_record = [
            camera_id
            for camera_id in cameras.camera_ids
            if camera_id in config.DEFAULT_CAMERAS_TO_RECORD
        ]
    elif (
        query.cameras_ids_to_record is None and config.DEFAULT_CAMERAS_TO_RECORD is None
    ):
        # If the user has not specified cameras to record and there are no default cameras, we use all connected cameras
        cameras_ids_to_record = cameras.camera_ids
    elif query.cameras_ids_to_record is not None:
        # If the user has specified cameras to record, we remove duplicates
        # and intersect with the connected cameras
        cameras_ids_to_record = list(
            set(query.cameras_ids_to_record).intersection(cameras.camera_ids)
        )
    else:
        # Fallback: use all connected cameras
        cameras_ids_to_record = cameras.camera_ids

    # Check that the number of cameras and robots is consistent with the existing dataset
    number_of_connected_cameras = len(cameras_ids_to_record)

    # Compute the number of connected robots and remove leader arms
    number_of_connected_robots = 0
    robots_to_record = copy(await rcm.robots)
    for robot in await rcm.robots:
        if (
            hasattr(robot, "SERIAL_ID")
            and query.robot_serials_to_ignore is not None
            and robot.SERIAL_ID in query.robot_serials_to_ignore
        ):
            robots_to_record.remove(robot)
        else:
            number_of_connected_robots += 1

    if len(robots_to_record) == 0:
        raise HTTPException(
            status_code=400,
            detail="No robots to record. You should have at least one robot connected to start recording.",
        )

    format = query.episode_format or config.DEFAULT_EPISODE_FORMAT
    if format != "json":
        try:
            info_model = InfoModel.from_json(
                meta_folder_path=os.path.join(dataset_path, "meta"),
                format=format,
            )
            number_of_cameras_in_dataset = len(info_model.features.observation_images)
            # If we are in the github action, we add 2 simulated cameras for CICD (testing)
            if is_github_actions() and number_of_connected_cameras == 0:
                number_of_connected_cameras += 2
            if number_of_connected_cameras != number_of_cameras_in_dataset:
                raise KeyError(
                    f"Dataset {dataset_name} has {number_of_cameras_in_dataset} cameras but you have {number_of_connected_cameras} connected. Create a new dataset by changing the dataset name in Admin Settings."
                )

            # Get action dimensions from existing dataset
            dataset_action_dim = info_model.features.action.shape[0]
            # Calculate expected action dimensions from connected robots
            expected_action_dim = 0
            for robot in robots_to_record:
                assert isinstance(
                    robot, BaseManipulator
                ), "Robot must be an instance of BaseManipulator."
                base_robot_info = robot.get_info_for_dataset()
                expected_action_dim += base_robot_info.action.shape[0]

            if expected_action_dim != dataset_action_dim:
                raise KeyError(
                    f"Dataset {dataset_name} has action dimension {dataset_action_dim} but connected robots have total action dimension {expected_action_dim}. Create a new dataset by changing the dataset name in Admin Settings."
                )
        except ValueError:
            # This means the dataset does not exist yet
            pass
        except KeyError as e:
            # This means the dataset exists but the number of cameras or robots is not consistent
            logger.warning(
                "Number of cameras or robots is not consistent with the existing dataset. Create a new dataset by changing the dataset name in Admin Settings."
            )
            raise HTTPException(status_code=400, detail=str(e))

    # Check if the recorder is not currently saving
    if recorder.is_saving:
        raise HTTPException(
            status_code=400,
            detail="Recorder is still saving an episode. Please wait a few seconds and try again.",
        )

    # Update recorder's robots
    await recorder.start(
        background_tasks=background_tasks,
        # Replace all the None values with defaults
        episode_format=query.episode_format or config.DEFAULT_EPISODE_FORMAT,
        dataset_name=dataset_name,
        codec=query.video_codec or config.DEFAULT_VIDEO_CODEC,
        freq=query.freq or config.DEFAULT_FREQ,
        branch_path=query.branch_path,
        robots=robots_to_record,  # type: ignore
        target_size=query.target_video_size
        or (config.DEFAULT_VIDEO_SIZE[0], config.DEFAULT_VIDEO_SIZE[1]),
        cameras_ids_to_record=cameras_ids_to_record,
        instruction=query.instruction or config.DEFAULT_TASK_INSTRUCTION,
        enable_rerun=query.enable_rerun_visualization,
    )
    return StatusResponse()


@router.post("/recording/stop", response_model=RecordingStopResponse)
async def stop_recording_episode(
    query: RecordingStopRequest,
    background_tasks: BackgroundTasks,
    recorder: Recorder = Depends(get_recorder),
) -> RecordingStopResponse | HTTPException:
    """
    Stop the recording of the episode. The data is saved to disk to the user home directory, in the `phosphobot` folder.
    """
    if recorder.is_saving:
        raise HTTPException(
            status_code=400,
            detail="Recorder is still saving an episode. Please wait a few seconds and try again.",
        )

    if not recorder.is_recording:
        raise HTTPException(status_code=400, detail="No episode to stop")

    if recorder.episode is None:
        raise HTTPException(status_code=400, detail="No episode to stop")

    # This doesn't save the episode to disk, only stops the recording
    await recorder.stop()

    # Save the episode to disk
    if not query.save:
        logger.info(
            "Episode stopped but not saved. Use the `save` parameter to save the episode."
        )
        return RecordingStopResponse(episode_folder_path=None, episode_index=None)

    background_tasks.add_task(background_task_log_exceptions(recorder.save_episode))

    return RecordingStopResponse(
        episode_folder_path=str(recorder.episode.dataset_path),
        episode_index=recorder.episode.episode_index,
    )


@router.post("/recording/play", response_model=StatusResponse)
async def play_recording(
    query: RecordingPlayRequest,
    recorder: Recorder = Depends(get_recorder),
    rcm: RobotConnectionManager = Depends(get_rcm),
) -> StatusResponse | HTTPException:
    """
    Play a recorded episode.
    """

    if query.episode_path is not None:
        if not os.path.exists(query.episode_path):
            raise HTTPException(
                status_code=400,
                detail=f"Episode path {query.episode_path} does not exist.",
            )
        episode = BaseEpisode.load(query.episode_path, format=recorder.episode_format)
    elif query.dataset_name is not None and query.episode_id is None:
        # Load the latest episode
        dataset_path = os.path.join(
            get_home_app_path(),
            "recordings",
            query.dataset_format,
            query.dataset_name,
        )
        dataset = LeRobotDataset(path=dataset_path, enforce_path=True)
        dataset.load_episodes()
        if len(dataset.episodes) == 0:
            raise HTTPException(
                status_code=400,
                detail=f"No episode found in the dataset {query.dataset_name}.",
            )
        episode = dataset.episodes[-1]
        if not episode:
            raise HTTPException(
                status_code=400,
                detail=f"No episode found in the dataset {query.dataset_name}.",
            )
    elif query.dataset_name is not None and query.episode_id is not None:
        # Load the episode with the given ID
        dataset_path = os.path.join(
            get_home_app_path(),
            "recordings",
            query.dataset_format,
            query.dataset_name,
        )
        dataset = LeRobotDataset(path=dataset_path, enforce_path=True)
        dataset.load_episodes()
        if query.episode_id >= len(dataset.episodes):
            raise HTTPException(
                status_code=400,
                detail=f"Request to play episode with ID {query.episode_id} but the dataset {query.dataset_name} has only {len(dataset.episodes)} episodes.",
            )
        episode = dataset.episodes[query.episode_id]
    elif hasattr(recorder, "episode") and recorder.episode is not None:
        episode = recorder.episode
    else:
        raise HTTPException(
            status_code=400,
            detail="No episode path given and no episode stored in the recorder.",
        )

    if isinstance(query.robot_id, int):
        if query.robot_id >= len(await rcm.robots):
            raise HTTPException(
                status_code=400,
                detail=f"Robot with ID {query.robot_id} not found.",
            )
        robots = [await rcm.get_robot(query.robot_id)]
    elif isinstance(query.robot_id, list):
        robots = []
        for robot_id in query.robot_id:
            if robot_id >= len(await rcm.robots):
                raise HTTPException(
                    status_code=400,
                    detail=f"Robot with ID {robot_id} not found.",
                )
            robots.append(await rcm.get_robot(robot_id))
    elif query.robot_id is None:
        robots = copy(await rcm.robots)

    if query.robot_serials_to_ignore is not None:
        for robot in robots:
            if (
                hasattr(robot, "SERIAL_ID")
                and robot.SERIAL_ID in query.robot_serials_to_ignore
            ):
                robots.remove(robot)

    # the episode cannot be None since episode_path and recorder.episode cannot be none simultaneously
    await episode.play(  # type: ignore
        robots=robots,  # type: ignore
        playback_speed=query.playback_speed,
        interpolation_factor=query.interpolation_factor,
        replicate=query.replicate,
    )
    return StatusResponse()
