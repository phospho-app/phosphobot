import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, Optional, List, Tuple

import numpy as np
from fastapi import BackgroundTasks, Depends
from loguru import logger

from phosphobot.camera import AllCameras, get_all_cameras
from phosphobot.configs import config
from phosphobot.hardware import BaseRobot
from phosphobot.models import BaseDataset, Observation, Step

# New imports for refactored Episode structure
from phosphobot.models import (
    BaseEpisode,
    JsonEpisode,
    LeRobotEpisode,
    LeRobotDataset,
)
from phosphobot.robot import RobotConnectionManager, get_rcm
from phosphobot.types import VideoCodecs
from phosphobot.utils import background_task_log_exceptions, get_home_app_path
from phosphobot.rerun_visualizer import RerunVisualizer

recorder = None  # Global variable to store the recorder instance


class Recorder:
    episode_format: Literal["json", "lerobot_v2", "lerobot_v2.1"] = "lerobot_v2.1"

    is_saving: bool = False
    is_recording: bool = False
    episode: Optional[BaseEpisode] = None  # Link to an Episode instance
    start_ts: Optional[float]
    freq: int  # Stored from start() for use in record_loop

    cameras: AllCameras
    robots: list[BaseRobot]

    # Performance optimization: thread pools for concurrent operations
    _image_thread_pool: Optional[ThreadPoolExecutor] = None
    _robot_thread_pool: Optional[ThreadPoolExecutor] = None
    _max_image_workers: int = 4
    _max_robot_workers: int = 2

    # For push_to_hub, if Recorder handles it directly after save
    # _current_dataset_full_path_for_push: Optional[str] = None
    # _current_branch_path_for_push: Optional[str] = None
    # _use_push_to_hub_after_save: bool = False

    @property
    def episode_recording_folder(
        self,
    ) -> str:  # Base folder for all recordings (".../phosphobot/recordings")
        return str(get_home_app_path() / "recordings")

    def __init__(self, robots: list[BaseRobot], cameras: AllCameras):
        self.robots = robots
        self.cameras = cameras
        self.rerun_visualizer = RerunVisualizer()

        # Initialize thread pools for performance optimization
        self._max_image_workers = max(
            4, len(cameras.camera_ids)
        )  # At least one per camera
        self._max_robot_workers = min(
            2, len(robots) + 1
        )  # Adaptive based on robot count

        self._image_thread_pool = ThreadPoolExecutor(
            max_workers=self._max_image_workers, thread_name_prefix="recorder_images"
        )
        self._robot_thread_pool = ThreadPoolExecutor(
            max_workers=self._max_robot_workers, thread_name_prefix="recorder_robots"
        )

        logger.info(
            f"Recorder initialized with {self._max_image_workers} image workers and {self._max_robot_workers} robot workers"
        )

    async def start(
        self,
        background_tasks: BackgroundTasks,
        robots: list[BaseRobot],  # Can use self.robots or update if new list passed
        codec: VideoCodecs,
        freq: int,
        target_size: Optional[Tuple[int, int]],
        dataset_name: str,  # e.g., "my_robot_data"
        instruction: Optional[str],
        episode_format: Literal["json", "lerobot_v2", "lerobot_v2.1"],
        cameras_ids_to_record: Optional[List[int]],
        use_push_to_hf: bool = True,  # Stored for save_episode to decide
        # Stored for push_to_hub if initiated from here
        branch_path: Optional[str] = None,
        enable_rerun: bool = False,  # Enable real-time Rerun visualization
    ) -> None:
        if target_size is None:
            target_size = (config.DEFAULT_VIDEO_SIZE[0], config.DEFAULT_VIDEO_SIZE[1])

        if self.is_recording:
            logger.warning(
                "Stopping previous recording session before starting a new one."
            )
            await self.stop()  # Stop does not save, just halts the loop

        self.robots = robots  # Update robots if a new list is provided
        self.cameras.cameras_ids_to_record = cameras_ids_to_record  # type: ignore
        self.freq = freq  # Store for record_loop
        self.episode_format = episode_format
        self.use_push_to_hf = use_push_to_hf  # Store for save_episode
        self.branch_path = branch_path

        # Store for push_to_hub, to be used by save_episode
        # self._current_branch_path_for_push = branch_path
        # self._use_push_to_hub_after_save = use_push_to_hf

        logger.info(
            f"Attempting to start recording for dataset '{dataset_name}' in format '{episode_format}'"
        )

        if self.episode_format == "json":
            # JsonEpisode.start_new handles its own path creation within "recordings/json/dataset_name"
            self.episode = await JsonEpisode.start_new(
                base_recording_folder=self.episode_recording_folder,
                dataset_name=dataset_name,
                robots=self.robots,
                # Any other necessary params for JsonEpisode metadata
            )
            # if self._use_push_to_hub_after_save:
            #     self._current_dataset_full_path_for_push = str(Path(self.episode_recording_folder) / "json" / dataset_name)

        elif self.episode_format.startswith("lerobot"):
            # Path for LeRobotDataset: "recordings/lerobot_vX.Y/dataset_name"
            dataset_full_path = os.path.join(
                self.episode_recording_folder, episode_format, dataset_name
            )
            # LeRobotDataset constructor will ensure directories like meta, data, videos exist.
            lerobot_dataset_manager = LeRobotDataset(path=dataset_full_path)

            # if self._use_push_to_hub_after_save:
            #     self._current_dataset_full_path_for_push = lerobot_dataset_manager.folder_full_path

            self.episode = await LeRobotEpisode.start_new(
                dataset_manager=lerobot_dataset_manager,  # Pass the dataset manager
                robots=self.robots,
                codec=codec,
                freq=freq,
                target_size=target_size,
                instruction=instruction,
                all_camera_key_names=self.cameras.get_all_camera_key_names(),
            )
        else:
            logger.error(f"Unknown episode format: {self.episode_format}")
            raise ValueError(f"Unknown episode format: {self.episode_format}")

        if enable_rerun and self.rerun_visualizer:
            self.rerun_visualizer.enabled = True
            episode_index = self.episode.episode_index if self.episode else 0
            self.rerun_visualizer.initialize(dataset_name, episode_index)

        self.is_recording = True
        self.start_ts = time.perf_counter()

        background_tasks.add_task(
            background_task_log_exceptions(self.record_loop),
            target_size=target_size,
            language_instruction=instruction
            or config.DEFAULT_TASK_INSTRUCTION,  # Passed to Step
        )
        logger.success(
            f"Recording started for {self.episode_format} dataset '{dataset_name}'. Episode index: {self.episode.episode_index if self.episode else 'N/A'}"
        )

    async def stop(self) -> None:
        """
        Stop the recording without saving.
        """
        if self.is_recording:
            logger.info("Stopping current recording...")
            self.is_recording = False
            # Allow record_loop to finish its current iteration and exit
            await asyncio.sleep(
                1 / self.freq + 0.1
            )  # Ensure loop has time to see is_recording=False
            logger.info("Recording loop should now be stopped.")
        else:
            logger.info("No active recording to stop.")
        # self.episode remains as is, until save_episode or a new start clears it.

    async def save_episode(self) -> None:
        if self.is_saving:
            logger.warning("Already in the process of saving an episode. Please wait.")
            return None  # Or raise an error/return a specific status

        if not self.episode:
            logger.error(
                "No episode data found. Was recording started and were steps recorded?"
            )
            return None

        if not self.episode.steps:
            logger.warning("Episode contains no steps. Nothing to save.")
            self.episode = None  # Clear the empty episode
            return None

        self.is_saving = True
        # Ensure recording is stopped before saving
        if self.is_recording:
            logger.info("Stopping active recording before saving.")
            await self.stop()

        episode_to_save = self.episode  # Keep a reference
        dataset_name_for_log = episode_to_save.metadata.get(
            "dataset_name", "UnknownDataset"
        )
        episode_format_for_log = episode_to_save.metadata.get(
            "episode_format", "UnknownFormat"
        )

        logger.info(
            f"Starting to save episode for dataset '{dataset_name_for_log}' (format: {episode_format_for_log})..."
        )

        try:
            await episode_to_save.save()  # The episode handles all its saving logic
            logger.success(
                f"Episode saved successfully for dataset '{dataset_name_for_log}'."
            )

        except Exception as e:
            logger.error(f"An error occurred during episode saving: {e}", exc_info=True)
            # Depending on the severity, you might not want to clear self.episode here,
            # to allow for a retry or manual inspection. For now, it's cleared in finally.
            raise  # Re-throw for higher level handling if necessary
        finally:
            self.is_saving = False
            # self.episode = None # Clear episode if not cleared on success (e.g. if push fails but save was ok)

        if self.use_push_to_hf and isinstance(self.episode, LeRobotEpisode):
            self.push_to_hub(
                dataset_path=str(self.episode.dataset_path),
                branch_path=self.branch_path,
            )

        return None

    def push_to_hub(self, dataset_path: str, branch_path: Optional[str] = None) -> None:
        logger.info(
            f"Attempting to push dataset from {dataset_path} to Hugging Face Hub. Branch: {branch_path or 'main'}"
        )
        try:
            # Dataset class needs to be robust enough to be initialized with the full path
            # e.g., "recordings/lerobot_v2.1/my_dataset_name"
            dataset_obj = BaseDataset(path=dataset_path)
            dataset_obj.push_dataset_to_hub(branch_path=branch_path)
            logger.success(
                f"Successfully pushed dataset {dataset_path} to Hugging Face Hub."
            )
        except FileNotFoundError:
            logger.error(f"Dataset path not found for push_to_hub: {dataset_path}")
        except ValueError as ve:
            logger.error(
                f"Failed to initialize Dataset for push_to_hub. Path: {dataset_path}. Error: {ve}"
            )
        except Exception as e:
            logger.error(
                f"An unexpected error occurred while pushing dataset {dataset_path} to Hub: {e}",
                exc_info=True,
            )

    async def record_loop(
        self,
        target_size: tuple[int, int],
        language_instruction: str,  # This is the initial instruction
    ) -> None:
        if not self.episode:
            logger.error(
                "Record loop started but no episode is initialized in the recorder."
            )
            self.is_recording = False  # Stop the loop
            return
        if not self.start_ts:  # Should be set by start()
            logger.error("Record loop started but start_ts is not set.")
            self.is_recording = False  # Stop the loop
            return

        logger.info(
            f"Record loop engaged for episode {self.episode.episode_index if self.episode else 'N/A'}. Cameras: {self.cameras.camera_ids=} ({self.cameras.main_camera=})"
        )

        step_count = 0
        while self.is_recording:  # This flag is controlled by self.stop()
            loop_iteration_start_time = time.perf_counter()

            # --- Optimized Image Gathering with Parallel Processing ---
            main_frames, secondary_frames = await self._gather_frames_parallel(
                target_size
            )

            if main_frames and len(main_frames) > 0:
                main_frame = main_frames[0]
            else:
                main_frame = np.zeros(
                    (target_size[1], target_size[0], 3), dtype=np.uint8
                )

            # --- Optimized Robot Observation with Parallel Processing ---
            (
                final_state,
                final_joints_position,
            ) = await self._gather_robot_observations_parallel()

            current_time_in_episode = loop_iteration_start_time - self.start_ts

            # The language instruction for the step should be the one active for this episode.
            # If instructions can change mid-episode, this needs more complex handling.
            # For now, assume it's the instruction set at the start of the episode.
            current_instruction = (
                self.episode.instruction
                if self.episode.instruction
                else language_instruction
            )

            observation = Observation(
                main_image=main_frame,
                secondary_images=secondary_frames,
                state=final_state,  # Robot's end-effector state(s)
                language_instruction=current_instruction,
                joints_position=final_joints_position,  # Actual joint positions
                timestamp=current_time_in_episode,
            )

            # Action for a step is typically the joints_position that LED to the NEXT observation.
            # So, when we add step N, its action is observation N+1's joints_position.
            # The last step's action might be None or a repeat.
            # update_previous_step handles this.
            step = Step(
                observation=observation,
                action=None,  # Will be filled by update_previous_step for the *previous* step
                metadata={"created_at": loop_iteration_start_time},
            )

            if self.rerun_visualizer and self.rerun_visualizer.enabled:
                self.rerun_visualizer.log_step(
                    step=step,
                    robots=self.robots,
                    cameras=self.cameras,
                    step_index=step_count,
                )

            if step_count % 20 == 0:  # Log every 20 steps
                logger.debug(
                    f"Recording: Processing Step {step_count} for episode {self.episode.episode_index if self.episode else 'N/A'}"
                )

            # Order: update previous, then add current.
            if self.episode.steps:  # If there's a previous step
                # The 'action' of the previous step is the 'joints_position' of the current observation
                self.episode.update_previous_step(step)

            # Append the current step. Episode's append_step will handle its internal logic
            # (like updating meta files for LeRobot format).
            await self.episode.append_step(step)

            elapsed_this_iteration = time.perf_counter() - loop_iteration_start_time
            time_to_wait = max((1 / self.freq) - elapsed_this_iteration, 0)

            # Log performance metrics every 100 steps
            if step_count % 100 == 0:
                logger.debug(
                    f"Step {step_count}: Processing time: {elapsed_this_iteration:.3f}s, Target: {1/self.freq:.3f}s"
                )

            await asyncio.sleep(time_to_wait)
            step_count += 1

        if self.rerun_visualizer and self.rerun_visualizer.enabled:
            self.rerun_visualizer.finalize()

        logger.info(
            f"Recording loop for episode {self.episode.episode_index if self.episode else 'N/A'} has gracefully exited."
        )

    async def _gather_frames_parallel(
        self, target_size: tuple[int, int]
    ) -> tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Simple parallel frame capture - each camera runs independently.
        Returns (main_frames, secondary_frames).
        """
        loop = asyncio.get_event_loop()

        # Get all cameras and their roles
        main_camera = self.cameras.main_camera
        secondary_cameras = self.cameras.get_secondary_cameras()

        # Submit individual camera capture tasks
        camera_futures = []

        # Main camera
        if (
            main_camera
            and hasattr(main_camera, "camera_id")
            and main_camera.camera_id is not None
        ):
            future = loop.run_in_executor(
                self._image_thread_pool,
                self._capture_single_camera,
                main_camera,
                target_size,
            )
            camera_futures.append(("main", main_camera.camera_id, future))

        # Secondary cameras
        for camera in secondary_cameras:
            if hasattr(camera, "camera_id") and camera.camera_id is not None:
                future = loop.run_in_executor(
                    self._image_thread_pool,
                    self._capture_single_camera,
                    camera,
                    target_size,
                )
                camera_futures.append(("secondary", camera.camera_id, future))

        # Wait for all camera captures
        main_frames = []
        secondary_frames = []

        for role, camera_id, future in camera_futures:
            try:
                frame = await future
                if frame is not None:
                    if role == "main":
                        main_frames.append(frame)
                    else:
                        secondary_frames.append(frame)
            except Exception as e:
                logger.warning(f"Failed to capture frame from camera {camera_id}: {e}")

        return main_frames, secondary_frames

    def _capture_single_camera(
        self, camera, target_size: tuple[int, int]
    ) -> Optional[np.ndarray]:
        """
        Capture frame from a single camera.
        This runs in the thread pool.
        """
        try:
            return camera.get_rgb_frame(resize=target_size)
        except Exception as e:
            logger.warning(
                f"Exception capturing frame from camera {getattr(camera, 'camera_id', 'unknown')}: {e}"
            )
            return None

    async def _gather_robot_observations_parallel(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simple parallel robot observation gathering - each robot runs independently.
        Returns (final_state, final_joints_position).
        """
        if not self.robots:
            return np.array([]), np.array([])

        loop = asyncio.get_event_loop()

        # Submit individual robot observation tasks
        robot_futures = []
        for i, robot_instance in enumerate(self.robots):
            future = loop.run_in_executor(
                self._robot_thread_pool,
                self._get_single_robot_observation,
                robot_instance,
                i,
            )
            robot_futures.append(future)

        # Wait for all robot observations
        all_robot_states = []
        all_robot_joints_positions = []

        for future in robot_futures:
            try:
                robot_state, robot_joints = await future
                if robot_state is not None and robot_joints is not None:
                    all_robot_states.append(robot_state)
                    all_robot_joints_positions.append(robot_joints)
            except Exception as e:
                logger.warning(f"Failed to get robot observation: {e}")

        # Concatenate if multiple robots, otherwise use the first robot's data
        final_state = (
            np.concatenate(all_robot_states)
            if len(all_robot_states) > 1
            else (all_robot_states[0] if all_robot_states else np.array([]))
        )
        final_joints_position = (
            np.concatenate(all_robot_joints_positions)
            if len(all_robot_joints_positions) > 1
            else (
                all_robot_joints_positions[0]
                if all_robot_joints_positions
                else np.array([])
            )
        )

        return final_state, final_joints_position

    def _get_single_robot_observation(
        self, robot, robot_idx: int
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get observation from a single robot.
        This runs in the thread pool.
        """
        try:
            return robot.get_observation()
        except Exception as e:
            logger.warning(f"Exception getting observation from robot {robot_idx}: {e}")
            return None, None

    def __del__(self):
        """Cleanup thread pools on deletion."""
        if hasattr(self, "_image_thread_pool") and self._image_thread_pool:
            self._image_thread_pool.shutdown(wait=True)
        if hasattr(self, "_robot_thread_pool") and self._robot_thread_pool:
            self._robot_thread_pool.shutdown(wait=True)


async def get_recorder(
    rcm: RobotConnectionManager = Depends(get_rcm),
) -> Recorder:
    global recorder
    if recorder is not None:
        return recorder
    else:
        robots = await rcm.robots
        cameras = get_all_cameras()
        recorder = Recorder(
            robots=robots,  # type: ignore
            cameras=cameras,
        )
        return recorder
