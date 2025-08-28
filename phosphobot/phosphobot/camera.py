import asyncio
import atexit
import base64
import binascii
import json
import platform
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Literal, Optional, Tuple, cast

import cv2
import numpy as np
import zmq
from fastapi import Request
from loguru import logger

from phosphobot.configs import config
from phosphobot.models import AllCamerasStatus, SingleCameraStatus
from phosphobot.types import CameraTypes

cameras = None


def get_camera_names() -> List[str]:
    """
    This function returns the list of cameras connected to the computer.
    Example Output:
    ["Caméra FaceTime HD", "Integrated Camera", "USB Camera"]
    """

    import platform

    system_name = platform.system()
    camera_names = []

    if system_name == "Darwin":  # macOS
        # Run the system_profiler command to get camera information
        result = subprocess.run(
            ["system_profiler", "SPCameraDataType"], stdout=subprocess.PIPE, text=True
        )

        # Split the output into lines
        lines = result.stdout.split("\n")

        # Iterate over each line to find camera names
        for line in lines:
            if (
                "Model ID" in line
                or "Model Identifier" in line
                or "Identifiant du modèle" in line
            ):
                # Extract the camera name
                camera_name = line.split(":")[-1].strip()
                camera_names.append(camera_name)

    elif system_name == "Linux":
        # Use v4l2-ctl to list cameras on Linux
        try:
            result = subprocess.run(
                ["v4l2-ctl", "--list-devices"], stdout=subprocess.PIPE, text=True
            )
            lines = result.stdout.splitlines()

            # collect (device_path, camera_name)
            dev_name_pairs: List[Tuple[str, str]] = []
            i = 0
            while i < len(lines):
                line = lines[i]
                if line and not line.startswith("\t") and line.strip().endswith(":"):
                    header = line.strip().rstrip(":")
                    # Simplify the camera name
                    if "RealSense" in header:
                        name = "Realsense"
                    else:
                        name = header.split("(")[0].split(":")[0].strip()
                    # gather subsequent /dev/video* entries
                    i += 1
                    while i < len(lines) and lines[i].startswith("\t"):
                        dev = lines[i].strip()
                        if dev.startswith("/dev/video"):
                            dev_name_pairs.append((dev, name))
                        i += 1
                    continue
                i += 1

            # sort by the numeric index of /dev/videoN
            def video_index(dev_path: str) -> int:
                return int(dev_path.replace("/dev/video", ""))

            dev_name_pairs.sort(key=lambda dn: video_index(dn[0]))
            camera_names = [name for _, name in dev_name_pairs]

        except FileNotFoundError:
            logger.warning(
                "v4l2-ctl is not installed. Please install it to list cameras."
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"v4l2-ctl failed: {e}")

    elif system_name == "Windows":
        # Use PowerShell to list cameras on Windows
        try:
            result = subprocess.run(
                [
                    "powershell",
                    "-Command",
                    "Get-PnpDevice -Class Camera | Select-Object -ExpandProperty FriendlyName",
                ],
                stdout=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            lines = result.stdout.split("\n")
            for line in lines:
                if line.strip() != "":
                    camera_names.append(line.strip())
        except FileNotFoundError:
            logger.warning("PowerShell is not available. Cannot list cameras.")

    else:
        logger.error(f"Unsupported operating system: {system_name}")

    if config.SIMULATE_CAMERAS:
        camera_names.extend(["Main Simulated Camera", "Secondary Simulated Camera"])

    return camera_names


def detect_camera_type(
    index: int,
    camera_names: List[str] = [],
    possible_camera_ids: Optional[List[int]] = None,
) -> CameraTypes:
    """
    Detect the type of camera for the given index.
    Returns "classic" for a regular camera or "realsense" for a realsense camera.
    We check first check if the index corresponds to a realsense camera by matching the device name.
    Then we see if the camera is a stereo camera by checking the aspect ratio.

    To detect simulated cameras, we check if the index is the last one or the one before the last one..
    Pass all camera_ids for that.
    """
    # Check if the realsense device can be matched with the current index
    if index < len(camera_names):
        camera_name = camera_names[index]
        if "realsense" in camera_name.lower():
            return "realsense"
    if config.SIMULATE_CAMERAS and possible_camera_ids is not None:
        # The last two cameras indexes are simulated cameras
        if index == possible_camera_ids[-1]:
            return "dummy"
        if index == possible_camera_ids[-2]:
            return "dummy_stereo"

    # Check for stereo camera using OpenCV
    # For this, look at the resolution of the camera
    # If the ratio is 16:9, it is a classic camera
    # If it's 32:9, it's a stereo camera
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if width == 0 or height == 0:
            logger.warning(
                f"Camera {index} has invalid resolution: {width}x{height}. Assuming classic camera."
            )
            cap.release()
            return "classic"
        ratio = width / height
        if ratio >= 8 / 3:
            cap.release()
            return "stereo"
        cap.release()

    # Check for classic camera using OpenCV
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        cap.release()
        return "classic"

    # If no camera detected
    return "unknown"


# TODO: Handle multiple realsense cameras
def _find_cameras(
    possible_camera_ids: list[int],
    raise_when_empty=False,
    camera_names: List[str] = [],
) -> list[int]:
    """
    Utility function to find cameras from a list of possible camera ids.

    This tries to open the camera and check if it's opened.

    This ignores realsense cameras.
    """

    import cv2

    camera_ids = []
    for camera_idx in possible_camera_ids:
        try:
            camera = cv2.VideoCapture(camera_idx)
            is_open = camera.isOpened()
        except cv2.error as e:
            logger.warning(f"Failed to open camera at index {camera_idx}: {e}.")
            is_open = False
            continue

        if not is_open:
            continue

        if (
            detect_camera_type(
                index=camera_idx,
                camera_names=camera_names,
                possible_camera_ids=possible_camera_ids,
            )
            == "realsense"
        ):
            logger.info("Realsense camera detected, skipping")
            continue

        if is_open:
            logger.success(f"Camera found at index {camera_idx}")
            camera_ids.append(camera_idx)

        camera.release()

    if config.SIMULATE_CAMERAS:
        camera_ids.extend([len(camera_ids), len(camera_ids) + 1])

    if raise_when_empty and len(camera_ids) == 0:
        raise OSError(
            f"Not a single camera was detected in {possible_camera_ids}. Try replugging, rebooting your computer,"
            + "reinstalling `opencv2`, reinstalling your camera driver, and ensure your camera is compatible with opencv2."
        )
    elif len(camera_ids) == 0:
        logger.warning(f"No camera detected in {possible_camera_ids}")

    return camera_ids


# TODO: Handle multiple realsense cameras
def detect_video_indexes(
    max_index_search_range: Optional[int] = None,
    mock: bool = False,
    camera_names: List[str] = [],
) -> list[int]:
    """
    Return the indexes of all available cameras.

    Note: This list of int is not guaranteed to be continuous (e.g: [0, 1, 3, 4])

    This is only done once and the result is cached in self._available_camera_ids
    """
    if max_index_search_range is None:
        from phosphobot.configs import config

        max_index_search_range = config.MAX_OPENCV_INDEX

    cameras = []
    if platform.system() == "Linux":
        possible_ports = [str(port) for port in Path("/dev").glob("video*")]
        possible_camera_ids = [
            int(port.removeprefix("/dev/video")) for port in possible_ports
        ]
        # Sort by increasing
        possible_camera_ids = sorted(possible_camera_ids)
        logger.info(
            f"(Linux) Found possible ports through scanning '/dev/video*': {possible_camera_ids}"
        )
        # Filter out indexes > MAX_OPENCV_INDEX
        to_remove = [idx for idx in possible_camera_ids if idx > max_index_search_range]
        logger.info(
            f"Ignoring possible ports: {to_remove} (index > {max_index_search_range})"
        )
        possible_camera_ids = [
            idx for idx in possible_camera_ids if idx not in to_remove
        ]
        indices = _find_cameras(possible_camera_ids, camera_names=camera_names)
        cameras.extend(indices)
    else:
        logger.debug(
            f"Listing camera indexes through OpenCV, max {max_index_search_range} cameras"
        )
        possible_camera_ids = list(range(max_index_search_range))
        indices = _find_cameras(possible_camera_ids, camera_names=camera_names)
        cameras.extend(indices)

    return cameras


class BaseCamera(ABC):
    camera_type: CameraTypes
    is_active: bool = False
    width: int
    height: int
    fps: int

    def __init__(self):
        atexit.register(self.stop)

    def __del__(self):
        self.stop()

    @property
    def camera_name(self) -> str:
        return f"BaseCamera {self.camera_type}"

    @abstractmethod
    def get_rgb_frame(
        self, resize: Optional[tuple[int, int]] = None
    ) -> Optional[cv2.typing.MatLike]:
        """Get the latest frame from the camera."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop the camera from capturing frames."""
        raise NotImplementedError("Stop method not implemented")

    def get_depth_frame(self) -> Optional[cv2.typing.MatLike]:
        """Get the latest depth frame from the camera."""
        raise NotImplementedError("Depth frame not available")

    def get_jpeg_rgb_frame(
        self,
        target_size: Optional[tuple[int, int]],
        quality: Optional[int],
        is_video_frame: bool = True,
    ) -> Optional[bytes]:
        if is_video_frame:
            rgb_frame = self.get_rgb_frame(resize=target_size)
        else:
            rgb_frame = self.get_depth_frame()
        if rgb_frame is None or not isinstance(rgb_frame, np.ndarray):
            return None

        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        params = [cv2.IMWRITE_JPEG_QUALITY, quality] if quality else []
        success, jpeg = cv2.imencode(".jpg", bgr_frame, params)

        if not success:
            return None

        return jpeg.tobytes()

    async def generate_rgb_frames(
        self,
        target_size: Optional[tuple[int, int]],
        quality: Optional[int],
        is_video_frame: bool = True,
        request: Optional[Request] = None,
    ) -> AsyncGenerator:
        """Generator for video frames"""
        try:
            while self.is_active and (
                request is None or not await request.is_disconnected()
            ):
                time_start = time.perf_counter()
                frame = self.get_jpeg_rgb_frame(
                    is_video_frame=is_video_frame,
                    target_size=target_size,
                    quality=quality,
                )
                if frame is not None:
                    yield (
                        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                    )
                else:
                    logger.warning(
                        f"{self.camera_name} Skipped frame due to capture error"
                    )
                    # Prevent tight loop
                    await asyncio.sleep(0.02)
                # Wait according to the fps
                time_spent = time.perf_counter() - time_start
                time_to_wait = max(0, 1 / self.fps - time_spent)
                await asyncio.sleep(time_to_wait)
        except GeneratorExit:
            logger.info(f"{self.camera_name} Generator exited")
        except KeyboardInterrupt:
            logger.info(f"{self.camera_name} Keyboard interrupt")
            self.stop()
        except Exception as e:
            logger.warning(f"{self.camera_name} Error generating frames: {str(e)}")
            self.stop()

    async def generate_depth_frames(self):
        return await self.generate_rgb_frames(is_video_frame=False)


class VideoCamera(threading.Thread, BaseCamera):
    camera_type: CameraTypes = "classic"
    camera_id: Optional[int] = None
    last_frame: Optional[cv2.typing.MatLike] = None
    lock: threading.Lock
    _stop_event: threading.Event
    video: Optional[cv2.VideoCapture] = None

    def __init__(
        self,
        video: Optional[cv2.VideoCapture] = None,
        disable: bool = False,
        camera_id: Optional[int] = 0,
        camera_type: Optional[CameraTypes] = None,
    ):
        threading.Thread.__init__(self)
        BaseCamera.__init__(self)

        if camera_type:
            self.camera_type = camera_type

        self.camera_id = camera_id
        if disable:
            logger.info(f"{self.camera_name}: disabled")
            self.is_active = False
            return

        if video:
            self.video = video
        else:
            self.video = cv2.VideoCapture()

        self.is_active = self.init_camera()
        if self.is_active:
            self.lock = threading.Lock()
            self._stop_event = threading.Event()
            self.start()

    @property
    def camera_name(self) -> str:
        return f"VideoCamera {self.camera_type} {self.camera_id}"

    def stop(self) -> None:
        """Stop the video stream"""
        logger.debug(f"{self.camera_name}: Stopping. is_active={self.is_active}")
        if not self.is_active:
            return
        self.is_active = False
        self._stop_event.set()
        try:
            if self.video:
                # If you don't wait and the camera is currently used (eg: streamed or recording)
                # Then OpenCV will crash with error 139. This small waits for the .is_active=False
                # just above to propagate before releasing the OpenCV camera.
                time.sleep(0.1)
                self.video.release()
        except Exception:
            pass
        finally:
            self.video = None

    def init_camera(self) -> bool:
        if not self.video:
            return False

        try:
            if self.video.isOpened():
                # If on Windows, we need to set the height, width and fps (otherwise it will not work)
                # We set it to a default 16:9 value
                # TODO: Make this parameterizable
                if platform.system() == "Windows":
                    self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
                    self.video.set(cv2.CAP_PROP_FPS, 30)

                self.video.set(
                    cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc("M", "J", "P", "G")
                )
                # Get the width, height, and fps of the camera
                self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.fps = int(self.video.get(cv2.CAP_PROP_FPS))
            else:
                logger.warning(f"{self.camera_name}: Failed to open")
                return False

            success, _ = self.video.read()
            if not success:
                logger.warning(f"""{self.camera_name}: Failed to grab first frame
Camera id: {self.camera_id}
Camera type: {self.camera_type}""")
                return False

            return True

        except Exception as e:
            logger.warning(f"{self.camera_name}: Error initializing {str(e)}")

        return False

    def run(self) -> None:
        if not self.is_active:
            return None

        with self.lock:
            while (
                not self._stop_event.is_set()
                and self.video is not None
                and self.is_active
            ):
                if self.camera_type == "dummy" or self.camera_type == "dummy_stereo":
                    # No need to read frames from a dummy camera
                    time.sleep(0.1)
                    continue

                if not self.video or not self.video.isOpened():
                    logger.warning(f"{self.camera_name}: is not initialized")
                    self.last_frame = None
                    continue

                # The stereo camera fails on the first 2 attempts
                success, frame = False, None
                for _ in range(3):  # Try up to 3 times
                    success, frame = self.video.read()
                    if success:
                        break

                if not success:
                    logger.warning(f"{self.camera_name}: Failed to grab frame")
                    self.last_frame = None
                else:
                    self.last_frame = frame

    def get_rgb_frame(
        self, resize: Optional[tuple[int, int]] = None
    ) -> Optional[cv2.typing.MatLike]:
        """
        Read a frame from the camera. Returns None if the frame could not be read.

        Shape: (height, width, channels)
        type: np.uint8
        """
        if not self.is_active:
            logger.warning(f"{self.camera_name}: is not active")
        if self.last_frame is None:
            logger.warning(f"{self.camera_name}: No frame available")

        frame: Optional[np.ndarray] = None
        # Convert from BGR to RGB
        if self.last_frame is not None:
            frame = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2RGB)

        if resize is not None and frame is not None:
            frame = cv2.resize(src=frame, dsize=resize, interpolation=cv2.INTER_AREA)

        return frame


class DummyCamera(VideoCamera):
    camera_type: Literal["dummy", "dummy_stereo"] = "dummy"
    is_active: bool = False
    width: int
    height: int
    fps: int

    def __init__(
        self,
        camera_type: Literal["dummy", "dummy_stereo"],
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ):
        self.width = width
        self.height = height
        self.fps = fps
        super().__init__(camera_type=camera_type)

    def init_camera(self):
        """
        The simulated camera cannot be opened with opencv, so we return True.
        """
        self.last_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        return True

    def get_rgb_frame(
        self, resize: Optional[Tuple[int, int]] = None
    ) -> Optional[cv2.typing.MatLike]:
        """
        Read a frame from the camera. Returns None if the frame could not be read.

        Shape: (height, width, channels)
        type: np.uint8
        """
        # Return a white frame
        if resize is not None:
            frame = 255 * np.ones((resize[1], resize[0], 3), dtype=np.uint8)
        else:
            frame = 255 * np.ones((self.height, self.width, 3), dtype=np.uint8)

        return frame

    def stop(self) -> None:
        """Stop the video stream"""
        logger.debug(f"{self.camera_name}: Stopping. is_active={self.is_active}")
        self.is_active = False

    @property
    def camera_name(self) -> str:
        return f"DummyCamera {self.camera_type}"


class StereoCamera(VideoCamera):
    """
    A stereo camera captures two frames at the same time: left eye, right eye.

    The frames are concatenated into a single frame by a chip in the camera.

    On this model, we suppose that left=left half, right=right half. But you sometimes
    have top/bottom or other configurations.
    """

    camera_type: CameraTypes = "stereo"

    @property
    def camera_name(self) -> str:
        return f"StereoCamera {self.camera_id}"

    def get_left_eye_rgb_frame(
        self, resize: Optional[Tuple[int, int]] = None
    ) -> Optional[cv2.typing.MatLike]:
        last_frame = self.get_rgb_frame()
        if last_frame is None:
            return None
        # Split the frame into two parts
        width = last_frame.shape[1]
        left_frame = last_frame[:, : width // 2]
        if resize is not None:
            left_frame = cv2.resize(left_frame, resize, interpolation=cv2.INTER_AREA)
        return left_frame

    def get_right_eye_rgb_frame(
        self, resize: Optional[Tuple[int, int]] = None
    ) -> Optional[cv2.typing.MatLike]:
        last_frame = self.get_rgb_frame()
        if last_frame is None:
            return None
        # Split the frame into two parts
        width = last_frame.shape[1]
        right_frame = last_frame[:, width // 2 :]
        if resize is not None:
            right_frame = cv2.resize(right_frame, resize, interpolation=cv2.INTER_AREA)
        return right_frame


try:
    import pyrealsense2 as rs  # type: ignore

    REALSENSE_AVAILABLE = True

    class RealSenseCamera(BaseCamera):
        """
        A Realsense camera is an Intel camera that can capture depth and RGB frames.

        It's based on infrared technology and can be used to capture 3D images.
        """

        camera_type: CameraTypes = "realsense"
        last_rgb_frame: np.ndarray
        last_depth_frame: np.ndarray
        pipeline: rs.pipeline
        is_connected: bool = False
        device_info: str
        device_serial: str
        device_index: int

        def __init__(
            self,
            device_serial: Optional[str] = None,
            device_index: Optional[int] = None,
            disable: bool = False,
        ):
            super().__init__()

            self.device_serial = device_serial or "unknown"
            self.device_index = device_index if device_index is not None else 0
            self.is_connected = False
            self.is_active = False

            if disable:
                logger.debug(f"{self.camera_name} disabled")
                return

            # Configure depth and color streams
            self.pipeline = rs.pipeline()
            config = rs.config()

            # Get the RealSense context and devices
            ctx = rs.context()
            realsense_devices = ctx.query_devices()

            if realsense_devices.size() == 0:
                logger.warning("No RealSense devices found")
                return

            # Find the specific device if serial number is provided
            target_device = None
            if device_serial and device_serial != "unknown":
                for i in range(realsense_devices.size()):
                    device = realsense_devices[i]
                    if device.get_info(rs.camera_info.serial_number) == device_serial:
                        target_device = device
                        break

                if target_device is None:
                    logger.error(
                        f"RealSense device with serial {device_serial} not found"
                    )
                    return
            else:
                # Use device by index if no serial number provided
                if self.device_index < realsense_devices.size():
                    target_device = realsense_devices[self.device_index]
                    # Update serial number from the actual device
                    self.device_serial = target_device.get_info(
                        rs.camera_info.serial_number
                    )
                else:
                    logger.error(
                        f"RealSense device index {self.device_index} out of range"
                    )
                    return

            self.is_connected = True

            try:
                # Store device information
                self.device_info = target_device.get_info(rs.camera_info.name)

                # Enable the specific device using its serial number
                config.enable_device(self.device_serial)

                # Configure streams
                config.enable_stream(
                    stream_type=rs.stream.color,
                    format=rs.format.bgr8,
                )
                config.enable_stream(
                    stream_type=rs.stream.depth,
                    format=rs.format.z16,
                )

                # Add a small delay to ensure device is ready
                time.sleep(0.3)

                # Start streaming
                self.pipeline.start(config)

                # Wait a bit more for the pipeline to stabilize
                time.sleep(0.3)

                # Get the width, height, and fps of the camera
                profile = self.pipeline.get_active_profile()
                color_stream = profile.get_stream(rs.stream.color)
                if color_stream:
                    video_profile = color_stream.as_video_stream_profile()
                    self.width = video_profile.width()
                    self.height = video_profile.height()
                    self.fps = video_profile.fps()
                else:
                    # Set defaults if stream info is not available
                    self.width = 640
                    self.height = 480
                    self.fps = 30
                    logger.warning(
                        f"{self.camera_name}: Using default resolution and fps"
                    )

                self.is_active = True
                logger.debug(
                    f"{self.camera_name}: Successfully initialized ({self.width}x{self.height} @ {self.fps}fps)"
                )

            except Exception as e:
                logger.error(f"{self.camera_name}: Failed to initialize - {str(e)}")
                self.is_connected = False
                self.is_active = False
                if hasattr(self, "pipeline"):
                    try:
                        self.pipeline.stop()
                    except:  # noqa: E722
                        pass

        @property
        def camera_name(self) -> str:
            return f"RealsenseCamera {self.device_index} ({self.device_serial})"

        def get_rgb_frame(
            self, resize: Optional[Tuple[int, int]] = None
        ) -> Optional[cv2.typing.MatLike]:
            # To get the video frame, get the couple (video, depth) frame from the wait_for_frames method
            if not self.is_active:
                logger.warning(f"{self.camera_name} is not active")
                return None
            # Wait for a coherent pair of frames: depth and color
            last_bgr_frame = self.pipeline.wait_for_frames(
                timeout_ms=200
            ).get_color_frame()
            if last_bgr_frame is None:
                logger.warning(f"{self.camera_name} failed to grab frame")
                return None
            np_bgr_frame = np.asanyarray(last_bgr_frame.get_data())
            # Convert frame from BGR to RGB
            frame = cv2.cvtColor(np_bgr_frame, cv2.COLOR_BGR2RGB)
            if resize is not None:
                frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
            return frame

        def get_depth_frame(
            self, resize: Optional[Tuple[int, int]] = None
        ) -> Optional[cv2.typing.MatLike]:
            # To get the depth frame, also get the couple (video, depth) frame from the wait_for_frames method
            # The method get_depth and get_rgb_frame can be called simultaneously without lagging (tested)
            # One should load them together for recording to be sure that the depth and video frame are coherent

            if not self.is_active:
                logger.warning(f"{self.camera_name} is not active")
                return None
            last_bgr_depth_frame = self.pipeline.wait_for_frames().get_depth_frame()
            if last_bgr_depth_frame is None:
                logger.warning(f"{self.camera_name} Failed to grab frame")
                return None
            np_bgr_depth_frame = np.asanyarray(last_bgr_depth_frame.get_data())
            frame = cv2.cvtColor(np_bgr_depth_frame, cv2.COLOR_BGR2RGB)
            if resize is not None:
                frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
            return frame

        def stop(self) -> None:
            if self.is_active:
                self.is_active = False
                try:
                    time.sleep(0.1)
                    self.pipeline.stop()
                except Exception as e:
                    logger.warning(f"{self.camera_name} failed to stop: {str(e)}")

    class RealSenseVirtualCamera(VideoCamera):
        def __init__(
            self,
            realsense_camera: RealSenseCamera,
            frame_type: Literal["rgb", "depth"],
            camera_id: int,
            disable: bool = False,
        ):
            threading.Thread.__init__(self)

            self.width = realsense_camera.width
            self.height = realsense_camera.height
            self.fps = realsense_camera.fps
            self.is_active = not disable and realsense_camera.is_active

            self.realsense_camera = realsense_camera
            self.frame_type = frame_type
            self.camera_type = cast(CameraTypes, f"realsense_{frame_type}")
            self.camera_id = camera_id

            atexit.register(self.stop)

        @property
        def is_active(self) -> bool:
            return self.realsense_camera.is_active

        @is_active.setter
        def is_active(self, value):
            return

        def get_rgb_frame(
            self, resize: Optional[Tuple[int, int]] = None
        ) -> Optional[cv2.typing.MatLike]:
            if not self.is_active:
                return None
            if self.frame_type == "rgb":
                frame = self.realsense_camera.get_rgb_frame(resize=resize)
            else:
                depth_frame = self.realsense_camera.get_depth_frame(resize=resize)
                if depth_frame is not None:
                    # Normalize depth data for visualization
                    normalized_depth = cv2.normalize(
                        depth_frame, depth_frame, 0, 255, cv2.NORM_MINMAX
                    )
                    normalized_depth = normalized_depth.astype(np.uint8)
                    # Apply colormap for better visualization
                    frame = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    frame = None
            return frame

        def stop(self) -> None:
            self.realsense_camera.stop()

        @property
        def camera_name(self) -> str:
            return f"RealSenseVirtualCamera {self.camera_type} {self.camera_id}"

except ImportError:
    logger.debug(
        "phosphobot: pyrealsense2 not available, RealSenseCamera will not be available"
    )

    REALSENSE_AVAILABLE = False

    class RealSenseCamera(BaseCamera):  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("Install pyrealsense2 to add RealSense camera support.")

    class RealSenseVirtualCamera(VideoCamera):  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("Install pyrealsense2 to add RealSense camera support.")


class ZMQCamera(VideoCamera):
    """
    A camera that connects to a ZMQ PUSH socket, receiving JSON-serialized
    data with base64-encoded camera frames and performing manual topic filtering.
    """

    camera_type: CameraTypes = "zmq"
    connect_to: str
    topic: Optional[str]
    stream_initialized: bool = False
    context: Optional[zmq.Context] = None
    socket: Optional[zmq.Socket] = None
    poller: Optional[zmq.Poller] = None

    def __init__(
        self,
        connect_to: str = "tcp://localhost:5555",
        topic: Optional[str] = None,
        disable: bool = False,
        camera_id: Optional[int] = None,
    ):
        self.connect_to = connect_to
        self.topic = topic if topic and topic.strip() else None
        self.stream_initialized = False
        super().__init__(video=None, disable=disable, camera_id=camera_id)
        self.last_frame = None
        self.lock = threading.Lock()
        self._stop_event = threading.Event()
        self.thread = None

    @property
    def camera_name(self) -> str:
        if self.topic:
            return f"ZMQCamera(addr='{self.connect_to}', topic='{self.topic}')"
        return f"ZMQCamera(addr='{self.connect_to}', receives_all_topics)"

    def init_camera(self) -> bool:
        """Initializes the ZMQ PULL socket."""
        try:
            logger.info(f"{self.camera_name}: Connecting to ZMQ PUSH server...")
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.PULL)
            self.socket.setsockopt(zmq.RCVTIMEO, 2000)
            self.socket.connect(self.connect_to)

            if self.topic:
                logger.info(
                    f"{self.camera_name}: Will manually filter for topic '{self.topic}'."
                )
            else:
                logger.warning(
                    f"{self.camera_name}: No topic set, will process all received messages."
                )

            self.poller = zmq.Poller()
            self.poller.register(self.socket, zmq.POLLIN)
            self.width, self.height = 0, 0
            self.fps = 30

            logger.success(f"{self.camera_name}: ZMQ PULL connection established.")
            return True
        except Exception as e:
            logger.error(
                f"{self.camera_name}: Failed to initialize ZMQ connection: {e}"
            )
            return False

    def _process_frame_data(self, data: dict) -> None:
        """Helper function to process a received data dictionary and update the frame."""
        if not self.stream_initialized:
            shape = data["shape"]
            self.height, self.width, _ = shape
            self.stream_initialized = True
            logger.success(
                f"{self.camera_name}: Stream properties detected: {self.width}x{self.height}"
            )

        frame_bytes = base64.b64decode(data["frame_bytes"])
        frame = np.frombuffer(frame_bytes, dtype=np.dtype(data["dtype"]))
        reconstructed_frame = frame.reshape(data["shape"])
        with self.lock:
            self.last_frame = cv2.cvtColor(reconstructed_frame, cv2.COLOR_RGB2BGR)

    def run(self) -> None:
        """Polls the ZMQ PULL socket and manually filters messages by topic."""
        if not self.socket or not self.poller:
            logger.error(f"{self.camera_name}: Cannot run, ZMQ socket not initialized.")
            return

        while not self._stop_event.is_set():
            socks = dict(self.poller.poll(timeout=100))
            if self.socket in socks and socks[self.socket] == zmq.POLLIN:
                try:
                    message_parts = self.socket.recv_multipart(flags=zmq.NOBLOCK)
                    if len(message_parts) < 2:
                        continue  # Ignore malformed messages

                    received_topic = message_parts[0].decode()

                    # Process if this message is for us, or if we accept all topics
                    if self.topic is None or received_topic == self.topic:
                        json_bytes = message_parts[1]
                        data = json.loads(json_bytes.decode("utf-8"))
                        self._process_frame_data(data)

                except zmq.Again:
                    continue
                except (
                    json.JSONDecodeError,
                    binascii.Error,
                    KeyError,
                    IndexError,
                    TypeError,
                ) as e:
                    logger.warning(
                        f"{self.camera_name}: Malformed data packet. Error: {e}"
                    )
                except Exception as e:
                    logger.error(
                        f"{self.camera_name}: Unexpected error processing frame: {e}"
                    )

        logger.info(f"{self.camera_name}: Thread stopped.")

    def stop(self) -> None:
        """Extends the parent stop method to gracefully close ZMQ resources."""
        if self._stop_event.is_set():
            return
        logger.debug(f"{self.camera_name}: Stopping...")
        self._stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        try:
            if self.socket:
                self.socket.close()
            if self.context:
                self.context.term()
        except Exception as e:
            logger.error(f"{self.camera_name}: Error during ZMQ cleanup: {e}")


class AllCameras:
    disabled_cameras: Optional[List[int]]
    video_cameras: List[VideoCamera]
    realsense_cameras: List[RealSenseCamera]
    zmq_cameras: List[ZMQCamera]

    camera_ids: List[int]
    camera_names: List[str]
    _main_camera: Optional[BaseCamera] = None
    # If it's None, record everything. Otherwise, record only the corresponding cameras
    _cameras_ids_to_record: List[int]
    _is_detecting: bool = False

    def __init__(self, disabled_cameras: Optional[List[int]] = None):
        """
        AllCameras class to manage all cameras connected to the computer.
        Args:
            disabled_cameras: These cameras indexes will not be used by the application, set to [-1] to disable all cameras
        """
        if disabled_cameras is not None:
            self.disabled_cameras = disabled_cameras
        else:
            self.disabled_cameras = []

        self.detect_cameras()

        # Add atexit hook to stop the cameras
        atexit.register(self.stop)

    @property
    def cameras(self) -> List[BaseCamera]:
        """
        Returns a list of all cameras ordered as: video cameras by camera_id,
        realsense cameras by device_index, then zmq cameras in order added.
        """
        all_cameras: List[BaseCamera] = []

        # Video cameras first, ordered by camera_id
        sorted_video_cameras = sorted(
            self.video_cameras,
            key=lambda cam: cam.camera_id if cam.camera_id is not None else 0,
        )
        all_cameras.extend(sorted_video_cameras)

        # RealSense cameras second, ordered by device_index
        sorted_realsense_cameras = sorted(
            self.realsense_cameras, key=lambda cam: cam.device_index
        )
        all_cameras.extend(sorted_realsense_cameras)

        # ZMQ cameras last, in order they were added
        all_cameras.extend(self.zmq_cameras)

        return all_cameras

    def detect_cameras(self):
        """
        Detect all cameras connected to the computer and initialize them.
        """
        from phosphobot.configs import config

        if self._is_detecting:
            logger.warning("Cameras are already being detected, skipping")
            return

        self._is_detecting = True
        self.video_cameras = []
        self.camera_ids = []
        self.camera_names = []
        self._cameras_ids_to_record = []
        self.realsense_cameras = []
        self.zmq_cameras = []

        if not config.ENABLE_CAMERAS:
            logger.warning("Cameras are disabled")
            self.disabled_cameras = list(range(config.MAX_OPENCV_INDEX))
            return

        camera_names = get_camera_names()
        self.initialize_realsense_camera()

        # Get the available video indexes from a range of 0 to config.MAX_OPENCV_INDEX
        possible_camera_ids = detect_video_indexes()

        # For every of these index we will try to detect the camera type
        # If it corresponds to a classic or stereo camera, we initialize the camera accordingly
        for index in possible_camera_ids:
            camera_type = detect_camera_type(
                index=index,
                camera_names=camera_names,
                possible_camera_ids=possible_camera_ids,
            )
            if camera_type == "classic":
                # TODO: Do not hardcode the width, height and fps
                self.video_cameras.append(
                    VideoCamera(
                        video=cv2.VideoCapture(index),
                        disable=self.disabled_cameras is not None
                        and index in self.disabled_cameras,
                        camera_id=index,
                    )
                )
                self.camera_ids.append(index)
            # TODO: Support multiple stereo cameras
            elif camera_type == "stereo":
                stereo_camera = StereoCamera(
                    video=cv2.VideoCapture(index),
                    disable=self.disabled_cameras is not None
                    and index in self.disabled_cameras,
                    camera_id=index,
                )
                # Set the camera_id to the first position and reindex
                # the others
                stereo_camera.camera_id = 0
                self.video_cameras = [stereo_camera] + self.video_cameras
                self.camera_ids = [0] + self.camera_ids
                for i, camera_id in enumerate(self.camera_ids[1:]):
                    self.camera_ids[i + 1] = camera_id + 1

                # self.video_cameras.append(stereo_camera)
                # self.camera_ids.append(index)
            elif camera_type == "dummy":
                self.video_cameras.append(DummyCamera(camera_type="dummy"))
                self.camera_ids.append(index)
            elif camera_type == "dummy_stereo":
                self.video_cameras.append(
                    DummyCamera(camera_type="dummy_stereo", width=1280, height=480)
                )
                self.camera_ids.append(index)
            else:
                logger.debug(f"Ignoring camera {index}: {camera_type}")

        # Create virtual cameras for each RealSense device
        if len(self.realsense_cameras) > 0 and config.ENABLE_REALSENSE:
            # Generate unique camera IDs for virtual cameras
            max_id = max(self.camera_ids) if self.camera_ids else -1

            for device_index, realsense_camera in enumerate(self.realsense_cameras):
                if not realsense_camera.is_connected or not realsense_camera.is_active:
                    logger.debug(f"Skipping inactive RealSense device {device_index}")
                    continue

                # Generate unique IDs for this device's virtual cameras
                virtual_rgb_id = max_id + 1
                virtual_depth_id = max_id + 2
                max_id += 2  # Increment for next device

                # Check if virtual cameras are disabled
                disabled = (
                    self.disabled_cameras if self.disabled_cameras is not None else []
                )
                virtual_rgb_disabled = virtual_rgb_id in disabled
                virtual_depth_disabled = virtual_depth_id in disabled

                # Create virtual cameras for this RealSense device
                virtual_rgb = RealSenseVirtualCamera(
                    realsense_camera,
                    "rgb",
                    virtual_rgb_id,
                    disable=virtual_rgb_disabled,
                )
                virtual_depth = RealSenseVirtualCamera(
                    realsense_camera,
                    "depth",
                    virtual_depth_id,
                    disable=virtual_depth_disabled,
                )

                # Add to video cameras and camera IDs
                self.video_cameras.extend([virtual_rgb, virtual_depth])
                self.camera_ids.extend([virtual_rgb_id, virtual_depth_id])

                logger.info(
                    f"Added virtual cameras for RealSense device {device_index} "
                    f"(RGB: {virtual_rgb_id}, Depth: {virtual_depth_id})"
                )

        self._cameras_ids_to_record = self.camera_ids
        self._is_detecting = False

    def add_custom_camera(self, camera: BaseCamera):
        """
        Manually adds an initialized custom camera instance to the list of active cameras.
        This is useful for adding virtual or non-discoverable cameras like ZMQCamera.

        Args:
            camera: An instance of a class that inherits from BaseCamera.
        """

        # According to the camera type, add it to the appropriate list
        if isinstance(camera, VideoCamera):
            # Assign camera_id if not already set
            if camera.camera_id is None:
                max_id = max(self.camera_ids) if self.camera_ids else -1
                camera.camera_id = max_id + 1

            self.video_cameras.append(camera)
            self.camera_ids.append(camera.camera_id)

        elif isinstance(camera, RealSenseCamera):
            self.realsense_cameras.append(camera)

            # Create virtual cameras for the RealSense device if it's active
            if camera.is_connected and camera.is_active:
                # Generate unique camera IDs for virtual cameras
                max_id = max(self.camera_ids) if self.camera_ids else -1
                virtual_rgb_id = max_id + 1
                virtual_depth_id = max_id + 2

                # Create virtual cameras for this RealSense device
                virtual_rgb = RealSenseVirtualCamera(
                    camera,
                    "rgb",
                    virtual_rgb_id,
                    disable=False,
                )
                virtual_depth = RealSenseVirtualCamera(
                    camera,
                    "depth",
                    virtual_depth_id,
                    disable=False,
                )

                # Add to video cameras and camera IDs
                self.video_cameras.extend([virtual_rgb, virtual_depth])
                self.camera_ids.extend([virtual_rgb_id, virtual_depth_id])

                logger.info(
                    f"Added virtual cameras for RealSense device (RGB: {virtual_rgb_id}, Depth: {virtual_depth_id})"
                )

        elif isinstance(camera, ZMQCamera):
            # Assign camera_id if not already set
            if camera.camera_id is None:
                max_id = max(self.camera_ids) if self.camera_ids else -1
                camera.camera_id = max_id + 1

            self.zmq_cameras.append(camera)
            self.camera_ids.append(camera.camera_id)

        else:
            raise ValueError(
                "Custom camera must be an instance of VideoCamera, RealSenseCamera, or ZMQCamera"
            )

        # Add the camera name to the list
        self.camera_names.append(camera.camera_name)
        logger.info(
            f"Custom camera added: {camera.camera_name} Type: {camera.camera_type} ID: {getattr(camera, 'camera_id', 'N/A')}"
        )

    def refresh(self) -> None:
        """
        Refresh the list of cameras.
        This will reinitialize the cameras and update the camera_ids.
        """
        # First, stop all cameras
        self.stop()
        time.sleep(0.1)
        # Then, reinitialize the cameras
        self.detect_cameras()

    @property
    def has_connected_camera(self) -> bool:
        """
        Return True if at least one camera is active. False otherwise.
        """
        return any(camera.is_active for camera in self.cameras)

    def initialize_realsense_camera(self, max_retries: int = 3) -> None:
        """
        Initialize all available RealSense cameras with automatic retries on failure.
        Creates a list of RealSenseCamera instances, one for each connected device.
        """
        self.realsense_cameras = []

        if not config.ENABLE_REALSENSE:
            logger.debug("Realsense cameras are disabled")
            return

        if not REALSENSE_AVAILABLE:
            logger.debug(
                "pyrealsense2 is not available, RealSense cameras cannot be initialized"
            )
            return

        # Get all available RealSense devices
        ctx = rs.context()
        realsense_devices = ctx.query_devices()
        device_count = realsense_devices.size()

        if device_count == 0:
            logger.info("No RealSense devices detected")
            return

        logger.info(f"Found {device_count} RealSense device(s)")

        # Initialize each device
        for device_index in range(device_count):
            device = realsense_devices[device_index]
            device_serial = device.get_info(rs.camera_info.serial_number)
            device_name = device.get_info(rs.camera_info.name)

            logger.debug(
                f"Attempting to initialize RealSense device {device_index}: {device_name} (Serial: {device_serial})"
            )

            # Check if this specific device should be disabled
            device_disabled = (
                self.disabled_cameras is not None
                and -1 in self.disabled_cameras  # All cameras disabled
            )

            for attempt in range(max_retries):
                try:
                    realsense_camera = RealSenseCamera(
                        device_serial=device_serial,
                        device_index=device_index,
                        disable=device_disabled,
                    )

                    if realsense_camera.is_connected and realsense_camera.is_active:
                        self.realsense_cameras.append(realsense_camera)
                        logger.info(
                            f"RealSense camera {device_index} initialized: {device_name} (Serial: {device_serial})"
                        )
                        break
                    else:
                        logger.warning(
                            f"RealSense camera {device_index} failed to connect properly"
                        )

                except Exception as e:
                    logger.warning(
                        f"RealSense device {device_index} attempt {attempt + 1}/{max_retries} failed: {str(e)}"
                    )
                    if attempt < max_retries - 1:
                        time.sleep(1)
                    else:
                        logger.error(
                            f"Failed to initialize RealSense device {device_index} after {max_retries} attempts"
                        )

        if len(self.realsense_cameras) == 0:
            logger.info("No RealSense cameras initialized")
        else:
            logger.info(
                f"Successfully initialized {len(self.realsense_cameras)} RealSense camera(s)"
            )

    def status(self) -> AllCamerasStatus:
        """
        Return information about the status of all cameras.

        This is used to setup video in the app.
        """

        # Don't show realsense (deprecated usage)
        realsense_available = False

        return AllCamerasStatus(
            video_cameras_ids=self.camera_ids,
            realsense_available=realsense_available,
            is_stereo_camera_available=any(
                camera.camera_type == "stereo" for camera in self.cameras
            ),
            cameras_status=[
                SingleCameraStatus(
                    camera_id=camera.camera_id,
                    camera_type=camera.camera_type,
                    is_active=camera.is_active,
                    width=camera.width,
                    height=camera.height,
                    fps=camera.fps,
                )
                for camera in self.cameras
                if hasattr(camera, "camera_id") and camera.camera_id is not None
            ],
        )

    def stop(self):
        for camera in self.cameras:
            camera.stop()

    def get_camera_by_id(self, id: int) -> Optional[BaseCamera]:
        if id not in self.camera_ids:
            logger.warning(f"Camera with id {id} not available in {self.camera_ids}")
            return None

        for camera in self.cameras:
            if hasattr(camera, "camera_id") and camera.camera_id == id:
                return camera

        logger.warning(f"Camera with id {id} not available in {self.camera_ids}")
        return None

    def get_rgb_frame(
        self,
        camera_id: Optional[int] = None,
        resize: Optional[Tuple[int, int]] = None,
    ) -> Optional[cv2.typing.MatLike]:
        """
        Return the latest RGB frame from the specidied camera
        To use the realsense camera, set realsense to True
        If no Camera is specified, the first video camera recognized by OpenCV will be used

        raise ValueError if both camera_id and realsense are specified
        """
        rgb_camera: Optional[BaseCamera]

        if camera_id is not None:
            rgb_camera = self.get_camera_by_id(camera_id)
        else:
            logger.warning("No camera specified, the first video camera will be used")
            rgb_camera = self.get_camera_by_id(0)

        if rgb_camera is None:
            logger.warning(f"No camera found for camera_id={camera_id}")
            return None

        frame = rgb_camera.get_rgb_frame(resize=resize)

        return frame

    def get_rgb_frames_for_all_cameras(
        self, resize: Optional[Tuple[int, int]] = None
    ) -> Dict[str, Optional[cv2.typing.MatLike]]:
        """
        Get the RGB frames for all the cameras.

        Returns a dict with the camera id as key and the frame as value.
        """
        frames: Dict[str, Optional[cv2.typing.MatLike]] = {}
        for camera in self.video_cameras:
            if camera.camera_type == "stereo":
                camera = cast(StereoCamera, camera)
                left_side = camera.get_left_eye_rgb_frame(resize=resize)
                right_side = camera.get_right_eye_rgb_frame(resize=resize)
                frames[f"camera_{len(frames)}"] = left_side
                frames[f"camera_{len(frames)}"] = right_side
            elif camera.camera_type == "classic":
                frame = camera.get_rgb_frame(resize=resize)
                frames[f"camera_{len(frames)}"] = frame
            else:
                logger.warning(f"Unknown camera type: {camera.camera_type}")
        return frames

    @property
    def cameras_ids_to_record(self) -> List[int]:
        """
        Get the camera ids to record.
        """
        return self._cameras_ids_to_record

    @cameras_ids_to_record.setter
    def cameras_ids_to_record(self, camera_ids: Optional[List[int]]):
        """
        Set the camera ids to record.
        """
        # Remove the cached _main_camera, because it depends on the selected camera ids
        self._main_camera = None
        # Set the camera ids to record to be the intersection with available camera ids
        if camera_ids is not None:
            self._cameras_ids_to_record = list(set(camera_ids) & set(self.camera_ids))
        else:
            # Set to all cameras available
            self._cameras_ids_to_record = self.camera_ids

    @property
    def main_camera(self) -> Optional[BaseCamera]:
        """
        Get the main camera among the selected camera ids.
        If selected camera ids are not provided, we select amoung all available cameras.
        The id of the main camera is set in the config file.
        By default, the main camera is the first camera.
        """

        if self._main_camera is None:
            # The main camera is the one designated in the config or the first available index
            if (
                config.MAIN_CAMERA_ID is not None
                and config.MAIN_CAMERA_ID in self.cameras_ids_to_record
            ):
                self._main_camera = self.get_camera_by_id(config.MAIN_CAMERA_ID)
            elif len(self.cameras_ids_to_record) > 0:
                self._main_camera = self.get_camera_by_id(self.cameras_ids_to_record[0])
            else:
                # No camera detected
                logger.warning("No camera detected")
                self._main_camera = None

        return self._main_camera

    @main_camera.setter
    def main_camera(self, camera: BaseCamera):
        """
        Explicitly set the main camera.
        """
        self._main_camera = camera

    def get_main_camera_frames(
        self,
        target_video_size: Tuple[int, int],
    ) -> Optional[List[cv2.typing.MatLike]]:
        """
        Get the frames from the main camera.
        """
        if self.main_camera is None:
            return None

        if self.main_camera.camera_type == "stereo" and isinstance(
            self.main_camera, StereoCamera
        ):
            # If the main camera is a stereo camera, return the left and right eye frames
            frames = [
                self.main_camera.get_left_eye_rgb_frame(resize=target_video_size),
                self.main_camera.get_right_eye_rgb_frame(resize=target_video_size),
            ]
        else:
            # Otherwise, return the frame from the main camera
            frames = [self.main_camera.get_rgb_frame(resize=target_video_size)]

        # Remove None values
        return [f for f in frames if f is not None]

    def get_secondary_camera_ids(self) -> List[int]:
        """
        Get the camera ids for all the cameras selected except the main camera.
        """

        return [
            camera_id
            for camera_id in self.cameras_ids_to_record
            if self.get_camera_by_id(camera_id) != self.main_camera
        ]

    def get_secondary_cameras(self) -> List[BaseCamera]:
        """
        Get the camera objects for all the cameras except the main camera.
        """
        # All cameras except the main are secondary.
        cameras = cast(
            List[BaseCamera],
            [
                camera
                for camera in self.video_cameras
                if (
                    camera != self.main_camera
                    and camera.camera_id in self.cameras_ids_to_record
                )
            ],
        )
        return cameras

    def get_secondary_camera_frames(
        self,
        target_video_size: Tuple[int, int],
    ) -> List[cv2.typing.MatLike]:
        """
        Get the frames from every camera except the main camera.
        """

        # Get the frames from all the cameras
        frames = []
        for camera in self.get_secondary_cameras():
            if camera.camera_type == "stereo" and isinstance(camera, StereoCamera):
                # If the camera is a stereo camera, return the left and right eye frames
                frames.extend(
                    [
                        camera.get_left_eye_rgb_frame(resize=target_video_size),
                        camera.get_right_eye_rgb_frame(resize=target_video_size),
                    ]
                )

            else:
                # Otherwise, return the frame from the camera
                frames.append(camera.get_rgb_frame(resize=target_video_size))

        return [frame for frame in frames if frame is not None]

    def get_secondary_camera_key_names(self) -> List[str]:
        """
        Get the keys for the secondary cameras.
        """
        secondary_camera_key_names: list[str] = []
        camera_key_index = 0

        # If the main camera is a stereo camera, the first camera secondary key is the right eye
        if (
            self.main_camera is not None
            and self.main_camera.camera_type == "stereo"
            and isinstance(self.main_camera, StereoCamera)
        ):
            secondary_camera_key_names.append(
                f"observation.images.secondary_{camera_key_index}"
            )
            camera_key_index += 1

        # If we detect a stereo camera, we must add a key for the additional frame (left/right)
        for camera in self.get_secondary_cameras():
            if camera.camera_type == "stereo" and isinstance(camera, StereoCamera):
                secondary_camera_key_names.extend(
                    [
                        f"observation.images.secondary_{camera_key_index}",
                        f"observation.images.secondary_{camera_key_index + 1}",
                    ]
                )
                camera_key_index += 2
            else:
                secondary_camera_key_names.append(
                    f"observation.images.secondary_{camera_key_index}"
                )
                camera_key_index += 1

        return secondary_camera_key_names

    def get_all_camera_key_names(self) -> List[str]:
        """
        Get the keys for all the cameras, including the main camera.
        """
        camera_key_names = []

        # Add main camera key
        if self.main_camera is not None:
            camera_key_names.append("observation.images.main")

        # Add secondary cameras keys
        camera_key_names.extend(self.get_secondary_camera_key_names())

        return camera_key_names


@lru_cache()
def get_all_cameras() -> AllCameras:
    """
    Return the global AllCameras instance.
    """
    global cameras

    if not cameras:
        cameras = AllCameras(disabled_cameras=config.DEFAULT_CAMERAS_TO_DISABLE)

    return cameras


def get_all_cameras_no_init() -> Optional[AllCameras]:
    """
    Return the global AllCameras instance without initializing it.
    This is useful for testing purposes.
    """
    global cameras

    return cameras
