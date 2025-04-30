import base64
import functools
import inspect
import json
import os
import traceback
from pathlib import Path
from typing import Annotated, Any, Tuple, Union

import av
import cv2
import numpy as np
import pandas as pd
from huggingface_hub import HfApi
from loguru import logger
from pydantic import BeforeValidator, PlainSerializer

from phosphobot.types import VideoCodecs


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""

    def default(self, obj):
        logger.info(f"Encoding with NumpyEncoder object type({type(obj)}) {obj}")
        if isinstance(obj, np.ndarray):
            logger.debug(f"Encoding NumpyEncoder numpy array of shape {obj.shape}")
            return {
                "_type": "ndarray",
                "data": obj.tolist(),
                "dtype": str(obj.dtype),
                "shape": obj.shape,
            }
        return json.JSONEncoder.default(self, obj)


def decode_numpy(dct: dict) -> Union[np.ndarray, dict]:
    """Custom decoder. Reads the encoded numpy array from the JSON file."""
    if isinstance(dct, dict) and dct.get("_type") == "ndarray":
        return np.array(dct["data"], dtype=np.dtype(dct["dtype"]))
    if "__numpy__" in dct:
        data = base64.b64decode(dct["__numpy__"])
        arr = np.frombuffer(data, dtype=np.dtype(dct["dtype"]))
        arr = arr.reshape(dct["shape"])
        return arr
    return dct


def nd_array_custom_before_validator(x: Any) -> np.ndarray:
    # custom before validation logic
    if isinstance(x, list):
        return np.array(x)
    elif isinstance(x, np.ndarray):
        return x
    elif isinstance(x, float) or isinstance(x, int):
        # Convert scalar to 1D array
        # This shouldn't happen with normal serialization, but can happen due to bugs.
        return np.array([x])
    else:
        raise ValueError("Invalid type for numpy array")


def nd_array_custom_serializer(x: np.ndarray):
    # custom serialization logic: convert to list
    return x.tolist()


# Custom type for numpy array. Stores np.ndarray as list in JSON.
NdArrayAsList = Annotated[
    np.ndarray,
    BeforeValidator(nd_array_custom_before_validator),
    PlainSerializer(nd_array_custom_serializer, return_type=list),
]


def create_video_file(
    frames: np.ndarray,
    target_size: Tuple[int, int],
    output_path: str,
    fps: float,
    codec: VideoCodecs,
) -> Union[str, Tuple[str, str]]:
    """
    Create a video file from a 4D numpy array of frames and resize to target size.
    For stereo cameras (aspect ratio >= 8/3), creates separate left and right video files.

    Args:
        frames (np.ndarray): Array of shape (N, height, width, 3) in RGB format.
        target_size (Tuple[int, int]): Target dimensions (width, height) for the output video.
        output_path (str): Path to save the video file.
        fps (float): Frames per second for the video.
        codec (str): Codec name for PyAV (e.g. "mpeg4", "h264").

    Returns:
        Union[str, Tuple[str, str]]: Path(s) to created video file(s). Returns tuple of paths for stereo.

    Raises:
        ValueError: If frames array is empty or has incorrect shape.
        RuntimeError: If writing fails unexpectedly.
    """
    # Map FourCC-style codec literals to PyAV codec names
    CODEC_MAP = {
        "avc1": "h264",
        "avc3": "h264",
        "mp4v": "mpeg4",
        "hev1": "hevc",
        "hvc1": "hevc",
        "av01": "av1",
        "vp09": "vp9",
    }
    codec_av = CODEC_MAP.get(codec, codec)
    logger.info(f"Using codec: {codec}")

    # Validate input array
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(
            f"Frames must be a 4D array with shape (N, H, W, 3), got {frames.shape}"
        )
    num_frames, h, w, _ = frames.shape
    if num_frames == 0:
        raise ValueError("Frames array is empty (N=0)")

    aspect_ratio = w / h
    is_stereo = aspect_ratio >= 8 / 3
    logger.info(f"Stereo={is_stereo}, aspect_ratio={aspect_ratio:.2f}")

    def open_container(path: str, size: Tuple[int, int]):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        container = av.open(path, mode="w")

        # pick encoder options based on codec
        encoder_opts: dict[str, str] = {}
        if codec_av in ("h264", "mpeg4", "hevc"):
            # CRF = quality (lower = better), preset = speed/efficiency trade-off
            encoder_opts = {"crf": "18", "preset": "slow"}
        elif codec_av == "av1":
            # AV1 needs slightly higher CRF to match visually (~30),
            # and cpu-used trades speed vs. quality (0=slowest/best)
            encoder_opts = {
                "crf": "30",
                "cpu-used": "4",
                "row-mt": "1",  # multi-threading
                "tile-columns": "2",  # parallel tile encoding
            }
        elif codec_av == "vp9":
            # VP9: crf + speed (0=best, 5=fastest)
            encoder_opts = {"crf": "30", "speed": "1"}
        elif codec_av == "mpeg4":
            # old MPEG-4 Part 2: no CRF, use qscale OR fixed bitrate
            # Lower qscale = better quality. 2–5 is a good range.
            encoder_opts = {"qscale": "2"}
        # else: leave encoder_opts empty for codecs that don’t support these flags

        stream = container.add_stream(
            codec_av,
            rate=fps,
            options=encoder_opts or None,  # type: ignore
        )  # type: ignore
        # Force a minimum bitrate for mpeg4 to avoid artifacts
        if codec_av == "mpeg4":
            stream.bit_rate = 5_000_000  # ~5 Mb/s

        stream.width, stream.height = size
        stream.pix_fmt = "yuv420p"
        return container, stream

    def process_and_encode(frame: np.ndarray, stream, container, size: Tuple[int, int]):
        # Convert to uint8 RGB if needed
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        # Wrap as PyAV frame and resize/convert
        video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        video_frame = video_frame.reformat(
            width=size[0], height=size[1], format="yuv420p"
        )
        for packet in stream.encode(video_frame):
            container.mux(packet)

    if is_stereo:
        size = (target_size[0] // 2, target_size[1])
        base, suffix = output_path.rsplit("/episode", 1)
        left_path = f"{base}.left/episode{suffix}"
        right_path = f"{base}.right/episode{suffix}"

        left_ct = right_ct = None
        try:
            left_ct, left_stream = open_container(left_path, size)
            right_ct, right_stream = open_container(right_path, size)

            mid_w = w // 2
            for frame in frames:
                left_frame = frame[:, :mid_w, :]
                right_frame = frame[:, mid_w:, :]
                process_and_encode(left_frame, left_stream, left_ct, size)
                process_and_encode(right_frame, right_stream, right_ct, size)

            # flush encoders
            for packet in left_stream.encode():
                left_ct.mux(packet)
            for packet in right_stream.encode():
                right_ct.mux(packet)

            return left_path, right_path

        except Exception:
            logger.error("Error writing stereo video", exc_info=True)
            raise
        finally:
            if left_ct:
                left_ct.close()
            if right_ct:
                right_ct.close()

    else:
        size = target_size
        container = None
        try:
            container, stream = open_container(output_path, size)
            for frame in frames:
                process_and_encode(frame, stream, container, size)

            # flush encoder
            for packet in stream.encode():
                container.mux(packet)

            return output_path

        except Exception:
            logger.error("Error writing video", exc_info=True)
            raise
        finally:
            if container:
                container.close()


def get_home_app_path() -> Path:
    """
    Return the path to the app's folder in the user's home directory.
    This is used to store user-specific data.

    It's platform dependent.

    user_home/
        phosphobot/
            calibration/
            recordings/
            ...
    """
    home_path = Path.home() / "phosphobot"
    # Create the folder if it doesn't exist
    home_path.mkdir(parents=True, exist_ok=True)
    # Create subfolders
    (home_path / "calibration").mkdir(parents=True, exist_ok=True)
    (home_path / "recordings").mkdir(parents=True, exist_ok=True)
    return home_path


def compute_sum_squaresum_framecount_from_video(
    video_path: str,
    raise_if_not_found: bool = False,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Process a video file and calculate the sum of RGB values and sum of squares of RGB values for each frame.
    Returns a list of np.ndarray corresponding respectively to the sum of RGB values, sum of squares of RGB values and nb_pixel.
    We divide by 255.0 RGB values to normalize the values to the range [0, 1].
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        if raise_if_not_found:
            raise FileNotFoundError(
                f"Error: Could not open video at path: {video_path}"
            )
        logger.warning(
            f"Error: Could not open video at path: {video_path}. Returning 0."
        )
        return (np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32), 0)

    nb_pixel = 0
    total_sum_rgb = np.zeros(3, dtype=np.float32)  # To store sum of RGB values
    total_sum_squares = np.zeros(
        3, dtype=np.float32
    )  # To store sum of squares of RGB values
    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        # If the frame was not read successfully, break the loop
        if not ret:
            break

        # Convert the frame from BGR to RGB (OpenCV uses BGR by default)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
        # Calculate the sum of RGB values for the frame
        sum_rgb = np.sum(frame_rgb, axis=(0, 1))
        # Calculate the sum of squares of RGB values for the frame
        sum_squares = np.sum(frame_rgb**2, axis=(0, 1))
        # Accumulate the sums
        # Cannot cast ufunc 'add' output from dtype('float64') to dtype('uint64') with casting rule 'same_kind'
        total_sum_rgb = total_sum_rgb + sum_rgb
        total_sum_squares = total_sum_squares + sum_squares

        # nb Pixel
        nb_pixel += frame_rgb.shape[0] * frame_rgb.shape[1]

    # Release the video capture object
    # TODO: If problem of dimension maybe transposing arrays is needed.
    cap.release()
    return (total_sum_rgb, total_sum_squares, nb_pixel)


def get_field_min_max(df: pd.DataFrame, field_name: str) -> tuple:
    """
    Compute the minimum value for the given field in the DataFrame.

    If the field values are numeric, returns a scalar minimum.
    If the field values are lists/arrays, returns an element-wise minimum array.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        field_name (str): The name of the field/column to compute the min,max for.

    Returns:
        The minimum value(s) for the specified field.

    Raises:
        ValueError: If the field does not exist or if list/array values have inconsistent shapes.
    """
    if field_name not in df.columns:
        raise ValueError(f"Field '{field_name}' not found in DataFrame")

    # Get a sample value (skip any nulls)
    sample_value = df[field_name].dropna().iloc[0]

    # If the field values are lists or arrays, compute element-wise min, max
    if isinstance(sample_value, (list, np.ndarray)):
        # Check that df[field_name].values is a np.ndarray
        array_values = df[field_name].values

        # No overload variant of "vstack" matches argument type "ndarray[Any, Any]"
        return (
            np.min(np.vstack(array_values), axis=0),  # type: ignore
            np.max(np.vstack(array_values), axis=0),  # type: ignore
        )
    else:
        # Otherwise, assume the field is numeric and return the scalar min.
        return (df[field_name].min(), df[field_name].max())


def parse_hf_username_or_orgid(user_info: dict) -> str | None:
    """
    Extract the username or organization name from the user info dictionary.
    user_info = api.whoami(token=hf_token)
    """
    # Extract the username
    username = user_info.get("name", "Unknown")

    # If no fine grained permissions, return the username
    if user_info.get("auth", {}).get("accessToken", {}).get("role") == "write":
        return username

    # Extract fine-grained permissions
    fine_grained_permissions = (
        user_info.get("auth", {}).get("accessToken", {}).get("fineGrained", {})
    )
    scoped_permissions = fine_grained_permissions.get("scoped", [])

    # Check if the token has write access to the user account
    org_with_write_access = None

    for scope in scoped_permissions:
        entity = scope.get("entity", {})
        entity_type = entity.get("type")
        entity_name = entity.get("name")
        permissions = scope.get("permissions", [])

        # Check if the entity is the user and has write access
        if entity_type == "user" and "repo.write" in permissions:
            # Return the username
            return username

        # Check if the entity is an org and has write access
        if entity_type == "org" and "repo.write" in permissions:
            org_with_write_access = entity_name
            return org_with_write_access

    logger.warning(
        "No user or org with write access found. Wont be able to push to Hugging Face."
    )

    return None


def get_hf_username_or_orgid() -> str | None:
    """
    Returns the username or organization name from the Hugging Face token file.
    Returns None if we can't write anywhere.
    """
    token_file = get_home_app_path() / "huggingface.token"

    # Check the file exists
    if not token_file.exists():
        logger.info("Token file not found.")
        return None

    with open(token_file, "r") as file:
        hf_token = file.read().strip()
    if hf_token:
        try:
            api = HfApi()
            user_info = api.whoami(token=hf_token)
            # Get the username or org where we have write access
            username_or_orgid = parse_hf_username_or_orgid(user_info)
            return username_or_orgid
        except Exception as e:
            logger.warning(f"Error logging in to Hugging Face: {e}")
            return None
    else:
        logger.warning(f"Empty token file: {token_file}. Won't push to HuggingFace.")
        return None


def get_hf_token() -> str | None:
    """
    Returns the hf token from the token file.
    Returns None if there is no token.
    """
    token_file = get_home_app_path() / "huggingface.token"

    # Check the file exists
    if not token_file.exists():
        logger.info("Token file not found.")
        return None

    with open(token_file, "r") as file:
        hf_token = file.read().strip()
    if hf_token:
        return hf_token
    else:
        logger.warning(f"Empty token file: {token_file}.")
        return None


def background_task_log_exceptions(func):
    """
    Decorator to log exceptions in background tasks (works for both sync/async functions).
    Otherwise, the exception is silently swallowed.
    """

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Background task error: {str(e)}\n{traceback.format_exc()}")
            raise

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Background task error: {str(e)}\n{traceback.format_exc()}")
            raise

    if inspect.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper
