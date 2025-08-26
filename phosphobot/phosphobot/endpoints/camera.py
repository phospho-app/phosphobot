import asyncio
import base64
from typing import Dict, Optional

import cv2
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
)
from fastapi.responses import StreamingResponse
from loguru import logger

from phosphobot.camera import AllCameras, get_all_cameras, ZMQCamera
from phosphobot.models import AddZMQCameraRequest

router = APIRouter(tags=["camera"])


@router.get(
    "/video/{camera_id}",
    response_class=StreamingResponse,
    description="Stream video feed of the specified camera. "
    + "If no camera id is provided, the default camera is used. "
    + "Specify a target size and quality using query parameters.",
    responses={
        200: {"description": "Streaming video feed of the specified camera."},
        404: {"description": "Camera not available"},
    },
)
def video_feed_for_camera(
    request: Request,
    camera_id: Optional[int],
    height: Optional[int] = None,
    width: Optional[int] = None,
    quality: Optional[int] = None,
    cameras: AllCameras = Depends(get_all_cameras),
):
    """
    Stream video feed of the specified camera.
    """

    if width is None or height is None:
        target_size = None
    else:
        target_size = (width, height)
    logger.debug(
        f"Received request for camera {camera_id} with target size {target_size} and quality {quality}"
    )
    if camera_id is None:
        camera_id = 0

    # Convert to integer the parameter if read as a string
    if isinstance(camera_id, str) and camera_id.isdigit():
        camera_id = int(camera_id)

    if not (isinstance(camera_id, int) or isinstance(camera_id, str)):
        raise HTTPException(
            status_code=400,
            detail=f"Unprocessable type for camera id. Received {type(camera_id)}",
        )

    if quality and (quality < 0 or quality > 100):
        raise HTTPException(
            status_code=400,
            detail=f"Quality must be between 0 and 100. Received {quality}",
        )

    stream_params = {
        "target_size": target_size,
        "quality": quality,
    }

    camera = cameras.get_camera_by_id(camera_id)
    if camera is None or not camera.is_active:
        raise HTTPException(status_code=404, detail="Camera not available")
    logger.info(f"Starting video feed with params {stream_params}")
    return StreamingResponse(
        camera.generate_rgb_frames(
            target_size=target_size, quality=quality, request=request
        ),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get(
    "/frames",
    response_model=Dict[str, Optional[str]],
    description="Capture frames from all available cameras. "
    + "Returns a dictionary with camera IDs as keys and base64 encoded JPG images as values. "
    + "If a camera is not available or fails to capture, its value will be None.",
    responses={
        200: {
            "description": "Successfully captured frames from available cameras",
            "content": {
                "application/json": {
                    "example": {
                        "0": "base64_encoded_image_string",
                        "1": None,
                        "realsense": "base64_encoded_image_string",
                    }
                }
            },
        },
        500: {"description": "Server error while capturing frames"},
    },
)
async def get_all_camera_frames(
    cameras: AllCameras = Depends(get_all_cameras),
) -> Dict[str, Optional[str]]:
    """
    Capture and return frames from all available cameras.
    Returns:
        Dict[str, Optional[str]]: Dictionary mapping camera IDs to base64 encoded JPG images
        or None if camera is unavailable/failed to capture
    """
    logger.debug("Received request for all camera frames")

    # We can add a resize here if needed
    frames = cameras.get_rgb_frames_for_all_cameras()

    # Initialize response dictionary
    response: Dict[str, Optional[str]] = {}

    # Process each frame
    for camera_id, frame in frames.items():
        try:
            if frame is None:
                response[camera_id] = None
                continue

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Encode frame as JPG
            _, buffer = cv2.imencode(".jpg", rgb_frame)

            # Convert to base64 string
            base64_frame = base64.b64encode(buffer.tobytes()).decode("utf-8")

            response[camera_id] = base64_frame

        except Exception as e:
            logger.error(f"Error processing frame for camera {camera_id}: {str(e)}")
            response[camera_id] = None

    if not response:
        raise HTTPException(status_code=503, detail="No camera frames available")

    return response


@router.post(
    "/cameras/refresh",
    response_model=dict,
    description="Refresh the list of available cameras. "
    + "This operation can take a few seconds as it disconnects and reconnects to all cameras. "
    + "It is useful when cameras are added or removed while the application is running.",
)
async def refresh_camera_list(
    cameras: AllCameras = Depends(get_all_cameras),
):
    """
    Refresh the list of available cameras.
    This operation can take a few seconds as it disconnects and reconnects to all cameras.
    It is useful when cameras are added or removed while the application is running.
    """
    cameras.refresh()
    return {"message": "Camera list refreshed successfully"}


@router.post(
    "/cameras/add-zmq",
    response_model=dict,
    description="Add a camera feed from a ZMQ publisher. ",
)
async def add_zmq_camera_feed(
    query: AddZMQCameraRequest,
    cameras: AllCameras = Depends(get_all_cameras),
):
    """
    Add a camera feed from a ZMQ publisher.
    This allows the application to receive camera frames from a ZMQ publisher.
    """
    try:
        zmq_camera = ZMQCamera(connect_to=query.tcp_address, topic=query.topic)
        await asyncio.sleep(0.1)  # Allow some time for the camera to initialize
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to connect to ZMQ publisher at {query.tcp_address}: {str(e)}",
        )

    # Add to cameras
    cameras.add_custom_camera(zmq_camera)
    return {"message": "ZMQ camera added successfully"}
