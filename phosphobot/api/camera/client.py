# This file was auto-generated by Fern from our API Definition.

from ..core.client_wrapper import SyncClientWrapper
import typing
from ..core.request_options import RequestOptions
from ..core.pydantic_utilities import parse_obj_as
from ..errors.internal_server_error import InternalServerError
from json.decoder import JSONDecodeError
from ..core.api_error import ApiError
from .types.video_feed_for_camera_video_camera_id_get_request_camera_id import (
    VideoFeedForCameraVideoCameraIdGetRequestCameraId,
)
from ..core.jsonable_encoder import jsonable_encoder
from ..errors.not_found_error import NotFoundError
from ..errors.unprocessable_entity_error import UnprocessableEntityError
from ..types.http_validation_error import HttpValidationError
from ..core.client_wrapper import AsyncClientWrapper


class CameraClient:
    def __init__(self, *, client_wrapper: SyncClientWrapper):
        self._client_wrapper = client_wrapper

    def get_all_camera_frames(
        self, *, request_options: typing.Optional[RequestOptions] = None
    ) -> typing.Dict[str, typing.Optional[str]]:
        """
        Capture frames from all available cameras. Returns a dictionary with camera IDs as keys and base64 encoded JPG images as values. If a camera is not available or fails to capture, its value will be None.

        Parameters
        ----------
        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        typing.Dict[str, typing.Optional[str]]
            Successfully captured frames from available cameras

        Examples
        --------
        from phospho import PhosphoApi

        client = PhosphoApi(
            base_url="https://yourhost.com/path/to/api",
        )
        client.camera.get_all_camera_frames()
        """
        _response = self._client_wrapper.httpx_client.request(
            "frames",
            method="GET",
            request_options=request_options,
        )
        try:
            if 200 <= _response.status_code < 300:
                return typing.cast(
                    typing.Dict[str, typing.Optional[str]],
                    parse_obj_as(
                        type_=typing.Dict[str, typing.Optional[str]],  # type: ignore
                        object_=_response.json(),
                    ),
                )
            if _response.status_code == 500:
                raise InternalServerError(
                    typing.cast(
                        typing.Optional[typing.Any],
                        parse_obj_as(
                            type_=typing.Optional[typing.Any],  # type: ignore
                            object_=_response.json(),
                        ),
                    )
                )
            _response_json = _response.json()
        except JSONDecodeError:
            raise ApiError(status_code=_response.status_code, body=_response.text)
        raise ApiError(status_code=_response.status_code, body=_response_json)

    def video_feed_for_camera(
        self,
        camera_id: typing.Optional[VideoFeedForCameraVideoCameraIdGetRequestCameraId],
        *,
        height: typing.Optional[int] = None,
        width: typing.Optional[int] = None,
        quality: typing.Optional[int] = None,
        request_options: typing.Optional[RequestOptions] = None,
    ) -> None:
        """
        Stream video feed of the specified camera. If no camera id is provided, the default camera is used. If the camera id is 'realsense' or 'depth', the realsense camera is used.Specify a target size and quality using query parameters.

        Parameters
        ----------
        camera_id : typing.Optional[VideoFeedForCameraVideoCameraIdGetRequestCameraId]

        height : typing.Optional[int]

        width : typing.Optional[int]

        quality : typing.Optional[int]

        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        None

        Examples
        --------
        from phospho import PhosphoApi

        client = PhosphoApi(
            base_url="https://yourhost.com/path/to/api",
        )
        client.camera.video_feed_for_camera()
        """
        _response = self._client_wrapper.httpx_client.request(
            f"video/{jsonable_encoder(camera_id)}",
            method="GET",
            params={
                "height": height,
                "width": width,
                "quality": quality,
            },
            request_options=request_options,
        )
        try:
            if 200 <= _response.status_code < 300:
                return
            if _response.status_code == 404:
                raise NotFoundError(
                    typing.cast(
                        typing.Optional[typing.Any],
                        parse_obj_as(
                            type_=typing.Optional[typing.Any],  # type: ignore
                            object_=_response.json(),
                        ),
                    )
                )
            if _response.status_code == 422:
                raise UnprocessableEntityError(
                    typing.cast(
                        HttpValidationError,
                        parse_obj_as(
                            type_=HttpValidationError,  # type: ignore
                            object_=_response.json(),
                        ),
                    )
                )
            _response_json = _response.json()
        except JSONDecodeError:
            raise ApiError(status_code=_response.status_code, body=_response.text)
        raise ApiError(status_code=_response.status_code, body=_response_json)


class AsyncCameraClient:
    def __init__(self, *, client_wrapper: AsyncClientWrapper):
        self._client_wrapper = client_wrapper

    async def get_all_camera_frames(
        self, *, request_options: typing.Optional[RequestOptions] = None
    ) -> typing.Dict[str, typing.Optional[str]]:
        """
        Capture frames from all available cameras. Returns a dictionary with camera IDs as keys and base64 encoded JPG images as values. If a camera is not available or fails to capture, its value will be None.

        Parameters
        ----------
        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        typing.Dict[str, typing.Optional[str]]
            Successfully captured frames from available cameras

        Examples
        --------
        import asyncio

        from phospho import AsyncPhosphoApi

        client = AsyncPhosphoApi(
            base_url="https://yourhost.com/path/to/api",
        )


        async def main() -> None:
            await client.camera.get_all_camera_frames()


        asyncio.run(main())
        """
        _response = await self._client_wrapper.httpx_client.request(
            "frames",
            method="GET",
            request_options=request_options,
        )
        try:
            if 200 <= _response.status_code < 300:
                return typing.cast(
                    typing.Dict[str, typing.Optional[str]],
                    parse_obj_as(
                        type_=typing.Dict[str, typing.Optional[str]],  # type: ignore
                        object_=_response.json(),
                    ),
                )
            if _response.status_code == 500:
                raise InternalServerError(
                    typing.cast(
                        typing.Optional[typing.Any],
                        parse_obj_as(
                            type_=typing.Optional[typing.Any],  # type: ignore
                            object_=_response.json(),
                        ),
                    )
                )
            _response_json = _response.json()
        except JSONDecodeError:
            raise ApiError(status_code=_response.status_code, body=_response.text)
        raise ApiError(status_code=_response.status_code, body=_response_json)

    async def video_feed_for_camera(
        self,
        camera_id: typing.Optional[VideoFeedForCameraVideoCameraIdGetRequestCameraId],
        *,
        height: typing.Optional[int] = None,
        width: typing.Optional[int] = None,
        quality: typing.Optional[int] = None,
        request_options: typing.Optional[RequestOptions] = None,
    ) -> None:
        """
        Stream video feed of the specified camera. If no camera id is provided, the default camera is used. If the camera id is 'realsense' or 'depth', the realsense camera is used.Specify a target size and quality using query parameters.

        Parameters
        ----------
        camera_id : typing.Optional[VideoFeedForCameraVideoCameraIdGetRequestCameraId]

        height : typing.Optional[int]

        width : typing.Optional[int]

        quality : typing.Optional[int]

        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        None

        Examples
        --------
        import asyncio

        from phospho import AsyncPhosphoApi

        client = AsyncPhosphoApi(
            base_url="https://yourhost.com/path/to/api",
        )


        async def main() -> None:
            await client.camera.video_feed_for_camera()


        asyncio.run(main())
        """
        _response = await self._client_wrapper.httpx_client.request(
            f"video/{jsonable_encoder(camera_id)}",
            method="GET",
            params={
                "height": height,
                "width": width,
                "quality": quality,
            },
            request_options=request_options,
        )
        try:
            if 200 <= _response.status_code < 300:
                return
            if _response.status_code == 404:
                raise NotFoundError(
                    typing.cast(
                        typing.Optional[typing.Any],
                        parse_obj_as(
                            type_=typing.Optional[typing.Any],  # type: ignore
                            object_=_response.json(),
                        ),
                    )
                )
            if _response.status_code == 422:
                raise UnprocessableEntityError(
                    typing.cast(
                        HttpValidationError,
                        parse_obj_as(
                            type_=HttpValidationError,  # type: ignore
                            object_=_response.json(),
                        ),
                    )
                )
            _response_json = _response.json()
        except JSONDecodeError:
            raise ApiError(status_code=_response.status_code, body=_response.text)
        raise ApiError(status_code=_response.status_code, body=_response_json)
