# This file was auto-generated by Fern from our API Definition.

import typing
import httpx
from .core.client_wrapper import SyncClientWrapper
from .control.client import ControlClient
from .camera.client import CameraClient
from .recording.client import RecordingClient
from .core.request_options import RequestOptions
from .types.server_status import ServerStatus
from .core.pydantic_utilities import parse_obj_as
from json.decoder import JSONDecodeError
from .core.api_error import ApiError
from .core.client_wrapper import AsyncClientWrapper
from .control.client import AsyncControlClient
from .camera.client import AsyncCameraClient
from .recording.client import AsyncRecordingClient


class PhosphobotApi:
    """
    Use this class to access the different functions within the SDK. You can instantiate any number of clients with different configuration that will propagate to these functions.

    Parameters
    ----------
    base_url : str
        The base url to use for requests from the client.

    timeout : typing.Optional[float]
        The timeout to be used, in seconds, for requests. By default the timeout is 60 seconds, unless a custom httpx client is used, in which case this default is not enforced.

    follow_redirects : typing.Optional[bool]
        Whether the default httpx client follows redirects or not, this is irrelevant if a custom httpx client is passed in.

    httpx_client : typing.Optional[httpx.Client]
        The httpx client to use for making requests, a preconfigured client is used by default, however this is useful should you want to pass in any custom httpx configuration.

    Examples
    --------
    from phosphobot import PhosphobotApi

    client = PhosphobotApi(
        base_url="https://yourhost.com/path/to/api",
    )
    """

    def __init__(
        self,
        *,
        base_url: str,
        timeout: typing.Optional[float] = None,
        follow_redirects: typing.Optional[bool] = True,
        httpx_client: typing.Optional[httpx.Client] = None,
    ):
        _defaulted_timeout = (
            timeout if timeout is not None else 60 if httpx_client is None else None
        )
        self._client_wrapper = SyncClientWrapper(
            base_url=base_url,
            httpx_client=httpx_client
            if httpx_client is not None
            else httpx.Client(
                timeout=_defaulted_timeout, follow_redirects=follow_redirects
            )
            if follow_redirects is not None
            else httpx.Client(timeout=_defaulted_timeout),
            timeout=_defaulted_timeout,
        )
        self.control = ControlClient(client_wrapper=self._client_wrapper)
        self.camera = CameraClient(client_wrapper=self._client_wrapper)
        self.recording = RecordingClient(client_wrapper=self._client_wrapper)

    def status_status_get(
        self, *, request_options: typing.Optional[RequestOptions] = None
    ) -> ServerStatus:
        """
        Get the status of the server.

        Parameters
        ----------
        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        ServerStatus
            Successful Response

        Examples
        --------
        from phosphobot import PhosphobotApi

        client = PhosphobotApi(
            base_url="https://yourhost.com/path/to/api",
        )
        client.status_status_get()
        """
        _response = self._client_wrapper.httpx_client.request(
            "status",
            method="GET",
            request_options=request_options,
        )
        try:
            if 200 <= _response.status_code < 300:
                return typing.cast(
                    ServerStatus,
                    parse_obj_as(
                        type_=ServerStatus,  # type: ignore
                        object_=_response.json(),
                    ),
                )
            _response_json = _response.json()
        except JSONDecodeError:
            raise ApiError(status_code=_response.status_code, body=_response.text)
        raise ApiError(status_code=_response.status_code, body=_response_json)


class AsyncPhosphobotApi:
    """
    Use this class to access the different functions within the SDK. You can instantiate any number of clients with different configuration that will propagate to these functions.

    Parameters
    ----------
    base_url : str
        The base url to use for requests from the client.

    timeout : typing.Optional[float]
        The timeout to be used, in seconds, for requests. By default the timeout is 60 seconds, unless a custom httpx client is used, in which case this default is not enforced.

    follow_redirects : typing.Optional[bool]
        Whether the default httpx client follows redirects or not, this is irrelevant if a custom httpx client is passed in.

    httpx_client : typing.Optional[httpx.AsyncClient]
        The httpx client to use for making requests, a preconfigured client is used by default, however this is useful should you want to pass in any custom httpx configuration.

    Examples
    --------
    from phosphobot import AsyncPhosphobotApi

    client = AsyncPhosphobotApi(
        base_url="https://yourhost.com/path/to/api",
    )
    """

    def __init__(
        self,
        *,
        base_url: str,
        timeout: typing.Optional[float] = None,
        follow_redirects: typing.Optional[bool] = True,
        httpx_client: typing.Optional[httpx.AsyncClient] = None,
    ):
        _defaulted_timeout = (
            timeout if timeout is not None else 60 if httpx_client is None else None
        )
        self._client_wrapper = AsyncClientWrapper(
            base_url=base_url,
            httpx_client=httpx_client
            if httpx_client is not None
            else httpx.AsyncClient(
                timeout=_defaulted_timeout, follow_redirects=follow_redirects
            )
            if follow_redirects is not None
            else httpx.AsyncClient(timeout=_defaulted_timeout),
            timeout=_defaulted_timeout,
        )
        self.control = AsyncControlClient(client_wrapper=self._client_wrapper)
        self.camera = AsyncCameraClient(client_wrapper=self._client_wrapper)
        self.recording = AsyncRecordingClient(client_wrapper=self._client_wrapper)

    async def status_status_get(
        self, *, request_options: typing.Optional[RequestOptions] = None
    ) -> ServerStatus:
        """
        Get the status of the server.

        Parameters
        ----------
        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        ServerStatus
            Successful Response

        Examples
        --------
        import asyncio

        from phosphobot import AsyncPhosphobotApi

        client = AsyncPhosphobotApi(
            base_url="https://yourhost.com/path/to/api",
        )


        async def main() -> None:
            await client.status_status_get()


        asyncio.run(main())
        """
        _response = await self._client_wrapper.httpx_client.request(
            "status",
            method="GET",
            request_options=request_options,
        )
        try:
            if 200 <= _response.status_code < 300:
                return typing.cast(
                    ServerStatus,
                    parse_obj_as(
                        type_=ServerStatus,  # type: ignore
                        object_=_response.json(),
                    ),
                )
            _response_json = _response.json()
        except JSONDecodeError:
            raise ApiError(status_code=_response.status_code, body=_response.text)
        raise ApiError(status_code=_response.status_code, body=_response_json)