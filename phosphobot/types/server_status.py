# This file was auto-generated by Fern from our API Definition.

from ..core.pydantic_utilities import UniversalBaseModel
from .status import Status
import typing
from .all_cameras_status import AllCamerasStatus
from ..core.pydantic_utilities import IS_PYDANTIC_V2
import pydantic


class ServerStatus(UniversalBaseModel):
    """
    Contains the status of the app
    """

    status: Status
    name: str
    robots: typing.Optional[typing.List[str]] = None
    cameras: typing.Optional[AllCamerasStatus] = None

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
            extra="allow", frozen=True
        )  # type: ignore # Pydantic v2
    else:

        class Config:
            frozen = True
            smart_union = True
            extra = pydantic.Extra.allow