# This file was auto-generated by Fern from our API Definition.

from ..core.pydantic_utilities import UniversalBaseModel
import typing
import pydantic
from ..core.pydantic_utilities import IS_PYDANTIC_V2


class AllCamerasStatus(UniversalBaseModel):
    """
    Description of the status of all cameras. Use this to know which camera to stream.
    """

    video_cameras_ids: typing.Optional[typing.List[int]] = pydantic.Field(default=None)
    """
    List of available video cameras ids
    """

    realsense_available: typing.Optional[bool] = pydantic.Field(default=None)
    """
    True if a RealSense camera is available
    """

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
            extra="allow", frozen=True
        )  # type: ignore # Pydantic v2
    else:

        class Config:
            frozen = True
            smart_union = True
            extra = pydantic.Extra.allow