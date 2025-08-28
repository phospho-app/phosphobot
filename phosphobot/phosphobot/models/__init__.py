from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from phosphobot._version import __version__
from phosphobot.types import VideoCodecs
from phosphobot.utils import NetworkDevice

from .camera import AllCamerasStatus, SingleCameraStatus
from .dataset import BaseDataset, BaseEpisode, JsonEpisode, Observation, Step
from .lerobot_dataset import (
    BaseRobotInfo,
    EpisodesModel,
    InfoModel,
    LeRobotDataset,
    LeRobotEpisode,
)
from .robot import (
    BaseRobot,
    BaseRobotConfig,
    BaseRobotPIDGains,
    RobotConfigStatus,
    Temperature,
    RobotConfigResponse,
)


class ServerStatus(BaseModel):
    """Contains the status of the app"""

    status: Literal["ok", "error"]
    name: str
    robots: List[str] = Field(default_factory=list, deprecated=True)
    robot_status: List[RobotConfigStatus] = Field(default_factory=list)
    cameras: AllCamerasStatus = Field(default_factory=AllCamerasStatus)
    version_id: str = Field(
        default=__version__, description="Current version of the teleoperation server"
    )
    is_recording: bool = Field(
        False, description="Whether the server is currently recording an episode."
    )
    ai_running_status: Literal["stopped", "running", "paused", "waiting"] = Field(
        "stopped",
        description="Whether the robot is currently controlled by an AI model.",
    )
    leader_follower_status: bool = Field(
        False,
        description="Whether the leader-follower control is currently active.",
    )
    server_ip: str = Field(
        ..., description="IP address of the phosphobot server", examples=["192.168.1.X"]
    )
    server_port: int = Field(
        ..., description="Port of the phosphobot server", examples=[80, 8020, 8021]
    )


class RobotStatus(BaseModel):
    """
    Contains the status of the robot and the number of actions received in one second
    This is sent by the robot to the app.
    """

    is_object_gripped: Optional[bool] = None
    is_object_gripped_source: Optional[Literal["left", "right"]] = None
    nb_actions_received: int


class EndEffectorReadRequest(BaseModel):
    sync: bool = Field(
        False,
        description="If True, the simulation will first read the motor positions, synchronize them with the simulated robot, and then return the end effector position."
        + "Useful for measurements, however it will take more time to respond.",
    )
    only_gripper: bool = Field(
        False,
        description="If True, only return the gripper state. If False, return the full end effector position and orientation.",
    )


class EndEffectorPosition(BaseModel):
    """
    End effector position for a movement in absolute frame.
    All zeros means the initial position, that you get by calling /move/init
    """

    x: Optional[float] = Field(description="X position in centimeters")
    y: Optional[float] = Field(description="Y position in centimeters")
    z: Optional[float] = Field(description="Z position in centimeters")
    rx: Optional[float] = Field(description="Absolute Pitch in degrees")
    ry: Optional[float] = Field(description="Absolute Yaw in degrees")
    rz: Optional[float] = Field(description="Absolute Roll in degrees")
    open: float = Field(description="0 for closed, 1 for open")


class MoveAbsoluteRequest(BaseModel):
    """
    Move the robot to an absolute position. All zeros means the initial position,
    that you get by calling /move/init.
    """

    x: Optional[float] = Field(None, description="X position in centimeters")
    y: Optional[float] = Field(None, description="Y position in centimeters")
    z: Optional[float] = Field(None, description="Z position in centimeters")
    rx: Optional[float] = Field(
        None,
        description="Absolute Pitch in degrees. If None, inverse kinematics will be used to calculate the best position.",
    )
    ry: Optional[float] = Field(
        None,
        description="Absolute Yaw in degrees. If None, inverse kinematics will be used to calculate the best position.",
    )
    rz: Optional[float] = Field(
        None,
        description="Absolute Roll in degrees. If None, inverse kinematics will be used to calculate the best position.",
    )
    open: Optional[float] = Field(None, description="0 for closed, 1 for open")

    max_trials: int = Field(
        10,
        ge=1,
        description="The maximum number of trials to reach the target position.",
    )
    position_tolerance: float = Field(
        0.03,
        ge=0,
        description="Increase max_trials and decrease tolerance to get more precision."
        + "Position tolerance is the euclidean distance between the target and the current position.",
    )
    orientation_tolerance: float = Field(
        0.2,
        ge=0,
        description="Increase max_trials and decrease tolerance to get more precision."
        + "Orientation tolerance is the euclidean distance between the target and the current orientation.",
    )


class AppControlData(BaseModel):
    """
    Type of data sent by the Metaquest app.
    """

    x: float
    y: float
    z: float
    rx: float = Field(description="Absolute Pitch in degrees")
    ry: float = Field(description="Absolute Yaw in degrees")
    rz: float = Field(description="Absolute Roll in degrees")
    open: float = Field(description="0 for closed, 1 for open")
    source: Literal["left", "right"] = Field(
        "right", description="Which hand the data comes from. Can be left or right."
    )
    timestamp: Optional[float] = Field(
        None, description="Unix timestamp with milliseconds"
    )
    # For moving robots, we can have a direction vector.
    direction_x: float = Field(
        0.0,
        description="Direction vector X, normalized between -1 (left) and 1 (right)",
        le=1,
        ge=-1,
    )
    direction_y: float = Field(
        0.0,
        description="Direction vector Y, normalized between -1 (backward) and 1 (forward)",
        le=1,
        ge=-1,
    )

    def is_null(self, eps: float = 1e-6) -> bool:
        """
        Return True if the data received is null (below a certain threshold)
        """
        return (
            self.x < eps
            and self.y < eps
            and self.z < eps
            and self.rx < eps
            and self.ry < eps
            and self.rz < eps
            and self.open == 0
        )

    def has_null_position(self) -> bool:
        return self.x == 0 and self.y == 0 and self.z == 0

    def has_null_orientation(self) -> bool:
        return self.rx == 0 and self.ry == 0 and self.rz == 0

    def to_robot(
        self, robot_name: str = "so-100"
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Convert the MetaQuest data to the robot referential.

        - y and z are inverted
        - ry and rz are inverted
        - we take the opposite of ry for rz as the base is inverted

        This mutates the object.

        This returns a tuple with the position, orientation and gripper state.
        """
        # TODO:This function should become a more generic config that depends on a
        # succession of instructions at the beginning.
        # Example : please move your arm sideways as much as you can, then up and down, etc.
        # These will we used to calibrate the robot and the MetaQuest.
        # The calibration has to be saved in a file and loaded at the beginning of the program.

        # Unity axis and robot axis are different
        position = np.array([self.x, self.z, self.y])

        # Pitch axis is inverted in RollPitchYaw: https://simple.wikipedia.org/wiki/Pitch,_yaw,_and_roll
        if robot_name == "wx-250s" or robot_name == "koch-v1.1":
            orientation = np.array([-self.rx, -self.rz, -self.ry])
        if robot_name == "agilex-piper":
            orientation = np.array([self.ry, self.rx, self.rz])
        else:  # SO-100 configuration
            orientation = np.array([-self.rx, -self.rz, -self.ry])

        orientation = np.mod(orientation + 180, 360) - 180

        return position, orientation, self.open


class RelativeEndEffectorPosition(BaseModel):
    """
    Relative end effector position for a movement in relative frame.
    Useful for OpenVLA-like control.
    """

    ### How we store the robot state for dataset generation ###
    # Dataset are in RDLS format like the Bridge Data V2 dataset
    # See https://github.com/google-research/rlds for more information

    x: Optional[float] = Field(None, description="Delta X position in centimeters")
    y: Optional[float] = Field(None, description="Delta Y position in centimeters")
    z: Optional[float] = Field(None, description="Delta Z position in centimeters")
    rx: Optional[float] = Field(None, description="Relative Pitch in degrees")
    ry: Optional[float] = Field(None, description="Relative Yaw in degrees")
    rz: Optional[float] = Field(None, description="Relative Roll in degrees")
    open: Optional[float] = Field(
        None, description="0 for closed, 1 for open. If None, use the last value."
    )

    def init(self, np_array: np.ndarray) -> None:
        if np_array.shape != (7,):
            raise ValueError("Invalid array shape")

        self.x = np_array[0]
        self.y = np_array[1]
        self.z = np_array[2]
        self.rx = np_array[3]
        self.ry = np_array[4]
        self.rz = np_array[5]
        self.open = np_array[6]


class EmoteRequest(BaseModel):
    """
    Emote supported by the application.
    """

    emote_name: Literal["wave", "dance", "bow"] = Field(
        ...,
        description="Name of the emote to play. See the list of emotes in the documentation.",
        examples=["wave", "dance", "bow"],
    )


class AutoControlRequest(BaseModel):
    """
    Launch an auto control with a request to the OpenVLA or ACT server.
    """

    type_of_model: Literal["act", "openvla", "pi0"] = Field(
        ..., description="Type of model, either OpenVLA or ACT"
    )
    size_of_images: Optional[tuple[int, int]] = Field(
        None,
        description="Size of the images to send to the model (the size it was trained with)",
    )
    instruction: Optional[str] = Field(
        None, description="Prompt to be followed by the robot when using OpenVLA"
    )
    robot_ids: List[int] = Field(
        [0], description="List of robot ids to control, in order, defaults to [0]"
    )


class CalibrateResponse(BaseModel):
    """
    Response from the calibration endpoint.
    """

    calibration_status: Literal["error", "success", "in_progress"] = Field(
        ...,
        description="Status of the calibration. Ends when status is success or error.",
    )
    message: str
    current_step: int
    total_nb_steps: int


class JointsReadRequest(BaseModel):
    """
    Request to read the joints of the robot.
    """

    unit: Literal["rad", "motor_units", "degrees"] = Field(
        "rad",
        description="The unit of the angles. Defaults to radian.",
    )
    joints_ids: Optional[List[int]] = Field(
        None,
        description="If set, only read the joints with these ids. If None, read all joints.",
    )
    source: Literal["sim", "robot"] = Field(
        "robot",
        description="Source of the joint angles. 'sim' means the angles are read from the simulation, 'robot' means the angles are read from the hardware.",
    )


class JointsWriteRequest(BaseModel):
    """
    Request to set the joints of the robot.
    """

    angles: List[float] = Field(
        ...,
        description="A list with the position of each joint. The length of the list must be equal to the number of joints. The unit is given by the 'unit' field.",
    )
    unit: Literal["rad", "motor_units", "degrees"] = Field(
        "rad",
        description="The unit of the angles. Defaults to radian.",
    )
    joints_ids: Optional[List[int]] = Field(
        None,
        description="If set, only set the joints with these ids. If None, set all joints."
        "Example: 'angles'=[1,1,1], 'joints_ids'=[0,1,2] will set the first 3 joints to 1 radian.",
    )


class JointsReadResponse(BaseModel):
    """
    Response to read the joints of the robot.
    """

    angles: List[Optional[float]] = Field(
        ...,
        description="A list of length 6, with the position of each joint in the unit specified in the request. If a joint is not available, its value will be None.",
    )
    unit: Literal["rad", "motor_units", "degrees"] = Field(
        "rad",
        description="The unit of the angles. Defaults to radian.",
    )


class TorqueReadResponse(BaseModel):
    """
    Response to read the torque of the robot.
    """

    current_torque: List[float] = Field(
        ...,
        description="A list of length 6, with the current torque of each joint.",
    )


class VoltageReadResponse(BaseModel):
    """
    Response to read the torque of the robot.
    """

    current_voltage: Optional[List[float]] = Field(
        ...,
        description="A list of length 6, with the current voltage of each joint. If the robot is not connected, this will be None.",
    )


class TemperatureReadResponse(BaseModel):
    """
    Response to read the Temperature of the robot.
    """

    current_max_Temperature: Optional[List[Temperature]] = Field(
        ...,
        description=" A list of Temperature objects, one for each joint. If the robot is not connected, this will be None.",
    )


class TemperatureWriteRequest(BaseModel):
    """
    Request to set the maximum Temperature for joints of the robot.
    """

    maximum_temperature: List[int] = Field(
        ...,
        description="A list with the maximum temperature of each joint. The length of the list must be equal to the number of joints.",
    )


class InfoResponse(BaseModel):
    """
    Response to the /dataset/info endpoint.
    """

    status: Literal["ok", "error"] = "ok"
    robot_type: Optional[str] = None
    robot_dof: Optional[int] = None
    number_of_episodes: Optional[int] = None
    image_keys: Optional[List[str]] = None
    image_frames: Optional[Dict[str, str]] = None


class StatusResponse(BaseModel):
    """
    Default response. May contain other fields.
    """

    model_config = ConfigDict(extra="allow")

    status: Literal["ok", "error"] = "ok"
    message: Optional[str] = None


class TrainingInfoRequest(BaseModel):
    model_id: Optional[str] = Field(
        None, description="Hugging Face model id to get training info"
    )
    model_type: Literal["gr00t", "ACT", "ACT_BBOX", "custom"]


class TrainingInfoResponse(BaseModel):
    status: Literal["ok", "error"]
    message: Optional[str] = None
    training_body: Optional[dict] = None


class ServerInfoResponse(BaseModel):
    server_id: int
    url: str
    port: int
    tcp_socket: tuple[str, int]
    model_id: str
    timeout: int


class HFDownloadDatasetRequest(BaseModel):
    dataset_name: str


class DatasetRepairRequest(BaseModel):
    dataset_path: str = Field(
        ...,
        description="Path to the dataset to repair",
        examples=["/lerobot_v2.1/example_dataset"],
    )


class DatasetSplitRequest(BaseModel):
    dataset_path: str = Field(
        ...,
        description="Path to the dataset to split",
        examples=["/lerobot_v2.1/example_dataset"],
    )
    split_ratio: float = Field(
        0.8,
        ge=0,
        le=1,
        description="Ratio of the dataset to use for the first split. The second split will use the rest of the dataset.",
    )
    first_split_name: str = Field(
        ...,
        description="Name of the first split.",
        examples=["/lerobot_v2.1/example_dataset_training"],
    )
    second_split_name: str = Field(
        ...,
        description="Name of the second split.",
        examples=["/lerobot_v2.1/example_dataset_validation"],
    )


class DatasetShuffleRequest(BaseModel):
    dataset_path: str = Field(
        ...,
        description="Path to the dataset to shuffle",
        examples=["/lerobot_v2.1/example_dataset"],
    )


class SpawnStatusResponse(StatusResponse):
    """
    Response to spawn a server.
    """

    server_info: ServerInfoResponse


class AIControlStatusResponse(StatusResponse):
    """
    Response when starting the AI control.
    """

    server_info: Optional[ServerInfoResponse] = None
    ai_control_signal_id: str
    ai_control_signal_status: Literal["stopped", "running", "paused", "waiting"]


class RecordingStartRequest(BaseModel):
    """
    Request to start the recording of an episode.
    """

    dataset_name: Optional[str] = Field(
        None,
        description="Name of the dataset to save the episode in."
        + "If None, defaults to the value set in Admin Configuration.",
        examples=["example_dataset"],
    )
    episode_format: Optional[Literal["json", "lerobot_v2", "lerobot_v2.1"]] = Field(
        None,
        description="Format to save the episode.\n`json` is compatible with OpenVLA and stores videos as a series of npy.\n`lerobot_v2` is compatible with [lerobot training.](https://docs.phospho.ai/learn/ai-models)."
        + "If None, defaults to the value set in Admin Configuration.",
        examples=["lerobot_v2.1"],
    )
    video_codec: Optional[VideoCodecs] = Field(
        None,
        description="Codec to use for the video saving."
        + "If None, defaults to the value set in Admin Configuration.",
        examples=["avc1"],
    )
    freq: Optional[int] = Field(
        None,
        description="Records steps of the robot at this frequency."
        + "If None, defaults to the value set in Admin Configuration.",
        examples=[30],
    )
    branch_path: Optional[str] = Field(
        None,
        description="Path to the branch to push the dataset to, in addition to the main branch. If set to None, only push to the main branch. Defaults to None.",
    )
    target_video_size: Optional[Tuple[int, int]] = Field(
        None,
        description="Target video size for the recording, all videos in the dataset should have the same size. If set to None, defaults to the value set in Admin Configuration.",
        examples=[(320, 240)],
    )
    cameras_ids_to_record: Optional[List[int]] = Field(
        None,
        description="List of camera ids to record. If set to None, records all available cameras.",
        examples=[[0, 1]],
    )
    instruction: Optional[str] = Field(
        None,
        description="A text describing the recorded task. If set to None, defaults to the value set in Admin Configuration.",
        examples=["Pick up the orange brick and put it in the black box."],
    )
    robot_serials_to_ignore: Optional[List[str]] = Field(
        None,
        description="List of robot serial ids to ignore. If set to None, records all available robots.",
        examples=[["/dev/ttyUSB0"]],
    )
    enable_rerun_visualization: bool = Field(
        False,
        description="Enable rerun",
    )


class RecordingStopRequest(BaseModel):
    """
    Request to stop the recording of the episode.
    """

    save: bool = Field(
        True,
        description="Whether to save the episode to disk. Defaults to True.",
    )


class RecordingStopResponse(BaseModel):
    """
    Response when the recording is stopped. The episode is saved in the given path.
    """

    episode_folder_path: Optional[str] = Field(
        ...,
        description="Path to the folder where the episode is saved.",
    )
    episode_index: Optional[int] = Field(
        ...,
        description="Index of the recorded episode in the dataset.",
    )


class RecordingPlayRequest(BaseModel):
    """
    Request to play a recorded episode.
    """

    dataset_format: Literal["lerobot_v2", "lerobot_v2.1"] = Field(
        "lerobot_v2.1",
        description="Format of the dataset to play. This is used to determine how to read the episode data.",
    )
    dataset_name: Optional[str] = Field(
        None,
        description="Name of the dataset to play the episode from. If None, defaults to the last dataset recorded.",
        examples=["example_dataset"],
    )
    episode_id: Optional[int] = Field(
        None,
        description="ID of the episode to play. If a dataset_name is specified but episode_id is None, plays the last episode recorded of this dataset. "
        + "If dataset_name is None, this is ignored and plays the last episode recorded.",
        examples=[0],
    )
    episode_path: Optional[str] = Field(
        None,
        description="(Optional) If you recorded your data with LeRobot v2 compatible format, you can directly specifiy the path to the .parquet file of the episode to play. If specified, you don't have to pass a dataset_name or episode_id.",
        examples=[
            "~/phosphobot/lerobot_v2/example_dataset/chunk-000/episode_000000.json"
        ],
    )

    robot_id: Optional[Union[int, List[int]]] = Field(
        None,
        description="ID of the robot to play the episode on. If None, plays on all robots. If a list, plays on the robots with the given IDs.",
        examples=[0, [0, 1]],
    )
    robot_serials_to_ignore: Optional[List[str]] = Field(
        None,
        description="List of robot serial ids to ignore. If set to None, plays on all available robots.",
        examples=[["/dev/ttyUSB0"]],
    )
    replicate: bool = Field(
        True,
        description="If False and there are more robots than number of robots in the episode, extra robots will not move. If True, all the extras robots will replicate movements of the robots in the episode."
        + "Examples: If there are 4 robots and the episode has 2 robots, if replicate is True, robot 3 and 4 will replicate the movements of robot 1 and 2. If replicate is False, robot 3 and 4 will not move.",
    )

    playback_speed: float = Field(
        1.0,
        ge=0,
        description="Speed of the playback. 1.0 is normal speed, 0.5 is half speed, 2.0 is double speed. High speed may cause the robot to break.",
    )
    interpolation_factor: int = Field(
        4,
        ge=1,
        description="Smoothen the playback by interpolating between frames. 1 means no interpolation, 2 means 1 frame every 2 frames, etc. 4 is the recommended value.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "dataset_name": "example_dataset",
                    "episode_id": 0,
                },
                {
                    "episode_path": "~/phosphobot/lerobot_v2/example_dataset/chunk-000/episode_000000.json",
                    "robot_id": [0, 1],
                    "replicate": False,
                },
            ]
        }
    }


class HuggingFaceTokenRequest(BaseModel):
    """
    Hugging Face token saved by the user.
    """

    token: str


class WandBTokenRequest(BaseModel):
    """
    WandB token saved by the user.
    """

    token: str


class ItemInfo(BaseModel):
    """
    Contains the information of the items in a directory used in the browser page
    """

    name: str
    path: str
    absolute_path: str
    is_dir: bool
    is_dataset_dir: bool = False
    browseUrl: Optional[str] = None
    downloadUrl: Optional[str] = None
    previewUrl: Optional[str] = None
    huggingfaceUrl: Optional[str] = None
    canDeleteDataset: bool = False
    deleteDatasetAction: Optional[str] = None


class BrowseFilesResponse(BaseModel):
    """
    Represents a response for browsing directories or items in a robotic system.
    """

    directoryTitle: str
    tokenError: Optional[str] = None
    items: List[ItemInfo]
    episode_ids: List[int] = []
    episode_paths: List[str] = []


class BrowserFilesRequest(BaseModel):
    """
    Request to browse files in a directory.
    """

    path: str


class DeleteEpisodeRequest(BaseModel):
    """
    Request to delete an episode.
    """

    path: str
    episode_id: int


class ModelConfigurationRequest(BaseModel):
    model_id: str = Field(
        ...,
        description="Hugging Face model id to use",
        examples=["PLB/GR00T-N1-lego-pickup-mono-2"],
        # no empty string
        pattern=r"^\s*\S.*$",
    )
    model_type: Literal["gr00t", "ACT", "ACT_BBOX"] = Field(
        ...,
        description="Type of model to use.",
    )


class ModelConfigurationResponse(BaseModel):
    video_keys: List[str] = Field(
        ...,
        description="List of video keys for the model. These are the keys used to access the videos in the dataset.",
        examples=[["video_0", "video_1"]],
    )
    checkpoints: List[str] = Field(
        default_factory=list,
        description="List of available checkpoints for the model.",
        examples=[["100", "500"]],
    )


class AdminSettingsRequest(BaseModel):
    """
    Contains the admin settings
    """

    dataset_name: str
    episode_format: str
    freq: int
    video_codec: VideoCodecs
    video_size: List[int]  # size 2
    task_instruction: str
    cameras_to_record: Optional[List[int]] = None
    hf_private_mode: bool = False


class AdminSettingsResponse(BaseModel):
    """
    Contains the settings returned in the admin page
    """

    dataset_name: str
    freq: int
    episode_format: str
    video_codec: VideoCodecs
    video_size: List[int]  # size 2
    task_instruction: str
    cameras_to_record: Optional[List[int]]
    hf_private_mode: bool


class AdminSettingsTokenResponse(BaseModel):
    """
    To each provider is assigned a bool, which is True
    if the token is set and valid.
    """

    huggingface: bool = False
    wandb: bool = False


class VizSettingsResponse(BaseModel):
    """
    Settings for the vizualisation page.
    """

    width: int
    height: int
    quality: int


class MergeDatasetsRequest(BaseModel):
    # Contains a list of dataset paths to merge
    first_dataset: str = Field(
        ...,
        description="Path to the first dataset to merge",
        examples=["/lerobot_v2.1/example_dataset"],
    )
    second_dataset: str = Field(
        ...,
        description="Path to the second dataset to merge",
        examples=["/lerobot_v2.1/example_dataset_to_merge_with"],
    )
    new_dataset_name: str = Field(
        ...,
        description="Name of the new dataset to create",
        examples=["/lerobot_v2.1/example_dataset_merged"],
    )
    image_key_mappings: Dict[str, str] = Field(
        ...,
        description="Mapping of the image keys from the first dataset to the second dataset.",
        examples=[
            {"wrist_camera": "wrist_camera_2", "context_camera": "context_camera_2"}
        ],
    )


class DatasetListResponse(BaseModel):
    """
    List of datasets
    """

    pushed_datasets: List[str]
    local_datasets: List[str]


class StartServerRequest(BaseModel):
    """
    Request to start an inference server and get the server info.
    """

    model_id: str = Field(..., description="Hugging Face model id to use")
    robot_serials_to_ignore: Optional[List[str]] = Field(
        None,
        description="List of robot serial ids to ignore. If set to None, controls all available robots.",
        examples=[["/dev/ttyUSB0"]],
    )
    model_type: Literal["gr00t", "ACT"] = Field(
        ...,
        description="Type of model to use. Can be gr00t or act.",
    )


class StartAIControlRequest(BaseModel):
    """
    Request to start the AI control of the robot.
    """

    prompt: Optional[str] = Field(
        None, description="Prompt to be followed by the robot"
    )
    model_id: str = Field(..., description="Hugging Face model id to use")
    speed: float = Field(
        1.0,
        ge=0.1,
        le=2,
        description="Speed of the AI control. 1.0 is normal speed, 0.5 is half speed, 2.0 is double speed. The highest speed is still bottlenecked by the GPU inference time.",
    )

    robot_serials_to_ignore: Optional[List[str]] = Field(
        None,
        description="List of robot serial ids to ignore. If set to None, controls all available robots.",
        examples=[["/dev/ttyUSB0"]],
    )
    cameras_keys_mapping: Optional[Dict[str, int]] = Field(
        None,
        description="Mapping of the camera keys to the camera ids. If set to None, use the default mapping based on cameras order.",
        examples=[{"wrist_camera": 0, "context_camera": 1}],
    )
    model_type: Literal["gr00t", "ACT", "ACT_BBOX"] = Field(
        ...,
        description="Type of model to use. Can be gr00t or act.",
    )
    selected_camera_id: Optional[int] = Field(
        None,
        description="Name of the camera to use when ACT_BBOX model is used. This is only required for ACT_BBOX models, and is ignored for other models.",
    )
    verify_cameras: bool = Field(
        True,
        description="Whether to verify the setup before starting the AI control. If False, skips the verification step.",
    )
    checkpoint: Optional[int] = Field(
        None,
        description="Checkpoint to use for the model. If None, uses the latest checkpoint.",
        examples=[500],
    )
    angle_format: Literal["degrees", "rad", "other"] = Field(
        "rad",
        description="Format of the angles used in the model. Can be 'degrees', 'radians', or 'other'. If other is selected, you will need to specify a min and max angle value.",
        examples=["rad"],
    )
    min_angle: Optional[float] = Field(
        None,
        description="If angle_format is 'other', this is the minimum angle value used in the model. If None and angle_format is 'other', will raise an error.",
    )
    max_angle: Optional[float] = Field(
        None,
        description="If angle_format is 'other', this is the maximum angle value used in the model. If None and angle_format is 'other', will raise an error.",
    )

    @model_validator(mode="after")
    def check_angle_format(self) -> "StartAIControlRequest":
        """
        Validate the angle format and min/max angles if angle_format is 'other'.
        """
        if self.angle_format == "other":
            if self.min_angle is None or self.max_angle is None:
                raise ValueError(
                    "If angle_format is 'other', min_angle and max_angle must be set."
                )
            if self.min_angle >= self.max_angle:
                raise ValueError(
                    "min_angle must be less than max_angle when angle_format is 'other'."
                )
        return self


class AIStatusResponse(BaseModel):
    """
    Response to the AI status request.
    """

    status: Literal["stopped", "running", "paused", "waiting"] = Field(
        ..., description="Status of the AI control"
    )
    id: Optional[str] = Field(..., description="ID of the AI control session.")


class TorqueControlRequest(BaseModel):
    """
    Request to control the robot's torque.
    """

    torque_status: bool = Field(
        ..., description="Whether to enable or disable torque control."
    )


class ModelStatusResponse(BaseModel):
    model_url: str
    model_status: Literal["Done", "In progress", "Not started", "Failed"]


class ModelStatusRequest(BaseModel):
    model_url: str = Field(..., description="Hugging Face model URL")


class NetworkCredentials(BaseModel):
    ssid: str
    password: str


class ResetPasswordRequest(BaseModel):
    access_token: str = Field(..., description="Access token from the reset email")
    refresh_token: str = Field(..., description="Refresh token from the reset email")
    new_password: str = Field(..., description="New password to set for the user")


class LoginCredentialsRequest(BaseModel):
    email: str
    password: str


class VerifyEmailCodeRequest(BaseModel):
    email: str
    token: str


class ConfirmRequest(BaseModel):
    access_token: str
    refresh_token: str


class ForgotPasswordRequest(BaseModel):
    email: str


class Session(BaseModel):
    """
    Session model for storing supabase session details.
    """

    user_id: str
    user_email: str
    email_confirmed: bool
    access_token: str
    refresh_token: str
    expires_at: int


class SessionReponse(BaseModel):
    """
    Response for login/signup
    """

    message: str
    session: Optional[Session] = None
    is_pro_user: Optional[bool] = None


class AuthResponse(BaseModel):
    authenticated: bool
    session: Optional[Session] = None
    is_pro_user: Optional[bool] = None


class FeedbackRequest(BaseModel):
    feedback: Literal["positive", "negative"] = Field(
        ...,
        description="Feedback on the AI control. Can be positive or negative.",
    )
    ai_control_id: str = Field(
        ...,
        description="ID of the AI control session.",
    )


class RobotPairRequest(BaseModel):
    """
    Represents a pair of robots for leader-follower control.
    """

    model_config = ConfigDict(extra="ignore")

    leader_id: Optional[int] = Field(
        ..., description="Serial number of the leader robot"
    )
    follower_id: Optional[int] = Field(
        ..., description="Serial number of the follower robot"
    )


class StartLeaderArmControlRequest(BaseModel):
    """
    Request to set up leader-follower control. The leader robot will be controlled by the user,
    and the follower robot will mirror the leader's movements.

    You need two robots connected to the same computer to use this feature.
    """

    robot_pairs: List[RobotPairRequest] = Field(
        ...,
        description="List of robot pairs to control. Each pair contains the robot id of the leader and the corresponding follower.",
    )
    invert_controls: bool = Field(
        False, description="Mirror controls for the follower robots"
    )
    enable_gravity_compensation: bool = Field(
        False, description="Enable gravity compensation for the leader robots"
    )
    gravity_compensation_values: Optional[Dict[str, int]] = Field(
        {"shoulder": 100, "elbow": 50, "wrist": 10},
        description="Gravity compensation pourcentage values for shoulder, elbow, and wrist joints (0-100%)",
    )


class SupabaseTrainingModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: int
    status: Literal["succeeded", "failed", "running", "canceled"]
    user_id: str
    dataset_name: str
    model_name: str
    requested_at: str
    terminated_at: Optional[str]
    used_wandb: Optional[bool]
    model_type: str
    training_params: Optional[dict] = None
    modal_function_call_id: Optional[str] = None
    # Metrics
    session_count: int = 0
    success_rate: Optional[float] = None


class TrainingsList(BaseModel):
    models: list[SupabaseTrainingModel]


class UDPServerInformationResponse(BaseModel):
    host: str
    port: int


class StartTrainingResponse(StatusResponse):
    training_id: Optional[int] = Field(
        ...,
        description="ID of the training to start. This is the ID returned by the training request.",
    )
    model_url: Optional[str] = Field(
        None,
        description="URL to the Hugging Face model card.",
    )


class CancelTrainingRequest(BaseModel):
    training_id: int = Field(..., description="ID of the training to cancel.")


class CustomTrainingRequest(BaseModel):
    custom_command: str = Field(
        ...,
        description="Will run this custom command as a subprocess when pressing the train button.",
    )


class ScanNetworkRequest(BaseModel):
    """
    Request to scan the network for devices.
    """

    robot_name: Optional[str] = Field(
        None,
        description="Name of the robot to scan for. If None, scans for all devices on the network.",
    )


class ScanNetworkResponse(BaseModel):
    """
    Response to the network scan request.
    """

    devices: List[NetworkDevice] = Field(
        ...,
        description="List of devices found on the network.",
    )
    subnet: Optional[str] = Field(
        ...,
        description="Subnet of the network.",
        examples=["192.168.1.1/24"],
    )


class LocalDevice(BaseModel):
    name: str
    device: str
    serial_number: Optional[str] = None
    pid: Optional[int] = None
    interface: Optional[str] = None


class ScanDevicesResponse(BaseModel):
    """
    Response to the USB devices scan request.
    """

    devices: List[LocalDevice] = Field(
        ...,
        description="List of connected USB devices.",
    )


class RobotConnectionRequest(BaseModel):
    """
    Request to manually connect to a robot.
    """

    robot_name: str = Field(
        ...,
        description="Type of the robot to connect to.",
        examples=["so-100", "wx-250s", "koch-v1.1"],
    )
    connection_details: dict[str, Any] = Field(
        ...,
        description="Connection details for the robot. These are passed to the class constructor. This can include IP address, port, and other connection parameters.",
    )


class RobotConnectionResponse(StatusResponse):
    robot_id: int


class AddZMQCameraRequest(BaseModel):
    """
    Request model for adding a ZMQ camera feed.
    """

    tcp_address: str = Field(
        ...,
        description="TCP address of the ZMQ publisher. "
        + "Format: 'tcp://<host>:<port>'.",
        examples=["tcp://localhost:5555"],
    )
    topic: Optional[str] = Field(
        None,
        description="Topic to subscribe to. If None, will subscribes to all messages on the given TCP address.",
        examples=["camera", "wrist_camera"],
    )


class TeleopSettings(BaseModel):
    """
    Model representing current teleop settings.
    """

    vr_scaling: float = Field(
        ...,
        description="VR scaling factor for teleoperation control.",
        gt=0,
        examples=[1.0, 0.5, 2.0],
    )


class TeleopSettingsRequest(BaseModel):
    """
    Request model for updating teleop settings.
    """

    vr_scaling: float = Field(
        ...,
        description="VR scaling factor for teleoperation control.",
        gt=0,
        examples=[1.0, 0.5, 2.0],
    )
