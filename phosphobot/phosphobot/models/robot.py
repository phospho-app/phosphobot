import asyncio
import json
from abc import ABC, abstractmethod
from typing import List, Literal, Optional, Union

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field, model_validator
from serial.tools.list_ports_common import ListPortInfo

from phosphobot.utils import get_home_app_path

DEFAULT_FILE_ENCODING = "utf-8"


class Temperature(BaseModel):
    current: Optional[float]
    max: Optional[float]


class RobotConfigStatus(BaseModel):
    """
    Contains the configuration of a robot.
    """

    name: str
    robot_type: Literal["manipulator", "mobile", "other"] = "manipulator"
    device_name: Optional[str]
    temperature: Optional[List[Temperature]] = None


class BaseRobot(ABC):
    name: str
    is_connected: bool = False
    is_moving: bool = False

    @abstractmethod
    def set_motors_positions(
        self, q_target_rad: np.ndarray, enable_gripper: bool = False
    ) -> None:
        """
        Set the motor positions of the robot in radians.
        """
        raise NotImplementedError

    @abstractmethod
    def get_info_for_dataset(self):
        """
        Generate information about the robot useful for the dataset.
        Return a BaseRobotInfo object. (see models.dataset.BaseRobotInfo)
        Dict returned is info.json file at initialization
        """
        raise NotImplementedError

    @abstractmethod
    def get_observation(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the observation of the robot.
        This method should return the observation of the robot.
        Will be used to build an observation in a Step of an episode.
        Returns:
            - state: np.array state of the robot (7D)
            - joints_position: np.array joints position of the robot
        """
        raise NotImplementedError

    @abstractmethod
    async def connect(self) -> None:
        """
        Initialize communication with the robot.

        This method is called after the __init__ method.

        raise: Exception if the setup fails. For example, if the robot is not plugged in.
            This Exception will be caught by the __init__ method.
        """
        raise NotImplementedError

    @abstractmethod
    def disconnect(self) -> None:
        """
        Close the connection to the robot.

        This method is called on __del__ to disconnect the robot.
        """
        raise NotImplementedError

    def init_config(self) -> None:
        """
        Initialize the robot configuration.
        """
        pass

    def enable_torque(self) -> None:
        """
        Enable the torque of the robot.
        """
        pass

    def disable_torque(self) -> None:
        """
        Disable the torque of the robot.
        """
        pass

    @abstractmethod
    async def move_robot_absolute(
        self,
        target_position: np.ndarray,  # cartesian np.array
        target_orientation_rad: Optional[np.ndarray],  # rad np.array
        **kwargs,
    ) -> None:
        """
        Move the robot to the target position and orientation.
        This method should be implemented by the robot class.
        """
        raise NotImplementedError

    @classmethod
    def from_port(cls, port: ListPortInfo, **kwargs) -> Optional["BaseRobot"]:
        """
        Return the robot class from the port information.
        """
        logger.warning(
            f"For automatic detection of {cls.__name__}, the method from_port must be implemented. Skipping autodetection."
        )
        return None

    def status(self) -> RobotConfigStatus:
        return RobotConfigStatus(
            name=self.name,
            device_name=getattr(self, "SERIAL_ID", None),
        )

    @abstractmethod
    async def move_to_initial_position(self) -> None:
        """
        Move the robot to its initial position.
        The initial position is a safe position for the robot, where it is moved before starting the calibration.
        This method should be implemented by the robot class.

        This should update self.initial_position  and self.initial_orientation_rad
        """
        raise NotImplementedError

    @abstractmethod
    async def move_to_sleep(self) -> None:
        """
        Move the robot to its sleep position.
        The sleep position is a safe position for the robot, where it is moved before disabling the motors.
        This method should be implemented by the robot class.
        """
        raise NotImplementedError

    def move_to_sleep_sync(self):
        asyncio.run(self.move_to_sleep())


class BaseRobotPIDGains(BaseModel):
    """
    PID gains for servo motors
    """

    p_gain: float
    i_gain: float
    d_gain: float


class BaseRobotConfig(BaseModel):
    """
    Calibration configuration for a robot
    """

    name: str
    servos_voltage: float
    servos_offsets: List[float] = Field(
        default_factory=lambda: [
            2048.0,
            2048.0,
            2048.0,
            2048.0,
            2048.0,
            2048.0,
        ]
    )
    # Default factory: default offsets for SO-100
    servos_calibration_position: List[float]
    servos_offsets_signs: List[float] = Field(
        default_factory=lambda: [-1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    )
    pid_gains: List[BaseRobotPIDGains] = Field(default_factory=list)

    # Torque value to consider that an object is gripped
    gripping_threshold: int = Field(
        default=80,
        gt=0,
        description="Torque threshold to consider an object gripped. This will block the gripper position and prevent it from moving further.",
    )
    non_gripping_threshold: int = Field(
        default=10,
        gt=0,
        description="Torque threshold to consider an object not gripped. This will allow the gripper to move freely.",
    )

    @model_validator(mode="after")
    def validate_servos_arrays(self) -> "BaseRobotConfig":
        """Validate that servos_offsets and servos_calibration_position have same length
        and different values at each position"""
        if len(self.servos_offsets) != len(self.servos_calibration_position):
            raise ValueError(
                f"servos_offsets (length {len(self.servos_offsets)}) and "
                f"servos_calibration_position (length {len(self.servos_calibration_position)}) "
                f"must have the same length"
            )

        # Check that corresponding elements are different
        for i, (offset, cal_pos) in enumerate(
            zip(self.servos_offsets, self.servos_calibration_position)
        ):
            if offset == cal_pos:
                raise ValueError(
                    f"servos_offsets[{i}] ({offset}) must be different from "
                    f"servos_calibration_position[{i}] ({cal_pos})"
                )

        return self

    @classmethod
    def from_json(cls, filepath: str) -> Union["BaseRobotConfig", None]:
        """
        Load a configuration from a JSON file
        """
        try:
            with open(filepath, "r", encoding=DEFAULT_FILE_ENCODING) as f:
                data = json.load(f)

        except FileNotFoundError:
            return None

        # Fix issues with the JSON file
        servos_offsets = data.get("servos_offsets", [])
        if len(servos_offsets) == 0:
            data["servos_offsets"] = [2048.0] * 6

        servos_offsets_signs = data.get("servos_offsets_signs", [])
        if len(servos_offsets_signs) == 0:
            data["servos_offsets_signs"] = [-1.0] + [1.0] * 5

        try:
            return cls(**data)
        except Exception as e:
            logger.error(f"Error loading configuration from {filepath}: {e}")
            return None

    @classmethod
    def from_serial_id(
        cls, serial_id: str, name: str
    ) -> Union["BaseRobotConfig", None]:
        """
        Load a configuration from a serial ID and a name.
        """
        filename = f"{name}_{serial_id}_config.json"
        filepath = str(get_home_app_path() / "calibration" / filename)
        return cls.from_json(filepath)

    def to_json(self, filename: str) -> None:
        """
        Save the configuration to a JSON file
        """
        with open(filename, "w", encoding=DEFAULT_FILE_ENCODING) as f:
            f.write(self.model_dump_json(indent=4))

    def save_local(self, serial_id: str) -> str:
        """
        Save the configuration to the local calibration folder

        Returns:
            The path to the saved file
        """
        filename = f"{self.name}_{serial_id}_config.json"
        assert (
            "/" not in filename
        ), "Filename cannot contain '/'. Did you pass a device_name instead of SERIAL_ID?"
        filepath = str(get_home_app_path() / "calibration" / filename)
        logger.info(f"Saving configuration to {filepath}")
        self.to_json(filepath)
        return filepath


class RobotConfigResponse(BaseModel):
    """
    Response model for robot configuration.
    """

    robot_id: int
    name: str
    config: Optional[BaseRobotConfig]
    gripper_joint_index: Optional[int] = None
    servo_ids: List[int] = Field(default_factory=lambda: list(range(1, 7)))
    resolution: int = 4096
