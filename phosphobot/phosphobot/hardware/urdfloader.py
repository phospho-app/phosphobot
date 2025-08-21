from typing import Literal

import numpy as np
from loguru import logger

from phosphobot.hardware.base import BaseManipulator
from phosphobot.models.robot import BaseRobotConfig, BaseRobotPIDGains
from phosphobot.utils import get_resources_path
from phosphobot.models import RobotConfigStatus


class URDFLoader(BaseManipulator):
    name = "urdf_loader"

    RESOLUTION = 4096  # unused for now

    def __init__(
        self,
        urdf_path: str,
        end_effector_link_index: int,
        gripper_joint_index: int,
        axis_orientation: list[float] | None = None,
    ):
        self.URDF_FILE_PATH = urdf_path
        self.END_EFFECTOR_LINK_INDEX = end_effector_link_index
        self.GRIPPER_JOINT_INDEX = gripper_joint_index

        if axis_orientation is not None:
            self.AXIS_ORIENTATION = axis_orientation
        else:
            self.AXIS_ORIENTATION = [0, 0, 0, 1]

        super().__init__()

    async def connect(self):
        """
        Connect to the robot.
        """
        self.is_connected = True

    def disconnect(self):
        """
        Disconnect the robot.
        """
        self.is_connected = False

    def init_config(self):
        """
        This config is used for PID tuning, motors offsets, and other parameters.
        """
        self.config = BaseRobotConfig(
            name=self.name,
            servos_voltage=6.0,
            servos_offsets=[0] * len(self.SERVO_IDS),
            servos_calibration_position=[1] * len(self.SERVO_IDS),
            servos_offsets_signs=[1] * len(self.SERVO_IDS),
            pid_gains=[BaseRobotPIDGains(p_gain=0, i_gain=0, d_gain=0)]
            * len(self.SERVO_IDS),
            gripping_threshold=10,
            non_gripping_threshold=1,
        )

    def enable_torque(self):
        pass

    def disable_torque(self):
        pass

    def _set_pid_gains_motors(
        self, servo_id: int, p_gain: int = 32, i_gain: int = 0, d_gain: int = 32
    ):
        """
        Set the PID gains for the Feetech servo.

        :servo_id: Joint ID (0-6)
        :param p_gain: Proportional gain (0-255)
        :param i_gain: Integral gain (0-255)
        :param d_gain: Derivative gain (0-255)
        """
        pass

    def read_motor_position(self, servo_id: int, **kwargs) -> int | None:
        """
        Read the position of a Feetech servo.
        """
        pass

    def write_motor_position(self, servo_id: int, units: int, **kwargs) -> None:
        """
        Write a position to a Feetech servo.
        """
        pass

    def write_group_motor_position(
        self, q_target: np.ndarray, enable_gripper: bool = True
    ) -> None:
        """
        Write a position to all motors of the robot.
        """
        pass

    def read_group_motor_position(self) -> np.ndarray:
        """
        Read the position of all motors of the robot.
        """
        return np.zeros(len(self.SERVO_IDS), dtype=np.int32)

    def read_motor_torque(self, servo_id: int, **kwargs) -> float | None:
        """
        Read the torque of a Feetech servo.
        """
        pass

    def read_motor_voltage(self, servo_id: int, **kwargs) -> float | None:
        """
        Read the voltage of a Feetech servo.
        """
        pass

    def status(self) -> RobotConfigStatus:
        return RobotConfigStatus(
            name=self.name,
            device_name=self.URDF_FILE_PATH,
            temperature=None,
        )

    async def calibrate(self) -> tuple[Literal["success", "in_progress", "error"], str]:
        """
        Compute and save offsets and signs for the motors.

        This method has to be called multiple time, moving the robot to the same position as in the simulation beforehand.
        """

        return "success", "Calibration not implemented yet."

    def calibrate_motors(self, **kwargs) -> None:
        """
        Calibrate the motors.
        """
        pass
