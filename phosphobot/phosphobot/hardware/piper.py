import asyncio
import subprocess
import time
from typing import Any, List, Literal, Optional, Union

import numpy as np
from loguru import logger
from piper_sdk import C_PiperInterface_V2

from phosphobot.hardware.base import BaseManipulator
from phosphobot.models import BaseRobotConfig
from phosphobot.models.robot import RobotConfigStatus
from phosphobot.utils import get_resources_path, is_running_on_linux


class PiperHardware(BaseManipulator):
    name = "agilex-piper"
    device_name = "agilex-piper"

    URDF_FILE_PATH = str(
        get_resources_path() / "urdf" / "piper" / "urdf" / "piper.urdf"
    )

    AXIS_ORIENTATION = [0, 0, 0, 1]

    END_EFFECTOR_LINK_INDEX = 5
    GRIPPER_JOINT_INDEX = 6

    SERVO_IDS = [1, 2, 3, 4, 5, 6, 7]

    RESOLUTION = 360 * 1000  # In 0.001 degree

    SLEEP_POSITION = [0, 0, 0, 0, 0, 0]
    time_to_sleep: float = 1.8
    CALIBRATION_POSITION = [0, 0, 0, 0, 0, 0]

    is_object_gripped = False
    is_moving = False
    robot_connected = False

    GRIPPER_MAX_ANGLE = 99  # In degree
    ENABLE_GRIPPER = 0x01
    DISABLE_GRIPPER = 0x00

    GRIPPER_SERVO_ID = 7
    # When using the set zero of gripper control, we observe that current position is set to -1800 and not to zero
    GRIPPER_ZERO_POSITION = -1800
    # Strength with which the gripper will close. Similar to the gripping threshold value of other robots,
    GRIPPER_EFFORT = 600

    calibration_max_steps: int = 2

    # Reference: https://github.com/agilexrobotics/piper_sdk/blob/master/asserts/V2/INTERFACE_V2.MD#jointctrl
    #  |joint_name|     limit(rad)     |    limit(angle)    |
    # |----------|     ----------     |     ----------     |
    # |joint1    |   [-2.6179, 2.6179]  |    [-150.0, 150.0] |
    # |joint2    |   [0, 3.14]        |    [0, 180.0]      |
    # |joint3    |   [-2.967, 0]      |    [-170, 0]       |
    # |joint4    |   [-1.745, 1.745]  |    [-100.0, 100.0] |
    # |joint5    |   [-1.22, 1.22]    |    [-70.0, 70.0]   |
    # |joint6    |   [-2.09439, 2.09439]|    [-120.0, 120.0] |
    piper_limits_rad: dict = {
        1: {"min_angle_limit": -2.6179, "max_angle_limit": 2.6179},
        2: {"min_angle_limit": 0, "max_angle_limit": 3.14},
        3: {"min_angle_limit": -2.967, "max_angle_limit": 0},
        4: {"min_angle_limit": -1.745, "max_angle_limit": 1.745},
        5: {"min_angle_limit": -1.047, "max_angle_limit": 1.047},
        6: {"min_angle_limit": -2.09439, "max_angle_limit": 2.0943},
    }
    piper_limits_degrees: dict = {
        1: {"min_angle_limit": -150.0, "max_angle_limit": 150.0},
        2: {"min_angle_limit": 0, "max_angle_limit": 180.0},
        3: {"min_angle_limit": -170, "max_angle_limit": 0},
        4: {"min_angle_limit": -100.0, "max_angle_limit": 100.0},
        5: {"min_angle_limit": -60.0, "max_angle_limit": 60.0},
        6: {"min_angle_limit": -120.0, "max_angle_limit": 120.0},
    }

    def __init__(
        self,
        can_name: str = "can0",
        only_simulation: bool = False,
        axis: Optional[List[float]] = None,
    ) -> None:
        self.can_name = can_name
        super().__init__(
            only_simulation=only_simulation, axis=axis, enable_self_collision=True
        )
        self.SERIAL_ID = can_name
        self.is_torqued = False
        self._init_sim_gripper()

    def _init_sim_gripper(self) -> None:
        """Discover and cache gripper joints + limits in simulation."""

        self.gripper_initial_angle = None

        # Initialize gripper-related attributes with None/empty defaults
        self._gripper_joint_indices = []
        self._gripper_closed_positions = []
        self._gripper_open_positions = []
        self._gripper_joint_limits = []  # NEW: Cache joint limits

        try:
            name2idx = {
                self.sim.get_joint_info(robot_id=self.p_robot_id, joint_index=i)[
                    1
                ].decode("utf-8"): i
                for i in range(self.num_joints)
            }

            # Try to find joint7 and joint8 first
            if "joint7" in name2idx and "joint8" in name2idx:
                self._gripper_joint_indices = [name2idx["joint7"], name2idx["joint8"]]
            else:
                # fallback: last 2 prismatic joints
                found = [
                    i
                    for i in range(self.num_joints)
                    if self.sim.get_joint_info(robot_id=self.p_robot_id, joint_index=i)[
                        2
                    ]
                    == 1
                ]
                if len(found) >= 2:
                    self._gripper_joint_indices = found[-2:]
                    logger.debug(
                        f"Guessed prismatic joints: {self._gripper_joint_indices}"
                    )
                else:
                    logger.warning("Failed to auto-detect gripper joints; disabled")
                    return
        except Exception as e:
            logger.error(f"pybullet joint lookup failed: {e}")
            return

        # Cache joint limits and positions in one pass
        closed, opened, limits = [], [], []
        for jidx in self._gripper_joint_indices:
            info = self.sim.get_joint_info(robot_id=self.p_robot_id, joint_index=jidx)
            lower, upper = float(info[8]), float(info[9])

            # Cache the limits for later use
            limits.append((lower, upper))

            # Determine closed/open positions
            closed_pos, open_pos = (
                (upper, lower) if abs(upper) < abs(lower) else (lower, upper)
            )

            # Apply limits immediately
            closed.append(float(np.clip(closed_pos, lower, upper)))
            opened.append(float(np.clip(open_pos, lower, upper)))

        self._gripper_closed_positions = closed
        self._gripper_open_positions = opened
        self._gripper_joint_limits = limits  # NEW: Store cached limits

        logger.debug(
            f"Init gripper: {self._gripper_joint_indices=} {closed=} {opened=} {limits=}"
        )

    @classmethod
    def from_can_port(cls, can_name: str = "can0") -> Optional["PiperHardware"]:
        try:
            piper = cls(can_name=can_name)
            return piper
        except Exception as e:
            logger.warning(e)
            return None

    async def connect(self) -> None:
        """
        Setup the robot.
        can_number : 0 if only one robot is connected, 1 to connect to second robot
        """
        self.is_connected = False

        logger.info(f"Connecting to Agilex Piper on {self.can_name}")

        if not is_running_on_linux():
            logger.warning("Robot can only be connected on a Linux machine.")
            return

        try:
            proc = subprocess.Popen(
                ["bash", str(get_resources_path() / "agilex_can_activate.sh")],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if proc.stdout is None or proc.stderr is None:
                logger.error("Failed to start the CAN activation script.")
                return

            # Example: read lines one by one, log them
            for line in proc.stdout:
                logger.debug("[can-script] " + line.rstrip())

            for line in proc.stderr:
                logger.error("[can-script] " + line.rstrip())

            proc.wait(timeout=10)
            if proc.returncode != 0:
                logger.warning(f"Script exited with exit code: {proc.returncode}")
                return
        except subprocess.CalledProcessError as e:
            logger.warning(
                f"CAN Activation Failed!\nError: {e}\nOutput:\n{e.stdout}\nErrors:\n{e.stderr}"
            )
            return

        logger.debug(f"Attempting to connect to Agilex Piper on {self.can_name}")
        self.motors_bus = C_PiperInterface_V2(
            can_name=self.can_name, judge_flag=True, can_auto_init=True
        )
        await asyncio.sleep(0.1)
        # Check if CAN bus is OK
        is_ok = self.motors_bus.isOk()
        if not is_ok:
            logger.warning(
                f"Could not connect to Agilex Piper on {self.can_name}: CAN bus is not OK."
            )
            return

        self.motors_bus.ConnectPort(can_init=True)

        # Start by resetting the control mode (useful if arm stuck in teaching mode)
        self.motors_bus.MotionCtrl_1(0x02, 0, 0)  # 恢复
        self.motors_bus.MotionCtrl_2(0, 0, 0, 0x00)  # 位置速度模式
        # Reset the gripper
        self.motors_bus.GripperCtrl(0, 1000, 0x00, 0)
        await asyncio.sleep(1.5)

        self.firmware_version = self.motors_bus.GetPiperFirmwareVersion()
        logger.info(
            f"Connected to Agilex Piper on {self.can_name} with firmware version {self.firmware_version}"
        )

        self.motors_bus.ArmParamEnquiryAndConfig(
            param_setting=0x01,
            # data_feedback_0x48x=0x02,
            end_load_param_setting_effective=0,
            set_end_load=0x0,
        )
        await asyncio.sleep(0.1)
        # First, start standby mode (ctrl_mode=0x00). Then, switch to CAN command control mode (ctrl_mode=0x01)
        # Source: https://static.generation-robots.com/media/agilex-piper-user-manual.pdf
        self.motors_bus.MotionCtrl_2(
            ctrl_mode=0x00, move_mode=0x01, move_spd_rate_ctrl=100, is_mit_mode=0x00
        )
        await asyncio.sleep(0.1)
        self.motors_bus.MotionCtrl_2(
            ctrl_mode=0x01,
            move_mode=0x01,
            move_spd_rate_ctrl=100,
            is_mit_mode=0x00,
            installation_pos=0x01,
            # installation_pos:
            # 0x01: Horizontal installation
            # 0x02: Left side installation
            # 0x03: Right side installation
        )
        await asyncio.sleep(0.2)
        self.is_torqued = True

        self.init_config()
        self.is_connected = True

    def get_default_base_robot_config(
        self, voltage: str, raise_if_none: bool = False
    ) -> Union[BaseRobotConfig, None]:
        return BaseRobotConfig(
            name=self.name,
            servos_voltage=12.0,
            servos_offsets=[0] * len(self.SERVO_IDS),
            servos_calibration_position=[1e-6] * len(self.SERVO_IDS),
            servos_offsets_signs=[1] * len(self.SERVO_IDS),
            gripping_threshold=4500,
            non_gripping_threshold=500,
        )

    def disconnect(self) -> None:
        """
        Disconnect the robot.
        """

        if not self.is_connected:
            return

        # Reset the control mode
        self.motors_bus.MotionCtrl_1(0x02, 0, 0)  # 恢复
        self.motors_bus.MotionCtrl_2(0, 0, 0, 0x00)  # 位置速度模式
        time.sleep(0.1)

        # Disconnect
        self.motors_bus.DisconnectPort()
        self.is_connected = False
        self.is_torqued = False

    def init_config(self) -> None:
        """
        Load the config file.
        """
        self.config = self.get_default_base_robot_config(voltage="24v")

    def enable_torque(self) -> None:
        if not self.is_connected:
            return
        self.motors_bus.EnablePiper()
        self.is_torqued = True

    def disable_torque(self) -> None:
        # Disable torque
        if not self.is_connected:
            return
        self.motors_bus.DisableArm(7)
        # Disable the gripper with no change of zero position
        self.motors_bus.GripperCtrl(0, self.GRIPPER_EFFORT, self.DISABLE_GRIPPER, 0)
        self.is_torqued = False

    def read_motor_torque(self, servo_id: int) -> Optional[float]:
        """
        Read the torque of a motor

        raise: Exception if the routine has not been implemented
        """
        if servo_id >= self.GRIPPER_SERVO_ID:
            gripper_state = self.motors_bus.GetArmGripperMsgs().gripper_state
            return gripper_state.grippers_effort
        else:
            return 100 if self.is_torqued else 0

    def read_motor_voltage(self, servo_id: int) -> Optional[float]:
        """
        Read the voltage of a motor

        raise: Exception if the routine has not been implemented
        """
        # Not implemented
        return None

    def write_motor_position(self, servo_id: int, units: int, **kwargs: Any) -> None:
        """
        Move the motor to the specified position.

        Args:
            servo_id: The ID of the motor to move.
            units: The position to move the motor to. This is in the range 0 -> (self.RESOLUTION -1).
        Each position is mapped to an angle.
        """
        # If servo_id is 7 (gripper), write the gripper command
        if servo_id == self.GRIPPER_SERVO_ID:
            self.write_gripper_command(units)
            return

        # Otherwise, we need to write the position to the motor. We can only write all motors at once.
        current_position = self.read_joints_position(unit="motor_units", source="robot")
        # The last position is the gripper, so we drop it
        current_position = current_position[:-1]

        # Override the position of the specified servo_id
        current_position[servo_id - 1] = units

        # Clamp the position in the allowed range for the motors using self.piper_limits
        if servo_id in self.piper_limits_degrees:
            min_limit = self.piper_limits_degrees[servo_id]["min_angle_limit"] * 1000
            max_limit = self.piper_limits_degrees[servo_id]["max_angle_limit"] * 1000
            current_position[servo_id - 1] = np.clip(
                current_position[servo_id - 1], min_limit, max_limit
            )

        # Move robot
        self.motors_bus.ModeCtrl(
            ctrl_mode=0x01, move_mode=0x01, move_spd_rate_ctrl=100, is_mit_mode=0x00
        )
        self.joint_position = current_position.tolist()
        self.motors_bus.JointCtrl(*[int(q) for q in current_position])

    def set_motors_positions(
        self, q_target_rad: np.ndarray, enable_gripper: bool = False
    ) -> None:
        """
        Write the positions to the motors.

        If the robot is connected, the position is written to the motors.
        We always move the robot in the simulation.

        This does not control the gripper.

        q_target_rad is in radians.

        Args:
            q_target_rad: The position to move the motors to. This is in radians.
            enable_gripper: If True, the gripper will be moved to the position specified in q_target_rad.
        """
        logger.debug(
            f"Piper: Setting motors to {q_target_rad} rad, gripper enabled: {enable_gripper}"
        )
        joint_indices = (
            self.actuated_joints
        )  # size 6, the gripper is excluded from actuated_joints in the Piper class
        target_positions = [q_target_rad[i] for i in joint_indices]

        # Move in simulation first to validate the position
        self.sim.set_joints_states(
            robot_id=self.p_robot_id,
            joint_indices=joint_indices,
            target_positions=target_positions,
        )
        if enable_gripper and len(q_target_rad) >= self.GRIPPER_SERVO_ID:
            gripper_open = self._rad_to_open_command(
                q_target_rad[self.GRIPPER_SERVO_ID - 1]
            )
            self.move_gripper_in_sim(open=gripper_open)
        self.sim.step()

        if self.is_connected:
            validated_q_target = self.sim.get_joints_states(
                robot_id=self.p_robot_id, joint_indices=joint_indices
            )  # size 6
            validated_q_target_array = np.array(validated_q_target)
            q_target = self._radians_vec_to_motor_units(validated_q_target_array)
            if enable_gripper and len(q_target_rad) < self.GRIPPER_SERVO_ID:
                q_target = np.append(
                    q_target,
                    q_target_rad[-1]
                    * self.GRIPPER_MAX_ANGLE
                    * self.RESOLUTION
                    / (np.pi / 2),
                )
            self.write_group_motor_position(
                q_target=q_target, enable_gripper=enable_gripper
            )

    def write_group_motor_position(
        self,
        q_target: np.ndarray,  # in motor units
        enable_gripper: bool,
    ) -> None:
        # First 6 values of q_target are the joints position.
        # Clamp joints in the allowed range for the motors using self.piper_limits_degrees * 1000
        # Simply clamping the values is not strong enough, as in certain positions the angle limits are exceeded.
        # To avoid this, we first set the joints in pybullet, then read the validated position and use it to clamp the values.

        clamped_joints = []
        for i, joint in enumerate(q_target):
            # in self.piper_limits_degrees, there are only indexes 1 to 6, we forgo the gripper
            servo_id = i + 1
            if servo_id in self.piper_limits_degrees:
                min_limit = self.piper_limits_degrees[i + 1]["min_angle_limit"] * 1000
                max_limit = self.piper_limits_degrees[i + 1]["max_angle_limit"] * 1000
                # q_target[i] = np.clip(joint, min_limit, max_limit)  # noqa: F821
                clamped_joint = int(np.clip(joint, min_limit, max_limit))
                clamped_joints.append(clamped_joint)

        self.motors_bus.ModeCtrl(
            ctrl_mode=0x01, move_mode=0x01, move_spd_rate_ctrl=100, is_mit_mode=0x00
        )
        self.motors_bus.JointCtrl(*clamped_joints)

        # Move the gripper if it is enabled
        if enable_gripper and len(q_target) >= self.GRIPPER_SERVO_ID:
            # The last value q_target[-1] is the gripper position in motor units. Rescale it between (0, 1) to write the gripper command
            gripper_position = q_target[-1]
            gripper_command = gripper_position / (
                self.GRIPPER_MAX_ANGLE * self.RESOLUTION
            )
            self.write_gripper_command(gripper_command)

    def read_motor_position(self, servo_id: int, **kwargs: Any) -> Optional[int]:
        """
        Read the position of the motor. This should return the position in motor units.
        """
        return self.read_group_motor_position()[servo_id - 1]

    def read_joints_position(
        self,
        unit: Literal["rad", "motor_units", "degrees", "other"] = "rad",
        source: Literal["sim", "robot"] = "robot",
        joints_ids: Optional[List[int]] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> np.ndarray:
        """
        Read the position of the joints. This should return the position in motor units.
        """
        # The parent method reads the joints, but not the gripper.
        joints = super().read_joints_position(
            unit=unit,
            source=source,
            joints_ids=joints_ids,
            min_value=min_value,
            max_value=max_value,
        )

        # Add the gripper position if it is not already present
        if len(joints) < self.GRIPPER_SERVO_ID and (
            joints_ids is None or self.GRIPPER_SERVO_ID in joints_ids
        ):
            gripper_position = self.read_gripper_command(
                source=source, unit=unit, min_value=min_value, max_value=max_value
            )

            joints = np.array(joints.tolist() + [gripper_position]).astype(np.float32)
        return joints

    def read_group_motor_position(self) -> np.ndarray:
        """
        Read the position of all the motors. This should return the position in motor units.
        """
        joint_state = self.motors_bus.GetArmJointMsgs().joint_state
        # in 0.001 deg
        position_unit = np.array(
            [
                joint_state.joint_1,
                joint_state.joint_2,
                joint_state.joint_3,
                joint_state.joint_4,
                joint_state.joint_5,
                joint_state.joint_6,
            ]
        )

        return position_unit

    def calibrate_motors(self, **kwargs: Any) -> None:
        """
        This is called during the calibration phase of the robot.
        It sets the offset of all motors to self.RESOLUTION/2.
        """
        # Set zero positions for motors and gripper
        self.motors_bus.JointConfig(set_zero=0xAE)  # Set zero position of motors
        # Set zero position of gripper
        self.motors_bus.GripperCtrl(0, self.GRIPPER_EFFORT, 0x00, 0xAE)

    def _units_vec_to_radians(self, units: np.ndarray) -> np.ndarray:
        """
        Convert from motor discrete units (0 -> RESOLUTION) to radians
        """
        position_deg = units * 2 * np.pi / self.RESOLUTION  # in 0.001 deg
        return position_deg  # in deg

    def _radians_vec_to_motor_units(self, radians: np.ndarray) -> np.ndarray:
        """
        Convert from radians to motor discrete units (0 -> RESOLUTION)

        Note: The result can exceed the resolution of the motor, in the case of a continuous rotation motor.
        """
        position_deg = np.rad2deg(radians)  # in degrees
        position_units = (position_deg * 1000).astype(int)  # in motor units
        return position_units

    async def calibrate(self) -> tuple[Literal["success", "in_progress", "error"], str]:
        """
        This is called during the calibration phase of the robot.
        CAUTION : Set the robot in sleep mode where falling wont be an issue and close the gripper.
        """
        if not self.is_connected:
            logger.warning("Robot is not connected. Cannot calibrate.")
            return "error", "Robot is not connected. Cannot calibrate."

        if self.calibration_current_step == 0:
            self.calibration_current_step = 1
            return (
                "in_progress",
                "STEP 1/3: NEXT STEP, THE ROBOT WILL FALL. HOLD THE ROBOT to prevent it from falling.",
            )
        elif self.calibration_current_step == 1:
            self.disable_torque()
            self.calibration_current_step = 2
            return (
                "in_progress",
                "STEP 2/3: Move the robot to its sleep position. Close the gripper fully.",
            )
        elif self.calibration_current_step == 2:
            self.calibration_current_step = 0
            self.calibrate_motors()
            self.enable_torque()
            return (
                "success",
                "STEP 3/3: Calibration completed successfully. Offsets and signs saved to the robot.",
            )

        return (
            "error",
            "Calibration failed. Please try again.",
        )

    def read_gripper_command(
        self,
        source: Literal["sim", "robot"] = "robot",
        unit: Literal["motor_units", "rad", "degrees", "other"] = "motor_units",
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> float:
        """
        Read if gripper is open or closed.
        """

        if not self.is_connected:
            logger.warning("Robot not connected")
            return 0

        if source == "robot":
            gripper_ctrl = self.motors_bus.GetArmGripperMsgs().gripper_state
            gripper_position = gripper_ctrl.grippers_angle
            # Calculate normalized gripper position [0, 1]
            normalized = (gripper_position - self.GRIPPER_ZERO_POSITION) / (
                self.GRIPPER_MAX_ANGLE * 1000
            )
        elif source == "sim":
            if not self._gripper_joint_indices:
                return 0.0

            fractions = []
            for jidx, closed_pos, open_pos in zip(
                self._gripper_joint_indices,
                self._gripper_closed_positions,
                self._gripper_open_positions,
            ):
                pos = float(
                    self.sim.get_joint_state(
                        robot_id=self.p_robot_id, joint_index=jidx
                    )[0]
                )
                denom = open_pos - closed_pos
                f = (pos - closed_pos) / denom if abs(denom) > 1e-9 else 0.0
                fractions.append(float(np.clip(f, 0.0, 1.0)))
            normalized = float(np.mean(fractions)) if fractions else 0.0
        else:
            raise ValueError(f"Unknown source: {source}")

        if unit == "motor_units":
            # Don't do anything
            gripper_units = normalized
        elif unit == "rad":
            # Convert the gripper from (0, GRIPPER_MAX_ANGLE) to (0, pi / 2)
            gripper_units = normalized * (np.pi / 2)
        elif unit == "degrees":
            # Convert the gripper from (0, GRIPPER_MAX_ANGLE) to (0, 90)
            gripper_units = normalized * 90
        elif unit == "other":
            # Convert the gripper from (0, GRIPPER_MAX_ANGLE) to (min_value, max_value)
            if min_value is None or max_value is None:
                raise ValueError(
                    "min_value and max_value must be provided for 'other' unit."
                )
            gripper_units = normalized * (max_value - min_value) + min_value
        else:
            raise ValueError(f"Unknown unit: {unit}")

        return gripper_units

    def _rad_to_open_command(self, radians: float) -> float:
        """
        Convert radians to an open command for the gripper.
        The open command is in the range [0, 1], where 0 is fully closed and 1 is fully open.
        """
        # Clip to valid range and normalize to [0, 1]
        clipped_radians = np.clip(radians, 0, np.pi / 2)  # Max 90 degrees (π/2 rad)
        open_command = clipped_radians / (np.pi / 2)  # Normalize to [0, 1]
        return open_command

    def write_gripper_command(self, command: float) -> None:
        """
        Open or close the gripper.

        command: 0 to close, 1 to open
        """
        if not self.is_connected:
            logger.debug("Robot not connected, cannot write gripper command")
            return
        # Gripper -> Convert from 0->RESOLUTION to 0->GRIPPER_MAX_ANGLE
        unit_degree = command * self.GRIPPER_MAX_ANGLE
        unit_command = self.GRIPPER_ZERO_POSITION + int(unit_degree * 1000)
        self.motors_bus.GripperCtrl(
            gripper_angle=unit_command,
            gripper_effort=self.GRIPPER_EFFORT,
            gripper_code=self.ENABLE_GRIPPER,
            set_zero=0,
        )
        self.update_object_gripping_status()

    def is_powered_on(self) -> bool:
        """
        Check if the robot is powered on.
        """
        return self.is_connected

    def status(self) -> RobotConfigStatus:
        """
        Get the status of the robot.

        Returns:
            RobotConfigStatus object
        """
        return RobotConfigStatus(
            name=self.name, device_name=self.can_name, robot_type="manipulator"
        )

    def move_gripper_in_sim(self, open: float) -> None:
        """
        Move the AgileX Piper gripper in the simulation.
        `open` is normalized: 0.0 = fully closed, 1.0 = fully open.
        This updates both prismatic fingers (URDF joints `joint7` and `joint8`).
        """

        # Early returns for edge cases
        if not self._gripper_joint_indices:
            logger.debug("No gripper joints available")
            return

        if self.is_object_gripped:
            return

        # Clamp input to [0,1]
        open = float(np.clip(open, 0.0, 1.0))

        # Calculate target positions using cached data
        target_positions = []
        for i, (close, open_pos, (lower, upper)) in enumerate(
            zip(
                self._gripper_closed_positions,
                self._gripper_open_positions,
                self._gripper_joint_limits,
            )
        ):
            # Interpolate target position
            target = close + (open_pos - close) * open

            # Apply cached joint limits
            if lower <= upper:
                target = float(np.clip(target, lower, upper))

            target_positions.append(target)

        # Apply the joint positions
        self.sim.set_joints_states(
            robot_id=self.p_robot_id,
            joint_indices=self._gripper_joint_indices,
            target_positions=target_positions,
        )
